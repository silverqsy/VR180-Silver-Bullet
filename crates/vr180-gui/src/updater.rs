//! Seamless auto-update via GitHub Releases.
//!
//! Follows the well-proven Tauri-updater architecture — the same feed /
//! trust / install model, hand-rolled for eframe:
//!
//! * **Feed**: a Tauri-style `latest.json` uploaded as an asset on every
//!   GitHub release. The app polls the STABLE URL
//!   `releases/latest/download/latest.json`, which 302s to the newest
//!   release's asset — the URL baked into the binary never changes.
//! * **Trust**: every update artifact is minisign-signed on the build
//!   machine (`~/.vr180-updater/vr180-updater.key`, OFF-repo); the
//!   manifest carries the signature and the app verifies against
//!   [`PUBKEY`] before touching anything. Clients only trust the pubkey
//!   in the build they run — ship a new pubkey in a release BEFORE
//!   signing with it. HTTPS/GitHub is transport, not trust.
//! * **Install**: immediate download → verify → swap → relaunch (not
//!   install-on-quit). macOS swaps the WHOLE signed+notarized `.app`
//!   bundle atomically (per-file patching would break the seal) and
//!   relaunches with `open -n`. Windows spawns the Inno `-setup.exe`
//!   with `/SILENT` (per-user install, no UAC) and exits.
//!
//! Threading follows the app's export-job pattern: all network / disk
//! work runs on a plain `std::thread`, events flow back over a
//! crossbeam channel, the UI polls each frame.
//!
//! Testing: set `VR180_UPDATE_URL` to override the feed (e.g. a local
//! `python3 -m http.server` serving a hand-built latest.json).

#[cfg(target_os = "macos")]
use std::path::Path;
use std::path::PathBuf;

/// Stable feed URL — `releases/latest/download/<asset>` always redirects
/// to the newest release, so each release just ships a fresh latest.json.
const UPDATE_FEED_URL: &str =
    "https://github.com/silverqsy/VR180-Silver-Bullet/releases/latest/download/latest.json";

/// minisign public key (key id F1F8A4B3EA057A41). The matching secret
/// key lives OUTSIDE the repo at `~/.vr180-updater/vr180-updater.key`
/// on the build machines. Lose the key → no more updates, ever.
const PUBKEY: &str = "RWRBegXqs6T48TnCzQA6Cf6QUZ1upbnUBUb1uk8vYR0J8hotYTPpSI78";

/// Platform key in the manifest's `platforms` map (same convention as
/// the Tauri updater so tooling stays interchangeable).
#[cfg(target_os = "macos")]
const PLATFORM_KEY: &str = "darwin-aarch64";
#[cfg(target_os = "windows")]
const PLATFORM_KEY: &str = "windows-x86_64";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const PLATFORM_KEY: &str = "linux-x86_64";

fn feed_url() -> String {
    std::env::var("VR180_UPDATE_URL").unwrap_or_else(|_| UPDATE_FEED_URL.to_string())
}

// ─── Manifest ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Manifest {
    pub version: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub pub_date: String,
    pub platforms: std::collections::HashMap<String, PlatformEntry>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PlatformEntry {
    /// Full contents of the `.minisig` file, base64-encoded (the
    /// two-line minisign signature block — same field the Tauri
    /// manifest carries).
    pub signature: String,
    pub url: String,
}

/// A newer version advertised by the feed for THIS platform.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub version: String,
    pub notes: String,
    pub entry: PlatformEntry,
}

/// Parse "2.0.0"-style versions tolerantly (missing parts = 0, ignores
/// any pre-release suffix after `-`).
fn parse_ver(v: &str) -> (u64, u64, u64) {
    let v = v.trim().trim_start_matches('v');
    let v = v.split('-').next().unwrap_or(v);
    let mut it = v.split('.').map(|p| p.trim().parse::<u64>().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}

pub fn is_newer(candidate: &str, current: &str) -> bool {
    parse_ver(candidate) > parse_ver(current)
}

// ─── Check ───────────────────────────────────────────────────────────

/// Fetch the feed and return a newer-version entry for this platform,
/// or `None` when up to date (or the manifest lacks this platform).
pub fn check(current_version: &str) -> Result<Option<UpdateInfo>, String> {
    let url = feed_url();
    tracing::info!("updater: checking {url}");
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(20))
        .call()
        .map_err(|e| format!("update check: {e}"))?;
    let manifest: Manifest = resp
        .into_json()
        .map_err(|e| format!("update manifest parse: {e}"))?;
    if !is_newer(&manifest.version, current_version) {
        tracing::info!(
            "updater: up to date (feed {} vs current {current_version})",
            manifest.version
        );
        return Ok(None);
    }
    let Some(entry) = manifest.platforms.get(PLATFORM_KEY) else {
        tracing::warn!(
            "updater: {} available but no '{PLATFORM_KEY}' platform entry",
            manifest.version
        );
        return Ok(None);
    };
    tracing::info!("updater: {} available for {PLATFORM_KEY}", manifest.version);
    Ok(Some(UpdateInfo {
        version: manifest.version,
        notes: manifest.notes,
        entry: entry.clone(),
    }))
}

// ─── Download + verify + stage ───────────────────────────────────────

/// A verified, extracted update waiting to be installed.
#[derive(Debug, Clone)]
pub struct StagedUpdate {
    pub version: String,
    /// macOS: path of the extracted `.app` bundle in the staging dir.
    /// Windows: path of the verified `-setup.exe`.
    pub payload: PathBuf,
}

/// Staging directory next to settings.json (survives nothing — cleaned
/// on each new download).
fn staging_dir() -> Result<PathBuf, String> {
    let base = crate::decoder::Settings::config_path()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .ok_or("no config dir")?;
    Ok(base.join("updates"))
}

/// Download the platform asset with byte progress, verify its minisign
/// signature against [`PUBKEY`], and (macOS) extract the `.app`.
/// `progress(got, total)` is called from the worker thread.
pub fn download_and_stage(
    info: &UpdateInfo,
    mut progress: impl FnMut(u64, u64),
) -> Result<StagedUpdate, String> {
    let dir = staging_dir()?;
    let _ = std::fs::remove_dir_all(&dir); // drop any previous staging
    std::fs::create_dir_all(&dir).map_err(|e| format!("staging dir: {e}"))?;

    // ── Download ──
    let file_name = info
        .entry
        .url
        .rsplit('/')
        .next()
        .filter(|n| !n.is_empty())
        .ok_or("bad asset url")?;
    let artifact = dir.join(file_name);
    tracing::info!("updater: downloading {} → {}", info.entry.url, artifact.display());
    let resp = ureq::get(&info.entry.url)
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| format!("download: {e}"))?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&artifact).map_err(|e| format!("create {file_name}: {e}"))?,
    );
    let mut buf = [0u8; 128 * 1024];
    let mut got: u64 = 0;
    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|e| format!("download read: {e}"))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut out, &buf[..n]).map_err(|e| format!("write: {e}"))?;
        got += n as u64;
        progress(got, total);
    }
    std::io::Write::flush(&mut out).map_err(|e| format!("flush: {e}"))?;
    drop(out);

    // ── Verify (minisign over the whole artifact) ──
    let data = std::fs::read(&artifact).map_err(|e| format!("read back: {e}"))?;
    verify_minisign(&data, &info.entry.signature)?;
    tracing::info!("updater: signature OK ({} bytes)", data.len());
    drop(data);

    // ── Stage per platform ──
    #[cfg(target_os = "macos")]
    {
        // Extract with ditto: preserves the code signature, xattrs and
        // symlinks of the notarized bundle (plain unzip can subtly break
        // the seal).
        let extract = dir.join("extracted");
        std::fs::create_dir_all(&extract).map_err(|e| format!("extract dir: {e}"))?;
        let st = std::process::Command::new("ditto")
            .arg("-x")
            .arg("-k")
            .arg(&artifact)
            .arg(&extract)
            .status()
            .map_err(|e| format!("ditto: {e}"))?;
        if !st.success() {
            return Err(format!("ditto extract failed ({st})"));
        }
        let app = std::fs::read_dir(&extract)
            .map_err(|e| format!("scan extract: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map(|x| x == "app").unwrap_or(false))
            .ok_or("no .app in update zip")?;
        // Gatekeeper sanity: the new bundle must assess as notarized —
        // a broken/unsigned bundle would be unlaunchable after the swap.
        let assess = std::process::Command::new("spctl")
            .args(["--assess", "--type", "execute"])
            .arg(&app)
            .output()
            .map_err(|e| format!("spctl: {e}"))?;
        if !assess.status.success() {
            return Err(format!(
                "downloaded app failed Gatekeeper assessment: {}",
                String::from_utf8_lossy(&assess.stderr).trim()
            ));
        }
        tracing::info!("updater: staged + assessed {}", app.display());
        return Ok(StagedUpdate { version: info.version.clone(), payload: app });
    }

    #[cfg(target_os = "windows")]
    {
        return Ok(StagedUpdate {
            version: info.version.clone(),
            payload: artifact,
        });
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = artifact;
        Err("auto-update not supported on this platform".into())
    }
}

fn verify_minisign(data: &[u8], signature_b64: &str) -> Result<(), String> {
    use minisign_verify::{PublicKey, Signature};
    let pk = PublicKey::from_base64(PUBKEY).map_err(|e| format!("pubkey: {e}"))?;
    // The manifest field is the full .minisig file content, base64'd.
    let sig_text = base64_decode(signature_b64.trim())
        .and_then(|b| String::from_utf8(b).ok())
        .ok_or("signature: bad base64")?;
    let sig = Signature::decode(&sig_text).map_err(|e| format!("signature decode: {e}"))?;
    pk.verify(data, &sig, false)
        .map_err(|_| "SIGNATURE VERIFICATION FAILED — update rejected".to_string())
}

/// Minimal base64 decoder (standard alphabet, padding tolerant) — not
/// worth a crate for one field.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

// ─── Install + relaunch ──────────────────────────────────────────────

/// Swap the update into place and relaunch. On success this function
/// DOES NOT RETURN (the process exits). Errors return so the UI can
/// surface them without having killed the running app.
pub fn install_and_relaunch(staged: &StagedUpdate) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        // Locate the running bundle: …/VR180 Silver Bullet.app/Contents/MacOS/vr180-gui
        let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
        let bundle = exe
            .ancestors()
            .nth(3)
            .map(Path::to_path_buf)
            .filter(|p| p.extension().map(|x| x == "app").unwrap_or(false))
            .ok_or("not running from an installed .app (dev build?) — update by replacing the app manually")?;
        let parent = bundle.parent().ok_or("bundle has no parent dir")?;

        // Swap: rename the old bundle aside (atomic, same volume), move
        // the new one in. The aside copy is kept for manual rollback and
        // replaced on the next update.
        let aside = parent.join(format!(
            "{}.old",
            bundle.file_name().and_then(|n| n.to_str()).unwrap_or("VR180 Silver Bullet.app")
        ));
        let _ = std::fs::remove_dir_all(&aside);
        std::fs::rename(&bundle, &aside).map_err(|e| {
            format!("could not move the old app aside ({e}) — is it in a write-protected location?")
        })?;
        let moved = std::fs::rename(&staged.payload, &bundle);
        if let Err(e) = moved {
            // Cross-volume rename fails — fall back to ditto copy.
            tracing::warn!("updater: rename failed ({e}), ditto-copying instead");
            let st = std::process::Command::new("ditto")
                .arg(&staged.payload)
                .arg(&bundle)
                .status()
                .map_err(|e| format!("ditto copy: {e}"))?;
            if !st.success() {
                // Roll back the aside so the user still has a working app.
                let _ = std::fs::rename(&aside, &bundle);
                return Err(format!("install copy failed ({st}) — rolled back"));
            }
        }
        tracing::info!("updater: swapped to v{} — relaunching", staged.version);
        std::process::Command::new("open")
            .arg("-n")
            .arg(&bundle)
            .spawn()
            .map_err(|e| format!("relaunch: {e}"))?;
        std::process::exit(0);
    }

    #[cfg(target_os = "windows")]
    {
        // Inno Setup per-user install (PrivilegesRequired=lowest → no
        // UAC). The installer waits for this process to exit, replaces
        // the files and (via its [Run] section) relaunches the app.
        std::process::Command::new(&staged.payload)
            .args(["/SILENT", "/NORESTART"])
            .spawn()
            .map_err(|e| format!("launch installer: {e}"))?;
        std::process::exit(0);
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = staged;
        Err("auto-update not supported on this platform".into())
    }
}

// ─── Prefs (skip-version) ────────────────────────────────────────────

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UpdaterPrefs {
    /// Version the user chose to skip — no auto-prompt for it again
    /// (manual check still shows it).
    #[serde(default)]
    pub skip_version: Option<String>,
}

fn prefs_path() -> Option<PathBuf> {
    crate::decoder::Settings::config_path()
        .and_then(|p| p.parent().map(|d| d.join("updater.json")))
}

pub fn load_prefs() -> UpdaterPrefs {
    prefs_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

pub fn save_prefs(p: &UpdaterPrefs) {
    if let (Some(path), Ok(json)) = (prefs_path(), serde_json::to_string_pretty(p)) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, json);
    }
}

// ─── Background worker (export-job pattern) ──────────────────────────

/// Events from the updater worker thread to the UI.
#[derive(Debug, Clone)]
pub enum UpdateEvent {
    /// Check finished: newer version (Some) or up to date (None).
    Checked(Option<UpdateInfo>),
    /// Check failed (offline etc.) — shown only for manual checks.
    CheckFailed(String),
    /// Download progress in bytes (`total` may be 0 if unknown).
    Progress { got: u64, total: u64 },
    /// Downloaded, verified and staged — about to install.
    Installing,
    /// Windows: verified installer staged — the UI must close the app
    /// gracefully and spawn it from `App::on_exit`. (`process::exit(0)`
    /// from this worker thread wedges on Windows DLL teardown with live
    /// D3D11/WASAPI threads — found in the 2.1.0 E2E test: the installer
    /// launched but the app stayed open until closed by hand.)
    ReadyToInstall(StagedUpdate),
    /// Download / verify / install failed.
    Failed(String),
}

/// Spawn a background check. Sends exactly one `Checked`/`CheckFailed`.
pub fn spawn_check(tx: crossbeam_channel::Sender<UpdateEvent>) {
    let current = env!("CARGO_PKG_VERSION").to_string();
    std::thread::spawn(move || {
        let ev = match check(&current) {
            Ok(info) => UpdateEvent::Checked(info),
            Err(e) => UpdateEvent::CheckFailed(e),
        };
        let _ = tx.send(ev);
    });
}

/// Spawn download → verify → stage → install → relaunch. On success the
/// process exits inside; on failure a `Failed` event is delivered.
pub fn spawn_install(info: UpdateInfo, tx: crossbeam_channel::Sender<UpdateEvent>) {
    std::thread::spawn(move || {
        let txp = tx.clone();
        let staged = match download_and_stage(&info, move |got, total| {
            let _ = txp.try_send(UpdateEvent::Progress { got, total });
        }) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(UpdateEvent::Failed(e));
                return;
            }
        };
        let _ = tx.send(UpdateEvent::Installing);
        // Give the UI a beat to paint the "installing" state before the
        // process is replaced.
        std::thread::sleep(std::time::Duration::from_millis(300));
        // Windows: DON'T install-and-exit from this worker — hand the
        // staged payload to the UI, which closes the window and spawns
        // the installer during app teardown (see UpdateEvent::ReadyToInstall).
        #[cfg(target_os = "windows")]
        {
            let _ = tx.send(UpdateEvent::ReadyToInstall(staged));
        }
        #[cfg(not(target_os = "windows"))]
        if let Err(e) = install_and_relaunch(&staged) {
            let _ = tx.send(UpdateEvent::Failed(e));
        }
    });
}

/// Windows: spawn the staged silent installer WITHOUT exiting — called
/// from `App::on_exit` once the event loop has unwound (settings saved,
/// GPU/decode/audio threads torn down). The installer waits out any
/// stragglers via `CloseApplications`, replaces the files, and relaunches
/// the app through the `[Run] Check: WizardSilent` entry in windows.iss.
#[cfg(target_os = "windows")]
pub fn spawn_installer(staged: &StagedUpdate) -> Result<(), String> {
    std::process::Command::new(&staged.payload)
        .args(["/SILENT", "/NORESTART"])
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("launch installer: {e}"))
}

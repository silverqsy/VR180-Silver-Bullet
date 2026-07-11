# Auto-update

The app updates itself from GitHub Releases. The design follows the
well-proven Tauri-updater architecture (same manifest format, minisign
trust model, and install flow), hand-rolled for eframe in
`crates/vr180-gui/src/updater.rs`.

## How it works (client)

- **Check**: on launch (errors swallowed), every 4 hours, and manually by
  clicking the version label in the toolbar. The app fetches the STABLE
  URL — `https://github.com/silverqsy/VR180-Silver-Bullet/releases/latest/download/latest.json` —
  which 302s to the newest release's `latest.json` asset, so the URL baked
  into the binary never changes.
- **Trust**: every update artifact is **minisign-signed**; the manifest
  carries the signature and the app verifies it against the pubkey baked
  into `updater.rs` before touching anything. HTTPS/GitHub is transport,
  not trust — a compromised release cannot push code without the key.
  macOS additionally runs `spctl --assess` on the extracted bundle
  (must still be notarized).
- **Install** (immediate, not install-on-quit):
  - macOS: download `…-mac-arm64-app.zip` → verify → `ditto`-extract →
    Gatekeeper assess → atomically swap the whole `.app` (old bundle kept
    as `….app.old` for manual rollback) → `open -n` relaunch → exit.
  - Windows: download `…-windows-x64-setup.exe` → verify → spawn it with
    `/SILENT /NORESTART` → exit. The Inno installer (per-user, no UAC)
    replaces the files and relaunches the app (silent-mode `[Run]` entry
    in `installer/windows.iss`).
- **UX**: toolbar version label becomes a `⬆ update` badge; the popover
  shows release notes, byte progress, *Install & Restart* / *Skip this
  version* / *Check again*. Skip is persisted in
  `<config dir>/updater.json`. The auto-prompt never interrupts a running
  export.
- **Testing**: set `VR180_UPDATE_URL` to any URL serving a `latest.json`
  (e.g. `python3 -m http.server` in a staging dir).

## The signing key

- Private key: `~/.vr180-updater/vr180-updater.key` on **both** build
  machines — generated with `minisign -G -W` (passwordless; the key FILE
  is the secret). NEVER commit it. **Lose the key → no more updates, ever.**
- Public key (baked into `updater.rs`):
  `RWRBegXqs6T48TnCzQA6Cf6QUZ1upbnUBUb1uk8vYR0J8hotYTPpSI78`
- Key rotation: a release containing the NEW pubkey must ship (signed by
  the OLD key) before any release is signed with the new key — clients
  only trust the pubkey in the build they run.
- To copy the key to the Windows box: transfer
  `~/.vr180-updater/vr180-updater.key` to
  `C:\Users\<user>\.vr180-updater\vr180-updater.key` over a private
  channel. Install minisign there (`scoop install minisign` or the
  release binary).

## Release recipe

**Bump the version first** (or the updater has nothing to offer):
`Cargo.toml` (workspace) + `crates/vr180-gui/Cargo.toml` + the
`Info.plist` in the dist bundle skeleton + `installer/windows.iss`
`#define AppVer`.

### macOS box
1. Build, bundle (dylibbundler), sign inside-out, notarize + staple the
   `.app` and DMG as usual (see the release notes in project memory /
   BUILD.md).
2. Zip the stapled app as the **update asset** (ditto preserves the
   signature): `ditto -c -k --keepParent "dist/VR180 Silver Bullet.app" "dist/VR180-Silver-Bullet-<ver>-mac-arm64-app.zip"`
3. `node scripts/make-latest-json.mjs --version <ver> --notes-file RELEASE_NOTES.md`
   → signs the zip, writes/merges `release-staging/latest.json`.

### Windows box
1. Build the bundle folder + `ISCC.exe installer\windows.iss` → setup.exe.
2. Copy the mac box's `release-staging/latest.json` into `release-staging/`
   (so the entries MERGE), then
   `node scripts/make-latest-json.mjs --version <ver> --notes-file RELEASE_NOTES.md`.

### Publish
```
gh release create v<ver> --title "VR180 Silver Bullet <ver>" --notes-file RELEASE_NOTES.md \
    dist/VR180-Silver-Bullet-<ver>-mac-arm64.dmg \
    release-staging/VR180-Silver-Bullet-<ver>-mac-arm64-app.zip \
    release-staging/VR180-Silver-Bullet-<ver>-windows-x64-setup.exe \
    dist/VR180-Silver-Bullet-<ver>-windows-x64.zip \
    release-staging/latest.json
```
`latest.json` must be an asset on **every** release from now on — the
stable `releases/latest/download/latest.json` URL resolves against the
newest release. A release without it turns the update check into a 404
(harmless — clients log and move on — but they won't see the update).

The DMG and the portable Windows zip remain the **fresh-install**
vehicles; the app zip and setup.exe are the **update** vehicles.

## Gotchas

- The mac update asset must be zipped with `ditto -c -k --keepParent`
  (not plain `zip`) so the notarization seal/xattrs survive; the client
  extracts with `ditto -x -k` for the same reason.
- The manifest `url` must exactly match the uploaded asset name.
- `minisign -S` with the `-W` key does not prompt; if it ever prompts for
  a password, the wrong key file is being used.
- Dev builds (`cargo run`) refuse to install updates ("not running from
  an installed .app") — test the swap from a real installed copy.

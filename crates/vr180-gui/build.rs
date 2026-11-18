//! Build script — embeds the application icon and product name into
//! `vr180-gui.exe`.
//!
//! On Windows the executable carries its own icon in a Win32 resource;
//! without it the shell shows a blank default everywhere the *file* is
//! referenced: Explorer, the taskbar, and the Start-Menu / desktop shortcut
//! the installer creates. The live window icon is set separately at runtime
//! (`ViewportBuilder::with_icon` in `main.rs`).
//!
//! This is a no-op on macOS / Linux — macOS takes its icon from
//! `assets/icon.icns` via the `.app` bundle, so `winresource` is only a
//! build-dependency on Windows hosts (`[target.'cfg(windows)'.build-dependencies]`)
//! and the body below is `cfg`-gated to match.
fn main() {
    // macOS: keep the Swift helpers (`helpers/bin/*`, what the Dev bundles and
    // the release script copy next to the app) built from the CURRENT
    // sources. Reruns only when a helper source or the build script changes;
    // a missing toolchain is a warning, never a failed build.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let script = root.join("helpers/build_swift.sh");
        for f in ["apac_encode", "mvhevc_encode", "vt_denoise", "aivu_wrap"] {
            println!("cargo:rerun-if-changed={}", root.join("helpers/swift").join(format!("{f}.swift")).display());
        }
        println!("cargo:rerun-if-changed={}", script.display());
        match std::process::Command::new("bash").arg(&script).output() {
            Ok(o) if o.status.success() => {}
            Ok(o) => println!(
                "cargo:warning=vr180-gui: Swift helpers NOT rebuilt ({}); helpers/bin may be stale: {}",
                o.status, String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("")
            ),
            Err(e) => println!("cargo:warning=vr180-gui: could not run helpers/build_swift.sh: {e}"),
        }
    }
    #[cfg(windows)]
    {
        // Re-run if the artwork changes so a new `icon.ico` is picked up.
        println!("cargo:rerun-if-changed=../../assets/icon.ico");
        let mut res = winresource::WindowsResource::new();
        // Path is relative to this crate's manifest dir (where the build
        // script runs); the shared icon lives at the workspace root.
        res.set_icon("../../assets/icon.ico");
        // Product identity for the shell: Task Manager's Name column, the
        // taskbar hover and the file's Properties all read FileDescription /
        // ProductName. Without these winresource falls back to the crate
        // name and the app shows up as "vr180-gui". Deliberately NOT
        // version-specific (the version rides ProductVersion from Cargo).
        res.set("ProductName", "VR180 Silver Bullet");
        res.set("FileDescription", "VR180 Silver Bullet");
        if let Err(e) = res.compile() {
            // Don't fail the build over an icon — warn and ship without it.
            println!("cargo:warning=vr180-gui: failed to embed Windows icon: {e}");
        }
    }
}

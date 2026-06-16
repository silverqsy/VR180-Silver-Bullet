//! Build script — embeds the application icon into `vr180-gui.exe`.
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
    #[cfg(windows)]
    {
        // Re-run if the artwork changes so a new `icon.ico` is picked up.
        println!("cargo:rerun-if-changed=../../assets/icon.ico");
        let mut res = winresource::WindowsResource::new();
        // Path is relative to this crate's manifest dir (where the build
        // script runs); the shared icon lives at the workspace root.
        res.set_icon("../../assets/icon.ico");
        if let Err(e) = res.compile() {
            // Don't fail the build over an icon — warn and ship without it.
            println!("cargo:warning=vr180-gui: failed to embed Windows icon: {e}");
        }
    }
}

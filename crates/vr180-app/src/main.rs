//! `vr180-app` binary entry point. The actual app logic lives in the
//! library half (`vr180_app_lib::run`) — this binary just calls it,
//! so unit tests + integration tests can pull in the lib without
//! launching a Tauri window.

fn main() {
    vr180_app_lib::run();
}

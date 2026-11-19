//! Telling the system the app is at work.
//!
//! macOS rests an app that is not in front or whose window is covered
//! ("App Nap": every thread of it drops to the lowest priority and its
//! timers are held back), and lets the machine sleep when nobody touches
//! it. An export is exactly when the app is left alone. Measured on a batch
//! of exports, the same settings with the window in front and with another
//! app in front: 10.3 against 5.4 frames a second, and 1.65 against 0.46 on
//! a clip with noise reduction; the threads' priority read 31 to 47 in
//! front and 4 behind.
//!
//! While a [`Busy`] is held the app is exempt from both: it runs at its own
//! priority wherever its window is, and the machine stays awake (its
//! display may still sleep).

/// Held for as long as the app has unattended work in flight; the
/// exemption ends when it is dropped.
pub struct Busy(#[allow(dead_code)] platform::Held);

impl Busy {
    /// Tell the system the app is at work on what the user asked for
    /// (`reason`: a few words, shown by the system's own tools). `None`
    /// where the system has no such notion or declines.
    pub fn begin(reason: &str) -> Option<Self> {
        platform::begin(reason).map(Busy)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};

    /// `NSActivityUserInitiated`: work the user asked for. No App Nap, and
    /// no idle system sleep (bit 20 is part of it).
    const USER_INITIATED: u64 = 0x00FF_FFFF;

    /// The activity's token, with the process information object it is
    /// given back to.
    pub struct Held {
        info: Retained<AnyObject>,
        token: Retained<AnyObject>,
    }

    pub fn begin(reason: &str) -> Option<Held> {
        let reason = std::ffi::CString::new(reason).ok()?;
        unsafe {
            let info: Option<Retained<AnyObject>> = msg_send![class!(NSProcessInfo), processInfo];
            let info = info?;
            let text: Option<Retained<AnyObject>> =
                msg_send![class!(NSString), stringWithUTF8String: reason.as_ptr()];
            let text = text?;
            let token: Option<Retained<AnyObject>> =
                msg_send![&*info, beginActivityWithOptions: USER_INITIATED, reason: &*text];
            Some(Held { info, token: token? })
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![&*self.info, endActivity: &*self.token];
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    /// Nothing to hold.
    pub struct Held;

    pub fn begin(_reason: &str) -> Option<Held> {
        None
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// The system's own list of what keeps the machine awake.
    fn assertions() -> String {
        let out = std::process::Command::new("/usr/bin/pmset").args(["-g", "assertions"]).output();
        out.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
    }

    /// The system lists the app as at work while a `Busy` is held, under
    /// the reason given, and no longer once it is dropped.
    #[test]
    fn the_system_is_told_while_it_is_held_and_not_after() {
        let reason = format!("vr180 activity test {}", std::process::id());
        assert!(!assertions().contains(&reason), "listed before it began");
        let busy = Busy::begin(&reason).expect("the system takes the activity");
        let held = assertions();
        assert!(held.contains(&reason), "not listed while held:\n{held}");
        let line = held.lines().find(|l| l.contains(&reason)).unwrap_or_default();
        assert!(line.contains("PreventUserIdleSystemSleep"), "the machine is not kept awake: {line}");
        drop(busy);
        assert!(!assertions().contains(&reason), "still listed after it ended");
    }

    /// Two at once are two: one ending leaves the other in force.
    #[test]
    fn one_ending_leaves_another_in_force() {
        let (a, b) = (format!("vr180 activity a {}", std::process::id()), format!("vr180 activity b {}", std::process::id()));
        let first = Busy::begin(&a).expect("first");
        let second = Busy::begin(&b).expect("second");
        drop(first);
        let now = assertions();
        assert!(!now.contains(&a) && now.contains(&b), "after the first ended:\n{now}");
        drop(second);
        assert!(!assertions().contains(&b));
    }
}

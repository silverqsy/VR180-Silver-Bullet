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
//!
//! Windows has the same two habits under other names. A PC goes to sleep
//! when nobody has touched it for the time its power plan says, whatever a
//! program is computing: only a power request keeps it awake. And a process
//! whose windows are all minimised or covered may be run at reduced speed
//! (on battery: a low processor frequency, the efficient cores) unless it
//! says it is not to be. A [`Busy`] there is a power request of the kind
//! "the system is needed" and, for as long as any is held, the process's
//! speed taken out of the system's hands.

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

#[cfg(target_os = "windows")]
mod platform {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Power::{
        PowerClearRequest, PowerCreateRequest, PowerRequestSystemRequired, PowerSetRequest,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
        POWER_REQUEST_CONTEXT_SIMPLE_STRING, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE, REASON_CONTEXT,
        REASON_CONTEXT_0,
    };

    /// `POWER_REQUEST_CONTEXT_VERSION`.
    const CONTEXT_VERSION: u32 = 0;

    /// How many are held: the process's speed is one setting for all of
    /// them, taken when the first begins and given back when the last ends.
    static HELD: AtomicUsize = AtomicUsize::new(0);

    /// A power request that is set: while it is, the machine does not go to
    /// sleep by itself (the display still may).
    pub struct Held {
        request: HANDLE,
    }

    pub fn begin(reason: &str) -> Option<Held> {
        let mut text: Vec<u16> = reason.encode_utf16().chain(std::iter::once(0)).collect();
        let context = REASON_CONTEXT {
            Version: CONTEXT_VERSION,
            Flags: POWER_REQUEST_CONTEXT_SIMPLE_STRING,
            Reason: REASON_CONTEXT_0 { SimpleReasonString: text.as_mut_ptr() },
        };
        unsafe {
            // The system copies the reason: `text` need not outlive the call.
            let request = PowerCreateRequest(&context);
            if request.is_null() || request == INVALID_HANDLE_VALUE {
                return None;
            }
            if PowerSetRequest(request, PowerRequestSystemRequired) == 0 {
                CloseHandle(request);
                return None;
            }
            if HELD.fetch_add(1, Ordering::SeqCst) == 0 {
                speed_is_the_systems(false);
            }
            Some(Held { request })
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            unsafe {
                PowerClearRequest(self.request, PowerRequestSystemRequired);
                CloseHandle(self.request);
            }
            if HELD.fetch_sub(1, Ordering::SeqCst) == 1 {
                speed_is_the_systems(true);
            }
        }
    }

    /// Whether the system decides how fast the process runs (`true`: its
    /// default, by what its windows show) or the process runs at full speed
    /// wherever they are. A system that does not know the setting (before
    /// Windows 10 1709) refuses the call, and nothing is lost.
    fn speed_is_the_systems(yes: bool) {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: if yes { 0 } else { PROCESS_POWER_THROTTLING_EXECUTION_SPEED },
            StateMask: 0,
        };
        unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                (&state as *const PROCESS_POWER_THROTTLING_STATE).cast(),
                std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            );
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
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

#[cfg(all(test, target_os = "windows"))]
mod windows_tests {
    use super::*;
    use windows_sys::Win32::System::Power::{CallNtPowerInformation, SystemExecutionState, ES_SYSTEM_REQUIRED};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetProcessInformation, ProcessPowerThrottling,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
    };

    /// The two tests share the process: one at a time.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// What the system holds for the process: (it decides the speed, the
    /// process is slowed).
    fn speed() -> (bool, bool) {
        let mut state = PROCESS_POWER_THROTTLING_STATE { Version: 1, ControlMask: 0, StateMask: 0 };
        let ok = unsafe {
            GetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                (&mut state as *mut PROCESS_POWER_THROTTLING_STATE).cast(),
                std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            )
        };
        assert!(ok != 0, "the system does not answer for the process's speed");
        (
            state.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED == 0,
            state.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED != 0,
        )
    }

    /// Whether anything on the machine asks for the system to stay awake:
    /// this request or another program's. The list that names each request
    /// (`powercfg /requests`) is for administrators only, so where another
    /// program holds one already this says nothing about ours; what the
    /// tests can always tell is that the system took the request (`begin`
    /// answered) and what it holds for the process's speed.
    fn system_required() -> bool {
        let mut state: u32 = 0;
        let status = unsafe {
            CallNtPowerInformation(SystemExecutionState, std::ptr::null(), 0, (&mut state as *mut u32).cast(), 4)
        };
        assert_eq!(status, 0, "the system does not answer for its execution state");
        state & ES_SYSTEM_REQUIRED != 0
    }

    /// While a `Busy` is held the machine is asked to stay awake and the
    /// process runs at its own speed; once it is dropped the speed is the
    /// system's again.
    #[test]
    fn the_system_is_told_while_it_is_held_and_not_after() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(speed(), (true, false), "before it began");
        let busy = Busy::begin("vr180 activity test").expect("the system takes the request");
        assert!(system_required(), "the machine is not asked to stay awake");
        assert_eq!(speed(), (false, false), "while held");
        drop(busy);
        assert_eq!(speed(), (true, false), "after it ended");
    }

    /// Two at once are two: one ending leaves the other in force.
    #[test]
    fn one_ending_leaves_another_in_force() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let first = Busy::begin("vr180 activity a").expect("first");
        let second = Busy::begin("vr180 activity b").expect("second");
        drop(first);
        assert!(system_required(), "after the first ended");
        assert_eq!(speed(), (false, false), "after the first ended");
        drop(second);
        assert_eq!(speed(), (true, false), "after both ended");
    }
}

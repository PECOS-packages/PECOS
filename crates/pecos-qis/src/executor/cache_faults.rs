//! Process-local fault injection. This entire module and all call sites are test-only.

use std::collections::BTreeMap;
use std::fs::TryLockError;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Site {
    InitialLoad,
    PostLockLoad,
    FinalLoad,
    TimeoutLoad,
    RenameLoad,
    BeforeLock,
    Staged,
    Link,
    Publish,
    Published,
    Manifest,
    Lock,
    LockPoll,
    Budget,
    CleanupObserved,
    CleanupLocked,
}

#[derive(Clone)]
pub(super) enum Action {
    Fail(ErrorKind),
    RawError(i32),
    Pause { signal: PathBuf, resume: PathBuf },
}

#[derive(Default)]
struct State {
    actions: BTreeMap<(Site, usize), Action>,
    hits: BTreeMap<Site, Vec<PathBuf>>,
    budget: Option<Duration>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

pub(super) fn enable() {
    assert!(STATE.set(Mutex::new(State::default())).is_ok());
}

pub(super) fn on(site: Site, occurrence: usize, action: Action) {
    STATE
        .get()
        .expect("enabled")
        .lock()
        .expect("fault state")
        .actions
        .insert((site, occurrence), action);
}

pub(super) fn hits(site: Site) -> Vec<PathBuf> {
    STATE
        .get()
        .expect("enabled")
        .lock()
        .expect("fault state")
        .hits
        .get(&site)
        .cloned()
        .unwrap_or_default()
}

pub(super) fn check(site: Site, path: &Path) -> std::io::Result<()> {
    let Some(state) = STATE.get() else {
        return Ok(());
    };
    let action = {
        let mut state = state.lock().expect("fault state");
        let hits = state.hits.entry(site).or_default();
        hits.push(path.to_owned());
        let count = hits.len();
        state.actions.get(&(site, count)).cloned()
    };
    match action {
        Some(Action::Fail(kind)) => Err(Error::new(kind, format!("injected {site:?}"))),
        Some(Action::RawError(code)) => Err(Error::from_raw_os_error(code)),
        Some(Action::Pause { signal, resume }) => {
            std::fs::write(signal, path.as_os_str().as_encoded_bytes()).expect("signal barrier");
            wait_for(&resume);
            Ok(())
        }
        None => Ok(()),
    }
}

pub(super) fn point(site: Site, path: &Path) {
    check(site, path).expect("pause seam");
}

pub(super) fn lock_error() -> Result<(), TryLockError> {
    check(Site::Lock, Path::new("")).map_err(TryLockError::Error)
}

pub(super) fn set_budget(budget: Duration) {
    STATE
        .get()
        .expect("enabled")
        .lock()
        .expect("fault state")
        .budget = Some(budget);
}

pub(super) fn wait_budget(default: Duration) -> Duration {
    let Some(state) = STATE.get() else {
        return default;
    };
    let budget = state.lock().expect("fault state").budget;
    if let Some(budget) = budget {
        point(Site::Budget, Path::new(""));
        budget
    } else {
        default
    }
}

pub(super) fn wait_for(path: &Path) {
    let watchdog = Instant::now();
    while !path.exists() {
        assert!(
            watchdog.elapsed() < Duration::from_secs(60),
            "barrier timed out: {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

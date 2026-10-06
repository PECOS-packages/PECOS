//! Cache contracts exercised in isolated, genuinely concurrent test processes.

use super::cache_faults::{self as faults, Action, Site};
use super::program_cache::{
    self, EntryKind, LIB_SUFFIX, LockOutcome, bucket_lock_path, parse_entry,
};
use super::*;
use crate::test_env::{TestChild, finish_test_child, join_test_child, spawn_test_child};
use std::fs::{File, TryLockError};
use std::io::{ErrorKind, Write};
use std::time::Duration;

const PROGRAM: &[u8] = b"define i64 @qmain(i64 %arg) { ret i64 0 } ; cache protocol tests";
const CHILD_TEST: &str = "executor::cache_tests::cache_child";

mod bucket_tests;

struct ChildRun(Option<TestChild>);

impl ChildRun {
    fn wait(&mut self, path: &Path) -> Result<(), String> {
        let watchdog = std::time::Instant::now();
        while !path.exists() {
            let exited = self
                .0
                .as_mut()
                .expect("child")
                .try_wait()
                .expect("child status")
                .is_some();
            // A child can signal and exit between the file check and try_wait.
            if exited && path.exists() {
                return Ok(());
            }
            if exited || watchdog.elapsed() >= Duration::from_secs(60) {
                let result = finish_test_child(self.0.take().expect("child"), Duration::ZERO);
                let detail = match result {
                    Ok(output) => format!(
                        "child exited ({}) before signalling: {}\n{}",
                        output.status,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    ),
                    Err(error) => error,
                };
                return Err(format!("barrier {}: {detail}", path.display()));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    fn join(mut self) {
        join_test_child(self.0.take().expect("child"));
    }

    fn kill(mut self) {
        let mut child = self.0.take().expect("child");
        assert!(!child.kill_and_wait().success());
    }
}

struct Fixture {
    root: tempfile::TempDir,
    control: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self {
            root: tempfile::tempdir().expect("root"),
            control: tempfile::tempdir().expect("control"),
        }
    }

    fn spawn(&self, mode: &str, id: &str) -> ChildRun {
        spawn(mode, id, self.root.path(), self.control.path())
    }

    fn run(&self, mode: &str) {
        self.spawn(mode, "one").join();
    }

    fn cache(&self) -> PathBuf {
        self.root.path().join("qis-programs-v2")
    }

    fn signal(&self, name: &str) {
        signal(&self.control.path().join(name));
    }
    fn wait(&self, name: &str, child: &mut ChildRun) {
        child
            .wait(&self.control.path().join(name))
            .unwrap_or_else(|error| panic!("{error}"));
    }

    fn library(&self) -> PathBuf {
        entries(&self.cache())
            .into_iter()
            .find(|path| kind(path) == Some(EntryKind::Library))
            .expect("library")
    }
}

fn spawn(mode: &str, id: &str, root: &Path, control: &Path) -> ChildRun {
    ChildRun(Some(spawn_test_child(
        CHILD_TEST,
        &[
            ("PECOS_CACHE_TEST_MODE", mode.as_ref()),
            ("PECOS_CACHE_TEST_ID", id.as_ref()),
            ("PECOS_CACHE_TEST_CONTROL", control.as_os_str()),
            ("PECOS_CACHE_DIR", root.as_os_str()),
        ],
    )))
}

fn signal(path: &Path) {
    std::fs::write(path, b"ready").expect("signal");
}
fn control() -> PathBuf {
    std::env::var_os("PECOS_CACHE_TEST_CONTROL")
        .expect("control")
        .into()
}
fn root() -> PathBuf {
    std::env::var_os("PECOS_CACHE_DIR").expect("root").into()
}
fn count(site: Site) -> usize {
    faults::hits(site).len()
}
fn fail(site: Site) {
    faults::on(site, 1, Action::Fail(ErrorKind::Other));
}
fn pause(site: Site, occurrence: usize, ready: &str, go: &str) {
    faults::on(
        site,
        occurrence,
        Action::Pause {
            signal: control().join(ready),
            resume: control().join(go),
        },
    );
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .expect("cache directory")
        .map(|e| e.expect("entry").path())
        .collect()
}

fn kind(path: &Path) -> Option<EntryKind> {
    path.file_name()?
        .to_str()
        .and_then(parse_entry)
        .map(|(_, kind)| kind)
}

fn assert_layout(cache: &Path, manifest: bool) {
    let paths = entries(cache);
    assert_eq!(paths.len(), if manifest { 3 } else { 2 }, "{paths:?}");
    for expected in [EntryKind::Library, EntryKind::Lock] {
        assert_eq!(
            paths.iter().filter(|p| kind(p) == Some(expected)).count(),
            1,
            "{paths:?}"
        );
    }
    assert_eq!(
        paths
            .iter()
            .filter(|p| kind(p) == Some(EntryKind::Manifest))
            .count(),
        usize::from(manifest)
    );
}

fn assert_no_staging(cache: &Path) {
    assert!(
        entries(cache)
            .iter()
            .all(|p| kind(p) != Some(EntryKind::Staging))
    );
}

fn interface() -> QisHeliosInterface {
    let mut interface = QisHeliosInterface::new();
    interface.program = PROGRAM.to_vec();
    interface.format = ProgramFormat::LlvmIrText;
    interface
}

fn load() -> Result<PathBuf, InterfaceError> {
    interface().create_shared_library()
}

fn digest() -> String {
    QisHeliosInterface::get_qis_ffi_lib_singleton().expect("runtime");
    interface().program_digest().expect("digest")
}

fn force_build() {
    fail(Site::InitialLoad);
    fail(Site::PostLockLoad);
}

#[test]
fn cache_child() {
    let Ok(mode) = std::env::var("PECOS_CACHE_TEST_MODE") else {
        return;
    };
    faults::enable();
    match mode.as_str() {
        "seed" => {
            load().expect("compile");
            assert_eq!(count(Site::Link), 1);
        }
        "fresh" => {
            load().expect("fresh load");
            assert_eq!(count(Site::InitialLoad), 1);
            assert_eq!(count(Site::Link), 0);
        }
        "concurrent" => {
            let id = std::env::var("PECOS_CACHE_TEST_ID").expect("id");
            pause(Site::BeforeLock, 1, &format!("ready-{id}"), "go");
            load().expect("concurrent compile");
            assert_eq!(count(Site::BeforeLock), 1);
            std::fs::write(
                control().join(format!("count-{id}")),
                count(Site::Link).to_string(),
            )
            .expect("count");
        }
        "threads" => concurrent_threads(false),
        "staging" => concurrent_threads(true),
        "load-failures" => {
            force_build();
            fail(Site::Link);
            let error = load().expect_err("link failure");
            assert!(
                error.to_string().contains("Injected linking failure"),
                "{error}"
            );
            for site in [Site::InitialLoad, Site::PostLockLoad, Site::Link] {
                assert_eq!(count(site), 1);
            }
        }
        "final-failure" => {
            fail(Site::FinalLoad);
            let error = load().expect_err("final load failure");
            assert!(
                error.to_string().contains("Injected program load failure"),
                "{error}"
            );
            assert_eq!(count(Site::FinalLoad), 1);
            assert_eq!(count(Site::Publish), 1);
        }
        "final-locked" => {
            pause(Site::FinalLoad, 1, "final-ready", "finish-final");
            load().expect("load while holding compilation lock");
            assert_eq!(count(Site::FinalLoad), 1);
        }
        "raw-lock-unsupported" | "raw-cleanup-unsupported" => raw_lock_error(&mode),
        "noisy-success" | "noisy-failure" | "noisy-watchdog" => noisy_child(&mode),
        "holder" => hold_lock(),
        "dead-waiter" => {
            let cache = get_persistent_cache_dir().expect("cache");
            // Windows may release a dead process's locks asynchronously.
            let budget = if cfg!(windows) {
                Duration::from_secs(30)
            } else {
                Duration::ZERO
            };
            let guard = CompilationLock::acquire(&cache, &digest(), budget).expect("acquire");
            assert!(matches!(guard, LockOutcome::Acquired(_)));
            #[cfg(unix)]
            assert_eq!(count(Site::LockPoll), 0);
            assert_eq!(count(Site::Lock), count(Site::LockPoll) + 1);
        }
        "live-holder" => {
            pause(Site::Staged, 1, "acquired", "publish");
            pause(Site::Published, 1, "published", "release");
            load().expect("holder publication");
            assert_eq!(count(Site::Staged), 1);
            assert_eq!(count(Site::Published), 1);
        }
        "live-waiter" => live_waiter(),
        "probe" => {
            let path = bucket_lock_path(&root().join("qis-programs-v2"), &digest());
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .expect("fresh lock handle");
            assert!(
                matches!(file.try_lock(), Err(TryLockError::WouldBlock)),
                "holder must still own the pathname"
            );
        }
        "budget-present" | "budget-absent" | "budget-load-failure" => budget(&mode),
        "cleanup-owned" => cleanup_owned(),
        "cleanup" => {
            program_cache::cleanup_old_cache_files(&root().join("qis-programs-v2"), 86400);
            assert_eq!(count(Site::Lock), 1);
            assert_eq!(count(Site::CleanupObserved), 1);
        }
        "cleanup-locked" => {
            pause(Site::CleanupLocked, 1, "cleanup-held", "finish-cleanup");
            program_cache::cleanup_old_cache_files(&root().join("qis-programs-v2"), 86400);
            assert_eq!(count(Site::CleanupLocked), 1);
            assert_eq!(count(Site::Lock), 1);
        }
        "cleanup-race" => {
            pause(Site::CleanupObserved, 1, "observed", "clean");
            program_cache::cleanup_old_cache_files(&root().join("qis-programs-v2"), 86400);
            assert_eq!(count(Site::CleanupObserved), 1);
            assert_eq!(count(Site::Lock), 1);
        }
        "replace" => {
            force_build();
            load().expect("replace");
            for site in [
                Site::InitialLoad,
                Site::PostLockLoad,
                Site::Link,
                Site::Publish,
            ] {
                assert_eq!(count(site), 1);
            }
        }
        "replacement-driver" => replacement_driver(),
        "replacement-child" => replacement_child(),
        "rename-present"
        | "rename-absent"
        | "rename-invalid"
        | "rename-load-failure"
        | "unsupported"
        | "lock-error"
        | "manifest-error" => error_path(&mode),
        mode if mode.starts_with("bucket-") => bucket_tests::run(mode),
        _ => panic!("unknown mode {mode}"),
    }
}

fn concurrent_threads(unsupported: bool) {
    let n = if unsupported { 2 } else { 4 };
    for i in 1..=n {
        pause(Site::BeforeLock, i, &format!("before-{i}"), "start");
        if unsupported {
            faults::on(Site::Lock, i, Action::Fail(ErrorKind::Unsupported));
            pause(Site::Staged, i, &format!("staged-{i}"), "finish");
        }
    }
    let workers: Vec<_> = (0..n)
        .map(|_| std::thread::spawn(|| load().expect("thread load")))
        .collect();
    for i in 1..=n {
        faults::wait_for(&control().join(format!("before-{i}")));
    }
    signal(&control().join("start"));
    if unsupported {
        let watchdog = std::time::Instant::now();
        while !(1..=n).all(|i| control().join(format!("staged-{i}")).exists()) {
            assert!(
                workers.iter().all(|worker| !worker.is_finished()),
                "worker exited before both staging directories existed"
            );
            assert!(
                watchdog.elapsed() < Duration::from_secs(60),
                "staging barrier watchdog"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let paths = faults::hits(Site::Staged);
        assert_eq!(paths.len(), 2);
        assert_ne!(paths[0], paths[1]);
        for (i, path) in paths.iter().enumerate() {
            assert_eq!(kind(path), Some(EntryKind::Staging));
            std::fs::write(path.join("independent"), i.to_string())
                .expect("independent staging content");
        }
        for (i, path) in paths.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(path.join("independent")).expect("contents"),
                i.to_string()
            );
        }
        signal(&control().join("finish"));
    }
    for worker in workers {
        worker.join().expect("worker");
    }
    assert_eq!(count(Site::BeforeLock), n);
    assert_eq!(count(Site::Link), if unsupported { 2 } else { 1 });
    if unsupported {
        assert_eq!(count(Site::Lock), 2);
    }
    assert_layout(&root().join("qis-programs-v2"), true);
}

fn hold_lock() {
    let cache = get_persistent_cache_dir().expect("cache");
    let guard = CompilationLock::acquire(&cache, &digest(), Duration::ZERO).expect("lock");
    assert!(matches!(guard, LockOutcome::Acquired(_)));
    assert_eq!(count(Site::Lock), 1);
    signal(&control().join("acquired"));
    faults::wait_for(&control().join("release"));
    drop(guard);
}

fn live_waiter() {
    pause(Site::LockPoll, 1, "contended", "poll-again");
    pause(Site::LockPoll, 2, "polled-after-publication", "finish-poll");
    load().expect("waiter uses publication");
    assert!(count(Site::LockPoll) >= 2);
    assert_eq!(count(Site::PostLockLoad), 1);
    assert_eq!(count(Site::Link), 0);
}

fn budget(mode: &str) {
    faults::set_budget(Duration::ZERO);
    if mode != "budget-absent" {
        fail(Site::InitialLoad);
    }
    if mode == "budget-load-failure" {
        fail(Site::TimeoutLoad);
    }
    let result = load();
    if mode == "budget-present" {
        result.expect("timeout recheck uses published file");
        assert_eq!(count(Site::InitialLoad), 1);
        assert_eq!(count(Site::TimeoutLoad), 1);
    } else {
        let error = result.expect_err("timeout");
        assert!(error.to_string().contains("Timeout waiting"), "{error}");
    }
    if mode == "budget-load-failure" {
        assert_eq!(count(Site::InitialLoad), 1);
        assert_eq!(count(Site::TimeoutLoad), 1);
    }
    assert_eq!(count(Site::Budget), 1);
    assert_eq!(count(Site::LockPoll), 1);
    assert_eq!(count(Site::Link), 0);
}

#[test]
fn concurrent_processes_and_layout() {
    let fixture = Fixture::new();
    let mut children: Vec<_> = (0..4)
        .map(|i| fixture.spawn("concurrent", &i.to_string()))
        .collect();
    for (i, child) in children.iter_mut().enumerate() {
        fixture.wait(&format!("ready-{i}"), child);
    }
    fixture.signal("go");
    for child in children {
        child.join();
    }
    let compilations: usize = (0..4)
        .map(|i| {
            std::fs::read_to_string(fixture.control.path().join(format!("count-{i}")))
                .expect("count")
                .parse::<usize>()
                .expect("number")
        })
        .sum();
    assert_eq!(compilations, 1);
    assert_layout(&fixture.cache(), true);
    assert_eq!(entries(fixture.root.path()), vec![fixture.cache()]);
}

#[test]
fn concurrent_threads_and_staging_uniqueness() {
    Fixture::new().run("threads");
    Fixture::new().run("staging");
}

#[test]
fn failed_initial_and_post_lock_loads_preserve_publication() {
    let fixture = Fixture::new();
    fixture.run("seed");
    let library = fixture.library();
    let original = std::fs::read(&library).expect("original bytes");
    fixture.run("load-failures");
    assert_eq!(
        std::fs::read(&library).expect("publication must survive"),
        original
    );
    assert_layout(&fixture.cache(), true);
    fixture.run("fresh");
}

#[test]
fn failed_final_load_preserves_publication() {
    let fixture = Fixture::new();
    fixture.run("final-failure");
    let original = std::fs::read(fixture.library()).expect("published bytes");
    assert_ne!(original, [] as [u8; 0]);
    assert_layout(&fixture.cache(), true);
    fixture.run("fresh");
    assert_eq!(
        std::fs::read(fixture.library()).expect("unchanged bytes"),
        original
    );
}

#[test]
fn killed_holder_releases_lock() {
    let fixture = Fixture::new();
    let mut holder = fixture.spawn("holder", "holder");
    fixture.wait("acquired", &mut holder);
    holder.kill();
    fixture.run("dead-waiter");
}

#[test]
fn live_holder_keeps_lock_path_after_publication() {
    let fixture = Fixture::new();
    let mut holder = fixture.spawn("live-holder", "holder");
    fixture.wait("acquired", &mut holder);
    let mut waiter = fixture.spawn("live-waiter", "waiter");
    fixture.wait("contended", &mut waiter);
    fixture.signal("publish");
    fixture.wait("published", &mut holder);
    fixture.signal("poll-again");
    fixture.wait("polled-after-publication", &mut waiter);
    fixture.run("probe");
    fixture.signal("release");
    fixture.signal("finish-poll");
    holder.join();
    waiter.join();
    assert_layout(&fixture.cache(), true);
}

#[test]
fn exhausted_budget_rechecks_publication_without_compiling() {
    for mode in ["budget-present", "budget-absent", "budget-load-failure"] {
        let fixture = Fixture::new();
        let original = if mode == "budget-absent" {
            None
        } else {
            fixture.run("seed");
            Some((
                fixture.library(),
                std::fs::read(fixture.library()).expect("original bytes"),
            ))
        };
        let mut holder = fixture.spawn("holder", "holder");
        fixture.wait("acquired", &mut holder);
        fixture.run(mode);
        if let Some((library, bytes)) = original {
            assert_eq!(
                std::fs::read(library).expect("timeout load must preserve publication"),
                bytes
            );
            fixture.run("fresh");
        }
        fixture.signal("release");
        holder.join();
    }
}

fn make_old(path: &Path) {
    let mut options = File::options();
    options.write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        if path.is_dir() {
            options.custom_flags(0x0200_0000);
        }
    }
    #[cfg(not(windows))]
    if path.is_dir() {
        options.write(false).read(true);
    }
    options
        .open(path)
        .expect("open for age")
        .set_modified(std::time::SystemTime::now() - Duration::from_hours(48))
        .expect("set old age");
}

fn cleanup_owned() {
    let cache = root().join("qis-programs-v2");
    std::fs::create_dir_all(&cache).expect("cache");
    let digest = "a".repeat(64);
    let library = cache.join(format!("program_{digest}.{LIB_SUFFIX}"));
    let manifest = library.with_extension("manifest");
    let lock = bucket_lock_path(&cache, &digest);
    let staging = program_cache::staging_dir(&cache, &digest)
        .expect("staging")
        .keep();
    let owned = [&library, &manifest, &lock];
    for path in owned {
        std::fs::write(path, b"old owned").expect("fixture");
        make_old(path);
    }
    std::fs::write(staging.join("partial.lib"), b"partial linker output").expect("partial");
    make_old(&staging);
    let foreign = [
        root().join("dependency.tar.gz"),
        root().join(format!("program_{digest}.{LIB_SUFFIX}")),
        cache.join("dependency.tar.gz"),
        cache.join(".hidden"),
        // Uppercase hex is not owned. Use a digest that cannot name an owned file
        // on a case-insensitive filesystem (macOS, Windows).
        cache.join(format!("program_{}.{LIB_SUFFIX}", "C".repeat(64))),
        cache.join(format!("program_{}.so", "a".repeat(63))),
        cache.join(format!("program_{digest}.so.compiling.123")),
        cache.join(format!(".program_{digest}.bad-token.tmp")),
        cache.join(format!(".program_{digest}.abcde.tmp")),
        cache.join(format!(".program_{digest}.abcdefg.tmp")),
        cache.join(format!("program_{digest}.lock.extra")),
        cache.join(format!("program_{digest}.lock")),
    ];
    for path in &foreign {
        std::fs::write(path, b"foreign").expect("foreign");
        make_old(path);
    }
    let other_lock = bucket_lock_path(&cache, &"b".repeat(64));
    std::fs::write(&other_lock, b"retained lock").expect("other lock");
    make_old(&other_lock);
    program_cache::cleanup_old_cache_files(&cache, 86400);
    assert_eq!(count(Site::Lock), 1);
    assert_eq!(count(Site::CleanupObserved), 1);
    for path in [&library, &manifest, &staging] {
        assert!(!path.exists(), "{}", path.display());
    }
    for path in foreign.iter().chain([&lock, &other_lock]) {
        assert!(path.exists(), "{}", path.display());
    }
}

#[test]
fn cleanup_removes_only_owned_old_entries() {
    Fixture::new().run("cleanup-owned");
}

#[test]
fn cleanup_skips_locked_digest() {
    let fixture = Fixture::new();
    fixture.run("seed");
    let mut holder = fixture.spawn("holder", "holder");
    fixture.wait("acquired", &mut holder);
    let library = fixture.library();
    let name = library.file_name().expect("name").to_str().expect("utf8");
    let (digest, _) = parse_entry(name).expect("owned");
    let staging = program_cache::staging_dir(&fixture.cache(), digest)
        .expect("staging")
        .keep();
    let manifest = library.with_extension("manifest");
    for path in [&library, &manifest, &staging] {
        make_old(path);
    }
    fixture.run("cleanup");
    for path in [&library, &manifest, &staging] {
        assert!(path.exists(), "{}", path.display());
    }
    fixture.signal("release");
    holder.join();
}

#[test]
fn cleanup_rechecks_age_after_locking() {
    let fixture = Fixture::new();
    fixture.run("seed");
    let library = fixture.library();
    make_old(&library);
    let mut cleaner = fixture.spawn("cleanup-race", "cleaner");
    fixture.wait("observed", &mut cleaner);
    // The publishing child first observes the old file during its own startup
    // cleanup; reset its age so that child builds via the two failed-load seams.
    File::options()
        .write(true)
        .open(&library)
        .expect("library")
        .set_modified(std::time::SystemTime::now())
        .expect("protect from startup cleanup");
    fixture.run("replace");
    fixture.signal("clean");
    cleaner.join();
    assert!(library.exists(), "fresh publication was removed");
    fixture.run("fresh");
}

fn replacement_driver() {
    let mut loaded = interface();
    let path = loaded.create_shared_library().expect("first load");
    let before = std::fs::read(&path).expect("old bytes");
    let mut child = spawn("replacement-child", "replacement", &root(), &control());
    child
        .wait(&control().join("replace-ready"))
        .unwrap_or_else(|error| panic!("{error}"));
    let staging: Vec<_> = entries(&root().join("qis-programs-v2"))
        .into_iter()
        .filter(|p| kind(p) == Some(EntryKind::Staging))
        .collect();
    assert_eq!(staging.len(), 1);
    // Distinguish complete linked objects without modifying the mapped object.
    File::options()
        .append(true)
        .open(staging[0].join(format!("program.{LIB_SUFFIX}")))
        .expect("staged object")
        .write_all(b"replacement test marker")
        .expect("mark replacement");
    signal(&control().join("replace-go"));
    child.join();
    let after = std::fs::read(&path).expect("published bytes");
    let outcome = std::fs::read_to_string(control().join("replacement-outcome")).expect("outcome");
    match outcome.as_str() {
        "replaced" => {
            assert_ne!(before, after);
            assert!(after.ends_with(b"replacement test marker"));
        }
        "refused" => {
            assert_eq!(before, after);
            assert_eq!(
                std::env::consts::OS,
                "windows",
                "unexpected refusal on Unix"
            );
        }
        _ => panic!("unexpected publication outcome: {outcome}"),
    }
    loaded
        .collect_operations()
        .expect("existing process handle still works");
    spawn("fresh", "fresh", &root(), &control()).join();
    assert_eq!(count(Site::Link), 1);
    assert_layout(&root().join("qis-programs-v2"), true);
}

fn replacement_child() {
    force_build();
    pause(Site::Publish, 1, "replace-ready", "replace-go");
    load().expect("replacement or C5 existing-file fallback");
    for site in [
        Site::InitialLoad,
        Site::PostLockLoad,
        Site::Link,
        Site::Publish,
    ] {
        assert_eq!(count(site), 1);
    }
    let outcome = if count(Site::Published) == 1 {
        assert_eq!(count(Site::FinalLoad), 1);
        "replaced"
    } else {
        assert_eq!(count(Site::RenameLoad), 1);
        "refused"
    };
    std::fs::write(control().join("replacement-outcome"), outcome).expect("outcome");
}

#[test]
fn replacing_library_preserves_existing_process_handle() {
    Fixture::new().run("replacement-driver");
}

fn error_path(mode: &str) {
    match mode {
        "rename-present" | "rename-load-failure" => {
            force_build();
            fail(Site::Publish);
            if mode == "rename-load-failure" {
                fail(Site::RenameLoad);
            }
        }
        "rename-absent" | "rename-invalid" => fail(Site::Publish),
        "unsupported" => faults::on(Site::Lock, 1, Action::Fail(ErrorKind::Unsupported)),
        "lock-error" => faults::on(Site::Lock, 1, Action::Fail(ErrorKind::PermissionDenied)),
        "manifest-error" => fail(Site::Manifest),
        _ => unreachable!(),
    }
    let result = load();
    match mode {
        "rename-present" => {
            result.expect("use destination");
            for site in [
                Site::InitialLoad,
                Site::PostLockLoad,
                Site::Publish,
                Site::RenameLoad,
            ] {
                assert_eq!(count(site), 1);
            }
            assert_eq!(count(Site::Published), 0);
        }
        "rename-absent" | "rename-invalid" | "rename-load-failure" => {
            let error = result.expect_err("publication must fail");
            assert!(
                error.to_string().contains("Failed to publish program"),
                "{error}"
            );
            assert!(error.to_string().contains("injected Publish"), "{error}");
            assert_eq!(count(Site::Publish), 1);
            if mode != "rename-absent" {
                for site in [Site::InitialLoad, Site::PostLockLoad, Site::RenameLoad] {
                    assert_eq!(count(site), 1);
                }
            }
        }
        "lock-error" => {
            let error = result.expect_err("lock error");
            assert!(
                error.to_string().contains("Failed to lock program cache"),
                "{error}"
            );
            assert!(error.to_string().contains("injected Lock"), "{error}");
            assert_eq!(count(Site::Lock), 1);
        }
        "unsupported" => {
            result.expect("unsupported proceeds");
            assert_eq!(count(Site::Lock), 1);
        }
        "manifest-error" => {
            result.expect("manifest failure is best effort");
            assert_eq!(count(Site::Manifest), 1);
            assert_eq!(count(Site::FinalLoad), 1);
            assert_layout(&root().join("qis-programs-v2"), false);
        }
        _ => unreachable!(),
    }
    assert_eq!(count(Site::Link), usize::from(mode != "lock-error"));
    assert_no_staging(&root().join("qis-programs-v2"));
}

#[test]
fn publication_lock_and_manifest_error_paths() {
    for mode in [
        "rename-present",
        "rename-absent",
        "rename-load-failure",
        "rename-invalid",
        "unsupported",
        "lock-error",
        "manifest-error",
    ] {
        let fixture = Fixture::new();
        let original = if matches!(
            mode,
            "rename-present" | "rename-invalid" | "rename-load-failure"
        ) {
            fixture.run("seed");
            let library = fixture.library();
            if mode == "rename-invalid" {
                std::fs::write(&library, b"invalid library").expect("corrupt unloaded library");
            }
            Some((library.clone(), std::fs::read(library).expect("bytes")))
        } else {
            None
        };
        fixture.run(mode);
        if let Some((library, bytes)) = original {
            let after = std::fs::read(library)
                .unwrap_or_else(|error| panic!("{mode}: destination must survive: {error}"));
            assert_eq!(after, bytes);
        }
        if matches!(
            mode,
            "unsupported" | "manifest-error" | "rename-present" | "rename-load-failure"
        ) {
            fixture.run("fresh");
        }
    }
}

#[test]
fn owned_name_grammar() {
    let digest = "0123456789abcdef".repeat(4);
    for (name, expected) in [
        (format!("program_{digest}.{LIB_SUFFIX}"), EntryKind::Library),
        (format!("program_{digest}.manifest"), EntryKind::Manifest),
        (format!(".program_{digest}.aB09zZ.tmp"), EntryKind::Staging),
    ] {
        assert_eq!(parse_entry(&name), Some((digest.as_str(), expected)));
    }
    for byte in 0..=255 {
        let bucket = format!("{byte:02x}");
        assert_eq!(
            parse_entry(&format!("lock_{bucket}.lock")),
            Some((bucket.as_str(), EntryKind::Lock))
        );
    }
    assert_eq!(parse_entry(&format!("program_{digest}.lock")), None);
    for name in [
        "",
        "program",
        ".program_",
        "program_💣.so",
        "program_../x.so",
        "lock_0.lock",
        "lock_000.lock",
        "lock_AB.lock",
        "lock_ab.lock.x",
    ] {
        assert_eq!(parse_entry(name), None);
    }
}

fn raw_lock_error(mode: &str) {
    let code = std::env::var("PECOS_CACHE_TEST_ID")
        .expect("raw error")
        .parse()
        .expect("error number");
    faults::on(Site::Lock, 1, Action::RawError(code));
    if mode == "raw-lock-unsupported" {
        load().expect("raw unsupported error must allow unlocked compilation");
        assert_eq!(count(Site::Link), 1);
        assert_eq!(count(Site::FinalLoad), 1);
        assert_eq!(count(Site::LockPoll), 0);
        assert_layout(&root().join("qis-programs-v2"), true);
    } else {
        let cache = root().join("qis-programs-v2");
        std::fs::create_dir_all(&cache).expect("cache");
        let library = cache.join(format!("program_{}.{LIB_SUFFIX}", "a".repeat(64)));
        std::fs::write(&library, b"old publication").expect("fixture");
        make_old(&library);
        program_cache::cleanup_old_cache_files(&cache, 86400);
        assert_eq!(
            std::fs::read(library).expect("unsupported cleanup preserves file"),
            b"old publication"
        );
        assert_eq!(count(Site::CleanupObserved), 1);
        assert_eq!(count(Site::CleanupLocked), 0);
    }
    assert_eq!(count(Site::Lock), 1);
}

#[test]
fn raw_unsupported_lock_errors_allow_compilation_and_skip_cleanup() {
    #[cfg(unix)]
    let codes = [libc::ENOLCK, libc::ENOTSUP];
    #[cfg(windows)]
    let codes = [
        program_cache::ERROR_NOT_SUPPORTED,
        program_cache::ERROR_INVALID_FUNCTION,
    ];
    for code in codes {
        for mode in ["raw-lock-unsupported", "raw-cleanup-unsupported"] {
            let fixture = Fixture::new();
            fixture.spawn(mode, &code.to_string()).join();
            if mode == "raw-lock-unsupported" {
                fixture.run("fresh");
            }
        }
    }
}

#[test]
fn cleanup_holds_lock_through_deletion() {
    let fixture = Fixture::new();
    fixture.run("seed");
    let library = fixture.library();
    make_old(&library);
    let mut cleaner = fixture.spawn("cleanup-locked", "cleaner");
    fixture.wait("cleanup-held", &mut cleaner);
    assert!(
        !library.exists(),
        "cleanup must delete the old library before the barrier"
    );
    fixture.run("probe");
    fixture.signal("finish-cleanup");
    cleaner.join();
}

#[test]
fn compilation_holds_lock_through_final_load() {
    let fixture = Fixture::new();
    let mut compiler = fixture.spawn("final-locked", "compiler");
    fixture.wait("final-ready", &mut compiler);
    assert!(fixture.library().with_extension("manifest").exists());
    fixture.run("probe");
    fixture.signal("finish-final");
    compiler.join();
    assert_layout(&fixture.cache(), true);
    fixture.run("fresh");
}

fn noisy_child(mode: &str) {
    // Exceed pipe capacity on both streams before signalling the parent.
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&vec![b'o'; 256 * 1024])
        .expect("large stdout");
    writeln!(stdout, "\nCHILD_STDOUT_END").expect("stdout marker");
    stdout.flush().expect("stdout flush");
    drop(stdout);
    let mut stderr = std::io::stderr().lock();
    stderr
        .write_all(&vec![b'e'; 256 * 1024])
        .expect("large stderr");
    writeln!(stderr, "\nCHILD_STDERR_END").expect("stderr marker");
    stderr.flush().expect("stderr flush");
    drop(stderr);
    signal(&control().join("noisy-ready"));
    match mode {
        "noisy-failure" => panic!("intentional child failure"),
        "noisy-watchdog" => faults::wait_for(&control().join("never-release")),
        _ => (),
    }
}

#[test]
fn child_helper_drains_pipes_and_reports_failure_output() {
    let child = spawn_test_child("executor::cache_tests::nonexistent_child_test", &[]);
    let error = finish_test_child(child, Duration::from_secs(60))
        .expect_err("a misspelled filter must not silently succeed");
    assert!(
        error.contains("child did not run exactly one test"),
        "{error}"
    );

    for mode in ["noisy-success", "noisy-failure", "noisy-watchdog"] {
        let fixture = Fixture::new();
        let mut child = fixture.spawn(mode, "noisy");
        fixture.wait("noisy-ready", &mut child);
        let budget = if mode == "noisy-watchdog" {
            Duration::ZERO
        } else {
            Duration::from_secs(60)
        };
        let result = finish_test_child(child.0.take().expect("child"), budget);
        if mode == "noisy-success" {
            result.expect("verbose child must not block on pipes");
        } else {
            let error = result.expect_err("child failure must be reported");
            assert!(error.contains("CHILD_STDOUT_END"));
            assert!(error.contains("CHILD_STDERR_END"));
            if mode == "noisy-watchdog" {
                assert!(error.contains("child watchdog expired"));
            } else {
                assert!(error.contains("child failed"));
                assert!(error.contains("intentional child failure"));
            }
        }
    }
}

#[test]
fn cache_path_with_comma_keeps_linker_outputs_in_staging() {
    let fixture = Fixture {
        root: tempfile::Builder::new()
            .prefix("qis,cache ")
            .tempdir()
            .expect("cache path with comma and space"),
        control: tempfile::tempdir().expect("control"),
    };
    fixture.run("seed");
    assert_layout(&fixture.cache(), true);
    assert_eq!(entries(fixture.root.path()), vec![fixture.cache()]);
    fixture.run("fresh");
}

#[test]
fn barrier_wait_reports_early_child_exit() {
    for mode in ["noisy-success", "noisy-failure"] {
        let fixture = Fixture::new();
        let mut child = fixture.spawn(mode, "early-exit");
        let error = child
            .wait(&fixture.control.path().join("never-signalled"))
            .expect_err("a child that exits without signalling must fail the barrier");
        assert!(error.contains("CHILD_STDOUT_END"));
        assert!(error.contains("CHILD_STDERR_END"));
        assert!(!error.contains("watchdog"), "{error}");
        if mode == "noisy-failure" {
            assert!(error.contains("child failed"));
            assert!(error.contains("intentional child failure"));
        } else {
            assert!(error.contains("child exited"));
            assert!(error.contains("before signalling"));
        }
    }
}

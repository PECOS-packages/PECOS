//! Bucket contracts use production digests, with no compilation during selection.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn variant(value: usize) -> QisHeliosInterface {
    let mut interface = interface();
    interface.program = format!("define i64 @qmain(i64 %arg) {{ ret i64 {value} }}").into_bytes();
    interface
}

fn selected() -> [(usize, String); 3] {
    QisHeliosInterface::get_qis_ffi_lib_singleton().expect("runtime");
    let mut seen = BTreeMap::<String, (usize, String)>::new();
    let mut digests = BTreeSet::new();
    let mut pair = None;
    for value in 0..257 {
        let digest = variant(value).program_digest().expect("production digest");
        assert!(digests.insert(digest.clone()), "full digest collision");
        let bucket = digest[..2].to_owned();
        if let Some(first) = seen.get(&bucket) {
            pair = Some([first.clone(), (value, digest.clone())]);
        } else {
            seen.insert(bucket, (value, digest));
        }
        if let Some([first, second]) = &pair
            && let Some(other) = seen.values().find(|(_, d)| d[..2] != first.1[..2])
        {
            assert_ne!(first.1, second.1);
            assert_ne!(first.1, other.1);
            assert_ne!(second.1, other.1);
            assert_eq!(first.1[..2], second.1[..2]);
            assert_ne!(first.1[..2], other.1[..2]);
            return [first.clone(), second.clone(), other.clone()];
        }
    }
    panic!("257 distinct production digests did not yield a collision and a different bucket");
}

fn child(mode: &str, id: &str) -> ChildRun {
    spawn(mode, id, &root(), &control())
}

fn wait(child: &mut ChildRun, name: &str) {
    child
        .wait(&control().join(name))
        .unwrap_or_else(|error| panic!("{error}"));
}

fn hold(digest: &str) {
    let cache = root().join("qis-programs-v2");
    std::fs::create_dir_all(&cache).expect("cache");
    let guard = CompilationLock::acquire(&cache, digest, Duration::ZERO).expect("lock");
    assert!(matches!(guard, LockOutcome::Acquired(_)));
    signal(&control().join(format!("acquired-{digest}")));
    faults::wait_for(&control().join("release"));
    drop(guard);
}

pub(super) fn run(mode: &str) {
    match mode {
        "bucket-bounded" => bounded(),
        "bucket-holder" => hold(&std::env::var("PECOS_CACHE_TEST_ID").expect("digest")),
        "bucket-processes" => processes(),
        "bucket-threads" => threads(),
        "bucket-independent" => independent(),
        "bucket-budget-absent" | "bucket-budget-present" => budget_driver(mode),
        "bucket-cleanup" => cleanup(),
        "bucket-isolation" => isolation(),
        "bucket-isolation-check" => isolation_check(),
        "bucket-seed" | "bucket-waiter" | "bucket-timeout-absent" | "bucket-timeout-present" => {
            compile_child(mode);
        }
        _ => panic!("unknown bucket mode {mode}"),
    }
}

fn bounded() {
    let programs = selected();
    assert_eq!(count(Site::Link), 0);
    let cache = get_persistent_cache_dir().expect("cache");
    assert_eq!(entries(&cache), [] as [PathBuf; 0]);
    for (value, digest) in &programs {
        let path = variant(*value)
            .create_shared_library()
            .expect("compile selected");
        assert_eq!(path, cache.join(format!("program_{digest}.{LIB_SUFFIX}")));
        assert!(!cache.join(format!("program_{digest}.lock")).exists());
    }
    assert_eq!(count(Site::Link), 3);
    let expected: BTreeSet<_> = programs
        .iter()
        .map(|(_, d)| cache.join(format!("lock_{}.lock", &d[..2])))
        .collect();
    assert_eq!(expected.len(), 2);
    let actual: BTreeSet<_> = entries(&cache)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|ext| ext == "lock"))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(entries(&cache).len(), 8);
    assert_no_staging(&cache);
}

fn compile_child(mode: &str) {
    let value = std::env::var("PECOS_CACHE_TEST_ID")
        .expect("variant")
        .parse()
        .expect("constant");
    if mode == "bucket-waiter" {
        pause(Site::LockPoll, 1, "contended", "poll-again");
    }
    if mode.starts_with("bucket-timeout-") {
        faults::set_budget(Duration::from_millis(5));
        if mode == "bucket-timeout-present" {
            fail(Site::InitialLoad);
        }
    }
    let result = variant(value).create_shared_library();
    if mode == "bucket-timeout-absent" {
        let error = result.expect_err("timeout for unpublished digest");
        assert!(error.to_string().contains("Timeout waiting"), "{error}");
    } else {
        result.expect("selected program load");
    }
    if mode.starts_with("bucket-timeout-") {
        assert_eq!(count(Site::Budget), 1);
        assert!(count(Site::LockPoll) > 0);
        assert_eq!(count(Site::Staged), 0);
        assert_eq!(count(Site::Link), 0);
        assert_eq!(count(Site::PostLockLoad), 0);
        if mode == "bucket-timeout-present" {
            let digest = variant(value).program_digest().expect("digest");
            assert_eq!(
                faults::hits(Site::TimeoutLoad),
                [root()
                    .join("qis-programs-v2")
                    .join(format!("program_{digest}.{LIB_SUFFIX}"))]
            );
        }
    } else {
        assert_eq!(count(Site::Link), 1);
        if mode == "bucket-waiter" {
            assert!(count(Site::LockPoll) > 0);
        }
    }
}

fn processes() {
    let [(_, first), (second, _), _] = selected();
    let mut holder = child("bucket-holder", &first);
    wait(&mut holder, &format!("acquired-{first}"));
    let mut waiter = child("bucket-waiter", &second.to_string());
    wait(&mut waiter, "contended");
    signal(&control().join("release"));
    holder.join();
    signal(&control().join("poll-again"));
    waiter.join();
}

fn threads() {
    let [(_, first), (second, _), _] = selected();
    get_persistent_cache_dir().expect("initialize cleanup before holding a lock");
    let ready = control().join(format!("acquired-{first}"));
    let holder = std::thread::spawn(move || hold(&first));
    wait_thread(&ready, &holder);
    pause(Site::LockPoll, 1, "contended", "poll-again");
    let waiter = std::thread::spawn(move || variant(second).create_shared_library());
    wait_thread(&control().join("contended"), &waiter);
    assert_eq!(count(Site::Staged), 0);
    signal(&control().join("release"));
    holder.join().expect("holder");
    signal(&control().join("poll-again"));
    waiter
        .join()
        .expect("waiter")
        .expect("compile after release");
    assert_eq!(count(Site::Link), 1);
}

fn wait_thread<T>(path: &Path, worker: &std::thread::JoinHandle<T>) {
    let start = std::time::Instant::now();
    while !path.exists() {
        assert!(
            !worker.is_finished() || path.exists(),
            "worker exited before signalling {}",
            path.display()
        );
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "thread barrier watchdog"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn independent() {
    let [(_, first), _, (_, other)] = selected();
    let mut a = child("bucket-holder", &first);
    let mut b = child("bucket-holder", &other);
    wait(&mut a, &format!("acquired-{first}"));
    wait(&mut b, &format!("acquired-{other}"));
    signal(&control().join("release"));
    a.join();
    b.join();
}

fn budget_driver(mode: &str) {
    let [(_, first), (second, _), _] = selected();
    let present = mode == "bucket-budget-present";
    if present {
        child("bucket-seed", &second.to_string()).join();
    }
    let mut holder = child("bucket-holder", &first);
    wait(&mut holder, &format!("acquired-{first}"));
    child(
        if present {
            "bucket-timeout-present"
        } else {
            "bucket-timeout-absent"
        },
        &second.to_string(),
    )
    .join();
    holder.kill();
}

fn old_entries(cache: &Path, bucket: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for suffix in ["1", "2", "3"] {
        let digest = format!("{bucket}{}", suffix.repeat(62));
        let library = cache.join(format!("program_{digest}.{LIB_SUFFIX}"));
        let manifest = library.with_extension("manifest");
        for path in [&library, &manifest] {
            std::fs::write(path, b"old entry").expect("fixture");
        }
        let staging = program_cache::staging_dir(cache, &digest)
            .expect("staging")
            .keep();
        std::fs::write(staging.join("partial"), b"partial output").expect("partial");
        for path in [library, manifest, staging] {
            make_old(&path);
            paths.push(path);
        }
    }
    paths
}

fn cleanup() {
    let cache = get_persistent_cache_dir().expect("cache");
    let a = old_entries(&cache, "aa");
    let b = old_entries(&cache, "bb");
    let fresh = cache.join(format!("program_{}.manifest", "c".repeat(64)));
    std::fs::write(&fresh, b"fresh").expect("fresh");
    let locks: Vec<_> = ["aa", "dd", "ee"]
        .map(|bucket| bucket_lock_path(&cache, bucket))
        .into();
    for path in &locks {
        std::fs::write(path, b"permanent lock").expect("lock");
        make_old(path);
    }
    let mut holder = child("bucket-holder", &"a".repeat(64));
    wait(&mut holder, &format!("acquired-{}", "a".repeat(64)));
    program_cache::cleanup_old_cache_files(&cache, 86400);
    assert_eq!(
        faults::hits(Site::Lock),
        [
            bucket_lock_path(&cache, "aa"),
            bucket_lock_path(&cache, "bb")
        ]
    );
    assert!(a.iter().all(|p| p.exists()));
    assert!(b.iter().all(|p| !p.exists()));
    assert!(fresh.exists());
    assert!(!bucket_lock_path(&cache, "cc").exists());
    signal(&control().join("release"));
    holder.join();
    program_cache::cleanup_old_cache_files(&cache, 86400);
    assert_eq!(
        faults::hits(Site::Lock),
        [
            bucket_lock_path(&cache, "aa"),
            bucket_lock_path(&cache, "bb"),
            bucket_lock_path(&cache, "aa")
        ]
    );
    assert!(a.iter().all(|p| !p.exists()));
    assert!(fresh.exists());
    assert!(!bucket_lock_path(&cache, "cc").exists());
    for path in &locks {
        assert_eq!(
            std::fs::read(path).expect("lock retained without truncation"),
            b"permanent lock"
        );
    }
    assert!(bucket_lock_path(&cache, "bb").exists());
    assert_eq!(entries(&cache).len(), 5);
}

fn isolation() {
    child("seed", "seed").join();
    let old = root().join("qis-programs");
    std::fs::rename(root().join("qis-programs-v2"), &old).expect("old cache");
    let digest = digest();
    let legacy_lock = old.join(format!("program_{digest}.lock"));
    std::fs::write(&legacy_lock, b"legacy lock").expect("legacy lock");
    let staging = program_cache::staging_dir(&old, &digest)
        .expect("old staging")
        .keep();
    std::fs::write(staging.join("partial"), b"old partial").expect("old partial");
    let before: BTreeMap<_, _> = entries(&old)
        .into_iter()
        .map(|path| {
            make_old(&path);
            let bytes = if path.is_file() {
                std::fs::read(&path).expect("old bytes")
            } else {
                Vec::new()
            };
            let modified = std::fs::metadata(&path)
                .expect("metadata")
                .modified()
                .expect("mtime");
            (path, (bytes, modified))
        })
        .collect();
    let lock = File::options()
        .read(true)
        .write(true)
        .open(legacy_lock)
        .expect("legacy lock handle");
    lock.try_lock().expect("hold legacy digest lock");
    child("bucket-isolation-check", "check").join();
    drop(lock);
    assert_eq!(
        entries(&old).into_iter().collect::<BTreeSet<_>>(),
        before.keys().cloned().collect()
    );
    for (path, (bytes, modified)) in before {
        if path.is_file() {
            assert_eq!(std::fs::read(&path).expect("old file untouched"), bytes);
        }
        assert_eq!(
            std::fs::metadata(path)
                .expect("old entry retained")
                .modified()
                .expect("mtime"),
            modified
        );
    }
    assert_eq!(
        std::fs::read(staging.join("partial")).expect("old staging retained"),
        b"old partial"
    );
    assert_layout(&root().join("qis-programs-v2"), true);
}

fn isolation_check() {
    let path = load().expect("compile into isolated directory");
    let cache = root().join("qis-programs-v2");
    assert_eq!(path.parent(), Some(cache.as_path()));
    assert_eq!(count(Site::InitialLoad), 0);
    assert_eq!(count(Site::Link), 1);
    assert_eq!(
        faults::hits(Site::Lock),
        [bucket_lock_path(&cache, &digest())]
    );
    program_cache::cleanup_old_cache_files(&cache, 86400);
}

#[test]
fn bounded_real_digests() {
    Fixture::new().run("bucket-bounded");
}

#[test]
fn same_bucket_process_exclusion() {
    Fixture::new().run("bucket-processes");
}

#[test]
fn same_bucket_thread_exclusion() {
    Fixture::new().run("bucket-threads");
}

#[test]
fn different_buckets_acquire_before_release() {
    Fixture::new().run("bucket-independent");
}

#[test]
fn same_bucket_timeout_rechecks_without_compiling() {
    Fixture::new().run("bucket-budget-absent");
    Fixture::new().run("bucket-budget-present");
}

#[test]
fn cleanup_groups_old_entries_by_bucket() {
    Fixture::new().run("bucket-cleanup");
}

#[test]
fn old_directory_is_isolated() {
    Fixture::new().run("bucket-isolation");
}

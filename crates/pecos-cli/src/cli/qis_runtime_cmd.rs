//! Cargo is the authority for runtime artifact paths; staging is PECOS-owned.
use super::build_context::BuildContext;
use cargo_metadata::{Message, Metadata};
use pecos_build::Result;
use pecos_build::errors::Error;
use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const STAGING_TEMP_PREFIX: &str = ".pecos-runtime-";

pub struct RuntimeArtifacts {
    pub directory: PathBuf,
    pub libraries: Vec<PathBuf>,
}

pub fn prepare(context: &BuildContext, tests: bool) -> Result<RuntimeArtifacts> {
    let mut metadata_command = context.cargo(&["metadata", "--no-deps", "--format-version", "1"]);
    let output = metadata_command.output()?;
    if !output.status.success() {
        return Err(Error::Config(format!(
            "{metadata_command:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout).map_err(|error| {
        Error::Config(format!("Invalid output from {metadata_command:?}: {error}"))
    })?;
    let package = metadata
        .packages
        .iter()
        .find(|package| package.name == "pecos-qis")
        .ok_or_else(|| {
            Error::Config(
                "cargo metadata did not report pecos-qis; run from the PECOS workspace".into(),
            )
        })?;
    let mut libraries = declared_libraries(&package.metadata)?;
    let mut args = vec![
        "build",
        "--message-format=json-render-diagnostics",
        "-p",
        "pecos-qis",
        "--no-default-features",
        "--features",
        "pecos-qis/selene",
    ];
    if tests {
        args.extend(["-p", "pecos-qis-test-runtime"]);
        libraries.insert("pecos_qis_test_runtime".into());
    }
    args.extend(context.profile.cargo_args());
    let mut command = context.cargo(&args);
    let description = format!("{command:?}");
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::Config(format!("Missing stdout from {description}")))?;
    let artifacts = parse_artifacts(std::io::BufReader::new(stdout), &libraries, &description);
    let status = child.wait()?;
    if !status.success() {
        return Err(Error::Config(format!("{description} failed ({status})")));
    }
    let destination = metadata
        .target_directory
        .as_std_path()
        .join("pecos-qis-runtime")
        .join(context.profile.directory())
        .join(if tests { "tests" } else { "runtime" });
    let libraries = copy_named_artifacts(&artifacts?, &destination).map_err(|error| {
        Error::Config(format!(
            "Could not stage artifacts from {description}: {error}"
        ))
    })?;
    Ok(RuntimeArtifacts {
        directory: destination,
        libraries,
    })
}

/// Read `[package.metadata.pecos] runtime-libraries` from the pecos-qis manifest.
fn declared_libraries(package_metadata: &serde_json::Value) -> Result<BTreeSet<String>> {
    let libraries: BTreeSet<String> = serde_json::from_value(
        package_metadata["pecos"]["runtime-libraries"].clone(),
    )
    .map_err(|error| {
        Error::Config(format!(
            "Invalid pecos-qis package.metadata.pecos.runtime-libraries: {error}; \
                     provide an array of library names"
        ))
    })?;
    if libraries.is_empty() {
        return Err(Error::Config(
            "pecos-qis package.metadata.pecos.runtime-libraries is empty; \
             list the QIS runtime library names"
                .into(),
        ));
    }
    Ok(libraries)
}

fn parse_artifacts(
    reader: impl BufRead,
    libraries: &BTreeSet<String>,
    command: &str,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut artifacts: BTreeMap<String, BTreeSet<(PathBuf, String)>> = BTreeMap::new();
    let mut import_libraries = BTreeSet::new();
    let mut ffi_filenames = BTreeSet::new();
    for message in Message::parse_stream(reader) {
        match message.map_err(|error| {
            Error::Config(format!("Invalid Cargo message from {command}: {error}"))
        })? {
            Message::CompilerArtifact(artifact) if libraries.contains(&artifact.target.name) => {
                if artifact.target.name == "pecos_qis_ffi" {
                    ffi_filenames.extend(artifact.filenames.iter().cloned());
                }
                let paths = artifact
                    .filenames
                    .iter()
                    .filter_map(|file| match file.extension() {
                        Some(extension @ ("so" | "dylib" | "dll")) => {
                            Some((file, extension.to_owned()))
                        }
                        _ => None,
                    });
                for (file, extension) in paths {
                    if artifact.target.name == "pecos_qis_ffi" && extension == "dll" {
                        import_libraries.extend(
                            artifact
                                .filenames
                                .iter()
                                .filter(|path| path.as_str().ends_with(".dll.lib"))
                                .map(|path| path.as_std_path().to_path_buf()),
                        );
                    }
                    artifacts
                        .entry(artifact.target.name.clone())
                        .or_default()
                        .insert((file.as_std_path().to_path_buf(), extension));
                }
            }
            _ => {}
        }
    }
    let mut files = BTreeMap::new();
    for name in libraries {
        let paths = artifacts.remove(name).unwrap_or_default();
        if paths.len() != 1 {
            return Err(Error::Config(format!(
                "Expected one runtime artifact for {name}, found {} ({paths:?}) from {command}; rebuild with the required runtime features",
                paths.len()
            )));
        }
        if name == "pecos_qis_ffi" {
            for (ffi, extension) in &paths {
                if extension == "dll" {
                    if ffi
                        .file_name()
                        .is_none_or(|name| name != "pecos_qis_ffi.dll")
                        || import_libraries.iter().any(|path| {
                            path.file_name()
                                .is_none_or(|name| name != "pecos_qis_ffi.dll.lib")
                        })
                    {
                        return Err(Error::Config(format!(
                            "Cannot stage pecos_qis_ffi DLL {} with import libraries {import_libraries:?} from {command}: the import library names the DLL it was linked with, so the pair cannot be renamed; rebuild with canonical pecos_qis_ffi.dll and pecos_qis_ffi.dll.lib names",
                            ffi.display()
                        )));
                    }
                    if import_libraries.len() != 1 {
                        return Err(Error::Config(format!(
                            "Expected one pecos_qis_ffi import library in Cargo filenames, found {} ({import_libraries:?}) from {command}; PECOS stages the MSVC .dll.lib import library; Cargo reported {ffi_filenames:?}; rebuild for an MSVC Windows target",
                            import_libraries.len()
                        )));
                    }
                    for source in &import_libraries {
                        files.insert(format!("{name}.dll.lib"), source.clone());
                    }
                }
            }
        }
        for (source, extension) in paths {
            let prefix = if extension == "dll" { "" } else { "lib" };
            files.insert(format!("{prefix}{name}.{extension}"), source);
        }
    }
    Ok(files)
}

/// Copy already-staged libraries, preserving their canonical filenames.
pub fn copy_artifacts(files: &[PathBuf], destination: &Path) -> Result<Vec<PathBuf>> {
    let mut named = BTreeMap::new();
    for file in files {
        let name = file.file_name().ok_or_else(|| {
            Error::Config(format!(
                "Runtime artifact {} has no filename; rebuild it",
                file.display()
            ))
        })?;
        if let Some(previous) = named.insert(name.to_owned(), file.clone()) {
            return Err(Error::Config(format!(
                "Runtime artifacts {} and {} map to {}; provide one artifact per staged name",
                previous.display(),
                file.display(),
                Path::new(name).display()
            )));
        }
    }
    copy_named_artifacts(&named, destination)
}

fn copy_named_artifacts(
    files: &BTreeMap<impl AsRef<Path>, PathBuf>,
    destination: &Path,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(destination)
        .map_err(|error| staging_error("create directory", destination, &error))?;
    let mut staged = Vec::with_capacity(files.len());
    for (name, file) in files {
        let path = destination.join(name);
        let temporary = tempfile::Builder::new()
            .prefix(STAGING_TEMP_PREFIX)
            .tempfile_in(destination)
            .map_err(|error| staging_error("create temporary file in", destination, &error))?;
        std::fs::copy(file, temporary.path()).map_err(|error| {
            Error::Config(format!(
                "Cannot copy runtime artifact {} to {}: {error}; rebuild the runtime libraries",
                file.display(),
                temporary.path().display()
            ))
        })?;
        temporary.persist(&path).map_err(|error| {
            let remedy = if cfg!(windows) {
                "close processes that have the library loaded (for example a Python session using pecos_rslib), then retry runtime staging"
            } else {
                "check the path and permissions, then retry runtime staging"
            };
            Error::Config(format!(
                "Cannot rename runtime artifact {} to {}: {error}; {remedy}",
                error.file.path().display(),
                path.display()
            ))
        })?;
        staged.push(path);
    }
    for entry in std::fs::read_dir(destination)
        .map_err(|error| staging_error("list", destination, &error))?
    {
        let entry = entry.map_err(|error| staging_error("list entry in", destination, &error))?;
        if entry
            .file_name()
            .as_encoded_bytes()
            .starts_with(STAGING_TEMP_PREFIX.as_bytes())
        {
            continue;
        }
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|error| staging_error("inspect", &path, &error))?;
        if !staged.contains(&path) && (kind.is_file() || kind.is_symlink()) {
            std::fs::remove_file(&path).map_err(|error| staging_error("remove", &path, &error))?;
        }
    }
    Ok(staged)
}

fn staging_error(operation: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::Config(format!(
        "Cannot {operation} {}: {error}; check the path and permissions, then retry runtime staging",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    const RECORDING: &str = include_str!("fixtures/runtime-artifacts.jsonl");
    const COMMAND: &str =
        "cargo build --locked --message-format=json-render-diagnostics -p pecos-qis";

    fn libraries() -> BTreeSet<String> {
        [
            "pecos_qis_ffi",
            "selene_simple_runtime",
            "selene_soft_rz_runtime",
            "pecos_qis_test_runtime",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn runtime_library_declaration_must_be_a_non_empty_list() {
        for metadata in [
            serde_json::Value::Null,
            serde_json::json!({}),
            serde_json::json!({"pecos": {}}),
            serde_json::json!({"pecos": {"runtime-libraries": []}}),
            serde_json::json!({"pecos": {"runtime-libraries": "pecos_qis_ffi"}}),
        ] {
            let error = declared_libraries(&metadata)
                .expect_err("invalid declaration")
                .to_string();
            assert!(error.contains("runtime-libraries"), "{error}");
        }
        let declared = declared_libraries(
            &serde_json::json!({"pecos": {"runtime-libraries": ["pecos_qis_ffi"]}}),
        )
        .expect("valid declaration");
        assert_eq!(declared, BTreeSet::from(["pecos_qis_ffi".to_owned()]));
    }

    #[test]
    fn recorded_artifacts_select_shared_libraries() {
        let paths = parse_artifacts(RECORDING.as_bytes(), &libraries(), COMMAND)
            .expect("recorded artifacts");
        assert_eq!(paths.len(), 4);
        for name in libraries() {
            assert!(paths.values().any(
                |path| path.file_name().expect("filename") == format!("lib{name}.so").as_str()
            ));
        }
        assert!(
            paths
                .values()
                .all(|path| path.starts_with("/workspace/target/"))
        );
        // Fresh and repeated messages describe the same output, not ambiguity.
        let repeated = format!("{RECORDING}{RECORDING}");
        assert_eq!(
            parse_artifacts(repeated.as_bytes(), &libraries(), COMMAND).expect("same artifacts"),
            paths
        );
    }

    #[test]
    fn recorded_artifacts_reject_missing_and_ambiguous_outputs() {
        let missing = RECORDING
            .lines()
            .filter(|line| !line.contains("selene_soft_rz_runtime"))
            .collect::<Vec<_>>()
            .join("\n");
        let error = parse_artifacts(missing.as_bytes(), &libraries(), COMMAND)
            .expect_err("missing runtime")
            .to_string();
        for text in ["selene_soft_rz_runtime", "found 0", COMMAND] {
            assert!(error.contains(text), "{error}");
        }
        let different = RECORDING.replace("libpecos_qis_ffi.so", "libpecos_qis_ffi-other.so");
        let ambiguous = format!("{RECORDING}{different}");
        let error = parse_artifacts(ambiguous.as_bytes(), &libraries(), COMMAND)
            .expect_err("ambiguous runtime")
            .to_string();
        for text in ["pecos_qis_ffi", "found 2", COMMAND] {
            assert!(error.contains(text), "{error}");
        }
    }

    #[test]
    fn hashed_artifacts_stage_under_canonical_names() {
        for extension in ["so", "dylib", "dll"] {
            let temporary = tempfile::tempdir().expect("test directory");
            let deps = temporary.path().join("deps");
            std::fs::create_dir(&deps).expect("deps directory");
            let prefix = if extension == "dll" { "" } else { "lib" };
            let name = "selene_simple_runtime";
            let hashed = format!("{prefix}{name}-580b7c95dd39b09a.{extension}");
            let source = deps.join(&hashed);
            std::fs::write(&source, b"runtime").expect("source library");
            let line = RECORDING
                .lines()
                .find(|line| line.contains(name))
                .expect("recording");
            let mut message: serde_json::Value = serde_json::from_str(line).expect("JSON");
            message["filenames"] = serde_json::json!([source]);
            let artifacts = parse_artifacts(
                message.to_string().as_bytes(),
                &BTreeSet::from([name.into()]),
                COMMAND,
            )
            .expect("artifacts");
            let destination = temporary.path().join("staged");
            std::fs::create_dir(&destination).expect("destination");
            std::fs::write(destination.join(&hashed), b"obsolete").expect("old hashed output");
            let staged = copy_named_artifacts(&artifacts, &destination).expect("stage");
            let exact = destination.join(format!("{prefix}{name}.{extension}"));
            assert!(exact.is_file(), "exact-name lookup");
            assert_eq!(staged, vec![exact.clone()]);
            assert_eq!(std::fs::read(&exact).expect("staged bytes"), b"runtime");
            assert!(!destination.join(&hashed).exists());
            let native = temporary.path().join("_native");
            assert_eq!(
                copy_artifacts(&staged, &native).expect("Python CLI staging"),
                vec![native.join(exact.file_name().expect("canonical name"))]
            );
        }
    }

    #[test]
    fn staging_preserves_loaded_files_and_delays_cleanup() {
        let directory = tempfile::tempdir().expect("test directory");
        let source = directory.path().join("runtime.dll");
        std::fs::write(&source, b"new").expect("source");
        let destination = directory.path().join("staged");
        std::fs::create_dir(&destination).expect("destination");
        let target = destination.join("runtime.dll");
        std::fs::write(&target, b"old").expect("old library");
        let stale = destination.join("stale.dll.lib");
        std::fs::write(&stale, b"stale").expect("stale library");
        #[cfg(unix)]
        let mut loaded = std::fs::File::open(&target).expect("loaded library");
        copy_artifacts(std::slice::from_ref(&source), &destination).expect("stage");
        #[cfg(unix)]
        {
            use std::io::Read;
            let mut bytes = Vec::new();
            loaded.read_to_end(&mut bytes).expect("read old inode");
            assert_eq!(bytes, b"old");
        }
        assert_eq!(std::fs::read(&target).expect("new library"), b"new");
        assert!(!stale.exists());
        std::fs::write(&stale, b"stale").expect("stale library");
        std::fs::remove_file(&source).expect("missing source");
        let error = copy_artifacts(std::slice::from_ref(&source), &destination)
            .expect_err("copy fails")
            .to_string();
        for text in [
            "copy",
            &source.display().to_string(),
            &destination.display().to_string(),
        ] {
            assert!(error.contains(text), "{error}");
        }
        assert_eq!(
            std::fs::read(&target).expect("previous library remains"),
            b"new"
        );
        assert!(stale.exists(), "cleanup must wait until all copies succeed");
    }

    #[test]
    fn staging_io_errors_name_operation_and_path() {
        let directory = tempfile::tempdir().expect("test directory");
        let source = directory.path().join("runtime.dll");
        std::fs::write(&source, b"new").expect("source");
        let error = copy_artifacts(&[], &source)
            .expect_err("not a directory")
            .to_string();
        assert!(error.contains("create"), "{error}");
        assert!(error.contains(&source.display().to_string()), "{error}");
        let destination = directory.path().join("staged");
        let target = destination.join("runtime.dll");
        std::fs::create_dir_all(&target).expect("directory cannot be replaced by file");
        let error = copy_artifacts(&[source], &destination)
            .expect_err("rename fails")
            .to_string();
        for text in [
            "rename",
            "retry",
            if cfg!(windows) {
                "close processes"
            } else {
                "check the path and permissions"
            },
        ] {
            assert!(error.contains(text), "{error}");
        }
        assert!(error.contains(&target.display().to_string()), "{error}");
    }

    #[test]
    fn staging_preserves_other_stagers_temporary_files() {
        let directory = tempfile::tempdir().expect("test directory");
        let source = directory.path().join("runtime.dll");
        std::fs::write(&source, b"runtime").expect("source");
        let destination = directory.path().join("staged");
        std::fs::create_dir(&destination).expect("staging directory");
        let in_flight = destination.join(".pecos-runtime-other-stager");
        std::fs::write(&in_flight, b"in flight").expect("other stager temporary file");
        copy_artifacts(&[source], &destination).expect("stage");
        assert_eq!(
            std::fs::read(in_flight).expect("temporary file survives"),
            b"in flight"
        );
    }

    #[test]
    fn windows_ffi_pair_cannot_be_renamed() {
        let line = RECORDING
            .lines()
            .find(|line| line.contains("libpecos_qis_ffi.so"))
            .expect("FFI recording");
        for (dll, import) in [
            ("pecos_qis_ffi-hash.dll", "pecos_qis_ffi.dll.lib"),
            ("pecos_qis_ffi.dll", "pecos_qis_ffi-hash.dll.lib"),
            ("pecos_qis_ffi-hash.dll", "pecos_qis_ffi-hash.dll.lib"),
        ] {
            let mut message: serde_json::Value = serde_json::from_str(line).expect("JSON");
            let dll = format!("/cargo/deps/{dll}");
            let import = format!("/cargo/deps/{import}");
            message["filenames"] = serde_json::json!([dll, import]);
            let error = parse_artifacts(
                message.to_string().as_bytes(),
                &BTreeSet::from(["pecos_qis_ffi".into()]),
                COMMAND,
            )
            .expect_err("renaming breaks import library")
            .to_string();
            for text in [
                &dll,
                &import,
                "import library names the DLL it was linked with",
                "pair cannot be renamed",
                COMMAND,
            ] {
                assert!(error.contains(text), "{error}");
            }
        }
    }

    #[test]
    fn windows_gnu_import_error_lists_reported_files() {
        let line = RECORDING
            .lines()
            .find(|line| line.contains("libpecos_qis_ffi.so"))
            .expect("FFI recording");
        let mut message: serde_json::Value = serde_json::from_str(line).expect("JSON");
        message["filenames"] =
            serde_json::json!(["/cargo/pecos_qis_ffi.dll", "/cargo/libpecos_qis_ffi.dll.a"]);
        let error = parse_artifacts(
            message.to_string().as_bytes(),
            &BTreeSet::from(["pecos_qis_ffi".into()]),
            COMMAND,
        )
        .expect_err("GNU import unsupported")
        .to_string();
        for text in [
            "MSVC .dll.lib",
            "/cargo/pecos_qis_ffi.dll",
            "/cargo/libpecos_qis_ffi.dll.a",
            COMMAND,
        ] {
            assert!(error.contains(text), "{error}");
        }
    }

    #[test]
    fn windows_import_library_is_required_and_staged() {
        let temporary = tempfile::tempdir().expect("test directory");
        let directory = temporary.path();
        let dll = directory.join("pecos_qis_ffi.dll");
        let import_library = dll.with_extension("dll.lib");
        if import_library.exists() {
            std::fs::remove_file(&import_library).expect("remove prior import library");
        }
        let line = RECORDING
            .lines()
            .find(|line| line.contains("libpecos_qis_ffi.so"))
            .expect("FFI recording");
        let mut message: serde_json::Value = serde_json::from_str(line).expect("recorded JSON");
        message["filenames"] = serde_json::json!([dll]);
        let recording = message.to_string();
        let names = BTreeSet::from(["pecos_qis_ffi".into()]);
        let error = parse_artifacts(recording.as_bytes(), &names, COMMAND)
            .expect_err("missing Windows import library")
            .to_string();
        for text in ["pecos_qis_ffi", "import library", COMMAND] {
            assert!(error.contains(text), "{error}");
        }
        std::fs::write(&dll, b"dll").expect("fixture DLL");
        std::fs::write(&import_library, b"import").expect("fixture import library");
        assert!(
            parse_artifacts(recording.as_bytes(), &names, COMMAND).is_err(),
            "an adjacent unreported import library must not be accepted"
        );
        message["filenames"] =
            serde_json::json!([dll, import_library, directory.join("extra.dll.lib")]);
        assert!(parse_artifacts(message.to_string().as_bytes(), &names, COMMAND).is_err());
        message["filenames"] = serde_json::json!([dll, import_library]);
        let recording = message.to_string();
        let paths =
            parse_artifacts(recording.as_bytes(), &names, COMMAND).expect("Windows artifacts");
        let repeated = format!("{recording}\n{recording}");
        assert_eq!(
            parse_artifacts(repeated.as_bytes(), &names, COMMAND)
                .expect("repeated Windows artifacts"),
            paths
        );
        message["filenames"] =
            serde_json::json!([dll, directory.join("other/pecos_qis_ffi.dll.lib")]);
        let ambiguous = format!("{recording}\n{message}");
        let error = parse_artifacts(ambiguous.as_bytes(), &names, COMMAND)
            .expect_err("ambiguous imports across Cargo messages")
            .to_string();
        for text in ["pecos_qis_ffi", "import library", "found 2", COMMAND] {
            assert!(error.contains(text), "{error}");
        }
        assert_eq!(paths["pecos_qis_ffi.dll.lib"], import_library);
        assert_eq!(paths["pecos_qis_ffi.dll"], dll);
        let destination = directory.join("staged");
        std::fs::create_dir_all(&destination).expect("staging directory");
        std::fs::write(destination.join("stale.dll"), b"stale").expect("stale output");
        copy_named_artifacts(&paths, &destination).expect("stage artifacts");
        assert!(!destination.join("stale.dll").exists());
        assert_eq!(
            std::fs::read(destination.join("pecos_qis_ffi.dll.lib")).expect("staged import"),
            b"import"
        );
        assert_eq!(
            std::fs::read(destination.join("pecos_qis_ffi.dll")).expect("staged DLL"),
            b"dll"
        );
    }
}

//! Shared process environment for Cargo, runtime preparation, and Maturin.
use super::BuildProfile;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct BuildContext {
    pub profile: BuildProfile,
    env: BTreeMap<String, String>,
    removed: Vec<&'static str>,
}

impl BuildContext {
    pub fn new(profile: BuildProfile, extra_rustflags: Option<&str>) -> Self {
        let mut env = super::env_cmd::collect_env();
        let mut flags = std::env::var("RUSTFLAGS").unwrap_or_default();
        if profile == BuildProfile::Native {
            flags.push_str(" -C target-cpu=native");
        }
        if let Some(extra) = extra_rustflags {
            flags.push(' ');
            flags.push_str(extra);
        }
        if !flags.is_empty() {
            env.insert("RUSTFLAGS".into(), flags);
        }
        Self {
            profile,
            env,
            removed: Vec::new(),
        }
    }

    pub fn target_dir(&mut self, path: &Path) {
        self.env.insert(
            "CARGO_TARGET_DIR".into(),
            path.to_string_lossy().into_owned(),
        );
    }

    pub fn python_environment(&mut self, root: &Path) -> PathBuf {
        let bin = root.join(if cfg!(windows) {
            ".venv/Scripts"
        } else {
            ".venv/bin"
        });
        let separator = if cfg!(windows) { ';' } else { ':' };
        self.env.insert(
            "PATH".into(),
            format!(
                "{}{separator}{}",
                bin.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        );
        self.removed.push("CONDA_PREFIX");
        #[cfg(target_os = "macos")]
        {
            let flags = self.env.entry("RUSTFLAGS".into()).or_default();
            if !flags.contains("-rpath") {
                flags.push_str(" -C link-arg=-Wl,-rpath,/usr/lib");
            }
            self.removed.extend([
                "LIBRARY_PATH",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "DYLD_FALLBACK_LIBRARY_PATH",
            ]);
            self.env.insert("LIBRARY_PATH".into(), "/usr/lib".into());
            self.env.insert("CARGO_CACHE_RUSTC_INFO".into(), "0".into());
        }
        bin
    }

    pub fn apply(&self, command: &mut Command) {
        for key in &self.removed {
            command.env_remove(key);
        }
        command.envs(&self.env);
    }

    pub fn cargo(&self, args: &[&str]) -> Command {
        let mut command = Command::new("cargo");
        if let Some((subcommand, rest)) = args.split_first() {
            command.arg(subcommand);
            if matches!(
                *subcommand,
                "build" | "check" | "clippy" | "metadata" | "run" | "test"
            ) && !args
                .iter()
                .any(|arg| matches!(*arg, "--locked" | "--frozen" | "--offline"))
            {
                command.arg("--locked");
            }
            command.args(rest);
        }
        self.apply(&mut command);
        command
    }
}

//! TOML-aware reader/writer for `.cargo/config.toml`.
//!
//! Every in-repo writer (LLVM, cuQuantum, the Windows MSVC bootstrap) funnels
//! through here. Mutations go through `toml_edit`, so unrelated tables/keys are
//! preserved and we can never emit a duplicate `[env]` table -- which cargo
//! rejects as a TOML parse error, and which the previous independent
//! line-based writers could produce when run in sequence.

use crate::errors::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Table};

/// A structurally-parsed `.cargo/config.toml`. Open it, apply one or more
/// mutations, then `save()` once.
pub struct CargoConfig {
    path: PathBuf,
    original: String,
    doc: DocumentMut,
}

impl CargoConfig {
    /// Open the project's `.cargo/config.toml`, creating the `.cargo`
    /// directory if needed. A missing file parses as an empty document.
    ///
    /// # Errors
    /// Returns an error if `.cargo` cannot be created or the existing file is
    /// not valid TOML.
    pub fn open(project_root: &Path) -> Result<Self> {
        let cargo_dir = project_root.join(".cargo");
        fs::create_dir_all(&cargo_dir)?;
        let path = cargo_dir.join("config.toml");
        // Only a missing file means "start empty". A permission/UTF-8/other
        // read error must surface -- silently treating it as empty would let
        // save() clobber an existing, unreadable user config.
        let original = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(Error::Config(format!(
                    "could not read {}: {e}",
                    path.display()
                )));
            }
        };
        let doc = original
            .parse::<DocumentMut>()
            .map_err(|e| Error::Config(format!("{} is not valid TOML: {e}", path.display())))?;
        Ok(Self {
            path,
            original,
            doc,
        })
    }

    fn table_mut<'a>(doc: &'a mut DocumentMut, key: &str) -> Result<&'a mut Table> {
        doc.as_table_mut()
            .entry(key)
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .ok_or_else(|| Error::Config(format!("`{key}` in .cargo/config.toml is not a table")))
    }

    /// Set `[env].<key>`. With `force`, writes the
    /// `{ value = "...", force = true }` form so it overrides the inherited
    /// shell environment (required on Windows, where git-bash mangles the
    /// ambient value before cargo's child processes see it); otherwise a
    /// plain string.
    ///
    /// # Errors
    /// Returns an error if `[env]` exists but is not a table.
    pub fn set_env(&mut self, key: &str, value: &str, force: bool) -> Result<&mut Self> {
        let env = Self::table_mut(&mut self.doc, "env")?;
        if force {
            let mut inline = toml_edit::InlineTable::new();
            inline.insert("value", value.into());
            inline.insert("force", true.into());
            env[key] = toml_edit::value(inline);
        } else {
            env[key] = toml_edit::value(value);
        }
        Ok(self)
    }

    /// Set `[target.<triple>].linker`.
    ///
    /// # Errors
    /// Returns an error if `[target]` / `[target.<triple>]` exist but are not
    /// tables.
    pub fn set_target_linker(&mut self, triple: &str, linker: &str) -> Result<&mut Self> {
        let target = Self::table_mut(&mut self.doc, "target")?;
        // Implicit so it renders as `[target.<triple>]`, not a bare `[target]`.
        target.set_implicit(true);
        let triple_tbl = target
            .entry(triple)
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .ok_or_else(|| {
                Error::Config(format!(
                    "`target.{triple}` in .cargo/config.toml is not a table"
                ))
            })?;
        triple_tbl.set_implicit(false);
        triple_tbl["linker"] = toml_edit::value(linker);
        Ok(self)
    }

    /// Write back, but **only if the serialized content changed** -- this
    /// avoids bumping the file mtime on no-op runs, which build-freshness
    /// checks key off of. Returns `true` if the file was written.
    ///
    /// # Errors
    /// Returns an error if the file cannot be written.
    pub fn save(self) -> Result<bool> {
        let rendered = self.doc.to_string();
        if rendered == self.original {
            return Ok(false);
        }
        fs::write(&self.path, rendered)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(root: &Path) -> String {
        fs::read_to_string(root.join(".cargo").join("config.toml")).unwrap()
    }

    #[test]
    fn creates_forced_env_in_empty_project() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = CargoConfig::open(tmp.path()).unwrap();
        cfg.set_env("LLVM_SYS_140_PREFIX", "C:/llvm", true).unwrap();
        assert!(cfg.save().unwrap());

        let parsed: toml::Value = toml::from_str(&read(tmp.path())).unwrap();
        let env = &parsed["env"]["LLVM_SYS_140_PREFIX"];
        assert_eq!(env["value"].as_str().unwrap(), "C:/llvm");
        assert!(env["force"].as_bool().unwrap());
    }

    #[test]
    fn plain_env_is_a_bare_string() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = CargoConfig::open(tmp.path()).unwrap();
        cfg.set_env("CUQUANTUM_ROOT", "/opt/cuquantum", false)
            .unwrap();
        cfg.save().unwrap();

        let parsed: toml::Value = toml::from_str(&read(tmp.path())).unwrap();
        assert_eq!(
            parsed["env"]["CUQUANTUM_ROOT"].as_str().unwrap(),
            "/opt/cuquantum"
        );
    }

    #[test]
    fn second_write_preserves_unrelated_content_and_no_duplicate_env() {
        let tmp = tempfile::tempdir().unwrap();
        // First writer: LLVM.
        let mut a = CargoConfig::open(tmp.path()).unwrap();
        a.set_env("LLVM_SYS_140_PREFIX", "/llvm", true).unwrap();
        a.save().unwrap();
        // Second writer: cuQuantum -- must merge into the same [env].
        let mut b = CargoConfig::open(tmp.path()).unwrap();
        b.set_env("CUQUANTUM_ROOT", "/cq", true).unwrap();
        b.save().unwrap();

        let text = read(tmp.path());
        assert_eq!(text.matches("[env]").count(), 1, "duplicate [env]: {text}");
        // Both keys survive and parse.
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(
            parsed["env"]["LLVM_SYS_140_PREFIX"]["value"],
            "/llvm".into()
        );
        assert_eq!(parsed["env"]["CUQUANTUM_ROOT"]["value"], "/cq".into());
    }

    #[test]
    fn preserves_foreign_tables() {
        let tmp = tempfile::tempdir().unwrap();
        let cargo_dir = tmp.path().join(".cargo");
        fs::create_dir_all(&cargo_dir).unwrap();
        fs::write(
            cargo_dir.join("config.toml"),
            "[build]\njobs = 4\n\n[env]\nFOO = \"bar\"\n",
        )
        .unwrap();

        let mut cfg = CargoConfig::open(tmp.path()).unwrap();
        cfg.set_env("LLVM_SYS_140_PREFIX", "/llvm", true).unwrap();
        cfg.save().unwrap();

        let parsed: toml::Value = toml::from_str(&read(tmp.path())).unwrap();
        assert_eq!(parsed["build"]["jobs"].as_integer().unwrap(), 4);
        assert_eq!(parsed["env"]["FOO"].as_str().unwrap(), "bar");
        assert_eq!(
            parsed["env"]["LLVM_SYS_140_PREFIX"]["value"],
            "/llvm".into()
        );
    }

    #[test]
    fn target_linker_renders_dotted_header() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = CargoConfig::open(tmp.path()).unwrap();
        cfg.set_target_linker("x86_64-pc-windows-msvc", "C:/msvc/link.exe")
            .unwrap();
        cfg.set_env("LIB", "C:/sdk/lib", true).unwrap();
        assert!(cfg.save().unwrap());

        let text = read(tmp.path());
        assert!(
            text.contains("[target.x86_64-pc-windows-msvc]"),
            "missing dotted target header: {text}"
        );
        let parsed: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(
            parsed["target"]["x86_64-pc-windows-msvc"]["linker"]
                .as_str()
                .unwrap(),
            "C:/msvc/link.exe"
        );
    }

    #[test]
    fn save_is_idempotent_no_rewrite_when_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = CargoConfig::open(tmp.path()).unwrap();
        cfg.set_env("LLVM_SYS_140_PREFIX", "/llvm", true).unwrap();
        assert!(cfg.save().unwrap(), "first write should change the file");

        let mut again = CargoConfig::open(tmp.path()).unwrap();
        again.set_env("LLVM_SYS_140_PREFIX", "/llvm", true).unwrap();
        assert!(
            !again.save().unwrap(),
            "re-applying the same value must not rewrite the file"
        );
    }
}

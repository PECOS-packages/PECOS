// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! ripgrep detection and installation.

use crate::errors::{Error, Result};
use std::path::PathBuf;
use std::process::Command;

/// The docs URL we point users at for manual install instructions.
pub const DOCS_URL: &str = "https://github.com/BurntSushi/ripgrep#installation";

/// Find a usable `rg` on the system `PATH`.
#[must_use]
pub fn find_ripgrep() -> Option<PathBuf> {
    let output = Command::new("rg").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let extensions: &[&str] = if cfg!(windows) {
        &[".exe", ".bat", ""]
    } else {
        &[""]
    };
    crate::executable::which_in_path("rg", extensions)
}

/// Install ripgrep with Cargo, streaming build output to the terminal.
///
/// # Errors
///
/// Returns an error if Cargo cannot be started or the installation fails.
pub fn install_ripgrep(force: bool) -> Result<()> {
    let mut command = Command::new("cargo");
    command.args(["install", "ripgrep", "--locked"]);
    if force {
        command.arg("--force");
    }

    let status = command.status()?;
    if !status.success() {
        return Err(Error::Config(format!(
            "`cargo install ripgrep --locked` failed with {status}"
        )));
    }
    Ok(())
}

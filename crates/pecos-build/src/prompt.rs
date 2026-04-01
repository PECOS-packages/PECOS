//! Interactive prompt utilities for CLI commands
//!
//! Provides a simple Y/n confirmation prompt that respects TTY detection
//! and supports non-interactive overrides for CI environments.

use std::io::{self, BufRead, IsTerminal, Write};

/// How to resolve prompts: interactively, or with a forced answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    /// Ask the user interactively (falls back to default if not a TTY).
    Interactive,
    /// Accept all prompts without asking.
    AcceptAll,
    /// Decline all prompts without asking.
    DeclineAll,
}

/// Prompt the user with a yes/no question.
///
/// - `message`: The question to display (e.g. "Install LLVM 14 to ~/.pecos/deps/llvm/ (~400MB)?")
/// - `default_yes`: Whether the default answer is yes (`[Y/n]`) or no (`[y/N]`)
/// - `mode`: How to resolve the prompt
///
/// In `Interactive` mode, returns the default if stdin is not a TTY (e.g. piped input, CI).
#[must_use]
pub fn confirm(message: &str, default_yes: bool, mode: PromptMode) -> bool {
    match mode {
        PromptMode::AcceptAll => return true,
        PromptMode::DeclineAll => return false,
        PromptMode::Interactive => {}
    }

    // Non-interactive environment -> use default silently
    if !io::stdin().is_terminal() {
        return default_yes;
    }

    let hint = if default_yes { "[Y/n]" } else { "[y/N]" };
    print!("{message} {hint} ");
    io::stdout().flush().ok();

    let mut input = String::new();
    if io::stdin().lock().read_line(&mut input).is_err() {
        return default_yes;
    }

    match input.trim().to_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default_yes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_all_ignores_default() {
        assert!(confirm("test?", false, PromptMode::AcceptAll));
        assert!(confirm("test?", true, PromptMode::AcceptAll));
    }

    #[test]
    fn decline_all_ignores_default() {
        assert!(!confirm("test?", false, PromptMode::DeclineAll));
        assert!(!confirm("test?", true, PromptMode::DeclineAll));
    }
}

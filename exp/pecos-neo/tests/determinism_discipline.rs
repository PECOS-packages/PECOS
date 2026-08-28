// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

struct ForbiddenConstructor {
    pattern: &'static str,
    alternative: &'static str,
}

const FORBIDDEN_CONSTRUCTORS: &[ForbiddenConstructor] = &[
    ForbiddenConstructor {
        pattern: "SparseStab::new",
        alternative: "SparseStab::with_seed(num_qubits, 42)",
    },
    ForbiddenConstructor {
        pattern: "StateVec::new",
        alternative: "StateVec::with_seed(num_qubits, 42)",
    },
    ForbiddenConstructor {
        pattern: "StateVecEngine::new",
        alternative: "StateVecEngine::with_seed(num_qubits, 42)",
    },
];

// Entries are (crate-relative file, one-based line, forbidden pattern).
const ALLOWLIST: &[(&str, usize, &str)] = &[];

#[derive(Default)]
struct LexState {
    block_comment_depth: usize,
    in_string: bool,
    raw_string_hashes: Option<usize>,
}

#[derive(Default)]
struct ScanResults {
    files_visited: usize,
    seeded_constructors_in_tests: usize,
    violations: Vec<String>,
}

#[test]
fn pecos_neo_test_simulators_require_explicit_seeds() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tests_dir = manifest_dir.join("tests");
    let src_dir = manifest_dir.join("src");
    let mut results = ScanResults::default();

    for path in rust_files_under(&tests_dir) {
        scan_whole_test_file(&manifest_dir, &path, true, &mut results);
    }

    let mut external_test_modules = BTreeSet::new();
    for path in rust_files_under(&src_dir) {
        scan_cfg_test_modules(
            &manifest_dir,
            &path,
            &mut external_test_modules,
            &mut results,
        );
    }
    for path in external_test_modules {
        scan_whole_test_file(&manifest_dir, &path, false, &mut results);
    }

    assert!(
        results.files_visited > 0,
        "determinism scan did not visit any Rust files"
    );
    assert!(
        results.seeded_constructors_in_tests > 0,
        "determinism scan found no seeded simulator constructors under tests/; the scan may be vacuous"
    );
    assert!(
        results.violations.is_empty(),
        "pecos-neo test code must construct every simulator with an explicit seed; \
         entropy-seeded constructors are forbidden and the allowlist is intentionally empty:\n{}",
        results.violations.join("\n")
    );
}

fn rust_files_under(root: &Path) -> Vec<PathBuf> {
    fn collect(directory: &Path, files: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()));
        for entry in entries {
            let path = entry.expect("failed to read directory entry").path();
            if path.is_dir() {
                collect(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    files
}

fn scan_whole_test_file(
    manifest_dir: &Path,
    path: &Path,
    count_seeded_constructors: bool,
    results: &mut ScanResults,
) {
    let source = read_source(path);
    let relative_path = crate_relative_path(manifest_dir, path);
    let mut lex_state = LexState::default();

    results.files_visited += 1;
    for (line_index, line) in source.lines().enumerate() {
        let code = code_only(line, &mut lex_state);
        scan_code_line(
            &relative_path,
            line_index + 1,
            &code,
            count_seeded_constructors,
            results,
        );
    }
}

fn scan_cfg_test_modules(
    manifest_dir: &Path,
    path: &Path,
    external_test_modules: &mut BTreeSet<PathBuf>,
    results: &mut ScanResults,
) {
    let source = read_source(path);
    let relative_path = crate_relative_path(manifest_dir, path);
    let mut lex_state = LexState::default();
    let mut pending_cfg_test = false;
    let mut brace_depth = 0isize;
    let mut test_module_outer_depths = Vec::new();

    results.files_visited += 1;
    for (line_index, line) in source.lines().enumerate() {
        let code = code_only(line, &mut lex_state);
        let compact: String = code
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let already_in_test_module = !test_module_outer_depths.is_empty();

        if already_in_test_module {
            scan_code_line(&relative_path, line_index + 1, &code, false, results);
        }

        if compact == "#[cfg(test)]" {
            pending_cfg_test = true;
        } else if pending_cfg_test && (compact.is_empty() || compact.starts_with("#[")) {
            // Permit comments, blank lines, and other attributes between cfg(test) and mod.
        } else if pending_cfg_test {
            if let Some((module_name, inline)) = module_declaration(&code) {
                if inline {
                    if !already_in_test_module {
                        scan_code_line(&relative_path, line_index + 1, &code, false, results);
                    }
                    test_module_outer_depths.push(brace_depth);
                } else if let Some(module_path) = conventional_module_path(path, &module_name) {
                    external_test_modules.insert(module_path);
                }
            }
            pending_cfg_test = false;
        }

        brace_depth += brace_delta(&code);
        while test_module_outer_depths
            .last()
            .is_some_and(|outer_depth| brace_depth <= *outer_depth)
        {
            test_module_outer_depths.pop();
        }
    }
}

// This is deliberately a lightweight Rust lexer and brace tracker, not a full parser. It handles
// nested block comments plus ordinary/raw strings and ordinary character literals. The module
// detector expects cfg(test) on its own attribute line followed by a conventional mod declaration;
// external modules using #[path] are outside its supported syntax.
fn code_only(line: &str, state: &mut LexState) -> String {
    let bytes = line.as_bytes();
    let mut code = String::with_capacity(line.len());
    let mut index = 0;

    while index < bytes.len() {
        if state.block_comment_depth > 0 {
            if bytes[index..].starts_with(b"/*") {
                state.block_comment_depth += 1;
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                state.block_comment_depth -= 1;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }

        if let Some(hashes) = state.raw_string_hashes {
            if raw_string_closes_at(bytes, index, hashes) {
                state.raw_string_hashes = None;
                index += hashes + 1;
            } else {
                index += 1;
            }
            continue;
        }

        if state.in_string {
            match bytes[index] {
                b'\\' => index = (index + 2).min(bytes.len()),
                b'"' => {
                    state.in_string = false;
                    index += 1;
                }
                _ => index += 1,
            }
            continue;
        }

        if bytes[index..].starts_with(b"//") {
            break;
        }
        if bytes[index..].starts_with(b"/*") {
            state.block_comment_depth = 1;
            index += 2;
            continue;
        }
        if let Some((prefix_length, hashes)) = raw_string_opens_at(bytes, index) {
            state.raw_string_hashes = Some(hashes);
            index += prefix_length;
            continue;
        }
        if bytes[index] == b'"' {
            state.in_string = true;
            index += 1;
            continue;
        }
        if bytes[index] == b'\''
            && let Some(end) = character_literal_end(bytes, index)
        {
            index = end + 1;
            continue;
        }

        code.push(char::from(bytes[index]));
        index += 1;
    }

    code
}

fn raw_string_opens_at(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let raw_prefix = if bytes.get(index) == Some(&b'r') {
        index + 1
    } else if bytes.get(index..index + 2) == Some(b"br") {
        index + 2
    } else {
        return None;
    };

    let mut quote_index = raw_prefix;
    while bytes.get(quote_index) == Some(&b'#') {
        quote_index += 1;
    }
    (bytes.get(quote_index) == Some(&b'"'))
        .then_some((quote_index - index + 1, quote_index - raw_prefix))
}

fn raw_string_closes_at(bytes: &[u8], index: usize, hashes: usize) -> bool {
    bytes.get(index) == Some(&b'"')
        && bytes
            .get(index + 1..index + 1 + hashes)
            .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
}

fn character_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'\'' => return Some(index),
            byte if byte.is_ascii_whitespace() => return None,
            _ => index += 1,
        }
    }
    None
}

fn module_declaration(code: &str) -> Option<(String, bool)> {
    let tokens: Vec<&str> = code
        .split(|character: char| character.is_whitespace() || matches!(character, '{' | ';'))
        .filter(|token| !token.is_empty())
        .collect();
    let mod_index = tokens.iter().position(|token| *token == "mod")?;
    let module_name = tokens.get(mod_index + 1)?.trim().to_string();
    Some((module_name, code.contains('{')))
}

fn conventional_module_path(parent_file: &Path, module_name: &str) -> Option<PathBuf> {
    let parent_directory = parent_file.parent()?;
    let module_base = match parent_file.file_stem()?.to_str()? {
        "lib" | "main" | "mod" => parent_directory.to_path_buf(),
        stem => parent_directory.join(stem),
    };
    let file_module = module_base.join(format!("{module_name}.rs"));
    if file_module.is_file() {
        return Some(file_module);
    }
    let directory_module = module_base.join(module_name).join("mod.rs");
    directory_module.is_file().then_some(directory_module)
}

fn brace_delta(code: &str) -> isize {
    code.bytes().fold(0, |depth, byte| match byte {
        b'{' => depth + 1,
        b'}' => depth - 1,
        _ => depth,
    })
}

fn scan_code_line(
    relative_path: &str,
    line_number: usize,
    code: &str,
    count_seeded_constructors: bool,
    results: &mut ScanResults,
) {
    if count_seeded_constructors
        && [
            "SparseStab::with_seed(",
            "StateVec::with_seed(",
            "StateVecEngine::with_seed(",
            "CoinToss::with_seed(",
        ]
        .iter()
        .any(|pattern| code.contains(pattern))
    {
        results.seeded_constructors_in_tests += 1;
    }

    for forbidden in FORBIDDEN_CONSTRUCTORS {
        if code.contains(forbidden.pattern)
            && !ALLOWLIST.contains(&(relative_path, line_number, forbidden.pattern))
        {
            results.violations.push(format!(
                "{relative_path}:{line_number}: found `{}`; use `{}`",
                forbidden.pattern, forbidden.alternative
            ));
        }
    }
}

fn read_source(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn crate_relative_path(manifest_dir: &Path, path: &Path) -> String {
    path.strip_prefix(manifest_dir)
        .unwrap_or(path)
        .display()
        .to_string()
}

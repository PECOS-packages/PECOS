"""Regression tests for documentation authoring and Rust data staging."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

GENERATOR = runpy.run_path(str(Path(__file__).resolve().parents[4] / "scripts/docs/generate_doc_tests.py"))
extract_code_blocks = GENERATOR["extract_code_blocks"]
generate_test_function = GENERATOR["generate_test_function"]
generate_unified_rust_crate = GENERATOR["_generate_unified_rust_crate"]
stage_rust_test_data = GENERATOR["_stage_rust_test_data"]
CodeBlock = GENERATOR["CodeBlock"]

RUST_CODE = 'use pecos::prelude::*;\nlet data = std::fs::read("a.ext").unwrap();'


@pytest.fixture
def docs_tree(tmp_path: Path):
    docs = tmp_path / "docs"
    assets = docs / "assets/test-data"
    assets.mkdir(parents=True)
    crate = tmp_path / "rust_crate"
    crate.mkdir()
    markdown = docs / "example.md"
    return docs, assets, crate, markdown


def test_staging_refreshes_and_prunes_only_owned_files(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_bytes(b"first\x00")
    (assets / "stale.ext").write_bytes(b"stale")
    unrelated = crate / "unrelated.ext"
    unrelated.write_bytes(b"keep")
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext, stale.ext-->\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "a.ext").read_bytes() == b"first\x00"
    assert (crate / "stale.ext").read_bytes() == b"stale"
    assert not (crate / "tests/a.ext").exists()
    assert json.loads((crate / ".test-data-files.json").read_text()) == ["a.ext", "stale.ext"]

    (assets / "a.ext").write_bytes(b"refreshed")
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "a.ext").read_bytes() == b"refreshed"
    assert not (crate / "stale.ext").exists()
    assert unrelated.read_bytes() == b"keep"

    markdown.write_text(f"# Example\n\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert not (crate / "a.ext").exists()
    assert unrelated.read_bytes() == b"keep"


def test_missing_data_names_source_and_block(docs_tree):
    docs, _assets, crate, markdown = docs_tree
    markdown.write_text(f"# Example\n\n<!--test-data: missing.ext-->\n```rust\n{RUST_CODE}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1: missing test-data file 'missing\.ext'"):
        generate_unified_rust_crate([markdown], docs, crate)


@pytest.mark.parametrize("language", ["python", "javascript", "c#", "text", ""])
def test_non_rust_data_marker_is_rejected(docs_tree, language):
    docs, _assets, crate, markdown = docs_tree
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->\n```{language}\nprint('hello')\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1: test-data"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_continuation_inherits_and_stages_earlier_data(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_bytes(b"earlier")
    (assets / "b.ext").write_bytes(b"later")
    markdown.write_text(
        f"# Example\n\n<!--test-data: a.ext-->\n```rust\n{RUST_CODE}\n```\n\n"
        "<!--continuation-->\n<!--test-data: b.ext-->\n```rust\nassert!(!data.is_empty());\n```\n\n"
        '<!--continuation-->\n```rust\nassert!(std::fs::read("b.ext").is_ok());\n```\n\n'
        "```rust\nuse pecos::prelude::*;\nlet independent = 1;\n```\n",
    )
    blocks = extract_code_blocks(markdown, "rust")
    assert blocks[2].test_data == ["a.ext", "b.ext"]
    assert blocks[3].test_data == []
    # Stage the continuation in isolation: staging every block could mask lost inheritance.
    stage_rust_test_data([blocks[2]], docs, crate)
    assert (crate / "a.ext").read_bytes() == b"earlier"
    assert (crate / "b.ext").read_bytes() == b"later"
    generate_unified_rust_crate([markdown], docs, crate)
    generated = (crate / "tests/example.rs").read_text()
    assert generated.count('std::fs::read("a.ext")') == 3


@pytest.mark.parametrize("cargo_import", ["", "use pecos::prelude::*;\n"])
@pytest.mark.parametrize(
    "skip",
    ["<!--skip: illustrative only-->\n", "<!--skip-->\n", ",skip", ",ignore", ",no_run", ",notest"],
)
def test_incomplete_rust_requires_explicit_skip(docs_tree, cargo_import, skip):
    docs, _assets, crate, markdown = docs_tree
    code = cargo_import + "let runner = ...;"
    markdown.write_text(f"# Example\n\n```rust\n{code}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1:.*complete the example or mark it skipped"):
        generate_unified_rust_crate([markdown], docs, crate)
    # Direct pytest generation must enforce the same rule, without relying on extraction.
    block = CodeBlock(code=code, language="rust", block_number=1, source_file=markdown)
    with pytest.raises(ValueError, match=r"example\.md: block 1:.*complete the example or mark it skipped"):
        generate_test_function(block, "example")

    marker = skip if skip.startswith("<!--") else ""
    suffix = skip if skip.startswith(",") else ""
    markdown.write_text(f"# Example\n\n{marker}```rust{suffix}\n{code}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert list((crate / "tests").glob("*.rs")) == []
    [block] = extract_code_blocks(markdown, "rust")
    assert "@pytest.mark.skip(" in generate_test_function(block, "example")


@pytest.mark.parametrize(
    ("marker", "fence", "code"),
    [
        ("<!--skip: illustrative-->", "rust", RUST_CODE),
        ("", "rust,ignore", RUST_CODE),
        ("", "rust", "fn main() {}"),
        ("", "hidden-rust", RUST_CODE),
        ("<!--setup-->", "rust", RUST_CODE),
        ("<!--teardown-->", "rust", RUST_CODE),
    ],
)
def test_data_marker_requires_a_staged_block(docs_tree, marker, fence, code):
    docs, _assets, crate, markdown = docs_tree
    markdown.write_text(f"# Example\n\n{marker}\n<!--test-data: a.ext-->\n```{fence}\n{code}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1: test-data"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_absent_marker_stages_nothing(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("unused")
    markdown.write_text(f"# Example\n\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert sorted(path.name for path in crate.iterdir()) == ["tests"]


@pytest.mark.parametrize(
    "name",
    ["../outside.ext", "/absolute.ext", "Cargo.toml", "Cargo.lock", ".test-data-files.json"],
)
def test_data_paths_and_reserved_names_are_rejected(docs_tree, name):
    docs, _assets, crate, markdown = docs_tree
    markdown.write_text(f"# Example\n\n<!--test-data: {name}-->\n```rust\n{RUST_CODE}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1:.*bare, non-reserved"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_staging_refuses_to_overwrite_unrelated_files(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("source")
    (crate / "a.ext").write_text("unrelated")
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->\n```rust\n{RUST_CODE}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1: cannot overwrite unrelated"):
        generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "a.ext").read_text() == "unrelated"


def test_data_filename_does_not_imply_skip(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "skip.ext").write_text("data")
    markdown.write_text(f"# Example\n\n<!--test-data: skip.ext-->\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "skip.ext").read_text() == "data"

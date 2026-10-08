"""Regression tests for documentation authoring and Rust data staging."""

from __future__ import annotations

import json
import re
import runpy
from pathlib import Path

import pytest

GENERATOR = runpy.run_path(str(Path(__file__).resolve().parents[4] / "scripts/docs/generate_doc_tests.py"))
extract_code_blocks = GENERATOR["extract_code_blocks"]
generate_test_function = GENERATOR["generate_test_function"]
generate_unified_rust_crate = GENERATOR["_generate_unified_rust_crate"]
stage_rust_test_data = GENERATOR["_stage_rust_test_data"]
CodeBlock = GENERATOR["CodeBlock"]
parse_marker_comment = GENERATOR["_parse_marker_comment"]

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
    assert not (crate / ".test-data-files.json").exists()
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
    with pytest.raises(ValueError, match=r"example\.md:3: test-data marker"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_continuation_inherits_and_stages_earlier_data(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_bytes(b"earlier")
    (assets / "b.ext").write_bytes(b"later")
    markdown.write_text(
        f"# Example\n\n<!--test-data: a.ext-->\n```rust\n{RUST_CODE}\n```\n\n"
        "<!--continuation test-data: b.ext-->\n```rust\nassert!(!data.is_empty());\n```\n\n"
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
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->\n{marker}\n```{fence}\n{code}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md:3: test-data marker"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_absent_marker_stages_nothing(docs_tree):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("unused")
    markdown.write_text(f"# Example\n\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert sorted(path.name for path in crate.iterdir()) == ["tests"]


@pytest.mark.parametrize(
    "name",
    [
        "../outside.ext",
        "/absolute.ext",
        "Cargo.toml",
        "Cargo.lock",
        ".test-data-files.json",
        "build.rs",
        "rust-toolchain",
        "rust-toolchain.toml",
        ".cargo",
        ".env",
        ".hidden.ext",
    ],
)
def test_data_paths_and_reserved_names_are_rejected(docs_tree, name):
    docs, assets, crate, markdown = docs_tree
    if Path(name).name == name:
        (assets / name).write_text("source")
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


def test_data_marker_allows_space_before_colon(docs_tree):
    # The placement check accepts `test-data :`; the parser must stage it too,
    # or an accepted marker would silently stage nothing.
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("data")
    markdown.write_text(f"# Example\n\n<!--test-data : a.ext-->\n```rust\n{RUST_CODE}\n```\n")
    generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "a.ext").read_text() == "data"


PYTHON_BLOCKS = "```python\nprint(1)\n```\n\n```python\nprint(2)\n```\n"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Use ``` to open a fence.\n\n" + PYTHON_BLOCKS, ["print(1)", "print(2)"]),
        ("Wrap code in ```` ``` ```` fences.\n\n" + PYTHON_BLOCKS, ["print(1)", "print(2)"]),
        ("```text\nsome ``` inside\n```\n\n```python\nprint(1)\n```\n", ["print(1)"]),
        ("```bash\nunclosed\n\n" + PYTHON_BLOCKS, ["print(1)", "print(2)"]),
        (
            "```markdown\n\\```python\nprint(0)\n\\```\n```\n\n```python\nprint(1)\n```\n",
            ["print(0)\n\\", "print(1)"],
        ),
    ],
    ids=["prose", "inline", "text-body", "unclosed-bash", "escaped-markdown"],
)
def test_fence_extraction_matches_origin_dev(tmp_path, content, expected):
    markdown = tmp_path / "example.md"
    markdown.write_text(content)
    # These are the origin/dev patterns, deliberately independent of the generator.
    pattern = r"(<!--[^>]*-->\s*)?" + r"```(hidden-)?python(,(?:skip|ignore|no_run|notest))?\n(.*?)```"
    legacy_code = [match.group(4).strip() for match in re.finditer(pattern, content, re.DOTALL)]
    blocks = extract_code_blocks(markdown, "python")
    assert len(blocks) == len(expected)
    assert [block.code for block in blocks] == legacy_code == expected


@pytest.mark.parametrize(
    ("prefix", "fence", "code"),
    [
        ("Use ``` to open a fence.\n\n", "python", "print(1)"),
        ("Use ``` to open a fence.\n\n", "rust,skip", RUST_CODE),
        ("```bash\nunclosed\n", "rust", "fn main() {}"),
        ("```bash\nunclosed\n", "rust,ignore", RUST_CODE),
    ],
    ids=["stray-python", "stray-skipped-rust", "unclosed-noncargo-rust", "unclosed-skipped-rust"],
)
def test_misplaced_data_in_stray_fence_file_fails(docs_tree, prefix, fence, code):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("data")
    markdown.write_text(f"{prefix}<!--test-data: a.ext-->\n```{fence}\n{code}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md:3: test-data marker"):
        generate_unified_rust_crate([markdown], docs, crate)


def test_document_skip_exempts_illustrative_data_markers(docs_tree):
    docs, _assets, crate, markdown = docs_tree
    example = "```markdown\n<!--test-data: a.ext-->\n\\```rust\nfn main() {}\n\\```\n```\n"
    markdown.write_text(f"# Example\n\n<!--skip: illustrative-->\n\n{example}")
    generate_unified_rust_crate([markdown], docs, crate)
    assert not (crate / "a.ext").exists()
    markdown.write_text(f"# Example\n\n{example}")
    with pytest.raises(ValueError, match=r"example\.md:4: test-data marker"):
        generate_unified_rust_crate([markdown], docs, crate)


@pytest.mark.parametrize("following", ["", "prose\n", "<!--test-name: x-->\n", "<!--test-data: b.ext-->\n"])
def test_unattached_data_marker_fails(docs_tree, following):
    docs, _assets, crate, markdown = docs_tree
    code = f"```rust\n{RUST_CODE}\n```\n" if following else ""
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->{following}{code}")
    with pytest.raises(ValueError, match=r"example\.md:3: test-data marker"):
        generate_unified_rust_crate([markdown], docs, crate)


@pytest.mark.parametrize(
    ("comment", "expected_skip", "reason"),
    [
        ("<!--skip-->", True, None),
        ("<!-- skip -->", True, None),
        ("<!--skip: r-->", True, "r"),
        ("<!--skip reason-->", True, None),
        ("<!-- skip : reason -->", True, "reason"),
        ("<!--skip - flaky-->", True, None),
        ("<!--SKIP: r-->", True, "r"),
        ("<!--skip-if-no-cuda-->", False, None),
        ("<!--skip-if-no-cuda-rust-->", False, None),
        ("<!--test-name: skip_this-->", False, None),
        ("<!--test-data: skip.txt-->", False, None),
        ("<!--skipper-->", False, None),
    ],
)
def test_skip_comment_spellings(comment, expected_skip, reason):
    attrs = parse_marker_comment(comment)
    assert attrs["skip"] is expected_skip
    assert attrs["skip_reason"] == reason


@pytest.mark.parametrize("last_comment", ["<!--test-name: x-->", "<!--expect-error: X-->"])
def test_only_immediately_preceding_comment_applies(tmp_path, last_comment):
    markdown = tmp_path / "example.md"
    # Introductory prose keeps the first comment from being a document-level skip.
    markdown.write_text(f"Intro.\n\n<!--skip-->\n{last_comment}\n```python\nprint(1)\n```\n")
    [block] = extract_code_blocks(markdown)
    assert not block.skip
    assert block.line_number == 4
    assert block.test_name == ("x" if "test-name" in last_comment else None)
    assert block.expect_error == ("X" if "expect-error" in last_comment else None)


def test_expect_error_overrides_skip():
    attrs = parse_marker_comment("<!--skip: illustrative; expect-error: X-->")
    assert attrs["expect_error"] == "X"
    assert not attrs["skip"]


@pytest.mark.parametrize("target_exists", [True, False])
def test_staging_refuses_symlink_destination(docs_tree, target_exists):
    docs, assets, crate, markdown = docs_tree
    (assets / "a.ext").write_text("source")
    target = crate.parent / "target.ext"
    if target_exists:
        target.write_text("unrelated")
    (crate / "a.ext").symlink_to(target)
    # Even a previously owned destination must not follow a replacement symlink.
    (crate / ".test-data-files.json").write_text('["a.ext"]')
    markdown.write_text(f"# Example\n\n<!--test-data: a.ext-->\n```rust\n{RUST_CODE}\n```\n")
    with pytest.raises(ValueError, match=r"example\.md: block 1: cannot overwrite unrelated"):
        generate_unified_rust_crate([markdown], docs, crate)
    assert (crate / "a.ext").is_symlink()
    assert target.read_text() == "unrelated" if target_exists else not target.exists()


@pytest.mark.parametrize(
    "contents",
    ['{"a.ext": 1}', "[1]", '["../outside.ext"]', '["build.rs"]', '[".cargo"]', "{"],
)
def test_invalid_staging_manifest_is_rejected(docs_tree, contents):
    docs, _assets, crate, markdown = docs_tree
    manifest = crate / ".test-data-files.json"
    manifest.write_text(contents)
    outside = crate.parent / "outside.ext"
    outside.write_text("keep")
    markdown.write_text(f"# Example\n\n```rust\n{RUST_CODE}\n```\n")
    with pytest.raises(ValueError, match=r"Invalid staged test-data manifest|Expecting property name"):
        generate_unified_rust_crate([markdown], docs, crate)
    assert manifest.read_text() == contents
    assert outside.read_text() == "keep"

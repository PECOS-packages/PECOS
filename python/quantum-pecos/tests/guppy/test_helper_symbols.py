"""PECOS helper normalization must preserve the runtime ABI without LLVM bindings."""

import pytest
from pecos_rslib.hugr_lowering import PECOS_HELPER_ABIS, normalize_pecos_helper_symbols

_HELPER = "pecos_qis_trace_metadata_qubit_hugr"


@pytest.mark.parametrize("prefix", ["__hugr__.", "__hugr__.__main__.", "__hugr__.a.b."])
@pytest.mark.parametrize("suffix", ["", ".14", ".001"])
def test_rename_every_symbol_occurrence(prefix: str, suffix: str) -> None:
    name = f"{prefix}{_HELPER}{suffix}"
    source = (
        f"declare i64 @{name}(i64, ptr, ptr) local_unnamed_addr\n%q = call i64 @{name}(i64 0, ptr null, ptr null)\n"
    )
    assert normalize_pecos_helper_symbols(source) == source.replace("@" + name, "@" + _HELPER)


@pytest.mark.parametrize(("helper", "abi"), PECOS_HELPER_ABIS.items())
def test_all_helper_abis(helper: str, abi: str) -> None:
    result, parameters = abi.split(" (", 1)
    declaration = f"declare {result} @__hugr__.{helper}({parameters}\n"
    expected = declaration.replace("@__hugr__.", "@")
    assert normalize_pecos_helper_symbols(declaration) == expected
    assert normalize_pecos_helper_symbols(expected) == expected


def test_identical_declarations_merge_after_attributes_and_whitespace() -> None:
    first = f"declare i64 @__hugr__.__main__.{_HELPER}.14(i64, ptr, ptr) local_unnamed_addr #0\n"
    second = f"declare noundef i64 @__hugr__.other.{_HELPER}.92( i64 noundef, ptr nonnull, ptr ) nounwind\n"
    assert normalize_pecos_helper_symbols(first + second) == first.replace(f"__hugr__.__main__.{_HELPER}.14", _HELPER)


def test_conflicting_declarations_report_both_signatures() -> None:
    source = f"declare i64 @{_HELPER}(i64, ptr, ptr)\ndeclare void @__hugr__.{_HELPER}.2(ptr, ptr)\n"
    with pytest.raises(ValueError, match=_HELPER) as error:
        normalize_pecos_helper_symbols(source)
    assert _HELPER in str(error.value)
    assert "i64 (i64, ptr, ptr)" in str(error.value)
    assert "void (ptr, ptr)" in str(error.value)
    assert "conflicting" in str(error.value)


@pytest.mark.parametrize(
    "actual",
    ["i32 (i64, ptr, ptr)", "i64 (i32, ptr, ptr)", "i64 (i64, ptr addrspace(1), ptr)", "i64 (i64, ptr, ptr, ...)"],
)
def test_wrong_abi_reports_expected_and_actual(actual: str) -> None:
    result, parameters = actual.split(" (", 1)
    with pytest.raises(ValueError, match=_HELPER) as error:
        normalize_pecos_helper_symbols(f"declare {result} @{_HELPER}({parameters}\n")
    assert _HELPER in str(error.value)
    assert PECOS_HELPER_ABIS[_HELPER] in str(error.value)
    assert actual in str(error.value)


@pytest.mark.parametrize("name", [_HELPER, f"__hugr__.__main__.{_HELPER}.14"])
def test_helper_definition_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=f"{_HELPER}.*defined"):
        normalize_pecos_helper_symbols(f"define i64 @{name}(i64 %q, ptr %a, ptr %b) {{\nret i64 %q\n}}\n")


def test_non_helpers_and_non_numeric_suffixes_are_untouched() -> None:
    source = f"declare i64 @__hugr__.other.14(i64)\ndeclare i64 @__hugr__.{_HELPER}.user(i64)\n"
    assert normalize_pecos_helper_symbols(source) == source


def test_strings_and_comments_are_untouched() -> None:
    source = f'@data = constant [90 x i8] c"escaped \\22 @__hugr__.__main__.{_HELPER}.14\\00"\n; @{_HELPER}\n'
    assert normalize_pecos_helper_symbols(source) == source


def test_function_body_reference_is_not_a_definition() -> None:
    source = f"define ptr @user() {{ ret ptr @__hugr__.{_HELPER}.14 }}\n"
    assert normalize_pecos_helper_symbols(source) == source.replace(f"__hugr__.{_HELPER}.14", _HELPER)


def test_quoted_symbol_with_guppy_scope_is_renamed() -> None:
    name = f'"__hugr__.tests.test_probe.<locals>.{_HELPER}.12"'
    source = f"declare i64 @{name}(i64, ptr, ptr)\n%q = call i64 @{name}(i64 0, ptr null, ptr null)\n"
    assert normalize_pecos_helper_symbols(source) == source.replace(name, _HELPER)


def test_declaration_text_inside_multiline_string_is_untouched() -> None:
    source = f'@data = constant [99 x i8] c"\ndeclare void @{_HELPER}()\n"\n'
    assert normalize_pecos_helper_symbols(source) == source


def test_aggregate_declarations_merge_with_spacing_and_return_attributes() -> None:
    helper = "pecos_qis_runtime_barrier_qubits2_hugr"
    first = f"declare {{i64,i64}} @__hugr__.a.{helper}.1(i64,i64) local_unnamed_addr\n"
    second = f"declare noundef {{ i64, i64 }} @{helper}(i64 noundef, i64) #1\n"
    assert normalize_pecos_helper_symbols(first + second) == first.replace(f"__hugr__.a.{helper}.1", helper)


@pytest.mark.parametrize("modifier", ["dso_local", "hidden", "ccc", "noundef range(i64 0, 4)"])
def test_declaration_modifiers_are_not_return_types(modifier: str) -> None:
    source = f"declare {modifier} i64 @{_HELPER}(i64 noundef, ptr align 8, ptr) local_unnamed_addr\n"
    assert normalize_pecos_helper_symbols(source) == source


def test_declarations_with_different_modifiers_merge() -> None:
    first = f"declare i64 @{_HELPER}(i64, ptr, ptr)\n"
    second = f"declare dso_local i64 @__hugr__.{_HELPER}.14(i64, ptr, ptr)\n"
    assert normalize_pecos_helper_symbols(first + second) == first


def test_parameter_names_do_not_change_the_llvm_type() -> None:
    first = f'declare i64 @{_HELPER}(i64 %"a,b)", ptr %key, ptr %value)\n'
    duplicate = first.replace("declare i64", "declare noundef i64")
    assert normalize_pecos_helper_symbols(first + duplicate) == first
    assert normalize_pecos_helper_symbols(first + first.replace("a,b)", "different")) == first


@pytest.mark.parametrize("control", ["\x0c", "\r", "\u2028"])
def test_control_characters_inside_string_do_not_split_declarations(control: str) -> None:
    literal = f'@data = constant [99 x i8] c"{control}@__hugr__.{_HELPER}.14"\n'
    declaration = f"declare i64 @__hugr__.{_HELPER}.14(i64, ptr, ptr)\n"
    assert normalize_pecos_helper_symbols(literal + declaration) == literal + declaration.replace(
        "__hugr__.",
        "",
    ).replace(".14", "")

"""Tests for the Zluppy Python bindings."""

import json

import pytest

import zluppy

# =============================================================================
# Version Tests
# =============================================================================


def test_version():
    """Test that version returns a non-empty string."""
    v = zluppy.version()
    assert isinstance(v, str)
    assert len(v) > 0
    assert "." in v  # Should be semantic version like "0.1.0"


# =============================================================================
# compile_to_slr Tests
# =============================================================================


def test_compile_to_slr_bell_state():
    """Test compiling a Bell state program to SLR-AST dict."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }"""

    result = zluppy.compile_to_slr(source)

    assert isinstance(result, dict)
    assert result["type"] == "Program"
    assert result["name"] == "main"
    assert result["allocator"]["name"] == "q"
    assert result["allocator"]["capacity"] == 2

    body = result["body"]
    assert len(body) == 2
    assert body[0]["gate"] == "H"
    assert body[1]["gate"] == "CX"


def test_compile_to_slr_rotation_gate():
    """Test compiling a rotation gate program."""
    source = """fn main() -> void {
        var q = qalloc(1);
        rz(q[0], 1.57);
    }"""

    result = zluppy.compile_to_slr(source)

    body = result["body"]
    assert len(body) == 1
    assert body[0]["gate"] == "RZ"
    assert len(body[0]["params"]) > 0


def test_compile_to_slr_child_allocator():
    """Test compiling with child allocator."""
    source = """fn main() -> void {
        var base = qalloc(4);
        var q = base.child(2);
        h(q[0]);
    }"""

    result = zluppy.compile_to_slr(source)

    # Should have multiple allocator declarations
    decls = result["declarations"]
    assert len(decls) >= 2


def test_compile_to_slr_strict_mode():
    """Test compiling with strict mode enabled."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
    }"""

    # Strict mode should still compile valid code
    result = zluppy.compile_to_slr(source, strict=True)
    assert result["type"] == "Program"


def test_compile_to_slr_parse_error():
    """Test that parse errors raise ZluppyError."""
    source = "fn main() -> void { h(q[0] }"  # Missing closing paren

    with pytest.raises(zluppy.ZluppyError) as exc_info:
        zluppy.compile_to_slr(source)

    assert "parse error" in str(exc_info.value).lower() or "expected" in str(exc_info.value).lower()


def test_compile_to_slr_semantic_error():
    """Test that semantic errors raise ZluppyError."""
    source = """fn main() -> void {
        h(undefined_var[0]);
    }"""

    with pytest.raises(zluppy.ZluppyError) as exc_info:
        zluppy.compile_to_slr(source)

    assert "semantic" in str(exc_info.value).lower() or "undefined" in str(exc_info.value).lower()


# =============================================================================
# compile_to_slr_json Tests
# =============================================================================


def test_compile_to_slr_json_pretty():
    """Test compiling to pretty-printed JSON string."""
    source = """fn main() -> void {
        var q = qalloc(1);
        h(q[0]);
    }"""

    result = zluppy.compile_to_slr_json(source)

    assert isinstance(result, str)
    # Pretty-printed JSON has newlines
    assert "\n" in result

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed["type"] == "Program"


def test_compile_to_slr_json_compact():
    """Test compiling to compact JSON string."""
    source = """fn main() -> void {
        var q = qalloc(1);
        h(q[0]);
    }"""

    result = zluppy.compile_to_slr_json(source, compact=True)

    assert isinstance(result, str)
    # Compact JSON should not have indented newlines
    assert "\n  " not in result

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed["type"] == "Program"


def test_compile_to_slr_json_strict():
    """Test compiling to JSON with strict mode."""
    source = """fn main() -> void {
        var q = qalloc(1);
        x(q[0]);
    }"""

    result = zluppy.compile_to_slr_json(source, strict=True)
    parsed = json.loads(result)
    assert parsed["type"] == "Program"


# =============================================================================
# check Tests
# =============================================================================


def test_check_valid_program():
    """Test checking a valid program."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }"""

    # Should not raise
    zluppy.check(source)


def test_check_strict_mode():
    """Test checking in strict mode."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
    }"""

    # Should not raise for valid code
    zluppy.check(source, strict=True)


def test_check_parse_error():
    """Test that check raises on parse error."""
    source = "fn main( void { }"  # Missing closing paren

    with pytest.raises(zluppy.ZluppyError):
        zluppy.check(source)


def test_check_semantic_error():
    """Test that check raises on semantic error."""
    source = """fn main() -> void {
        UnknownGate(q[0]);
    }"""

    with pytest.raises(zluppy.ZluppyError):
        zluppy.check(source)


# =============================================================================
# parse_debug Tests
# =============================================================================


def test_parse_debug():
    """Test getting debug AST string."""
    source = "fn main() -> void { var q = qalloc(1); }"

    result = zluppy.parse_debug(source)

    assert isinstance(result, str)
    assert "Program" in result
    assert "FnDecl" in result


def test_parse_debug_error():
    """Test that parse_debug raises on parse error."""
    with pytest.raises(zluppy.ZluppyError):
        zluppy.parse_debug("fn main( void { }")


# =============================================================================
# SlrProgram Builder Tests
# =============================================================================


def test_slr_program_create():
    """Test creating an SlrProgram."""
    prog = zluppy.SlrProgram("test")

    assert repr(prog) == 'SlrProgram(name="test")'


def test_slr_program_add_allocator():
    """Test adding an allocator to SlrProgram."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 2)

    result = prog.to_dict()

    assert result["allocator"]["name"] == "q"
    assert result["allocator"]["capacity"] == 2


def test_slr_program_add_gate():
    """Test adding gates to SlrProgram."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 2)
    prog.add_gate("H", [("q", 0)])
    prog.add_gate("CX", [("q", 0), ("q", 1)])

    result = prog.to_dict()

    body = result["body"]
    assert len(body) == 2
    assert body[0]["gate"] == "H"
    assert body[1]["gate"] == "CX"


def test_slr_program_add_rotation_gate():
    """Test adding rotation gate with parameters."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 1)
    prog.add_gate("RZ", [("q", 0)], [3.14159])

    result = prog.to_dict()

    body = result["body"]
    assert len(body) == 1
    assert body[0]["gate"] == "RZ"
    assert len(body[0]["params"]) == 1


def test_slr_program_add_prepare():
    """Test adding prepare operation."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 2)
    prog.add_prepare("q")  # Prepare all

    result = prog.to_dict()

    body = result["body"]
    assert len(body) == 1
    assert body[0]["type"] == "PrepareOp"


def test_slr_program_add_prepare_slots():
    """Test adding prepare operation with specific slots."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 3)
    prog.add_prepare("q", [0, 1])  # Prepare specific slots

    result = prog.to_dict()

    body = result["body"]
    assert len(body) == 1
    assert body[0]["type"] == "PrepareOp"


def test_slr_program_to_json():
    """Test converting SlrProgram to JSON."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 1)
    prog.add_gate("H", [("q", 0)])

    result = prog.to_json()

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["name"] == "test"


def test_slr_program_to_json_compact():
    """Test converting SlrProgram to compact JSON."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 1)
    prog.add_gate("X", [("q", 0)])

    result = prog.to_json(compact=True)

    assert isinstance(result, str)
    assert "\n  " not in result  # No pretty-printing


def test_slr_program_unknown_gate():
    """Test that unknown gates raise ValueError."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 1)

    with pytest.raises(ValueError, match="Unknown gate") as exc_info:
        prog.add_gate("UnknownGate", [("q", 0)])

    assert "Unknown gate" in str(exc_info.value)


# =============================================================================
# Integration Tests
# =============================================================================


def test_roundtrip_compile_and_build():
    """Test that compiled and built programs produce similar structure."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }"""

    # Compile from source
    compiled = zluppy.compile_to_slr(source)

    # Build equivalent program
    prog = zluppy.SlrProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("H", [("q", 0)])
    prog.add_gate("CX", [("q", 0), ("q", 1)])
    built = prog.to_dict()

    # Both should have same structure
    assert compiled["type"] == built["type"]
    assert compiled["name"] == built["name"]
    assert len(compiled["body"]) == len(built["body"])


def test_all_single_qubit_gates():
    """Test all supported single-qubit gates."""
    gates = ["H", "X", "Y", "Z", "SX", "SY", "SZ", "SXdg", "SYdg", "SZdg", "T", "Tdg", "F", "Fdg", "F4", "F4dg"]

    for gate in gates:
        prog = zluppy.SlrProgram("test")
        prog.add_allocator("q", 1)
        prog.add_gate(gate, [("q", 0)])

        result = prog.to_dict()
        assert result["body"][0]["gate"] == gate


def test_all_two_qubit_gates():
    """Test all supported two-qubit gates."""
    gates = ["CX", "CY", "CZ", "CH", "SWAP", "iSWAP", "SXX", "SYY", "SZZ", "SXXdg", "SYYdg", "SZZdg"]

    for gate in gates:
        prog = zluppy.SlrProgram("test")
        prog.add_allocator("q", 2)
        prog.add_gate(gate, [("q", 0), ("q", 1)])

        result = prog.to_dict()
        assert result["body"][0]["gate"] == gate


def test_three_qubit_gates():
    """Test three-qubit gates (CCX/Toffoli)."""
    prog = zluppy.SlrProgram("test")
    prog.add_allocator("q", 3)
    prog.add_gate("CCX", [("q", 0), ("q", 1), ("q", 2)])

    result = prog.to_dict()
    assert result["body"][0]["gate"] == "CCX"


def test_all_rotation_gates():
    """Test all supported rotation gates."""
    # Single-qubit rotations
    for gate in ["RX", "RY", "RZ"]:
        prog = zluppy.SlrProgram("test")
        prog.add_allocator("q", 1)
        prog.add_gate(gate, [("q", 0)], [1.57])

        result = prog.to_dict()
        assert result["body"][0]["gate"] == gate
        assert len(result["body"][0]["params"]) == 1


def test_two_qubit_rotation_gates():
    """Test two-qubit rotation gates."""
    # CRZ and RZZ need 2 qubits + 1 angle
    for gate in ["CRZ", "RZZ"]:
        prog = zluppy.SlrProgram("test")
        prog.add_allocator("q", 2)
        prog.add_gate(gate, [("q", 0), ("q", 1)], [1.57])

        result = prog.to_dict()
        assert result["body"][0]["gate"] == gate
        assert len(result["body"][0]["params"]) == 1


def test_lowercase_gate_aliases():
    """Test lowercase gate aliases in builder API."""
    # Lowercase names should map to uppercase SLR-AST output
    for gate, expected in [("h", "H"), ("x", "X"), ("cx", "CX"), ("rx", "RX")]:
        prog = zluppy.SlrProgram("test")
        prog.add_allocator("q", 2)
        if gate in ["cx"]:
            prog.add_gate(gate, [("q", 0), ("q", 1)])
        elif gate in ["rx"]:
            prog.add_gate(gate, [("q", 0)], [0.5])
        else:
            prog.add_gate(gate, [("q", 0)])

        result = prog.to_dict()
        assert result["body"][0]["gate"] == expected


# =============================================================================
# ZluppyEngine Tests
# =============================================================================


def test_zluppy_engine_source():
    """Test ZluppyEngine with source code."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()

    assert isinstance(hugr_bytes, bytes)
    assert len(hugr_bytes) > 0


def test_zluppy_engine_file(tmp_path):
    """Test ZluppyEngine with file input."""
    # Create a temporary .zlp file
    zlp_file = tmp_path / "test.zlp"
    zlp_file.write_text("""fn main() -> void {
        var q = qalloc(1);
        h(q[0]);
    }""")

    engine = zluppy.ZluppyEngine().file(str(zlp_file))
    hugr_bytes = engine.to_hugr_bytes()

    assert isinstance(hugr_bytes, bytes)
    assert len(hugr_bytes) > 0


def test_zluppy_engine_run():
    """Test ZluppyEngine.run() executes through simulator."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
        cx(q[0], q[1]);
    }"""

    result = zluppy.ZluppyEngine().source(source).run(shots=10)

    # Check we got results
    assert result is not None
    result_dict = result.to_dict()
    assert "measurements" in result_dict
    assert len(result_dict["measurements"]) == 10

    # Verify Bell state correlations (q0 == q1 for each shot)
    for shot in result_dict["measurements"]:
        assert shot[0] == shot[1], f"Bell state violation: {shot}"


def test_zluppy_engine_run_single_qubit():
    """Test ZluppyEngine with single qubit circuit."""
    source = """fn main() -> void {
        var q = qalloc(1);
        x(q[0]);
    }"""

    result = zluppy.ZluppyEngine().source(source).run(shots=5)
    result_dict = result.to_dict()

    # X gate should always give |1⟩
    for shot in result_dict["measurements"]:
        assert shot == [1], f"Expected [1], got {shot}"


def test_zluppy_engine_no_source_error():
    """Test that to_hugr_bytes raises without source."""
    engine = zluppy.ZluppyEngine()

    with pytest.raises(ValueError, match="No source compiled") as exc_info:
        engine.to_hugr_bytes()

    assert "No source compiled" in str(exc_info.value)


def test_zluppy_engine_parse_error():
    """Test that ZluppyEngine raises on parse error."""
    source = "fn main() -> void { h(q[0] }"  # Missing closing paren

    with pytest.raises(zluppy.ZluppyError):
        zluppy.ZluppyEngine().source(source)


def test_zluppy_engine_semantic_error():
    """Test that ZluppyEngine raises on semantic error."""
    source = """fn main() -> void {
        h(undefined_var[0]);
    }"""

    with pytest.raises(zluppy.ZluppyError):
        zluppy.ZluppyEngine().source(source)


def test_zluppy_engine_strict_mode():
    """Test ZluppyEngine with strict mode."""
    source = """fn main() -> void {
        var q = qalloc(2);
        h(q[0]);
    }"""

    # Strict mode should work for valid code
    engine = zluppy.ZluppyEngine(strict=True).source(source)
    hugr_bytes = engine.to_hugr_bytes()

    assert len(hugr_bytes) > 0


def test_zluppy_engine_chaining():
    """Test that ZluppyEngine methods return self for chaining."""
    source = """fn main() -> void {
        var q = qalloc(1);
        h(q[0]);
    }"""

    # All methods should be chainable
    result = zluppy.ZluppyEngine().source(source).run(shots=1)
    assert result is not None


def test_zluppy_engine_repr():
    """Test ZluppyEngine repr."""
    engine = zluppy.ZluppyEngine()
    assert "not compiled" in repr(engine)

    engine.source("fn main() -> void { var q = qalloc(1); }")
    assert "compiled" in repr(engine)


def test_zluppy_engine_file_not_found():
    """Test ZluppyEngine raises on missing file."""
    with pytest.raises(OSError, match="Failed to read"):
        zluppy.ZluppyEngine().file("/nonexistent/path/to/file.zlp")


# =============================================================================
# End-to-End Gate Tests (compile through HUGR)
# =============================================================================


def test_engine_ch_gate():
    """Test CH (controlled Hadamard) gate compiles to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(2);
        ch(q[0], q[1]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_ising_gates():
    """Test Ising gates (SXX, SYY, SZZ) compile to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(2);
        sxx(q[0], q[1]);
        syy(q[0], q[1]);
        szz(q[0], q[1]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_ising_dagger_gates():
    """Test Ising dagger gates compile to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(2);
        sxxdg(q[0], q[1]);
        syydg(q[0], q[1]);
        szzdg(q[0], q[1]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_rzz_gate():
    """Test RZZ (ZZ rotation) gate compiles to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(2);
        rzz(q[0], q[1], 1.57);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_f_gates():
    """Test F gates (Clifford face rotation) compile to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(1);
        f(q[0]);
        fdg(q[0]);
        f4(q[0]);
        f4dg(q[0]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_ccx_gate():
    """Test CCX (Toffoli) gate compiles to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(3);
        ccx(q[0], q[1], q[2]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_engine_swap_gates():
    """Test SWAP and iSWAP gates compile to HUGR."""
    source = """fn main() -> void {
        var q = qalloc(2);
        swap(q[0], q[1]);
        iswap(q[0], q[1]);
    }"""

    engine = zluppy.ZluppyEngine().source(source)
    hugr_bytes = engine.to_hugr_bytes()
    assert len(hugr_bytes) > 0


def test_compile_all_new_gates():
    """Test compiling a program with all new gates to SLR."""
    source = """fn main() -> void {
        var q = qalloc(3);
        // F gates
        f(q[0]);
        fdg(q[0]);
        f4(q[0]);
        f4dg(q[0]);
        // Two-qubit controlled
        ch(q[0], q[1]);
        // Ising gates
        sxx(q[0], q[1]);
        syy(q[0], q[1]);
        szz(q[0], q[1]);
        sxxdg(q[0], q[1]);
        syydg(q[0], q[1]);
        szzdg(q[0], q[1]);
        // Swap gates
        swap(q[0], q[1]);
        iswap(q[0], q[1]);
        // Rotation
        rzz(q[0], q[1], 0.5);
        crz(q[0], q[1], 0.5);
        // Three-qubit
        ccx(q[0], q[1], q[2]);
    }"""

    result = zluppy.compile_to_slr(source)
    assert result["type"] == "Program"
    # Should have 16 operations in body
    assert len(result["body"]) == 16


# =============================================================================
# ZlupProgram Builder Tests
# =============================================================================


def test_zlup_program_create():
    """Test creating a ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    assert repr(prog) == 'ZlupProgram(name="main", statements=0, strict=false)'


def test_zlup_program_add_allocator():
    """Test adding an allocator to ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)

    source = prog.to_source()
    assert "var q = qalloc(2);" in source


def test_zlup_program_add_gate():
    """Test adding a gate to ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    source = prog.to_source()
    assert "h(q[0]);" in source
    assert "cx(q[0], q[1]);" in source


def test_zlup_program_add_rotation_gate():
    """Test adding a rotation gate with parameters."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 1)
    prog.add_gate("rz", [("q", 0)], params=[1.57])

    source = prog.to_source()
    assert "rz(q[0], 1.57);" in source


def test_zlup_program_compile_to_slr():
    """Test compiling ZlupProgram to SLR-AST."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    slr_json = prog.compile_to_slr()
    result = json.loads(slr_json)

    assert result["type"] == "Program"
    assert result["name"] == "main"
    assert len(result["body"]) == 2
    assert result["body"][0]["gate"] == "H"
    assert result["body"][1]["gate"] == "CX"


def test_zlup_program_compile_to_slr_dict():
    """Test compiling ZlupProgram to SLR-AST dict."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 1)
    prog.add_gate("x", [("q", 0)])

    result = prog.compile_to_slr_dict()

    assert isinstance(result, dict)
    assert result["body"][0]["gate"] == "X"


def test_zlup_program_compile_to_hugr():
    """Test compiling ZlupProgram to HUGR."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    hugr_bytes = prog.compile_to_hugr()

    assert isinstance(hugr_bytes, bytes)
    assert len(hugr_bytes) > 0


def test_zlup_program_method_chaining():
    """Test that ZlupProgram methods support chaining."""
    prog = (
        zluppy.ZlupProgram("main").add_allocator("q", 2).add_gate("h", [("q", 0)]).add_gate("cx", [("q", 0), ("q", 1)])
    )

    source = prog.to_source()
    assert "var q = qalloc(2);" in source
    assert "h(q[0]);" in source
    assert "cx(q[0], q[1]);" in source


def test_zlup_program_all_single_qubit_gates():
    """Test all single-qubit gates in ZlupProgram."""
    gates = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "sy", "sydg", "sz", "szdg"]

    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 1)
    for gate in gates:
        prog.add_gate(gate, [("q", 0)])

    result = json.loads(prog.compile_to_slr())
    assert len(result["body"]) == len(gates)


def test_zlup_program_all_two_qubit_gates():
    """Test all two-qubit gates in ZlupProgram."""
    gates = ["cx", "cy", "cz", "ch", "sxx", "syy", "szz", "sxxdg", "syydg", "szzdg"]

    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    for gate in gates:
        prog.add_gate(gate, [("q", 0), ("q", 1)])

    result = json.loads(prog.compile_to_slr())
    assert len(result["body"]) == len(gates)


def test_zlup_program_rotation_gates():
    """Test rotation gates in ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("rx", [("q", 0)], params=[0.5])
    prog.add_gate("ry", [("q", 0)], params=[0.5])
    prog.add_gate("rz", [("q", 0)], params=[0.5])
    prog.add_gate("rzz", [("q", 0), ("q", 1)], params=[0.5])

    result = json.loads(prog.compile_to_slr())
    assert len(result["body"]) == 4
    assert result["body"][0]["gate"] == "RX"
    assert result["body"][3]["gate"] == "RZZ"


def test_zlup_program_prepare():
    """Test prepare operation in ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_prepare("q")  # Prepare all
    prog.add_gate("h", [("q", 0)])

    source = prog.to_source()
    assert "q.prepare_all();" in source


def test_zlup_program_measure():
    """Test measure operation in ZlupProgram."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_measure([("q", 0), ("q", 1)])

    source = prog.to_source()
    assert "measure(q[0], q[1]);" in source


def test_zlup_program_unknown_gate_error():
    """Test that unknown gates raise an error."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 1)

    with pytest.raises(ValueError, match="Unknown gate") as exc_info:
        prog.add_gate("unknown_gate", [("q", 0)])

    assert "Unknown gate" in str(exc_info.value)


def test_zlup_program_strict_mode():
    """Test ZlupProgram with strict mode."""
    prog = zluppy.ZlupProgram("main", strict=True)
    assert "strict=true" in repr(prog)


def test_zlup_program_save(tmp_path):
    """Test saving ZlupProgram to a .zlp file."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    path = tmp_path / "bell.zlp"
    prog.save(str(path))

    # Verify file contents
    content = path.read_text()
    assert "fn main() -> void {" in content
    assert "var q = qalloc(2);" in content
    assert "h(q[0]);" in content
    assert "cx(q[0], q[1]);" in content

    # Verify the saved file can be compiled
    result = zluppy.compile_file(str(path))
    assert result["type"] == "Program"


def test_zlup_program_compile_via_source_to_slr():
    """Test compile_via_source_to_slr round-trips correctly."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    # Compile via AST
    slr_ast = prog.compile_to_slr()

    # Compile via source (generate -> parse -> compile)
    slr_source = prog.compile_via_source_to_slr()

    # Both should produce equivalent results
    ast_data = json.loads(slr_ast)
    source_data = json.loads(slr_source)

    assert ast_data["name"] == source_data["name"]
    assert len(ast_data["body"]) == len(source_data["body"])
    assert [g["gate"] for g in ast_data["body"]] == [g["gate"] for g in source_data["body"]]


def test_zlup_program_compile_via_source_to_hugr():
    """Test compile_via_source_to_hugr works."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 2)
    prog.add_gate("h", [("q", 0)])
    prog.add_gate("cx", [("q", 0), ("q", 1)])

    hugr_bytes = prog.compile_via_source_to_hugr()

    assert isinstance(hugr_bytes, bytes)
    assert len(hugr_bytes) > 0


def test_zlup_program_roundtrip_all_gates():
    """Test that all gates round-trip through source generation."""
    prog = zluppy.ZlupProgram("main")
    prog.add_allocator("q", 3)

    # Add various gates
    single_qubit = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"]
    two_qubit = ["cx", "cy", "cz", "ch", "sxx", "syy", "szz"]

    for gate in single_qubit:
        prog.add_gate(gate, [("q", 0)])
    for gate in two_qubit:
        prog.add_gate(gate, [("q", 0), ("q", 1)])
    prog.add_gate("rz", [("q", 0)], params=[1.57])

    # Should compile successfully via source
    slr = prog.compile_via_source_to_slr()
    data = json.loads(slr)

    expected_count = len(single_qubit) + len(two_qubit) + 1  # +1 for rz
    assert len(data["body"]) == expected_count

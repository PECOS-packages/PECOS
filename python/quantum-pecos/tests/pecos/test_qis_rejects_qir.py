"""QIR must be converted before its shared intrinsic names reach the QIS ABI."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

QIR = """
define void @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr inttoptr (i64 1 to ptr))
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)
  ret void
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="2" "required_num_results"="2" }
"""


def test_qis_rejects_qir_before_execution(tmp_path: Path) -> None:
    """A subprocess contains crashes if a wrong-ABI program reaches execution."""
    script = textwrap.dedent(
        f"""
        import pecos

        try:
            pecos.sim(pecos.Qis({QIR!r})).qubits(2).quantum(
                pecos.state_vector()
            ).seed(1).workers(1).run(5)
        except (RuntimeError, ValueError) as error:
            assert "convert QIR to QIS first" in str(error), str(error)
        else:
            raise AssertionError("Qis accepted QIR input and executed its incompatible ABI")
        print("QIR rejected; Python remains alive")
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PECOS_CACHE_DIR": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QIR rejected; Python remains alive" in result.stdout


def test_converted_qir_preserves_the_requested_result(tmp_path: Path) -> None:
    """The native converter output is valid QIS and keeps result 1 distinct."""
    script = textwrap.dedent(
        f"""
        import subprocess
        import pecos
        import pecos_rslib
        import qir_qis

        bitcode = qir_qis.qir_to_qis(qir_qis.qir_ll_to_bc({QIR!r}), target="native")
        ir = subprocess.run(
            [pecos_rslib.find_llvm_tool("llvm-dis"), "-o", "-"],
            input=bitcode, capture_output=True, check=True,
        ).stdout.decode()
        result = pecos.sim(pecos.Qis(ir)).qubits(2).quantum(
            pecos.state_vector()
        ).seed(1).workers(1).run(5).to_dict()
        assert result.get("USER:RESULT:result_1") == [1] * 5, result
        print("converted QIR accepted")
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PECOS_CACHE_DIR": str(tmp_path / "cache")},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "converted QIR accepted" in result.stdout

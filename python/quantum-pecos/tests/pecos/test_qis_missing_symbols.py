"""Missing program imports must not terminate the Python interpreter."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_missing_qis_symbol_raises_and_process_survives(tmp_path: Path) -> None:
    """The dynamic loader reports the missing import before executing QIS."""
    program = r"""
declare void @__quantum__qis__h__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)
declare i64 @get_current_shot()

define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    %s = call i64 @get_current_shot()
    %r = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    call void @__quantum__rt__result_record_output(i64 0, i8* null)
    ret i64 0
}

attributes #0 = { "EntryPoint" }
"""
    script = textwrap.dedent(
        """
        import sys
        import pecos

        code = sys.argv[1]
        def run(source):
            return (
                pecos.sim(pecos.Qis(source)).qubits(1)
                .quantum(pecos.state_vector()).seed(1).workers(1).run(4)
            )

        try:
            run(code)
        except RuntimeError as error:
            assert "get_current_shot" in str(error), str(error)
            print("caught missing get_current_shot", flush=True)
        else:
            raise AssertionError("An undefined QIS import must raise")

        valid = code.replace("%s = call i64 @get_current_shot()", "")
        result = run(valid).to_dict()
        assert result and all(len(values) == 4 for values in result.values()), result
        print("after", flush=True)
        """,
    )
    cache_dir = tmp_path / "program-cache"
    env = {**os.environ, "PECOS_CACHE_DIR": str(cache_dir)}
    # A second fresh interpreter loads the valid program from the persistent
    # cache and must compile, and reject, the missing import again.
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script, program],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "caught missing get_current_shot" in completed.stdout
        assert "after" in completed.stdout
    cached = [path for path in (cache_dir / "qis-programs").iterdir() if path.suffix in {".so", ".dll"}]
    assert len(cached) == 1, cached

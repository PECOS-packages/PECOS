"""Exercise WasmForeignObject from a built wheel on compatibility runners."""

from pathlib import Path

from pecos_rslib import WasmForeignObject

repo_root = Path(__file__).resolve().parents[2]
wat_path = repo_root / "crates" / "pecos-wasm" / "tests" / "wat" / "finite_start_loop.wat"

wasm = WasmForeignObject.from_file(wat_path)
try:
    wasm.init()
finally:
    wasm.teardown()

print(f"WASM compatibility smoke test passed: {wat_path.name}")

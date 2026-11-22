"""Public runner adapters; source imports and compilation stay inside preparation."""

import re
from contextlib import contextmanager

TRAP_MESSAGE = "WebAssembly function 'trap' trapped: wasm trap: wasm `unreachable` instruction executed"


def normalized_failure(error: Exception, message: str, code: int) -> dict:
    """Recognize the suite's exact Wasmtime trap, never stdout/stderr or generic errors."""
    if code == 2001 and message == TRAP_MESSAGE:
        return {"error": "wasm_trap"}
    raise error


def pecos_failure(error: RuntimeError) -> dict:
    """Extract the public QIS panic diagnostic; reject unrelated RuntimeErrors."""
    text = str(error)
    match = re.search(r"QIS program panic: code=(\d+), message=(.*)\Z", text)
    if text.startswith("Simulation failed:") and match:
        return normalized_failure(error, match[2], int(match[1]))
    raise error


def records_to_shot(records) -> dict:
    """Preserve the unique-label contract rather than dropping duplicate records."""
    shot = {}
    for label, value in records:
        if label in shot:
            message = "Duplicate result label"
            raise ValueError(message)
        shot[label] = value
    return shot


@contextmanager
def selene(program, modules, settings, build_dir):
    import selene_sim
    from pecos_selene_wasm import build
    from selene_sim.exceptions import SelenePanicError

    runner = build(
        program,
        modules,
        platform=settings["platform"],
        build_dir=build_dir,
        wasm_int_width=settings.get("wasm_int_width", 64),
    )

    def run():
        try:
            return {
                "shots": [
                    records_to_shot(shot)
                    for shot in runner.run_shots(
                        simulator=selene_sim.Quest(),
                        n_qubits=settings["qubits"],
                        n_shots=settings["shots"],
                        n_processes=settings["workers"],
                        random_seed=settings["seed"],
                    )
                ],
            }
        except SelenePanicError as error:
            return normalized_failure(error, error.message, error.code)

    yield run


@contextmanager
def pecos(program, modules, settings, _build_dir):
    import pecos as pc

    runner = (
        pc.sim(pc.Guppy(program, wasm_modules=modules, wasm_int_width=settings.get("wasm_int_width", 64)))
        .qubits(settings["qubits"])
        .quantum(pc.state_vector())
        .seed(settings["seed"])
        .workers(settings["workers"])
        .build()
    )

    def run():
        try:
            columns = runner.run(settings["shots"]).to_dict()
            return {"shots": [dict(zip(columns, row, strict=True)) for row in zip(*columns.values(), strict=True)]}
        except RuntimeError as error:
            return pecos_failure(error)

    yield run

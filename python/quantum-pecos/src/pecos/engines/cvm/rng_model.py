"""This Module is responsible for keeping track of the state for generating a sequence of random numbers.

It handles RNG platform function calls that are handled by the pcg_rng library.

"""

# TODO: A Rust version of this RNGModel should be created for Rust side usage and then exposed via pecos-rslib/PyO3

from __future__ import annotations

from pecos_rslib import RngPcg

from pecos import BitUInt


class RNGModel:
    """This class is responsible for the functionality of generating a sequence of random numbers."""

    def __init__(
        self,
        shot_id: int,
        seed: int = 0,
        current_bound: int | None = 0,
    ) -> None:
        """Constructs an RNGModel object."""
        self.shot_id = shot_id
        self.current_bound = self._normalize_bound(current_bound)
        self.count = 0
        self._draw_bound_runs: list[tuple[int, int]] = []
        self.pcg = RngPcg()
        self._replay_base_pcg = self.pcg.clone()
        self.set_seed(seed)

    def __str__(self) -> str:
        """Returns the str representation of the model."""
        return f"RNG Model bounded by {self.current_bound} with current count {self.count}"

    def set_seed(self, seed: int) -> None:
        """Setting the seed for generating random numbers."""
        self.seed = seed
        self.pcg.srandom(seed)
        self.count = 0
        self._draw_bound_runs = []
        self._replay_base_pcg = self.pcg.clone()

    def start_shot(self, shot_id: int) -> None:
        """Reset shot-local replay state while preserving the current stream position."""
        self.shot_id = shot_id
        self.count = 0
        self._draw_bound_runs = []
        self._replay_base_pcg = self.pcg.clone()

    def set_bound(self, bound: int | None) -> None:
        """Setting the current bound for generating random numbers."""
        self.current_bound = self._normalize_bound(bound)

    @staticmethod
    def _require_non_negative(name: str, value: int) -> None:
        """Raise a clear error when an RNG parameter is negative."""
        if value < 0:
            error_msg = f"RNG {name} must be non-negative: got {value}"
            raise ValueError(error_msg)

    @classmethod
    def _normalize_bound(cls, bound: int | None) -> int:
        """Normalize optional bounds to the unbounded sentinel used by the RNG model."""
        if bound is None:
            return 0
        cls._require_non_negative("bound", bound)
        return bound

    def rng_random(self) -> int:
        """Generating a random number and keeping track of how many we have generated."""
        rng_num = self.pcg.random() if self.current_bound == 0 else self.pcg.boundedrand(self.current_bound)
        if self._draw_bound_runs and self._draw_bound_runs[-1][0] == self.current_bound:
            bound, run_length = self._draw_bound_runs[-1]
            self._draw_bound_runs[-1] = (bound, run_length + 1)
        else:
            self._draw_bound_runs.append((self.current_bound, 1))
        self.count += 1
        return rng_num

    def set_index(self, index: int) -> None:
        """Setting the index for the random number sequence.

        The number after from the stream will be the idx of interest.
        """
        self._require_non_negative("index", index)
        if self.count > index:
            error_msg = f"RNGindex({index}) cannot move backward: current stream index is {self.count}"
            raise ValueError(error_msg)
        while self.count < index:
            self.rng_random()

    def set_relative_index(self, delta: int) -> None:
        """Move relative to the current random-number stream index."""
        target_index = self.count + delta
        if target_index < 0:
            error_msg = (
                f"RNGadvance({delta}) cannot move before the start of the stream: "
                f"current stream index is {self.count}"
            )
            raise ValueError(error_msg)

        if delta < 0:
            bound_runs = list(self._draw_bound_runs)
            active_bound = self.current_bound
            self.pcg = self._replay_base_pcg.clone()
            self.count = 0
            self._draw_bound_runs = []

            remaining = target_index
            for historical_bound, run_length in bound_runs:
                if remaining <= 0:
                    break
                self.current_bound = historical_bound
                replay_count = min(run_length, remaining)
                for _ in range(replay_count):
                    self.rng_random()
                remaining -= replay_count

            self.current_bound = active_bound
        else:
            while self.count < target_index:
                self.rng_random()

    def extract_val(self, param: str | int, output: dict) -> int:
        """Responsible for extracting the value of interest depending on the type of the parameter being passed in."""
        if isinstance(param, int):
            return param

        try:
            return int(param)
        except (TypeError, ValueError):
            pass

        if "[" in param:
            idx_creg = param.split("[")
            creg = output[idx_creg[0]]
            idx = int(idx_creg[-1][:-1])
            return int(creg[idx])
        if param == "JOB_shotnum":
            return self.shot_id

        reg = output[param]
        return int(reg)

    def eval_func(self, params: dict, output: dict) -> None:
        """Calling the appropriate functions dependent on RNG Function call passed in."""
        func_name = params.get("func")
        if func_name == "RNGseed":
            seed_var = params.get("args")[0]
            seed = self.extract_val(seed_var, output)
            self.set_seed(seed)
        elif func_name == "RNGbound":
            bound_var = params.get("args")[0]
            bound = self.extract_val(bound_var, output)
            self.set_bound(bound)
        elif func_name == "RNGindex":
            index_var = params.get("args")[0]
            index = self.extract_val(index_var, output)
            self.set_index(index)
        elif func_name == "RNGadvance":
            delta_var = params.get("args")[0]
            delta = self.extract_val(delta_var, output)
            self.set_relative_index(delta)
        elif func_name == "RNGnum":
            creg_name = params.get("assign_vars")[0]
            creg = output[creg_name]
            rng = self.rng_random()
            binary_val = BitUInt(creg.size, rng)
            creg.set(binary_val)
        else:
            error_msg = f"Unknown RNG Function '{func_name}'"
            raise ValueError(error_msg)

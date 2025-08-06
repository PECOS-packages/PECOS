"""This Module is responsible for keeping track of the state for generating a sequence of random numbers.

It handles RNG platform function calls that are handled by the pcg_rng library.

"""

# TODO: A Rust version of this RNGModel should be created for Rust side usage and then exposed via pecos-rslib/PyO3

from __future__ import annotations

from pecos_rslib._pecos_rslib import RngPcg

from pecos.engines.cvm.binarray import BinArray


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
        self.current_bound = current_bound
        self.count = 0
        self.pcg = RngPcg()
        self.seed = self.set_seed(seed)

    def __str__(self) -> str:
        """Returns the str representation of the model."""
        return f"RNG Model with bound {self.current_bound} with count {self.count}"

    def set_seed(self, seed: int) -> None:
        """Setting the seed for generating random numbers."""
        self.seed = seed
        self.pcg.srandom(seed)

    def set_bound(self, bound: int) -> None:
        """Setting the current bound for generating random numbers."""
        self.current_bound = bound

    def rng_random(self) -> int:
        """Generating a random number and keeping track of how many we have generated."""
        rng_num = (
            self.pcg.random()
            if self.current_bound == 0
            else self.pcg.boundedrand(self.current_bound)
        )
        self.count += 1
        return rng_num

    def set_index(self, index: int) -> None:
        """Setting the index for the random number sequence.

        The number after from the stream will be the idx of interest.
        """
        if self.count > index:
            error_msg = "rngindex called after specified already generated"
            raise BufferError(error_msg)
        while self.count < index:
            self.rng_random()

    def extract_val(self, param: str, output: dict) -> int:
        """Responsible for extracting the value of interest depending on the type of the parameter being passed in."""
        if param.isdigit():
            val = int(param)
        elif "[" in param:
            idx_creg = param.split("[")
            creg = output[idx_creg[0]]
            idx = int(idx_creg[-1][:-1])
            val = int(creg[idx])
        elif param == "JOB_shotnum":
            val = self.shot_id
        else:
            reg = output[param]
            val = int(reg)
        return val

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
        elif func_name == "RNGnum":
            creg_name = params.get("assign_vars")[0]
            creg = output[creg_name]
            rng = self.rng_random()
            binary_val = BinArray(creg.size, rng)
            creg.set(binary_val)
        else:
            error_msg = f"RNG function not supported {func_name}"
            raise ValueError(error_msg)

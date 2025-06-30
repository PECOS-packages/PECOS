import pecos_rng_pcg
from typing import Optional
from pecos.engines.cvm.binarray import BinArray

class RNGModel:
    def __init__(self, shot_id: int, seed:int=0, current_bound: Optional[int]=0) -> None:
        self.shot_id = shot_id
        self.current_bound = current_bound
        self.count = 0
        self.last_rand = 0
        self.seed = self.set_seed(seed)

    def __str__(self) -> str:
        return f'RNG Model with bound {self.current_bound} with count {self.count}'   

    def set_seed(self, seed:int) -> None:
        self.seed = seed
        pecos_rng_pcg.pcg32_srandom(seed)

    def set_bound(self, bound:int) -> None:
        self.current_bound = bound

    def rng_random(self) -> int:
        if self.current_bound == 0:
            rng_num = pecos_rng_pcg.pcg32_random()
        else:
            rng_num = pecos_rng_pcg.pcg32_boundedrand(self.current_bound)
        self.count+=1
        self.last_rand = rng_num
        return rng_num

    def set_index(self, index: int) -> None:
        if self.count > index:
            raise BufferError("rngindex called after specified already generated")
        # number after from the stream will be the idx of interest
        while self.count < index:
            self.rng_random()
    
    def extract_val(self, param, output):
        if param.isdigit():
            val = int(param)
        elif '[' in param:
            idx_creg = param.split('[')
            creg = output[idx_creg[0]]
            idx = int(idx_creg[-1][:-1])
            val = int(creg[idx])
        else:
            if param == 'JOB_shotnum':
                val = self.shot_id
            else:
                reg = output[param]
                val = int(reg)
        return val

    def eval_func(self, params, output):
        func_name = params.get('func')
        if func_name == 'RNGseed':
            seed_var = params.get('args')[0]
            seed = self.extract_val(seed_var, output)
            self.set_seed(seed)
        elif func_name == 'RNGbound':
            bound_var = params.get('args')[0]
            bound = self.extract_val(bound_var, output)
            self.set_bound(bound)
        elif func_name == 'RNGindex':
            index_var = params.get('args')[0]
            index = self.extract_val(index_var, output)
            self.set_index(index)
        elif func_name == 'RNGnum':
            creg_name = params.get('assign_vars')[0]
            creg = output[creg_name]
            rng = self.rng_random()
            binary_val = BinArray(creg.size, rng)
            creg.set(binary_val)
        else:
            raise ValueError(f'RNG function not supported {func_name}')
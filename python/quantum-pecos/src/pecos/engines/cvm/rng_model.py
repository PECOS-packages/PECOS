import pecos_rng_pcg
from typing import Optional

class RNGModel:
    def __init__(self, seed:int=0, current_bound: Optional[int]=0) -> None:
        self.current_bound = current_bound
        self.count = 0
        self.last_rand = 0
        self.seed = self.set_seed(seed)    

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
    
    def eval_func(self, op):
        pass


from .gen_guppy import GuppyGenerator, run_guppy_from_str

class HUGRGenerator:
    """Compiles SLR code to HUGR JSON"""

    @staticmethod
    def generate_block(block):
        """Compiles a SLR block to a target code"""
        guppy_str = block.gen(GuppyGenerator())
        mod = run_guppy_from_str(guppy_str)
        json_str = mod.compile_hugr().to_json()
        return json_str

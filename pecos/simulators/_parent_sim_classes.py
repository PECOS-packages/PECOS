
class BaseSim:

    def __init__(self):
        self.gate_dict = {}

    def run_gate(self, symbol, locations, **gate_kwargs):
        """

        Args:
            symbol:
            locations:
            **gate_kwargs:

        Returns:

        """

        output = {}
        for location in locations:
            results = self.gate_dict[symbol](self, location, **gate_kwargs)

            if results:
                output[location] = results

        return output

    def run_circuit(self, circuit):
        """

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """

        results = []

        for symbol, locations, gate_kwargs in circuit.items():
            gate_output = self.run_gate(symbol, locations, **gate_kwargs)
            results.append(gate_output)

        return results

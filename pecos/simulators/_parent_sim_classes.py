
class BaseSim(object):
    """
    A parent class to provide standard methods for simulators.
    """

    def __init__(self):
        self.bindings = {}

    def run_gate(self, symbol, locations, **params):
        """

        Args:
            symbol:
            locations:
            **params:

        Returns:

        """

        output = {}
        for location in locations:
            results = self.bindings[symbol](self, location, **params)

            if results:
                output[location] = results

        return output

    def run_circuit(self, circuit, removed_locations=None):
        """

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            removed_locations:

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """

        # TODO: removed_locations doesn't make sense except if circuit is tick_circuit
        # because can't say not to do gates for particular ticks....

        if removed_locations is None:
            removed_locations = set([])

        results = {}
        for symbol, locations, params in circuit.items():
            gate_results = self.run_gate(symbol, locations - removed_locations, **params)
            results.update(gate_results)

        return results

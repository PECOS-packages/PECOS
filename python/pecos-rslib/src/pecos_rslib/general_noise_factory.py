"""Factory class for creating GeneralNoiseModelBuilder from dict/JSON configuration.

This module provides a mapping between configuration keys and builder methods,
allowing general noise models to be constructed from dictionaries or JSON while
maintaining type safety and validation.
"""

from typing import Dict, Any, Callable, Optional
import json
import warnings
from dataclasses import dataclass
from pecos_rslib import GeneralNoiseModelBuilder


@dataclass
class MethodMapping:
    """Defines how a config key maps to a builder method."""

    method_name: str
    converter: Optional[Callable[[Any], Any]] = None
    description: str = ""
    apply_to_list: bool = False  # If True, apply method to each item in list

    def apply(
        self, builder: GeneralNoiseModelBuilder, value: Any
    ) -> GeneralNoiseModelBuilder:
        """Apply this mapping to the builder with the given value."""
        method = getattr(builder, self.method_name)

        if self.apply_to_list and isinstance(value, list):
            # Apply the method to each item in the list
            for item in value:
                converted_item = self.converter(item) if self.converter else item
                builder = method(converted_item)
            return builder
        else:
            # Normal single-value application
            if self.converter:
                value = self.converter(value)
            return method(value)


class GeneralNoiseFactory:
    """Factory for creating GeneralNoiseModelBuilder from configuration dictionaries.

    This class provides a mapping between configuration keys and builder methods,
    with support for type conversion, validation, and default values.

    Example:
        >>> config = {
        ...     "seed": 42,
        ...     "p1": 0.001,
        ...     "p2": 0.01,
        ...     "p_meas_0": 0.002,
        ...     "p_meas_1": 0.002,
        ...     "scale": 1.5,
        ...     "noiseless_gates": ["H", "MEASURE"],
        ... }
        >>> factory = GeneralNoiseFactory()
        >>> builder = factory.create_from_dict(config)
        >>> sim = qasm_sim(qasm).noise(builder).build()
    """

    # Standard parameter mappings - extracted as class constant for clarity
    _STANDARD_MAPPINGS = {
        # Global parameters
        "seed": MethodMapping("with_seed", int, "Random seed for reproducibility"),
        "scale": MethodMapping("with_scale", float, "Global error rate scaling factor"),
        "leakage_scale": MethodMapping(
            "with_leakage_scale", float, "Leakage vs depolarizing ratio (0-1)"
        ),
        "emission_scale": MethodMapping(
            "with_emission_scale", float, "Spontaneous emission scaling"
        ),
        "seepage_prob": MethodMapping(
            "with_seepage_prob", float, "Global seepage probability for leaked qubits"
        ),
        # Single noiseless gate (string -> with_noiseless_gate)
        "noiseless_gate": MethodMapping(
            "with_noiseless_gate", str, "Single gate to make noiseless"
        ),
        # Multiple noiseless gates (list -> multiple with_noiseless_gate calls)
        "noiseless_gates": MethodMapping(
            "with_noiseless_gate",
            str,
            "List of gates to make noiseless",
            apply_to_list=True,
        ),
        # Idle noise parameters
        "p_idle_coherent": MethodMapping(
            "with_p_idle_coherent", bool, "Use coherent vs incoherent dephasing"
        ),
        "p_idle_linear_rate": MethodMapping(
            "with_p_idle_linear_rate", float, "Idle noise linear rate"
        ),
        "p_idle_average_linear_rate": MethodMapping(
            "with_average_p_idle_linear_rate", float, "Average idle noise linear rate"
        ),
        "p_idle_linear_model": MethodMapping(
            "with_p_idle_linear_model", dict, "Idle noise Pauli distribution"
        ),
        "p_idle_quadratic_rate": MethodMapping(
            "with_p_idle_quadratic_rate", float, "Idle noise quadratic rate"
        ),
        "p_idle_average_quadratic_rate": MethodMapping(
            "with_average_p_idle_quadratic_rate",
            float,
            "Average idle noise quadratic rate",
        ),
        "p_idle_coherent_to_incoherent_factor": MethodMapping(
            "with_p_idle_coherent_to_incoherent_factor",
            float,
            "Coherent to incoherent conversion factor",
        ),
        "idle_scale": MethodMapping(
            "with_idle_scale", float, "Idle noise scaling factor"
        ),
        # State preparation
        "p_prep": MethodMapping(
            "with_prep_probability", float, "State preparation error probability"
        ),
        "p_prep_leak_ratio": MethodMapping(
            "with_prep_leak_ratio", float, "Fraction of prep errors that leak"
        ),
        "p_prep_crosstalk": MethodMapping(
            "with_p_prep_crosstalk", float, "Preparation crosstalk probability"
        ),
        "prep_scale": MethodMapping(
            "with_prep_scale", float, "Preparation error scaling factor"
        ),
        "p_prep_crosstalk_scale": MethodMapping(
            "with_p_prep_crosstalk_scale", float, "Preparation crosstalk scaling"
        ),
        # Single-qubit gates
        "p1": MethodMapping(
            "with_p1_probability", float, "Single-qubit gate error probability"
        ),
        "p1_average": MethodMapping(
            "with_average_p1_probability", float, "Average single-qubit error"
        ),
        "p1_emission_ratio": MethodMapping(
            "with_p1_emission_ratio", float, "Fraction that are emission errors"
        ),
        "p1_emission_model": MethodMapping(
            "with_p1_emission_model", dict, "Single-qubit emission error distribution"
        ),
        "p1_seepage_prob": MethodMapping(
            "with_p1_seepage_prob", float, "Probability of seeping leaked qubits"
        ),
        "p1_pauli_model": MethodMapping(
            "with_p1_pauli_model",
            dict,
            "Pauli error distribution for single-qubit gates",
        ),
        "p1_scale": MethodMapping(
            "with_p1_scale", float, "Single-qubit error scaling factor"
        ),
        # Two-qubit gates
        "p2": MethodMapping(
            "with_p2_probability", float, "Two-qubit gate error probability"
        ),
        "p2_average": MethodMapping(
            "with_average_p2_probability", float, "Average two-qubit error"
        ),
        "p2_angle_params": MethodMapping(
            "with_p2_angle_params", tuple, "RZZ angle-dependent error params (a,b,c,d)"
        ),
        "p2_angle_power": MethodMapping(
            "with_p2_angle_power", float, "Power parameter for angle-dependent errors"
        ),
        "p2_emission_ratio": MethodMapping(
            "with_p2_emission_ratio", float, "Fraction that are emission errors"
        ),
        "p2_emission_model": MethodMapping(
            "with_p2_emission_model", dict, "Two-qubit emission error distribution"
        ),
        "p2_seepage_prob": MethodMapping(
            "with_p2_seepage_prob", float, "Probability of seeping leaked qubits"
        ),
        "p2_pauli_model": MethodMapping(
            "with_p2_pauli_model", dict, "Pauli error distribution for two-qubit gates"
        ),
        "p2_idle": MethodMapping(
            "with_p2_idle", float, "Idle noise after two-qubit gates"
        ),
        "p2_scale": MethodMapping(
            "with_p2_scale", float, "Two-qubit error scaling factor"
        ),
        # Measurement
        "p_meas": MethodMapping(
            "with_meas_probability",
            float,
            "Symmetric measurement error (sets both 0->1 and 1->0)",
        ),
        "p_meas_0": MethodMapping(
            "with_meas_0_probability", float, "Probability of 0->1 measurement flip"
        ),
        "p_meas_1": MethodMapping(
            "with_meas_1_probability", float, "Probability of 1->0 measurement flip"
        ),
        "p_meas_crosstalk": MethodMapping(
            "with_p_meas_crosstalk", float, "Measurement crosstalk probability"
        ),
        "meas_scale": MethodMapping(
            "with_meas_scale", float, "Measurement error scaling factor"
        ),
        "p_meas_crosstalk_scale": MethodMapping(
            "with_p_meas_crosstalk_scale", float, "Measurement crosstalk scaling"
        ),
    }

    def __init__(self, use_defaults: bool = True):
        """Initialize the factory with optional default mappings.

        Args:
            use_defaults: If True, initialize with standard parameter mappings.
                         If False, start with empty mappings.
        """
        if use_defaults:
            self.mappings = dict(self._STANDARD_MAPPINGS)
            self._default_mappings = dict(self._STANDARD_MAPPINGS)
        else:
            self.mappings: Dict[str, MethodMapping] = {}
            self._default_mappings: Dict[str, MethodMapping] = {}

        # Default values to apply if not specified by user
        self.defaults: Dict[str, Any] = {}

    def add_mapping(
        self,
        key: str,
        method_name: str,
        converter: Optional[Callable] = None,
        description: str = "",
    ) -> None:
        """Add or update a configuration key mapping.

        Args:
            key: Configuration dictionary key
            method_name: Builder method name to call
            converter: Optional function to convert the value
            description: Human-readable description
        """
        # Check if we're overriding a default mapping
        if key in self._default_mappings and key in self.mappings:
            old_method = self.mappings[key].method_name
            if old_method != method_name:
                warnings.warn(
                    f"Overriding default mapping for '{key}': "
                    f"'{old_method}' -> '{method_name}'. "
                    f"This may cause unexpected behavior.",
                    UserWarning,
                    stacklevel=2,
                )

        self.mappings[key] = MethodMapping(method_name, converter, description)

    def remove_mapping(self, key: str) -> bool:
        """Remove a parameter mapping.

        Args:
            key: Configuration key to remove

        Returns:
            True if the key was removed, False if it didn't exist

        Example:
            >>> factory = GeneralNoiseFactory()
            >>> factory.remove_mapping("p1_total")  # Remove alias
            >>> factory.remove_mapping("p_meas_0")  # Remove another alias
        """
        if key in self.mappings:
            del self.mappings[key]
            return True
        return False

    def set_default(self, key: str, value: Any) -> None:
        """Set a default value for a configuration key.

        Args:
            key: Configuration key
            value: Default value to use if not provided
        """
        self.defaults[key] = value

    def create_from_dict(
        self, config: Dict[str, Any], strict: bool = True, apply_defaults: bool = True
    ) -> GeneralNoiseModelBuilder:
        """Create a GeneralNoiseModelBuilder from a configuration dictionary.

        Args:
            config: Configuration dictionary
            strict: If True, raise error for unknown keys. If False, ignore them.
            apply_defaults: If True, apply factory defaults before user config

        Returns:
            Configured GeneralNoiseModelBuilder

        Raises:
            ValueError: If strict=True and unknown keys are found
            ValueError: If a mapped value fails validation
        """
        # Start with a fresh builder
        builder = GeneralNoiseModelBuilder()

        # Apply defaults first if requested
        if apply_defaults:
            for key, value in self.defaults.items():
                if key in self.mappings:
                    mapping = self.mappings[key]
                    builder = mapping.apply(builder, value)

        # Check for unknown keys if strict mode
        if strict:
            unknown_keys = set(config.keys()) - set(self.mappings.keys())
            if unknown_keys:
                raise ValueError(
                    f"Unknown configuration keys: {unknown_keys}. "
                    f"Valid keys are: {sorted(self.mappings.keys())}"
                )

        # Apply user configuration
        for key, value in config.items():
            if key not in self.mappings:
                if not strict:
                    continue  # Skip unknown keys in non-strict mode

            mapping = self.mappings[key]

            # Apply mapping
            try:
                builder = mapping.apply(builder, value)
            except Exception as e:
                raise ValueError(f"Error applying '{key}': {e}") from e

        return builder

    def create_from_json(self, json_str: str, **kwargs) -> GeneralNoiseModelBuilder:
        """Create a GeneralNoiseModelBuilder from a JSON string.

        Args:
            json_str: JSON string containing configuration
            **kwargs: Additional arguments passed to create_from_dict

        Returns:
            Configured GeneralNoiseModelBuilder
        """
        config = json.loads(json_str)
        return self.create_from_dict(config, **kwargs)

    def get_available_keys(self) -> Dict[str, str]:
        """Get all available configuration keys with descriptions.

        Returns:
            Dictionary mapping keys to their descriptions
        """
        return {key: mapping.description for key, mapping in self.mappings.items()}

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate a configuration dictionary without creating a builder.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        # Check for unknown keys
        unknown_keys = set(config.keys()) - set(self.mappings.keys())
        if unknown_keys:
            errors["unknown_keys"] = f"Unknown keys: {unknown_keys}"

        # Try to apply each mapping to check for type errors
        test_builder = GeneralNoiseModelBuilder()
        for key, value in config.items():
            if key in self.mappings:
                try:
                    mapping = self.mappings[key]
                    mapping.apply(test_builder, value)
                except Exception as e:
                    errors[key] = str(e)

        return errors

    def show_mappings(self, show_descriptions: bool = True) -> None:
        """Display the current parameter mappings in a readable format.

        Args:
            show_descriptions: If True, include parameter descriptions
        """
        print("\nCurrent Parameter Mappings:")
        print("=" * 80)

        if show_descriptions:
            print(
                f"{'Configuration Key':<20} → {'Builder Method':<35} {'Description':<30}"
            )
            print("-" * 80)
            for key, mapping in sorted(self.mappings.items()):
                # Mark overridden defaults
                marker = (
                    "*"
                    if (
                        key in self._default_mappings
                        and self._default_mappings[key].method_name
                        != mapping.method_name
                    )
                    else " "
                )
                print(
                    f"{marker}{key:<19} → {mapping.method_name:<35} {mapping.description[:30]}"
                )
        else:
            print(f"{'Configuration Key':<20} → {'Builder Method':<35}")
            print("-" * 55)
            for key, mapping in sorted(self.mappings.items()):
                # Mark overridden defaults
                marker = (
                    "*"
                    if (
                        key in self._default_mappings
                        and self._default_mappings[key].method_name
                        != mapping.method_name
                    )
                    else " "
                )
                print(f"{marker}{key:<19} → {mapping.method_name:<35}")

        # Show defaults if any
        if self.defaults:
            print("\nDefault Values:")
            for key, value in sorted(self.defaults.items()):
                print(f"  {key}: {value}")

        # Show legend if there are overrides
        has_overrides = any(
            key in self._default_mappings
            and self._default_mappings[key].method_name != mapping.method_name
            for key, mapping in self.mappings.items()
        )
        if has_overrides:
            print("\n* = Overridden default mapping")

        print("=" * 80)

    @classmethod
    def with_defaults(cls) -> "GeneralNoiseFactory":
        """Create a factory with standard default mappings.

        Returns:
            GeneralNoiseFactory with all predefined mappings
        """
        return cls(use_defaults=True)

    @classmethod
    def empty(cls) -> "GeneralNoiseFactory":
        """Create an empty factory with no predefined mappings.

        Returns:
            GeneralNoiseFactory with no mappings

        Example:
            >>> factory = GeneralNoiseFactory.empty()
            >>> factory.add_mapping("my_p1", "with_p1_probability", float)
            >>> factory.add_mapping("my_p2", "with_p2_probability", float)
        """
        return cls(use_defaults=False)


# Global instance for convenience functions - created lazily to avoid import issues
_default_factory = None


def _get_default_factory() -> GeneralNoiseFactory:
    """Get or create the default factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = GeneralNoiseFactory()
    return _default_factory


def create_noise_from_dict(
    config: Dict[str, Any], **kwargs
) -> GeneralNoiseModelBuilder:
    """Convenience function to create noise model from dict using default factory.

    Args:
        config: Configuration dictionary
        **kwargs: Arguments passed to factory.create_from_dict()

    Returns:
        Configured GeneralNoiseModelBuilder

    Example:
        >>> noise = create_noise_from_dict(
        ...     {"seed": 42, "p1": 0.001, "p2": 0.01, "scale": 1.2}
        ... )
        >>> sim = qasm_sim(qasm).noise(noise).run(1000)
    """
    return _get_default_factory().create_from_dict(config, **kwargs)


def create_noise_from_json(json_str: str, **kwargs) -> GeneralNoiseModelBuilder:
    """Convenience function to create noise model from JSON using default factory.

    Args:
        json_str: JSON configuration string
        **kwargs: Arguments passed to factory.create_from_dict()

    Returns:
        Configured GeneralNoiseModelBuilder
    """
    return _get_default_factory().create_from_json(json_str, **kwargs)


# Example custom factory for specific use cases
class IonTrapNoiseFactory(GeneralNoiseFactory):
    """Specialized factory for ion trap noise models with appropriate defaults."""

    def __init__(self):
        super().__init__()

        # Ion trap specific defaults
        self.defaults = {
            "p_prep": 0.001,  # Typical state prep error
            "p1": 0.0001,  # Very good single-qubit gates
            "p2": 0.003,  # Two-qubit gates are limiting factor
            "p_meas_0": 0.001,  # Dark state error
            "p_meas_1": 0.005,  # Bright state error (typically higher)
            "scale": 1.0,
        }

        # Add ion trap specific mappings
        self.add_mapping(
            "motional_heating",
            "with_scale",
            lambda x: 1.0 + x * 0.1,  # Convert heating rate to scale
            "Motional heating rate (0-10)",
        )

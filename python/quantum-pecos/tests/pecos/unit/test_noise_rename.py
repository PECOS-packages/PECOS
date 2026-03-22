# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for module renames and deprecation shims (error_models -> noise, tools -> analysis)."""

import warnings

import pecos


def test_noise_module_accessible() -> None:
    """pecos.noise should be directly importable."""
    from pecos.noise import DepolarModel

    assert DepolarModel is not None


def test_noise_accessible_via_attribute() -> None:
    """pecos.noise should be accessible as an attribute."""
    assert hasattr(pecos, "noise")
    assert hasattr(pecos.noise, "DepolarModel")


def test_error_models_deprecation_warning() -> None:
    """Importing pecos.error_models should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import importlib

        # Force reimport to trigger the warning
        import pecos.error_models

        importlib.reload(pecos.error_models)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning from pecos.error_models"
        assert "pecos.noise" in str(deprecation_warnings[0].message)


def test_error_models_shim_exports_same_classes() -> None:
    """pecos.error_models should re-export everything from pecos.noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import pecos.error_models

        assert hasattr(pecos.error_models, "DepolarModel")
        assert pecos.error_models.DepolarModel is pecos.noise.DepolarModel


# ============================================================================
# tools -> analysis rename
# ============================================================================


def test_analysis_module_accessible() -> None:
    """pecos.analysis should be directly importable."""
    from pecos.analysis import VerifyStabilizers

    assert VerifyStabilizers is not None


def test_analysis_accessible_via_attribute() -> None:
    """pecos.analysis should be accessible as an attribute."""
    assert hasattr(pecos, "analysis")
    assert hasattr(pecos.analysis, "VerifyStabilizers")


def test_tools_deprecation_warning() -> None:
    """Importing pecos.tools should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import importlib

        import pecos.tools

        importlib.reload(pecos.tools)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning from pecos.tools"
        assert "pecos.analysis" in str(deprecation_warnings[0].message)


def test_tools_shim_exports_same_classes() -> None:
    """pecos.tools should re-export everything from pecos.analysis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import pecos.tools

        assert hasattr(pecos.tools, "VerifyStabilizers")
        assert pecos.tools.VerifyStabilizers is pecos.analysis.VerifyStabilizers

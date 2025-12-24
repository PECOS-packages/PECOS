# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Test configuration and shared fixtures."""

# Configure matplotlib to use non-interactive backend for tests (if available)
# This must be done before importing matplotlib.pyplot to avoid GUI backend issues on Windows
try:
    import matplotlib as mpl

    mpl.use("Agg")
except ImportError:
    # matplotlib is optional - only needed for visualization tests
    pass

# Note: llvmlite functionality is now always available via Rust (pecos_rslib.ir and pecos_rslib.binding)
# No need for conditional test skipping

# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""AST analysis passes.

This package provides analysis passes that operate on the AST representation,
including validation, resource counting, circuit depth analysis, and data flow.
"""

from pecos.slr.ast.analysis.connectivity_analyzer import (
    ConnectivityAnalyzer,
    ConnectivityResult,
    analyze_connectivity,
)
from pecos.slr.ast.analysis.data_flow import (
    DataFlowAnalyzer,
    DataFlowInfo,
    DataFlowResult,
    ValueUse,
    analyze_data_flow,
)
from pecos.slr.ast.analysis.dependency_analyzer import (
    DependencyAnalyzer,
    DependencyResult,
    analyze_dependencies,
)
from pecos.slr.ast.analysis.depth_analyzer import (
    DepthAnalyzer,
    DepthResult,
    analyze_depth,
)
from pecos.slr.ast.analysis.parallelism_analyzer import (
    ParallelismAnalyzer,
    ParallelismResult,
    analyze_parallelism,
)
from pecos.slr.ast.analysis.qubit_state_validator import (
    AstQubitStateValidator,
    QubitStateTracker,
    StateViolation,
    ValidationSlotState,
    validate_ast_qubit_states,
)
from pecos.slr.ast.analysis.qubit_usage_analyzer import (
    QubitRole,
    QubitUsageAnalyzer,
    QubitUsageResult,
    QubitUsageStats,
    analyze_qubit_usage,
)
from pecos.slr.ast.analysis.resource_counter import (
    ResourceCount,
    ResourceCounter,
    count_resources,
)
from pecos.slr.ast.analysis.t_count_analyzer import (
    TCountAnalyzer,
    TCountResult,
    analyze_t_count,
)

__all__ = [
    # Qubit state validation
    "AstQubitStateValidator",
    # Connectivity analysis
    "ConnectivityAnalyzer",
    "ConnectivityResult",
    # Data flow analysis
    "DataFlowAnalyzer",
    "DataFlowInfo",
    "DataFlowResult",
    # Dependency analysis
    "DependencyAnalyzer",
    "DependencyResult",
    # Depth analysis
    "DepthAnalyzer",
    "DepthResult",
    # Parallelism analysis
    "ParallelismAnalyzer",
    "ParallelismResult",
    # Qubit usage analysis
    "QubitRole",
    "QubitStateTracker",
    "QubitUsageAnalyzer",
    "QubitUsageResult",
    "QubitUsageStats",
    # Resource counting
    "ResourceCount",
    "ResourceCounter",
    "StateViolation",
    # T-count analysis
    "TCountAnalyzer",
    "TCountResult",
    "ValidationSlotState",
    "ValueUse",
    "analyze_connectivity",
    "analyze_data_flow",
    "analyze_dependencies",
    "analyze_depth",
    "analyze_parallelism",
    "analyze_qubit_usage",
    "analyze_t_count",
    "count_resources",
    "validate_ast_qubit_states",
]

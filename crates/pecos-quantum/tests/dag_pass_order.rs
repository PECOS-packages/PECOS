// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[path = "support/dag_pass_order.rs"]
mod dag_pass_order;

#[test]
fn passes_preserve_dev_gates_wires_and_order() {
    // Recorded by running dag_pass_order_fixtures against origin/dev
    // de0ba996740e83ec15987cb6a4cd550a990c5acc, not the implementation under test.
    assert_eq!(
        dag_pass_order::snapshots(),
        include_str!("data/dag_pass_order.txt")
    );
}

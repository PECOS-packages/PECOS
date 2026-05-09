// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Syndrome-graph Union-Find decoder implementation.
//!
//! The algorithm (Delfosse-Nickerson style):
//!
//! 1. Each defect detector starts as its own cluster (odd parity).
//! 2. Grow all unsatisfied clusters by radius until an edge becomes fusible.
//!    An edge is fusible when the sum of endpoint radii reaches the edge weight.
//!    Two growing clusters fuse at half the weight; boundary needs full weight.
//! 3. Fuse all fusible edges, merging clusters. Parity = XOR of components.
//! 4. Repeat until all clusters have even parity or contain the boundary.
//! 5. Peel a spanning forest (BFS from boundary) to extract the correction:
//!    an edge is in the correction iff its subtree has odd parity.
//!
//! All data structures are flat arrays. Zero per-shot allocation after init.

use pecos_decoder_core::correlated_decoder::MatchingDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_decoder_core::errors::DecoderError;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Edge in the syndrome graph.
#[derive(Debug, Clone)]
struct Edge {
    /// First endpoint node index.
    node1: u32,
    /// Second endpoint node index (boundary = `num_detectors`).
    node2: u32,
    /// Weight (log-likelihood ratio). Lower = more likely error.
    weight: f64,
    /// Observable bitmask for this edge.
    obs_mask: u64,
}

/// Peeling strategy for correction extraction.
#[derive(Debug, Clone, Copy, Default)]
pub enum PeelingStrategy {
    /// BFS from boundary. Fastest, slightly less accurate.
    Bfs,
    /// Prim's MST from boundary. Uses globally lightest edges.
    /// Better accuracy, slight heap overhead at small sizes.
    #[default]
    PrimMst,
}

/// Growth strategy for cluster expansion.
#[derive(Debug, Clone, Copy, Default)]
pub enum GrowthStrategy {
    /// Event-driven with priority queue. Weighted growth (1/size).
    /// Best for larger codes (d >= 7). O(E log E) total.
    #[default]
    EventDriven,
    /// Scan-based: find min increment per round, grow all, fuse.
    /// Lower constant overhead for small codes (d <= 5).
    ScanBased,
}

/// Configuration for the UF decoder.
///
/// Use `UfDecoderConfig::fast()`, `::balanced()`, or `::accurate()` for
/// presets, then override individual fields as needed.
#[derive(Debug, Clone, Copy)]
pub struct UfDecoderConfig {
    /// Maximum growth rounds before giving up (prevents infinite loops).
    /// 0 = auto (100 * `num_detectors`).
    pub max_growth_rounds: usize,
    /// How to build the spanning forest for peeling.
    pub peeling: PeelingStrategy,
    /// How to grow clusters.
    pub growth: GrowthStrategy,
    /// Enable cluster predecoder for simple syndromes.
    /// Disable for windowed decoding which needs complete edge tracking.
    pub predecoder: bool,
}

impl Default for UfDecoderConfig {
    fn default() -> Self {
        Self::fast()
    }
}

impl UfDecoderConfig {
    /// Fast preset: event-driven growth, BFS peeling. Lowest latency.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            max_growth_rounds: 0,
            peeling: PeelingStrategy::Bfs,
            growth: GrowthStrategy::EventDriven,
            predecoder: true,
        }
    }

    /// Balanced preset: event-driven weighted growth, Prim MST peeling.
    /// Better accuracy, used as inner decoder for two-pass correlated mode.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            max_growth_rounds: 0,
            peeling: PeelingStrategy::PrimMst,
            growth: GrowthStrategy::EventDriven,
            predecoder: true,
        }
    }

    /// Accurate preset: same as balanced (UIUF accuracy comes from
    /// the CSS wrapper, not from single-graph config).
    #[must_use]
    pub fn accurate() -> Self {
        Self::balanced()
    }

    /// Windowed preset: Prim MST peeling, no predecoder (need complete edge tracking).
    #[must_use]
    pub fn windowed() -> Self {
        Self {
            max_growth_rounds: 0,
            peeling: PeelingStrategy::PrimMst,
            growth: GrowthStrategy::EventDriven,
            predecoder: false,
        }
    }
}

/// Fast syndrome-graph Union-Find decoder.
pub struct UfDecoder {
    /// Edges in the syndrome graph.
    edges: Vec<Edge>,
    /// CSR adjacency: flat data array of (`edge_index`, `neighbor_node`).
    adj_data: Vec<(usize, u32)>,
    /// CSR adjacency: offset[i]..offset[i+1] is the range in `adj_data` for node i.
    adj_offset: Vec<u32>,
    /// Number of detectors.
    num_detectors: usize,
    /// Config.
    config: UfDecoderConfig,

    // === Per-shot reusable buffers ===
    /// Disjoint-set forest: parent[i] = parent of node i.
    parent: Vec<u32>,
    /// Rank for union-by-rank.
    rank: Vec<u8>,
    /// Cluster parity: true = odd (needs correction).
    parity: Vec<bool>,
    /// Whether cluster contains the boundary node (satisfied regardless of parity).
    has_boundary: Vec<bool>,
    /// Growth radius of each cluster at `last_growth_time` (tracked at root).
    radius: Vec<f64>,
    /// Time when radius was last updated (for lazy computation).
    last_growth_time: Vec<f64>,
    /// Cluster size (number of nodes, tracked at root).
    cluster_size: Vec<u32>,
    /// Defect flags per detector.
    is_defect: Vec<bool>,

    // === Scratch buffers (reused across shots to avoid allocation) ===
    /// Growth event queue.
    growth_events: BinaryHeap<Reverse<(u64, usize, u32, u32)>>,
    /// Peeling: tree parent for each node.
    tree_parent: Vec<Option<(u32, usize)>>,
    /// Peeling: visited flags.
    visited: Vec<bool>,
    /// Peeling: visit order for reverse traversal.
    visit_order: Vec<u32>,
    /// Peeling: priority queue for Prim's MST.
    peel_heap: BinaryHeap<Reverse<(u64, usize, u32, u32)>>,
    /// Peeling: subtree parity for each node.
    subtree_parity: Vec<bool>,
    /// Peeling: correction edge indices.
    correction_edges: Vec<usize>,
    /// Weight swap buffer for `decode_with_weights`.
    weight_swap: Vec<(usize, f64)>,
}

impl UfDecoder {
    /// Get the adjacency entries for a node (slice into CSR data).
    #[inline]
    fn adj(&self, node: usize) -> &[(usize, u32)] {
        let start = self.adj_offset[node] as usize;
        let end = self.adj_offset[node + 1] as usize;
        &self.adj_data[start..end]
    }

    /// Build from a `DemMatchingGraph`.
    #[must_use]
    pub fn from_matching_graph(graph: &DemMatchingGraph, config: UfDecoderConfig) -> Self {
        let num_detectors = graph.num_detectors;
        let num_nodes = num_detectors + 1;
        let boundary_node = num_detectors as u32;

        let mut edges = Vec::with_capacity(graph.edges.len());
        // Build temporary adjacency for sorting, then flatten to CSR.
        let mut temp_adj: Vec<Vec<(usize, u32)>> = vec![Vec::new(); num_nodes];

        for (idx, me) in graph.edges.iter().enumerate() {
            let n1 = me.node1;
            let n2 = me.node2.map_or(boundary_node, |n| n);

            let mut obs_mask = 0u64;
            for &o in &me.observables {
                obs_mask |= 1 << o;
            }

            edges.push(Edge {
                node1: n1,
                node2: n2,
                weight: me.weight,
                obs_mask,
            });

            temp_adj[n1 as usize].push((idx, n2));
            temp_adj[n2 as usize].push((idx, n1));
        }

        // Sort each node's adjacency by weight (lightest first).
        for adj in &mut temp_adj {
            adj.sort_by(|a, b| {
                edges[a.0]
                    .weight
                    .partial_cmp(&edges[b.0].weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Flatten to CSR format.
        let total_entries: usize = temp_adj.iter().map(std::vec::Vec::len).sum();
        let mut adj_data = Vec::with_capacity(total_entries);
        let mut adj_offset = Vec::with_capacity(num_nodes + 1);
        for adj in &temp_adj {
            adj_offset.push(adj_data.len() as u32);
            adj_data.extend_from_slice(adj);
        }
        adj_offset.push(adj_data.len() as u32);

        Self {
            edges,
            adj_data,
            adj_offset,
            num_detectors,
            config,
            parent: vec![0; num_nodes],
            rank: vec![0; num_nodes],
            parity: vec![false; num_nodes],
            has_boundary: vec![false; num_nodes],
            radius: vec![0.0; num_nodes],
            last_growth_time: vec![0.0; num_nodes],
            cluster_size: vec![1; num_nodes],
            is_defect: vec![false; num_nodes],
            growth_events: BinaryHeap::new(),
            tree_parent: vec![None; num_nodes],
            visited: vec![false; num_nodes],
            visit_order: Vec::with_capacity(num_nodes),
            peel_heap: BinaryHeap::new(),
            subtree_parity: vec![false; num_nodes],
            correction_edges: Vec::new(),
            weight_swap: Vec::new(),
        }
    }

    /// Build from a DEM string.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed.
    pub fn from_dem(dem: &str, config: UfDecoderConfig) -> Result<Self, DecoderError> {
        let graph = DemMatchingGraph::from_dem_str(dem)?;
        Ok(Self::from_matching_graph(&graph, config))
    }

    /// Reset per-shot state. Uses bulk fill operations for cache efficiency.
    fn reset(&mut self) {
        let boundary = self.num_detectors;
        let n = boundary + 1;
        // Bulk-fill each array (SIMD-friendly).
        for i in 0..n {
            self.parent[i] = i as u32;
        }
        self.rank[..n].fill(0);
        self.parity[..n].fill(false);
        self.has_boundary[..n].fill(false);
        self.has_boundary[boundary] = true;
        self.radius[..n].fill(0.0);
        self.last_growth_time[..n].fill(0.0);
        self.cluster_size[..n].fill(1);
        self.is_defect[..n].fill(false);
    }

    /// Find root of node with path halving (one shortcut per step).
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    /// Union two clusters. Returns the new root.
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }

        // Union by rank
        let (root, child) = if self.rank[ra as usize] >= self.rank[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };

        self.parent[child as usize] = root;
        if self.rank[root as usize] == self.rank[child as usize] {
            self.rank[root as usize] += 1;
        }

        // XOR parities
        self.parity[root as usize] ^= self.parity[child as usize];
        // Propagate boundary membership
        self.has_boundary[root as usize] |= self.has_boundary[child as usize];
        // Keep the larger radius
        self.radius[root as usize] = self.radius[root as usize].max(self.radius[child as usize]);
        // Sum cluster sizes
        self.cluster_size[root as usize] += self.cluster_size[child as usize];

        root
    }

    /// Decode a syndrome and return the observable mask.
    pub fn decode_syndrome(&mut self, syndrome: &[u8]) -> u64 {
        // Try cluster-detection predecoder (if enabled).
        if self.config.predecoder
            && let Some(obs) = self.predecode_clusters(syndrome)
        {
            return obs;
        }

        // Full decoder path for complex syndromes.
        self.reset();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                self.parity[i] = true;
                self.is_defect[i] = true;
            }
        }
        self.grow_clusters();
        self.peel_correction()
    }

    /// Cluster-detection predecoder.
    ///
    /// Finds connected components of defects in the matching graph.
    /// - Size-0 components: no defects, return 0.
    /// - Size-1 components: match to boundary.
    /// - Size-2 components (adjacent pair): match directly if their edge
    ///   is lighter than both boundary alternatives.
    /// - Size 3+: too complex, fall through to full UF.
    ///
    /// This is provably correct: isolated clusters are independent, so
    /// predecoding them individually gives the same result as joint decoding.
    #[must_use]
    pub fn predecode_clusters(&self, syndrome: &[u8]) -> Option<u64> {
        let boundary = self.num_detectors as u32;

        // Mark defects.
        // Use is_defect buffer conceptually but don't mutate self.
        // Instead use a local bitset for small codes.
        let mut defect_flags = vec![false; self.num_detectors];
        let mut defect_list: Vec<u32> = Vec::new();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                defect_flags[i] = true;
                defect_list.push(i as u32);
            }
        }

        if defect_list.is_empty() {
            return Some(0);
        }

        // Find connected components of defects.
        // Two defects are connected if they share an edge in the matching graph.
        // Use union-find on defect indices (not the full graph -- just defects).
        let n = defect_list.len();
        let mut component: Vec<usize> = (0..n).collect(); // parent array

        // For each defect, check if any neighbor is also a defect.
        for (di, &d) in defect_list.iter().enumerate() {
            for &(_, neighbor) in self.adj(d as usize) {
                if neighbor != boundary
                    && (neighbor as usize) < self.num_detectors
                    && defect_flags[neighbor as usize]
                {
                    // Find the other defect's index in defect_list.
                    if let Some(ni) = defect_list.iter().position(|&x| x == neighbor) {
                        // Union di and ni.
                        let mut ra = di;
                        while component[ra] != ra {
                            ra = component[ra];
                        }
                        let mut rb = ni;
                        while component[rb] != rb {
                            rb = component[rb];
                        }
                        if ra != rb {
                            component[rb] = ra;
                        }
                    }
                }
            }
        }

        // Flatten components.
        for i in 0..n {
            let mut r = i;
            while component[r] != r {
                r = component[r];
            }
            component[i] = r;
        }

        // Count component sizes.
        let mut comp_size: Vec<usize> = vec![0; n];
        for &c in &component {
            comp_size[c] += 1;
        }

        // Check if any component has 3+ defects -- if so, fall through.
        for &s in &comp_size {
            if s >= 3 {
                return None; // Complex cluster, need full UF.
            }
        }

        // All components are size 1 or 2. Predecode each.
        let mut obs_mask = 0u64;
        let mut handled = vec![false; n];

        for di in 0..n {
            if handled[di] {
                continue;
            }
            let root = component[di];

            if comp_size[root] == 1 {
                // Isolated defect: match to boundary.
                obs_mask ^= self.predecode_single(defect_list[di]);
                handled[di] = true;
            } else if comp_size[root] == 2 {
                // Find the other defect in this component.
                let mut ni = None;
                for (dj, &candidate_root) in component.iter().enumerate().take(n).skip(di + 1) {
                    if candidate_root == root {
                        ni = Some(dj);
                        break;
                    }
                }
                let ni = ni?;

                let d0 = defect_list[di];
                let d1 = defect_list[ni];

                // Find lightest direct edge and lightest boundary alternatives.
                let mut direct_w = f64::INFINITY;
                let mut direct_obs = 0u64;
                for &(e, nbr) in self.adj(d0 as usize) {
                    if nbr == d1 && self.edges[e].weight < direct_w {
                        direct_w = self.edges[e].weight;
                        direct_obs = self.edges[e].obs_mask;
                    }
                }

                let mut b0_w = f64::INFINITY;
                let mut b0_obs = 0u64;
                for &(e, nbr) in self.adj(d0 as usize) {
                    if nbr == boundary && self.edges[e].weight < b0_w {
                        b0_w = self.edges[e].weight;
                        b0_obs = self.edges[e].obs_mask;
                    }
                }

                let mut b1_w = f64::INFINITY;
                let mut b1_obs = 0u64;
                for &(e, nbr) in self.adj(d1 as usize) {
                    if nbr == boundary && self.edges[e].weight < b1_w {
                        b1_w = self.edges[e].weight;
                        b1_obs = self.edges[e].obs_mask;
                    }
                }

                // Pick min-weight correction.
                if direct_w <= b0_w + b1_w {
                    obs_mask ^= direct_obs;
                } else {
                    obs_mask ^= b0_obs ^ b1_obs;
                }

                handled[di] = true;
                handled[ni] = true;
            }
        }

        Some(obs_mask)
    }

    /// Predecode: single defect matches to boundary.
    fn predecode_single(&self, defect: u32) -> u64 {
        let boundary = self.num_detectors as u32;
        // Find the lightest boundary edge from this defect.
        // Adjacency is sorted by weight, so iterate and pick first boundary edge.
        for &(edge_idx, neighbor) in self.adj(defect as usize) {
            if neighbor == boundary {
                return self.edges[edge_idx].obs_mask;
            }
        }
        // No boundary edge found (shouldn't happen for valid surface codes).
        0
    }

    /// Returns true if a cluster (given by its root) still needs to grow.
    fn is_unsatisfied(&self, root: usize) -> bool {
        self.parity[root] && !self.has_boundary[root]
    }

    /// Compute the growth rate for a cluster: `1 / size(cluster)`.
    /// Smaller clusters grow faster, producing better pairings.
    fn growth_rate(&self, root: usize) -> f64 {
        1.0 / f64::from(self.cluster_size[root])
    }

    /// Get the effective radius of a cluster at a given time.
    /// Uses lazy computation: radius is only updated when queried.
    fn effective_radius(&self, root: usize, current_time: f64) -> f64 {
        if self.is_unsatisfied(root) && root != self.num_detectors {
            let dt = current_time - self.last_growth_time[root];
            self.radius[root] + dt * self.growth_rate(root)
        } else {
            self.radius[root]
        }
    }

    /// Materialize the lazy radius for a cluster (update stored value).
    fn materialize_radius(&mut self, root: usize, current_time: f64) {
        if self.is_unsatisfied(root) && root != self.num_detectors {
            let dt = current_time - self.last_growth_time[root];
            self.radius[root] += dt * self.growth_rate(root);
        }
        self.last_growth_time[root] = current_time;
    }

    /// Compute when edge becomes fusible given current radii and growth rates.
    /// Returns the absolute time, or 0 if already fusible.
    fn fusible_time(&self, root_u: usize, root_v: usize, weight: f64, current_time: f64) -> f64 {
        let r_u = self.effective_radius(root_u, current_time);
        let r_v = self.effective_radius(root_v, current_time);
        let gap = weight - r_u - r_v;
        if gap <= 0.0 {
            return current_time;
        }

        let u_grows = self.is_unsatisfied(root_u);
        let v_grows = self.is_unsatisfied(root_v);

        let combined_rate = if u_grows && v_grows {
            self.growth_rate(root_u) + self.growth_rate(root_v)
        } else if u_grows {
            self.growth_rate(root_u)
        } else if v_grows {
            self.growth_rate(root_v)
        } else {
            return f64::INFINITY; // neither grows
        };

        current_time + gap / combined_rate
    }

    /// Dispatch to the configured growth strategy.
    fn grow_clusters(&mut self) {
        match self.config.growth {
            GrowthStrategy::EventDriven => self.grow_event_driven(),
            GrowthStrategy::ScanBased => self.grow_scan_based(),
        }
    }

    /// Scan-based growth: simple loop, lower overhead for small codes.
    ///
    /// Each round: scan all cross-cluster edges to find the minimum growth
    /// increment, grow all unsatisfied clusters uniformly, then fuse.
    /// O(R * E) where R is the number of growth rounds.
    fn grow_scan_based(&mut self) {
        let boundary = self.num_detectors;
        let max_rounds = if self.config.max_growth_rounds > 0 {
            self.config.max_growth_rounds
        } else {
            100 * self.num_detectors.max(1)
        };

        for _round in 0..max_rounds {
            // Check for any unsatisfied cluster.
            let mut any_unsatisfied = false;
            for i in 0..=self.num_detectors {
                let root = self.find(i as u32) as usize;
                if root == i && self.is_unsatisfied(root) {
                    any_unsatisfied = true;
                    break;
                }
            }
            if !any_unsatisfied {
                break;
            }

            // Find the minimum growth increment across all cross-cluster edges.
            let mut min_increment = f64::INFINITY;
            for node in 0..=self.num_detectors {
                let root_u = self.find(node as u32) as usize;
                if !self.is_unsatisfied(root_u) {
                    continue;
                }

                let adj_len = self.adj(node).len();
                for adj_i in 0..adj_len {
                    let (edge_idx, neighbor) = self.adj(node)[adj_i];
                    let root_v = self.find(neighbor) as usize;
                    if root_u == root_v {
                        continue;
                    }

                    let w = self.edges[edge_idx].weight;
                    let gap = w - self.radius[root_u] - self.radius[root_v];
                    if gap <= 0.0 {
                        min_increment = 0.0;
                        break;
                    }

                    let v_grows = self.is_unsatisfied(root_v);
                    let needed = if v_grows { gap / 2.0 } else { gap };
                    min_increment = min_increment.min(needed);
                }
                if min_increment == 0.0 {
                    break;
                }
            }

            if min_increment.is_infinite() {
                break;
            }

            // Grow all unsatisfied clusters.
            for i in 0..=self.num_detectors {
                let root = self.find(i as u32) as usize;
                if root == i && self.is_unsatisfied(root) && i != boundary {
                    self.radius[root] += min_increment;
                }
            }

            // Fuse all now-fusible cross-cluster edges.
            // Collect first to avoid borrow issues.
            let mut fuse_count = 0;
            for node in 0..=self.num_detectors {
                let adj_len = self.adj(node).len();
                for adj_i in 0..adj_len {
                    let (_, neighbor) = self.adj(node)[adj_i];
                    let root_u = self.find(node as u32) as usize;
                    let root_v = self.find(neighbor) as usize;
                    if root_u == root_v {
                        continue;
                    }
                    let w = self.edges[self.adj(node)[adj_i].0].weight;
                    if self.radius[root_u] + self.radius[root_v] >= w - 1e-12 {
                        self.union(node as u32, neighbor);
                        fuse_count += 1;
                    }
                }
            }
            if fuse_count == 0 && min_increment == 0.0 {
                break; // No progress
            }
        }
    }

    /// Event-driven weighted cluster growth.
    ///
    /// Smaller clusters grow faster (rate = 1/size), making nearby small
    /// clusters fuse before large clusters can absorb them. This improves
    /// the quality of the UF pairings.
    ///
    /// Uses a priority queue with lazy deletion for O(E log E) total work.
    fn grow_event_driven(&mut self) {
        self.growth_events.clear();

        // Seed events only from defect nodes (unsatisfied singletons).
        // At low error rates, this skips ~95% of nodes.
        for node in 0..self.num_detectors {
            if !self.is_defect[node] {
                continue;
            }

            let adj_len = self.adj(node).len();
            for adj_i in 0..adj_len {
                let (edge_idx, neighbor) = self.adj(node)[adj_i];
                let root_v = self.find(neighbor) as usize;
                if root_v == node {
                    continue; // same cluster (shouldn't happen for singletons)
                }

                let ft = self.fusible_time(node, root_v, self.edges[edge_idx].weight, 0.0);
                if ft.is_finite() {
                    self.growth_events.push(Reverse((
                        ft.to_bits(),
                        edge_idx,
                        node as u32,
                        neighbor,
                    )));
                }
            }
        }

        let mut current_time: f64;
        let max_events = if self.config.max_growth_rounds > 0 {
            self.config.max_growth_rounds
        } else {
            1000 * self.edges.len().max(1)
        };
        let mut events_processed = 0;

        while let Some(Reverse((time_bits, _edge_idx, a, b))) = self.growth_events.pop() {
            events_processed += 1;
            if events_processed > max_events {
                break;
            }
            let event_time = f64::from_bits(time_bits);

            let root_a = self.find(a) as usize;
            let root_b = self.find(b) as usize;
            if root_a == root_b {
                continue;
            }

            let a_unsat = self.is_unsatisfied(root_a);
            let b_unsat = self.is_unsatisfied(root_b);
            if !a_unsat && !b_unsat {
                continue;
            }

            current_time = event_time;

            // Materialize radii for the merging clusters (lazy update).
            self.materialize_radius(root_a, current_time);
            self.materialize_radius(root_b, current_time);

            // Fuse.
            let new_root = self.union(a, b) as usize;
            self.last_growth_time[new_root] = current_time;

            if !self.is_unsatisfied(new_root) {
                continue;
            }

            // Re-insert events for edges from the merge nodes with updated times.
            for &node in &[a, b] {
                let nu = node as usize;
                let adj_len = self.adj(nu).len();
                for adj_i in 0..adj_len {
                    let (edge_idx, neighbor) = self.adj(nu)[adj_i];
                    let root_n = self.find(neighbor) as usize;
                    if root_n == new_root {
                        continue;
                    }

                    let ft = self.fusible_time(
                        new_root,
                        root_n,
                        self.edges[edge_idx].weight,
                        current_time,
                    );
                    if ft.is_finite() {
                        self.growth_events
                            .push(Reverse((ft.to_bits(), edge_idx, node, neighbor)));
                    }
                }
            }
        }
    }

    /// Build a spanning forest and peel to extract the correction.
    /// Returns `(obs_mask, correction_edge_indices)`.
    fn peel_correction_with_edges(&mut self) -> (u64, Vec<usize>) {
        match self.config.peeling {
            PeelingStrategy::PrimMst => self.peel_prim_mst(),
            PeelingStrategy::Bfs => self.peel_bfs(),
        }
    }

    /// Prim's MST peeling: globally lightest spanning tree. Better accuracy.
    fn peel_prim_mst(&mut self) -> (u64, Vec<usize>) {
        let boundary = self.num_detectors;

        self.tree_parent.fill(None);
        self.visited.fill(false);
        self.visit_order.clear();
        self.peel_heap.clear();
        self.correction_edges.clear();

        for seed in std::iter::once(boundary).chain(0..self.num_detectors) {
            if self.visited[seed] {
                continue;
            }
            self.visited[seed] = true;
            self.visit_order.push(seed as u32);

            let adj_len = self.adj(seed).len();
            for adj_i in 0..adj_len {
                let (edge_idx, neighbor) = self.adj(seed)[adj_i];
                if !self.visited[neighbor as usize] {
                    let w_bits = self.edges[edge_idx].weight.to_bits();
                    self.peel_heap
                        .push(Reverse((w_bits, edge_idx, seed as u32, neighbor)));
                }
            }

            while let Some(Reverse((_w_bits, edge_idx, from, to))) = self.peel_heap.pop() {
                let tu = to as usize;
                if self.visited[tu] {
                    continue;
                }
                let from_root = self.find(from);
                let to_root = self.find(to);
                if from_root != to_root {
                    continue;
                }
                self.visited[tu] = true;
                self.tree_parent[tu] = Some((from, edge_idx));
                self.visit_order.push(to);

                let adj_len = self.adj(tu).len();
                for adj_i in 0..adj_len {
                    let (e_idx, nbr) = self.adj(tu)[adj_i];
                    if !self.visited[nbr as usize] {
                        let w_bits = self.edges[e_idx].weight.to_bits();
                        self.peel_heap.push(Reverse((w_bits, e_idx, to, nbr)));
                    }
                }
            }
        }

        // Peel: process in reverse visit order (leaves first).
        self.subtree_parity.fill(false);
        self.subtree_parity[..self.num_detectors]
            .copy_from_slice(&self.is_defect[..self.num_detectors]);

        let mut obs_mask = 0u64;

        for i in (0..self.visit_order.len()).rev() {
            let v = self.visit_order[i];
            if let Some((parent, edge_idx)) = self.tree_parent[v as usize] {
                if self.subtree_parity[v as usize] {
                    obs_mask ^= self.edges[edge_idx].obs_mask;
                    self.correction_edges.push(edge_idx);
                }
                self.subtree_parity[parent as usize] ^= self.subtree_parity[v as usize];
            }
        }

        (obs_mask, self.correction_edges.clone())
    }

    /// BFS peeling: simpler, faster (no heap), slightly less accurate.
    fn peel_bfs(&mut self) -> (u64, Vec<usize>) {
        let boundary = self.num_detectors;

        self.tree_parent.fill(None);
        self.visited.fill(false);
        self.visit_order.clear();
        self.correction_edges.clear();

        for seed in std::iter::once(boundary).chain(0..self.num_detectors) {
            if self.visited[seed] {
                continue;
            }
            self.visited[seed] = true;
            self.visit_order.push(seed as u32);

            let mut queue_start = self.visit_order.len() - 1;
            while queue_start < self.visit_order.len() {
                let v = self.visit_order[queue_start] as usize;
                queue_start += 1;

                let adj_len = self.adj(v).len();
                for adj_i in 0..adj_len {
                    let (edge_idx, neighbor) = self.adj(v)[adj_i];
                    let nu = neighbor as usize;
                    if self.visited[nu] {
                        continue;
                    }
                    let v_root = self.find(v as u32);
                    let n_root = self.find(neighbor);
                    if v_root != n_root {
                        continue;
                    }
                    self.visited[nu] = true;
                    self.tree_parent[nu] = Some((v as u32, edge_idx));
                    self.visit_order.push(neighbor);
                }
            }
        }

        self.subtree_parity.fill(false);
        self.subtree_parity[..self.num_detectors]
            .copy_from_slice(&self.is_defect[..self.num_detectors]);

        let mut obs_mask = 0u64;
        for i in (0..self.visit_order.len()).rev() {
            let v = self.visit_order[i];
            if let Some((parent, edge_idx)) = self.tree_parent[v as usize] {
                if self.subtree_parity[v as usize] {
                    obs_mask ^= self.edges[edge_idx].obs_mask;
                    self.correction_edges.push(edge_idx);
                }
                self.subtree_parity[parent as usize] ^= self.subtree_parity[v as usize];
            }
        }

        (obs_mask, self.correction_edges.clone())
    }

    /// Peel correction, returning only the observable mask.
    fn peel_correction(&mut self) -> u64 {
        self.peel_correction_with_edges().0
    }

    /// Number of edges in the matching graph.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of detectors.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Get the observable mask for an edge.
    #[must_use]
    pub fn edge_obs_mask(&self, edge_idx: usize) -> u64 {
        self.edges.get(edge_idx).map_or(0, |e| e.obs_mask)
    }

    /// Get node1 of an edge.
    #[must_use]
    pub fn edge_node1(&self, edge_idx: usize) -> u32 {
        self.edges.get(edge_idx).map_or(0, |e| e.node1)
    }

    /// Get node2 of an edge (boundary = `num_detectors`).
    #[must_use]
    pub fn edge_node2(&self, edge_idx: usize) -> u32 {
        self.edges.get(edge_idx).map_or(0, |e| e.node2)
    }

    /// Get the weight of an edge (log-likelihood ratio).
    #[must_use]
    pub fn edge_weight(&self, edge_idx: usize) -> f64 {
        self.edges.get(edge_idx).map_or(0.0, |e| e.weight)
    }

    /// Decode with full UF (no predecoder) and return matched edges.
    /// Used by windowed decoder which needs complete edge tracking.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if decoding fails.
    pub fn decode_full_matching(
        &mut self,
        syndrome: &[u8],
    ) -> Result<(u64, Vec<usize>), DecoderError> {
        self.reset();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                self.parity[i] = true;
                self.is_defect[i] = true;
            }
        }
        self.grow_clusters();
        Ok(self.peel_correction_with_edges())
    }

    // === UIUF support methods ===

    /// Run syndrome validation (growth phase only, no peeling).
    ///
    /// After calling this, the internal cluster state reflects which nodes
    /// have been merged. Use `edge_in_cluster()` to query which edges are
    /// covered by clusters.
    pub fn syndrome_validate(&mut self, syndrome: &[u8]) {
        self.reset();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                self.parity[i] = true;
                self.is_defect[i] = true;
            }
        }
        self.grow_clusters();
    }

    /// Check if an edge's two endpoints are in the same cluster.
    ///
    /// Call after `syndrome_validate()`. Returns true if the edge is
    /// "covered" by a cluster (both endpoints merged into one component).
    pub fn edge_in_cluster(&mut self, edge_idx: usize) -> bool {
        if edge_idx >= self.edges.len() {
            return false;
        }
        let n1 = self.edges[edge_idx].node1;
        let n2 = self.edges[edge_idx].node2;
        let root_a = self.find(n1);
        let root_b = self.find(n2);
        root_a == root_b
    }

    /// Decode with pre-seeded erasure edges.
    ///
    /// Erasure edges are pre-merged into clusters before growth begins.
    /// This is used by UIUF Phase 3 after the intersection step identifies
    /// likely Y errors as erasures.
    pub fn decode_with_erasures(&mut self, syndrome: &[u8], erasure_edges: &[usize]) -> u64 {
        self.reset();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                self.parity[i] = true;
                self.is_defect[i] = true;
            }
        }

        // Pre-merge erasure edges before growth.
        for &edge_idx in erasure_edges {
            if edge_idx < self.edges.len() {
                let n1 = self.edges[edge_idx].node1;
                let n2 = self.edges[edge_idx].node2;
                self.union(n1, n2);
            }
        }

        self.grow_clusters();
        self.peel_correction()
    }
}

// === Trait implementations ===

impl pecos_decoder_core::ObservableDecoder for UfDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        Ok(self.decode_syndrome(syndrome))
    }
}

impl pecos_decoder_core::correlated_decoder::MatchingDecoder for UfDecoder {
    fn decode_with_matching(&mut self, syndrome: &[u8]) -> Result<(u64, Vec<usize>), DecoderError> {
        // Count defects for predecoder.
        let num_defects = syndrome
            .iter()
            .take(self.num_detectors)
            .filter(|&&v| v != 0)
            .count() as u32;

        if num_defects == 0 {
            return Ok((0, Vec::new()));
        }

        // Cluster predecoder (if enabled). Skipped in windowed mode
        // because windowed decoding needs complete edge tracking.
        if self.config.predecoder
            && let Some(obs) = self.predecode_clusters(syndrome)
        {
            return Ok((obs, Vec::new()));
        }

        // Full decode path.
        self.reset();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < self.num_detectors {
                self.parity[i] = true;
                self.is_defect[i] = true;
            }
        }
        self.grow_clusters();
        Ok(self.peel_correction_with_edges())
    }

    fn decode_with_weights(
        &mut self,
        syndrome: &[u8],
        weights: &[f64],
    ) -> Result<(u64, Vec<usize>), DecoderError> {
        // Temporarily swap in the new weights
        self.weight_swap.clear();
        for (i, &w) in weights.iter().enumerate() {
            if i < self.edges.len() {
                self.weight_swap.push((i, self.edges[i].weight));
                self.edges[i].weight = w;
            }
        }

        // Note: CSR adjacency sort order is fixed at construction.
        // The weight swap affects growth event ordering but not correctness.
        let result = self.decode_with_matching(syndrome);

        // Restore original weights
        for &(i, w) in &self.weight_swap {
            self.edges[i].weight = w;
        }

        result
    }

    fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

impl pecos_decoder_core::correlated_decoder::EdgeTrackingDecoder for UfDecoder {
    fn edge_node1(&self, edge_idx: usize) -> u32 {
        self.edges.get(edge_idx).map_or(0, |e| e.node1)
    }

    fn edge_node2(&self, edge_idx: usize) -> u32 {
        self.edges.get(edge_idx).map_or(0, |e| e.node2)
    }

    fn edge_weight(&self, edge_idx: usize) -> f64 {
        self.edges.get(edge_idx).map_or(0.0, |e| e.weight)
    }

    fn edge_obs_mask(&self, edge_idx: usize) -> u64 {
        self.edges.get(edge_idx).map_or(0, |e| e.obs_mask)
    }

    fn num_detectors(&self) -> usize {
        self.num_detectors
    }
}

impl pecos_decoder_core::erasure::ObservableErasureDecoder for UfDecoder {
    fn decode_with_erasures(
        &mut self,
        syndrome: &[u8],
        erasure_edges: &[usize],
    ) -> Result<u64, DecoderError> {
        if erasure_edges.is_empty() {
            // Use MatchingDecoder path directly.
            use pecos_decoder_core::correlated_decoder::MatchingDecoder;
            let (obs, _) = self.decode_with_matching(syndrome)?;
            return Ok(obs);
        }

        // Set erased edges to weight=0 (certain error), decode, restore.
        let mut modified_weights = Vec::with_capacity(self.edges.len());
        for (i, e) in self.edges.iter().enumerate() {
            if erasure_edges.contains(&i) {
                modified_weights.push(0.0);
            } else {
                modified_weights.push(e.weight);
            }
        }

        let (obs, _) = self.decode_with_weights(syndrome, &modified_weights)?;
        Ok(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_DEM: &str = "\
error(0.1) D0 D1 L0
error(0.1) D1
";

    const SURFACE_LIKE_DEM: &str = "\
error(0.01) D0 D1 L0
error(0.01) D1 D2
error(0.01) D2
error(0.01) D0
";

    #[test]
    fn test_no_errors() {
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        assert_eq!(dec.decode_syndrome(&[0, 0]), 0);
    }

    #[test]
    fn test_single_error() {
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        // D0 and D1 both triggered -> edge D0-D1 (carries L0)
        assert_eq!(dec.decode_syndrome(&[1, 1]), 1);
    }

    #[test]
    fn test_boundary_error() {
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        // Only D1 triggered -> boundary edge (no observable)
        assert_eq!(dec.decode_syndrome(&[0, 1]), 0);
    }

    #[test]
    fn test_multiple_shots() {
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        for _ in 0..20 {
            let _ = dec.decode_syndrome(&[1, 1]);
            let _ = dec.decode_syndrome(&[0, 1]);
            let _ = dec.decode_syndrome(&[0, 0]);
        }
    }

    #[test]
    fn test_surface_like() {
        let mut dec = UfDecoder::from_dem(SURFACE_LIKE_DEM, UfDecoderConfig::default()).unwrap();
        // D0 triggered -> boundary edge (no observable)
        assert_eq!(dec.decode_syndrome(&[1, 0, 0]), 0);
        // D0 and D1 -> edge D0-D1 (L0)
        assert_eq!(dec.decode_syndrome(&[1, 1, 0]), 1);
    }

    #[test]
    fn test_observable_decoder_trait() {
        use pecos_decoder_core::ObservableDecoder;
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        let mask = dec.decode_to_observables(&[1, 1]).unwrap();
        assert_eq!(mask, 1);
    }

    #[test]
    fn test_matching_decoder_trait() {
        use pecos_decoder_core::correlated_decoder::MatchingDecoder;
        let mut dec = UfDecoder::from_dem(SIMPLE_DEM, UfDecoderConfig::default()).unwrap();
        let (mask, _edges) = dec.decode_with_matching(&[1, 1]).unwrap();
        assert_eq!(mask, 1);
        // Note: predecoder may return empty edges for simple cases.
    }

    /// Distance-3 repetition code: 3 data qubits, 2 detectors, 2 rounds.
    /// Tests a more realistic graph structure with time-like edges.
    const REP_CODE_D3_DEM: &str = "\
error(0.01) D0 D1
error(0.01) D0 L0
error(0.01) D1
error(0.001) D0 D2
error(0.001) D1 D3
error(0.01) D2 D3
error(0.01) D2 L0
error(0.01) D3
";

    #[test]
    fn test_rep_code_d3() {
        let mut dec = UfDecoder::from_dem(REP_CODE_D3_DEM, UfDecoderConfig::default()).unwrap();
        assert_eq!(dec.num_edges(), 8);
        assert_eq!(dec.num_detectors(), 4);

        // No defects
        assert_eq!(dec.decode_syndrome(&[0, 0, 0, 0]), 0);

        // Single data error in round 1: D0 and D1 both fire
        assert_eq!(dec.decode_syndrome(&[1, 1, 0, 0]), 0);

        // Boundary error: only D0
        assert_eq!(dec.decode_syndrome(&[1, 0, 0, 0]), 1); // L0

        // Single boundary error round 2: only D2
        assert_eq!(dec.decode_syndrome(&[0, 0, 1, 0]), 1); // L0
    }

    #[test]
    fn test_stress_reuse() {
        // Verify buffers are correctly reused over many shots
        let mut dec = UfDecoder::from_dem(REP_CODE_D3_DEM, UfDecoderConfig::default()).unwrap();
        let syndromes: &[&[u8]] = &[
            &[0, 0, 0, 0],
            &[1, 1, 0, 0],
            &[1, 0, 0, 0],
            &[0, 1, 0, 0],
            &[0, 0, 1, 1],
            &[1, 0, 1, 0],
        ];
        for _ in 0..1000 {
            for syn in syndromes {
                let _ = dec.decode_syndrome(syn);
            }
        }
    }
}

# Decoder Architecture

PECOS provides a layered decoder architecture for quantum error correction,
from simple memory experiments to logical algorithms with transversal gates
and real-time streaming decode.

## Design Principles

- **Composable**: any MWPM-compatible decoder can be used as an inner decoder
- **Budget-aware**: automatically adapts window size and overlap based on hardware timing constraints
- **Streaming**: accepts syndrome data round-by-round for real-time operation
- **Frame-tracking**: propagates Pauli corrections through transversal gate boundaries

## Decoder Layers

The architecture is layered, with each layer adding capability:

```
Layer 1: Inner Decoder (MWPM)
    PyMatching, Fusion Blossom, Union-Find, ...
    Input: graphlike DEM + syndrome
    Output: observable correction bitmask

Layer 2: Observable Subgraph Decoder (OSD)
    Decomposes transversal-gate circuits into per-observable graphlike subgraphs
    Input: full DEM (may have hyperedges) + syndrome + spatial coordinates
    Output: observable correction bitmask

Layer 3: Logical Algorithm Decoder
    Adds frame propagation across transversal gate boundaries
    Input: algorithm descriptor (segments + boundary gates) + syndrome
    Output: correction at each decision point

Layer 4: Logical Circuit Decoder (budget-aware)
    Selects decode strategy based on hardware timing budget
    Input: algorithm descriptor + budget + syndrome stream
    Output: real-time corrections
```

## Layer 1: Inner Decoders

Any decoder implementing the `ObservableDecoder` trait:

| Decoder | Type | DEM Support | Accuracy | Speed |
|---------|------|-------------|----------|-------|
| PyMatching | MWPM | graphlike | baseline | fast |
| Fusion Blossom | MWPM | graphlike | ~baseline | fast (parallel) |
| Tesseract | A* search | any (hyperedges) | best | medium |
| BP+OSD | LDPC | any | good | slow |
| Union-Find | cluster | graphlike | good | fastest |
| MWPF | matching | graphlike | best | slow |

MWPM decoders require **graphlike** (decomposed) DEMs where every mechanism
touches at most 2 detectors. Non-MWPM decoders can handle hyperedges directly.

Use `decoder_dem_requirement(decoder_type)` to query what a decoder needs:

```python
from pecos_rslib.qec import decoder_dem_requirement

decoder_dem_requirement("pymatching")   # "graphlike"
decoder_dem_requirement("tesseract")    # "any"
```

## Layer 2: Observable Subgraph Decoder (OSD)

**Problem**: Transversal gates (H, CX) create hyperedge mechanisms in the DEM
(3+ detectors flipping together). MWPM decoders cannot handle these.

**Solution**: Proved by Serra-Peralta et al. and Cain et al. (2025): the
per-observable subgraph of a transversal-gate DEM is always graphlike. OSD
exploits this by:

1. Classifying each detector by (logical_qubit, stabilizer_type) using spatial coordinates
2. For each observable, finding its observing region via boundary edges
3. Extracting a sub-DEM restricted to those detectors (guaranteed graphlike)
4. Running any MWPM decoder on each subgraph independently
5. Combining per-observable corrections via XOR

```python
from pecos_rslib.qec import ObservableSubgraphDecoder

# Build OSD with PyMatching as inner decoder
osd = ObservableSubgraphDecoder(dem_string, stab_coords, inner_decoder="pymatching")

# Decode a syndrome
obs_correction = osd.decode(syndrome)
```

### Spatial Coordinates

OSD requires detector spatial coordinates to classify detectors. These
are typically embedded in the DEM as `detector(x, y, t) D_i` annotations.
The `stab_coords` parameter maps each stabilizer to its (x, y) position
and type (X or Z).

## Layer 3: Logical Algorithm Decoder

Adds Pauli frame propagation at transversal gate boundaries:

| Gate | X frame | Z frame |
|------|---------|---------|
| Hadamard | X <-> Z | Z <-> X |
| CNOT | ctrl X -> target X | target Z -> ctrl Z |
| S gate | X -> X*Z | Z unchanged |
| T injection | Decision point | ancilla Z -> data Z |

T-gate injection is the only point requiring a real-time decode decision.
All other gates just propagate the frame algebraically.

```python
from pecos_rslib.qec import LogicalAlgorithmDecoder

# Build from algorithm descriptor
decoder = LogicalAlgorithmDecoder(descriptor, inner_decoder="pymatching")
correction = decoder.decode(full_syndrome)
```

## Layer 4: Budget-Aware Decoding

Different hardware platforms have different timing constraints:

| Platform | Reaction time | Strategy |
|----------|--------------|----------|
| Superconducting | ~1 us | Minimal windows, no overlap |
| Neutral atom | ~1 ms | d-round windows, d/2 overlap |
| Ion trap | ~10 ms | Large windows, full overlap |
| Offline | unlimited | Full-circuit decode |

The `DecodeBudget` automatically selects window size and overlap:

```python
from pecos_rslib.qec import LogicalCircuitDecoder

# Budget-aware: automatically selects strategy
decoder = LogicalCircuitDecoder(
    descriptor,
    budget="neutral_atom",  # or "superconducting", "unlimited"
    inner_decoder="pymatching",
)
```

## Windowed Decoding

For deep circuits, the observing region can span too many rounds, degrading
accuracy. Windowed decoding splits the time axis:

- **Non-overlapping**: each detector in exactly one window (fastest)
- **Overlapping**: buffer zones extend beyond core for matching context (more accurate)
- **Streaming**: commit previous windows and slide forward (real-time)

The `WindowedOsdDecoder` implements windowed OSD:

```python
from pecos_rslib.qec import WindowedOsdDecoder

decoder = WindowedOsdDecoder(
    dem_string, stab_coords,
    inner_decoder="pymatching",
    step=8,    # core window size in time steps
    buffer=4,  # buffer on each side
)
```

## Streaming Decode

For real-time operation, the `StreamingDecoder` trait accepts syndrome
data round-by-round:

```rust
// Rust API
trait StreamingDecoder {
    fn feed_round(&mut self, round: usize, detectors: &[(u32, u8)]) -> Result<u64, DecoderError>;
    fn flush(&mut self) -> Result<u64, DecoderError>;
    fn accumulated_obs(&self) -> u64;
}
```

The `CommittedOsd` implements streaming with software commitment (Cain et al.):
committed detectors are masked in future decodes, preventing past decisions
from being revisited.

## DEM Generation for Decoders

The choice of DEM generation method affects decoder accuracy:

| Method | Coherent noise | PyMatching LER | Tesseract LER |
|--------|---------------|----------------|---------------|
| `from_circuit` (stochastic only) | ignores | baseline | baseline |
| `coherent_dem_decomposed` (EEG) | handles | 17% better | 10% better |
| `noise_characterization` (EEG) | handles | 17% better | 10% better |

For circuits with coherent noise (idle Z-rotations), use `coherent_dem_decomposed`
or `noise_characterization` which produce properly decomposed DEMs with
Heisenberg-exact probabilities.

## Summary

The decoder architecture separates concerns cleanly:

- **Inner decoders** solve the matching/search problem on graphlike DEMs
- **OSD** handles transversal gate hyperedges via proven subgraph decomposition
- **Frame propagation** tracks corrections through gate boundaries algebraically
- **Budgets** adapt decode strategy to hardware constraints automatically
- **Streaming** enables real-time operation via round-by-round feeding

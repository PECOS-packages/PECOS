# Design: pecos-zx Interactive Editor

## Vision

A visual editor for ZX diagrams and QEC circuits, analogous to Blender for 3D modeling. The editor allows interactive construction, manipulation, and analysis of ZX graphs with real-time feedback on Pauli webs, simplification, and error correction properties.

## Motivation

ZX calculus and QEC circuit design are inherently spatial and visual, but current tooling is almost entirely text-driven. Key pain points:

- **No interactive exploration**: Pauli web overlays on static SVGs overlap and are hard to read. Toggling individual webs, hovering for details, and zooming into dense regions all require interactivity.
- **No visual editing**: Building circuits requires writing code, converting to ZX, rendering, inspecting, then modifying code and repeating. A drag-and-drop editor with live ZX preview would dramatically shorten the feedback loop.
- **No step-through**: ZX simplification is a sequence of graph rewrites. Stepping through them visually (like a debugger) would build intuition and catch bugs.

## Architecture

```
pecos-zx (library)          pecos-zx-editor (binary)
+-----------------------+    +---------------------------+
| ZxGraph, PauliWeb,    |    | Bevy app                  |
| simplify, DEM,        |<---| - 2D graph renderer       |
| convert, tableau,     |    | - UI panels (egui)        |
| viz (SVG/ASCII/TikZ)  |    | - Edit tools              |
+------|----------------+    | - Web overlay toggles     |
       |                     | - Simplification stepper  |
       v                     +---------------------------+
  Static output (SVG,
  TikZ, HTML viewer)
```

The existing `pecos-zx` crate remains the computation backend. The editor is a separate crate that depends on it. Static renderers (SVG, ASCII, TikZ, HTML) stay for non-interactive use cases.

## Phased Roadmap

### Phase 1: HTML Viewer (current)

Generate a self-contained HTML file with:
- All webs as toggleable SVG overlays (`<g>` groups with checkbox controls)
- Pauli labels on edges
- Legend synced with toggle state
- Pan and zoom via mouse/trackpad
- No new Rust dependencies -- just string generation like the SVG renderer

**Status**: Implementing now.

### Phase 2: Read-Only Bevy Viewer

Bevy app that loads and displays a ZX graph:
- Vertices as 2D sprites (circles, squares) with color scheme support
- Edges as line segments or bezier curves
- Camera: pan, zoom, fit-to-view
- Hover tooltips: vertex type, phase, ID; edge type, Pauli operator
- Web overlay toggle panel (checkboxes per web with classification labels)
- Reuse existing layout algorithms from `pecos-zx::viz::layout`

**Key decisions**:
- Use `bevy_prototype_lyon` or custom mesh generation for graph drawing
- Use `bevy_egui` for UI panels
- Graph loaded from a serialized `ZxGraph` or computed live from a `DagCircuit`

### Phase 3: Graph Editing

Interactive graph manipulation:
- **Vertex operations**: add, delete, drag to reposition, change type (Z/X/H/B), edit phase
- **Edge operations**: add, delete, change type (normal/Hadamard)
- **Spider fusion**: drag one spider onto another to fuse
- **Undo/redo**: command pattern stack
- **Selection**: click, shift-click, box select
- **Keyboard shortcuts**: delete, copy/paste subgraphs, type-change hotkeys

### Phase 4: QEC-Specific Tools

- **Circuit view**: side-by-side circuit and ZX representations with live sync
- **Simplification stepper**: step forward/back through rewrite rules with highlighting of the affected region
- **Noise overlay**: visualize `NoiseModel` error rates as edge thickness or color intensity
- **DEM inspector**: click a detector to highlight its web, click an error mechanism to see which detectors it triggers
- **Code builder**: templates for common QEC codes (surface, color, repetition) with parameterized construction

## Technology Choice: Why Bevy

| Requirement | Bevy | egui standalone | Web (Leptos/Yew) |
|---|---|---|---|
| 2D rendering performance | Excellent (wgpu) | Adequate | Adequate |
| 3D (future: topological codes) | Built-in | Not practical | Needs Three.js |
| Rich input handling | Built-in | Basic | DOM events |
| UI panels | bevy_egui plugin | Native | HTML/CSS |
| Plugin ecosystem | Large | Small | Large (JS) |
| WASM target | Supported | Supported | Native |
| Learning curve | Steep (ECS) | Low | Medium |

Bevy's ECS maps well to graph structures: vertices and edges are entities, types and phases are components, layout and rendering are systems. The steep learning curve is justified by the long-term feature set.

## Data Flow

```
DagCircuit --[dag_to_zx]--> ZxGraph --[compute_layout]--> positions
                                |                              |
                                v                              v
                     compute_pauli_webs              Bevy: spawn entities
                                |                    at positions
                                v
                         PauliWebResult --[classify]--> WebOverlay
                                                           |
                                                           v
                                                   Bevy: toggle overlays
```

## Open Questions

- **Serialization format**: Should the editor save/load ZX graphs in a custom format, or use an existing one (e.g., the `.qgraph` format from PyZX)?
- **Live circuit sync**: Should editing the ZX graph back-propagate changes to the circuit representation? This is hard in general (ZX simplification is lossy).
- **Collaboration**: Is multi-user editing a goal? (Probably not initially.)
- **3D timeline**: When does 3D visualization of topological codes become a priority?

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

//! Visualization of ZX diagrams as SVG, TikZ, and ASCII.

pub mod ascii;
pub mod circuit_ascii;
pub mod circuit_layout;
pub mod circuit_svg;
pub mod colors;
pub mod html;
pub mod layout;
pub mod render;
pub mod svg;
pub mod tikz;

pub use ascii::{AsciiOptions, render_ascii};
pub use circuit_ascii::{CircuitAsciiOptions, render_circuit_ascii};
pub use circuit_layout::{CircuitLayout, GateSlot, layout_from_dag, layout_from_tick_circuit};
pub use circuit_svg::{CircuitSvgOptions, render_circuit_svg};
pub use colors::ColorScheme;
pub use html::{render_html, render_html_with_rewrites};
pub use layout::{LayoutAlgorithm, compute_layout};
pub use render::Renderer;
pub use svg::{SvgOptions, WebOverlay, render_svg};
pub use tikz::{TikzOptions, render_tikz, standalone_document, tikz_preamble};

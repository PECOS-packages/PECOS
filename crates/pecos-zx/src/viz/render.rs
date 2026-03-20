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

//! Unified renderer for ZX diagrams.
//!
//! The [`Renderer`] struct holds options for ASCII, SVG, and TikZ output
//! and provides a single [`render()`](Renderer::render) method that produces
//! all three formats from one call.

use std::path::{Path, PathBuf};

use quizx::graph::GraphLike;

use super::ascii::{AsciiOptions, render_ascii};
use super::layout::LayoutAlgorithm;
use super::svg::{SvgOptions, render_svg};
use super::tikz::{TikzOptions, render_tikz, standalone_document};

/// Unified renderer that produces ASCII, SVG, and TikZ output for a ZX graph.
#[derive(Default)]
pub struct Renderer {
    /// Options for ASCII rendering.
    pub ascii: AsciiOptions,
    /// Options for SVG rendering.
    pub svg: SvgOptions,
    /// Options for TikZ rendering.
    pub tikz: TikzOptions,
    /// Directory for output files. If set, files are written under this
    /// directory (created automatically). Defaults to the current directory.
    pub output_dir: Option<PathBuf>,
}

impl Renderer {
    /// Set the layout algorithm across all three formats.
    pub fn set_layout(&mut self, layout: LayoutAlgorithm) {
        self.ascii.layout = layout;
        self.svg.layout = layout;
        self.tikz.layout = layout;
    }

    /// Return the path for an output file, creating the output directory if needed.
    fn output_path(&self, filename: &str) -> PathBuf {
        match &self.output_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir).expect("failed to create output directory");
                dir.join(filename)
            }
            None => PathBuf::from(filename),
        }
    }

    /// Convenience setter for `output_dir`.
    pub fn set_output_dir(&mut self, dir: impl AsRef<Path>) {
        self.output_dir = Some(dir.as_ref().to_path_buf());
    }

    /// Render the graph in all three formats.
    ///
    /// - Prints ASCII to stdout with a `--- {name} ---` header
    /// - Writes `{name}.svg`
    /// - Writes `{name}.tex` (standalone TikZ document)
    pub fn render(&self, graph: &impl GraphLike, name: &str) {
        println!("--- {name} ---\n");
        let ascii = render_ascii(graph, &self.ascii);
        println!("{ascii}\n");

        let svg = render_svg(graph, &self.svg);
        let svg_path = self.output_path(&format!("{name}.svg"));
        std::fs::write(&svg_path, &svg).expect("failed to write SVG");
        println!("  Wrote {}", svg_path.display());

        let tikz = render_tikz(graph, &self.tikz);
        let doc = standalone_document(&tikz, self.tikz.color_scheme);
        let tex_path = self.output_path(&format!("{name}.tex"));
        std::fs::write(&tex_path, &doc).expect("failed to write .tex");
        println!("  Wrote {}", tex_path.display());
    }
}

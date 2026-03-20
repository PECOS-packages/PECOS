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

//! Color infrastructure for ZX diagram rendering.
//!
//! Supports two color schemes:
//! - **PECOS** (default): RGB = XYZ convention (Red=X, Green=Y, Blue=Z)
//! - **ZX Canonical**: traditional ZX calculus convention (Green=Z, Red=X)
//!
//! ## PECOS color algebra
//!
//! Gate colors in the PECOS scheme are derived from additive RGB mixing based
//! on how each gate acts on Pauli axes (under conjugation). A gate that
//! interconverts axes P1 and P2 gets the additive mix of their colors, which
//! is the complement of the axis it leaves invariant:
//!
//! | Axis pair | Gates                    | RGB mix     | Color       | Complement of |
//! |-----------|--------------------------|-------------|-------------|---------------|
//! | X <-> Z   | H, SY, SY†             | Red + Blue  | Magenta     | Y (Green)     |
//! | X <-> Y   | SZ, SZ†                | Red + Green | Yellow      | Z (Blue)      |
//! | Y <-> Z   | SX, SX†                | Blue + Green| Cyan        | X (Red)       |
//! | X, Y, Z   | F (X->Y->Z->X), F†     | R + G + B   | Grey/White  | (none)        |
//!
//! ## Brightness axis (forward vs inverse)
//!
//! A second visual dimension encodes the rotation direction using brightness.
//! The "forward" direction follows the cyclic order **X -> Y -> Z -> X**
//! (which is exactly the F gate). The "inverse" direction goes against it
//! (the dagger).
//!
//! | Gate | Action     | Direction | Brightness |
//! |------|------------|-----------|------------|
//! | SZ   | X -> Y     | forward   | bright     |
//! | SZ†  | Y -> X     | inverse   | dim        |
//! | SX   | Y -> Z     | forward   | bright     |
//! | SX†  | Z -> Y     | inverse   | dim        |
//! | SY   | Z -> X     | forward   | bright     |
//! | SY†  | X -> Z     | inverse   | dim        |
//! | H    | Z <-> X    | self-adj  | bright     |
//! | F    | X->Y->Z->X | forward   | bright     |
//! | F†   | X->Z->Y->X | inverse   | dim        |
//!
//! The algebra is sign-blind on the hue axis: daggers get the same hue as
//! their non-dagger counterparts since the hue represents which axes are
//! coupled. The brightness axis encodes direction.

/// Color scheme for ZX diagram rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorScheme {
    /// PECOS convention: RGB = XYZ (Blue=Z, Red=X).
    #[default]
    Pecos,
    /// Traditional ZX calculus convention (Green=Z, Red=X).
    ZxCanonical,
}

impl ColorScheme {
    /// Returns the color palette for this scheme.
    #[must_use]
    pub fn palette(self) -> &'static Palette {
        match self {
            Self::Pecos => &PECOS_PALETTE,
            Self::ZxCanonical => &ZX_CANONICAL_PALETTE,
        }
    }

    /// ANSI color for Z spiders.
    #[must_use]
    pub fn ansi_z(self) -> AnsiColor {
        match self {
            Self::Pecos => AnsiColor::Blue,
            Self::ZxCanonical => AnsiColor::Green,
        }
    }

    /// ANSI color for X spiders.
    #[must_use]
    pub fn ansi_x(self) -> AnsiColor {
        AnsiColor::Red
    }

    /// ANSI color for H boxes.
    #[must_use]
    pub fn ansi_h(self) -> AnsiColor {
        AnsiColor::Yellow
    }

    /// ANSI color for Hadamard edges.
    ///
    /// PECOS: Magenta (Blue+Red = Z+X, the axes H interconverts).
    #[must_use]
    pub fn ansi_hadamard(self) -> AnsiColor {
        match self {
            Self::Pecos => AnsiColor::Magenta,
            Self::ZxCanonical => AnsiColor::Blue,
        }
    }

    /// ANSI color for a forward X<->Z gate (H, SY).
    #[must_use]
    pub fn ansi_gate_xz_fwd(self) -> AnsiColor {
        match self {
            Self::Pecos => AnsiColor::Magenta,
            Self::ZxCanonical => AnsiColor::Blue,
        }
    }

    /// ANSI color for an inverse X<->Z gate (SY†).
    #[must_use]
    pub fn ansi_gate_xz_inv(self) -> AnsiColor {
        match self {
            Self::Pecos => AnsiColor::BrightMagenta,
            Self::ZxCanonical => AnsiColor::BrightBlue,
        }
    }

    /// ANSI color for a forward X<->Y gate (SZ).
    #[must_use]
    pub fn ansi_gate_xy_fwd(self) -> AnsiColor {
        AnsiColor::Yellow
    }

    /// ANSI color for an inverse X<->Y gate (SZ†).
    #[must_use]
    pub fn ansi_gate_xy_inv(self) -> AnsiColor {
        AnsiColor::BrightYellow
    }

    /// ANSI color for a forward Y<->Z gate (SX).
    #[must_use]
    pub fn ansi_gate_yz_fwd(self) -> AnsiColor {
        AnsiColor::Cyan
    }

    /// ANSI color for an inverse Y<->Z gate (SX†).
    #[must_use]
    pub fn ansi_gate_yz_inv(self) -> AnsiColor {
        AnsiColor::BrightCyan
    }

    /// ANSI color for a forward all-axis gate (F).
    #[must_use]
    pub fn ansi_gate_xyz_fwd(self) -> AnsiColor {
        AnsiColor::White
    }

    /// ANSI color for an inverse all-axis gate (F†).
    #[must_use]
    pub fn ansi_gate_xyz_inv(self) -> AnsiColor {
        AnsiColor::BrightWhite
    }
}

/// A complete set of colors for rendering ZX diagrams.
pub struct Palette {
    // Spider / vertex colors
    pub z_fill: &'static str,
    pub z_stroke: &'static str,
    pub x_fill: &'static str,
    pub x_stroke: &'static str,
    pub h_fill: &'static str,
    pub h_stroke: &'static str,
    pub boundary_fill: &'static str,
    pub boundary_stroke: &'static str,

    // Edge colors
    pub edge_normal: &'static str,
    pub edge_hadamard: &'static str,
    pub hadamard_square: &'static str,

    // Gate colors from the color algebra.
    // Forward (bright): follows X -> Y -> Z -> X cycle.
    // Inverse (dim): dagger, against the cycle.
    /// X <-> Z forward (H, SY): saturated magenta in PECOS.
    pub gate_xz_fwd: &'static str,
    /// X <-> Z inverse (SY†): desaturated magenta in PECOS.
    pub gate_xz_inv: &'static str,
    /// X <-> Y forward (SZ): saturated yellow in PECOS.
    pub gate_xy_fwd: &'static str,
    /// X <-> Y inverse (SZ†): desaturated yellow in PECOS.
    pub gate_xy_inv: &'static str,
    /// Y <-> Z forward (SX): saturated cyan in PECOS.
    pub gate_yz_fwd: &'static str,
    /// Y <-> Z inverse (SX†): desaturated cyan in PECOS.
    pub gate_yz_inv: &'static str,
    /// All-axis forward (F): dark grey in PECOS.
    pub gate_xyz_fwd: &'static str,
    /// All-axis inverse (F†): light grey in PECOS.
    pub gate_xyz_inv: &'static str,

    // Text and background
    pub phase_text: &'static str,
    pub label_text: &'static str,
    pub background: &'static str,
}

/// PECOS color palette: Blue=Z, Red=X.
static PECOS_PALETTE: Palette = Palette {
    z_fill: "#6495ED",
    z_stroke: "#1E3A8A",
    x_fill: "#FF6B6B",
    x_stroke: "#8B0000",
    h_fill: "#FFD700",
    h_stroke: "#B8860B",
    boundary_fill: "#333333",
    boundary_stroke: "#000000",
    edge_normal: "#444444",
    edge_hadamard: "#C850C0",
    hadamard_square: "#DDA0DD",
    gate_xz_fwd: "#C850C0",  // Magenta (H, SY)
    gate_xz_inv: "#E8A0E0",  // Light magenta (SY†)
    gate_xy_fwd: "#DAA520",  // Goldenrod (SZ)
    gate_xy_inv: "#F0D080",  // Light goldenrod (SZ†)
    gate_yz_fwd: "#00B4D8",  // Cyan (SX)
    gate_yz_inv: "#80D8E8",  // Light cyan (SX†)
    gate_xyz_fwd: "#707070", // Dark grey (F)
    gate_xyz_inv: "#B0B0B0", // Light grey (F†)
    phase_text: "#000000",
    label_text: "#666666",
    background: "#FFFFFF",
};

/// ZX canonical color palette: Green=Z, Red=X.
static ZX_CANONICAL_PALETTE: Palette = Palette {
    z_fill: "#98FB98",
    z_stroke: "#2E8B57",
    x_fill: "#FF6B6B",
    x_stroke: "#8B0000",
    h_fill: "#FFD700",
    h_stroke: "#B8860B",
    boundary_fill: "#333333",
    boundary_stroke: "#000000",
    edge_normal: "#444444",
    edge_hadamard: "#4169E1",
    hadamard_square: "#FFD700",
    gate_xz_fwd: "#4169E1",  // Royal blue (H, SY)
    gate_xz_inv: "#8CA0E8",  // Light blue (SY†)
    gate_xy_fwd: "#DAA520",  // Goldenrod (SZ)
    gate_xy_inv: "#F0D080",  // Light goldenrod (SZ†)
    gate_yz_fwd: "#00CED1",  // Dark turquoise (SX)
    gate_yz_inv: "#80E8E8",  // Light turquoise (SX†)
    gate_xyz_fwd: "#707070", // Dark grey (F)
    gate_xyz_inv: "#B0B0B0", // Light grey (F†)
    phase_text: "#000000",
    label_text: "#666666",
    background: "#FFFFFF",
};

/// ANSI terminal color for colored ASCII output.
///
/// Standard colors are used for "forward" gate directions (saturated),
/// bright variants for "inverse" / dagger directions (lighter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnsiColor {
    Red,
    Green,
    Blue,
    Yellow,
    Magenta,
    Cyan,
    White,
    // Bright variants (for inverse/dagger gate directions)
    BrightRed,
    BrightGreen,
    BrightBlue,
    BrightYellow,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl AnsiColor {
    /// Returns the ANSI escape code to set this foreground color.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Red => "\x1b[31m",
            Self::Green => "\x1b[32m",
            Self::Yellow => "\x1b[33m",
            Self::Blue => "\x1b[34m",
            Self::Magenta => "\x1b[35m",
            Self::Cyan => "\x1b[36m",
            Self::White => "\x1b[37m",
            Self::BrightRed => "\x1b[91m",
            Self::BrightGreen => "\x1b[92m",
            Self::BrightYellow => "\x1b[93m",
            Self::BrightBlue => "\x1b[94m",
            Self::BrightMagenta => "\x1b[95m",
            Self::BrightCyan => "\x1b[96m",
            Self::BrightWhite => "\x1b[97m",
        }
    }

    /// Returns the ANSI reset escape code.
    #[must_use]
    pub fn reset() -> &'static str {
        "\x1b[0m"
    }
}

// Pauli web overlay colors (semi-transparent, not scheme-dependent)
pub const WEB_COLORS: &[&str] = &[
    "rgba(255, 0, 0, 0.4)",
    "rgba(0, 0, 255, 0.4)",
    "rgba(0, 180, 0, 0.4)",
    "rgba(255, 165, 0, 0.4)",
    "rgba(148, 0, 211, 0.4)",
    "rgba(0, 206, 209, 0.4)",
    "rgba(255, 20, 147, 0.4)",
    "rgba(128, 128, 0, 0.4)",
];

/// Opaque versions of `WEB_COLORS` for text labels and legend swatches.
pub const WEB_COLORS_OPAQUE: &[&str] = &[
    "#FF0000", "#0000FF", "#00B400", "#FFA500", "#9400D3", "#00CED1", "#FF1493", "#808000",
];

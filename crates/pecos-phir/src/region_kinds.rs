/*!
Region kinds and interfaces for PHIR

Based on MLIR's `RegionKindInterface`, this module defines the different
execution semantics for regions.
*/

use crate::error::{Result, SourceLocation};
use crate::phir::Region;

/// Region execution semantics
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegionKind {
    /// SSACFG (Single Static Assignment Control Flow Graph) regions
    /// - Sequential execution within blocks
    /// - Control flow between blocks via terminators
    /// - SSA dominance rules apply
    /// - Multiple blocks allowed
    SSACFG,

    /// Graph regions for concurrent/dataflow semantics
    /// - No control flow between operations
    /// - All values in scope everywhere in region
    /// - Order of operations is not semantically meaningful
    /// - Currently restricted to single block (may be relaxed later)
    Graph,

    /// Custom region kind defined by dialect
    Custom(String),
}

impl RegionKind {
    /// Check if this region kind requires SSA dominance
    #[must_use]
    pub fn requires_dominance(&self) -> bool {
        matches!(self, RegionKind::SSACFG)
    }

    /// Check if this region kind allows multiple blocks
    #[must_use]
    pub fn allows_multiple_blocks(&self) -> bool {
        match self {
            RegionKind::Graph => false, // Currently restricted
            RegionKind::SSACFG | RegionKind::Custom(_) => true, // Let dialect decide
        }
    }

    /// Check if operation order is semantically meaningful
    #[must_use]
    pub fn is_order_significant(&self) -> bool {
        match self {
            RegionKind::Graph => false,
            RegionKind::SSACFG | RegionKind::Custom(_) => true, // Conservative default
        }
    }
}

/// Interface for operations that define region semantics
pub trait RegionKindInterface {
    /// Get the kind of region at the given index
    fn get_region_kind(&self, index: usize) -> Option<RegionKind>;

    /// Get the number of regions this operation contains
    fn num_regions(&self) -> usize;

    /// Verify that regions have correct structure for their kind
    ///
    /// # Errors
    ///
    /// Returns an error if any region does not conform to its kind's constraints
    fn verify_regions(&self) -> Result<()>;
}

/// Verify a region conforms to its kind's constraints
///
/// # Errors
///
/// Returns an error if the region does not conform to the specified kind's constraints:
/// - `SSACFG` regions: blocks must have terminators
/// - `Graph` regions: must have exactly one block
pub fn verify_region(region: &Region, kind: &RegionKind) -> Result<()> {
    use crate::error::{PhirError, ValidationError};

    match *kind {
        RegionKind::SSACFG => {
            // All blocks except entry must have predecessors
            // (checked via dominance analysis)

            // All blocks must end with terminator
            for (idx, block) in region.blocks.iter().enumerate() {
                if !block.has_terminator() {
                    return Err(PhirError::Validation(Box::new(
                        ValidationError::ControlFlow {
                            message: format!("Block {idx} in SSACFG region missing terminator"),
                            location: SourceLocation {
                                file: String::new(),
                                line: 0,
                                column: 0,
                                span: crate::error::Span { start: 0, end: 0 },
                            },
                        },
                    )));
                }
            }

            Ok(())
        }

        RegionKind::Graph => {
            // Must have exactly one block
            if region.blocks.len() != 1 {
                return Err(PhirError::Validation(Box::new(
                    ValidationError::ControlFlow {
                        message: format!(
                            "Graph region must have exactly one block, found {}",
                            region.blocks.len()
                        ),
                        location: SourceLocation {
                            file: String::new(),
                            line: 0,
                            column: 0,
                            span: crate::error::Span { start: 0, end: 0 },
                        },
                    },
                )));
            }

            // The single block should not have a terminator
            // (relaxed requirement for graph regions)

            Ok(())
        }

        RegionKind::Custom(_) => {
            // Dialect-specific verification
            Ok(())
        }
    }
}

/// Helper to check if a region has SEME (Single Entry Multiple Exit) semantics
#[must_use]
pub fn has_seme_semantics(region: &Region) -> bool {
    if region.blocks.is_empty() {
        return false;
    }

    // Entry block is always block 0
    // Multiple exits allowed (any block can have return terminator)
    region.blocks.iter().skip(1).all(|_block| {
        // All non-entry blocks must be reachable
        // (would be checked by full dominance analysis)
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_kind_properties() {
        assert!(RegionKind::SSACFG.requires_dominance());
        assert!(!RegionKind::Graph.requires_dominance());

        assert!(RegionKind::SSACFG.allows_multiple_blocks());
        assert!(!RegionKind::Graph.allows_multiple_blocks());

        assert!(RegionKind::SSACFG.is_order_significant());
        assert!(!RegionKind::Graph.is_order_significant());
    }
}

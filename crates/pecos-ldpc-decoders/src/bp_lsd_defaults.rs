//! BP+LSD paper settings (Hillmann et al., arXiv 2406.18655, section 4.3).

use crate::{BpMethod, BpSchedule, OsdMethod};

/// Maximum BP iterations.
pub const MAX_ITER: usize = 30;
/// BP algorithm.
pub const BP_METHOD: BpMethod = BpMethod::MinimumSum;
/// BP update schedule.
pub const BP_SCHEDULE: BpSchedule = BpSchedule::Parallel;
/// Minimum-sum scaling factor (ignored by product-sum).
pub const MS_SCALING_FACTOR: f64 = 0.625;
/// LSD-0, without reprocessing.
pub const LSD_METHOD: OsdMethod = OsdMethod::Osd0;
/// Reprocessing order: 0 = LSD-0, no reprocessing; > 0 = combination-sweep (OSD-CS) reprocessing of that order.
pub const LSD_ORDER: usize = 0;
/// Bits added per cluster-growth step; an explicit zero grows all candidates.
pub const BITS_PER_STEP: usize = 1;

/// Select LSD-0 for order zero and combination-sweep reprocessing otherwise.
#[must_use]
pub fn lsd_method_for_order(order: usize) -> OsdMethod {
    if order == 0 {
        LSD_METHOD
    } else {
        OsdMethod::OsdCs
    }
}

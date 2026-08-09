# PECOS BP Trellis Decoder

PECOS's degeneracy-aware BP-guided trellis decoder is an approximate logical
maximum-likelihood decoder that is exact in the unpruned limit. Its optimality
is relative to the supplied detector error model, not the underlying physics,
and pruned results have no certified bound on discarded posterior mass. Belief
propagation guides only which states pruning retains; it does not change branch
probabilities or mass arithmetic. This is not a wrap or port of an external
project.

**Experimental** (`exp/`): the defaults and optional no-path escalation ladder
remain provisional pending broader validation. The shared trellis engine lives
in `pecos-frontier`; this crate contains PECOS's configuration and decoder
facade.

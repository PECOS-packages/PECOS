// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

struct Params {
    dimension: u32,
    level: u32,
    row_stride: u32,
    logical_count: u32,
    table_width: u32,
    rank_offset_lo: u32,
    rank_offset_hi: u32,
    rank_count: u32,
    ranks_per_thread: u32,
    invocation_count: u32,
    _padding_0: u32,
    _padding_1: u32,
}

struct U64Parts {
    lo: u32,
    hi: u32,
}

struct Minimum {
    weight: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> generator_rows: array<u32>;
@group(0) @binding(2) var<storage, read> logical_rows: array<u32>;
@group(0) @binding(3) var<storage, read> binomial_table: array<U64Parts>;
@group(0) @binding(4) var<storage, read_write> minimum: Minimum;

fn less_u64(left: U64Parts, right: U64Parts) -> bool {
    return left.hi < right.hi || (left.hi == right.hi && left.lo < right.lo);
}

fn subtract_u64(left: U64Parts, right: U64Parts) -> U64Parts {
    let borrow = select(0u, 1u, left.lo < right.lo);
    return U64Parts(left.lo - right.lo, left.hi - right.hi - borrow);
}

fn add_u32(value: U64Parts, increment: u32) -> U64Parts {
    let lo = value.lo + increment;
    let carry = select(0u, 1u, lo < value.lo);
    return U64Parts(lo, value.hi + carry);
}

fn binomial(n: u32, k: u32) -> U64Parts {
    return binomial_table[n * params.table_width + k];
}

fn unrank_lexicographic(rank_value: U64Parts, combination: ptr<function, array<u32, 256>>) {
    var rank = rank_value;
    var position = 0u;
    var candidate = 0u;
    loop {
        if position >= params.level {
            break;
        }
        let remaining_k = params.level - position - 1u;
        loop {
            let remaining_n = params.dimension - candidate - 1u;
            let block_size = binomial(remaining_n, remaining_k);
            if less_u64(rank, block_size) {
                (*combination)[position] = candidate;
                candidate += 1u;
                break;
            }
            rank = subtract_u64(rank, block_size);
            candidate += 1u;
        }
        position += 1u;
    }
}

fn advance_lexicographic(combination: ptr<function, array<u32, 256>>) -> bool {
    var position = i32(params.level) - 1;
    loop {
        if position < 0 {
            return false;
        }
        let unsigned_position = u32(position);
        let limit = params.dimension - params.level + unsigned_position;
        if (*combination)[unsigned_position] < limit {
            (*combination)[unsigned_position] += 1u;
            var following = unsigned_position + 1u;
            loop {
                if following >= params.level {
                    break;
                }
                (*combination)[following] = (*combination)[following - 1u] + 1u;
                following += 1u;
            }
            return true;
        }
        position -= 1;
    }
    return false;
}

fn examine_combination(combination: ptr<function, array<u32, 256>>) {
    var codeword: array<u32, 8>;
    var word = 0u;
    loop {
        if word >= params.row_stride {
            break;
        }
        codeword[word] = 0u;
        word += 1u;
    }

    var selected = 0u;
    loop {
        if selected >= params.level {
            break;
        }
        let row = (*combination)[selected];
        word = 0u;
        loop {
            if word >= params.row_stride {
                break;
            }
            codeword[word] ^= generator_rows[row * params.row_stride + word];
            word += 1u;
        }
        selected += 1u;
    }

    var triggers_logical = false;
    var logical = 0u;
    loop {
        if logical >= params.logical_count {
            break;
        }
        var parity = 0u;
        word = 0u;
        loop {
            if word >= params.row_stride {
                break;
            }
            parity ^= countOneBits(
                codeword[word] & logical_rows[logical * params.row_stride + word]
            ) & 1u;
            word += 1u;
        }
        if parity != 0u {
            triggers_logical = true;
            break;
        }
        logical += 1u;
    }

    if triggers_logical {
        var weight = 0u;
        word = 0u;
        loop {
            if word >= params.row_stride {
                break;
            }
            weight += countOneBits(codeword[word]);
            word += 1u;
        }
        atomicMin(&minimum.weight, weight);
    }
}

@compute @workgroup_size(64)
fn enumerate_level(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let invocation = global_id.x;
    if invocation >= params.invocation_count {
        return;
    }
    let local_start = invocation * params.ranks_per_thread;
    if local_start >= params.rank_count {
        return;
    }
    let local_end = min(local_start + params.ranks_per_thread, params.rank_count);
    let first_rank = add_u32(
        U64Parts(params.rank_offset_lo, params.rank_offset_hi),
        local_start,
    );
    var combination: array<u32, 256>;
    unrank_lexicographic(first_rank, &combination);

    var local_rank = local_start;
    loop {
        if local_rank >= local_end {
            break;
        }
        examine_combination(&combination);
        local_rank += 1u;
        if local_rank < local_end && !advance_lexicographic(&combination) {
            break;
        }
    }
}

/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

#include <math.h>
#include "rng_pcg.h"

// RNG state structure
typedef struct pcg_state_setseq_64 {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

// global RNG state
static pcg32_random_t pcg32_global = {
    0x853c49e6748fea9bULL,
    0xda3e39cb94b95bdbULL
};

// default multi[plier]
#define PCG_DEFAULT_MULTIPLIER_64  6364136223846793005ULL

// helper functions
static inline uint32_t pcg_rotr_32(uint32_t value, unsigned int urot) {
    int rot = (int)urot;
    return (value >> rot) | (value << ((-rot) & 31));
}

static inline void pcg_setseq_64_step_r(pcg32_random_t* rng) {
    rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_64 + rng->inc;
}

static inline uint32_t pcg_output_xsh_rr_64_32(uint64_t state) {
    return pcg_rotr_32(((state >> 18u) ^ state) >> 27u, state >> 59u);
}

static inline uint32_t pcg32_random_r(pcg32_random_t* rng) {
    const uint64_t oldstate = rng->state;
    pcg_setseq_64_step_r(rng);
    return pcg_output_xsh_rr_64_32(oldstate);
}

static inline uint32_t pcg32_boundedrand_r(pcg32_random_t* rng, uint32_t ubound) {
    int32_t bound = (int32_t)ubound;
    uint32_t threshold = -bound % bound;
    for (;;) {
        const uint32_t r = pcg32_random_r(rng);
        if (r >= threshold)
            return r % bound;
    }
}

static inline void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg_setseq_64_step_r(rng);
    rng->state += initstate;
    pcg_setseq_64_step_r(rng);
}

// public interface to RNG

uint32_t pcg32_random() {
    return pcg32_random_r(&pcg32_global);
}

uint32_t pcg32_boundedrand(uint32_t bound) {
    return pcg32_boundedrand_r(&pcg32_global, bound);
}

double pcg32_frandom() {
    return ldexp(pcg32_random(), -32);
}

void pcg32_srandom(uint64_t seq) {
    pcg32_srandom_r(&pcg32_global, 42u, seq);
}

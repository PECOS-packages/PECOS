#include <nanobind/nanobind.h>
#include "rng_pcg.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pecos_rng, m) {
    m.def("pcg32_random", &pcg32_random,"generate random numbers");
    m.def("pcg32_frandom", &pcg32_frandom, "Generate random floating point number");
    m.def("pcg32_boundedrand", &pcg32_boundedrand,  "Generate bounded random number");
    m.def("pcg32_srandom", &pcg32_srandom,  "seeded random");
}

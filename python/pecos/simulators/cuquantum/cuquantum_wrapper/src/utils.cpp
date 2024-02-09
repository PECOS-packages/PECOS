// Copyright 2022 The PECOS developers
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// Initial author: Tyson Lawrence

#include "utils.hpp"


/*
 *
 * Binary string conversion
 *
 */
std::string num_to_binary(int num, size_t len)
{
    std::string bstr;
    for (size_t i = (1 << (len - 1)); i > 0; i = i / 2)
        bstr += (num & i) ? "1" : "0";
    return bstr;
}

/*
 *
 * Get the power of two exponent
 *
 */
size_t get_power_of_two_exponent(size_t value)
{
    size_t x = 0;
    while(static_cast<size_t>( 1 << x ) != value)
        x ++;
    return x;
}

/*
 *
 * Random numbers
 *
 */
double rand_in_range(double rmin, double rmax)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dist(rmin, rmax);
    return dist(rng);
}

/*
 *
 * Bit string (vector of int 0s and 1s) helpers
 *
 */
// Compress the bit string, which is a vector of int 0s and 1s to an
// unsigned integer with the same bit representation
uint64_t compress_bit_string(std::vector<int32_t> const &bit_string)
{
    if (bit_string.size() > 64) {
        throw std::runtime_error("Can not compress bit string larger than 64 bits");
    }

    uint64_t result = 0;
    for (size_t i = 0; i < bit_string.size(); i++) {
        if (bit_string[i])
            result |= (1 << i);

    }
    return result;
}

// Print a bit string to stdout
void print_bit_string(std::vector<int32_t> const &bit_string, bool add_newline)
{
    auto it = bit_string.rbegin();
    for (size_t i = 0; i < bit_string.size(); i++) {
        std::cout << *it;
        it++;
    }
    if (add_newline)
        std::cout << std::endl;
}

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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

/*
 *
 * Helper macros for during development and in examples
 *
 */
#define PRINT(x) std::cout << x << std::endl;
#define ELAPSED(timer) std::cout << "Elapsed: " << timer.elapsed() << " s" << std::endl;

/*
 *
 * Timer class with Matlab-like semantics (tic and toc)
 *
 */
template <typename T = std::chrono::duration<double>,
          typename CLOCK_T = std::chrono::high_resolution_clock>
class Timer 
{
  public:
    Timer() noexcept : t_start{CLOCK_T::now()}, t_stop{t_start} {}

    // Start the timer
    Timer& tic() noexcept 
    {
        t_start = t_stop = CLOCK_T::now();
        return *this;
    }

    // Stop the timer
    Timer& toc() noexcept 
    {
        t_stop = CLOCK_T::now();
        return *this;
    }

    // Return the elapsed tics/counts between tic and now
    double elapsed(bool do_toc=true) noexcept 
    {
        if (do_toc)
            toc();

        return static_cast<double>(
            std::chrono::duration_cast<T>(t_stop - t_start).count());
    }

private:

    // Start and stop time points
    typename CLOCK_T::time_point t_start, t_stop;

}; // class Timer


/*
 *
 * Create a range of integeral values as a vector (like Python's arange)
 *
 */
template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) 
{
    static_assert(std::is_integral<T>::value, "Integral required.");
    std::vector<T> values;
    for (T value = start; value != stop; value += step)
        values.push_back(value);
    return values;
}

template <typename T>
void print_vector(const std::vector<T> &vec, size_t wrap=16)
{
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if ( i < vec.size()-1 )
            std::cout << ", ";
        if ( (i+1) % wrap == 0 )
            std::cout << "\n";
    }
    std::cout << std::endl;
}


/*
 *
 * Binary string conversion
 *
 */
std::string num_to_binary(int num, size_t len);

/*
 *
 * Get the power of two exponent
 *
 */
size_t get_power_of_two_exponent(size_t value);

/*
 *
 * Is a number odd or even
 *
 */
template <typename T>
bool is_odd(T n)
{
    return n % 2 != 0;
}

template <typename T>
bool is_even(T n)
{
    return n % 2 == 0;
}

/*
 *
 * Random numbers
 *
 */
double rand_in_range(double rmin=0.0, double rmax=1.0);

/*
 *
 * Math
 *
 */
template <typename T>
T mean(std::vector<T> const& v)
{
    if (v.empty())
        return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<T>(v.size());
}


/*
 *
 * Bit string (vector of int 0s and 1s) helpers
 *
 */
// Compress the bit string, which is a vector of int 0s and 1s to an
// unsigned integer with the same bit representation
uint64_t compress_bit_string(std::vector<int32_t> const &bit_string);

// Print a bit string to stdout
void print_bit_string(std::vector<int32_t> const &bit_string, bool add_newline=true);



#endif // UTILS_HPP
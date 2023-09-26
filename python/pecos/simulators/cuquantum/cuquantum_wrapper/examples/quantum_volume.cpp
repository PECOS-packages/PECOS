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

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <bitset>
#include <array>
#include <algorithm>
#include <random>

#include "argparse.hpp"

#include "custatevec_workspace.hpp"
#include "gate.hpp"         
#include "state_vector.hpp"         
#include "utils.hpp"
#include "quantum_volume.hpp"         
#include "version.h"         

/*
 *
 * Run a quantum volume simulation on the device
 *
 */
void run_quantum_volume(size_t num_qubits=2, size_t num_shots=1, bool verbose=true)
{
    Timer timer;
    timer.tic();

    // Initialize the workspace
    CuStatevecWorkspace ws;

    // Initialize the state vector to |000...0>
    StateVector sv(num_qubits);
    sv.init_on_device();

    // Create, apply, and free the QV circuit
    QuantumVolume qv(num_qubits);
    qv.copy_to_device();
    qv.apply(sv,ws);
    qv.free_on_device();

    // DISABLING FOR NOW
    // Make measurements for the shots
    // constexpr bool collapse = false;
    // double randnum = rand_in_range(0.0, 1.0);
    // std::vector<uint64_t> results(num_shots);
    // std::vector<double> times(num_shots);
    // BasisBits bit_string(num_qubits, -1);
    // for (size_t shot = 0; shot < num_shots; shot++) {
    //     randnum = rand_in_range(0.0, 1.0);
    //     sv.batch_measure_all(ws, bit_string, randnum, collapse);
    //     results[shot] = compress_bit_string(bit_string);
    // }

    std::vector<double> probs = sv.get_probabilities(ws);

    sv.read_from_device();
    sv.free_on_device();

    double total_time = timer.elapsed();

    // Optionally, print some stuff to stdout
    std::cout << "   num_qubits: " << num_qubits << std::endl;
    std::cout << "    num_shots: " << num_shots << std::endl;
    std::cout << "      elapsed: " << total_time << " s" << std::endl;
    if (verbose) {
        std::cout << "probabilities: " << std::endl;
        for (auto p : probs)
            std::cout << p << std::endl;
    }

    // DISABLING FOR NOW
    // Write the results to file
    // std::ofstream f;
    // std::string fname = "results_" + std::to_string(num_qubits) + ".txt";
    // f.open (fname);
    // f << num_qubits << "\n";
    // for (auto result : results)
    //     f << result << "\n";
    // f.close();
}


/*
 *
 * Main
 *
 */
int main(int argc, char *argv[]) 
{
    argparse::ArgumentParser args(argv[0], "0", argparse::default_arguments::help);

    args.add_argument("-n", "--num-qubits")
        .default_value(10)
        .scan<'i', int>()
        .help("Number of qubits (should be even)");

    args.add_argument("-s", "--num-shots")
        .default_value(3)
        .scan<'i', int>()
        .help("Number of shots (iterations) to run the circuit");

    args.add_argument("-v", "--verbose")
        .default_value(false)
        .implicit_value(true)
        .help("Verbose output");

    try {
        args.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << args;
        std::exit(1);
    }

    // Get command line arguments
    size_t num_qubits = args.get<int>("num-qubits");
    num_qubits = is_odd(num_qubits) ?  num_qubits++ : num_qubits;

    size_t num_shots = args.get<int>("num-shots");
    num_shots = (num_shots < 1) ? 1 : num_shots;

    bool verbose = args.get<bool>("verbose");

    run_quantum_volume(num_qubits, num_shots, verbose);

    return 0;
}

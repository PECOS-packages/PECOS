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

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "custatevec_workspace.hpp"
#include "state_vector.hpp"
#include "gate.hpp"
#include "quantum_volume.hpp"
#include "version.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cuquantum_wrapper, m) {

    m.attr("__version__") = VERSION_STR;

    m.doc() = R"pbdoc(
        cuQuantum Wrapper library Python bindings
        ---------------------------------
        CuStatevecWorkspace
        StateVector
        Gate
        QuantumVolume
    )pbdoc";

    /*
     *
     * Bindings (limited) for some of the main classes
     *
     */
    py::class_<CuStatevecWorkspace>(m, "CuStatevecWorkspace")
        .def(py::init<>(), R"pbdoc(Workspace Workspace for state vector simulations)pbdoc")
        .def("get_extra_sz", &CuStatevecWorkspace::get_extra_sz);

    py::class_<StateVector>(m, "StateVector")
        .def(py::init<size_t>(), R"pbdoc(A state vector)pbdoc")
        .def(py::init<const Eigen::VectorXcd>(), R"pbdoc(A state vector)pbdoc")
        .def("set", &StateVector::set, "Set the on host state vector from a numpy array")
        .def("get", &StateVector::get, "Get the on host state vector as a numpy array")
        .def("init_on_device", &StateVector::init_on_device, "Initialize the on device state vector to the zero state")
        .def("copy_to_device", &StateVector::copy_to_device, "Copy the on host state vector to the device")
        .def("free_on_device", &StateVector::free_on_device, "Free the state vector on the device")
        .def("read_from_device", &StateVector::read_from_device, "Read the state vector on the device back to the host")
        .def("print", &StateVector::print, "Print the on host state vector to stdout")
        .def("reset", static_cast<
            void (StateVector::*)(CuStatevecWorkspace&)
            >(&StateVector::reset), "Reset the on device statevector")
        .def("measure", static_cast<
            int32_t (StateVector::*)(CuStatevecWorkspace&, const BasisBits&, double, bool)
            >(&StateVector::measure), "Measure")
        .def("batch_measure", static_cast<
            std::vector<int32_t> (StateVector::*)(CuStatevecWorkspace&, const BasisBits&, double, bool)
            >(&StateVector::batch_measure), "Batch measure select qubits")
        .def("batch_measure_all", &StateVector::batch_measure_all, "Batch measure all qubits")
        .def("get_probabilities", &StateVector::get_probabilities, "Get the state probabilities");

    py::class_<Gate>(m, "Gate")
        .def(py::init<size_t, size_t>(), R"pbdoc(A gate)pbdoc")
        .def(py::init<const Eigen::MatrixXcd>(), R"pbdoc(A gate)pbdoc")
        .def("copy_to_device", &Gate::copy_to_device, "Copy the gate from the host to the device")
        .def("free_on_device", &Gate::free_on_device, "Free the gate on the device")
        .def("apply", static_cast<
            void (Gate::*)(StateVector&, CuStatevecWorkspace&, const Controls&, const Targets&, bool)
            >(&Gate::apply), "Apply the gate to the target state vector");

    py::class_<QuantumVolume>(m, "QuantumVolume")
        .def(py::init<size_t>(), R"pbdoc(A randomized quantum volume circuit)pbdoc")
        .def("print", &QuantumVolume::print, "Print the QV circuit to stdout")
        .def("copy_to_device", &QuantumVolume::copy_to_device, "Copy QV circuit from the host to the device")
        .def("free_on_device", &QuantumVolume::free_on_device, "Free the QV circuit on the device")
        .def("apply", &QuantumVolume::apply, "Run the QV circuit");

    /*
     *
     * Bindings for select gate matrices
     *
     */
    // Single qubit gates
    m.def("Rx", &Rx, R"pbdoc(Rx gate)pbdoc");
    m.def("Ry", &Ry, R"pbdoc(Ry gate)pbdoc");
    m.def("Rz", &Rz, R"pbdoc(Rz gate)pbdoc");
    m.def("U1q", &U1q, R"pbdoc(U1q gate)pbdoc");
    m.def("U", &U, R"pbdoc(U gate)pbdoc");
    m.def("PauliX", &PauliX, R"pbdoc(PauliX gate)pbdoc");
    m.def("PauliY", &PauliY, R"pbdoc(PauliY gate)pbdoc");
    m.def("PauliZ", &PauliZ, R"pbdoc(PauliZ gate)pbdoc");

    // Two qubit gates
    m.def("SqrtZZ", &SqrtZZ, R"pbdoc(SqrtZZ gate)pbdoc");
    m.def("RZZ", &RZZ, R"pbdoc(RZZ gate)pbdoc");
    m.def("Hadamard", &Hadamard, R"pbdoc(Hadamard gate)pbdoc");
    m.def("CNOT", &CNOT, R"pbdoc(CNOT gate)pbdoc");


}

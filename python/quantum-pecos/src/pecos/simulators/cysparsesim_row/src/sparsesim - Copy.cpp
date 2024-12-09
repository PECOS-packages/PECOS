// Copyright 2018 The PECOS Developers
// Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "sparsesim.h"

int_set_vec build_empty(int_num size) {
    int_set_vec matrix(size);

    return matrix;
}

int_set_vec build_ones(int_num size){

    int_set_vec matrix = build_empty(size);

    for( int_num i = 0; i < size; i++) {
        matrix[i].insert(i);
    }

    return matrix;
}

State::State(const int_num& num_qubits)
    :num_qubits(num_qubits)
{
    //num_qubits = num_qubits;
    clear();
}

void State::clear() {  // Allows state to reinitialize.

    // Initialize stabilizers.
    stabs.col_x = build_empty(num_qubits);
    stabs.col_z = build_ones(num_qubits);
    stabs.row_x = build_empty(num_qubits);
    stabs.row_z = build_ones(num_qubits);

    // Initialize destabilizers.
    destabs.col_x = build_ones(num_qubits);
    destabs.col_z = build_empty(num_qubits);
    destabs.row_x = build_ones(num_qubits);
    destabs.row_z = build_empty(num_qubits);

    //Inilialize signs columns
    signs_minus.clear();
    signs_i.clear();
}

void State::hadamard(const int_num& qubit) {
    /*
    X -> Z
    Z -> X
    W -> -W
    Y -> -Y
    */

    // X and Z -> -1
    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_z[s].count(qubit) & stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);

            } else {
                signs_minus.insert(s);

            }

        } else {  // Only X or Z

            if(stabs.row_z[s].count(qubit)) {
                stabs.row_x[s].insert(qubit);
                stabs.row_z[s].erase(qubit);

            } else {

                if(stabs.row_x[s].count(qubit)) {
                    stabs.row_z[s].insert(qubit);
                    stabs.row_x[s].erase(qubit);

                }
            }

        }

        if(!(destabs.row_z[s].count(qubit) & destabs.row_x[s].count(qubit))) {  // Only X or Z

            if(destabs.row_z[s].count(qubit)) {
                destabs.row_x[s].insert(qubit);
                destabs.row_z[s].erase(qubit);

            } else {

                if(destabs.row_x[s].count(qubit)) {
                    destabs.row_z[s].insert(qubit);
                    destabs.row_x[s].erase(qubit);

                }
            }

        }


    }

     stabs.col_x[qubit].swap(stabs.col_z[qubit]);
     destabs.col_x[qubit].swap(destabs.col_z[qubit]);


}

void hadamard_gen_mod(Generators& gen, const int_num& qubit) {
    // X <-> Z


    // Swap for rows
    for (const int_num& x_gen_id: gen.col_x[qubit]) {
        if (gen.col_z[qubit].count(x_gen_id) == 0) {
            gen.row_x[x_gen_id].erase(qubit);
            gen.row_z[x_gen_id].insert(qubit);
        }
    }

    for (const int_num& z_gen_id: gen.col_z[qubit]) {
        if (gen.col_x[qubit].count(z_gen_id) == 0) {
            gen.row_z[z_gen_id].erase(qubit);
            gen.row_x[z_gen_id].insert(qubit);
        }
    }

    // Swap for columns
    gen.col_x[qubit].swap(gen.col_z[qubit]);

}

void State::bitflip(const int_num& qubit) {

    /*
    for (const int_num& elem: stabs.col_z[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);
        } else {
            signs_minus.insert(elem);

        }

    } // end for
    */

    // Z -> -1
    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_z[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);

            } else {
                signs_minus.insert(s);

            }
        }
    }

}

void State::phaseflip(const int_num& qubit) {

    // X -> -1
    // Z -> -1
    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);

            } else {
                signs_minus.insert(s);

            }
        }
    }
}

void State::Y(const int_num& qubit) {

    // Z -> -1
    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);

            } else {
                signs_minus.insert(s);

            }
        }

        if(stabs.row_z[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);

            } else {
                signs_minus.insert(s);

            }
        }
    }

}

void State::phaserot(const int_num& qubit) {
    /*
    X -> iW = Y
    Z -> Z
    W -> iX
    Y -> -X
    */

    // X -> i
    // signs_i ^= stabs.col_x[qubit]
    // plus: i * i = -1
    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_x[s].count(qubit)) {
            if (signs_i.count(s)) {
                signs_i.erase(s);

                // Now add it to signs_minus
                if(signs_minus.count(s)) {
                    signs_minus.erase(s);
                } else {
                    signs_minus.insert(s);
                }

            } else {
                signs_i.insert(s);
            }

        }

    }


    /*
    for (const int_num& i: stabs.col_x[qubit]) {
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }*/


    phaserot_gen_mod(stabs, qubit);
    phaserot_gen_mod(destabs, qubit);

}

void phaserot_gen_mod(Generators& gen, const int_num& qubit) {
    /*
    X -> iW
    Z -> Z
    W -> iX
    */
    // X -> Z


    for (const int_num& x_gen_id: gen.col_x[qubit]) {

        // Update row
        if (gen.row_z[x_gen_id].count(qubit)) {
            gen.row_z[x_gen_id].erase(qubit);
        } else {
            gen.row_z[x_gen_id].insert(qubit);
        }

        // Update column
        if ( gen.col_z[qubit].count(x_gen_id)) {
            gen.col_z[qubit].erase(x_gen_id);
        } else {
            gen.col_z[qubit].insert(x_gen_id);
        }
    }

}

void State::cnot(const int_num& tqubit, const int_num& cqubit) {
     // Xt ^= Xc  X propagates from control to target

    /*
    for (const int_num& x_gen_id: stabs.col_x[tqubit]) {
        if (stabs.row_x[x_gen_id].count(cqubit)) {
            stabs.row_x[x_gen_id].erase(cqubit);
            stabs.col_x[cqubit].erase(x_gen_id);
        } else {
            stabs.row_x[x_gen_id].insert(cqubit);
            stabs.col_x[cqubit].insert(x_gen_id);
        }
    }*/
    for (int_num s = 0; s < num_qubits; s++){

        if (stabs.row_x[s].count(tqubit)) {

            if (stabs.row_x[s].count(cqubit)) {
                stabs.row_x[s].erase(cqubit);
                stabs.col_x[cqubit].erase(s);
            } else {
                stabs.row_x[s].insert(cqubit);
                stabs.col_x[cqubit].insert(s);
            }
        }



        // Zt ^= Zc  Z propagates from target to control
        if (stabs.row_z[s].count(cqubit)) {
            if (stabs.row_z[s].count(tqubit)) {
                stabs.row_z[s].erase(tqubit);
                stabs.col_z[tqubit].erase(s);
            } else {
                stabs.row_z[s].insert(tqubit);
                stabs.col_z[tqubit].insert(s);
            }
        }

        if (destabs.row_x[s].count(tqubit)) {
            if (destabs.row_x[s].count(cqubit)) {
                destabs.row_x[s].erase(cqubit);
                destabs.col_x[cqubit].erase(s);
            } else {
                destabs.row_x[s].insert(cqubit);
                destabs.col_x[cqubit].insert(s);
            }
        }

        // Zt ^= Zc  Z propagates from target to control
        if (destabs.row_z[s].count(cqubit)) {
            if (destabs.row_z[s].count(tqubit)) {
                destabs.row_z[s].erase(tqubit);
                destabs.col_z[tqubit].erase(s);
            } else {
                destabs.row_z[s].insert(tqubit);
                destabs.col_z[tqubit].insert(s);
            }
        }
    }
}

void cnot_gen_mod(Generators& gen, const int_num& tqubit,
                  const int_num& cqubit) {

    // Xt ^= Xc  X propagates from control to target
    for (const int_num& x_gen_id: gen.col_x[tqubit]) {
        if (gen.row_x[x_gen_id].count(cqubit)) {
            gen.row_x[x_gen_id].erase(cqubit);
            gen.col_x[cqubit].erase(x_gen_id);
        } else {
            gen.row_x[x_gen_id].insert(cqubit);
            gen.col_x[cqubit].insert(x_gen_id);
        }
    }

        // Zt ^= Zc  Z propagates from target to control
    for (const int_num& z_gen_id: gen.col_z[cqubit]) {
        if (gen.row_z[z_gen_id].count(tqubit)) {
            gen.row_z[z_gen_id].erase(tqubit);
            gen.col_z[tqubit].erase(z_gen_id);
        } else {
            gen.row_z[z_gen_id].insert(tqubit);
            gen.col_z[tqubit].insert(z_gen_id);
        }
    }

}



void State::swap(const int_num& qubit1, const int_num& qubit2) {
    swap_gen_mod(stabs, qubit1, qubit2);
    swap_gen_mod(destabs, qubit1, qubit2);
}

void swap_gen_mod(Generators& gen, const int_num& qubit1,
                  const int_num& qubit2) {

    // Xs
    for (const int_num& x_gen_id: gen.col_x[qubit1]) {
        if (gen.col_x[qubit2].count(x_gen_id) == 0){
            gen.row_x[x_gen_id].erase(qubit1);
            gen.row_x[x_gen_id].insert(qubit2);
        }

    }

    for (const int_num& x_gen_id: gen.col_x[qubit2]) {
        if (gen.col_x[qubit1].count(x_gen_id) == 0){
            gen.row_x[x_gen_id].erase(qubit2);
            gen.row_x[x_gen_id].insert(qubit1);
        }

    }

    // Zs
    for (const int_num& z_gen_id: gen.col_z[qubit1]) {
        if (gen.col_z[qubit2].count(z_gen_id) == 0){
            gen.row_z[z_gen_id].erase(qubit1);
            gen.row_z[z_gen_id].insert(qubit2);
        }

    }

    for (const int_num& z_gen_id: gen.col_z[qubit2]) {
        if (gen.col_z[qubit1].count(z_gen_id) == 0){
            gen.row_z[z_gen_id].erase(qubit2);
            gen.row_z[z_gen_id].insert(qubit1);
        }

    }

    // Swap for columns
    gen.col_x[qubit1].swap(gen.col_x[qubit2]);
    gen.col_z[qubit1].swap(gen.col_z[qubit2]);

}

unsigned int State::measure(const int_num& qubit, int force=-1) {

    if (stabs.col_x[qubit].size() == 0) {  // There are no anticommuting stabilizers
        return deterministic_measure(qubit);
    } else {
        return nondeterministic_measure(qubit, force);
    }

}

unsigned int State::deterministic_measure(const int_num& qubit) {

    int_set cumulative_x;
    int_num num_minuses = 0;
    int_num num_is = 0;


    // When no stabilizers anti-commute with the measurement, this means that
    // we are measuring a stabilizer of the state. The task then is to
    // determine the sign of the measured stabilizer. The generators that have
    // destabilizers that anticommute with the measurement multiply to give
    // the measured stabilzer. Therefore, we loop through these generators.


    // Count the is and -1s out front of the stabilizers being multiplied
    // together.

    for (int_num gen_id = 0; gen_id < num_qubits; gen_id++){

        if (destabs.row_x[gen_id].count(qubit)) {



            // Determine the overall minus sign of the generators.
            if (signs_minus.count(gen_id)) {
                num_minuses += 1;
            }

            // Determine the sign contribution due to is.
            if (signs_i.count(gen_id)) {
                num_is += 1;
            }

            // When determine the sign of the measured stabilizer we are
            // effectively multiplying generators together to get the measured
            // stabilzier. We therefore have to correct the sign due to ZX -> -ZX.
            for (const int_num& q: stabs.row_z[gen_id]) {
                if (cumulative_x.count(q)) {
                    num_minuses += 1;
                }
            }

            // We only need to know the Xs contribution of the generator
            // multiplications. => cumulative_x
            for (const int_num& q: stabs.row_x[gen_id]) {
                if (cumulative_x.count(q)) {
                    cumulative_x.erase(q);
                } else {
                    cumulative_x.insert(q);
                }
            }



        }

    }


    if (num_is % 4) {  // Can only be 0 or 2 (otherwise => an overall i or -i)
        num_minuses += 1;
    }

    return num_minuses % 2;

}

unsigned int State::nondeterministic_measure(const int_num& qubit,
                                             int force=-1) {
    // Here at least one stabilizer anticommutes with the measured stabilizer.
    // Therefore, we will have to update the stabilizers and destabilizers.

    int meas_outcome;
    int_num num_minuses;
    // int_set removed_row_x, removed_row_z;
    int_set anticom_stabs, anticom_destabs;

    // anticom_stabs = int_set(stabs.col_x[qubit]);
    // anticom_destabs = int_set(destabs.col_x[qubit]);

    // Choose an anti-commuting stabilizer to remove.
    // int_num removed_id = *(stabs.col_x[qubit].begin());

    // int_num removed_id = *(stabs.col_x[qubit].begin());
    int_num removed_id = 2*num_qubits + 1;
    int_num smallest_wt = 2*num_qubits;
    int_num temp_wt;

    for (int_num gen_id = 0; gen_id < num_qubits; gen_id++){
        if (stabs.row_x[gen_id].count(qubit)) {
           anticom_stabs.insert(gen_id);
        }

        if (destabs.row_x[gen_id].count(qubit)) {
           anticom_destabs.insert(gen_id);
        }


        if (stabs.row_x[gen_id].count(qubit)) {

            temp_wt = stabs.row_x[gen_id].size() + stabs.row_z[gen_id].size();

            if (temp_wt < smallest_wt) {
                removed_id = gen_id;
                smallest_wt = temp_wt;
            }
        }


    }

    anticom_stabs.erase(removed_id);
    anticom_destabs.erase(removed_id);

    const int_set removed_row_x = int_set(stabs.row_x[removed_id]);
    const int_set removed_row_z = int_set(stabs.row_z[removed_id]);


    if (signs_minus.count(removed_id)) {

        // Update all the anti-commuting stabs signs with that of the removed stab.
        for (const int_num& gen_id: anticom_stabs) {
            if (signs_minus.count(gen_id)) {
                signs_minus.erase(gen_id);
            } else {
                signs_minus.insert(gen_id);
            }
        }

    }

    if (signs_i.count(removed_id)) {

        signs_i.erase(removed_id);  // Throw away the sign for the removed stab.

        for (const int_num& gen_id: anticom_stabs) {
            // i*i = -1
            if(signs_i.count(gen_id)) {
                signs_i.erase(gen_id);

                if (signs_minus.count(gen_id)) {
                    signs_minus.erase(gen_id);
                } else {
                    signs_minus.insert(gen_id);
                }

            } else {

                signs_i.insert(gen_id);
            }
        }
    }

    for (const int_num& q: destabs.row_x[removed_id]) {
        destabs.col_x[q].erase(removed_id);
    }

    for (const int_num& q : destabs.row_z[removed_id]) {
        destabs.col_z[q].erase(removed_id);
    }


    // for (const int_num& gen_id: stabs.col_x[qubit]) {
    for (const int_num& gen_id: anticom_stabs) {


        num_minuses  = 0;
        // Correct signs due to ZX -> -XZ
        // Count the number of minuses due to this


        for (const int_num& q: removed_row_z) {

            if (stabs.row_x[gen_id].count(q)) {
                num_minuses += 1;
            }

            if (stabs.row_z[gen_id].count(q)) {
                stabs.row_z[gen_id].erase(q);
                stabs.col_z[q].erase(gen_id);
            } else {
                stabs.row_z[gen_id].insert(q);
                stabs.col_z[q].insert(gen_id);
            }

        }

        if (num_minuses % 2) {
            if (signs_minus.count(gen_id)) {
                signs_minus.erase(gen_id);
            } else {
                signs_minus.insert(gen_id);
            }
        }

        for (const int_num& q: removed_row_x) {
            if (stabs.row_x[gen_id].count(q)) {
                stabs.row_x[gen_id].erase(q);
                stabs.col_x[q].erase(gen_id);
            } else {
                stabs.row_x[gen_id].insert(q);
                stabs.col_x[q].insert(gen_id);
            }

        }

    } // end big for loop


    // ------------------------------------------------------------------------
    // Update destabilziers



    // Add in/Multiply by the new destabilizer
    // This makes all destabilizers commute with the new stabilizer.
    for (const int_num& q: removed_row_x) {

        stabs.col_x[q].erase(removed_id);
        destabs.col_x[q].insert(removed_id);

        for (const int_num& row: anticom_destabs) {
            if (destabs.col_x[q].count(row)) {
                destabs.col_x[q].erase(row);
                destabs.row_x[row].erase(q);
            }   else {
                destabs.col_x[q].insert(row);
                destabs.row_x[row].insert(q);
            }

        }
    }


    for (const int_num& q: removed_row_z) {

        stabs.col_z[q].erase(removed_id);
        destabs.col_z[q].insert(removed_id);

        for (const int_num& row: anticom_destabs) {
            if (destabs.col_z[q].count(row)) {
                destabs.col_z[q].erase(row);
                destabs.row_z[row].erase(q);
            }   else {
                destabs.col_z[q].insert(row);
                destabs.row_z[row].insert(q);
            }

        }
    }


    // Remove replaced stabilizer with the measured stabilizer
    stabs.col_z[qubit].insert(removed_id);

    // Row update
    stabs.row_x[removed_id].clear();
    stabs.row_z[removed_id].clear();
    stabs.row_z[removed_id].insert(qubit);

    destabs.row_x[removed_id] = removed_row_x;
    destabs.row_z[removed_id] = removed_row_z;

    meas_outcome = force;

    if (meas_outcome) {
        signs_minus.insert(removed_id);
    } else {
        signs_minus.erase(removed_id);
    }

    return meas_outcome;

}

// ----------------------------------------------------------------------------

void State::R(const int_num& qubit) {
    /*
    Applies a R rotation to stabilizers/destabilizers

    R = \sqrt{XZ} = SQS^{\dagger}

    R = R
    XZ = R^2
    R^{\dagger} = R^3
    I = R^4

    X -> -Z
    Z -> X
    W -> W
    Y -> Y
    */

    // Change the sign appropriately

    // X not Z -> -1
    // -------------------
    for (const int_num& i: stabs.col_x[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // Swap X <-> Z
    hadamard_gen_mod(stabs, qubit);
    hadamard_gen_mod(destabs, qubit);

}



void State::Rd(const int_num& qubit) {
    /*
    Applies a R rotation to stabilizers/destabilizers

    R^{\dagger} = \sqrt{XZ} = SQ^{\dagger}S^{\dagger}

    R = R
    XZ = R^2
    R^{\dagger} = R^3
    I = R^4

    X -> Z
    Z -> -X
    W -> W
    Y -> Y
    */

    // Change the sign appropriately

    // Z not X -> -1
    // -------------------
    for (const int_num& i: stabs.col_z[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // Swap X <-> Z
    hadamard_gen_mod(stabs, qubit);
    hadamard_gen_mod(destabs, qubit);

}

void State::Sd(const int_num& qubit) {
    /*
    X -> -iW = -Y
    Z -> Z
    W -> -iX
    Y -> X
    */



    for (const int_num& i: stabs.col_x[qubit]) {


        // X -> -1
        if(signs_minus.count(i)) {
            signs_minus.erase(i);
        } else {
            signs_minus.insert(i);
        }


        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    phaserot_gen_mod(stabs, qubit);
    phaserot_gen_mod(destabs, qubit);

}


void State::Q(const int_num& qubit) {
    /*
    X -> X
    Z -> -iW = -Y
    W -> -iZ
    Y -> Z
    */



    for (const int_num& i: stabs.col_z[qubit]) {


        // Z -> -1
        if(signs_minus.count(i)) {
            signs_minus.erase(i);
        } else {
            signs_minus.insert(i);
        }


        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    Q_gen_mod(stabs, qubit);
    Q_gen_mod(destabs, qubit);

}

void Q_gen_mod(Generators& gen, const int_num& qubit) {
    /*
    X -> X
    Z -> Y
    */


    for (const int_num& z_gen_id: gen.col_z[qubit]) {

        // Update row
        if (gen.row_x[z_gen_id].count(qubit)) {
            gen.row_x[z_gen_id].erase(qubit);
            gen.col_x[qubit].erase(z_gen_id);
        } else {
            gen.row_x[z_gen_id].insert(qubit);
            gen.col_x[qubit].insert(z_gen_id);
        }

    }

}

void State::Qd(const int_num& qubit) {
    /*
    X -> X
    Z -> iW = Y
    W -> iZ
    Y -> -Z
    */



    for (const int_num& i: stabs.col_z[qubit]) {


        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    Q_gen_mod(stabs, qubit);
    Q_gen_mod(destabs, qubit);

}

void State::H2(const int_num& qubit) {
    /*
    X -> -Z
    Z -> -X
    W -> -W
    Y -> -Y
    */

    // X or Z -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);

        } else {
            signs_minus.insert(elem);

        }
    } // end for


    for (const int_num& elem: stabs.col_z[qubit]) {

        if(stabs.col_x[qubit].count(elem) == 0) {
            if (signs_minus.count(elem)) {
                signs_minus.erase(elem);

            } else {
                signs_minus.insert(elem);

            }
        }
    } // end for

    hadamard_gen_mod(stabs, qubit);
    hadamard_gen_mod(destabs, qubit);

}

void State::H3(const int_num& qubit) {
    /*
    X -> iW = Y
    Z -> -Z
    W -> -iX
    Y -> X
    */

    for (const int_num& i: stabs.col_z[qubit]) {

        // Z -> -1
        if(signs_minus.count(i)) {
            signs_minus.erase(i);
        } else {
            signs_minus.insert(i);
        }
    }

    for (const int_num& i: stabs.col_x[qubit]) {

        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    phaserot_gen_mod(stabs, qubit);
    phaserot_gen_mod(destabs, qubit);

}

void State::H4(const int_num& qubit) {
    /*
    X -> -iW = -Y
    Z -> -Z
    W -> iX
    Y -> -X
    */

    // X not Z -> -1
    // -------------------
    for (const int_num& i: stabs.col_x[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // Z not X -> -1
    // -------------------
    for (const int_num& i: stabs.col_z[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    for (const int_num& i: stabs.col_x[qubit]) {

        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    phaserot_gen_mod(stabs, qubit);
    phaserot_gen_mod(destabs, qubit);

}

void State::H5(const int_num& qubit) {
    /*
    X -> -X
    Z -> iW = Y
    W -> -iZ
    Y -> Z
    */

    for (const int_num& i: stabs.col_x[qubit]) {

        // X -> -1
        if(signs_minus.count(i)) {
            signs_minus.erase(i);
        } else {
            signs_minus.insert(i);
        }
    }


    for (const int_num& i: stabs.col_z[qubit]) {


        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    Q_gen_mod(stabs, qubit);
    Q_gen_mod(destabs, qubit);

}

void State::H6(const int_num& qubit) {
    /*
    X -> -X
    Z -> -iW = -Y
    W -> iZ
    Y -> -Z
    */

    // X not Z -> -1
    // -------------------
    for (const int_num& i: stabs.col_x[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // Z not X -> -1
    // -------------------
    for (const int_num& i: stabs.col_z[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }


    for (const int_num& i: stabs.col_z[qubit]) {


        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    Q_gen_mod(stabs, qubit);
    Q_gen_mod(destabs, qubit);

}


void State::F1(const int_num& qubit) {
    /*
    X -> Z
    Z -> X
    W -> -W
    Y -> -Y
    */

    // X and Z -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {

        if (stabs.col_z[qubit].count(elem)) {

            if (signs_minus.count(elem)) {
                signs_minus.erase(elem);

            } else {
                signs_minus.insert(elem);

            }
        }
    } // end for

    for (const int_num& i: stabs.col_x[qubit]) {

        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F1_gen_mod(stabs, qubit);
    F1_gen_mod(destabs, qubit);

}

void F1_gen_mod(Generators& gen, const int_num& qubit) {
    // X -> W
    // Z -> X

    for (const int_num& x_gen_id: gen.col_x[qubit]) {

        if (gen.col_z[qubit].count(x_gen_id)) {
            gen.row_x[x_gen_id].erase(qubit);
        } else {
            gen.row_z[x_gen_id].insert(qubit);
        }
    }

    for (const int_num& z_gen_id: gen.col_z[qubit]) {
        if (gen.col_x[qubit].count(z_gen_id) == 0) {
            gen.row_z[z_gen_id].erase(qubit);
            gen.row_x[z_gen_id].insert(qubit);
        }

    }

    // Swap for columns
    gen.col_x[qubit].swap(gen.col_z[qubit]);

    for (const int_num& z_gen_id: gen.col_z[qubit]) {
        if (gen.col_x[qubit].count(z_gen_id)) {
            gen.col_x[qubit].erase(z_gen_id);
        } else {
            gen.col_x[qubit].insert(z_gen_id);
        }
    }

}


void State::F2(const int_num& qubit) {
    /*
    X -> -Z
    Z -> iW = Y
    W -> iX
    Y -> -X
    */

    // X not Z -> -1
    // -------------------
    for (const int_num& i: stabs.col_x[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    for (const int_num& i: stabs.col_z[qubit]) {


        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F2_gen_mod(stabs, qubit);
    F2_gen_mod(destabs, qubit);

}

void F2_gen_mod(Generators& gen, const int_num& qubit) {
    // X -> W
    // Z -> X

    // X -> Z
    // Z -> W

    for (const int_num& z_gen_id: gen.col_z[qubit]) {

        if (gen.col_x[qubit].count(z_gen_id)) {
            gen.row_z[z_gen_id].erase(qubit);
        } else {
            gen.row_x[z_gen_id].insert(qubit);
        }
    }

    for (const int_num& x_gen_id: gen.col_x[qubit]) {
        if (gen.col_z[qubit].count(x_gen_id) == 0) {
            gen.row_x[x_gen_id].erase(qubit);
            gen.row_z[x_gen_id].insert(qubit);
        }

    }

    // Swap for columns
    gen.col_z[qubit].swap(gen.col_x[qubit]);

    for (const int_num& x_gen_id: gen.col_x[qubit]) {
        if (gen.col_z[qubit].count(x_gen_id)) {
            gen.col_z[qubit].erase(x_gen_id);
        } else {
            gen.col_z[qubit].insert(x_gen_id);
        }
    }

}

void State::F3(const int_num& qubit) {
    /*
    X -> iW = Y
    Z -> -X
    W -> iZ
    Y -> -Z
    */

    // Z not X -> -1
    // -------------------
    for (const int_num& i: stabs.col_z[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // X -> i
    // signs_i ^= stabs.col_x[qubit]
    // plus: i * i = -1
    for (const int_num& i: stabs.col_x[qubit]) {
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F1_gen_mod(stabs, qubit);
    F1_gen_mod(destabs, qubit);

}

void State::F4(const int_num& qubit) {
    /*
    X -> Z
    Z -> -iW = -Y
    W -> iX
    Y -> -X
    */

    // Z not X -> -1
    // -------------------
    for (const int_num& i: stabs.col_z[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_x[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }


    for (const int_num& i: stabs.col_z[qubit]) {

        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F2_gen_mod(stabs, qubit);
    F2_gen_mod(destabs, qubit);

}

void State::F1d(const int_num& qubit) {
    /*
    X -> Z
    Z -> iW = Y
    W -> -iX
    Y -> X
    */

    // X and Z -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {

        if (stabs.col_z[qubit].count(elem)) {

            if (signs_minus.count(elem)) {
                signs_minus.erase(elem);

            } else {
                signs_minus.insert(elem);

            }
        }
    } // end for

    for (const int_num& i: stabs.col_z[qubit]) {

        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F2_gen_mod(stabs, qubit);
    F2_gen_mod(destabs, qubit);

}

void State::F2d(const int_num& qubit) {
    /*
    X -> -iW = -Y
    Z -> -X
    W -> -iZ
    Y -> Z
    */



    // X or Z -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);

        } else {
            signs_minus.insert(elem);

        }
    } // end for


    for (const int_num& elem: stabs.col_z[qubit]) {

        if(stabs.col_x[qubit].count(elem) == 0) {
            if (signs_minus.count(elem)) {
                signs_minus.erase(elem);

            } else {
                signs_minus.insert(elem);

            }
        }
    } // end for

    // X -> i
    // signs_i ^= stabs.col_x[qubit]
    // plus: i * i = -1
    for (const int_num& i: stabs.col_x[qubit]) {
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F1_gen_mod(stabs, qubit);
    F1_gen_mod(destabs, qubit);

}

void State::F3d(const int_num& qubit) {
    /*
    X -> -Z
    Z -> -iW = -Y
    W -> -iX
    Y -> X
    */

    // X or Z -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);

        } else {
            signs_minus.insert(elem);

        }
    } // end for


    for (const int_num& elem: stabs.col_z[qubit]) {

        if(stabs.col_x[qubit].count(elem) == 0) {
            if (signs_minus.count(elem)) {
                signs_minus.erase(elem);

            } else {
                signs_minus.insert(elem);

            }
        }
    } // end for


    for (const int_num& i: stabs.col_z[qubit]) {

        // Z -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F2_gen_mod(stabs, qubit);
    F2_gen_mod(destabs, qubit);

}

void State::F4d(const int_num& qubit) {
    /*
    X -> -iW = -Y
    Z -> X
    W -> iZ
    Y -> -Z
    */

    // X not Z -> -1
    // -------------------
    for (const int_num& i: stabs.col_x[qubit]) {
        if(signs_minus.count(i)) {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.erase(i);
            }
        } else {
            if(stabs.col_z[qubit].count(i) == 0) {
                signs_minus.insert(i);
            }
        }
    }

    // X -> i
    // signs_i ^= stabs.col_x[qubit]
    // plus: i * i = -1
    for (const int_num& i: stabs.col_x[qubit]) {
        if (signs_i.count(i)) {
            signs_i.erase(i);

            // Now add it to signs_minus
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }

        } else {
            signs_i.insert(i);
        }

    }

    F1_gen_mod(stabs, qubit);
    F1_gen_mod(destabs, qubit);

}

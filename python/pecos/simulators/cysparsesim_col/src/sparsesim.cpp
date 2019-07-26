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

void State::clear() {  // Allows state to reinitilize.

    // Initialize stabilizers.
    stabs.col_x = build_empty(num_qubits);
    stabs.col_z = build_ones(num_qubits);
    
    // Initialize destabilizers.
    destabs.col_x = build_ones(num_qubits);
    destabs.col_z = build_empty(num_qubits);
    
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
    for (const int_num& elem: stabs.col_x[qubit]) {
    
        if (stabs.col_z[qubit].count(elem)) {
        
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

void hadamard_gen_mod(Generators& gen, const int_num& qubit) {
    // X <-> Z   
    
    // Swap for columns
    gen.col_x[qubit].swap(gen.col_z[qubit]);

}

void State::bitflip(const int_num& qubit) {
    
    // Z -> -1
    for (const int_num& elem: stabs.col_z[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);
        } else {
            signs_minus.insert(elem);

        }

    } // end for
}

void State::phaseflip(const int_num& qubit) {
    for (const int_num& elem: stabs.col_x[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);
        } else {
            signs_minus.insert(elem);

        }

    } // end for
}

void State::Y(const int_num& qubit) {
    
    // Z -> -1
    for (const int_num& elem: stabs.col_z[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);
        } else {
            signs_minus.insert(elem);
        }

    } // end for
    
    // X -> -1
    for (const int_num& elem: stabs.col_x[qubit]) {
        if (signs_minus.count(elem)) {
            signs_minus.erase(elem);
        } else {
            signs_minus.insert(elem);
        }

    } // end for
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
        
        // Update column
        if ( gen.col_z[qubit].count(x_gen_id)) {
            gen.col_z[qubit].erase(x_gen_id);
        } else {
            gen.col_z[qubit].insert(x_gen_id);
        }
    }

}

void State::cnot(const int_num& tqubit, const int_num& cqubit) {
    cnot_gen_mod(stabs, tqubit, cqubit);
    cnot_gen_mod(destabs, tqubit, cqubit);
}

void cnot_gen_mod(Generators& gen, const int_num& tqubit, 
                  const int_num& cqubit) {
                  
    
        // Xt ^= Xc  X propogates from control to target
    for (const int_num& x_gen_id: gen.col_x[tqubit]) {
        
        // column update
        if (gen.col_x[cqubit].count(x_gen_id)) {
            gen.col_x[cqubit].erase(x_gen_id);
        } else {
            gen.col_x[cqubit].insert(x_gen_id);
        }
    }
    
        // Zt ^= Zc  Z propogates from target to control
    for (const int_num& z_gen_id: gen.col_z[cqubit]) {
        
        // column update
        if (gen.col_z[tqubit].count(z_gen_id)) {
            gen.col_z[tqubit].erase(z_gen_id);
        } else {
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
    
    // Swap for columns
    gen.col_x[qubit1].swap(gen.col_x[qubit2]);
    gen.col_z[qubit1].swap(gen.col_z[qubit2]);
    
}

unsigned int State::measure(const int_num& qubit, int force=-1) {

    if (stabs.col_x[qubit].size() == 0) {  // There are no anticommuting stabilizers
        return deterministic_measure(qubit);
    } else {
        return nondeterministic_measure(qubit, force);
        // return 1;
    }

}


unsigned int State::deterministic_measure(const int_num& qubit) {

    int_set cumulative_x;
    int_num num_minuses = 0;
    int_num num_is = 0;
    bool has_x, has_minus;
    int_set mul_stabs = destabs.col_x[qubit];
    
    // When no stabilizers anti-commute with the measurement, this means that
    // we are measuring a stabilizer of the state. The task then is to 
    // determine the sign of the measured stabilizer. The generators that have
    // destabilizers that anticommute with the measurement multiply to give 
    // the measured stabilzer. Therefore, we loop through these generators.
    
    
    has_minus = false;
    for (int_num q = 0; q < num_qubits; q++){
        has_x = false;
        for (const int_num& stab_id: mul_stabs) {
        
            if (has_x){
        
                if (stabs.col_z[q].count(stab_id)) {
                    has_minus = !has_minus;
                }
                    
            }
            
            if (stabs.col_x[q].count(stab_id)) {
                has_x = !has_x;
            }
            
        }
    }
    
    if (has_minus) {
        num_minuses = 1;
    }
    
    
    
    // Count the is and -1s out front of the stabilizers being multiplied 
    // together.
    for (const int_num& gen_id: mul_stabs) {
    
        // Determine the overall minus sign of the generators.
        if (signs_minus.count(gen_id)) {
            num_minuses += 1;
        }
        
        // Determine the sign contribution due to is.
        if (signs_i.count(gen_id)) {
            num_is += 1;
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
    
    int_set removed_row_x, removed_row_z;
    // int_set destab_removed_row_x, destab_removed_row_z;
    int_set anticom_stabs, anticom_destabs;
    
    anticom_stabs = int_set(stabs.col_x[qubit]);
    anticom_destabs = int_set(destabs.col_x[qubit]);
    
    // Choose an anti-commuting stabilizer to remove.
    // int_num removed_id = *(stabs.col_x[qubit].begin());
    
    int_num removed_id = *(stabs.col_x[qubit].begin());
    
    /*int_num smallest_wt = 2*num_qubits + 2;
    int_num temp_wt;
    
    for (const int_num& stab_id: stabs.col_x[qubit]) {
    
        temp_wt = 0;
        
        for (int_num q = 0; q < num_qubits; q++){
            if (stabs.col_z[q].count(stab_id)) {
                temp_wt += 1;
            }
            if (stabs.col_x[q].count(stab_id)) {
                temp_wt += 1;
            }
        }
        
        if (temp_wt < smallest_wt) {
            removed_id = stab_id;
            smallest_wt = temp_wt;
        } 
    
    }*/
    
    anticom_stabs.erase(removed_id);
    anticom_destabs.erase(removed_id);
    
    for (int_num q = 0; q < num_qubits; q++){
        if (stabs.col_z[q].count(removed_id)) {
            removed_row_z.insert(q);
        }
        if (stabs.col_x[q].count(removed_id)) {
            removed_row_x.insert(q);
        }
        if (destabs.col_z[q].count(removed_id)) {
            destabs.col_z[q].erase(removed_id);
        }
        if (destabs.col_x[q].count(removed_id)) {
            destabs.col_x[q].erase(removed_id);
        }
    
    }
    
    // const int_set removed_row_x = int_set(stabs.row_x[removed_id]);
    // const int_set removed_row_z = int_set(stabs.row_z[removed_id]);
    
    
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

    
        
    // for (const int_num& gen_id: stabs.col_x[qubit]) {
    for (const int_num& gen_id: anticom_stabs) {
    
        
        num_minuses  = 0;
        // Correct signs due to ZX -> -XZ
        // Count the number of minuses due to this
           
        
        for (const int_num& q: removed_row_z) {
            
            if (stabs.col_x[q].count(gen_id)) {
                num_minuses += 1;
            }
        
            if (stabs.col_z[q].count(gen_id)) {
                stabs.col_z[q].erase(gen_id);
            } else {
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
            if (stabs.col_x[q].count(gen_id)) {
                stabs.col_x[q].erase(gen_id);
            } else {
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
            }   else {
                destabs.col_x[q].insert(row);
            }
            
        }
    }

    
    for (const int_num& q: removed_row_z) {
    
        stabs.col_z[q].erase(removed_id);
        destabs.col_z[q].insert(removed_id);
            
        for (const int_num& row: anticom_destabs) {
            if (destabs.col_z[q].count(row)) {
                destabs.col_z[q].erase(row);
            }   else {
                destabs.col_z[q].insert(row);
            }
            
        }
    }
    
    
    // Remove replaced stabilizer with the measured stabilizer
    stabs.col_z[qubit].insert(removed_id);
    
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
    
        if (gen.col_x[qubit].count(z_gen_id)) {
            gen.col_x[qubit].erase(z_gen_id);
        } else {
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
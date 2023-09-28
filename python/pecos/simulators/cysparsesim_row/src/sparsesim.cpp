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
    stabs.row_x = build_empty(num_qubits);
    stabs.row_z = build_ones(num_qubits);
    
    // Initialize destabilizers.
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
    
    for (int_num s = 0; s < num_qubits; s++){
    
        if(stabs.row_z[s].count(qubit)) {
            if(stabs.row_x[s].count(qubit)) {        
                  
                
                    if (signs_minus.count(s)) {
                        signs_minus.erase(s);
        
                    } else {
                        signs_minus.insert(s);
        
                    }
                
            } else {
            
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }    
        
        } else {  // No Z.
        
            if(stabs.row_x[s].count(qubit)) { 
                stabs.row_x[s].erase(qubit);
                stabs.row_z[s].insert(qubit);
            }    
        }
    

    
        // Swap for rows
        if(destabs.row_z[s].count(qubit)) { 
            if (destabs.row_x[s].count(qubit) == 0) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }    
        } else {
            if(destabs.row_x[s].count(qubit)) { 
            
                destabs.row_x[s].erase(qubit);
                destabs.row_z[s].insert(qubit);
            }    
        }
    
    }
    


}


void State::bitflip(const int_num& qubit) {
    
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
        
            // Update row
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
        }
        
        if (destabs.row_x[s].count(qubit)) {
        
            // Update row
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
        }
    
    }
    
    
}


void State::cnot(const int_num& tqubit, const int_num& cqubit) {
     // Xt ^= Xc  X propogates from control to target
     
    for (int_num s = 0; s < num_qubits; s++){
    
        if (stabs.row_x[s].count(tqubit)) {
    
            if (stabs.row_x[s].count(cqubit)) {
                stabs.row_x[s].erase(cqubit);
            } else {
                stabs.row_x[s].insert(cqubit);
            }
        }
        

    
        // Zt ^= Zc  Z propogates from target to control
        if (stabs.row_z[s].count(cqubit)) {
            if (stabs.row_z[s].count(tqubit)) {
                stabs.row_z[s].erase(tqubit);
            } else {
                stabs.row_z[s].insert(tqubit);
            }
        }
    
        if (destabs.row_x[s].count(tqubit)) {
            if (destabs.row_x[s].count(cqubit)) {
                destabs.row_x[s].erase(cqubit);
            } else {
                destabs.row_x[s].insert(cqubit);
            }
        }
    
        // Zt ^= Zc  Z propogates from target to control
        if (destabs.row_z[s].count(cqubit)) {
            if (destabs.row_z[s].count(tqubit)) {
                destabs.row_z[s].erase(tqubit);
            } else {
                destabs.row_z[s].insert(tqubit);
            }
        }
    }
}




void State::swap(const int_num& qubit1, const int_num& qubit2) {
    State::cnot(qubit1, qubit2);
    State::cnot(qubit2, qubit1);
    State::cnot(qubit1, qubit2);

}


unsigned int State::measure(const int_num& qubit, int force=-1) {

    bool found = false;

    for (int_num s = 0; s < num_qubits; s++){
        if (stabs.row_x[s].count(qubit)) {
            found = true;
            break;
        }
    
    }

    if (found) {  // There are no anticommuting stabilizers
        return nondeterministic_measure(qubit, force);
        // return 1;
    } else {
        return deterministic_measure(qubit);
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
    bool id_set = false;
    
    int_num removed_id = 2*num_qubits + 1;
    int_num smallest_wt = 2*num_qubits + 2;
    int_num temp_wt;
    
    for (int_num gen_id = 0; gen_id < num_qubits; gen_id++){
        if (stabs.row_x[gen_id].count(qubit)) { 
           anticom_stabs.insert(gen_id);
           
        // }
        
        // if (stabs.row_x[gen_id].count(qubit)) {
    
            temp_wt = stabs.row_x[gen_id].size() + stabs.row_z[gen_id].size();
        
            if (temp_wt < smallest_wt) {
                removed_id = gen_id;
                smallest_wt = temp_wt;
            } 
        }
        
        
        if (destabs.row_x[gen_id].count(qubit)) { 
           anticom_destabs.insert(gen_id);
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
            } else {
                stabs.row_z[gen_id].insert(q);
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
            } else {
                stabs.row_x[gen_id].insert(q);
            }
            
        }
                
    } // end big for loop
    
    
    // ------------------------------------------------------------------------
    // Update destabilziers

    
    
    // Add in/Multiply by the new destabilizer
    // This makes all destabilizers commute with the new stabilizer.
    for (const int_num& q: removed_row_x) {
    
        for (const int_num& row: anticom_destabs) {
            if (destabs.row_x[row].count(q)) {
                destabs.row_x[row].erase(q);
            }   else {
                destabs.row_x[row].insert(q);
            }
            
        }
    }

    
    for (const int_num& q: removed_row_z) {
                
        for (const int_num& row: anticom_destabs) {
            
            if (destabs.row_z[row].count(q)) {
                destabs.row_z[row].erase(q);
            }   else {
                destabs.row_z[row].insert(q);
            }
            
        }
    }
    
    
    // Remove replaced stabilizer with the measured stabilizer
    
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

    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_x[s].count(qubit)) {
            if(signs_minus.count(s)) {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.erase(s);
                }
            } else {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.insert(s);
                }
            }    
        }
    
    
            // Swap for rows
        if(stabs.row_z[s].count(qubit)) { // Is there a Z?
        
            if (stabs.row_x[s].count(qubit) == 0) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }    
        
        } else {  // No Z.
        
            if(stabs.row_x[s].count(qubit)) { 
                stabs.row_x[s].erase(qubit);
                stabs.row_z[s].insert(qubit);
            }    
        }
    

    
        // Swap for rows
        if(destabs.row_z[s].count(qubit)) { 
            if (destabs.row_x[s].count(qubit) == 0) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }    
        } else {
            if(destabs.row_x[s].count(qubit)) { 
            
                destabs.row_x[s].erase(qubit);
                destabs.row_z[s].insert(qubit);
            }    
        }
    
    }

}



void State::Rd(const int_num& qubit) {

    for (int_num s = 0; s < num_qubits; s++){
    
    
    
        // Z not X -> -1
        // -------------------
        if(stabs.row_z[s].count(qubit)) {
            if(signs_minus.count(s)) {
                if(stabs.row_x[s].count(qubit) == 0) {
                    signs_minus.erase(s);
                }
            } else {
                if(stabs.row_x[s].count(qubit) == 0) {
                    signs_minus.insert(s);
                }
            }    
        }
    
    
    
    
    
            // Swap for rows
        if(stabs.row_z[s].count(qubit)) { // Is there a Z?
        
            if (stabs.row_x[s].count(qubit) == 0) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }    
        
        } else {  // No Z.
        
            if(stabs.row_x[s].count(qubit)) { 
                stabs.row_x[s].erase(qubit);
                stabs.row_z[s].insert(qubit);
            }    
        }
    

    
        // Swap for rows
        if(destabs.row_z[s].count(qubit)) { 
            if (destabs.row_x[s].count(qubit) == 0) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }    
        } else {
            if(destabs.row_x[s].count(qubit)) { 
            
                destabs.row_x[s].erase(qubit);
                destabs.row_z[s].insert(qubit);
            }    
        }
        
        
    
    }

}

void State::Sd(const int_num& qubit) {

     for (int_num s = 0; s < num_qubits; s++){
     
        if(stabs.row_x[s].count(qubit)) {
        
        
            // X -> -1
            if(signs_minus.count(s)) {
                signs_minus.erase(s);
            } else {
                signs_minus.insert(s);
            }
            
        
            // X -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
        
        
        
        
        
        // phaserot_gen_mod

        if(stabs.row_x[s].count(qubit)) {
        
            // Update row
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
            
        }
        
        if(destabs.row_x[s].count(qubit)) {
        
            // Update row
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
            
        }

        
        
        
        
        
        
    }

}


void State::Q(const int_num& qubit) {


    for (int_num s = 0; s < num_qubits; s++){
    
        if(stabs.row_z[s].count(qubit)) {        
        
            // Z -> -1
            if(signs_minus.count(s)) {
                signs_minus.erase(s);
            } else {
                signs_minus.insert(s);
            }
            
        
            // Z -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
        
        
        
        
        
        // Q_gen_mod
        if(stabs.row_z[s].count(qubit)) {
        
            // Update row
            if (stabs.row_x[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_x[s].insert(qubit);
            }
            
        }
        
                
        if(destabs.row_z[s].count(qubit)) {
        
            // Update row
            if (destabs.row_x[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_x[s].insert(qubit);
            }
            
        }
        
        
    
    }

}


void State::Qd(const int_num& qubit) {

    for (int_num s = 0; s < num_qubits; s++){

        if(stabs.row_z[s].count(qubit)) {
                
        
            // Z -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
        
        
        
        // Q_gen_mod
        if(stabs.row_z[s].count(qubit)) {
        
            // Update row
            if (stabs.row_x[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_x[s].insert(qubit);
            }
            
        }
        
                
        if(destabs.row_z[s].count(qubit)) {
        
            // Update row
            if (destabs.row_x[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_x[s].insert(qubit);
            }
            
        }
        
    }
    
}

void State::H2(const int_num& qubit) { 

    for (int_num s = 0; s < num_qubits; s++){
           
        
        
        
        // X or Z -> -1
        if(stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);
    
            } else {
                signs_minus.insert(s);
    
            }
        } // end for
        
        
        if(stabs.row_z[s].count(qubit)) {
        
            if(stabs.row_x[s].count(qubit) == 0) {
                if (signs_minus.count(s)) {
                    signs_minus.erase(s);
        
                } else {
                    signs_minus.insert(s);
        
                }
            }
        } // end for
    
    
    

    
        // Swap for rows
        if(stabs.row_z[s].count(qubit)) { // Is there a Z?
        
            if (stabs.row_x[s].count(qubit) == 0) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }    
        
        } else {  // No Z.
        
            if(stabs.row_x[s].count(qubit)) { 
                stabs.row_x[s].erase(qubit);
                stabs.row_z[s].insert(qubit);
            }    
        }
    

    
        // Swap for rows
        if(destabs.row_z[s].count(qubit)) { 
            if (destabs.row_x[s].count(qubit) == 0) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }    
        } else {
            if(destabs.row_x[s].count(qubit)) { 
            
                destabs.row_x[s].erase(qubit);
                destabs.row_z[s].insert(qubit);
            }    
        }
    
    }
    

}




void State::H3(const int_num& qubit) {

    for (int_num s = 0; s < num_qubits; s++){
    
    
        if(stabs.row_z[s].count(qubit)) {            
            // Z -> -1
            if(signs_minus.count(s)) {
                signs_minus.erase(s);
            } else {
                signs_minus.insert(s);
            }
        }
    
        if(stabs.row_x[s].count(qubit)) {
        
            // X -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
    
    
    
    
    
        // phaserot_gen_mod

        if(stabs.row_x[s].count(qubit)) {
        
            // Update row
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
            
        }
        
        if(destabs.row_x[s].count(qubit)) {
        
            // Update row
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
            
        }    
    
    
    }
    
}

void State::H4(const int_num& qubit) {

    for (int_num i = 0; i < num_qubits; i++){


        // X not Z -> -1
        // -------------------
        if(stabs.row_x[i].count(qubit)) {
            if(signs_minus.count(i)) {
                if(stabs.row_z[i].count(qubit) == 0) {
                    signs_minus.erase(i);
                }
            } else {
                if(stabs.row_z[i].count(qubit) == 0) {
                    signs_minus.insert(i);
                }
            }    
        }
        
        
        
        
        
        // Z not X -> -1
        // -------------------
        if(stabs.row_z[i].count(qubit)) {
            if(signs_minus.count(i)) {
                if(stabs.row_x[i].count(qubit) == 0) {
                    signs_minus.erase(i);
                }
            } else {
                if(stabs.row_x[i].count(qubit) == 0) {
                    signs_minus.insert(i);
                }
            }    
        }
        
        
        
        
        
    
        if(stabs.row_x[i].count(qubit)) {
        
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














        // phaserot_gen_mod

        if(stabs.row_x[i].count(qubit)) {
        
            // Update row
            if (stabs.row_z[i].count(qubit)) {
                stabs.row_z[i].erase(qubit);
            } else {
                stabs.row_z[i].insert(qubit);
            }
            
        }
        
        if(destabs.row_x[i].count(qubit)) {
        
            // Update row
            if (destabs.row_z[i].count(qubit)) {
                destabs.row_z[i].erase(qubit);
            } else {
                destabs.row_z[i].insert(qubit);
            }
            
        }


    }    
}

void State::H5(const int_num& qubit) {

    for (int_num i = 0; i < num_qubits; i++){
    
    
    
    
        if(stabs.row_x[i].count(qubit)) {
            
            // X -> -1
            if(signs_minus.count(i)) {
                signs_minus.erase(i);
            } else {
                signs_minus.insert(i);
            }
        }
        
    
        if(stabs.row_z[i].count(qubit)) {
                
        
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
    
    
    
    
    
    
    
    
        // Q_gen_mod
        if(stabs.row_z[i].count(qubit)) {
        
            // Update row
            if (stabs.row_x[i].count(qubit)) {
                stabs.row_x[i].erase(qubit);
            } else {
                stabs.row_x[i].insert(qubit);
            }
            
        }
        
                
        if(destabs.row_z[i].count(qubit)) {
        
            // Update row
            if (destabs.row_x[i].count(qubit)) {
                destabs.row_x[i].erase(qubit);
            } else {
                destabs.row_x[i].insert(qubit);
            }
            
        }    
    
    
    
    
    
    
    
    }
    
}

void State::H6(const int_num& qubit) {

    for (int_num i = 0; i < num_qubits; i++){















        // X not Z -> -1
        // -------------------
        if(stabs.row_x[i].count(qubit)) {
            if(signs_minus.count(i)) {
                if(stabs.row_z[i].count(qubit) == 0) {
                    signs_minus.erase(i);
                }
            } else {
                if(stabs.row_z[i].count(qubit) == 0) {
                    signs_minus.insert(i);
                }
            }    
        }
        
        // Z not X -> -1
        // -------------------
        if(stabs.row_z[i].count(qubit)) {
            if(signs_minus.count(i)) {
                if(stabs.row_x[i].count(qubit) == 0) {
                    signs_minus.erase(i);
                }
            } else {
                if(stabs.row_x[i].count(qubit) == 0) {
                    signs_minus.insert(i);
                }
            }    
        }
        
    
        if(stabs.row_z[i].count(qubit)) {
                
        
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












        // Q_gen_mod
        if(stabs.row_z[i].count(qubit)) {
        
            // Update row
            if (stabs.row_x[i].count(qubit)) {
                stabs.row_x[i].erase(qubit);
            } else {
                stabs.row_x[i].insert(qubit);
            }
            
        }
        
                
        if(destabs.row_z[i].count(qubit)) {
        
            // Update row
            if (destabs.row_x[i].count(qubit)) {
                destabs.row_x[i].erase(qubit);
            } else {
                destabs.row_x[i].insert(qubit);
            }
            
        }



    }

}


void State::F1(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
    
    
    
        // X and Z -> -1
        if(stabs.row_x[s].count(qubit)) {
        
            if (stabs.row_z[s].count(qubit)) {
            
                if (signs_minus.count(s)) {
                    signs_minus.erase(s);
    
                } else {
                    signs_minus.insert(s);
    
                }
            }
        } // end for
        
        if(stabs.row_x[s].count(qubit)) {
        
            // X -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
    
    
    
    
    
        // F1_gen_mod
        if(stabs.row_x[s].count(qubit)) {
            
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }
        
        }   
        
    
    
        if(destabs.row_x[s].count(qubit)) {
            
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }
        
        } 

    
    
   }
    
}


void State::F2(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
    
        // X not Z -> -1
        // -------------------
        if(stabs.row_x[s].count(qubit)) {
            if(signs_minus.count(s)) {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.erase(s);
                }
            } else {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.insert(s);
                }
            }    
        }
        
         if(stabs.row_z[s].count(qubit)) {
            
        
            // Z -> i
            // plus: i * i = -1
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
    
    
    
    
    
    
    
    
    
    
    // F2_gen_mod
    if(stabs.row_z[s].count(qubit)) {
        
        if (stabs.row_x[s].count(qubit)) {
            stabs.row_z[s].erase(qubit);
        } else {
            stabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(stabs.row_x[s].count(qubit)) {
            stabs.row_x[s].erase(qubit);
            stabs.row_z[s].insert(qubit);
        }
    
    }   
    


    if(destabs.row_z[s].count(qubit)) {
        
        if (destabs.row_x[s].count(qubit)) {
            destabs.row_z[s].erase(qubit);
        } else {
            destabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(destabs.row_x[s].count(qubit)) {

            destabs.row_x[s].erase(qubit);
            destabs.row_z[s].insert(qubit);
        }
    
    }     
    
    
    
    
    }
    
}

void State::F3(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
        // Z not X -> -1
        // -------------------
        if(stabs.row_z[s].count(qubit)) {
            if(signs_minus.count(s)) {
                if(stabs.row_x[s].count(qubit) == 0) {
                    signs_minus.erase(s);
                }
            } else {
                if(stabs.row_x[s].count(qubit) == 0) {
                    signs_minus.insert(s);
                }
            }    
        }
        
        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
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
    
    
    
    
    
        // F1_gen_mod
        if(stabs.row_x[s].count(qubit)) {
            
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }
        
        }   
        
    
    
        if(destabs.row_x[s].count(qubit)) {
            
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }
        
        } 
    
    
    
    }

}

void State::F4(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
    
    
    // Z not X -> -1
    // -------------------
    if(stabs.row_z[s].count(qubit)) {
        if(signs_minus.count(s)) {
            if(stabs.row_x[s].count(qubit) == 0) {
                signs_minus.erase(s);
            }
        } else {
            if(stabs.row_x[s].count(qubit) == 0) {
                signs_minus.insert(s);
            }
        }    
    }
    
    
    if(stabs.row_z[s].count(qubit)) {
        
        // Z -> i
        // plus: i * i = -1
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
    
    



    
    
    
    
    // F2_gen_mod
    if(stabs.row_z[s].count(qubit)) {
        
        if (stabs.row_x[s].count(qubit)) {
            stabs.row_z[s].erase(qubit);
        } else {
            stabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(stabs.row_x[s].count(qubit)) {
            stabs.row_x[s].erase(qubit);
            stabs.row_z[s].insert(qubit);
        }
    
    }   
    


    if(destabs.row_z[s].count(qubit)) {
        
        if (destabs.row_x[s].count(qubit)) {
            destabs.row_z[s].erase(qubit);
        } else {
            destabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(destabs.row_x[s].count(qubit)) {

            destabs.row_x[s].erase(qubit);
            destabs.row_z[s].insert(qubit);
        }
    
    }     
    
    
    
    
    
    
    
    }
    
}

void State::F1d(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    

        // X and Z -> -1
        if(stabs.row_x[s].count(qubit)) {
        
            if (stabs.row_z[s].count(qubit)) {
            
                if (signs_minus.count(s)) {
                    signs_minus.erase(s);
    
                } else {
                    signs_minus.insert(s);
    
                }
            }
        } // end for
        
        if(stabs.row_z[s].count(qubit)) {
            
            // Z -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
        
        
        
        
    // F2_gen_mod
    if(stabs.row_z[s].count(qubit)) {
        
        if (stabs.row_x[s].count(qubit)) {
            stabs.row_z[s].erase(qubit);
        } else {
            stabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(stabs.row_x[s].count(qubit)) {
            stabs.row_x[s].erase(qubit);
            stabs.row_z[s].insert(qubit);
        }
    
    }   
    


    if(destabs.row_z[s].count(qubit)) {
        
        if (destabs.row_x[s].count(qubit)) {
            destabs.row_z[s].erase(qubit);
        } else {
            destabs.row_x[s].insert(qubit);
        }
    } else {
    
        if(destabs.row_x[s].count(qubit)) {

            destabs.row_x[s].erase(qubit);
            destabs.row_z[s].insert(qubit);
        }
    
    }     
    
    
    

    }
    
}

void State::F2d(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
    
        // X or Z -> -1
        if(stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);
    
            } else {
                signs_minus.insert(s);
    
            }
        } // end for
        
        
        if(stabs.row_z[s].count(qubit)) {
        
            if(stabs.row_x[s].count(qubit) == 0) {
                if (signs_minus.count(s)) {
                    signs_minus.erase(s);
        
                } else {
                    signs_minus.insert(s);
        
                }
            }
        } // end for
        
        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
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
    
    
    
    
    
        // F1_gen_mod
        if(stabs.row_x[s].count(qubit)) {
            
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }
        
        }   
        
    
    
        if(destabs.row_x[s].count(qubit)) {
            
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }
        
        }     
    
    
    
    
    
    }
    
}

void State::F3d(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    
    
    
    
    
    
    
    
        // X or Z -> -1
        if(stabs.row_x[s].count(qubit)) {
            if (signs_minus.count(s)) {
                signs_minus.erase(s);
    
            } else {
                signs_minus.insert(s);
    
            }
        } // end for
        
        
        if(stabs.row_z[s].count(qubit)) {
        
            if(stabs.row_x[s].count(qubit) == 0) {
                if (signs_minus.count(s)) {
                    signs_minus.erase(s);
        
                } else {
                    signs_minus.insert(s);
        
                }
            }
        } // end for
        
        
        if(stabs.row_z[s].count(qubit)) {
            
            // Z -> i
            // signs_i ^= stabs.col_x[qubit]
            // plus: i * i = -1
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
        
    
    
    
    
        // F2_gen_mod
        if(stabs.row_z[s].count(qubit)) {
            
            if (stabs.row_x[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
            } else {
                stabs.row_x[s].insert(qubit);
            }
        } else {
        
            if(stabs.row_x[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
                stabs.row_z[s].insert(qubit);
            }
        
        }   
        
    
    
        if(destabs.row_z[s].count(qubit)) {
            
            if (destabs.row_x[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
            } else {
                destabs.row_x[s].insert(qubit);
            }
        } else {
        
            if(destabs.row_x[s].count(qubit)) {
    
                destabs.row_x[s].erase(qubit);
                destabs.row_z[s].insert(qubit);
            }
        
        }     
    
    
    
    }

    
}

void State::F4d(const int_num& qubit) {
    for (int_num s = 0; s < num_qubits; s++){
    






        // X not Z -> -1
        // -------------------
        if(stabs.row_x[s].count(qubit)) {
            if(signs_minus.count(s)) {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.erase(s);
                }
            } else {
                if(stabs.row_z[s].count(qubit) == 0) {
                    signs_minus.insert(s);
                }
            }    
        }
        
        // X -> i
        // signs_i ^= stabs.col_x[qubit]
        // plus: i * i = -1
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







        // F1_gen_mod
        if(stabs.row_x[s].count(qubit)) {
            
            if (stabs.row_z[s].count(qubit)) {
                stabs.row_x[s].erase(qubit);
            } else {
                stabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(stabs.row_z[s].count(qubit)) {
                stabs.row_z[s].erase(qubit);
                stabs.row_x[s].insert(qubit);
            }
        
        }   
        
    
    
        if(destabs.row_x[s].count(qubit)) {
            
            if (destabs.row_z[s].count(qubit)) {
                destabs.row_x[s].erase(qubit);
            } else {
                destabs.row_z[s].insert(qubit);
            }
        } else {
        
            if(destabs.row_z[s].count(qubit)) {
                destabs.row_z[s].erase(qubit);
                destabs.row_x[s].insert(qubit);
            }
        
        }    

    
    
    }
    
}
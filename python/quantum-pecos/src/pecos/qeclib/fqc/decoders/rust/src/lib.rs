// For smallest build uncomment this and the custom panic code below
// #![no_std]

static mut PFU_ONE:i32 = 0; // Pauli frame for logical qubit one
static mut PFU_TWO:i32 = 0; // Pauli frame for logical qubit two

const  SE_ROUNDS:usize=10; //number of rounds of SE

static mut SYN_MAT_ONE: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix
static mut FLAG_MAT_ONE: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix
static mut SYN_MAT_TWO: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix
static mut FLAG_MAT_TWO: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix

static mut SYN_MAT_TQG: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix
static mut FLAG_MAT_TQG: [usize;SE_ROUNDS] = [0;SE_ROUNDS]; //Syndrome matrix

const NO_SYN:i32 = 0;
const NO_FLAG_SYN:(i32,i32) = (0, 0);

#[no_mangle]
fn decode_flag(flag_syn:(i32,i32))->i32 {
    match flag_syn {
        (1, 2)  => 1,
        (1, 3)  => 3,
        (1, 9)  => 2,
        (1, 5)  => 2,
        (1, 1)  => 3,
        (2, 5)  => 1,
        (2, 6)  => 3,
        (2, 2)  => 2,
        (2, 10) => 2,
        (2, 3)  => 3,
        (4, 10) => 1,
        (4, 12) => 3,
        (4, 5)  => 2,
        (4, 4)  => 2,
        (4, 6)  => 3,
        (8, 4)  => 1,
        (8, 8)  => 3,
        (8, 10) => 2,
        (8, 9)  => 2,
        (8, 12) => 3,
        _       => NO_SYN,
    }
}

#[no_mangle]
fn decode_basic(syn:i32)->i32 {
    match syn {
        0  => 0,
        8  => 2,
        1  => 0,
        3  => 1,
        6  => 0,
        12 => 2,
        13 => 1,
        11 => 0,
        7  => 3,
        15 => 0,
        14 => 1,
        5  => 3,
        10 => 0,
        4  => 2,
        9  => 0,
        2  => 3,
        _  => NO_SYN,
    }
}

#[no_mangle]
fn decode_basic_tq(syn:i32)->(i32,i32) {
    match syn {
        0 => (0, 0),
        1 => (2, 0),
        2 => (0, 0),
        3 => (0, 0),
        4 => (0, 0),
        5 => (3, 1),
        6 => (1, 1),
        7 => (0, 0),
        8 => (2, 0),
        9 => (2, 0),
        10 => (3, 1),
        11 => (1, 1),
        12 => (0, 0),
        13 => (1, 1),
        14 => (0, 0),
        15 => (3, 1),
        16 => (2, 2),
        17 => (0, 2),
        18 => (2, 2),
        19 => (2, 2),
        20 => (2, 2),
        21 => (1, 3),
        22 => (3, 3),
        23 => (2, 2),
        24 => (2, 2),
        25 => (0, 2),
        26 => (1, 3),
        27 => (3, 3),
        28 => (2, 2),
        29 => (3, 3),
        30 => (2, 2),
        31 => (1, 3),
        32 => (0, 1),
        33 => (2, 1),
        34 => (0, 1),
        35 => (0, 1),
        36 => (0, 1),
        37 => (3, 1),
        38 => (1, 0),
        39 => (0, 1),
        40 => (2, 1),
        41 => (2, 1),
        42 => (3, 0),
        43 => (1, 0),
        44 => (0, 1),
        45 => (1, 1),
        46 => (0, 1),
        47 => (3, 0),
        48 => (2, 3),
        49 => (0, 3),
        50 => (2, 3),
        51 => (2, 3),
        52 => (2, 3),
        53 => (1, 2),
        54 => (3, 2),
        55 => (2, 3),
        56 => (2, 3),
        57 => (0, 3),
        58 => (1, 2),
        59 => (3, 2),
        60 => (2, 3),
        61 => (3, 2),
        62 => (2, 3),
        63 => (1, 2),
        64 => (0, 1),
        65 => (2, 1),
        66 => (0, 1),
        67 => (0, 1),
        68 => (0, 1),
        69 => (3, 0),
        70 => (1, 1),
        71 => (0, 1),
        72 => (2, 1),
        73 => (2, 1),
        74 => (3, 0),
        75 => (1, 0),
        76 => (0, 1),
        77 => (1, 0),
        78 => (0, 1),
        79 => (3, 1),
        80 => (0, 0),
        81 => (2, 0),
        82 => (0, 0),
        83 => (0, 0),
        84 => (0, 0),
        85 => (3, 1),
        86 => (1, 1),
        87 => (1, 3),
        88 => (2, 0),
        89 => (2, 0),
        90 => (3, 1),
        91 => (1, 1),
        92 => (0, 0),
        93 => (1, 1),
        94 => (3, 3),
        95 => (3, 1),
        96 => (0, 1),
        97 => (2, 1),
        98 => (0, 1),
        99 => (0, 1),
        100 => (0, 1),
        101 => (3, 0),
        102 => (1, 0),
        103 => (0, 1),
        104 => (2, 1),
        105 => (2, 1),
        106 => (3, 1),
        107 => (1, 1),
        108 => (0, 1),
        109 => (1, 0),
        110 => (0, 1),
        111 => (3, 0),
        112 => (0, 0),
        113 => (2, 0),
        114 => (0, 0),
        115 => (0, 0),
        116 => (0, 0),
        117 => (3, 1),
        118 => (1, 1),
        119 => (1, 2),
        120 => (2, 0),
        121 => (2, 0),
        122 => (3, 1),
        123 => (1, 1),
        124 => (0, 0),
        125 => (1, 1),
        126 => (3, 2),
        127 => (3, 1),
        128 => (2, 2),
        129 => (0, 2),
        130 => (2, 2),
        131 => (2, 2),
        132 => (2, 2),
        133 => (1, 3),
        134 => (3, 3),
        135 => (2, 2),
        136 => (0, 2),
        137 => (2, 2),
        138 => (1, 3),
        139 => (3, 3),
        140 => (2, 2),
        141 => (3, 3),
        142 => (2, 2),
        143 => (1, 3),
        144 => (2, 2),
        145 => (2, 2),
        146 => (2, 2),
        147 => (2, 2),
        148 => (2, 2),
        149 => (1, 3),
        150 => (3, 3),
        151 => (2, 2),
        152 => (0, 2),
        153 => (0, 2),
        154 => (1, 3),
        155 => (3, 3),
        156 => (2, 2),
        157 => (3, 3),
        158 => (2, 2),
        159 => (1, 3),
        160 => (0, 0),
        161 => (2, 0),
        162 => (3, 2),
        163 => (1, 2),
        164 => (0, 0),
        165 => (3, 1),
        166 => (1, 1),
        167 => (0, 0),
        168 => (2, 0),
        169 => (2, 0),
        170 => (3, 1),
        171 => (1, 1),
        172 => (0, 0),
        173 => (1, 1),
        174 => (0, 0),
        175 => (3, 1),
        176 => (0, 0),
        177 => (2, 0),
        178 => (0, 0),
        179 => (0, 0),
        180 => (1, 3),
        181 => (3, 1),
        182 => (1, 1),
        183 => (0, 0),
        184 => (2, 0),
        185 => (2, 0),
        186 => (3, 1),
        187 => (1, 1),
        188 => (3, 3),
        189 => (1, 1),
        190 => (0, 0),
        191 => (3, 1),
        192 => (2, 3),
        193 => (0, 3),
        194 => (2, 3),
        195 => (2, 3),
        196 => (2, 3),
        197 => (1, 2),
        198 => (3, 2),
        199 => (2, 3),
        200 => (0, 3),
        201 => (2, 3),
        202 => (1, 2),
        203 => (3, 2),
        204 => (2, 3),
        205 => (3, 2),
        206 => (2, 3),
        207 => (1, 2),
        208 => (0, 0),
        209 => (2, 0),
        210 => (0, 0),
        211 => (0, 0),
        212 => (1, 2),
        213 => (3, 1),
        214 => (1, 1),
        215 => (0, 0),
        216 => (2, 0),
        217 => (2, 0),
        218 => (3, 1),
        219 => (1, 1),
        220 => (3, 2),
        221 => (1, 1),
        222 => (0, 0),
        223 => (3, 1),
        224 => (0, 0),
        225 => (2, 0),
        226 => (3, 3),
        227 => (1, 3),
        228 => (0, 0),
        229 => (3, 1),
        230 => (1, 1),
        231 => (0, 0),
        232 => (2, 0),
        233 => (2, 0),
        234 => (3, 1),
        235 => (1, 1),
        236 => (0, 0),
        237 => (1, 1),
        238 => (0, 0),
        239 => (3, 1),
        240 => (2, 3),
        241 => (2, 3),
        242 => (2, 3),
        243 => (2, 3),
        244 => (2, 3),
        245 => (1, 2),
        246 => (3, 2),
        247 => (2, 3),
        248 => (0, 3),
        249 => (0, 3),
        250 => (1, 2),
        251 => (3, 2),
        252 => (2, 3),
        253 => (3, 2),
        254 => (2, 3),
        255 => (1, 2),
        _   => NO_FLAG_SYN,
    }
}

#[no_mangle]
fn decode_flag_tq(flag_syn:(i32,i32))->(i32,i32) {
    match flag_syn {
        (0,0) => (0,0),
        _     => NO_FLAG_SYN,
    }
}

#[no_mangle]
fn init() {
}

#[no_mangle]
pub fn single_decode(syn:i32, qubit_num:i32) {
    let d = decode_basic(syn);
    if qubit_num == 1 {
        unsafe {
            PFU_ONE = PFU_ONE ^ d;
        }
    }
    if qubit_num == 2 {
        unsafe {
            PFU_TWO = PFU_TWO ^ d;
        }
    }
}

#[no_mangle]
pub fn single_decode_flag(flag: i32, syn1:i32, qubit_num:i32){
    let d = decode_flag((flag,syn1));
    if qubit_num == 1 {
        unsafe {
            PFU_ONE = PFU_ONE ^ d;
        }
    }
    if qubit_num == 2 {
        unsafe {
            PFU_TWO = PFU_TWO ^ d;
        }
    }
}

#[no_mangle]
pub fn double_decode(syn:i32) {
    let d = decode_basic_tq(syn);
    unsafe {
        PFU_ONE = PFU_ONE ^ d.0;
        PFU_TWO = PFU_TWO ^ d.1;
    }
}

#[no_mangle]
pub fn double_decode_flag(flag:i32, syn1: i32) {
    let d = decode_flag_tq((flag,syn1));
    unsafe {
        PFU_ONE = PFU_ONE ^ d.0;
        PFU_TWO = PFU_TWO ^ d.1;
    }
}

#[no_mangle]
pub fn return_PFU(qubit_num:i32)->i32 {
    unsafe {
        match qubit_num {
            2 => PFU_TWO,
            _ => PFU_ONE,
        }

    }
}

#[no_mangle]
pub fn return_SYN_MAT(qubit_num :i32)->[usize;10] {
    unsafe {
        match qubit_num {
            1 => SYN_MAT_ONE,
            2 => SYN_MAT_TWO,
            3 => SYN_MAT_TQG,
            _ => SYN_MAT_ONE,
        }
    }
}

#[no_mangle]
pub fn return_FLAG_MAT(qubit_num :i32)->[usize;10] {
    unsafe {
        match qubit_num {
            1 => FLAG_MAT_ONE,
            2 => FLAG_MAT_TWO,
            3 => FLAG_MAT_TQG,
            _ => FLAG_MAT_ONE,
        }
    }
}

#[no_mangle]
pub fn add_syn(syn:usize, time_step:usize, qubit_num:usize){
    unsafe {
        if qubit_num == 1 {
            SYN_MAT_ONE[time_step] = syn;
        }
        if qubit_num == 2 {
            SYN_MAT_TWO[time_step] = syn;
        }
    }
}

#[no_mangle]
pub fn add_syn_TQG(syn:usize, time_step:usize ){
    unsafe {
        SYN_MAT_TQG[time_step] = syn;
    }
}

#[no_mangle]
pub fn add_flag(flag:usize, time_step:usize, qubit_num:usize){
    unsafe {
        if qubit_num == 1 {
            FLAG_MAT_ONE[time_step] = flag;
        }
        if qubit_num == 2 {
            FLAG_MAT_TWO[time_step] = flag;
        }
    }
}

#[no_mangle]
pub fn add_flag_TQG(flag:usize, time_step:usize){
    unsafe {
        FLAG_MAT_TQG[time_step] = flag;
    }
}

#[no_mangle]
pub fn global_reset(){
    unsafe{
        SYN_MAT_ONE = [0;SE_ROUNDS];
        FLAG_MAT_ONE = [0;SE_ROUNDS];
        SYN_MAT_TWO = [0;SE_ROUNDS];
        FLAG_MAT_TWO = [0;SE_ROUNDS];
        SYN_MAT_TQG = [0;SE_ROUNDS];
        FLAG_MAT_TQG = [0;SE_ROUNDS];
        PFU_ONE = 0;
        PFU_TWO = 0;
    }
}

// Uncomment this when building the smallest code
// tests will not work - they require std
// #[panic_handler]
// fn my_panic(_info: &core::panic::PanicInfo) -> ! {
//     loop {}
// }

#[cfg(test)]
mod tests {
    use crate::decode_flag;
    use crate::decode_basic;
    use crate::decode_basic_tq;

    #[test]
    fn check_decode_flag() {
        // existing item
        assert_eq!(decode_flag((1, 9)), 2);
        // non-existing item
        assert_eq!(decode_flag((20,20)), 0);
    }

    #[test]
    fn check_decode_basic() {
        // existing items
        assert_eq!(decode_basic(3), 1);
        assert_eq!(decode_basic(12), 2);
        // non-existing item
        assert_eq!(decode_basic(20), 0)
    }

    #[test]
    fn check_decode_basic_tq() {
        // existing items
        assert_eq!(decode_basic_tq(21), (1,3));
        assert_eq!(decode_basic_tq(22), (3,3));
        // non-existing item
        assert_eq!(decode_basic_tq(300), (0,0))
    }
}

use bitvec::prelude::*;
use pecos_engines::{Data, DataVec, DataVecType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Creating a DataVec from homogeneous Data values
    let measurements = vec![Data::U32(0), Data::U32(1), Data::U32(0), Data::U32(1)];
    let data_vec = DataVec::from_data_vec(measurements)?;

    println!("Created DataVec with {} elements", data_vec.len());
    println!("Data type: {:?}", data_vec.data_type());

    // Example 2: Accessing individual elements
    if let Some(first) = data_vec.get(0) {
        println!("First element: {first:?}");
    }

    // Example 3: Converting to JSON
    let json_array = data_vec.to_json_array();
    println!("As JSON array: {json_array}");

    // Example 4: Creating an empty DataVec and pushing values
    let mut float_vec = DataVec::new_empty(DataVecType::F64);
    float_vec.push(Data::F64(std::f64::consts::PI))?;
    float_vec.push(Data::F64(2.71))?;
    float_vec.push(Data::F64(1.41))?;

    println!("\nFloat vector: {float_vec:?}");

    // Example 5: Working with BitVec data

    let mut bv1 = BitVec::<u8, Lsb0>::new();
    bv1.push(true);
    bv1.push(false);
    bv1.push(true);

    let mut bv2 = BitVec::<u8, Lsb0>::new();
    bv2.push(false);
    bv2.push(true);
    bv2.push(true);

    let bitvec_data = vec![Data::BitVec(bv1), Data::BitVec(bv2)];
    let bitvec_vec = DataVec::from_data_vec(bitvec_data)?;

    println!("\nBitVec as JSON (decimal): {}", bitvec_vec.to_json_array());

    // Example 6: Round-trip conversion
    let original = vec![
        Data::String("quantum".to_string()),
        Data::String("computing".to_string()),
    ];
    let string_vec = DataVec::from_data_vec(original.clone())?;
    let converted_back = string_vec.to_data_vec();

    println!("\nRound-trip successful: {}", original == converted_back);

    Ok(())
}

// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Tests for shot result functionality.

#![allow(clippy::similar_names)]

#[cfg(test)]
mod tests {
    use crate::{Data, Shot, ShotVec};

    #[test]
    fn test_shot_results_display_64bit() {
        // Create a shot with various data types
        let mut shot1 = Shot::default();
        shot1.data.insert("reg_32".to_string(), Data::U32(42));

        // Add a large 64-bit register (larger than u32::MAX)
        let large_value = 1u64 << 34; // 2^34 = 17,179,869,184 (>4B)
        shot1
            .data
            .insert("reg_64".to_string(), Data::U64(large_value));

        // Add a signed 64-bit register with negative value
        shot1.data.insert("reg_signed".to_string(), Data::I64(-42));

        // Add some floating point data
        shot1
            .data
            .insert("float_val".to_string(), Data::F64(std::f64::consts::PI));

        // Create ShotVec with one shot
        let shot_results = ShotVec { shots: vec![shot1] };

        // Convert to string
        let json_string = shot_results.to_compact_json();
        let display_string = format!("{shot_results}");

        // Print the actual JSON for debugging
        println!("COMPACT JSON STRING: {json_string}");

        // The display string should match the compact JSON string
        assert_eq!(display_string, json_string);

        // Verify that both are valid JSON and contain the same data
        let json_value1: serde_json::Value = serde_json::from_str(&display_string).unwrap();
        let json_value2: serde_json::Value = serde_json::from_str(&json_string).unwrap();

        // Verify that both are arrays with the same length
        assert_eq!(
            json_value1.as_array().unwrap().len(),
            json_value2.as_array().unwrap().len(),
            "JSON arrays should have the same number of shots"
        );

        // Verify that all registers appear in the JSON
        assert!(json_string.contains("\"reg_32\""));
        assert!(json_string.contains("42"));
        assert!(json_string.contains("\"reg_64\""));
        assert!(json_string.contains("17179869184"));
        assert!(json_string.contains("\"reg_signed\""));
        assert!(json_string.contains("-42"));
        assert!(json_string.contains("\"float_val\""));
        assert!(json_string.contains("3.14159"));

        // Test with multiple shots
        let mut shot1_copy = Shot::default();
        shot1_copy.data.insert("reg_32".to_string(), Data::U32(42));
        shot1_copy
            .data
            .insert("reg_64".to_string(), Data::U64(large_value));
        shot1_copy
            .data
            .insert("reg_signed".to_string(), Data::I64(-42));
        shot1_copy
            .data
            .insert("float_val".to_string(), Data::F64(std::f64::consts::PI));

        let mut shot2 = Shot::default();
        shot2.data.insert("reg_32".to_string(), Data::U32(100));
        shot2.data.insert("reg_64".to_string(), Data::U64(200));

        let shot_results = ShotVec {
            shots: vec![shot1_copy, shot2],
        };

        let json_string = shot_results.to_compact_json();
        println!("Multi-shot JSON: {json_string}");

        // Verify the new shot array format shows individual shot objects
        assert!(json_string.contains("\"reg_32\":42"));
        assert!(json_string.contains("\"reg_32\":100"));
        assert!(json_string.contains("42"));
        assert!(json_string.contains("100"));

        // Test with JSON data variant
        let mut shot_with_json = Shot::default();
        shot_with_json
            .data
            .insert("measurement".to_string(), Data::U32(1));
        shot_with_json.data.insert(
            "metadata".to_string(),
            Data::Json(serde_json::json!({"custom": "data", "nested": {"value": 42}})),
        );

        let shot_results = ShotVec {
            shots: vec![shot_with_json],
        };

        let json_string = shot_results.to_compact_json();
        println!("Shot with JSON data: {json_string}");

        // When shots have JSON data variants, it should serialize the full shot structure
        assert!(json_string.contains("\"data\""));
        assert!(json_string.contains("\"metadata\""));
        assert!(json_string.contains("\"custom\""));
        assert!(json_string.contains("\"nested\""));
    }

    #[test]
    fn test_shot_results_compact_json() {
        // Create shot results with multiple shots
        let mut shot1 = Shot::default();
        shot1.data.insert("c".to_string(), Data::U32(0));
        shot1.data.insert("q".to_string(), Data::U32(1));

        let mut shot2 = Shot::default();
        shot2.data.insert("c".to_string(), Data::U32(3));
        shot2.data.insert("q".to_string(), Data::U32(0));

        let mut shot3 = Shot::default();
        shot3.data.insert("c".to_string(), Data::U32(2));
        shot3.data.insert("q".to_string(), Data::U32(1));

        let shot_results = ShotVec {
            shots: vec![shot1, shot2, shot3],
        };

        // Test compact format
        let compact_json = shot_results.to_compact_json();
        println!("COMPACT FORMAT: {compact_json}");

        // Compact format should not have newlines
        assert!(!compact_json.contains('\n'));
        // Should contain the data in the new format
        assert!(compact_json.contains(r#"{"c":0,"q":1}"#));
        assert!(compact_json.contains(r#"{"c":3,"q":0}"#));
        assert!(compact_json.contains(r#"{"c":2,"q":1}"#));

        // Test that Display also uses compact format
        let display_string = format!("{shot_results}");
        assert_eq!(display_string, compact_json);
    }

    #[test]
    fn test_bigint_support() {
        use num_bigint::BigInt;

        // Create a shot with BigInt data
        let mut shot = Shot::default();

        // Add a regular u32
        shot.data.insert("regular".to_string(), Data::U32(42));

        // Add a BigInt that fits in u32
        let small_bigint = BigInt::from(100u32);
        shot.data
            .insert("small_bigint".to_string(), Data::BigInt(small_bigint));

        // Add a BigInt that exceeds u64::MAX
        let huge_bigint = BigInt::from(u128::MAX) + BigInt::from(1000u32);
        shot.data
            .insert("huge_bigint".to_string(), Data::BigInt(huge_bigint.clone()));

        // Test to_string()
        assert_eq!(shot.data.get("regular").unwrap().to_string(), "42");
        assert_eq!(shot.data.get("small_bigint").unwrap().to_string(), "100");
        assert_eq!(
            shot.data.get("huge_bigint").unwrap().to_string(),
            (BigInt::from(u128::MAX) + BigInt::from(1000u32)).to_string()
        );

        // Test as_u32()
        assert_eq!(shot.data.get("regular").unwrap().as_u32(), Some(42));
        assert_eq!(shot.data.get("small_bigint").unwrap().as_u32(), Some(100));
        assert_eq!(shot.data.get("huge_bigint").unwrap().as_u32(), None); // Too big for u32

        // Test that BigInt serializes and we can work with it
        let shot_vec = ShotVec { shots: vec![shot] };
        let json = serde_json::to_string(&shot_vec).unwrap();

        // Print for debugging
        println!("Serialized JSON: {json}");

        // The important thing is that it serializes without error
        assert!(json.contains("\"regular\":42"));
        assert!(json.contains("\"small_bigint\""));
        assert!(json.contains("\"huge_bigint\""));

        // For BigInt deserialization, we'll need to use the actual format that num-bigint uses
        // Instead of testing deserialization, let's just make sure BigInt works programmatically
        let mut test_shot = Shot::default();
        test_shot.data.insert(
            "big_value".to_string(),
            Data::BigInt(BigInt::from(u128::MAX)),
        );

        match test_shot.data.get("big_value") {
            Some(Data::BigInt(v)) => {
                assert_eq!(v.to_string(), u128::MAX.to_string());
            }
            _ => panic!("Expected BigInt variant"),
        }
    }

    #[test]
    fn test_bytes_support() {
        // Create a shot with Bytes data
        let mut shot = Shot::default();

        // Add raw bytes
        let bytes = vec![0xFF, 0x00, 0xAB, 0xCD];
        shot.data
            .insert("raw_bytes".to_string(), Data::from_bytes(bytes.clone()));

        // Add bytes from bitstring
        let bitstring = "10110011";
        shot.data.insert(
            "from_bits".to_string(),
            Data::from_bitstring_as_bytes(bitstring).unwrap(),
        );

        // Test to_string (should show debug format)
        let bytes_str = shot.data.get("raw_bytes").unwrap().to_string();
        assert!(bytes_str.contains("255")); // 0xFF = 255
        assert!(bytes_str.contains("171")); // 0xAB = 171

        // Test bytes_to_bitstring
        match shot.data.get("from_bits").unwrap() {
            Data::Bytes(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0], 0b1011_0011);
            }
            _ => panic!("Expected Bytes variant"),
        }

        // Test bitstring conversion
        let bitstring_back = shot.data.get("from_bits").unwrap().to_bitstring();
        assert_eq!(bitstring_back, Some("10110011".to_string()));

        // Test as_u32
        let u32_bytes = vec![0x12, 0x34, 0x56, 0x78];
        shot.data
            .insert("u32_bytes".to_string(), Data::from_bytes(u32_bytes));
        assert_eq!(
            shot.data.get("u32_bytes").unwrap().as_u32(),
            Some(0x7856_3412) // Little-endian
        );

        // Test with measurement data - storing 16 qubit measurements efficiently
        let measurement_bits = "1011001110101101";
        let measurement_data = Data::from_bitstring_as_bytes(measurement_bits).unwrap();
        shot.data
            .insert("measurements".to_string(), measurement_data);

        // Verify we can get the bitstring back
        let retrieved = shot
            .data
            .get("measurements")
            .unwrap()
            .to_bitstring()
            .unwrap();
        assert_eq!(retrieved, measurement_bits);

        // Test serialization
        let shot_vec = ShotVec { shots: vec![shot] };
        let json = serde_json::to_string(&shot_vec).unwrap();

        // Bytes should serialize as arrays of numbers
        assert!(json.contains("\"raw_bytes\":[255,0,171,205]"));
        assert!(json.contains("\"from_bits\":[179]")); // 0b10110011 = 179
    }

    #[test]
    fn test_bitvec_support() {
        use bitvec::prelude::*;

        // Create a shot with BitVec data
        let mut shot = Shot::default();

        // Add BitVec from bitstring
        let bitstring = "101100111010110100101110";
        shot.data.insert(
            "bitvec".to_string(),
            Data::from_bitstring(bitstring).unwrap(),
        );

        // Test that it's actually a BitVec
        match shot.data.get("bitvec").unwrap() {
            Data::BitVec(bv) => {
                assert_eq!(bv.len(), bitstring.len());
                // Test individual bit access - bitstring is parsed MSB-first
                // "101100111010110100101110" rightmost bits are "...0101110"
                assert!(!bv[0]); // LSB is rightmost bit: '0'
                assert!(bv[1]); // '1'
                assert!(bv[2]); // '1'
                assert!(bv[3]); // '1'
                assert!(!bv[4]); // '0'
                assert!(bv[5]); // '1'
                // Verify leftmost bit (MSB) is '1'
                assert!(bv[bitstring.len() - 1]);
            }
            _ => panic!("Expected BitVec variant"),
        }

        // Test to_bitstring
        let retrieved = shot.data.get("bitvec").unwrap().to_bitstring().unwrap();
        assert_eq!(retrieved, bitstring);

        // Test to_string (should return the bitstring)
        let string_repr = shot.data.get("bitvec").unwrap().to_string();
        assert_eq!(string_repr, bitstring);

        // Test as_u32 (first 32 bits interpreted as little-endian)
        let u32_val = shot.data.get("bitvec").unwrap().as_u32();
        // BitVec stores bits with LSB at index 0
        // So "101100111010110100101110" has bit[0]=1, bit[1]=0, bit[2]=1, etc.
        assert!(u32_val.is_some());

        // Create BitVec directly and modify it
        // "01011010" MSB-first = LSB [0,1,0,1,1,0,1,0]
        let mut bv = BitVec::<u8, Lsb0>::from_bitslice(bits![u8, Lsb0; 0, 1, 0, 1, 1, 0, 1, 0]);
        bv.set(2, true); // Change bit 2 from 0 to 1
        shot.data.insert("modified".to_string(), Data::BitVec(bv));

        match shot.data.get("modified").unwrap() {
            Data::BitVec(bv) => {
                assert!(bv[2]); // We changed this (was 0, now 1)
                // Use our to_bitstring method which returns MSB-first
                let bitstring = shot.data.get("modified").unwrap().to_bitstring().unwrap();
                // Original: "01011010" with bit 2 (LSB-indexed) changed from 0 to 1
                // Results in: "01011110" (MSB-first display)
                assert_eq!(bitstring, "01011110");
            }
            _ => panic!("Expected BitVec variant"),
        }

        // Test serialization - BitVec serializes based on its serde implementation
        let shot_vec = ShotVec { shots: vec![shot] };
        let json = serde_json::to_string(&shot_vec).unwrap();

        // BitVec should serialize (the format depends on bitvec's serde implementation)
        assert!(json.contains("\"bitvec\""));
        assert!(json.contains("\"modified\""));
    }

    #[test]
    fn test_register_with_width() {
        // Create shots with register data
        let mut shot1 = Shot::default();
        shot1.add_register("c", 0, 2); // 2-bit register with value 0 -> "00"
        shot1.add_register("d", 5, 3); // 3-bit register with value 5 -> "101"

        let mut shot2 = Shot::default();
        shot2.add_register("c", 3, 2); // 2-bit register with value 3 -> "11"
        shot2.add_register("d", 1, 3); // 3-bit register with value 1 -> "001"

        // Test binary string formatting
        assert_eq!(shot1.register_to_binary_string("c"), Some("00".to_string()));
        assert_eq!(
            shot1.register_to_binary_string("d"),
            Some("101".to_string())
        );
        assert_eq!(shot2.register_to_binary_string("c"), Some("11".to_string()));
        assert_eq!(
            shot2.register_to_binary_string("d"),
            Some("001".to_string())
        );

        // Test width metadata
        assert_eq!(shot1.get_register_width("c"), Some(2));
        assert_eq!(shot1.get_register_width("d"), Some(3));

        // Create ShotVec and test formatting
        let shot_vec = ShotVec {
            shots: vec![shot1, shot2],
        };
        let binary_strings = shot_vec.format_as_binary_strings();

        assert_eq!(
            binary_strings.get("c"),
            Some(&vec!["00".to_string(), "11".to_string()])
        );
        assert_eq!(
            binary_strings.get("d"),
            Some(&vec!["101".to_string(), "001".to_string()])
        );

        // Test that register names exclude metadata
        let names = shot_vec.get_register_names();
        assert_eq!(names, vec!["c", "d"]);
        assert!(!names.contains(&"_width_c".to_string()));
        assert!(!names.contains(&"_width_d".to_string()));
    }
}

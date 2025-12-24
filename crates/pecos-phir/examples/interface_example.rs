//! Example of using MLIR's interface approach for quantum algorithms and patterns
//!
//! This demonstrates how PHIR uses attributes to implement semantic interfaces,
//! allowing operations and regions to declare which protocols they implement.

use pecos_phir::{
    attributes::{AttributeBuilder, helpers},
    ops::{Operation, QuantumOp},
    phir::{Function, Instruction, Module, Region},
    region_kinds::RegionKind,
    types::{FunctionType, Type},
};

// Example-specific interface tags
mod tags {
    pub const QFT: &str = "qft";
    pub const SYNDROME_EXTRACTION: &str = "syndrome_extraction";
}

#[allow(clippy::too_many_lines)] // Example demonstrating comprehensive interface usage
fn main() {
    // Example 1: QFT circuit implementing the QFT interface
    println!("=== Interface Example: QFT Circuit ===\n");

    let _module = Module::new("qft_example");

    // Create a function implementing QFT
    let qft_signature = FunctionType {
        inputs: vec![Type::Array(
            Box::new(Type::Qubit),
            pecos_phir::types::ArraySize::Fixed(4),
        )],
        outputs: vec![Type::Array(
            Box::new(Type::Qubit),
            pecos_phir::types::ArraySize::Fixed(4),
        )],
        variadic: false,
    };
    let _qft_func = Function::new_with_visibility(
        "quantum_fourier_transform",
        qft_signature,
        pecos_phir::phir::Visibility::Public,
    );

    // Create a region for the QFT algorithm
    let mut qft_region = Region::new(RegionKind::SSACFG);

    // Attach interface attributes to indicate this region implements QFT
    qft_region.attributes = AttributeBuilder::new()
        .with_tag(tags::QFT)
        .with_algorithm("quantum_fourier_transform")
        .with_attr("num_qubits", pecos_phir::phir::AttributeValue::Int(4))
        .with_attr("circuit_depth", pecos_phir::phir::AttributeValue::Int(16))
        .parallelizable()
        .build();

    println!("QFT Region Interface Attributes:");
    for (key, value) in &qft_region.attributes {
        println!("  {key}: {value:?}");
    }

    // Example 2: Syndrome extraction implementing QEC protocol interface
    println!("\n=== Interface Example: Syndrome Extraction ===\n");

    let mut syndrome_region = Region::new(RegionKind::SSACFG);

    // Attach interface attributes for syndrome extraction protocol
    syndrome_region.attributes = AttributeBuilder::new()
        .with_tag(tags::SYNDROME_EXTRACTION)
        .with_interface(
            vec![
                "data_qubits[5]".to_string(),
                "ancilla_qubits[4]".to_string(),
            ],
            vec!["syndrome_bits[4]".to_string()],
        )
        .with_attr(
            "stabilizer_type",
            pecos_phir::phir::AttributeValue::String("X".to_string()),
        )
        .with_attr(
            "measurement_order",
            pecos_phir::phir::AttributeValue::String("sequential".to_string()),
        )
        .build();

    println!("Syndrome Extraction Interface Attributes:");
    for (key, value) in &syndrome_region.attributes {
        println!("  {key}: {value:?}");
    }

    // Example 3: Optimization pass recognizing interface implementations
    println!("\n=== Interface Recognition Example ===\n");

    // Simulate an optimization pass checking interface implementations
    let regions = vec![
        ("QFT Region", &qft_region),
        ("Syndrome Region", &syndrome_region),
    ];

    for (name, region) in regions {
        println!("Analyzing {name}");

        // Check interface tags
        if helpers::has_tag(&region.attributes, tags::QFT) {
            println!("  Found QFT interface - can apply QFT-specific optimizations");
            if helpers::is_parallelizable(&region.attributes) {
                println!("  Marked as parallelizable - can distribute phase rotations");
            }
        }

        if helpers::has_tag(&region.attributes, tags::SYNDROME_EXTRACTION) {
            println!("  Found syndrome extraction interface - can optimize for fault tolerance");
            if let Some(stab_type) = region.attributes.get("stabilizer_type") {
                println!("  Stabilizer type: {stab_type:?}");
            }
        }

        // Check interface declarations
        if let Some(inputs) = region.attributes.get("input_interface") {
            println!("  → Input interface: {inputs:?}");
        }
        if let Some(outputs) = region.attributes.get("output_interface") {
            println!("  → Output interface: {outputs:?}");
        }
    }

    // Example 4: Operations implementing specific interfaces
    println!("\n=== Operation Interface Implementation ===\n");

    let mut magic_state_prep = Instruction::new(
        Operation::Quantum(QuantumOp::InitState(vec![])),
        vec![],
        vec![],
        vec![Type::Qubit],
    );

    // Tag operation as implementing magic state preparation interface
    magic_state_prep.attributes = AttributeBuilder::new()
        .with_tag("magic_state_preparation")
        .with_attr(
            "state_type",
            pecos_phir::phir::AttributeValue::String("T".to_string()),
        )
        .with_attr(
            "fidelity_required",
            pecos_phir::phir::AttributeValue::Float(0.999),
        )
        .build();

    println!("Magic State Preparation Interface Attributes:");
    for (key, value) in &magic_state_prep.attributes {
        println!("  {key}: {value:?}");
    }

    // Example 5: Nested interfaces - algorithms containing sub-protocols
    println!("\n=== Nested Interface Example ===\n");

    let mut shor_region = Region::new(RegionKind::SSACFG);
    shor_region.attributes = AttributeBuilder::new()
        .with_tag("shor_algorithm")
        .with_algorithm("integer_factorization")
        .with_attr("contains_qft", pecos_phir::phir::AttributeValue::Bool(true))
        .with_attr(
            "contains_modular_exp",
            pecos_phir::phir::AttributeValue::Bool(true),
        )
        .build();

    println!("Shor's Algorithm Interface:");
    println!(
        "  - Contains QFT interface: {:?}",
        shor_region.attributes.get("contains_qft")
    );
    println!(
        "  - Contains Modular Exp interface: {:?}",
        shor_region.attributes.get("contains_modular_exp")
    );
    println!(
        "\nOptimization passes can recognize nested interfaces and optimize each sub-protocol!"
    );
}

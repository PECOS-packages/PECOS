use pecos_phir_json::{
    PhirJsonEngineBuilder, PhirJsonEngineProgram, phir_json_to_module,
    v0_1::{
        EnhancedV0_1, V0_1,
        ast::{Operation, PHIRProgram},
        block_executor::BlockExecutor,
        block_iterative_executor::BlockIterativeExecutor,
        classical_interpreter::PhirClassicalInterpreter,
        engine::PhirJsonEngine,
        operations::OperationProcessor,
    },
    version_traits::PhirImplementation,
};
use serde_json::{Value, json};

type Entry = (&'static str, fn(&str, &[Operation]) -> Result<(), String>);
const ENTRIES: &[Entry] = &[
    ("file RON converter", |input, _| {
        with_file(input, |path| {
            pecos_phir_json::convert_phir_json_file_to_ron(path).map(|_| ())
        })
    }),
    ("environment", |_, ops| {
        let mut env = pecos_phir_json::v0_1::environment::Environment::new();
        ops.iter().try_for_each(|op| {
            if let Operation::VariableDefinition {
                data_type,
                variable,
                size,
                ..
            } = op
            {
                env.add_variable(variable, data_type.parse().unwrap(), size.unwrap())
                    .map_err(|e| e.to_string())
            } else {
                Ok(())
            }
        })
    }),
    ("processor add helpers", |_, ops| {
        let mut processor = OperationProcessor::new();
        ops.iter().try_for_each(|op| {
            if let Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } = op
            {
                if data == "qvar_define" {
                    processor.add_quantum_variable(variable, size.unwrap())
                } else {
                    processor.add_classical_variable(variable, data_type, size.unwrap())
                }
                .map_err(|e| e.to_string())
            } else {
                Ok(())
            }
        })
    }),
    ("block sequence", |_, ops| {
        BlockExecutor::new()
            .execute_sequence(ops)
            .map_err(|e| e.to_string())
    }),
    ("RON converter", |input, _| {
        pecos_phir_json::phir_json_to_ron(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("file converter", |input, _| {
        with_file(input, |path| {
            pecos_phir_json::convert_phir_json_file_to_module(path).map(|_| ())
        })
    }),
    ("file engine", |input, _| {
        with_file(input, |path| PhirJsonEngine::new(path).map(|_| ()))
    }),
    ("automatic setup", |input, _| {
        with_file(input, |path| {
            pecos_phir_json::setup_phir_json_engine(path).map(|_| ())
        })
    }),
    ("versioned setup", |input, _| {
        with_file(input, |path| {
            pecos_phir_json::v0_1::setup_phir_json_v0_1_engine(path).map(|_| ())
        })
    }),
    ("enhanced setup", |input, _| {
        with_file(input, |path| {
            pecos_phir_json::v0_1::setup_enhanced_phir_json_v0_1_engine(path).map(|_| ())
        })
    }),
    #[cfg(feature = "wasm")]
    ("versioned setup with WASM", |input, _| {
        with_wasm(
            input,
            pecos_phir_json::v0_1::setup_phir_json_v0_1_engine_with_wasm,
        )
    }),
    #[cfg(feature = "wasm")]
    ("enhanced setup with WASM", |input, _| {
        with_wasm(
            input,
            pecos_phir_json::v0_1::setup_enhanced_phir_json_v0_1_engine_with_wasm,
        )
    }),
    ("builder JSON", |input, _| {
        PhirJsonEngineBuilder::new()
            .json(input)
            .and_then(pecos_engines::ClassicalControlEngineBuilder::build)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("builder shared program", |input, _| {
        pecos_engines::ClassicalControlEngineBuilder::build(
            PhirJsonEngineBuilder::new().program(pecos_programs::PhirJson::from_json(input)),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }),
    ("converter", |input, _| {
        phir_json_to_module(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("processor", |_, ops| {
        let mut processor = OperationProcessor::new();
        ops.iter().try_for_each(|op| {
            if let Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } = op
            {
                processor
                    .handle_variable_definition(data, data_type, variable, size.unwrap())
                    .map_err(|e| e.to_string())
            } else {
                Ok(())
            }
        })
    }),
    ("engine JSON", |input, _| {
        PhirJsonEngine::from_json(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("engine AST", |_, ops| {
        PhirJsonEngine::from_program(PHIRProgram {
            format: "PHIR/JSON".into(),
            version: "0.1.0".into(),
            metadata: std::collections::BTreeMap::default(),
            ops: ops.to_vec(),
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
    }),
    ("block executor", |_, ops| {
        BlockExecutor::new()
            .execute_program(ops)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("iterative executor", |_, ops| {
        BlockIterativeExecutor::new(&mut BlockExecutor::new())
            .with_operations(ops)
            .process()
            .map_err(|e| e.to_string())
    }),
    ("Rust interpreter", |input, _| {
        PhirClassicalInterpreter::new()
            .init(input, None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("AST parser", |input, _| {
        serde_json::from_str::<PHIRProgram>(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("versioned parser", |input, _| {
        V0_1::parse_program(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("enhanced parser", |input, _| {
        EnhancedV0_1::parse_program(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
    ("builder program", |input, _| {
        PhirJsonEngineProgram::from_json(input)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }),
];
#[cfg(feature = "wasm")]
fn with_wasm(
    input: &str,
    run: impl FnOnce(
        &std::path::Path,
        &std::path::Path,
    ) -> Result<
        Box<dyn pecos_engines::ClassicalControlEngine>,
        pecos_core::errors::PecosError,
    >,
) -> Result<(), String> {
    with_file(input, |path| {
        let wasm = path.with_extension("wasm");
        std::fs::write(&wasm, b"\0asm\x01\0\0\0").unwrap();
        run(path, &wasm).map(|_| ())
    })
}

fn with_file(
    input: &str,
    run: impl FnOnce(&std::path::Path) -> Result<(), pecos_core::errors::PecosError>,
) -> Result<(), String> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("program.phir.json");
    std::fs::write(&path, input).unwrap();
    run(&path).map_err(|e| e.to_string())
}
const SHAPES: &[(&str, &str)] = &[
    ("quantum", "quantum"),
    ("classical", "classical"),
    ("quantum", "classical"),
    ("classical", "quantum"),
];
fn declaration(kind: &str, name: &str) -> Value {
    json!({"data":if kind == "quantum" {"qvar_define"} else {"cvar_define"},
        "variable":name, "data_type":if kind == "quantum" {"qubits"} else {"u32"}, "size":1})
}
#[test]
fn redeclaration_matrix() {
    let mut failures = Vec::new();
    for &(first, second) in SHAPES {
        let values = vec![declaration(first, "a"), declaration(second, "a")];
        let ops: Vec<Operation> = values
            .iter()
            .cloned()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        let input = json!({"format":"PHIR/JSON", "version":"0.1.0", "ops":values}).to_string();
        for &(entry, run) in ENTRIES {
            let result = run(&input, &ops);
            let expected = format!(
                "Variable 'a' is already declared as {first}; cannot redeclare as {second}"
            );
            if !matches!(&result, Err(error) if error.contains(&expected)) {
                failures.push(format!("{first}/{second}: {entry}: {result:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn distinct_names_and_repeated_references() {
    let input = include_str!("fixtures/unique_declarations.phir.json");
    let program: PHIRProgram = serde_json::from_str(input).unwrap();
    for &(entry, run) in ENTRIES {
        run(input, &program.ops).unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}

#[test]
fn unreachable_redeclarations_are_invalid() {
    let values = vec![
        declaration("quantum", "a"),
        json!({"block":"if", "condition":{"cop":"==", "args":[0,1]}, "true_branch":[declaration("classical", "a")]}),
    ];
    let ops: Vec<Operation> = values
        .iter()
        .cloned()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
    let input = json!({"format":"PHIR/JSON", "version":"0.1.0", "ops":values}).to_string();
    let mut failures = Vec::new();
    for &(entry, run) in ENTRIES {
        // These APIs register individual declarations, rather than walking blocks.
        if matches!(entry, "processor" | "processor add helpers" | "environment") {
            continue;
        }
        let result = run(&input, &ops);
        if !matches!(&result, Err(error) if error.contains("Variable 'a' is already declared as quantum; cannot redeclare as classical"))
        {
            failures.push(format!("{entry}: {result:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn classical_declaration_kind_is_independent_of_storage_type() {
    for (first, second) in [
        ("qvar_define", "cvar_define"),
        ("cvar_define", "qvar_define"),
    ] {
        let mut processor = OperationProcessor::new();
        processor
            .handle_variable_definition(first, "qubits", "a", 1)
            .unwrap();
        let error = processor
            .handle_variable_definition(second, "qubits", "a", 1)
            .unwrap_err();
        let first_kind = if first == "qvar_define" {
            "quantum"
        } else {
            "classical"
        };
        let second_kind = if second == "qvar_define" {
            "quantum"
        } else {
            "classical"
        };
        assert_eq!(
            error.to_string(),
            format!(
                "Input error: Variable 'a' is already declared as {first_kind}; cannot redeclare as {second_kind}"
            )
        );
    }
}

#[test]
fn redeclaration_precedes_storage_validation() {
    for &(first, second) in SHAPES {
        let mut processor = OperationProcessor::new();
        let data = |kind| {
            if kind == "quantum" {
                "qvar_define"
            } else {
                "cvar_define"
            }
        };
        let data_type = |kind| if kind == "quantum" { "qubits" } else { "i64" };
        processor
            .handle_variable_definition(data(first), data_type(first), "a", 1)
            .unwrap();
        let error = processor
            .handle_variable_definition(data(second), data_type(second), "a", usize::MAX)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "Input error: Variable 'a' is already declared as {first}; cannot redeclare as {second}"
            )
        );
    }
}

//! Python bindings for Zluppy via PyO3.
//!
//! This module provides Python access to Zluppy's compiler functionality:
//!
//! ```python
//! import zluppy
//!
//! # Compile to SLR-AST (returns dict)
//! ast = zluppy.compile_to_slr("""
//!     fn main() -> void {
//!         var q = qalloc(2);
//!         H(q[0]);
//!         CX(q[0], q[1]);
//!     }
//! """)
//!
//! # Compile to SLR-AST JSON string
//! json_str = zluppy.compile_to_slr_json(source)
//!
//! # Check source for errors
//! zluppy.check(source)  # Raises ZluppyError on failure
//! zluppy.check(source, strict=True)  # NASA Power of 10 mode
//!
//! # Parse and get AST as string (for debugging)
//! ast_str = zluppy.parse_debug(source)
//! ```

use std::path::Path;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use ::zlup::codegen::{HugrCodegen, SlrCodegen};
use ::zlup::semantic::SemanticAnalyzer;

// =============================================================================
// Error Types
// =============================================================================

pyo3::create_exception!(_zluppy, ZluppyError, pyo3::exceptions::PyException);

/// Convert a parse error to a Python exception.
fn parse_error_to_py(e: ::zlup::parser::ParseError) -> PyErr {
    ZluppyError::new_err(format!(
        "Parse error at {}:{}: {}",
        e.location.line, e.location.column, e.message
    ))
}

/// Convert a semantic error to a Python exception.
fn semantic_error_to_py(e: ::zlup::semantic::SemanticError) -> PyErr {
    ZluppyError::new_err(format!("Semantic error: {}", e))
}

/// Convert a codegen error to a Python exception.
fn codegen_error_to_py(e: ::zlup::codegen::slr::SlrError) -> PyErr {
    ZluppyError::new_err(format!("Codegen error: {}", e))
}

/// Convert a HUGR codegen error to a Python exception.
fn hugr_error_to_py(e: ::zlup::codegen::hugr::HugrError) -> PyErr {
    ZluppyError::new_err(format!("HUGR error: {}", e))
}

// =============================================================================
// Core Functions
// =============================================================================

/// Compile Zluppy source to SLR-AST and return as a Python dict.
///
/// Args:
///     source: Zluppy source code as a string
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     dict: SLR-AST as a Python dictionary
///
/// Raises:
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (source, strict = false))]
fn compile_to_slr(py: Python<'_>, source: &str, strict: bool) -> PyResult<Py<PyAny>> {
    // Parse
    let program = ::zlup::parse(source).map_err(parse_error_to_py)?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    // Code generation
    let mut codegen = SlrCodegen::new();
    let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

    // Convert to JSON then to Python dict
    let json_str = codegen.to_json(&slr_program).map_err(codegen_error_to_py)?;

    // Parse JSON into Python object
    let json_module = py.import("json")?;
    let result = json_module.call_method1("loads", (json_str,))?;
    Ok(result.into())
}

/// Compile Zluppy source to SLR-AST JSON string.
///
/// Args:
///     source: Zluppy source code as a string
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///     compact: Return compact JSON (no pretty-printing). Default: False
///
/// Returns:
///     str: SLR-AST as a JSON string
///
/// Raises:
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (source, strict = false, compact = false))]
fn compile_to_slr_json(source: &str, strict: bool, compact: bool) -> PyResult<String> {
    // Parse
    let program = ::zlup::parse(source).map_err(parse_error_to_py)?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    // Code generation
    let mut codegen = SlrCodegen::new();
    let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

    // Convert to JSON
    if compact {
        codegen
            .to_json_compact(&slr_program)
            .map_err(codegen_error_to_py)
    } else {
        codegen.to_json(&slr_program).map_err(codegen_error_to_py)
    }
}

/// Check Zluppy source for errors without compiling.
///
/// Args:
///     source: Zluppy source code as a string
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     None: If the source is valid
///
/// Raises:
///     ZluppyError: If parsing or semantic analysis fails
#[pyfunction]
#[pyo3(signature = (source, strict = false))]
fn check(source: &str, strict: bool) -> PyResult<()> {
    // Parse
    let program = ::zlup::parse(source).map_err(parse_error_to_py)?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    Ok(())
}

/// Parse Zluppy source and return AST as debug string.
///
/// This is primarily for debugging and inspection purposes.
///
/// Args:
///     source: Zluppy source code as a string
///
/// Returns:
///     str: AST in Rust Debug format
///
/// Raises:
///     ZluppyError: If parsing fails
#[pyfunction]
fn parse_debug(source: &str) -> PyResult<String> {
    let program = ::zlup::parse(source).map_err(parse_error_to_py)?;
    Ok(format!("{:#?}", program))
}

/// Get the Zluppy version.
///
/// Returns:
///     str: Version string
#[pyfunction]
fn version() -> &'static str {
    ::zlup::VERSION
}

// =============================================================================
// File-based Functions
// =============================================================================

/// Read and return the contents of a Zluppy source file.
fn read_file(path: &str) -> PyResult<String> {
    std::fs::read_to_string(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to read {}: {}", path, e)))
}

/// Get the filename from a path for error reporting.
fn filename_from_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

/// Compile a Zluppy source file to SLR-AST and return as a Python dict.
///
/// Args:
///     path: Path to a .zlp file
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     dict: SLR-AST as a Python dictionary
///
/// Raises:
///     IOError: If the file cannot be read
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (path, strict = false))]
fn compile_file(py: Python<'_>, path: &str, strict: bool) -> PyResult<Py<PyAny>> {
    let source = read_file(path)?;
    let filename = filename_from_path(path);

    // Parse with filename for better error messages
    let program = ::zlup::parse_file(&source, filename).map_err(parse_error_to_py)?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    // Code generation
    let mut codegen = SlrCodegen::new();
    let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

    // Convert to JSON then to Python dict
    let json_str = codegen.to_json(&slr_program).map_err(codegen_error_to_py)?;

    let json_module = py.import("json")?;
    let result = json_module.call_method1("loads", (json_str,))?;
    Ok(result.into())
}

/// Compile a Zluppy source file to SLR-AST JSON string.
///
/// Args:
///     path: Path to a .zlp file
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///     compact: Return compact JSON (no pretty-printing). Default: False
///
/// Returns:
///     str: SLR-AST as a JSON string
///
/// Raises:
///     IOError: If the file cannot be read
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (path, strict = false, compact = false))]
fn compile_file_json(path: &str, strict: bool, compact: bool) -> PyResult<String> {
    let source = read_file(path)?;
    let filename = filename_from_path(path);

    let program = ::zlup::parse_file(&source, filename).map_err(parse_error_to_py)?;

    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    let mut codegen = SlrCodegen::new();
    let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

    if compact {
        codegen
            .to_json_compact(&slr_program)
            .map_err(codegen_error_to_py)
    } else {
        codegen.to_json(&slr_program).map_err(codegen_error_to_py)
    }
}

/// Check a Zluppy source file for errors without compiling.
///
/// Args:
///     path: Path to a .zlp file
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     None: If the source is valid
///
/// Raises:
///     IOError: If the file cannot be read
///     ZluppyError: If parsing or semantic analysis fails
#[pyfunction]
#[pyo3(signature = (path, strict = false))]
fn check_file(path: &str, strict: bool) -> PyResult<()> {
    let source = read_file(path)?;
    let filename = filename_from_path(path);

    let program = ::zlup::parse_file(&source, filename).map_err(parse_error_to_py)?;

    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    Ok(())
}

// =============================================================================
// HUGR Compilation Functions
// =============================================================================

/// Compile Zluppy source to HUGR bytes.
///
/// The returned bytes can be passed directly to hugr_engine() or sim().
///
/// Args:
///     source: Zluppy source code as a string
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     bytes: HUGR in binary envelope format
///
/// Raises:
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (source, strict = false))]
fn compile_to_hugr(
    py: Python<'_>,
    source: &str,
    strict: bool,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let program = ::zlup::parse(source).map_err(parse_error_to_py)?;

    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    let mut codegen = HugrCodegen::new();
    let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
    let bytes = codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?;

    Ok(pyo3::types::PyBytes::new(py, &bytes).into())
}

/// Compile a Zluppy source file to HUGR bytes.
///
/// The returned bytes can be passed directly to hugr_engine() or sim().
///
/// Args:
///     path: Path to a .zlp file
///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
///
/// Returns:
///     bytes: HUGR in binary envelope format
///
/// Raises:
///     IOError: If the file cannot be read
///     ZluppyError: If parsing, semantic analysis, or codegen fails
#[pyfunction]
#[pyo3(signature = (path, strict = false))]
fn compile_file_hugr(
    py: Python<'_>,
    path: &str,
    strict: bool,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let source = read_file(path)?;
    let filename = filename_from_path(path);

    let program = ::zlup::parse_file(&source, filename).map_err(parse_error_to_py)?;

    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(semantic_error_to_py)?;

    let mut codegen = HugrCodegen::new();
    let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
    let bytes = codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?;

    Ok(pyo3::types::PyBytes::new(py, &bytes).into())
}

// =============================================================================
// SLR-AST Types (for direct construction)
// =============================================================================

/// SLR Program builder for constructing AST directly from Python.
///
/// Example:
///     ```python
///     prog = zluppy.SlrProgram("main")
///     prog.add_allocator("q", 2)
///     prog.add_gate("H", [("q", 0)])
///     prog.add_gate("CX", [("q", 0), ("q", 1)])
///     json_str = prog.to_json()
///     ```
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct SlrProgram {
    inner: ::zlup::codegen::slr::SlrProgram,
}

#[pymethods]
impl SlrProgram {
    /// Create a new SLR program.
    #[new]
    fn new(name: &str) -> Self {
        Self {
            inner: ::zlup::codegen::slr::SlrProgram::new(name),
        }
    }

    /// Add an allocator declaration.
    fn add_allocator(&mut self, name: &str, capacity: usize) {
        let decl = ::zlup::codegen::slr::SlrAllocatorDecl::new(name, capacity);
        if self.inner.allocator.is_none() {
            self.inner.allocator = Some(decl.clone());
        }
        self.inner
            .declarations
            .push(::zlup::codegen::slr::SlrDeclaration::Allocator(decl));
    }

    /// Add a gate operation.
    ///
    /// Args:
    ///     gate: Gate name (e.g., "H", "CX", "RZ")
    ///     targets: List of (allocator_name, index) tuples
    ///     params: Optional list of parameter values (for parameterized gates)
    #[pyo3(signature = (gate, targets, params = None))]
    fn add_gate(
        &mut self,
        gate: &str,
        targets: Vec<(String, usize)>,
        params: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let gate_name = match gate {
            // Single-qubit Pauli gates
            "H" | "h" => "H",
            "X" | "x" => "X",
            "Y" | "y" => "Y",
            "Z" | "z" => "Z",
            // Square root gates (single-qubit)
            "SX" | "sx" => "SX",
            "SY" | "sy" => "SY",
            "SZ" | "sz" => "SZ",
            "SXdg" | "sxdg" => "SXdg",
            "SYdg" | "sydg" => "SYdg",
            "SZdg" | "szdg" => "SZdg",
            // T gates
            "T" | "t" => "T",
            "Tdg" | "tdg" => "Tdg",
            // F gates (Clifford face rotations)
            "F" | "f" => "F",
            "Fdg" | "fdg" => "Fdg",
            "F4" | "f4" => "F4",
            "F4dg" | "f4dg" => "F4dg",
            // Two-qubit controlled gates
            "CX" | "cx" => "CX",
            "CY" | "cy" => "CY",
            "CZ" | "cz" => "CZ",
            "CH" | "ch" => "CH",
            // Two-qubit Ising gates
            "SXX" | "sxx" => "SXX",
            "SYY" | "syy" => "SYY",
            "SZZ" | "szz" => "SZZ",
            "SXXdg" | "sxxdg" => "SXXdg",
            "SYYdg" | "syydg" => "SYYdg",
            "SZZdg" | "szzdg" => "SZZdg",
            // Swap gates
            "SWAP" | "swap" => "SWAP",
            "iSWAP" | "iswap" => "iSWAP",
            // Rotation gates (single-qubit, parameterized)
            "RX" | "rx" => "RX",
            "RY" | "ry" => "RY",
            "RZ" | "rz" => "RZ",
            // Rotation gates (two-qubit, parameterized)
            "CRZ" | "crz" => "CRZ",
            "RZZ" | "rzz" => "RZZ",
            // Three-qubit gates
            "CCX" | "ccx" => "CCX",
            _ => {
                return Err(PyValueError::new_err(format!("Unknown gate: {}", gate)));
            }
        };

        let slot_refs: Vec<_> = targets
            .into_iter()
            .map(|(alloc, idx)| ::zlup::codegen::slr::SlrSlotRef::new(alloc, idx))
            .collect();

        let param_exprs: Vec<_> = params
            .unwrap_or_default()
            .into_iter()
            .map(|v| {
                ::zlup::codegen::slr::SlrExpression::Literal(
                    ::zlup::codegen::slr::SlrLiteralExpr::float(v),
                )
            })
            .collect();

        let gate_op =
            ::zlup::codegen::slr::SlrGateOp::new(gate_name, slot_refs).with_params(param_exprs);
        self.inner
            .body
            .push(::zlup::codegen::slr::SlrStatement::Gate(gate_op));

        Ok(())
    }

    /// Add a prepare operation (reset qubits to |0⟩).
    ///
    /// Args:
    ///     allocator: Allocator name
    ///     slots: Optional list of slot indices. If None, prepares all slots.
    #[pyo3(signature = (allocator, slots = None))]
    fn add_prepare(&mut self, allocator: &str, slots: Option<Vec<usize>>) {
        let prepare_op = match slots {
            Some(s) => ::zlup::codegen::slr::SlrPrepareOp::slots(allocator, s),
            None => ::zlup::codegen::slr::SlrPrepareOp::all(allocator),
        };
        self.inner
            .body
            .push(::zlup::codegen::slr::SlrStatement::Prepare(prepare_op));
    }

    /// Convert to JSON string.
    #[pyo3(signature = (compact = false))]
    fn to_json(&self, compact: bool) -> PyResult<String> {
        if compact {
            serde_json::to_string(&self.inner)
                .map_err(|e| PyValueError::new_err(format!("JSON error: {}", e)))
        } else {
            serde_json::to_string_pretty(&self.inner)
                .map_err(|e| PyValueError::new_err(format!("JSON error: {}", e)))
        }
    }

    /// Convert to Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json_str = self.to_json(false)?;
        let json_module = py.import("json")?;
        let result = json_module.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!("SlrProgram(name={:?})", self.inner.name)
    }
}

// =============================================================================
// ZlupProgram Builder (builds Zlup AST directly)
// =============================================================================

/// Builder for constructing Zlup programs programmatically.
///
/// This builds the Zlup AST directly, which can then be compiled to
/// either SLR-AST or HUGR through the normal compilation pipeline.
///
/// Example:
///     ```python
///     prog = zluppy.ZlupProgram("main")
///     prog.add_allocator("q", 2)
///     prog.add_gate("h", [("q", 0)])
///     prog.add_gate("cx", [("q", 0), ("q", 1)])
///
///     # Compile to SLR
///     slr_json = prog.compile_to_slr()
///
///     # Or compile to HUGR
///     hugr_bytes = prog.compile_to_hugr()
///
///     # Or generate source code
///     source = prog.to_source()
///     ```
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ZlupProgram {
    name: String,
    statements: Vec<::zlup::ast::Stmt>,
    strict: bool,
}

#[pymethods]
impl ZlupProgram {
    /// Create a new Zlup program builder.
    ///
    /// Args:
    ///     name: Program/function name (default: "main")
    ///     strict: Enable strict mode for compilation (default: False)
    #[new]
    #[pyo3(signature = (name = "main", strict = false))]
    fn new(name: &str, strict: bool) -> Self {
        Self {
            name: name.to_string(),
            statements: Vec::new(),
            strict,
        }
    }

    /// Add a qubit allocator declaration.
    ///
    /// Args:
    ///     name: Allocator variable name
    ///     capacity: Number of qubits to allocate
    ///
    /// Returns:
    ///     self: For method chaining
    fn add_allocator(&mut self, name: &str, capacity: usize) -> Self {
        // Build: var {name} = qalloc({capacity});
        let alloc_call = ::zlup::ast::Expr::Call(Box::new(::zlup::ast::CallExpr {
            callee: ::zlup::ast::Expr::Ident(::zlup::ast::Ident {
                name: "qalloc".to_string(),
                location: None,
            }),
            args: vec![::zlup::ast::Expr::IntLit(::zlup::ast::IntLit {
                value: capacity as i128,
                suffix: None,
                location: None,
            })],
            location: None,
        }));

        let binding = ::zlup::ast::Binding {
            name: name.to_string(),
            ty: None,
            value: Some(alloc_call),
            is_mutable: true,
            is_pub: false,
            doc_comment: None,
            location: None,
        };

        self.statements.push(::zlup::ast::Stmt::Binding(binding));
        self.clone()
    }

    /// Add a gate operation.
    ///
    /// Args:
    ///     gate: Gate name (e.g., "h", "cx", "rz")
    ///     targets: List of (allocator_name, index) tuples
    ///     params: Optional list of parameter values (for rotation gates)
    ///
    /// Returns:
    ///     self: For method chaining
    #[pyo3(signature = (gate, targets, params = None))]
    fn add_gate(
        &mut self,
        gate: &str,
        targets: Vec<(String, usize)>,
        params: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let gate_kind = match gate.to_lowercase().as_str() {
            // Single-qubit Paulis
            "x" => ::zlup::ast::GateKind::X,
            "y" => ::zlup::ast::GateKind::Y,
            "z" => ::zlup::ast::GateKind::Z,
            "h" => ::zlup::ast::GateKind::H,
            // Phase gates
            "s" => ::zlup::ast::GateKind::SZ,
            "sdg" => ::zlup::ast::GateKind::SZdg,
            "t" => ::zlup::ast::GateKind::T,
            "tdg" => ::zlup::ast::GateKind::Tdg,
            // Square root gates
            "sx" => ::zlup::ast::GateKind::SX,
            "sy" => ::zlup::ast::GateKind::SY,
            "sz" => ::zlup::ast::GateKind::SZ,
            "sxdg" => ::zlup::ast::GateKind::SXdg,
            "sydg" => ::zlup::ast::GateKind::SYdg,
            "szdg" => ::zlup::ast::GateKind::SZdg,
            // Rotation gates
            "rx" => ::zlup::ast::GateKind::RX,
            "ry" => ::zlup::ast::GateKind::RY,
            "rz" => ::zlup::ast::GateKind::RZ,
            // Two-qubit gates
            "cx" => ::zlup::ast::GateKind::CX,
            "cy" => ::zlup::ast::GateKind::CY,
            "cz" => ::zlup::ast::GateKind::CZ,
            "ch" => ::zlup::ast::GateKind::CH,
            // Two-qubit Ising gates
            "sxx" => ::zlup::ast::GateKind::SXX,
            "syy" => ::zlup::ast::GateKind::SYY,
            "szz" => ::zlup::ast::GateKind::SZZ,
            "sxxdg" => ::zlup::ast::GateKind::SXXdg,
            "syydg" => ::zlup::ast::GateKind::SYYdg,
            "szzdg" => ::zlup::ast::GateKind::SZZdg,
            "rzz" => ::zlup::ast::GateKind::RZZ,
            // Face rotations
            "f" => ::zlup::ast::GateKind::F,
            "fdg" => ::zlup::ast::GateKind::Fdg,
            "f4" => ::zlup::ast::GateKind::F4,
            "f4dg" => ::zlup::ast::GateKind::F4dg,
            _ => return Err(PyValueError::new_err(format!("Unknown gate: {}", gate))),
        };

        // Build slot references
        let slot_refs: Vec<_> = targets
            .into_iter()
            .map(|(alloc, idx)| ::zlup::ast::SlotRef {
                allocator: alloc,
                index: Box::new(::zlup::ast::Expr::IntLit(::zlup::ast::IntLit {
                    value: idx as i128,
                    suffix: None,
                    location: None,
                })),
                location: None,
            })
            .collect();

        // Build parameter expressions
        let param_exprs: Vec<_> = params
            .unwrap_or_default()
            .into_iter()
            .map(|v| {
                ::zlup::ast::Expr::FloatLit(::zlup::ast::FloatLit {
                    value: v,
                    suffix: None,
                    location: None,
                })
            })
            .collect();

        let gate_op = ::zlup::ast::GateOp {
            kind: gate_kind,
            targets: slot_refs,
            params: param_exprs,
            attrs: Vec::new(),
            location: None,
        };

        self.statements.push(::zlup::ast::Stmt::Gate(gate_op));
        Ok(self.clone())
    }

    /// Add a prepare operation (reset qubits to |0⟩).
    ///
    /// Args:
    ///     allocator: Allocator name
    ///     slots: Optional list of slot indices. If None, prepares all slots.
    ///
    /// Returns:
    ///     self: For method chaining
    #[pyo3(signature = (allocator, slots = None))]
    fn add_prepare(&mut self, allocator: &str, slots: Option<Vec<u32>>) -> Self {
        let prepare_op = ::zlup::ast::PrepareOp {
            allocator: allocator.to_string(),
            slots,
            location: None,
        };

        self.statements.push(::zlup::ast::Stmt::Prepare(prepare_op));
        self.clone()
    }

    /// Add a measure operation.
    ///
    /// Args:
    ///     targets: List of (allocator_name, index) tuples to measure
    ///
    /// Returns:
    ///     self: For method chaining
    fn add_measure(&mut self, targets: Vec<(String, usize)>) -> Self {
        let slot_refs: Vec<_> = targets
            .into_iter()
            .map(|(alloc, idx)| ::zlup::ast::SlotRef {
                allocator: alloc,
                index: Box::new(::zlup::ast::Expr::IntLit(::zlup::ast::IntLit {
                    value: idx as i128,
                    suffix: None,
                    location: None,
                })),
                location: None,
            })
            .collect();

        let measure_op = ::zlup::ast::MeasureOp {
            targets: slot_refs,
            results: Vec::new(), // No explicit result register
            location: None,
        };

        self.statements.push(::zlup::ast::Stmt::Measure(measure_op));
        self.clone()
    }

    /// Compile to SLR-AST JSON.
    ///
    /// Args:
    ///     compact: Return compact JSON (default: False)
    ///
    /// Returns:
    ///     str: SLR-AST as JSON string
    #[pyo3(signature = (compact = false))]
    fn compile_to_slr(&self, compact: bool) -> PyResult<String> {
        let program = self.build_ast();

        // Semantic analysis
        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        // SLR codegen
        let mut codegen = SlrCodegen::new();
        let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

        if compact {
            codegen
                .to_json_compact(&slr_program)
                .map_err(codegen_error_to_py)
        } else {
            codegen.to_json(&slr_program).map_err(codegen_error_to_py)
        }
    }

    /// Compile to SLR-AST as Python dict.
    fn compile_to_slr_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json_str = self.compile_to_slr(false)?;
        let json_module = py.import("json")?;
        let result = json_module.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    /// Compile to HUGR bytes.
    ///
    /// Returns:
    ///     bytes: HUGR in binary envelope format
    fn compile_to_hugr(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let program = self.build_ast();

        // Semantic analysis
        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        // HUGR codegen
        let mut codegen = HugrCodegen::new();
        let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
        let bytes = codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?;

        Ok(pyo3::types::PyBytes::new(py, &bytes).into())
    }

    /// Generate Zlup source code from the built AST.
    ///
    /// Returns:
    ///     str: Zlup source code
    fn to_source(&self) -> String {
        let program = self.build_ast();
        ::zlup::pretty::pretty_print(&program, &::zlup::pretty::PrettyOptions::default())
    }

    /// Save the program source code to a .zlp file.
    ///
    /// Args:
    ///     path: Path to write the .zlp file
    ///
    /// Raises:
    ///     IOError: If the file cannot be written
    fn save(&self, path: &str) -> PyResult<()> {
        let source = self.to_source();
        std::fs::write(path, source)
            .map_err(|e| PyIOError::new_err(format!("Failed to write {}: {}", path, e)))
    }

    /// Compile via source code generation and parsing.
    ///
    /// This generates source code, parses it back, and compiles to SLR.
    /// Useful for testing that the generated source is valid Zlup code.
    ///
    /// Args:
    ///     compact: Return compact JSON (default: False)
    ///
    /// Returns:
    ///     str: SLR-AST as JSON string
    #[pyo3(signature = (compact = false))]
    fn compile_via_source_to_slr(&self, compact: bool) -> PyResult<String> {
        let source = self.to_source();

        // Parse the generated source
        let program = ::zlup::parse(&source).map_err(parse_error_to_py)?;

        // Semantic analysis
        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        // SLR codegen
        let mut codegen = SlrCodegen::new();
        let slr_program = codegen.compile(&program).map_err(codegen_error_to_py)?;

        if compact {
            codegen
                .to_json_compact(&slr_program)
                .map_err(codegen_error_to_py)
        } else {
            codegen.to_json(&slr_program).map_err(codegen_error_to_py)
        }
    }

    /// Compile via source code generation and parsing to HUGR.
    ///
    /// This generates source code, parses it back, and compiles to HUGR.
    /// Useful for testing that the generated source is valid Zlup code.
    ///
    /// Returns:
    ///     bytes: HUGR in binary envelope format
    fn compile_via_source_to_hugr(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let source = self.to_source();

        // Parse the generated source
        let program = ::zlup::parse(&source).map_err(parse_error_to_py)?;

        // Semantic analysis
        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        // HUGR codegen
        let mut codegen = HugrCodegen::new();
        let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
        let bytes = codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?;

        Ok(pyo3::types::PyBytes::new(py, &bytes).into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ZlupProgram(name={:?}, statements={}, strict={})",
            self.name,
            self.statements.len(),
            self.strict
        )
    }
}

// Internal methods for ZlupProgram (not exposed to Python)
impl ZlupProgram {
    /// Build the Zlup AST Program.
    fn build_ast(&self) -> ::zlup::ast::Program {
        // Create main function with all statements
        let main_fn = ::zlup::ast::FnDecl {
            name: self.name.clone(),
            params: Vec::new(),
            return_type: Some(::zlup::ast::TypeExpr::Unit),
            body: ::zlup::ast::Block {
                label: None,
                attrs: Vec::new(),
                statements: self.statements.clone(),
                trailing_expr: None,
                location: None,
            },
            is_pub: false,
            is_inline: false,
            error_mode: None,
            doc_comment: None,
            location: None,
        };

        ::zlup::ast::Program {
            name: self.name.clone(),
            declarations: vec![::zlup::ast::TopLevelDecl::Fn(main_fn)],
            location: None,
        }
    }
}

// =============================================================================
// ZluppyEngine
// =============================================================================

/// Engine for compiling and running Zluppy programs.
///
/// Provides a fluent interface for compiling Zluppy source to HUGR
/// and running it through PECOS's simulator.
///
/// Example:
///     ```python
///     result = zluppy.ZluppyEngine().source('''
///         fn main() -> void {
///             var q = qalloc(2);
///             H(q[0]);
///             CX(q[0], q[1]);
///         }
///     ''').run(shots=100)
///     print(result.to_dict())
///     ```
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ZluppyEngine {
    strict: bool,
    hugr_bytes: Option<Vec<u8>>,
}

#[pymethods]
impl ZluppyEngine {
    /// Create a new ZluppyEngine.
    ///
    /// Args:
    ///     strict: Enable strict mode (NASA Power of 10 checks). Default: False
    #[new]
    #[pyo3(signature = (strict = false))]
    fn new(strict: bool) -> Self {
        Self {
            strict,
            hugr_bytes: None,
        }
    }

    /// Compile Zluppy source code.
    ///
    /// Args:
    ///     code: Zluppy source code as a string
    ///
    /// Returns:
    ///     self: For method chaining
    fn source(&mut self, code: &str) -> PyResult<Self> {
        let program = ::zlup::parse(code).map_err(parse_error_to_py)?;

        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        let mut codegen = HugrCodegen::new();
        let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
        self.hugr_bytes = Some(codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?);

        Ok(self.clone())
    }

    /// Compile a .zlp file.
    ///
    /// Args:
    ///     path: Path to a .zlp file
    ///
    /// Returns:
    ///     self: For method chaining
    fn file(&mut self, path: &str) -> PyResult<Self> {
        let source = read_file(path)?;
        let filename = filename_from_path(path);

        let program = ::zlup::parse_file(&source, filename).map_err(parse_error_to_py)?;

        let mut analyzer = if self.strict {
            SemanticAnalyzer::new()
        } else {
            SemanticAnalyzer::new_permissive()
        };
        analyzer.analyze(&program).map_err(semantic_error_to_py)?;

        let mut codegen = HugrCodegen::new();
        let hugr = codegen.compile(&program).map_err(hugr_error_to_py)?;
        self.hugr_bytes = Some(codegen.to_bytes(&hugr).map_err(hugr_error_to_py)?);

        Ok(self.clone())
    }

    /// Return the compiled HUGR bytes.
    fn to_hugr_bytes(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let hugr_bytes = self.hugr_bytes.as_ref().ok_or_else(|| {
            PyValueError::new_err("No source compiled. Call .source() or .file() first.")
        })?;
        Ok(pyo3::types::PyBytes::new(py, hugr_bytes).into())
    }

    fn __repr__(&self) -> String {
        let status = if self.hugr_bytes.is_some() {
            "compiled"
        } else {
            "not compiled"
        };
        format!("ZluppyEngine(strict={}, status={})", self.strict, status)
    }
}

// =============================================================================
// Module Definition
// =============================================================================

/// Zluppy Python module (internal).
///
/// A Zig/SLR/NASA Power of 10 reflection of Guppy's approach to quantum programming.
#[pymodule]
fn _zluppy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add exception
    m.add("ZluppyError", m.py().get_type::<ZluppyError>())?;

    // Add functions
    m.add_function(wrap_pyfunction!(compile_to_slr, m)?)?;
    m.add_function(wrap_pyfunction!(compile_to_slr_json, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(parse_debug, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // File-based functions
    m.add_function(wrap_pyfunction!(compile_file, m)?)?;
    m.add_function(wrap_pyfunction!(compile_file_json, m)?)?;
    m.add_function(wrap_pyfunction!(check_file, m)?)?;

    // HUGR compilation functions
    m.add_function(wrap_pyfunction!(compile_to_hugr, m)?)?;
    m.add_function(wrap_pyfunction!(compile_file_hugr, m)?)?;

    // Add classes
    m.add_class::<SlrProgram>()?;
    m.add_class::<ZlupProgram>()?;
    m.add_class::<ZluppyEngine>()?;

    Ok(())
}

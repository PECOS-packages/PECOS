//! Zero-dependency program types for PECOS quantum simulation
//!
//! Pure data types for quantum programs that can be used
//! across different PECOS engine crates without creating dependencies between them.

pub mod prelude;

use std::fmt;
use std::io;
use std::path::Path;

/// A QASM program
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qasm {
    /// The QASM source code
    pub source: String,
}

impl Qasm {
    /// Create a QASM program from a string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self { source: s.into() }
    }

    /// Create a QASM program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let source = std::fs::read_to_string(path)?;
        Ok(Self { source })
    }

    /// Get the source code
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl fmt::Display for Qasm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)
    }
}

/// Content types for QIS programs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QisContent {
    /// LLVM IR text format
    Ir(String),
    /// LLVM bitcode binary format
    Bitcode(Vec<u8>),
}

/// A QIS (Quantum Instruction Set) program
///
/// This represents LLVM IR that uses Selene QIS functions (___qalloc, ___`lazy_measure`, etc.)
/// as opposed to QIR functions. This is the output of HUGR compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qis {
    /// The QIS content (IR text or bitcode)
    pub content: QisContent,
}

impl Qis {
    /// Create a QIS program from IR text
    ///
    /// Create a QIS program from LLVM IR text
    ///
    /// Stores raw IR to let the JIT executor handle all preprocessing consistently.
    /// This avoids double preprocessing issues while ensuring compatibility with
    /// both raw QIS IR and HUGR-generated IR.
    pub fn from_string(s: impl Into<String>) -> Self {
        let raw_ir = s.into();
        Self {
            content: QisContent::Ir(raw_ir),
        }
    }

    /// Preprocess LLVM IR to filter out problematic metadata
    ///
    /// Removes metadata lines that can cause parsing issues in QIS compilation,
    /// such as HUGR-generated metadata that's not needed for execution.
    fn preprocess_llvm_ir(llvm_ir: &str) -> String {
        let mut filtered_lines = Vec::new();

        for line in llvm_ir.lines() {
            let line_trimmed = line.trim();
            // Skip all metadata lines that aren't needed for QIS execution
            // This includes both definitions (!0 = ...) and references (!name = ...)
            if line_trimmed.starts_with('!') {
                // Skip this metadata line
                continue;
            }
            // Skip completely empty lines to prevent parsing issues
            if line_trimmed.is_empty() {
                continue;
            }
            filtered_lines.push(line.trim_end());
        }

        // Join with newlines and ensure proper termination
        let mut result = filtered_lines.join("\n");
        if !result.ends_with('\n') {
            result.push('\n');
        }
        result
    }

    /// Create a QIS program from IR text (alias for `from_string`)
    pub fn from_ir(s: impl Into<String>) -> Self {
        Self::from_string(s)
    }

    /// Preprocess LLVM IR without creating a `Qis` (for debugging)
    pub fn preprocess_ir(llvm_ir: impl Into<String>) -> String {
        Self::preprocess_llvm_ir(&llvm_ir.into())
    }

    /// Create a QIS program from bitcode
    pub fn from_bitcode(bitcode: impl Into<Vec<u8>>) -> Self {
        Self {
            content: QisContent::Bitcode(bitcode.into()),
        }
    }

    /// Create a QIS program by reading from a file
    /// Auto-detects format based on extension (.ll for IR, .bc for bitcode)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let path = path.as_ref();
        if path.extension().and_then(|s| s.to_str()) == Some("bc") {
            // Read as bitcode
            let bitcode = std::fs::read(path)?;
            Ok(Self::from_bitcode(bitcode))
        } else {
            // Read as IR text (default for .ll or no extension)
            let ir = std::fs::read_to_string(path)?;
            Ok(Self::from_ir(ir))
        }
    }

    /// Create a QIS program from an IR text file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_ir_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let ir = std::fs::read_to_string(path)?;
        Ok(Self::from_ir(ir))
    }

    /// Create a QIS program from a bitcode file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_bitcode_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let bitcode = std::fs::read(path)?;
        Ok(Self::from_bitcode(bitcode))
    }

    /// Get the IR source code (if this is IR text)
    #[must_use]
    pub fn ir(&self) -> Option<&str> {
        match &self.content {
            QisContent::Ir(ir) => Some(ir),
            QisContent::Bitcode(_) => None,
        }
    }

    /// Get the source code (backward compatibility - returns IR if available)
    #[must_use]
    pub fn source(&self) -> &str {
        self.ir().unwrap_or("")
    }

    /// Get the bitcode (if this is bitcode)
    #[must_use]
    pub fn bitcode(&self) -> Option<&[u8]> {
        match &self.content {
            QisContent::Ir(_) => None,
            QisContent::Bitcode(bc) => Some(bc),
        }
    }

    /// Check if this is IR text
    #[must_use]
    pub fn is_ir(&self) -> bool {
        matches!(self.content, QisContent::Ir(_))
    }

    /// Check if this is bitcode
    #[must_use]
    pub fn is_bitcode(&self) -> bool {
        matches!(self.content, QisContent::Bitcode(_))
    }
}

impl fmt::Display for Qis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.content {
            QisContent::Ir(ir) => write!(f, "{ir}"),
            QisContent::Bitcode(bc) => write!(f, "Qis(bitcode, {} bytes)", bc.len()),
        }
    }
}

/// A HUGR program
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hugr {
    /// The HUGR data (serialized bytes)
    pub hugr: Vec<u8>,
}

impl Hugr {
    /// Create a HUGR program from bytes
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { hugr: bytes }
    }

    /// Create a HUGR program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let hugr = std::fs::read(path)?;
        Ok(Self { hugr })
    }

    /// Get the HUGR bytes
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.hugr
    }

    /// Get the HUGR bytes as a Vec (consuming self)
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.hugr
    }
}

impl fmt::Display for Hugr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hugr({} bytes)", self.hugr.len())
    }
}

/// A WebAssembly program (binary .wasm format)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wasm {
    /// The WASM binary data
    pub wasm: Vec<u8>,
}

impl Wasm {
    /// Create a WASM program from bytes
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self { wasm: bytes.into() }
    }

    /// Create a WASM program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let wasm = std::fs::read(path)?;
        Ok(Self { wasm })
    }

    /// Get the WASM bytes
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.wasm
    }

    /// Get the WASM bytes as a Vec (consuming self)
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.wasm
    }
}

impl fmt::Display for Wasm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wasm({} bytes)", self.wasm.len())
    }
}

/// A WebAssembly Text program (.wat format)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wat {
    /// The WAT source code
    pub source: String,
}

impl Wat {
    /// Create a WAT program from a string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self { source: s.into() }
    }

    /// Create a WAT program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let source = std::fs::read_to_string(path)?;
        Ok(Self { source })
    }

    /// Get the source code
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl fmt::Display for Wat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)
    }
}

/// A PHIR JSON program
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhirJson {
    /// The PHIR JSON source code
    pub source: String,
}

impl PhirJson {
    /// Create a PHIR JSON program from a string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self { source: s.into() }
    }

    /// Create a PHIR JSON program from JSON (alias for `from_string`)
    pub fn from_json(s: impl Into<String>) -> Self {
        Self::from_string(s)
    }

    /// Create a PHIR JSON program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let source = std::fs::read_to_string(path)?;
        Ok(Self { source })
    }

    /// Get the source code
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the JSON source (alias for source)
    #[must_use]
    pub fn json(&self) -> &str {
        &self.source
    }
}

impl fmt::Display for PhirJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)
    }
}

/// A Selene Interface Program (compiled plugin)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeleneInterface {
    /// The compiled plugin data (shared library bytes) or executable metadata
    pub plugin: Vec<u8>,
    /// Optional: Path to the Selene executable (for pre-compiled executables)
    pub executable_path: Option<String>,
    /// Optional: Path to the artifacts directory
    pub artifacts_path: Option<String>,
}

impl SeleneInterface {
    /// Create a Selene Interface program from plugin bytes
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            plugin: bytes,
            executable_path: None,
            artifacts_path: None,
        }
    }

    /// Create a Selene Interface program with executable paths
    #[must_use]
    pub fn from_executable(
        executable_path: String,
        artifacts_path: String,
        plugin_bytes: Vec<u8>,
    ) -> Self {
        Self {
            plugin: plugin_bytes,
            executable_path: Some(executable_path),
            artifacts_path: Some(artifacts_path),
        }
    }

    /// Create a Selene Interface program by reading from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let plugin = std::fs::read(path)?;
        Ok(Self {
            plugin,
            executable_path: None,
            artifacts_path: None,
        })
    }

    /// Get the plugin bytes
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.plugin
    }

    /// Get the plugin bytes as a Vec (consuming self)
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.plugin
    }
}

impl fmt::Display for SeleneInterface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeleneInterface({} bytes)", self.plugin.len())
    }
}

/// Enum for runtime dispatch of program types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Program {
    /// A QASM program
    Qasm(Qasm),
    /// A QIS program (Quantum Instruction Set - LLVM IR format)
    Qis(Qis),
    /// A HUGR program
    Hugr(Hugr),
    /// A WebAssembly program
    Wasm(Wasm),
    /// A WebAssembly Text program
    Wat(Wat),
    /// A PHIR JSON program
    PhirJson(PhirJson),
    /// A Selene Interface program (compiled plugin)
    SeleneInterface(SeleneInterface),
}

impl Program {
    /// Get the program type as a string
    #[must_use]
    pub fn program_type(&self) -> &'static str {
        match self {
            Program::Qasm(_) => "QASM",
            Program::Qis(_) => "QIS",
            Program::Hugr(_) => "HUGR",
            Program::Wasm(_) => "WASM",
            Program::Wat(_) => "WAT",
            Program::PhirJson(_) => "PHIR-JSON",
            Program::SeleneInterface(_) => "SELENE-INTERFACE",
        }
    }
}

impl From<Qasm> for Program {
    fn from(program: Qasm) -> Self {
        Program::Qasm(program)
    }
}

impl From<Qis> for Program {
    fn from(program: Qis) -> Self {
        Program::Qis(program)
    }
}

impl From<Hugr> for Program {
    fn from(program: Hugr) -> Self {
        Program::Hugr(program)
    }
}

impl From<Wasm> for Program {
    fn from(program: Wasm) -> Self {
        Program::Wasm(program)
    }
}

impl From<Wat> for Program {
    fn from(program: Wat) -> Self {
        Program::Wat(program)
    }
}

impl From<PhirJson> for Program {
    fn from(program: PhirJson) -> Self {
        Program::PhirJson(program)
    }
}

impl From<SeleneInterface> for Program {
    fn from(program: SeleneInterface) -> Self {
        Program::SeleneInterface(program)
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Program::Qasm(p) => write!(f, "QASM: {p}"),
            Program::Qis(p) => write!(f, "QIS: {p}"),
            Program::Hugr(p) => write!(f, "{p}"),
            Program::Wasm(p) => write!(f, "{p}"),
            Program::Wat(p) => write!(f, "WAT: {p}"),
            Program::PhirJson(p) => write!(f, "PHIR-JSON: {p}"),
            Program::SeleneInterface(p) => write!(f, "{p}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_qasm() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];";
        let program = Qasm::from_string(qasm);
        assert_eq!(program.source(), qasm);
        assert_eq!(program.to_string(), qasm);
    }

    #[test]
    fn test_qis() {
        let ir = "define void @main() { ret void }";
        let program = Qis::from_string(ir);
        assert_eq!(program.ir(), Some(ir));
        assert_eq!(program.to_string(), ir);

        // Test bitcode
        let bitcode = vec![0xDE, 0xC0, 0xDE, 0xCA, 0xFE];
        let program = Qis::from_bitcode(bitcode.clone());
        assert_eq!(program.bitcode(), Some(&bitcode[..]));
        assert_eq!(program.ir(), None);
        assert_eq!(program.to_string(), "Qis(bitcode, 5 bytes)");
    }

    #[test]
    fn test_hugr() {
        let bytes = vec![1, 2, 3, 4, 5];
        let program = Hugr::from_bytes(bytes.clone());
        assert_eq!(program.bytes(), &bytes[..]);
        assert_eq!(program.to_string(), "Hugr(5 bytes)");
    }

    #[test]
    fn test_wasm() {
        let wasm_bytes = vec![0x00, 0x61, 0x73, 0x6D]; // WASM magic number
        let program = Wasm::from_bytes(wasm_bytes.clone());
        assert_eq!(program.bytes(), &wasm_bytes[..]);
        assert_eq!(program.to_string(), "Wasm(4 bytes)");

        let program2 = Wasm::from_bytes(&wasm_bytes[..]);
        assert_eq!(program2.bytes(), &wasm_bytes[..]);
    }

    #[test]
    fn test_wat() {
        let wat = "(module (func $main))";
        let program = Wat::from_string(wat);
        assert_eq!(program.source(), wat);
        assert_eq!(program.to_string(), wat);
    }

    #[test]
    fn test_program_enum() {
        let qasm = Qasm::from_string("OPENQASM 2.0;");
        let program: Program = qasm.into();
        assert_eq!(program.program_type(), "QASM");

        let qis = Qis::from_string("define void @main() {}");
        let program: Program = qis.into();
        assert_eq!(program.program_type(), "QIS");

        let hugr = Hugr::from_bytes(vec![1, 2, 3]);
        let program: Program = hugr.into();
        assert_eq!(program.program_type(), "HUGR");

        let wasm = Wasm::from_bytes(vec![0x00, 0x61, 0x73, 0x6D]);
        let program: Program = wasm.into();
        assert_eq!(program.program_type(), "WASM");

        let wat = Wat::from_string("(module)");
        let program: Program = wat.into();
        assert_eq!(program.program_type(), "WAT");
    }

    #[test]
    fn test_from_file() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = tempfile::tempdir()?;

        // Test QASM from file
        let qasm_path = temp_dir.path().join("test.qasm");
        let mut file = std::fs::File::create(&qasm_path)?;
        writeln!(file, "OPENQASM 2.0;")?;
        writeln!(file, "qreg q[2];")?;
        drop(file);

        let qasm_program = Qasm::from_file(&qasm_path)?;
        assert_eq!(qasm_program.source().trim(), "OPENQASM 2.0;\nqreg q[2];");

        // Test QIS from file
        let qis_path = temp_dir.path().join("test.ll");
        let mut file = std::fs::File::create(&qis_path)?;
        writeln!(file, "define void @main() {{")?;
        writeln!(file, "  ret void")?;
        writeln!(file, "}}")?;
        drop(file);

        let qis_program = Qis::from_file(&qis_path)?;
        assert!(qis_program.ir().unwrap().contains("define void @main()"));

        // Test QIS bitcode from file
        let bc_path = temp_dir.path().join("test.bc");
        let bitcode_data = vec![0xDE, 0xC0, 0xDE, 0x42, 0x01, 0x0C];
        std::fs::write(&bc_path, &bitcode_data)?;

        let bc_program = Qis::from_file(&bc_path)?;
        assert!(bc_program.is_bitcode());
        assert_eq!(bc_program.bitcode(), Some(&bitcode_data[..]));

        // Test HUGR from file
        let hugr_path = temp_dir.path().join("test.hugr");
        let hugr_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        std::fs::write(&hugr_path, &hugr_data)?;

        let hugr_program = Hugr::from_file(&hugr_path)?;
        assert_eq!(hugr_program.bytes(), &hugr_data[..]);

        // Test WASM from file
        let wasm_path = temp_dir.path().join("test.wasm");
        let wasm_data = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        std::fs::write(&wasm_path, &wasm_data)?;

        let wasm_program = Wasm::from_file(&wasm_path)?;
        assert_eq!(wasm_program.bytes(), &wasm_data[..]);

        // Test WAT from file
        let wat_path = temp_dir.path().join("test.wat");
        let wat_content = "(module\n  (func $main)\n)";
        std::fs::write(&wat_path, wat_content)?;

        let wat_program = Wat::from_file(&wat_path)?;
        assert_eq!(wat_program.source(), wat_content);

        Ok(())
    }
}

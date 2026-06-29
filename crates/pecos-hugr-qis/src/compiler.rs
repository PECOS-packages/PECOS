//! HUGR to QIS LLVM IR compiler
//!
//! HUGR to LLVM IR compilation that generates
//! Selene QIS-compatible LLVM IR. It matches the full functionality
//! of tket2's qis-compiler but without Python bindings.

use anyhow::{Result, anyhow};
use pecos_core::errors::PecosError;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

/// Extension trait providing `exactly_one()` for iterators.
trait ExactlyOneExt: Iterator + Sized {
    /// Returns the single element of an iterator, or an error if there are zero or multiple.
    fn exactly_one(mut self) -> std::result::Result<Self::Item, ExactlyOneError> {
        match self.next() {
            None => Err(ExactlyOneError::Empty),
            Some(first) => {
                if self.next().is_some() {
                    Err(ExactlyOneError::Multiple)
                } else {
                    Ok(first)
                }
            }
        }
    }
}

impl<I: Iterator> ExactlyOneExt for I {}

/// Error returned when `exactly_one()` fails.
#[derive(Debug)]
enum ExactlyOneError {
    Empty,
    Multiple,
}
use tket::hugr::envelope::EnvelopeConfig;
use tket::hugr::llvm::extension::int::IntCodegenExtension;
use tket::hugr::llvm::inkwell::OptimizationLevel;
use tket::hugr::llvm::inkwell::context::Context;
use tket::hugr::llvm::inkwell::module::Module;
use tket::hugr::llvm::inkwell::passes::PassBuilderOptions;
use tket::hugr::llvm::inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use tket::hugr::llvm::utils::fat::FatExt as _;
use tket::hugr::llvm::{
    CodegenExtsBuilder,
    custom::CodegenExtsMap,
    emit::{EmitHugr, Namer},
};
use tket::hugr::ops::DataflowParent;
use tket::hugr::{Hugr, HugrView, Node};
use tket::llvm::rotation::RotationCodegenExtension;
use tket::passes::ComposablePass;
use tket_qsystem::llvm::array_utils::ArrayLowering;
use tket_qsystem::llvm::futures::FuturesCodegenExtension;
use tket_qsystem::llvm::{
    debug::DebugCodegenExtension, prelude::QISPreludeCodegen, qsystem::QSystemCodegenExtension,
    random::RandomCodegenExtension, result::ResultsCodegenExtension, utils::UtilsCodegenExtension,
};
use tket_qsystem::{QSystemPass, QSystemPlatform};

// Import read_hugr_envelope from utils module
use crate::utils::read_hugr_envelope;

const LLVM_MAIN: &str = "qmain";
const METADATA: &[(&str, &[&str])] = &[("name", &["mainlib"])];
const HUGR_SYMBOL_PREFIX: &str = "__hugr__.";
const TRACE_METADATA_HUGR_SYMBOL: &str = "pecos_qis_trace_metadata_hugr";
const TRACE_METADATA_QUBIT_HUGR_SYMBOL: &str = "pecos_qis_trace_metadata_qubit_hugr";
const RUNTIME_BARRIER_QUBIT_HUGR_SYMBOL: &str = "pecos_qis_runtime_barrier_qubit_hugr";
const RUNTIME_BARRIER_QUBITS2_HUGR_SYMBOL: &str = "pecos_qis_runtime_barrier_qubits2_hugr";

// Extension registry is defined in the parent module

/// Compilation arguments
#[derive(Debug, Clone)]
pub struct CompileArgs {
    /// Entry point symbol
    pub entry: Option<String>,
    /// LLVM module name
    pub name: String,
    /// Save HUGR to file
    pub save_hugr: Option<PathBuf>,
    /// Target triple (defaults to native)
    pub target_triple: Option<String>,
    /// Optimization level
    pub opt_level: OptimizationLevel,
    /// Target `QSystem` platform for lowering and codegen.
    ///
    /// PECOS targets the Quantinuum Helios QIS runtime (the Selene Helios
    /// plugin), so this defaults to [`QSystemPlatform::Helios`]. Set it
    /// explicitly to select another supported platform such as
    /// [`QSystemPlatform::Sol`]; unsupported platforms are rejected with a
    /// clear error when compilation starts.
    pub platform: QSystemPlatform,
}

impl Default for CompileArgs {
    fn default() -> Self {
        Self {
            entry: None,
            name: "hugr".to_string(),
            save_hugr: None,
            target_triple: None,
            opt_level: OptimizationLevel::Default,
            // PECOS targets the Selene Helios QIS runtime by default.
            platform: QSystemPlatform::Helios,
        }
    }
}

/// Reject `QSystem` platforms that PECOS has not wired through its QIS pipeline.
///
/// [`QSystemPlatform`] is `#[non_exhaustive]`; fail loudly on any future variant
/// rather than silently lowering for a platform PECOS has not validated
/// end-to-end (codegen extensions + Selene runtime).
fn ensure_supported_platform(platform: QSystemPlatform) -> Result<()> {
    match platform {
        QSystemPlatform::Helios | QSystemPlatform::Sol => Ok(()),
        other => Err(anyhow!(
            "Unsupported QSystem platform {other:?}: pecos-hugr-qis supports Helios and Sol. \
             Wire a newer tket-qsystem platform through the QIS codegen and Selene runtime \
             before selecting it."
        )),
    }
}

/// Process HUGR by applying required passes.
///
/// Note: `QSystemPass` internally calls `inline_constant_functions` when the
/// `llvm` feature is enabled, so we don't need to call it separately.
fn process_hugr(hugr: &mut Hugr, platform: QSystemPlatform) -> Result<()> {
    QSystemPass::defaults(platform).run(hugr)?;
    Ok(())
}

/// Build codegen extensions for LLVM generation
fn codegen_extensions(platform: QSystemPlatform) -> CodegenExtsMap<'static, Hugr> {
    use crate::array::SeleneHeapArrayCodegen;
    let pcg = QISPreludeCodegen;

    CodegenExtsBuilder::default()
        .add_prelude_extensions(pcg.clone())
        .add_extension(IntCodegenExtension::new(pcg.clone()))
        .add_float_extensions()
        .add_conversion_extensions()
        .add_logic_extensions()
        .add_extension(SeleneHeapArrayCodegen::LOWERING.codegen_extension())
        .add_default_static_array_extensions()
        .add_default_borrow_array_extensions(pcg.clone())
        .add_extension(FuturesCodegenExtension)
        .add_extension(QSystemCodegenExtension::new(platform, pcg.clone()))
        .add_extension(RandomCodegenExtension)
        .add_extension(ResultsCodegenExtension::new(
            SeleneHeapArrayCodegen::LOWERING,
        ))
        .add_extension(RotationCodegenExtension::new(pcg))
        .add_extension(UtilsCodegenExtension)
        .add_extension(DebugCodegenExtension::new(SeleneHeapArrayCodegen::LOWERING))
        .finish()
}

/// Get the entry point name from the HUGR
fn get_entry_point_name(namer: &Namer, hugr: &impl HugrView<Node = Node>) -> Result<String> {
    const HUGR_MAIN: &str = "main";

    let (name, entry_point_node) = if hugr.entrypoint_optype().is_module() {
        // For backwards compatibility: assume entrypoint is "main" function in module
        let node = hugr
            .children(hugr.module_root())
            .filter(|&n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .is_some_and(|f| f.func_name() == HUGR_MAIN)
            })
            .exactly_one()
            .map_err(|_| {
                anyhow!("Module entrypoint must have a single function named {HUGR_MAIN} as child")
            })?;
        (HUGR_MAIN, node)
    } else {
        let func_defn = hugr
            .entrypoint_optype()
            .as_func_defn()
            .ok_or_else(|| anyhow!("Entry point node is not a function definition"))?;

        if func_defn.inner_signature().input_count() != 0 {
            return Err(anyhow!(
                "Entry point function must have no input parameters (found {})",
                func_defn.inner_signature().input_count()
            ));
        }
        (func_defn.func_name().as_ref(), hugr.entrypoint())
    };

    Ok(namer.name_func(name, entry_point_node))
}

/// Generate LLVM module from HUGR
fn get_hugr_llvm_module<'c>(
    context: &'c Context,
    namer: Rc<Namer>,
    hugr: &Hugr,
    module_name: &str,
    exts: Rc<CodegenExtsMap<'static, Hugr>>,
) -> Result<Module<'c>> {
    let module = context.create_module(module_name);
    let emit = EmitHugr::new(context, module, namer, exts);
    Ok(emit
        .emit_module(
            hugr.try_fat(hugr.module_root())
                .expect("module root must be a valid fat node"),
        )?
        .finish())
}

/// Given an LLVM context and hugr, compile to an LLVM module
fn get_module_with_std_exts<'c>(
    args: &CompileArgs,
    context: &'c Context,
    namer: Rc<Namer>,
    hugr: &'c mut Hugr,
) -> Result<Module<'c>> {
    process_hugr(hugr, args.platform)?;

    if let Some(filename) = &args.save_hugr {
        let file = fs::File::create(filename)?;
        hugr.store(file, EnvelopeConfig::text())?;
    }

    get_hugr_llvm_module(
        context,
        namer,
        hugr,
        &args.name,
        Rc::new(codegen_extensions(args.platform)),
    )
}

/// Wrap the HUGR entry point with setup/teardown calls
fn wrap_main<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    hugr_entry: &str,
    module_entry: &str,
) -> Result<()> {
    let entry_ty = ctx.i64_type().fn_type(&[ctx.i64_type().into()], false);
    let entry_fun = module.add_function(module_entry, entry_ty, None);

    // Add EntryPoint attribute to the function
    entry_fun.add_attribute(
        tket::hugr::llvm::inkwell::attributes::AttributeLoc::Function,
        ctx.create_string_attribute("EntryPoint", ""),
    );

    let setup_type = ctx.void_type().fn_type(&[ctx.i64_type().into()], false);
    let setup = module.add_function("setup", setup_type, None);

    let teardown_type = ctx.i64_type().fn_type(&[], false);
    let teardown = module.add_function("teardown", teardown_type, None);

    let block = ctx.append_basic_block(entry_fun, "entry");
    let builder = ctx.create_builder();
    builder.position_at_end(block);

    let initial_tc = entry_fun
        .get_nth_param(0)
        .expect("entry function must have at least one parameter")
        .into_int_value();
    let hugr_main = module
        .get_function(hugr_entry)
        .ok_or_else(|| anyhow!("Entrypoint function '{hugr_entry}' not found in Module"))?;

    builder.build_call(setup, &[initial_tc.into()], "")?;
    builder.build_call(hugr_main, &[], "")?;
    let tc = builder
        .build_call(teardown, &[], "")?
        .try_as_basic_value()
        .basic()
        .ok_or_else(|| anyhow!("teardown has no return value"))?;
    builder.build_return(Some(&tc))?;

    Ok(())
}

/// Get the native target machine for LLVM
///
/// # Errors
/// Returns an error if target machine creation fails.
///
/// # Panics
/// Panics if native target initialization fails.
pub fn get_native_target_machine(opt_level: OptimizationLevel) -> Result<TargetMachine> {
    let reloc_mode = RelocMode::PIC;
    let code_model = CodeModel::Default;
    Target::initialize_native(&InitializationConfig::default())
        .expect("native LLVM target initialization failed");
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| anyhow!("{e}"))?;

    target
        .create_target_machine(
            &triple,
            &TargetMachine::get_host_cpu_name().to_string_lossy(),
            &TargetMachine::get_host_cpu_features().to_string_lossy(),
            opt_level,
            reloc_mode,
            code_model,
        )
        .ok_or_else(|| anyhow!("Failed to create target machine"))
}

/// Get the target machine from triple
///
/// # Errors
/// Returns an error if the target triple is invalid or target machine creation fails.
pub fn get_target_machine_from_triple(
    target_triple: &str,
    opt_level: OptimizationLevel,
) -> Result<TargetMachine> {
    let reloc_mode = RelocMode::PIC;
    let code_model = CodeModel::Default;
    Target::initialize_all(&InitializationConfig::default());
    let triple = TargetTriple::create(target_triple);
    log::debug!("Using target triple: {triple}");

    let target = Target::from_triple(&triple).map_err(|e| anyhow!("{e}"))?;
    log::debug!("Using target: {:?}", target.get_name());
    // Use the target name as CPU (matches tket2 behavior)
    let cpu: String = target.get_name().to_string_lossy().to_string();

    target
        .create_target_machine(&triple, &cpu, "", opt_level, reloc_mode, code_model)
        .ok_or_else(|| anyhow!("Failed to create target machine"))
}

/// Optimize the module using LLVM passes
fn optimize_module(
    module: &Module,
    target_machine: &TargetMachine,
    opt_level: OptimizationLevel,
) -> Result<()> {
    let opt_str = match opt_level {
        OptimizationLevel::Aggressive => "default<O3>",
        OptimizationLevel::Less => "default<O1>",
        OptimizationLevel::None => "default<O0>",
        OptimizationLevel::Default => "default<O2>",
    };

    module
        .run_passes(opt_str, target_machine, PassBuilderOptions::create())
        .map_err(|e| anyhow!("Failed to run optimization passes: {e}"))?;
    Ok(())
}

/// Compile the given HUGR to an LLVM module
/// This function is the primary entry point for the compiler
fn compile<'c, 'hugr: 'c>(
    args: &CompileArgs,
    ctx: &'c Context,
    hugr: &'hugr mut Hugr,
) -> Result<Module<'c>> {
    // Fail fast before any expensive work if the platform is unsupported.
    ensure_supported_platform(args.platform)?;

    log::debug!("starting primary compilation");
    let namer = Rc::new(Namer::new("__hugr__.", true));

    // Find the entry point
    let hugr_entry = get_entry_point_name(&namer, hugr)?;

    // The name of the entry point in the LLVM module
    let module_entry = args.entry.as_ref().map_or(LLVM_MAIN, |x| x.as_ref());

    // Create a new LLVM module using hugr-llvm
    let module = get_module_with_std_exts(args, ctx, namer, hugr)?;

    // Get the target machine
    let target_machine = if let Some(ref triple) = args.target_triple {
        get_target_machine_from_triple(triple, args.opt_level)?
    } else {
        get_native_target_machine(args.opt_level)?
    };

    // Set target-specific information
    module.set_triple(&target_machine.get_triple());
    module.set_data_layout(&target_machine.get_target_data().get_data_layout());

    // Wrap with setup/teardown
    wrap_main(ctx, &module, &hugr_entry, module_entry)?;

    // Add metadata
    for (key, values) in METADATA {
        let md_vec = values
            .iter()
            .map(|v| ctx.metadata_string(v).into())
            .collect::<Vec<_>>();
        let node = ctx.metadata_node(md_vec.as_slice());
        module
            .add_global_metadata(key, &node)
            .map_err(|e| anyhow!("Failed to add metadata: {e}"))?;
    }

    // Optimize
    optimize_module(&module, &target_machine, args.opt_level)?;

    // Verify
    module
        .verify()
        .map_err(|e| anyhow!("Module verification failed: {e}"))?;

    // Ensure the EntryPoint attribute is properly applied
    // This is a workaround - re-add the attribute after optimization
    if let Some(entry_fun) = module.get_function(module_entry) {
        entry_fun.add_attribute(
            tket::hugr::llvm::inkwell::attributes::AttributeLoc::Function,
            ctx.create_string_attribute("EntryPoint", ""),
        );
    }

    Ok(module)
}

fn is_llvm_symbol_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-')
}

/// Normalize PECOS-owned helper declarations that Guppy/HUGR lowers under a
/// private `__hugr__.*` symbol name.
///
/// These helpers are part of PECOS's runtime ABI, not ordinary user functions,
/// so they need stable external symbols for dynamic linking. Keep this rewrite
/// deliberately narrow: only PECOS-owned QIS helper declarations receive this
/// treatment.
/// If `symbol` is a PECOS-owned helper lowered under the private `__hugr__.*`
/// namespace, return its stable public name; otherwise `None`.
///
/// Guppy qualifies the symbol with the defining scope (module/function, plus a
/// `<locals>` segment for function-local declarations) and tket appends a numeric
/// node id, so the helper name sits second-from-last:
/// `__hugr__.<scope...>.<helper>.<id>`.
fn pecos_helper_public_name<'a>(symbol: &str, helper_symbols: &[&'a str]) -> Option<&'a str> {
    let rest = symbol.strip_prefix(HUGR_SYMBOL_PREFIX)?;
    let mut parts = rest.rsplit('.');
    let _suffix = parts.next()?;
    let helper_name = parts.next()?;
    helper_symbols
        .iter()
        .copied()
        .find(|helper| *helper == helper_name)
}

fn normalize_pecos_helper_symbols_in_llvm(llvm_ir: String) -> String {
    let helper_symbols = [
        TRACE_METADATA_HUGR_SYMBOL,
        TRACE_METADATA_QUBIT_HUGR_SYMBOL,
        RUNTIME_BARRIER_QUBIT_HUGR_SYMBOL,
        RUNTIME_BARRIER_QUBITS2_HUGR_SYMBOL,
    ];
    let mut normalized = String::with_capacity(llvm_ir.len());
    let mut cursor = 0;

    while let Some(relative_start) = llvm_ir[cursor..].find('@') {
        let start = cursor + relative_start;
        normalized.push_str(&llvm_ir[cursor..start]);
        let after_at = start + 1;

        // Quoted LLVM symbol: `@"..."`. Guppy quotes the private symbol whenever
        // the qualified name contains characters illegal in a bare LLVM identifier
        // -- notably the `<locals>` segment of a function-local declaration. The
        // bare-identifier scan below would skip these (a `"` is not a symbol char),
        // leaving the `__hugr__.*` name unnormalized, so handle the quoted form
        // explicitly. A literal `"` inside is LLVM-escaped as `\22`, so the next
        // unescaped `"` always terminates the symbol.
        if llvm_ir[after_at..].starts_with('"') {
            let body_start = after_at + 1;
            if let Some(rel_close) = llvm_ir[body_start..].find('"') {
                let body_end = body_start + rel_close;
                let symbol = &llvm_ir[body_start..body_end];
                if let Some(public) = pecos_helper_public_name(symbol, &helper_symbols) {
                    // The public name is a valid bare identifier, so drop the quotes.
                    normalized.push('@');
                    normalized.push_str(public);
                } else {
                    normalized.push_str("@\"");
                    normalized.push_str(symbol);
                    normalized.push('"');
                }
                cursor = body_end + 1;
                continue;
            }
            // Unterminated quote (malformed IR): emit the `@` and keep scanning.
            normalized.push('@');
            cursor = after_at;
            continue;
        }

        let symbol_start = after_at;
        let symbol_len = llvm_ir[symbol_start..]
            .chars()
            .take_while(|ch| is_llvm_symbol_char(*ch))
            .map(char::len_utf8)
            .sum::<usize>();
        if symbol_len == 0 {
            normalized.push('@');
            cursor = symbol_start;
            continue;
        }

        let symbol_end = symbol_start + symbol_len;
        let symbol = &llvm_ir[symbol_start..symbol_end];
        if let Some(public) = pecos_helper_public_name(symbol, &helper_symbols) {
            normalized.push('@');
            normalized.push_str(public);
        } else {
            normalized.push('@');
            normalized.push_str(symbol);
        }
        cursor = symbol_end;
    }

    normalized.push_str(&llvm_ir[cursor..]);
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_trace_metadata_helper_symbol() {
        let llvm = concat!(
            "declare void @__hugr__.pecos_qis_trace_metadata_hugr.16(i8*, i8*)\n",
            "call void @__hugr__.pecos_qis_trace_metadata_hugr.16(i8* %0, i8* %1)\n",
            "call void @__hugr__.__main__.pecos_qis_trace_metadata_hugr.18(i8* %0, i8* %1)\n",
            "declare i64 @__hugr__.pecos_qis_trace_metadata_qubit_hugr.19(i64, i8*, i8*)\n",
            "%q = call i64 @__hugr__.pecos_qis_trace_metadata_qubit_hugr.19(i64 %0, i8* %1, i8* %2)\n",
            "%q2 = call i64 @__hugr__.__main__.pecos_qis_trace_metadata_qubit_hugr.21(i64 %0, i8* %1, i8* %2)\n",
            "%q3 = call i64 @__hugr__.pecos_qis_runtime_barrier_qubit_hugr.22(i64 %0)\n",
            "%q4 = call i64 @__hugr__.__main__.pecos_qis_runtime_barrier_qubit_hugr.23(i64 %0)\n",
            "%q5 = call { i64, i64 } @__hugr__.pecos_qis_runtime_barrier_qubits2_hugr.24(i64 %0, i64 %1)\n",
            "%q6 = call { i64, i64 } @__hugr__.__main__.pecos_qis_runtime_barrier_qubits2_hugr.25(i64 %0, i64 %1)\n",
            "call void @__hugr__.other_helper.16()\n",
        )
        .to_string();
        let normalized = normalize_pecos_helper_symbols_in_llvm(llvm);
        assert!(normalized.contains("declare void @pecos_qis_trace_metadata_hugr(i8*, i8*)"));
        assert!(normalized.contains("call void @pecos_qis_trace_metadata_hugr(i8* %0, i8* %1)"));
        assert!(
            normalized.contains("declare i64 @pecos_qis_trace_metadata_qubit_hugr(i64, i8*, i8*)")
        );
        assert!(normalized.contains(
            "%q = call i64 @pecos_qis_trace_metadata_qubit_hugr(i64 %0, i8* %1, i8* %2)"
        ));
        assert!(normalized.contains(
            "%q2 = call i64 @pecos_qis_trace_metadata_qubit_hugr(i64 %0, i8* %1, i8* %2)"
        ));
        assert!(
            normalized.contains("%q3 = call i64 @pecos_qis_runtime_barrier_qubit_hugr(i64 %0)")
        );
        assert!(
            normalized.contains("%q4 = call i64 @pecos_qis_runtime_barrier_qubit_hugr(i64 %0)")
        );
        assert!(normalized.contains(
            "%q5 = call { i64, i64 } @pecos_qis_runtime_barrier_qubits2_hugr(i64 %0, i64 %1)"
        ));
        assert!(normalized.contains(
            "%q6 = call { i64, i64 } @pecos_qis_runtime_barrier_qubits2_hugr(i64 %0, i64 %1)"
        ));
        assert!(normalized.contains("@__hugr__.other_helper.16"));
    }

    #[test]
    fn normalize_quoted_function_local_helper_symbol() {
        // Function-local Guppy declarations qualify the symbol with a `<locals>`
        // segment, whose angle brackets force LLVM to quote the whole symbol
        // (`@"..."`). The normalizer must handle the quoted form, not skip it.
        let llvm = concat!(
            "declare { i64, i64 } @\"__hugr__.test_mod.test_fn.<locals>.pecos_qis_runtime_barrier_qubits2_hugr.23\"(i64, i64)\n",
            "%1 = call { i64, i64 } @\"__hugr__.test_mod.test_fn.<locals>.pecos_qis_runtime_barrier_qubits2_hugr.23\"(i64 %0, i64 %2)\n",
            "%q = call i64 @\"__hugr__.m.f.<locals>.pecos_qis_trace_metadata_qubit_hugr.7\"(i64 %0, i8* %1, i8* %2)\n",
            "%x = call i64 @\"__hugr__.m.f.<locals>.other_helper.9\"(i64 %0)\n",
        )
        .to_string();
        let normalized = normalize_pecos_helper_symbols_in_llvm(llvm);
        assert!(
            normalized
                .contains("declare { i64, i64 } @pecos_qis_runtime_barrier_qubits2_hugr(i64, i64)")
        );
        assert!(normalized.contains(
            "%1 = call { i64, i64 } @pecos_qis_runtime_barrier_qubits2_hugr(i64 %0, i64 %2)"
        ));
        assert!(normalized.contains(
            "%q = call i64 @pecos_qis_trace_metadata_qubit_hugr(i64 %0, i8* %1, i8* %2)"
        ));
        // PECOS helpers lose the private prefix entirely (no `<locals>` residue).
        assert!(!normalized.contains("<locals>.pecos_qis_"));
        // A non-PECOS quoted symbol is left untouched (still quoted).
        assert!(normalized.contains("@\"__hugr__.m.f.<locals>.other_helper.9\""));
    }
}

/// Compile HUGR bytes to LLVM IR string
///
/// This is the main entry point for the compiler.
///
/// # Errors
/// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
pub fn compile_hugr_bytes_to_string(hugr_bytes: &[u8]) -> Result<String, PecosError> {
    compile_hugr_bytes_to_string_with_options(hugr_bytes, &CompileArgs::default())
}

/// Compile HUGR bytes to LLVM IR string with custom options
///
/// # Errors
/// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
pub fn compile_hugr_bytes_to_string_with_options(
    hugr_bytes: &[u8],
    args: &CompileArgs,
) -> Result<String, PecosError> {
    log::info!("Compiling HUGR to LLVM IR");

    // Read HUGR
    let mut hugr = read_hugr_envelope(hugr_bytes)
        .map_err(|e| PecosError::Generic(format!("Failed to read HUGR: {e}")))?;

    // Create LLVM context
    let context = Context::create();

    // Compile
    let module = compile(args, &context, &mut hugr)
        .map_err(|e| PecosError::Generic(format!("Compilation failed: {e}")))?;

    // Get the module string
    let mut llvm_str = module.to_string();
    llvm_str = normalize_pecos_helper_symbols_in_llvm(llvm_str);

    // Workaround: Manually add the EntryPoint attribute if it's missing
    // This is needed because inkwell sometimes doesn't properly serialize string attributes
    let entry_name = args.entry.as_ref().map_or(LLVM_MAIN, |x| x.as_ref());
    if !llvm_str.contains("\"EntryPoint\"")
        && llvm_str.contains(&format!("define i64 @{entry_name}"))
    {
        // Find where entry is defined and add an attribute reference
        llvm_str = llvm_str.replace(
            &format!("define i64 @{entry_name}(i64 %0) local_unnamed_addr {{"),
            &format!("define i64 @{entry_name}(i64 %0) local_unnamed_addr #1 {{"),
        );
        // Add the attribute definition at the end
        if !llvm_str.contains("attributes #1") {
            llvm_str.push_str("\nattributes #1 = { \"EntryPoint\" }\n");
        }
    }

    Ok(llvm_str)
}

/// Compile HUGR bytes to LLVM bitcode
///
/// # Errors
/// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
pub fn compile_hugr_bytes_to_bitcode(hugr_bytes: &[u8]) -> Result<Vec<u8>, PecosError> {
    compile_hugr_bytes_to_bitcode_with_options(hugr_bytes, &CompileArgs::default())
}

/// Get the optimization level for the given integer value
///
/// Maps integer values to LLVM optimization levels:
/// - 0 -> None (O0)
/// - 1 -> Less (O1)
/// - 2 -> Default (O2)
/// - 3 -> Aggressive (O3)
///
/// # Errors
/// Returns an error if the optimization level is invalid (not 0-3)
pub fn get_opt_level(opt_level: u32) -> Result<OptimizationLevel> {
    match opt_level {
        0 => Ok(OptimizationLevel::None),
        1 => Ok(OptimizationLevel::Less),
        2 => Ok(OptimizationLevel::Default),
        3 => Ok(OptimizationLevel::Aggressive),
        _ => Err(anyhow!(
            "Invalid optimization level: {opt_level}. Must be 0-3"
        )),
    }
}

/// Compile HUGR bytes to LLVM bitcode with custom options
///
/// # Errors
/// Returns an error if HUGR parsing, validation, or LLVM compilation fails.
pub fn compile_hugr_bytes_to_bitcode_with_options(
    hugr_bytes: &[u8],
    args: &CompileArgs,
) -> Result<Vec<u8>, PecosError> {
    log::info!("Compiling HUGR to LLVM bitcode");

    // Read HUGR
    let mut hugr = read_hugr_envelope(hugr_bytes)
        .map_err(|e| PecosError::Generic(format!("Failed to read HUGR: {e}")))?;

    // Create LLVM context
    let context = Context::create();

    // Compile
    let module = compile(args, &context, &mut hugr)
        .map_err(|e| PecosError::Generic(format!("Compilation failed: {e}")))?;

    // Write to memory buffer and get bitcode. `as_slice()` includes LLVM's
    // trailing C-string NUL, which is not part of the bitcode stream.
    let buffer = module.write_bitcode_to_memory();
    let bitcode = buffer.as_slice();
    Ok(bitcode[..bitcode.len().saturating_sub(1)].to_vec())
}

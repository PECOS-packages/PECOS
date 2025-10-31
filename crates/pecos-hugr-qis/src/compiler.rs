//! HUGR to QIS LLVM IR compiler
//!
//! This module provides HUGR to LLVM IR compilation that generates
//! Selene QIS-compatible LLVM IR. It matches the full functionality
//! of tket2's qis-compiler but without Python bindings.

use anyhow::{Result, anyhow};
use pecos_core::errors::PecosError;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use itertools::Itertools;
use tket::hugr::envelope::EnvelopeConfig;
#[allow(deprecated)]
use tket::hugr::llvm::extension::int::IntCodegenExtension;
use tket::hugr::llvm::inkwell::OptimizationLevel;
use tket::hugr::llvm::inkwell::context::Context;
use tket::hugr::llvm::inkwell::module::Module;
use tket::hugr::llvm::inkwell::passes::PassBuilderOptions;
use tket::hugr::llvm::inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use tket::hugr::llvm::utils::fat::FatExt as _;
use tket::hugr::llvm::utils::inline_constant_functions;
use tket::hugr::llvm::{
    CodegenExtsBuilder,
    custom::CodegenExtsMap,
    emit::{EmitHugr, Namer},
};
use tket::hugr::ops::DataflowParent;
use tket::hugr::{Hugr, HugrView, Node};
use tket::llvm::rotation::RotationCodegenExtension;
use tket_qsystem::QSystemPass;
use tket_qsystem::llvm::array_utils::ArrayLowering;
use tket_qsystem::llvm::futures::FuturesCodegenExtension;
use tket_qsystem::llvm::{
    debug::DebugCodegenExtension, prelude::QISPreludeCodegen, qsystem::QSystemCodegenExtension,
    random::RandomCodegenExtension, result::ResultsCodegenExtension, utils::UtilsCodegenExtension,
};
use tracing::{Level, event, instrument};

// Import read_hugr_envelope from utils module
use crate::utils::read_hugr_envelope;

const LLVM_MAIN: &str = "qmain";
const METADATA: &[(&str, &[&str])] = &[("name", &["mainlib"])];

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
}

impl Default for CompileArgs {
    fn default() -> Self {
        Self {
            entry: None,
            name: "hugr".to_string(),
            save_hugr: None,
            target_triple: None,
            opt_level: OptimizationLevel::Default,
        }
    }
}

/// Process HUGR by applying required passes
fn process_hugr(hugr: &mut Hugr) -> Result<()> {
    QSystemPass::default().run(hugr)?;
    inline_constant_functions(hugr)?;
    Ok(())
}

/// Build codegen extensions for LLVM generation
#[allow(deprecated)]
fn codegen_extensions() -> CodegenExtsMap<'static, Hugr> {
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
        .add_extension(QSystemCodegenExtension::from(pcg.clone()))
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
        .emit_module(hugr.try_fat(hugr.module_root()).unwrap())?
        .finish())
}

/// Given an LLVM context and hugr, compile to an LLVM module
fn get_module_with_std_exts<'c>(
    args: &CompileArgs,
    context: &'c Context,
    namer: Rc<Namer>,
    hugr: &'c mut Hugr,
) -> Result<Module<'c>> {
    process_hugr(hugr)?;

    if let Some(filename) = &args.save_hugr {
        let file = fs::File::create(filename)?;
        hugr.store(file, EnvelopeConfig::text())?;
    }

    get_hugr_llvm_module(
        context,
        namer,
        hugr,
        &args.name,
        Rc::new(codegen_extensions()),
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

    let initial_tc = entry_fun.get_nth_param(0).unwrap().into_int_value();
    let hugr_main = module
        .get_function(hugr_entry)
        .ok_or_else(|| anyhow!("Entrypoint function '{hugr_entry}' not found in Module"))?;

    builder.build_call(setup, &[initial_tc.into()], "")?;
    builder.build_call(hugr_main, &[], "")?;
    let tc = builder
        .build_call(teardown, &[], "")?
        .try_as_basic_value()
        .left()
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
    Target::initialize_native(&InitializationConfig::default()).unwrap();
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
#[instrument(skip(args, ctx, hugr), parent = None)]
fn compile<'c, 'hugr: 'c>(
    args: &CompileArgs,
    ctx: &'c Context,
    hugr: &'hugr mut Hugr,
) -> Result<Module<'c>> {
    event!(Level::DEBUG, "starting primary compilation");
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

    // Write to memory buffer and get bitcode
    let buffer = module.write_bitcode_to_memory();
    Ok(buffer.as_slice().to_vec())
}

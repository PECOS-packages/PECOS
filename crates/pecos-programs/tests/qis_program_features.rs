//! Tests to verify all `Qis` features work correctly

use pecos_programs::{Qis, QisContent};

#[test]
fn test_qis_ir_methods() {
    let ir = "define void @main() { ret void }";

    // Test from_string
    let prog1 = Qis::from_string(ir);
    assert!(prog1.is_ir());
    assert!(!prog1.is_bitcode());
    assert_eq!(prog1.ir(), Some(ir));
    assert_eq!(prog1.bitcode(), None);

    // Test from_ir (alias)
    let prog2 = Qis::from_ir(ir);
    assert_eq!(prog1, prog2);
}

#[test]
fn test_qis_bitcode_methods() {
    let bitcode = vec![0xDE, 0xC0, 0xDE, 0x42, 0x01, 0x0C];

    // Test from_bitcode
    let prog = Qis::from_bitcode(bitcode.clone());
    assert!(!prog.is_ir());
    assert!(prog.is_bitcode());
    assert_eq!(prog.ir(), None);
    assert_eq!(prog.bitcode(), Some(bitcode.as_slice()));
}

#[test]
fn test_qis_file_auto_detection() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;

    // Test .ll file (IR text)
    let ll_path = temp_dir.path().join("test.ll");
    let ir_content = "define void @test() { ret void }";
    std::fs::write(&ll_path, ir_content)?;

    let ll_prog = Qis::from_file(&ll_path)?;
    assert!(ll_prog.is_ir());
    assert_eq!(ll_prog.ir(), Some(ir_content));

    // Test .bc file (bitcode)
    let bc_path = temp_dir.path().join("test.bc");
    let bc_content = vec![0xDE, 0xC0, 0xDE, 0x42];
    std::fs::write(&bc_path, &bc_content)?;

    let bc_prog = Qis::from_file(&bc_path)?;
    assert!(bc_prog.is_bitcode());
    assert_eq!(bc_prog.bitcode(), Some(bc_content.as_slice()));

    // Test file with no extension (defaults to IR)
    let no_ext_path = temp_dir.path().join("test");
    std::fs::write(&no_ext_path, ir_content)?;

    let no_ext_prog = Qis::from_file(&no_ext_path)?;
    assert!(no_ext_prog.is_ir());
    assert_eq!(no_ext_prog.ir(), Some(ir_content));

    Ok(())
}

#[test]
fn test_qis_specific_file_methods() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;

    // Test from_ir_file
    let ir_path = temp_dir.path().join("test.ll");
    let ir_content = "define void @test() { ret void }";
    std::fs::write(&ir_path, ir_content)?;

    let ir_prog = Qis::from_ir_file(&ir_path)?;
    assert!(ir_prog.is_ir());
    assert_eq!(ir_prog.ir(), Some(ir_content));

    // Test from_bitcode_file
    let bc_path = temp_dir.path().join("test.bc");
    let bc_content = vec![0xBC, 0xC0, 0xDE, 0x35, 0x14];
    std::fs::write(&bc_path, &bc_content)?;

    let bc_prog = Qis::from_bitcode_file(&bc_path)?;
    assert!(bc_prog.is_bitcode());
    assert_eq!(bc_prog.bitcode(), Some(bc_content.as_slice()));

    Ok(())
}

#[test]
fn test_qis_display() {
    // IR display shows the content
    let ir = "define void @main() { ret void }";
    let ir_prog = Qis::from_ir(ir);
    assert_eq!(format!("{ir_prog}"), ir);

    // Bitcode display shows size info
    let bc = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
    let bc_prog = Qis::from_bitcode(bc);
    assert_eq!(format!("{bc_prog}"), "Qis(bitcode, 6 bytes)");
}

#[test]
fn test_qis_content_enum() {
    let ir = "define void @main() {}";
    let prog1 = Qis::from_ir(ir);

    match &prog1.content {
        QisContent::Ir(content) => assert_eq!(content, ir),
        QisContent::Bitcode(_) => panic!("Expected IR, got bitcode"),
    }

    let bc = vec![1, 2, 3, 4];
    let prog2 = Qis::from_bitcode(bc.clone());

    match &prog2.content {
        QisContent::Ir(_) => panic!("Expected bitcode, got IR"),
        QisContent::Bitcode(content) => assert_eq!(content, &bc),
    }
}

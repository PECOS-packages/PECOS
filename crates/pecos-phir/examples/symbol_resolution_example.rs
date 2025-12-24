//! Example showing how symbol resolution works in PMIR without a separate AST

use std::collections::BTreeMap;

fn main() {
    println!("=== Symbol Resolution in PMIR ===\n");

    // Example: Parse and resolve this quantum program:
    // ```
    // module @quantum {
    //   global @phase : f64 = 0.5
    //
    //   func @prepare_state(%q: !quantum.qubit) {
    //     %theta = global.load @phase : f64
    //     quantum.ry %theta, %q : f64, !quantum.qubit
    //   }
    //
    //   func @main() {
    //     %q = quantum.alloc : !quantum.qubit
    //     call @prepare_state(%q) : (!quantum.qubit) -> ()
    //     %result = quantum.measure %q : !quantum.qubit -> i1
    //     return %result : i1
    //   }
    // }
    // ```

    example_multi_pass_resolution();
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum DeclKind {
    Global { ty: String },
    Function { signature: String },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DeclInfo {
    name: String,
    kind: DeclKind,
    location: usize, // line number
}

#[derive(Debug)]
struct UnresolvedRef {
    name: String,
    location: usize,
    context: String,
}

#[allow(clippy::too_many_lines)] // Example code demonstrating multiple resolution passes
fn example_multi_pass_resolution() {
    // Simulated parsing passes

    println!("Pass 1: Collect Declarations");
    println!("----------------------------");

    let mut declarations = BTreeMap::new();

    // First pass: collect all declarations
    declarations.insert(
        "@phase",
        DeclInfo {
            name: "@phase".to_string(),
            kind: DeclKind::Global {
                ty: "f64".to_string(),
            },
            location: 2,
        },
    );

    declarations.insert(
        "@prepare_state",
        DeclInfo {
            name: "@prepare_state".to_string(),
            kind: DeclKind::Function {
                signature: "(!quantum.qubit) -> ()".to_string(),
            },
            location: 4,
        },
    );

    declarations.insert(
        "@main",
        DeclInfo {
            name: "@main".to_string(),
            kind: DeclKind::Function {
                signature: "() -> i1".to_string(),
            },
            location: 9,
        },
    );

    for (name, decl) in &declarations {
        println!("  Found declaration: {} = {:?}", name, decl.kind);
    }

    println!("\nPass 2: Parse Function Bodies with Unresolved Refs");
    println!("--------------------------------------------------");

    let mut unresolved_refs = vec![];

    // Parsing @prepare_state
    println!("  Parsing @prepare_state:");
    unresolved_refs.push(UnresolvedRef {
        name: "@phase".to_string(),
        location: 5,
        context: "global.load".to_string(),
    });
    println!("    - Found unresolved ref: @phase");

    // Parsing @main
    println!("  Parsing @main:");
    unresolved_refs.push(UnresolvedRef {
        name: "@prepare_state".to_string(),
        location: 11,
        context: "call".to_string(),
    });
    println!("    - Found unresolved ref: @prepare_state");

    println!("\nPass 3: Resolve References");
    println!("--------------------------");

    for unresolved in &unresolved_refs {
        if let Some(decl) = declarations.get(unresolved.name.as_str()) {
            println!(
                "  Resolved {} at line {} -> {:?}",
                unresolved.name, unresolved.location, decl.kind
            );

            // Type checking would happen here
            match (unresolved.context.as_str(), &decl.kind) {
                ("global.load", DeclKind::Global { .. }) => {
                    println!("    Type check: global.load is valid for global");
                }
                ("call", DeclKind::Function { .. }) => {
                    println!("    Type check: call is valid for function");
                }
                _ => {
                    println!(
                        "    Type error: {} cannot be used in {} context",
                        unresolved.name, unresolved.context
                    );
                }
            }
        } else {
            println!("  Error: {} not found in scope", unresolved.name);
        }
    }

    println!("\nPass 4: Lower to Final PMIR");
    println!("---------------------------");
    println!("  - Replace UnresolvedCall with proper func.call");
    println!("  - Replace UnresolvedRef with global.load or SSA value");
    println!("  - All symbols now resolved!");

    // Show how scoping works
    println!("\n\nScoped Symbol Resolution");
    println!("========================");

    example_scoped_resolution();
}

fn example_scoped_resolution() {
    #[derive(Debug)]
    struct Scope {
        level: usize,
        symbols: BTreeMap<String, String>, // name -> type
        parent: Option<Box<Scope>>,
    }

    impl Scope {
        fn new(level: usize) -> Self {
            Self {
                level,
                symbols: BTreeMap::new(),
                parent: None,
            }
        }

        fn with_parent(level: usize, parent: Scope) -> Self {
            Self {
                level,
                symbols: BTreeMap::new(),
                parent: Some(Box::new(parent)),
            }
        }

        fn lookup(&self, name: &str) -> Option<(usize, &String)> {
            if let Some(ty) = self.symbols.get(name) {
                Some((self.level, ty))
            } else if let Some(parent) = &self.parent {
                parent.lookup(name)
            } else {
                None
            }
        }
    }

    // Example with nested scopes:
    // ```
    // func @nested(%x: i32) {
    //   %a = constant 1 : i32
    //   scf.if %condition {
    //     %b = constant 2 : i32
    //     %sum1 = addi %a, %b : i32  // Can see %a from outer scope
    //     scf.if %inner_cond {
    //       %c = constant 3 : i32
    //       %sum2 = addi %b, %c : i32  // Can see %b from parent
    //     }
    //     // %c not visible here
    //   }
    //   // %b not visible here
    // }
    // ```

    // Build scope chain
    let mut module_scope = Scope::new(0);
    module_scope
        .symbols
        .insert("@nested".to_string(), "func".to_string());

    let mut func_scope = Scope::with_parent(1, module_scope);
    func_scope
        .symbols
        .insert("%x".to_string(), "i32".to_string());
    func_scope
        .symbols
        .insert("%a".to_string(), "i32".to_string());

    let mut if_scope = Scope::with_parent(2, func_scope);
    if_scope.symbols.insert("%b".to_string(), "i32".to_string());

    let inner_if_scope = Scope::with_parent(3, if_scope);

    // Test lookups
    println!("  Testing scope chain lookups:");

    let test_lookups = vec![
        ("%x", &inner_if_scope),
        ("%a", &inner_if_scope),
        ("%b", &inner_if_scope),
        ("@nested", &inner_if_scope),
    ];

    for (name, scope) in test_lookups {
        if let Some((level, ty)) = scope.lookup(name) {
            println!("    {name} found at scope level {level} with type {ty}");
        } else {
            println!("    {name} not found!");
        }
    }

    println!("\n  Scoped resolution working correctly!");
}

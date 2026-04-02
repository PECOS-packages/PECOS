//! Implementation of the `migrate` command
//!
//! Moves legacy top-level installs (LLVM, CUDA, cuQuantum) from `~/.pecos/`
//! into `~/.pecos/deps/` to match the new directory layout.

use pecos_build::Result;
use pecos_build::home::{find_legacy_deps, migrate_legacy_dep};

/// Run the migrate command.
pub fn run() -> Result<()> {
    let legacy = find_legacy_deps()?;

    if legacy.is_empty() {
        println!("Nothing to migrate. All dependencies are already under ~/.pecos/deps/.");
        return Ok(());
    }

    println!("Migrating legacy dependencies to ~/.pecos/deps/:");
    for dep in &legacy {
        print!(
            "  {} : {} -> {}",
            dep.name,
            dep.old.display(),
            dep.new.display()
        );
        migrate_legacy_dep(dep)?;
        println!(" ... done");
    }

    println!();
    println!(
        "Migration complete. You may want to run `pecos llvm configure` to update .cargo/config.toml."
    );

    Ok(())
}

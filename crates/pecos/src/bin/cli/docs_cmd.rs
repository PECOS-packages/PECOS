//! Implementation of the `docs` subcommand (documentation server)

use pecos_build::Result;
use pecos_build::errors::Error;
use std::net::TcpStream;
use std::process::Command;
use std::thread;
use std::time::Duration;

/// Check if a port is already in use
fn is_port_in_use(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// Wait for server to be ready (port to be listening)
fn wait_for_server(port: u16, timeout_secs: u64) -> bool {
    for _ in 0..timeout_secs {
        if is_port_in_use(port) {
            return true;
        }
        thread::sleep(Duration::from_secs(1));
    }
    false
}

/// Open URL in default browser (cross-platform)
fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .is_ok()
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn().is_ok()
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(url).spawn().is_ok()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        let _ = url;
        false
    }
}

/// Run the docs server
pub fn run(port: u16, no_browser: bool) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}");

    // Check if port is already in use
    if is_port_in_use(port) {
        println!("Error: Port {port} is already in use.");
        println!();
        println!("Possible causes:");
        println!("  - Another mkdocs server is already running");
        println!("  - Another application is using port {port}");
        println!();
        println!("Solutions:");
        println!("  - Use a different port: pecos docs --port {}", port + 1);
        println!("  - Find and stop the existing process:");
        println!("      lsof -i :{port}  (Linux/macOS)");
        println!("      netstat -ano | findstr :{port}  (Windows)");
        println!(
            "      Get-NetTCPConnection -LocalPort {port} | Select-Object OwningProcess  (PowerShell)"
        );
        return Err(Error::Config(format!("Port {port} is already in use")));
    }

    // Start browser opener in background thread (before blocking on server)
    if !no_browser {
        let url_clone = url.clone();
        thread::spawn(move || {
            if wait_for_server(port, 30) && !open_browser(&url_clone) {
                println!("Could not open browser. Visit {url_clone}");
            }
        });
    }

    // Run mkdocs server (inherits stdio for colored output)
    // This blocks until the server exits (Ctrl+C)
    let status = Command::new("uv")
        .args(["run", "mkdocs", "serve", "-a", &format!("127.0.0.1:{port}")])
        .status()
        .map_err(|e| Error::Config(format!("Failed to start mkdocs server: {e}")))?;

    if !status.success() {
        return Err(Error::Config("mkdocs server exited with error".to_string()));
    }

    Ok(())
}

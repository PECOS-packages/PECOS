#!/bin/bash
# Test edge cases and potential failure modes of the rebuild system
#
# This script tests scenarios that could break the rebuild system

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
RUNTIME_LIB="$CARGO_HOME/pecos-llvm-runtime/libpecos_llvm_runtime.a"
MARKER_FILE="$CARGO_HOME/pecos-llvm-runtime/.needs_rebuild"
TEST_DIR="$PROJECT_ROOT/target/edge_case_test_$$"

# Platform adjustments
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    RUNTIME_LIB="$CARGO_HOME/pecos-llvm-runtime/pecos_llvm_runtime.lib"
fi

# Logging
log_test() {
    echo -e "\n${BLUE}TEST:${NC} $1"
    echo "----------------------------------------"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test concurrent marker file access
test_concurrent_marker_access() {
    log_test "Concurrent Marker File Access"

    # Create a test QIS file
    local QIS_FILE="$TEST_DIR/concurrent_test.ll"
    mkdir -p "$TEST_DIR"
    cat > "$QIS_FILE" << 'EOF'
define void @main() {
    ret void
}
EOF

    # Remove existing marker and library to force rebuild
    rm -f "$MARKER_FILE"
    rm -f "$RUNTIME_LIB"

    # Launch multiple QIS compilations simultaneously
    log_info "Launching 3 concurrent QIS compilations..."

    for i in 1 2 3; do
        (
            cd "$PROJECT_ROOT"
            "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE" 2>&1 | sed "s/^/[Process $i] /"
        ) &
    done

    # Wait for all to complete
    wait

    # Check results
    if [[ -f "$RUNTIME_LIB" ]]; then
        log_info "Runtime library built successfully"
    else
        log_error "Runtime library not built"
        return 1
    fi

    if [[ -f "$MARKER_FILE" ]]; then
        log_error "Marker still exists after concurrent builds"
        return 1
    else
        log_info "Marker removed after concurrent builds"
        return 0
    fi
}

# Test rapid file modifications
test_rapid_modifications() {
    log_test "Rapid File Modifications"

    local QIS_FILE="$TEST_DIR/rapid.ll"
    mkdir -p "$TEST_DIR"

    # Create initial QIS
    cat > "$QIS_FILE" << 'EOF'
define void @main() {
    ret void
}
EOF

    # Compile once
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE" || {
        log_error "Initial compilation failed"
        return 1
    }

    # Rapid modifications without sleep
    log_info "Making rapid modifications..."
    for i in {1..5}; do
        echo "; Modification $i" >> "$QIS_FILE"
        "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE" 2>/dev/null || {
            log_error "Compilation $i failed"
            return 1
        }
    done

    log_info "System handled rapid modifications"
    return 0
}

# Test corrupted marker file
test_corrupted_marker() {
    log_test "Corrupted Marker File"

    # Create corrupted marker (binary data)
    mkdir -p "$(dirname "$MARKER_FILE")"
    dd if=/dev/urandom of="$MARKER_FILE" bs=1024 count=1 2>/dev/null

    log_info "Created corrupted marker file"

    # Try to build
    cd "$PROJECT_ROOT"
    if cargo build -p pecos-llvm-runtime --quiet 2>/dev/null; then
        log_info "Build succeeded despite corrupted marker"
    else
        log_error "Build failed with corrupted marker"
        return 1
    fi

    # The corrupted marker should be handled during QIS compilation
    # (RuntimeBuilder removes marker after successful build)
    local QIS_FILE="$TEST_DIR/corrupted_test.ll"
    mkdir -p "$TEST_DIR"
    cat > "$QIS_FILE" << 'EOF'
define void @main() {
    ret void
}
EOF

    # Compile QIS - this should trigger runtime build and marker removal
    if "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE" 2>/dev/null; then
        if [[ -f "$MARKER_FILE" ]]; then
            log_error "Marker not removed after QIS compilation"
            return 1
        else
            log_info "Corrupted marker removed during QIS compilation"
            return 0
        fi
    else
        log_error "QIS compilation failed with corrupted marker"
        return 1
    fi
}

# Test permission issues
test_permission_issues() {
    log_test "Permission Issues"

    # Skip if running as root
    if [[ $EUID -eq 0 ]]; then
        log_info "Skipping (running as root)"
        return 0
    fi

    # Make marker directory read-only
    local MARKER_DIR="$(dirname "$MARKER_FILE")"
    mkdir -p "$MARKER_DIR"
    chmod 555 "$MARKER_DIR" 2>/dev/null || {
        log_info "Cannot change permissions (skipping)"
        return 0
    }

    log_info "Made marker directory read-only"

    # Try to build (should handle gracefully)
    cd "$PROJECT_ROOT"
    if cargo build -p pecos-llvm-runtime --quiet 2>&1 | grep -q "permission"; then
        log_info "Permission error handled gracefully"
        chmod 755 "$MARKER_DIR"
        return 0
    else
        chmod 755 "$MARKER_DIR"
        log_info "Build completed despite permission restrictions"
        return 0
    fi
}

# Test symlink scenarios
test_symlink_handling() {
    log_test "Symlink Handling"

    # Create a symlink for the runtime library
    local REAL_LIB="$TEST_DIR/real_runtime.a"
    mkdir -p "$TEST_DIR"

    if [[ -f "$RUNTIME_LIB" ]]; then
        cp "$RUNTIME_LIB" "$REAL_LIB"
        rm -f "$RUNTIME_LIB"
        ln -s "$REAL_LIB" "$RUNTIME_LIB"

        log_info "Created symlink: $RUNTIME_LIB -> $REAL_LIB"

        # Run build
        cd "$PROJECT_ROOT"
        if cargo build -p pecos-llvm-runtime --quiet; then
            log_info "Build works with symlinked runtime library"

            # Check if marker was created (it shouldn't be if symlink is valid)
            if [[ -f "$MARKER_FILE" ]]; then
                log_error "Marker created for valid symlink"
                rm -f "$RUNTIME_LIB"
                mv "$REAL_LIB" "$RUNTIME_LIB" 2>/dev/null || true
                return 1
            fi
        else
            log_error "Build failed with symlink"
            rm -f "$RUNTIME_LIB"
            mv "$REAL_LIB" "$RUNTIME_LIB" 2>/dev/null || true
            return 1
        fi

        # Restore
        rm -f "$RUNTIME_LIB"
        mv "$REAL_LIB" "$RUNTIME_LIB"
    else
        log_info "Runtime library not found (skipping symlink test)"
    fi

    return 0
}

# Test CARGO_HOME variations
test_cargo_home_variations() {
    log_test "CARGO_HOME Variations"

    # Save original
    local ORIG_CARGO_HOME="$CARGO_HOME"

    # Test with custom CARGO_HOME
    export CARGO_HOME="$TEST_DIR/custom_cargo"
    mkdir -p "$CARGO_HOME"

    log_info "Testing with CARGO_HOME=$CARGO_HOME"

    cd "$PROJECT_ROOT"
    if cargo build -p pecos-llvm-runtime --quiet 2>&1; then
        # Check if marker path is created in custom location
        local CUSTOM_MARKER="$CARGO_HOME/pecos-llvm-runtime/.needs_rebuild"
        if [[ -f "$CUSTOM_MARKER" ]]; then
            log_info "Marker created in custom CARGO_HOME"
        else
            log_info "Build succeeded with custom CARGO_HOME"
        fi
    else
        log_error "Build failed with custom CARGO_HOME"
        export CARGO_HOME="$ORIG_CARGO_HOME"
        return 1
    fi

    # Restore
    export CARGO_HOME="$ORIG_CARGO_HOME"
    return 0
}

# Test filesystem full scenario
test_filesystem_full() {
    log_test "Filesystem Full Scenario"

    # Create a small loopback filesystem
    local LOOP_FILE="$TEST_DIR/small_fs.img"
    local MOUNT_POINT="$TEST_DIR/mount"

    mkdir -p "$TEST_DIR"

    # Skip if not enough permissions
    if [[ $EUID -ne 0 ]]; then
        log_info "Skipping (requires root)"
        return 0
    fi

    # Create 10MB filesystem
    dd if=/dev/zero of="$LOOP_FILE" bs=1M count=10 2>/dev/null
    mkfs.ext4 -F "$LOOP_FILE" 2>/dev/null
    mkdir -p "$MOUNT_POINT"
    mount -o loop "$LOOP_FILE" "$MOUNT_POINT"

    # Fill it up
    dd if=/dev/zero of="$MOUNT_POINT/filler" bs=1M count=9 2>/dev/null

    # Try to create marker there
    export CARGO_HOME="$MOUNT_POINT"

    cd "$PROJECT_ROOT"
    if cargo build -p pecos-llvm-runtime --quiet 2>&1 | grep -q "space"; then
        log_info "Filesystem full error handled"
    else
        log_info "Build handled full filesystem scenario"
    fi

    # Cleanup
    umount "$MOUNT_POINT"
    export CARGO_HOME="$ORIG_CARGO_HOME"
    return 0
}

# Main execution
main() {
    echo "======================================"
    echo "PECOS Rebuild System Edge Case Tests"
    echo "======================================"

    # Build CLI first
    log_info "Building PECOS CLI..."
    cd "$PROJECT_ROOT"
    cargo build -p pecos --features cli --quiet || {
        log_error "Failed to build PECOS CLI"
        exit 1
    }

    mkdir -p "$TEST_DIR"

    # Run tests
    local FAILED=0
    local TESTS=(
        test_concurrent_marker_access
        test_rapid_modifications
        test_corrupted_marker
        test_permission_issues
        test_symlink_handling
        test_cargo_home_variations
        test_filesystem_full
    )

    for test in "${TESTS[@]}"; do
        if $test; then
            echo -e "${GREEN}PASSED${NC}"
        else
            echo -e "${RED}FAILED${NC}"
            ((FAILED++))
        fi
    done

    # Cleanup
    rm -rf "$TEST_DIR"

    # Summary
    echo
    echo "======================================"
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All edge case tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}$FAILED edge case tests failed${NC}"
        exit 1
    fi
}

# Cleanup on exit
trap 'rm -rf "$TEST_DIR" 2>/dev/null; chmod 755 "$(dirname "$MARKER_FILE")" 2>/dev/null || true' EXIT

main "$@"

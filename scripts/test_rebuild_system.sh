#!/bin/bash
# Test script for the PECOS QIS rebuild system
#
# This script tests the complete rebuild system including:
# - build.rs marker file creation
# - Runtime library building
# - QIS executable caching and rebuilding

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
RUNTIME_LIB="$CARGO_HOME/pecos-llvm-runtime/libpecos_llvm_runtime.a"
MARKER_FILE="$CARGO_HOME/pecos-llvm-runtime/.needs_rebuild"
TEST_DIR="$PROJECT_ROOT/target/rebuild_test_$$"
QIS_FILE="$TEST_DIR/test.ll"

# Platform-specific adjustments
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    RUNTIME_LIB="$CARGO_HOME/pecos-llvm-runtime/pecos_llvm_runtime.lib"
fi

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_file() {
    if [[ -f "$1" ]]; then
        echo -e "  $1 exists"
        return 0
    else
        echo -e "  $1 missing"
        return 1
    fi
}

get_mtime() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f %m "$1" 2>/dev/null || echo "0"
    else
        stat -c %Y "$1" 2>/dev/null || echo "0"
    fi
}

create_test_qis() {
    mkdir -p "$TEST_DIR"
    cat > "$QIS_FILE" << 'EOF'
%Qubit = type opaque
%Result = type opaque

declare void @__quantum__rt__initialize(i8*)
declare %Qubit* @__quantum__rt__qubit_allocate()
declare void @__quantum__rt__qubit_release(%Qubit*)
declare void @__quantum__qis__h__body(%Qubit*)

define void @main() {
entry:
    call void @__quantum__rt__initialize(i8* null)
    %q = call %Qubit* @__quantum__rt__qubit_allocate()
    call void @__quantum__qis__h__body(%Qubit* %q)
    call void @__quantum__rt__qubit_release(%Qubit* %q)
    ret void
}
EOF
}

# Test functions
test_marker_creation() {
    log_info "Testing marker file creation..."

    # Clean state
    rm -f "$MARKER_FILE"

    # Case 1: Missing runtime library
    if [[ -f "$RUNTIME_LIB" ]]; then
        mv "$RUNTIME_LIB" "$RUNTIME_LIB.backup"
    fi

    log_info "Running cargo build with missing runtime library..."
    cd "$PROJECT_ROOT"
    # Force a rebuild by cleaning first
    cargo clean -p pecos-llvm-runtime --quiet
    cargo build -p pecos-llvm-runtime --quiet

    if check_file "$MARKER_FILE"; then
        log_info "Marker created for missing library"
    else
        log_error "Marker not created for missing library"
        # Restore if we failed
        if [[ -f "$RUNTIME_LIB.backup" ]]; then
            mv "$RUNTIME_LIB.backup" "$RUNTIME_LIB"
        fi
        return 1
    fi

    # Restore library
    if [[ -f "$RUNTIME_LIB.backup" ]]; then
        mv "$RUNTIME_LIB.backup" "$RUNTIME_LIB"
    fi

    # Case 2: Up-to-date library
    rm -f "$MARKER_FILE"
    log_info "Running cargo build with up-to-date library..."
    cargo build -p pecos-llvm-runtime --quiet

    if [[ -f "$MARKER_FILE" ]]; then
        log_error "Marker created when library is up-to-date"
        return 1
    else
        log_info "No marker created when up-to-date"
    fi

    return 0
}

test_runtime_building() {
    log_info "Testing runtime library building..."

    # Ensure marker exists
    mkdir -p "$(dirname "$MARKER_FILE")"
    echo "rebuild" > "$MARKER_FILE"

    # Remove library
    rm -f "$RUNTIME_LIB"

    # Create test QIS
    create_test_qis

    log_info "Compiling QIS (should trigger runtime build)..."
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE" || {
        log_error "QIS compilation failed"
        return 1
    }

    # Check results
    if check_file "$RUNTIME_LIB"; then
        log_info "Runtime library built"
    else
        log_error "Runtime library not built"
        return 1
    fi

    if [[ -f "$MARKER_FILE" ]]; then
        log_error "Marker not removed after build"
        return 1
    else
        log_info "Marker removed after build"
    fi

    return 0
}

test_qis_caching() {
    log_info "Testing QIS executable caching..."

    create_test_qis
    local OUTPUT_DIR="$TEST_DIR/build"

    # First compilation
    log_info "First QIS compilation..."
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE"

    local LIB1="$OUTPUT_DIR/libtest.so"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LIB1="$OUTPUT_DIR/libtest.dylib"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        LIB1="$OUTPUT_DIR/test.dll"
    fi

    local MTIME1=$(get_mtime "$LIB1")
    log_info "Created: $LIB1 (mtime: $MTIME1)"

    # Wait to ensure timestamp difference
    sleep 2

    # Second compilation (no changes)
    log_info "Second compilation (should use cache)..."
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE"

    local MTIME2=$(get_mtime "$LIB1")
    if [[ "$MTIME1" == "$MTIME2" ]]; then
        log_info "Cached library used (same mtime)"
    else
        log_error "Library rebuilt unnecessarily"
        return 1
    fi

    # Modify QIS file
    log_info "Modifying QIS file..."
    echo "; Modified" >> "$QIS_FILE"
    sleep 1

    # Third compilation (should rebuild)
    log_info "Third compilation (should rebuild)..."
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE"

    local MTIME3=$(get_mtime "$LIB1")
    if [[ "$MTIME3" -gt "$MTIME2" ]]; then
        log_info "Library rebuilt after source change"
    else
        log_error "Library not rebuilt after source change"
        return 1
    fi

    return 0
}

test_source_change_flow() {
    log_info "Testing complete source change flow..."

    create_test_qis

    # Initial state
    rm -f "$MARKER_FILE"

    # Modify a source file
    local SRC_FILE="$PROJECT_ROOT/crates/pecos-llvm-runtime/src/lib.rs"
    local ORIG_CONTENT=$(cat "$SRC_FILE")

    log_info "Modifying pecos-llvm-runtime source file..."
    echo "// Test modification" >> "$SRC_FILE"

    # Run cargo build
    log_info "Running cargo build after source change..."
    cd "$PROJECT_ROOT"
    cargo build -p pecos-llvm-runtime --quiet

    if check_file "$MARKER_FILE"; then
        log_info "Marker created after source change"
    else
        log_error "Marker not created after source change"
        # Restore file
        echo "$ORIG_CONTENT" > "$SRC_FILE"
        return 1
    fi

    # Get runtime library mtime before
    local RT_MTIME_BEFORE=$(get_mtime "$RUNTIME_LIB")

    # Compile QIS (should rebuild runtime)
    log_info "Compiling QIS (should rebuild runtime)..."
    "$PROJECT_ROOT/target/debug/pecos" compile "$QIS_FILE"

    local RT_MTIME_AFTER=$(get_mtime "$RUNTIME_LIB")

    if [[ "$RT_MTIME_AFTER" -ge "$RT_MTIME_BEFORE" ]]; then
        log_info "Runtime library updated"
    else
        log_error "Runtime library not updated"
    fi

    if [[ -f "$MARKER_FILE" ]]; then
        log_error "Marker still exists after rebuild"
    else
        log_info "Marker removed after rebuild"
    fi

    # Restore source file
    echo "$ORIG_CONTENT" > "$SRC_FILE"

    return 0
}

# Main test execution
main() {
    echo "======================================"
    echo "PECOS QIS Rebuild System Test"
    echo "======================================"
    echo

    # Build the CLI first
    log_info "Building PECOS CLI..."
    cd "$PROJECT_ROOT"
    cargo build -p pecos-cli --quiet || {
        log_error "Failed to build PECOS CLI"
        exit 1
    }

    # Create test directory
    mkdir -p "$TEST_DIR"

    # Track test results
    local FAILED=0

    # Run tests
    echo
    echo "Test 1: Marker File Creation"
    echo "-----------------------------"
    if test_marker_creation; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
    fi

    echo
    echo "Test 2: Runtime Library Building"
    echo "--------------------------------"
    if test_runtime_building; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
    fi

    echo
    echo "Test 3: QIS Executable Caching"
    echo "------------------------------"
    if test_qis_caching; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
    fi

    echo
    echo "Test 4: Source Change Flow"
    echo "--------------------------"
    if test_source_change_flow; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
    fi

    # Cleanup
    rm -rf "$TEST_DIR"

    # Summary
    echo
    echo "======================================"
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}$FAILED tests failed${NC}"
        exit 1
    fi
}

# Handle interrupts
trap 'echo -e "\n${YELLOW}Interrupted. Cleaning up...${NC}"; rm -rf "$TEST_DIR"; exit 130' INT TERM

# Run main
main "$@"

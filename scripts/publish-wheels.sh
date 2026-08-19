#!/bin/bash
# Script to help publish PECOS wheels to PyPI from GitHub Actions artifacts
#
# Publish order matters: quantum-pecos pins pecos-rslib and pecos-rslib-llvm at
# exact versions, so the dependencies must exist on PyPI before quantum-pecos.
# All-packages mode preflights the complete artifact set and aborts on any
# missing package or declined prompt rather than continuing (a partial publish
# leaves quantum-pecos with unresolvable pins -- this has happened).

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ARTIFACT_FILE="pecos-distribution.zip"
DRY_RUN=false
PACKAGE=""
ASSUME_YES=false

ALL_PACKAGES=(pecos-rslib pecos-rslib-llvm quantum-pecos)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            ARTIFACT_FILE="$2"
            shift 2
            ;;
        -p|--package)
            PACKAGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -y|--yes)
            ASSUME_YES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --file FILE      Artifact zip OR extracted artifact directory (default: pecos-distribution.zip)"
            echo "  -p, --package PKG    Publish only specific package (pecos-rslib, pecos-rslib-llvm, or quantum-pecos)"
            echo "  --dry-run            Show what would be uploaded without actually uploading"
            echo "  -y, --yes            Skip per-package confirmation prompts (for non-interactive use)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Publish all packages"
            echo "  $0 -p pecos-rslib                    # Publish only pecos-rslib"
            echo "  $0 -p quantum-pecos --dry-run        # Dry run for quantum-pecos"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if the artifact exists: either the zip, or an already-extracted
# directory (`gh run download` auto-extracts artifacts).
if [ ! -f "$ARTIFACT_FILE" ] && [ ! -d "$ARTIFACT_FILE" ]; then
    echo -e "${RED}Error: Artifact '$ARTIFACT_FILE' not found!${NC}"
    echo "Please download the 'pecos-distribution' artifact from GitHub Actions."
    exit 1
fi

# Prompts need a terminal; refuse to guess in non-interactive contexts.
if [ "$ASSUME_YES" = false ] && [ "$DRY_RUN" = false ] && [ ! -t 0 ]; then
    echo -e "${RED}Error: stdin is not a terminal. Use --yes for non-interactive publishing.${NC}"
    exit 1
fi

# Check if uv is installed (preferred) or twine directly
if command -v uv &> /dev/null; then
    TWINE_CMD="uv run twine"
    echo -e "${GREEN}Using uv to run twine ($($TWINE_CMD --version 2>/dev/null | head -1))${NC}"
elif command -v twine &> /dev/null; then
    TWINE_CMD="twine"
    echo -e "${YELLOW}Using system twine ($(twine --version 2>/dev/null | head -1)); consider using uv${NC}"
else
    echo -e "${RED}Error: Neither uv nor twine is installed!${NC}"
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or install twine with: pip install twine"
    exit 1
fi

if [ -d "$ARTIFACT_FILE" ]; then
    # Already-extracted artifact directory: read it in place.
    echo -e "${GREEN}Using extracted distribution directory...${NC}"
    DIST_DIR="$ARTIFACT_FILE"
else
    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf -- "${TEMP_DIR:?}"' EXIT
    echo -e "${GREEN}Extracting distribution bundle...${NC}"
    unzip -q "$ARTIFACT_FILE" -d "$TEMP_DIR"
    DIST_DIR="$TEMP_DIR"
fi

# Resolve a package's distribution directory (with or without dist/ prefix).
package_dir_for() {
    local package_name=$1
    local dir="$DIST_DIR/$package_name"
    [ -d "$dir" ] || dir="$DIST_DIR/dist/$package_name"
    printf '%s\n' "$dir"
}

# Preflight one package: directory exists, contains only expected distribution
# files for that project, and report the single version they carry.
# Prints the version on stdout; fails the script otherwise.
preflight_package() {
    local package_name=$1
    local dir
    dir=$(package_dir_for "$package_name")
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: $package_name directory not found in distribution${NC}" >&2
        return 1
    fi
    local prefix="${package_name//-/_}"
    local versions=()
    local f base ver
    shopt -s nullglob
    local files=("$dir"/*)
    shopt -u nullglob
    if [ "${#files[@]}" -eq 0 ]; then
        echo -e "${RED}Error: no files found in $dir${NC}" >&2
        return 1
    fi
    for f in "${files[@]}"; do
        base=$(basename "$f")
        case "$base" in
            "$prefix"-*.whl|"$prefix"-*.tar.gz) ;;
            *)
                echo -e "${RED}Error: unexpected file in $dir: $base (expected only $prefix-*.whl / $prefix-*.tar.gz)${NC}" >&2
                return 1
                ;;
        esac
        ver=${base#"$prefix"-}
        ver=${ver%%-*}
        ver=${ver%.tar.gz}
        versions+=("$ver")
    done
    local unique
    unique=$(printf '%s\n' "${versions[@]}" | sort -u)
    if [ "$(printf '%s\n' "$unique" | wc -l)" -ne 1 ]; then
        echo -e "${RED}Error: mixed versions in $dir: ${unique//$'\n'/ }${NC}" >&2
        return 1
    fi
    printf '%s\n' "$unique"
}

# Run twine's own validation for one package.
#
# Deliberately separate from publish_package: every package must be validated before
# ANY package uploads. Validating inside the upload loop means a checker that rejects
# the last package only speaks up once the earlier ones are public, and a PyPI upload
# cannot be taken back. A twine too old for the metadata the build tools now emit is
# exactly that failure -- it rejected quantum-pecos while the two wheels it pins had
# already gone up.
check_package_distributions() {
    local package_name=$1
    local package_dir
    package_dir=$(package_dir_for "$package_name")

    # --strict: fail on warnings too. Current maturin/hatchling artifacts pass
    # strict checks cleanly, so any warning is a real signal.
    local check_output
    if check_output=$($TWINE_CMD check --strict "$package_dir"/* 2>&1); then
        echo -e "${GREEN}Distribution checks passed: $package_name${NC}"
    else
        echo "$check_output"
        echo -e "${RED}Distribution checks failed: $package_name${NC}"
        echo -e "${YELLOW}If this names a metadata version, the checker is behind the build tools:" \
             "upgrade it with 'uv tool upgrade twine' and re-run.${NC}"
        return 1
    fi
}

# Publish a package (assumed preflighted and checked). Fails the script on any error or
# on a declined prompt -- callers rely on this to abort dependent uploads.
publish_package() {
    local package_name=$1
    local package_dir
    package_dir=$(package_dir_for "$package_name")

    echo -e "\n${GREEN}=== Publishing $package_name ===${NC}"
    ls -la "$package_dir"

    if [ "$DRY_RUN" = true ]; then
        echo -e "\n${YELLOW}DRY RUN: Would upload the following files:${NC}"
        ls -1 "$package_dir"
        return 0
    fi

    echo -e "\n${GREEN}Uploading to PyPI...${NC}"
    if [ "$ASSUME_YES" = true ]; then
        REPLY="y"
    else
        # Full-line read: single-character reads mis-consume piped input. A
        # read failure (EOF) aborts loudly rather than guessing.
        if ! read -p "Are you sure you want to upload $package_name to PyPI? (y/N) " -r; then
            echo -e "${RED}Error: could not read confirmation (EOF); aborting.${NC}"
            return 1
        fi
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $TWINE_CMD upload "$package_dir"/*
        echo -e "${GREEN}Successfully uploaded $package_name!${NC}"
    else
        echo -e "${RED}Declined uploading $package_name; aborting (later packages depend on it).${NC}"
        return 1
    fi
}

# Verify an exact pin exists on PyPI (used to guard publishing quantum-pecos
# on its own: its exact pecos-rslib/-llvm pins must already be resolvable).
pin_exists_on_pypi() {
    local name=$1 version=$2
    curl -fsSL -o /dev/null "https://pypi.org/pypi/$name/$version/json"
}

# Main execution
if [ -n "$PACKAGE" ]; then
    if ! printf '%s\n' "${ALL_PACKAGES[@]}" | grep -qx "$PACKAGE"; then
        echo -e "${RED}Error: Invalid package name '$PACKAGE'${NC}"
        echo "Valid options are: ${ALL_PACKAGES[*]}"
        exit 1
    fi
    version=$(preflight_package "$PACKAGE")
    echo -e "${GREEN}Preflight OK: $PACKAGE $version${NC}"
    check_package_distributions "$PACKAGE"
    if [ "$PACKAGE" = "quantum-pecos" ] && [ "$DRY_RUN" = false ]; then
        for dep in pecos-rslib pecos-rslib-llvm; do
            if ! pin_exists_on_pypi "$dep" "$version"; then
                echo -e "${RED}Error: quantum-pecos==$version pins $dep==$version, which is not on PyPI.${NC}"
                echo "Publish $dep first (or use all-packages mode)."
                exit 1
            fi
        done
    fi
    publish_package "$PACKAGE"
else
    # All-packages mode: preflight EVERYTHING before uploading anything, and
    # require one consistent version across the set.
    echo -e "${GREEN}Preflighting all PECOS packages${NC}"
    versions=()
    for pkg in "${ALL_PACKAGES[@]}"; do
        v=$(preflight_package "$pkg")
        echo -e "${GREEN}Preflight OK: $pkg $v${NC}"
        check_package_distributions "$pkg"
        versions+=("$v")
    done
    unique=$(printf '%s\n' "${versions[@]}" | sort -u)
    if [ "$(printf '%s\n' "$unique" | wc -l)" -ne 1 ]; then
        echo -e "${RED}Error: packages carry different versions: ${unique//$'\n'/ }${NC}"
        exit 1
    fi
    echo -e "${GREEN}Publishing all PECOS packages at version $unique${NC}"
    # Dependencies first; publish_package aborts the script on failure or
    # decline, so quantum-pecos cannot publish without its pinned deps.
    for pkg in "${ALL_PACKAGES[@]}"; do
        publish_package "$pkg"
    done
fi

echo -e "\n${GREEN}Done!${NC}"

#!/bin/bash
# Script to help publish PECOS wheels to PyPI from GitHub Actions artifacts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ARTIFACT_FILE="pecos-distribution.zip"
DRY_RUN=false
PACKAGE=""

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --file FILE      Path to the GitHub Actions artifact zip (default: pecos-distribution.zip)"
            echo "  -p, --package PKG    Publish only specific package (pecos-rslib or quantum-pecos)"
            echo "  --dry-run            Show what would be uploaded without actually uploading"
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

# Check if artifact file exists
if [ ! -f "$ARTIFACT_FILE" ]; then
    echo -e "${RED}Error: Artifact file '$ARTIFACT_FILE' not found!${NC}"
    echo "Please download the 'pecos-distribution' artifact from GitHub Actions."
    exit 1
fi

# Check if uv is installed (preferred) or twine directly
if command -v uv &> /dev/null; then
    TWINE_CMD="uv run twine"
    echo -e "${GREEN}Using uv to run twine${NC}"
elif command -v twine &> /dev/null; then
    TWINE_CMD="twine"
    echo -e "${YELLOW}Using system twine (consider using uv)${NC}"
else
    echo -e "${RED}Error: Neither uv nor twine is installed!${NC}"
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or install twine with: pip install twine"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo -e "${GREEN}Extracting distribution bundle...${NC}"
unzip -q "$ARTIFACT_FILE" -d "$TEMP_DIR"

# Function to publish a package
publish_package() {
    local package_name=$1
    # Try both possible locations: with and without dist/ prefix
    local package_dir="$TEMP_DIR/$package_name"
    if [ ! -d "$package_dir" ]; then
        package_dir="$TEMP_DIR/dist/$package_name"
    fi

    if [ ! -d "$package_dir" ]; then
        echo -e "${YELLOW}Warning: $package_name directory not found in distribution${NC}"
        return
    fi

    local file_count=$(ls -1 "$package_dir" | wc -l)
    if [ "$file_count" -eq 0 ]; then
        echo -e "${YELLOW}Warning: No files found in $package_name directory${NC}"
        return
    fi

    echo -e "\n${GREEN}=== Publishing $package_name ===${NC}"
    echo "Found $file_count distribution file(s):"
    ls -la "$package_dir"

    # Run twine check
    echo -e "\n${GREEN}Running twine check...${NC}"
    if $TWINE_CMD check "$package_dir"/* 2>&1 | grep -v "license-file"; then
        echo -e "${GREEN}Distribution checks passed${NC}"
    else
        # Check if there are errors other than license-file
        if $TWINE_CMD check "$package_dir"/* 2>&1 | grep -v "license-file" | grep -q "ERROR"; then
            echo -e "${RED}Distribution checks failed${NC}"
            echo "Run '$TWINE_CMD check $package_dir/*' to see details"
            return 1
        else
            echo -e "${YELLOW}Only license-file warnings found (safe to ignore for maturin wheels)${NC}"
        fi
    fi

    if [ "$DRY_RUN" = true ]; then
        echo -e "\n${YELLOW}DRY RUN: Would upload the following files:${NC}"
        ls -1 "$package_dir"
    else
        echo -e "\n${GREEN}Uploading to PyPI...${NC}"
        read -p "Are you sure you want to upload $package_name to PyPI? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $TWINE_CMD upload "$package_dir"/*
            echo -e "${GREEN}Successfully uploaded $package_name!${NC}"
        else
            echo -e "${YELLOW}Skipped uploading $package_name${NC}"
        fi
    fi
}

# Main execution
if [ -n "$PACKAGE" ]; then
    # Publish specific package
    if [[ "$PACKAGE" != "pecos-rslib" && "$PACKAGE" != "quantum-pecos" ]]; then
        echo -e "${RED}Error: Invalid package name '$PACKAGE'${NC}"
        echo "Valid options are: pecos-rslib, quantum-pecos"
        exit 1
    fi
    publish_package "$PACKAGE"
else
    # Publish all packages
    echo -e "${GREEN}Publishing all PECOS packages${NC}"
    publish_package "pecos-rslib"
    publish_package "quantum-pecos"
fi

echo -e "\n${GREEN}Done!${NC}"

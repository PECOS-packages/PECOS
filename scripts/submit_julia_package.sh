#!/bin/bash
# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

set -e

echo "PECOS Julia Package Submission Helper"
echo "===================================="
echo

# Check if bundle file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pecos-julia-release-bundle.tar.gz>"
    echo
    echo "Steps:"
    echo "1. Download the release bundle from GitHub Actions artifacts"
    echo "2. Run this script with the bundle file"
    echo "3. Follow the prompts to prepare for submission"
    exit 1
fi

BUNDLE_FILE="$1"

if [ ! -f "$BUNDLE_FILE" ]; then
    echo "Error: Bundle file not found: $BUNDLE_FILE"
    exit 1
fi

# Create working directory
WORK_DIR="julia-submission-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Extracting bundle..."
tar -xzf "../$BUNDLE_FILE"

echo
echo "Bundle contents:"
find release-bundle -type f | sort

echo
echo "Choose submission method:"
echo "1) Julia General Registry (recommended for packages)"
echo "2) Yggdrasil (for creating JLL packages)"
echo "3) Both (prepare files for both methods)"
read -p "Enter choice (1-3): " choice

case $choice in
    1|3)
        echo
        echo "Preparing for Julia General Registry submission..."
        echo

        # Create Artifacts.toml template
        cat > artifacts_template.toml << 'EOF'
# Template Artifacts.toml for PECOS.jl
#
# Instructions:
# 1. Upload the binaries from release-bundle/binaries/ to a GitHub release
# 2. Replace the URLs below with the actual download URLs
# 3. The SHA256 hashes are already filled in from the bundle
# 4. Copy this file to julia/PECOS.jl/deps/Artifacts.toml

[pecos_julia]
git-tree-sha1 = "COMPUTE_THIS_VALUE"  # See instructions below

EOF

        # Add entries for each platform
        for tarball in release-bundle/binaries/*.tar.gz; do
            if [ -f "$tarball" ]; then
                basename=$(basename "$tarball")
                platform=$(echo "$basename" | sed 's/pecos_julia-\(.*\)\.tar\.gz/\1/')
                sha256=$(cat "release-bundle/checksums/${basename}.sha256" | awk '{print $1}')

                # Parse platform details
                case $platform in
                    linux-x86_64)
                        arch="x86_64"
                        os="linux"
                        libc='libc = "glibc"'
                        ;;
                    linux-aarch64)
                        arch="aarch64"
                        os="linux"
                        libc='libc = "glibc"'
                        ;;
                    macos-x86_64)
                        arch="x86_64"
                        os="macos"
                        libc=""
                        ;;
                    macos-aarch64)
                        arch="aarch64"
                        os="macos"
                        libc=""
                        ;;
                    windows-x86_64)
                        arch="x86_64"
                        os="windows"
                        libc=""
                        ;;
                esac

                cat >> artifacts_template.toml << EOF
# $platform
[[pecos_julia.download]]
url = "https://github.com/PECOS-packages/PECOS/releases/download/jl-vX.Y.Z/$basename"
sha256 = "$sha256"

[[pecos_julia.platform]]
arch = "$arch"
os = "$os"
EOF
                if [ -n "$libc" ]; then
                    echo "$libc" >> artifacts_template.toml
                fi
                echo "" >> artifacts_template.toml
            fi
        done

        # Add instructions for computing git-tree-sha1
        cat >> artifacts_template.toml << 'EOF'
# To compute git-tree-sha1:
# 1. Extract one of the tarballs to a temporary directory
# 2. Run this Julia code:
#
# using Pkg.GitTools
# tree_hash = GitTools.tree_hash("/path/to/extracted/contents")
# println("git-tree-sha1 = \"$tree_hash\"")
EOF

        echo "Created: artifacts_template.toml"
        echo
        echo "Next steps for Julia General Registry:"
        echo "1. Create a GitHub release and upload binaries from release-bundle/binaries/"
        echo "2. Update artifacts_template.toml with:"
        echo "   - Actual GitHub release URLs"
        echo "   - Computed git-tree-sha1"
        echo "3. Copy to julia/PECOS.jl/deps/Artifacts.toml"
        echo "4. Submit package:"
        echo "   julia> using Registrator"
        echo "   julia> Registrator.register(\"https://github.com/PECOS-packages/PECOS.git\", subdir=\"julia/PECOS.jl\")"
        ;;
esac

case $choice in
    2|3)
        echo
        echo "Preparing for Yggdrasil submission..."
        echo

        # Check if build_tarballs.jl exists
        BUILD_TARBALLS="../julia/PECOS.jl/deps/build_tarballs.jl"
        if [ -f "$BUILD_TARBALLS" ]; then
            # Create Yggdrasil structure
            mkdir -p yggdrasil_submission/P/PECOS_julia

            # Get the commit from the bundle
            CURRENT_COMMIT=$(grep "Commit:" release-bundle/SUBMISSION_INSTRUCTIONS.md 2>/dev/null | awk '{print $2}' || echo "main")
            echo "Using commit: $CURRENT_COMMIT"

            # Copy and update build_tarballs.jl
            cp "$BUILD_TARBALLS" yggdrasil_submission/P/PECOS_julia/build_tarballs.jl

            # Replace the commit reference
            sed -i "s|get(ENV, \"PECOS_BUILD_COMMIT\", \"main\")|\"$CURRENT_COMMIT\"|g" \
                yggdrasil_submission/P/PECOS_julia/build_tarballs.jl

            echo "Created Yggdrasil submission in: yggdrasil_submission/"
            echo
            echo "Next steps for Yggdrasil:"
            echo "1. Fork https://github.com/JuliaPackaging/Yggdrasil"
            echo "2. Copy yggdrasil_submission/P/ to your fork"
            echo "3. Create pull request with title: 'New package: PECOS_julia v0.1.0'"
            echo "4. Once merged, PECOS_julia_jll will be created automatically"
            echo
            echo "The build will use commit: $CURRENT_COMMIT"
        else
            echo "Warning: build_tarballs.jl not found at $BUILD_TARBALLS"
        fi
        ;;
esac

echo
echo "All files prepared in: $PWD"
echo
echo "Bundle information:"
cat release-bundle/SUBMISSION_INSTRUCTIONS.md | grep -E "Version:|Commit:|Branch:|Date:" || true

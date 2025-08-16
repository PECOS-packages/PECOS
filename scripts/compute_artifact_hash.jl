#!/usr/bin/env julia
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

# Helper script to compute git-tree-sha1 for Julia artifacts

using Pkg.GitTools
using SHA

if length(ARGS) < 1
    println("Usage: julia compute_artifact_hash.jl <tarball.tar.gz>")
    println()
    println("This script extracts the tarball and computes the git-tree-sha1")
    println("needed for Artifacts.toml")
    exit(1)
end

tarball_path = ARGS[1]

if !isfile(tarball_path)
    println("Error: File not found: $tarball_path")
    exit(1)
end

# Create temporary directory
mktempdir() do tmpdir
    println("Extracting $tarball_path...")

    # Extract tarball
    run(`tar -xzf $tarball_path -C $tmpdir`)

    # Compute git-tree-sha1
    tree_hash = bytes2hex(GitTools.tree_hash(tmpdir))

    println()
    println("git-tree-sha1 = \"$tree_hash\"")
    println()
    println("Add this value to your Artifacts.toml file")

    # Also show what's in the artifact
    println()
    println("Artifact contents:")
    for (root, dirs, files) in walkdir(tmpdir)
        level = count(==('/'), relpath(root, tmpdir))
        indent = "  " ^ level
        println(indent * basename(root) * "/")
        for file in files
            println(indent * "  " * file)
        end
    end
end

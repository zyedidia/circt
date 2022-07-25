#!/usr/bin/env bash
##===- utils/build-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with the standard options. Intended to be called from 
# the github workflows.
#
##===----------------------------------------------------------------------===##

BUILD_DIR=${1:-"build"}
BUILD_TYPE=${2:-"Release"}
CC=${3:-"clang"}
CXX=${4:-"clang++"}
EXTRA_ARGS=${@:5}

mkdir -p llvm/$BUILD_DIR
cd llvm/$BUILD_DIR
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_PROJECTS='mlir' \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_STATIC_LINK_CXX_STDLIB=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_TARGETS_TO_BUILD="host"
ninja

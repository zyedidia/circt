//===- DomainDialect.cpp - Domain dialect definition ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Domain operation definitions.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"

#include "circt/Dialect/Domain/DomainOps.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Domain/Domain.cpp.inc"

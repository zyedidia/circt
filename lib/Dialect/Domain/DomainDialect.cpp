//===- DomainDialect.cpp - Domain dialect definition ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Domain dialect definition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Domain/DomainDialect.h"
#include "circt/Dialect/Domain/DomainOps.h"

#include "circt/Dialect/Domain/DomainDialect.cpp.inc"

void circt::domain::DomainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Domain/Domain.cpp.inc"
      >();
}

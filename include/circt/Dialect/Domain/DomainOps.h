//===- DomainOps.h - Domain operation declarations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Domain operation declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DOMAIN_DOMAINOPS_H
#define CIRCT_DIALECT_DOMAIN_DOMAINOPS_H

#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Domain/Domain.h.inc"

#endif // CIRCT_DIALECT_DOMAIN_DOMAINOPS_H

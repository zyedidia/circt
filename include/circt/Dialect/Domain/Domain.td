//===- Domain.td - Domain dialect definition ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Domain dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DOMAIN_TD
#define DOMAIN_TD

include "mlir/IR/OpBase.td"

include "circt/Dialect/Domain/DomainDialect.td"

class DomainOp<string mnemonic, list<Trait> traits = []> :
    Op<DomainDialect, mnemonic, traits>;

include "circt/Dialect/Domain/DomainOps.td"

#endif // DOMAIN_TD

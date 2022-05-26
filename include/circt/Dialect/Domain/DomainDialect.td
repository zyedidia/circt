//===- DomainDialect.td - Domain dialect definition -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the DomainDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DOMAIN_DOMAINDIALECT
#define CIRCT_DIALECT_DOMAIN_DOMAINDIALECT

def DomainDialect : Dialect {
  let name = "domain";
  let cppNamespace = "::circt::domain";
  let summary = "The Domain dialect models design intent in various domains";
  let description = [{
    The Domain dialect models non-RTL, domain-specific information that is vital
    to capturing hardware designer intent. A very general structure is modelled,
    and domain models can be built on this foundation.

    The core concepts are:
      * Containers: which can have a handful of known, required properties, as
        well as arbitrary optional properties
      * Properties: which can be of known or user-defined types
      * References: which link together different containers in some way

    These basic structures can model a diverse set of domain models in a way
    that enables the compiler to verify, analyze, and transform them.
  }];
}

#endif // CIRCT_DIALECT_DOMAIN_DOMAINDIALECT

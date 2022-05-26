//===- DomainOps.td - Domain dialect ops ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the DomainDialect operation definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DOMAIN_DOMAINOPS
#define CIRCT_DIALECT_DOMAIN_DOMAINOPS

def ContainerOp : DomainOp<"container", [SingleBlock, NoTerminator]> {
  let summary = "A container for domain models.";
  let description = [{
    A container holds known, required properties as attributes and arbitrary,
    optional properties declared in the body block. Containers have identifiers
    so they can be referenced by other containers.

    The list of required properties is:
      * `id`: a unique identifier for the container, so it can be referenced
  }];

  let arguments = (ins
    StrAttr:$type,
    StrAttr:$id
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    $type `(` `id` `=` $id `)` $body attr-dict
  }];
}

def PropertyOp : DomainOp<"property", [HasParent<"ContainerOp">]> {
  let summary = "A property for arbitrary domain model data.";
  let description = [{
    A property is a name value pair of arbitrary data, which can have a builtin
    type or a user-defined type. It must exist in the body of a container.
  }];

  let arguments = (ins
    StrAttr:$name,
    AnyAttr:$value
  );

  let assemblyFormat = [{
    $name `,` $value attr-dict
  }];
}

def ReferenceOp : DomainOp<"reference", [HasParent<"ContainerOp">]> {
  let summary = "A reference to domain model data.";
  let description = [{
    A reference refers to a `domain.container` by its `id`. It must exist in the
    body of a container.
  }];

  let arguments = (ins
    StrAttr:$id
  );

  let assemblyFormat = [{
    $id attr-dict
  }];

  // TODO: verify the referenced id somehow.
  // let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_DOMAIN_DOMAINOPS

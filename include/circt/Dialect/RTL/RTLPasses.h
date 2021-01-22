//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLPASSES_H
#define CIRCT_DIALECT_RTL_RTLPASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
    namespace rtl {

std::unique_ptr<mlir::Pass> createBlackboxCalloutPass();

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/RTL/RTLPasses.h.inc"
    } // namespace rtl
} // namespace circt

#endif // CIRCT_DIALECT_RTL_PASSES_H

//===- ExportCpp.h - ExportCpp Conversion --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARC_EXPORT_CPP_H
#define ARC_EXPORT_CPP_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace arc {

mlir::LogicalResult exportCppFile(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerToCppFileTranslation();

} // namespace arc
} // namespace circt

#endif // ARC_EXPORT_CPP_H

//===- BlackboxCallout.cpp - External handling of external modules --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call arbitrary programs and pass them the attributes attached to external 
// modules.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/RTL/RTLPasses.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Process.h"

using namespace circt;
using namespace mlir;
using namespace rtl;
using namespace sv;

//===----------------------------------------------------------------------===//
// BlackboxCallout
//===----------------------------------------------------------------------===//

namespace {
struct BlackboxCallout : public BlackboxCalloutBase<BlackboxCallout> {

  void runOnOperation() override {
    llvm::Regex matcher(genString);
    llvm::errs() << "Searching for " << genString << " (" << matcher.isValid() << ")\n";
    RTLExternModuleOp module = getOperation();
      if (auto genAttr = module->getAttrOfType<StringAttr>("generator")) {
        if (matcher.match(genAttr.getValue())) {
          auto moduleName = module.getVerilogModuleName();
          std::vector<StringRef> args;

          for (auto attr : module.getAttrs()) {
              attr.first.dump(); llvm::errs() << " ";
              attr.second.dump(); llvm::errs() << " ";
              StringRef flag = "-" + attr.first.str();
              llvm::errs() << flag << "\n";
            if (auto strattr = attr.second.dyn_cast<StringAttr>()) {
              args.push_back(flag);
//              args.push_back("\"" + strattr.getValue().str() + "\"");
            } else if (auto intattr = attr.second.dyn_cast<IntegerAttr>()) {
              args.push_back(flag);
//              args.push_back(intattr.getValue().toString(10, false));              
            } else {
              llvm::errs() << "Unknown attribute type\n";
              attr.first.dump();
              attr.second.dump();
            }
          }
          llvm::errs() << "Executing: " << execString;
          for (auto s : args)
            llvm::errs() << " " << s;
          llvm::errs() << "\n";
          auto result = llvm::sys::ExecuteAndWait(execString, args);
          llvm::errs() << "Exited, got " << result << "\n";
        }
      }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::rtl::createBlackboxCalloutPass() {
  return std::make_unique<BlackboxCallout>();
}

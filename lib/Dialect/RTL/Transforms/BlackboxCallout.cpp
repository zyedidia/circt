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
          std::vector<std::string> args;
          args.emplace_back(moduleName.str());
          for (auto attr : module.getAttrs()) {
            std::string flag = "-" + attr.first.str();
            if (auto strattr = attr.second.dyn_cast<StringAttr>()) {
              args.emplace_back(flag);
              args.emplace_back("\"" + strattr.getValue().str() + "\"");
            } else if (auto intattr = attr.second.dyn_cast<IntegerAttr>()) {
              args.emplace_back(flag);
              args.emplace_back(intattr.getValue().toString(10, false));              
            }
          }
          std::vector<StringRef> stupidIndirection(args.size());
          for (auto& s : args)
            stupidIndirection.emplace_back(s);
          auto result = llvm::sys::ExecuteAndWait(execString, stupidIndirection);
          llvm::errs() << "Exited, got " << result << "\n";
        }
      }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::rtl::createBlackboxCalloutPass() {
  return std::make_unique<BlackboxCallout>();
}

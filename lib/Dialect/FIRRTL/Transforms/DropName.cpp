//===- DropName.cpp - Drop Names  -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropName pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

static bool isDeadOp(Operation *op,
                     llvm::SmallDenseSet<Operation *> &deadOperations) {
  for (auto *user : op->getUsers()) {
    if (deadOperations.count(user))
      continue;

    auto connect = dyn_cast<FConnectLike>(user);
    // If the user is neither node nor connect, we consider the op to be alive.
    if (!connect)
      return false;
    if (connect.src().getDefiningOp() != op)
      continue;
    auto *destOp = connect.dest().getDefiningOp();
    if (!destOp)
      return false;
    // Check the connect is self-connection. If not, check whether `dest` is
    // known to be dead.
    if (destOp != op && !deadOperations.count(destOp))
      return false;
  }

  return true;
}

namespace {
struct DropNamePass : public DropNameBase<DropNamePass> {
  DropNamePass(bool dropOnlyDeadNameFlag) {
    dropOnlyDeadNames = dropOnlyDeadNameFlag;
  }
  void runOnOperation() override {
    // If `dropOnlyDeadNames` is false, we drop all names regardless of their
    // liveness.
    if (!dropOnlyDeadNames) {
      getOperation().walk([](FNamableOp op) { op.dropName(); });
      return;
    }

    llvm::SmallDenseSet<Operation *> deadOperations;
    auto *block = getOperation().getBody();
    for (auto &op : llvm::reverse(*block)) {
      auto namable = dyn_cast<FNamableOp>(op);
      if (!namable)
        continue;

      bool isDead = isDeadOp(namable, deadOperations);
      if (!namable.hasDroppableName() && isDead)
        namable.dropName();
      if (isDead)
        deadOperations.insert(namable);
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createDropNamePass(bool dropOnlyDeadNames) {
  return std::make_unique<DropNamePass>(dropOnlyDeadNames);
}

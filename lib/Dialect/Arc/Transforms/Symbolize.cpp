//===- Symbolize.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

#define DEBUG_TYPE "arc-symbolize"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_SYMBOLIZE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

namespace {
struct SymbolizePass : public arc::impl::SymbolizeBase<SymbolizePass> {
  void visitFuncDef(func::FuncOp func);
  void visitModel(ModelOp model);
  void visitClockTree(ClockTreeOp clock, ModelOp model);
  void visitPassThrough(PassThroughOp pass, ModelOp model);
  void visitModelBodyOp(Value storage, Operation *op);
  void visitClockBodyOp(Operation *op);

  void visitComb(comb::AddOp op);
  void visitComb(comb::MulOp op);
  void visitComb(comb::DivUOp op);
  void visitComb(comb::DivSOp op);
  void visitComb(comb::ModUOp op);
  void visitComb(comb::ModSOp op);
  void visitComb(comb::ShlOp op);
  void visitComb(comb::ShrUOp op);
  void visitComb(comb::ShrSOp op);
  void visitComb(comb::SubOp op);
  void visitComb(comb::AndOp op);
  void visitComb(comb::OrOp op);
  void visitComb(comb::XorOp op);
  void visitComb(comb::ICmpOp op);
  void visitComb(comb::ParityOp op);
  void visitComb(comb::ExtractOp op);
  void visitComb(comb::ConcatOp op);
  void visitComb(comb::ReplicateOp op);
  void visitComb(comb::MuxOp op);

  void visitHW(hw::ConstantOp op);
  void visitHW(hw::ArrayCreateOp op);
  void visitHW(hw::ArrayGetOp op);
  void visitHW(hw::AggregateConstantOp op);

  void visitFunc(func::CallOp op);
  void visitFunc(func::ReturnOp op);
  void visitArc(arc::StateReadOp op);
  void visitArc(arc::StateWriteOp op);
  void visitArc(arc::MemoryReadOp op);
  void visitArc(arc::MemoryWriteOp op);
  void visitArc(arc::ClockGateOp op);

  void runOnOperation() override;

private:
  mlir::ModuleOp module;

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

  mlir::LLVM::LLVMFuncOp insertFunc(std::string name, Type signature);

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;
};
} // namespace

mlir::LLVM::LLVMFuncOp SymbolizePass::insertFunc(std::string name,
                                                 Type signature) {
  auto func = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
  if (!func) {
    OpBuilder builder(module.getBodyRegion());
    return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name,
                                                  signature);
  }
  return func;
}

void SymbolizePass::visitComb(comb::AddOp op) {
  OpBuilder builder(op);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(op->getContext());
  auto funcOp = insertFunc("_sym_build_add",
                           mlir::LLVM::LLVMFunctionType::get(voidTy, {}));
  builder.create<mlir::LLVM::CallOp>(op.getLoc(), funcOp, ArrayRef<Value>({}));
}
void SymbolizePass::visitComb(comb::MulOp op) {}
void SymbolizePass::visitComb(comb::DivUOp op) {}
void SymbolizePass::visitComb(comb::DivSOp op) {}
void SymbolizePass::visitComb(comb::ModUOp op) {}
void SymbolizePass::visitComb(comb::ModSOp op) {}
void SymbolizePass::visitComb(comb::ShlOp op) {}
void SymbolizePass::visitComb(comb::ShrUOp op) {}
void SymbolizePass::visitComb(comb::ShrSOp op) {}
void SymbolizePass::visitComb(comb::SubOp op) {}
void SymbolizePass::visitComb(comb::AndOp op) {
  OpBuilder builder(op);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(op->getContext());
  auto funcOp = insertFunc("_sym_build_and",
                           mlir::LLVM::LLVMFunctionType::get(voidTy, {}));
  builder.create<mlir::LLVM::CallOp>(op.getLoc(), funcOp, ArrayRef<Value>({}));
}
void SymbolizePass::visitComb(comb::OrOp op) {}
void SymbolizePass::visitComb(comb::XorOp op) {}
void SymbolizePass::visitComb(comb::ICmpOp op) {}
void SymbolizePass::visitComb(comb::ParityOp op) {}
void SymbolizePass::visitComb(comb::ExtractOp op) {}
void SymbolizePass::visitComb(comb::ConcatOp op) {}
void SymbolizePass::visitComb(comb::ReplicateOp op) {}
void SymbolizePass::visitComb(comb::MuxOp op) {
  OpBuilder builder(op);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(op->getContext());
  auto funcOp = insertFunc("_sym_build_mux",
                           mlir::LLVM::LLVMFunctionType::get(voidTy, {}));
  builder.create<mlir::LLVM::CallOp>(op.getLoc(), funcOp, ArrayRef<Value>({}));
}

void SymbolizePass::visitHW(hw::ConstantOp op) {}
void SymbolizePass::visitHW(hw::ArrayCreateOp op) {}
void SymbolizePass::visitHW(hw::ArrayGetOp op) {}
void SymbolizePass::visitHW(hw::AggregateConstantOp op) {}

void SymbolizePass::visitFunc(func::CallOp op) {}
void SymbolizePass::visitFunc(func::ReturnOp op) {}

void SymbolizePass::visitArc(arc::StateReadOp op) {}
void SymbolizePass::visitArc(arc::StateWriteOp op) {}
void SymbolizePass::visitArc(arc::MemoryReadOp op) {}
void SymbolizePass::visitArc(arc::MemoryWriteOp op) {}
void SymbolizePass::visitArc(arc::ClockGateOp op) {}

void SymbolizePass::visitModelBodyOp(Value storage, Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<arc::AllocStateOp>([&](auto op) {
          OpBuilder builder(op);
          auto sym = builder.create<arc::AllocStateOp>(op.getLoc(), StateType::get(builder.getI64Type()), storage);
          // if (StringAttr name = op->template getAttrOfType<StringAttr>("name")) {
          //   sym->setAttr("name", name);
          // }
      })
      .Default([&](auto op) {});
}

void SymbolizePass::visitClockBodyOp(Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<comb::AddOp, comb::MulOp, comb::DivUOp, comb::DivSOp, comb::ModUOp,
            comb::ModSOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp, comb::SubOp,
            comb::AndOp, comb::OrOp, comb::XorOp, comb::ICmpOp, comb::ParityOp,
            comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp, comb::MuxOp>(
          [&](auto op) { visitComb(op); })
      .Case<arc::StateReadOp, arc::StateWriteOp,
            arc::MemoryWriteOp, arc::MemoryReadOp, arc::ClockGateOp>(
          [&](auto op) { visitArc(op); })
      .Case<hw::ConstantOp, hw::ArrayCreateOp, hw::ArrayGetOp,
            hw::AggregateConstantOp>([&](auto op) { visitHW(op); })
      .Case<func::CallOp>([&](auto op) {})
      .Default([&](auto op) {
        if (!isa<AllocStateOp, AllocMemoryOp, RootInputOp, RootOutputOp,
                 ClockTreeOp, PassThroughOp>(op))
          emitOpError(op, "clock body op cannot be emitted");
      });
}

void SymbolizePass::visitClockTree(ClockTreeOp clock, ModelOp model) {
  for (auto &op : clock.getBodyBlock()) {
    visitClockBodyOp(&op);
  }
}

void SymbolizePass::visitPassThrough(PassThroughOp pass, ModelOp model) {
  for (auto &op : pass.getBodyBlock()) {
    visitClockBodyOp(&op);
  }
}

void SymbolizePass::visitFuncDef(func::FuncOp func) {
  for (auto &block : func.getBody()) {
    for (auto &op : block.getOperations()) {
      TypeSwitch<Operation *>(&op)
          .Case<comb::AddOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
                comb::ModUOp, comb::ModSOp, comb::ShlOp, comb::ShrUOp,
                comb::ShrSOp, comb::SubOp, comb::AndOp, comb::OrOp, comb::XorOp,
                comb::ICmpOp, comb::ParityOp, comb::ExtractOp, comb::ConcatOp,
                comb::ReplicateOp, comb::MuxOp>([&](auto op) { visitComb(op); })
          .Case<func::CallOp, func::ReturnOp>([&](auto op) { visitFunc(op); })
          .Case<hw::ConstantOp, hw::ArrayCreateOp, hw::ArrayGetOp,
                hw::AggregateConstantOp>([&](auto op) { visitHW(op); })
          .Default([&](auto op) { emitOpError(op, "op cannot be exported"); });
    }
  }
}

void SymbolizePass::visitModel(ModelOp model) {
  auto storage = model.getBody().getArgument(0);
  for (auto &op : model.getBodyBlock()) {
    visitModelBodyOp(storage, &op);
  }
  for (auto clock : model.getOps<ClockTreeOp>()) {
    visitClockTree(clock, model);
  }
  for (auto pass : model.getOps<PassThroughOp>()) {
    visitPassThrough(pass, model);
  }
}

void SymbolizePass::runOnOperation() {
  module = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "Symbolize PASS\n");

  for (auto op : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    visitFuncDef(op);
  }

  for (auto op : llvm::make_early_inc_range(module.getOps<ModelOp>())) {
    visitModel(op);
  }
}

std::unique_ptr<Pass> arc::createSymbolizePass() {
  return std::make_unique<SymbolizePass>();
}

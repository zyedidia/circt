//===- SVVisitors.h - SV Dialect Visitors -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with SV IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVVISITORS_H
#define CIRCT_DIALECT_SV_SVVISITORS_H

#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace sv {

template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class Visitor {
public:
  ResultType dispatchSVVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Expressions
            ReadInOutOp, ArrayIndexInOutOp, VerbatimExprOp, VerbatimExprSEOp,
            IndexedPartSelectInOutOp, IndexedPartSelectOp, StructFieldInOutOp,
            ConstantXOp, ConstantZOp, MacroRefExprOp,
            // Declarations.
            RegOp, WireOp, LogicOp, LocalParamOp, XMROp,
            // Control flow.
            OrderedOutputOp, IfDefOp, IfDefProceduralOp, IfOp, AlwaysOp,
            AlwaysCombOp, AlwaysFFOp, InitialOp, CaseOp,
            // Other Statements.
            AssignOp, BPAssignOp, PAssignOp, ForceOp, ReleaseOp, AliasOp,
            FWriteOp, VerbatimOp,
            // Type declarations.
            InterfaceOp, InterfaceSignalOp, InterfaceModportOp,
            InterfaceInstanceOp, GetModportOp, AssignInterfaceSignalOp,
            ReadInterfaceSignalOp,
            // Verification statements.
            AssertOp, AssumeOp, CoverOp, AssertConcurrentOp, AssumeConcurrentOp,
            CoverConcurrentOp,
            // Bind Statements
            BindOp,
            // Memory initialization statements
            ReadmemOp,
            // Simulator control tasks
            StopOp, FinishOp, ExitOp,
            // Severity message tasks
            FatalOp, ErrorOp, WarningOp, InfoOp,
            // Generate statements
            GenerateOp, GenerateCaseOp,
            // Sampled value functiions
            SampledOp>([&](auto expr) -> ResultType {
          return thisCast->visitSV(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSV(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidSV(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown SV node");
    abort();
  }

  /// This callback is invoked on any SV operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledSV(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitSV(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##SV(op, args...);  \
  }

  // Declarations
  HANDLE(RegOp, Unhandled);
  HANDLE(WireOp, Unhandled);
  HANDLE(LogicOp, Unhandled);
  HANDLE(LocalParamOp, Unhandled);
  HANDLE(XMROp, Unhandled);

  // Expressions
  HANDLE(ReadInOutOp, Unhandled);
  HANDLE(ArrayIndexInOutOp, Unhandled);
  HANDLE(VerbatimExprOp, Unhandled);
  HANDLE(VerbatimExprSEOp, Unhandled);
  HANDLE(IndexedPartSelectInOutOp, Unhandled);
  HANDLE(IndexedPartSelectOp, Unhandled);
  HANDLE(StructFieldInOutOp, Unhandled);
  HANDLE(ConstantXOp, Unhandled);
  HANDLE(ConstantZOp, Unhandled);
  HANDLE(MacroRefExprOp, Unhandled);

  // Control flow.
  HANDLE(OrderedOutputOp, Unhandled);
  HANDLE(IfDefOp, Unhandled);
  HANDLE(IfDefProceduralOp, Unhandled);
  HANDLE(IfOp, Unhandled);
  HANDLE(AlwaysOp, Unhandled);
  HANDLE(AlwaysCombOp, Unhandled);
  HANDLE(AlwaysFFOp, Unhandled);
  HANDLE(InitialOp, Unhandled);
  HANDLE(CaseOp, Unhandled);

  // Other Statements.
  HANDLE(AssignOp, Unhandled);
  HANDLE(BPAssignOp, Unhandled);
  HANDLE(PAssignOp, Unhandled);
  HANDLE(ForceOp, Unhandled);
  HANDLE(ReleaseOp, Unhandled);
  HANDLE(AliasOp, Unhandled);
  HANDLE(FWriteOp, Unhandled);
  HANDLE(VerbatimOp, Unhandled);

  // Type declarations.
  HANDLE(InterfaceOp, Unhandled);
  HANDLE(InterfaceInstanceOp, Unhandled);
  HANDLE(InterfaceSignalOp, Unhandled);
  HANDLE(InterfaceModportOp, Unhandled);
  HANDLE(GetModportOp, Unhandled);
  HANDLE(AssignInterfaceSignalOp, Unhandled);
  HANDLE(ReadInterfaceSignalOp, Unhandled);

  // Verification statements.
  HANDLE(AssertOp, Unhandled);
  HANDLE(AssumeOp, Unhandled);
  HANDLE(CoverOp, Unhandled);
  HANDLE(AssertConcurrentOp, Unhandled);
  HANDLE(AssumeConcurrentOp, Unhandled);
  HANDLE(CoverConcurrentOp, Unhandled);

  // Bind statements.
  HANDLE(BindOp, Unhandled);

  // Memory initialization statements
  HANDLE(ReadmemOp, Unhandled);

  // Simulator control tasks
  HANDLE(StopOp, Unhandled);
  HANDLE(FinishOp, Unhandled);
  HANDLE(ExitOp, Unhandled);

  // Severity message tasks
  HANDLE(FatalOp, Unhandled);
  HANDLE(ErrorOp, Unhandled);
  HANDLE(WarningOp, Unhandled);
  HANDLE(InfoOp, Unhandled);

  // Generate statements
  HANDLE(GenerateOp, Unhandled);
  HANDLE(GenerateCaseOp, Unhandled);

  // Sampled Value Functions
  HANDLE(SampledOp, Unhandled);
#undef HANDLE
};

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVVISITORS_H

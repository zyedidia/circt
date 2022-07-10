//===- GoEmitter.cpp - FIRRTL dialect to .go emitter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .go simulator emitter.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/GoEmitter.h"
#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "export-firrtl-go"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

namespace {
/// An emitter for FIRRTL dialect operations to .go simulation.
struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Indentation
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 4; }
  void reduceIndent() {
    assert(currentIndent >= 4);
    currentIndent -= 4;
  }

  // Circuit/module emission
  void emitCircuit(CircuitOp op);
  void emitModule(FModuleOp op);
  void emitModule(FExtModuleOp op);
  void emitModulePorts(ArrayRef<PortInfo> ports,
                       Block::BlockArgListType arguments = {});

  // Statement emission
  void emitStatementsInBlock(Block &block);
  void emitStatement(WhenOp op, bool noIndent = false);
  void emitStatement(WireOp op);
  void emitStatement(RegOp op);
  void emitStatement(RegResetOp op);
  void emitStatement(NodeOp op);
  void emitStatement(StopOp op);
  void emitStatement(SkipOp op);
  void emitStatement(PrintFOp op);
  void emitStatement(ConnectOp op);
  void emitStatement(StrictConnectOp op);
  void emitStatement(InstanceOp op);
  void emitStatement(AttachOp op);
  void emitStatement(MemOp op);
  void emitStatement(InvalidValueOp op);
  void emitStatement(CombMemOp op);
  void emitStatement(SeqMemOp op);
  void emitStatement(MemoryPortOp op);
  void emitStatement(MemoryPortAccessOp op);

  template <class T>
  void emitVerifStatement(T op, StringRef mnemonic);
  void emitStatement(AssertOp op) { emitVerifStatement(op, "assert"); }
  void emitStatement(AssumeOp op) { emitVerifStatement(op, "assume"); }
  void emitStatement(CoverOp op) { emitVerifStatement(op, "cover"); }

  // Exprsesion emission
  void emitExpression(Value value);
  void emitExpression(ConstantOp op);
  void emitExpression(SpecialConstantOp op);
  void emitExpression(SubfieldOp op);
  void emitExpression(SubindexOp op);
  void emitExpression(SubaccessOp op);

  void emitPrimExpr(StringRef mnemonic, Operation *op,
                    ArrayRef<uint32_t> attrs = {});

  void emitExpression(BitsPrimOp op) {
    emitPrimExpr("bits", op, {op.hi(), op.lo()});
  }
  void emitExpression(HeadPrimOp op) { emitPrimExpr("head", op, op.amount()); }
  void emitExpression(TailPrimOp op) { emitPrimExpr("tail", op, op.amount()); }
  void emitExpression(PadPrimOp op) { emitPrimExpr("pad", op, op.amount()); }
  void emitExpression(ShlPrimOp op) { emitPrimExpr("shl", op, op.amount()); }
  void emitExpression(ShrPrimOp op) { emitPrimExpr("shr", op, op.amount()); }

  // Funnel all ops without attrs into `emitPrimExpr`.
#define HANDLE(OPTYPE, MNEMONIC)                                               \
  void emitExpression(OPTYPE op) { emitPrimExpr(MNEMONIC, op); }
  HANDLE(AddPrimOp, "add");
  HANDLE(SubPrimOp, "sub");
  HANDLE(MulPrimOp, "mul");
  HANDLE(DivPrimOp, "div");
  HANDLE(RemPrimOp, "rem");
  HANDLE(AndPrimOp, "and");
  HANDLE(OrPrimOp, "or");
  HANDLE(XorPrimOp, "xor");
  HANDLE(LEQPrimOp, "leq");
  HANDLE(LTPrimOp, "lt");
  HANDLE(GEQPrimOp, "geq");
  HANDLE(GTPrimOp, "gt");
  HANDLE(EQPrimOp, "eq");
  HANDLE(NEQPrimOp, "neq");
  HANDLE(CatPrimOp, "cat");
  HANDLE(DShlPrimOp, "dshl");
  HANDLE(DShlwPrimOp, "dshlw");
  HANDLE(DShrPrimOp, "dshr");
  HANDLE(MuxPrimOp, "mux");
  HANDLE(AsSIntPrimOp, "asSInt");
  HANDLE(AsUIntPrimOp, "asUInt");
  HANDLE(AsAsyncResetPrimOp, "asAsyncReset");
  HANDLE(AsClockPrimOp, "asClock");
  HANDLE(CvtPrimOp, "cvt");
  HANDLE(NegPrimOp, "neg");
  HANDLE(NotPrimOp, "not");
  HANDLE(AndRPrimOp, "andr");
  HANDLE(OrRPrimOp, "orr");
  HANDLE(XorRPrimOp, "xorr");
#undef HANDLE

  // Attributes
  void emitAttribute(MemDirAttr attr);
  void emitAttribute(RUWAttr attr);

  // Types
  void emitType(Type type);

  // Locations
  void emitLocation(Location loc);
  void emitLocation(Operation *op) { emitLocation(op->getLoc()); }
  template <typename... Args>
  void emitLocationAndNewLine(Args... args) {
    emitLocation(args...);
    os << "\n";
  }

private:
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

  /// Return the name used during emission of a `Value`, or none if the value
  /// has not yet been emitted or it was emitted inline.
  Optional<StringRef> lookupEmittedName(Value value) {
    auto it = valueNames.find(value);
    if (it != valueNames.end())
      return {it->second};
    return {};
  }

private:
  /// The stream we are emitting into.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// Current level of indentation. See `indent()` and
  /// `addIndent()`/`reduceIndent()`.
  unsigned currentIndent = 0;

  /// The names used to emit values already encountered. Anything that gets a
  /// name in the output FIR is listed here, such that future expressions can
  /// reference it.
  DenseMap<Value, StringRef> valueNames;
  StringSet<> valueNamesStorage;

  void addValueName(Value value, StringAttr attr) {
    valueNames.insert({value, attr.getValue()});
  }
  void addValueName(Value value, StringRef str) {
    auto it = valueNamesStorage.insert(str);
    valueNames.insert({value, it.first->getKey()});
  }

  /// The current circuit namespace valid within the call to `emitCircuit`.
  CircuitNamespace circuitNamespace;
};
} // namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire circuit.
void Emitter::emitCircuit(CircuitOp op) {
  circuitNamespace.add(op);
  indent() << "package " << op.name() << "\n";
  for (auto &bodyOp : *op.getBody()) {
    if (encounteredError)
      break;
    TypeSwitch<Operation *>(&bodyOp)
        .Case<FModuleOp>([&](auto op) {
          emitModule(op);
          os << "\n";
        })
        .Case<FExtModuleOp>([&](auto op) {
          emitOpError(op, "extmodule not supported");
        })
        .Default([&](auto op) {
          emitOpError(op, "not supported for emission inside circuit");
        });
  }
  circuitNamespace.clear();
}

/// Emit an entire module.
void Emitter::emitModule(FModuleOp op) {
  indent() << "type " << op.getName() << " struct {\n";
  addIndent();

  // Emit the ports.
  auto ports = op.getPorts();
  emitModulePorts(ports, op.getArguments());
  if (!ports.empty() && !op.getBody()->empty())
    os << "\n";

  reduceIndent();
  indent() << "}\n";

  indent() << "func (m *" << op.getName() << ") Eval() {\n";
  addIndent();

  // Emit the module body.
  emitStatementsInBlock(*op.getBody());

  reduceIndent();
  indent() << "}\n";
  valueNames.clear();
  valueNamesStorage.clear();
}

/// Emit the ports of a module or extmodule. If the `arguments` array is
/// non-empty, it is used to populate `emittedNames` with the port names for use
/// during expression emission.
void Emitter::emitModulePorts(ArrayRef<PortInfo> ports,
                              Block::BlockArgListType arguments) {
  for (unsigned i = 0, e = ports.size(); i < e; ++i) {
    const auto &port = ports[i];
    indent() << (port.direction == Direction::In ? "I" : "O");
    if (!arguments.empty())
      addValueName(arguments[i], port.name);
    os << port.name.getValue() << " ";
    emitType(port.type);
    os << "\n";
  }
}

/// Check if an operation is inlined into the emission of their users. For
/// example, subfields are always inlined.
static bool isEmittedInline(Operation *op) {
  return isExpression(op) && !isa<InvalidValueOp>(op);
}

void Emitter::emitStatementsInBlock(Block &block) {
  for (auto &bodyOp : block) {
    if (encounteredError)
      return;
    if (isEmittedInline(&bodyOp))
      continue;
    TypeSwitch<Operation *>(&bodyOp)
        .Case<WhenOp, WireOp, RegOp, RegResetOp, NodeOp, StopOp, SkipOp,
              PrintFOp, AssertOp, AssumeOp, CoverOp, ConnectOp, StrictConnectOp,
              InstanceOp, AttachOp, MemOp, InvalidValueOp, SeqMemOp, CombMemOp,
              MemoryPortOp, MemoryPortAccessOp>(
            [&](auto op) { emitStatement(op); })
        .Default([&](auto op) {
          indent() << "// operation " << op->getName() << "\n";
          emitOpError(op, "not supported as statement");
        });
  }
}

void Emitter::emitStatement(WhenOp op, bool noIndent) {
  (noIndent ? os : indent()) << "if ";
  emitExpression(op.condition());
  os << " {";
  emitLocationAndNewLine(op);
  addIndent();
  emitStatementsInBlock(op.getThenBlock());
  reduceIndent();
  if (!op.hasElseRegion()) {
    indent() << "}\n";
    return;
  }

  indent() << "} else ";
  // Sugar to `else when ...` if there's only a single when statement in the
  // else block.
  auto &elseBlock = op.getElseBlock();
  if (!elseBlock.empty() && &elseBlock.front() == &elseBlock.back()) {
    if (auto whenOp = dyn_cast<WhenOp>(&elseBlock.front())) {
      emitStatement(whenOp, true);
      return;
    }
  }
  // Otherwise print the block as `else :`.
  os << "{\n";
  addIndent();
  emitStatementsInBlock(elseBlock);
  reduceIndent();
  indent() << "}\n";
}

void Emitter::emitStatement(WireOp op) {
  addValueName(op, op.nameAttr());
  indent() << "var wire_" << op.name() << " ";
  emitType(op.getType());
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RegOp op) {
  addValueName(op, op.nameAttr());
  indent() << "var reg_" << op.name() << " ";
  emitType(op.getType());
  os << ", ";
  emitExpression(op.clockVal());
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RegResetOp op) {
  addValueName(op, op.nameAttr());
  indent() << "reg " << op.name() << " : ";
  emitType(op.getType());
  os << ", ";
  emitExpression(op.clockVal());
  os << " with :\n";
  indent() << "  reset => (";
  emitExpression(op.resetSignal());
  os << ", ";
  emitExpression(op.resetValue());
  os << ")";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(NodeOp op) {
  addValueName(op, op.nameAttr());
  indent() << "node " << op.name() << " = ";
  emitExpression(op.input());
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(StopOp op) {
  indent() << "stop(";
  emitExpression(op.clock());
  os << ", ";
  emitExpression(op.cond());
  os << ", " << op.exitCode() << ")";
  if (!op.name().empty()) {
    os << " : " << op.name();
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(SkipOp op) {
  indent() << "skip";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(PrintFOp op) {
  indent() << "printf(";
  emitExpression(op.clock());
  os << ", ";
  emitExpression(op.cond());
  os << ", \"";
  os.write_escaped(op.formatString());
  os << "\"";
  for (auto operand : op.operands()) {
    os << ", ";
    emitExpression(operand);
  }
  os << ")";
  if (!op.name().empty()) {
    os << " : " << op.name();
  }
  emitLocationAndNewLine(op);
}

template <class T>
void Emitter::emitVerifStatement(T op, StringRef mnemonic) {
  indent() << mnemonic << "(";
  emitExpression(op.clock());
  os << ", ";
  emitExpression(op.predicate());
  os << ", ";
  emitExpression(op.enable());
  os << ", \"";
  os.write_escaped(op.message());
  os << "\"";
  os << ")";
  if (!op.name().empty()) {
    os << " : " << op.name();
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(ConnectOp op) {
  indent();
  emitExpression(op.dest());
  if (op.src().getDefiningOp<InvalidValueOp>()) {
    os << ".invalidate()";
  } else {
    os << " = ";
    emitExpression(op.src());
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(StrictConnectOp op) {
  indent();
  emitExpression(op.dest());
  if (op.src().getDefiningOp<InvalidValueOp>()) {
    os << ".invalidate()";
  } else {
    os << " = ";
    emitExpression(op.src());
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(InstanceOp op) {
  indent() << "// module instance " << op.name() << " of " << op.moduleName();
  emitLocationAndNewLine(op);

  // Make sure we have a name like `<inst>.<port>` for each of the instance
  // result values.
  SmallString<16> portName(op.name());
  portName.push_back('.');
  unsigned baseLen = portName.size();
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    portName.append(op.getPortNameStr(i));
    addValueName(op.getResult(i), portName);
    portName.resize(baseLen);
  }
}

void Emitter::emitStatement(AttachOp op) {
  indent() << "// TODO: AttachOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(MemOp op) {
  indent() << "// TODO: MemOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(SeqMemOp op) {
  indent() << "// TODO: SeqMemOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(CombMemOp op) {
  indent() << "// TODO: CombMemOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(MemoryPortOp op) {
  // Nothing to output for this operation.
  addValueName(op.data(), op.name());
}

void Emitter::emitStatement(MemoryPortAccessOp op) {
  indent() << "// TODO: MemoryPortAccessOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(InvalidValueOp op) {
  // Only emit this invalid value if it is used somewhere else than the RHS of
  // a connect.
  if (llvm::all_of(op->getUses(), [&](OpOperand &use) {
        return use.getOperandNumber() == 1 &&
               isa<ConnectOp, StrictConnectOp>(use.getOwner());
      }))
    return;

  auto name = circuitNamespace.newName("_invalid");
  addValueName(op, name);
  indent() << "wire " << name << " : ";
  emitType(op.getType());
  emitLocationAndNewLine(op);
  indent() << name << " is invalid";
  emitLocationAndNewLine(op);
}

void Emitter::emitExpression(Value value) {
  // Handle the trivial case where we already have a name for this value which
  // we can use.
  if (auto name = lookupEmittedName(value)) {
    os << name;
    return;
  }

  auto *op = value.getDefiningOp();
  assert(op && "value must either be a block arg or the result of an op");
  TypeSwitch<Operation *>(op)
      .Case<
          // Basic expressions
          ConstantOp, SpecialConstantOp, SubfieldOp, SubindexOp, SubaccessOp,
          // Binary
          AddPrimOp, SubPrimOp, MulPrimOp, DivPrimOp, RemPrimOp, AndPrimOp,
          OrPrimOp, XorPrimOp, LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp,
          EQPrimOp, NEQPrimOp, CatPrimOp, DShlPrimOp, DShlwPrimOp, DShrPrimOp,
          // Unary
          AsSIntPrimOp, AsUIntPrimOp, AsAsyncResetPrimOp, AsClockPrimOp,
          CvtPrimOp, NegPrimOp, NotPrimOp, AndRPrimOp, OrRPrimOp, XorRPrimOp,
          // Miscellaneous
          BitsPrimOp, HeadPrimOp, TailPrimOp, PadPrimOp, MuxPrimOp, ShlPrimOp,
          ShrPrimOp>([&](auto op) { emitExpression(op); })
      .Default([&](auto op) {
        emitOpError(op, "not supported as expression");
        os << "// <unsupported-expr-" << op->getName().stripDialect() << ">";
      });
}

void Emitter::emitExpression(ConstantOp op) {
  emitType(op.getType());
  // TODO: Add option to control base-2/8/10/16 output here.
  os << "(" << op.value() << ")";
}

void Emitter::emitExpression(SpecialConstantOp op) {
  indent() << "// TODO: SpecialConstantOp";
  emitLocationAndNewLine(op);
}

void Emitter::emitExpression(SubfieldOp op) {
  auto type = op.input().getType().cast<BundleType>();
  emitExpression(op.input());
  os << "." << type.getElementName(op.fieldIndex());
}

void Emitter::emitExpression(SubindexOp op) {
  emitExpression(op.input());
  os << "[" << op.index() << "]";
}

void Emitter::emitExpression(SubaccessOp op) {
  emitExpression(op.input());
  os << "[";
  emitExpression(op.index());
  os << "]";
}

void Emitter::emitPrimExpr(StringRef mnemonic, Operation *op,
                           ArrayRef<uint32_t> attrs) {
  os << mnemonic << "(";
  llvm::interleaveComma(op->getOperands(), os,
                        [&](Value arg) { emitExpression(arg); });
  for (auto attr : attrs)
    os << ", " << attr;
  os << ")";
}

void Emitter::emitAttribute(MemDirAttr attr) {
  switch (attr) {
  case MemDirAttr::Infer:
    os << "infer";
    break;
  case MemDirAttr::Read:
    os << "read";
    break;
  case MemDirAttr::Write:
    os << "write";
    break;
  case MemDirAttr::ReadWrite:
    os << "rdwr";
    break;
  }
}

void Emitter::emitAttribute(RUWAttr attr) {
  switch (attr) {
  case RUWAttr::Undefined:
    os << "undefined";
    break;
  case RUWAttr::Old:
    os << "old";
    break;
  case RUWAttr::New:
    os << "new";
    break;
  }
}

/// Emit a FIRRTL type into the output.
void Emitter::emitType(Type type) {
  auto emitWidth = [&](Optional<int32_t> width) {
    if (!width.hasValue())
      return;
    // os << "<" << *width << ">";
  };
  TypeSwitch<Type>(type)
      .Case<ClockType>([&](auto) { os << "Clock"; })
      .Case<ResetType>([&](auto) { os << "Reset"; })
      .Case<AsyncResetType>([&](auto) { os << "AsyncReset"; })
      .Case<UIntType>([&](auto type) {
        os << "UInt";
        emitWidth(type.getWidth());
      })
      .Case<SIntType>([&](auto type) {
        os << "SInt";
        emitWidth(type.getWidth());
      })
      .Case<AnalogType>([&](auto type) {
        os << "Analog";
        emitWidth(type.getWidth());
      })
      .Case<BundleType>([&](auto type) {
        os << "struct{";
        bool anyEmitted = false;
        for (auto &element : type.getElements()) {
          os << (anyEmitted ? "\n" : "");
          os << element.name.getValue() << " : ";
          emitType(element.type);
          anyEmitted = true;
        }
        if (anyEmitted)
          os << "\n";
        os << "}";
      })
      .Case<FVectorType>([&](auto type) {
        os << "[" << type.getNumElements() << "]";
        emitType(type.getElementType());
      })
      .Case<CMemoryType>([&](auto type) {
        llvm_unreachable("TODO: CMemoryType");
        // emitType(type.getElementType());
        // os << "[" << type.getNumElements() << "]";
      })
      .Default([&](auto type) {
        llvm_unreachable("all types should be implemented");
      });
}

/// Emit a location as `@[<filename> <line>:<column>]` annotation, including a
/// leading space.
void Emitter::emitLocation(Location loc) {
  // TODO: Handle FusedLoc and uniquify locations, avoid repeated file names.
  if (auto fileLoc = loc->dyn_cast_or_null<FileLineColLoc>()) {
    os << " //" << fileLoc.getFilename().getValue();
    if (auto line = fileLoc.getLine()) {
      os << " " << line;
      if (auto col = fileLoc.getColumn())
        os << ":" << col;
    }
  }
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified FIRRTL circuit into the given output stream.
mlir::LogicalResult circt::firrtl::exportGoFile(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  Emitter emitter(os);
  for (auto &op : *module.getBody()) {
    if (auto circuitOp = dyn_cast<CircuitOp>(op))
      emitter.emitCircuit(circuitOp);
  }
  return emitter.finalize();
}

void circt::firrtl::registerToGoFileTranslation() {
  static mlir::TranslateFromMLIRRegistration toFIR(
      "export-firrtl-go",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportGoFile(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<chirrtl::CHIRRTLDialect>();
        registry.insert<firrtl::FIRRTLDialect>();
      });
}

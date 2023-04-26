//===- ExportCpp.cpp ------------------------------------------------------===//
//
// Proprietary and Confidential Software of SiFive Inc. All Rights Reserved.
// See the LICENSE file for license information.
// SPDX-License-Identifier: UNLICENSED
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {

struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Indentation
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() {
    assert(currentIndent >= 2);
    currentIndent -= 2;
  }

  void clear();

  void accumulateStorage(Value storage, SmallVector<RootInputOp> &inputs,
                         SmallVector<RootOutputOp> &outputs);

  void emitHeader();
  void emitDesign(mlir::ModuleOp mod);
  void emitDefine(DefineOp op);
  void emitClockTree(ClockTreeOp op, ModelOp model);
  void emitPassThrough(PassThroughOp op, ModelOp model);
  void emitModel(ModelOp op);
  void emitStorage(Value storage);
  void emitIterators(Value storage);
  void emitType(IntegerType type);
  void emitType(hw::ArrayType type);
  void emitType(MemoryType type);
  void emitType(Type type);
  void emitValue(Value val);
  void emitDef(Value val, bool output = false);
  void emitModelBodyOp(Operation *op);

  void emitComb(comb::AddOp op);
  void emitComb(comb::MulOp op);
  void emitComb(comb::DivUOp op);
  void emitComb(comb::DivSOp op);
  void emitComb(comb::ModUOp op);
  void emitComb(comb::ModSOp op);
  void emitComb(comb::ShlOp op);
  void emitComb(comb::ShrUOp op);
  void emitComb(comb::ShrSOp op);
  void emitComb(comb::SubOp op);
  void emitComb(comb::AndOp op);
  void emitComb(comb::OrOp op);
  void emitComb(comb::XorOp op);
  void emitComb(comb::ICmpOp op);
  void emitComb(comb::ParityOp op);
  void emitComb(comb::ExtractOp op);
  void emitComb(comb::ConcatOp op);
  void emitComb(comb::ReplicateOp op);
  void emitComb(comb::MuxOp op);

  void emitHW(hw::ConstantOp op);
  void emitHW(hw::ArrayCreateOp op);
  void emitHW(hw::ArrayGetOp op);
  void emitHW(hw::AggregateConstantOp op);

  void emitArc(arc::StateOp op);
  void emitArc(arc::OutputOp op);
  void emitArc(arc::StateReadOp op);
  void emitArc(arc::StateWriteOp op);
  void emitArc(arc::MemoryReadOp op);
  void emitArc(arc::MemoryWriteOp op);
  void emitArc(arc::ClockGateOp op);

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

  /// The names used to emit values already encountered. Anything that gets a
  /// name in the output cpp is listed here, such that future expressions can
  /// reference it.
  DenseMap<Value, StringRef> valueNames;
  DenseMap<Value, StringRef> localNames;
  DenseMap<Value, StringRef> outNames;

  SmallVector<RootInputOp> inputs;
  SmallVector<RootOutputOp> outputs;
  DenseMap<StringRef, Operation *> internals;
  SmallVector<std::unique_ptr<std::string>> storageNames;

  Namespace vars;

  /// Return the name used during emission of a `Value`, or none if the value
  /// has not yet been emitted or it was emitted inline.
  llvm::Optional<StringRef> lookupEmittedName(Value value) {
    auto it = valueNames.find(value);
    if (it != valueNames.end())
      return {it->second};
    it = localNames.find(value);
    if (it != localNames.end())
      return {it->second};
    return {};
  }

  /// The stream we are emitting into.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// Current level of indentation. See `indent()` and
  /// `addIndent()`/`reduceIndent()`.
  unsigned currentIndent = 0;
};

} // namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

void Emitter::emitHeader() { os << "#include \"arc-cpp.h\"\n"; }

void Emitter::emitType(IntegerType type) {
  os << "Int<" << type.getWidth() << ">";
}

void Emitter::emitType(hw::ArrayType type) {
  os << "Array<" << type.getSize() << ", ";
  os << type.getElementType().cast<IntegerType>().getWidth();
  os << ">";
}

void Emitter::emitType(MemoryType type) {
  os << "Memory<" << type.getNumWords() << ", ";
  os << type.getWordType().cast<IntegerType>().getWidth();
  os << ">";
}

void Emitter::emitType(Type type) {
  if (auto t = type.dyn_cast<IntegerType>()) {
    emitType(t);
  } else if (auto t = type.dyn_cast<hw::ArrayType>()) {
    emitType(t);
  } else if (auto t = type.dyn_cast<MemoryType>()) {
    emitType(t);
  } else {
    encounteredError = true;
    return;
  }
}

void Emitter::emitDef(Value val, bool output) {
  // The value was already defined, so we just re-assign to it.
  if (!output) {
    if (auto name = lookupEmittedName(val)) {
      os << name;
      return;
    }
  }

  emitType(val.getType());
  if (!output) {
    auto name = vars.newName("val");
    localNames.insert({val, name});
    os << " " << name;
  } else {
    auto name = vars.newName("out");
    outNames.insert({val, name});
    os << " &" << name;
  }
}

void Emitter::emitValue(Value val) {
  if (auto name = lookupEmittedName(val)) {
    os << name;
    return;
  }
  auto *def = val.getDefiningOp();
  if (def) {
    emitOpError(val.getDefiningOp(), "was not defined (internal error)");
  }
  encounteredError = true;
}

void Emitter::clear() {
  valueNames.clear();
  localNames.clear();
  outNames.clear();
  vars.clear();
  storageNames.clear();
}

void Emitter::emitModel(ModelOp model) {
  indent() << "struct " << model.getName() << " {\n";
  auto storage = model.getBody().getArgument(0);
  addIndent();

  accumulateStorage(storage, inputs, outputs);

  emitStorage(storage);
  emitIterators(storage);

  for (auto clock : model.getOps<ClockTreeOp>()) {
    emitClockTree(clock, model);
    localNames.clear();
  }
  for (auto pass : model.getOps<PassThroughOp>()) {
    emitPassThrough(pass, model);
    localNames.clear();
  }

  reduceIndent();
  indent() << "};\n";
}

void Emitter::accumulateStorage(Value storage, SmallVector<RootInputOp> &inputs,
                                SmallVector<RootOutputOp> &outputs) {
  for (auto *op : storage.getUsers()) {
    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      accumulateStorage(substorage.getOutput(), inputs, outputs);
      continue;
    }
    if (!isa<AllocStateOp, AllocMemoryOp, RootInputOp, RootOutputOp>(op)) {
      emitOpError(op, "unknown storage user");
      continue;
    }

    if (!op->hasAttr("name")) {
      op->setAttr("name",
                  StringAttr::get(op->getContext(), vars.newName("tmp")));
    }
    std::string *str = new std::string(
        vars.newName(op->getAttrOfType<StringAttr>("name").getValue()).str());
    // replace '/' with '$' for C++ identifiers.
    std::replace((*str).begin(), (*str).end(), '/', '$');
    storageNames.emplace_back(str);
    llvm::StringRef name(*storageNames.back());
    TypeSwitch<Operation *>(op)
        .Case<RootInputOp>([&](RootInputOp input) {
          inputs.push_back(input);
          valueNames.insert({input, name});
        })
        .Case<RootOutputOp>([&](RootOutputOp output) {
          outputs.push_back(output);
          valueNames.insert({output, name});
        })
        .Case<AllocMemoryOp>([&](AllocMemoryOp mem) {
          internals.insert({name, mem});
          valueNames.insert({mem, name});
        })
        .Case<AllocStateOp>([&](AllocStateOp state) {
          internals.insert({name, state});
          valueNames.insert({state, name});
        });
  }
}

void Emitter::emitStorage(Value storage) {
  for (auto in : inputs) {
    indent();
    emitType(in.getType().getType());
    os << " " << in.getName() << ";\n";
  }

  for (auto out : outputs) {
    indent();
    emitType(out.getType().getType());
    os << " " << out.getName() << ";\n";
  }

  for (auto const &[name, op] : internals) {
    indent();
    if (auto mem = dyn_cast<AllocMemoryOp>(op)) {
      emitType(mem.getMemory().getType());
    } else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
      emitType(alloc.getType().getType());
    }
    os << " " << name << ";\n";
  }
}

void Emitter::emitIterators(Value storage) {
  indent() << "template<typename Lambda>\n";
  indent() << "void each_input(Lambda&& fn) {\n";
  addIndent();
  for (auto in : inputs) {
    indent() << "fn.template operator()<" << in.getType().getType().getWidth()
             << ">(\"" << in.getName() << "\", " << in.getName() << ");\n";
  }
  reduceIndent();
  indent() << "}\n";

  indent() << "template<typename Lambda>\n";
  indent() << "void each_output(Lambda&& fn) {\n";
  addIndent();
  for (auto out : outputs) {
    indent() << "fn.template operator()<" << out.getType().getType().getWidth()
             << ">(\"" << out.getName() << "\", " << out.getName() << ");\n";
  }
  reduceIndent();
  indent() << "}\n";

  indent() << "template<typename Lambda>\n";
  indent() << "void each_reg(Lambda&& fn) {\n";
  addIndent();
  for (auto const &[name, op] : internals) {
    if (auto alloc = dyn_cast<AllocStateOp>(op)) {
      indent() << "fn.template operator()<"
               << alloc.getType().getType().getWidth() << ">(\"" << name
               << "\", " << name << ");\n";
    }
  }
  reduceIndent();
  indent() << "}\n";
}

void Emitter::emitModelBodyOp(Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<comb::AddOp, comb::MulOp, comb::DivUOp, comb::DivSOp, comb::ModUOp,
            comb::ModSOp, comb::ShlOp, comb::ShrUOp, comb::ShrSOp, comb::SubOp,
            comb::AndOp, comb::OrOp, comb::XorOp, comb::ICmpOp, comb::ParityOp,
            comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp, comb::MuxOp>(
          [&](auto op) { emitComb(op); })
      .Case<arc::StateOp, arc::StateReadOp, arc::StateWriteOp,
            arc::MemoryWriteOp, arc::MemoryReadOp, arc::ClockGateOp>(
          [&](auto op) { emitArc(op); })
      .Case<hw::ConstantOp, hw::ArrayCreateOp, hw::ArrayGetOp,
            hw::AggregateConstantOp>([&](auto op) { emitHW(op); })
      .Default([&](auto op) {
        if (!isa<AllocStateOp, AllocMemoryOp, RootInputOp, RootOutputOp,
                 ClockTreeOp, PassThroughOp>(op))
          emitOpError(op, "cannot be emitted");
      });
}

void Emitter::emitClockTree(ClockTreeOp clock, ModelOp model) {
  indent() << "void " << vars.newName("clock") << "() {\n";
  addIndent();

  for (auto &op : model.getBodyBlock()) {
    emitModelBodyOp(&op);
  }

  for (auto &op : clock.getBodyBlock()) {
    emitModelBodyOp(&op);
  }

  reduceIndent();
  indent() << "}\n";
}

void Emitter::emitPassThrough(PassThroughOp pass, ModelOp model) {
  indent() << "void " << vars.newName("passthrough") << "() {\n";
  addIndent();
  for (auto &op : model.getBodyBlock()) {
    emitModelBodyOp(&op);
  }
  for (auto &op : pass.getBodyBlock()) {
    emitModelBodyOp(&op);
  }
  reduceIndent();
  indent() << "}\n";
}

void Emitter::emitDefine(DefineOp define) {
  indent() << "static inline void __attribute__((always_inline)) "
           << define.getName() << "(";

  unsigned i = 0;
  unsigned last = define.getNumArguments() - 1;
  for (auto &arg : define.getArguments()) {
    emitDef(arg);

    if (i != last) {
      os << ", ";
    }
    i++;
  }

  auto outputOp = cast<arc::OutputOp>(define.getBodyBlock().getTerminator());
  for (auto arg : outputOp.getOutputs()) {
    os << ", ";
    emitDef(arg, /*output=*/true);
    outNames.insert({arg, vars.newName("out")});
  }

  os << ") {\n";

  addIndent();
  for (auto &op : define.getBodyBlock()) {
    TypeSwitch<Operation *>(&op)
        .Case<comb::AddOp, comb::MulOp, comb::DivUOp, comb::DivSOp,
              comb::ModUOp, comb::ModSOp, comb::ShlOp, comb::ShrUOp,
              comb::ShrSOp, comb::SubOp, comb::AndOp, comb::OrOp, comb::XorOp,
              comb::ICmpOp, comb::ParityOp, comb::ExtractOp, comb::ConcatOp,
              comb::ReplicateOp, comb::MuxOp>([&](auto op) { emitComb(op); })
        .Case<arc::StateOp, arc::OutputOp>([&](auto op) { emitArc(op); })
        .Case<hw::ConstantOp, hw::ArrayCreateOp, hw::ArrayGetOp,
              hw::AggregateConstantOp>([&](auto op) { emitHW(op); })
        .Default([&](auto op) { emitOpError(op, "op cannot be exported"); });
  }
  reduceIndent();
  indent() << "}\n";

  clear();
}

void Emitter::emitHW(ConstantOp op) {
  indent();
  emitDef(op.getResult());
  os << " = ";
  os << "Const<" << op.getType().cast<IntegerType>().getWidth() << ">";
  os << "(" << op.getValueAttr().getValue() << ");\n";
}

void Emitter::emitHW(AggregateConstantOp op) {
  if (auto array = dyn_cast<hw::ArrayType>(op.getType())) {
    indent();
    emitDef(op.getResult());
    os << "({";
    auto fields = op.getFields();
    for (int i = fields.size() - 1; i >= 0; i--) {
      os << "Const<" << array.getElementType().cast<IntegerType>().getWidth()
         << ">";
      os << "(" << fields[i].cast<IntegerAttr>().getValue() << ")";
      if (i != 0) {
        os << ", ";
      }
    }
    os << "});\n";
  } else {
    emitError(op, "hw.aggregate_constant only supported with array type");
  }
}

void Emitter::emitHW(ArrayCreateOp op) {
  indent();
  emitDef(op.getResult());
  os << "({";
  unsigned last = op.getNumOperands() - 1;
  for (int i = last; i >= 0; i--) {
    auto arg = op.getOperand(i);
    emitValue(arg);
    if (i != 0) {
      os << ", ";
    }
  }
  os << "});\n";
}

void Emitter::emitHW(ArrayGetOp op) {
  indent();
  emitDef(op.getResult());
  os << " = ";
  emitValue(op.getInput());
  os << ".get(";
  emitValue(op.getIndex());
  os << ");\n";
}

// Macro to define a binary op emitter.
#define EMIT_BINOP(optype, name)                                               \
  void Emitter::emitComb(optype op) {                                          \
    auto result = op.getResult();                                              \
    indent();                                                                  \
    emitDef(result);                                                           \
    os << " = ";                                                               \
    emitValue(op.getLhs());                                                    \
    os << "." #name "(";                                                       \
    emitValue(op.getRhs());                                                    \
    os << ");\n";                                                              \
  }

// Macro to define a variadic op emitter.
#define EMIT_VAROP(optype, symbol)                                             \
  void Emitter::emitComb(optype op) {                                          \
    auto result = op.getResult();                                              \
    indent();                                                                  \
    emitDef(result);                                                           \
    os << " = ";                                                               \
    unsigned i = 0;                                                            \
    unsigned last = op.getNumOperands() - 1;                                   \
    for (auto arg : op.getOperands()) {                                        \
      emitValue(arg);                                                          \
      if (i != last) {                                                         \
        os << " " #symbol " ";                                                 \
      }                                                                        \
      i++;                                                                     \
    }                                                                          \
    os << ";\n";                                                               \
  }

EMIT_BINOP(comb::DivUOp, udiv)
EMIT_BINOP(comb::DivSOp, sdiv)
EMIT_BINOP(comb::ModUOp, umod)
EMIT_BINOP(comb::ModSOp, smod)
EMIT_BINOP(comb::ShlOp, shl)
EMIT_BINOP(comb::ShrUOp, ushr)
EMIT_BINOP(comb::ShrSOp, sshr)
EMIT_BINOP(comb::SubOp, sub)

EMIT_VAROP(comb::AddOp, +)
EMIT_VAROP(comb::MulOp, *)
EMIT_VAROP(comb::AndOp, &)
EMIT_VAROP(comb::XorOp, ^)
EMIT_VAROP(comb::OrOp, |)

void Emitter::emitComb(comb::ConcatOp op) {
  indent();
  emitDef(op.getResult());
  os << " = ";
  emitValue(op.getOperand(0));
  for (unsigned i = 1; i < op.getNumOperands(); i++) {
    auto arg = op.getOperand(i);
    os << ".concat<" << arg.getType().cast<IntegerType>().getWidth() << ">(";
    emitValue(arg);
    os << ")";
  }
  os << ";\n";
}

void Emitter::emitComb(comb::ICmpOp op) {
  indent();
  emitDef(op.getResult());
  // TODO: Try to find the built-in predicate printer.
  StringRef pred;
  switch (op.getPredicate()) {
  case comb::ICmpPredicate::eq:
    pred = "eq";
    break;
  case comb::ICmpPredicate::ne:
    pred = "ne";
    break;
  case comb::ICmpPredicate::slt:
    pred = "slt";
    break;
  case comb::ICmpPredicate::ult:
    pred = "ult";
    break;
  case comb::ICmpPredicate::sle:
    pred = "sle";
    break;
  case comb::ICmpPredicate::ule:
    pred = "ule";
    break;
  case comb::ICmpPredicate::sgt:
    pred = "sgt";
    break;
  case comb::ICmpPredicate::ugt:
    pred = "ugt";
    break;
  case comb::ICmpPredicate::sge:
    pred = "sge";
    break;
  case comb::ICmpPredicate::uge:
    pred = "uge";
    break;
  default:
    emitOpError(op, "uses unsupported predicate");
  }
  os << " = ";
  emitValue(op.getLhs());
  os << "." << pred << "(";
  emitValue(op.getRhs());
  os << ");\n";
}

void Emitter::emitComb(comb::ParityOp op) {
  auto result = op.getResult();
  indent();
  emitDef(result);
  os << " = ";
  emitValue(op.getInput());
  os << ".parity();\n";
}

void Emitter::emitComb(comb::ExtractOp op) {
  auto result = op.getResult();
  auto width = result.getType().cast<IntegerType>().getWidth();
  indent();
  emitDef(result);
  os << " = ";
  emitValue(op.getInput());
  os << ".extract<" << op.getLowBit() + width - 1 << ", " << op.getLowBit()
     << ">(";
  os << ");\n";
}

void Emitter::emitComb(comb::ReplicateOp op) {
  auto result = op.getResult();
  indent();
  emitDef(result);
  os << " = ";
  emitValue(op.getInput());
  os << ".replicate<" << op.getMultiple() << ">();\n";
}

void Emitter::emitComb(comb::MuxOp op) {
  auto result = op.getResult();
  indent();
  emitDef(result);
  os << " = ";
  emitValue(op.getTrueValue());
  os << ".mux(";
  emitValue(op.getCond());
  os << ", ";
  emitValue(op.getFalseValue());
  os << ");\n";
}

void Emitter::emitArc(arc::ClockGateOp op) {
  indent();
  emitDef(op.getOutput());
  os << " = ";
  emitValue(op.getInput());
  os << " & ";
  emitValue(op.getEnable());
  os << ";\n";
}

void Emitter::emitArc(arc::StateReadOp op) {
  indent();
  emitDef(op.getResult());
  os << " = ";
  emitValue(op.getState());
  os << ";\n";
}

void Emitter::emitArc(arc::StateWriteOp op) {
  indent();
  emitValue(op.getState());
  os << " = ";
  emitValue(op.getValue());
  os << ";\n";
}

void Emitter::emitArc(arc::MemoryWriteOp op) {
  indent();
  emitValue(op.getMemory());
  os << ".write(";
  emitValue(op.getAddress());
  os << ", ";
  emitValue(op.getEnable());
  os << ", ";
  emitValue(op.getData());
  os << ");\n";
}

void Emitter::emitArc(arc::MemoryReadOp op) {
  indent();
  emitDef(op.getResult());
  os << " = ";
  emitValue(op.getMemory());
  os << ".read(";
  emitValue(op.getAddress());
  os << ", Int<1>(1));\n";
}

void Emitter::emitArc(arc::StateOp op) {
  for (auto out : op.getResults()) {
    indent();
    emitDef(out);
    os << ";\n";
  }

  indent() << op.getArc() << "(";

  for (auto in : op.getInputs()) {
    emitValue(in);
    os << ", ";
  }
  unsigned i = 0;
  unsigned last = op.getNumResults() - 1;
  for (auto out : op.getResults()) {
    emitValue(out);
    if (i != last) {
      os << ", ";
    }
    i++;
  }
  os << ");\n";
}

void Emitter::emitArc(arc::OutputOp op) {
  for (auto out : op.getOutputs()) {
    indent();
    auto it = outNames.find(out);
    if (it == outNames.end()) {
      emitOpError(op, "invalid output");
      return;
    }
    os << it->second;
    os << " = ";
    emitValue(out);
    os << ";\n";
  }
  indent() << "return;\n";
}

void Emitter::emitDesign(mlir::ModuleOp module) {
  StringRef name = "Arc";
  if (module.getName().has_value()) {
    name = module.getName().value();
  } else if (!module.getOps<ModelOp>().empty()) {
    name = (*module.getOps<ModelOp>().begin()).getName();
  }

  indent() << "namespace " << name << " {\n";

  for (auto op : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    emitDefine(op);
  }

  for (auto op : llvm::make_early_inc_range(module.getOps<ModelOp>())) {
    emitModel(op);
  }

  indent() << "}\n";
}

namespace circt {
namespace arc {

mlir::LogicalResult exportCppFile(mlir::ModuleOp module,
                                  llvm::raw_ostream &os) {
  Emitter emitter(os);

  emitter.emitDesign(module);
  return emitter.finalize();
}

void registerToCppFileTranslation() {
  mlir::TranslateFromMLIRRegistration toCpp(
      "export-cpp", "emit arc simulator to C++",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportCppFile(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<arc::ArcDialect>();
        registry.insert<comb::CombDialect>();
        registry.insert<hw::HWDialect>();
        registry.insert<seq::SeqDialect>();
      });
}

} // namespace arc
} // namespace circt


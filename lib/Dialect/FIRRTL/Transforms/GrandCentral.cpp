//===- GrandCentral.cpp - Ingest black box sources --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Implement SiFive's Grand Central transform.  Currently, this supports
// SystemVerilog Interface generation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/YAMLTraits.h"
#include <variant>

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;
using llvm::Optional;

//===----------------------------------------------------------------------===//
// Collateral for generating a YAML representation of a SystemVerilog interface
//===----------------------------------------------------------------------===//

namespace {

// These macros are used to provide hard-errors if a user tries to use the YAML
// infrastructure improperly.  We only implement conversion to YAML and not
// conversion from YAML.  The LLVM YAML infrastructure doesn't provide the
// ability to differentitate this and we don't need it for the purposes of
// Grand Central.
#define UNIMPLEMENTED_DEFAULT(clazz)                                           \
  llvm_unreachable("default '" clazz                                           \
                   "' construction is an intentionally *NOT* implemented "     \
                   "YAML feature (you should never be using this)");
#define UNIMPLEMENTED_DENORM(clazz)                                            \
  llvm_unreachable("conversion from YAML to a '" clazz                         \
                   "' is intentionally *NOT* implemented (you should not be "  \
                   "converting from YAML to an interface)");

// This namespace provides YAML-related collateral that is specific to Grand
// Central and should not be placed in the `llvm::yaml` namespace.
namespace yaml {

/// Context information necessary for YAML generation.
struct Context {
  /// A symbol table consisting of _only_ the interfaces construted by the Grand
  /// Central pass.  This is not a symbol table because we do not have an
  /// up-to-date symbol table that includes interfaces at the time the Grand
  /// Central pass finishes.  This structure is easier to build up and is only
  /// the information we need.
  DenseMap<Attribute, sv::InterfaceOp> &interfaceMap;
};

/// A representation of an `sv::InterfaceSignalOp` that includes additional
/// description information.
///
/// TODO: This could be removed if we add `firrtl.DocStringAnnotation` support
/// or if FIRRTL dialect included support for ops to specify "comment"
/// information.
struct DescribedSignal {
  /// The comment associated with this signal.
  StringAttr description;

  /// The signal.
  sv::InterfaceSignalOp signal;
};

/// This exist to work around the fact that no interface can be instantiated
/// inside another interface.  This serves to represent an op like this for the
/// purposes of conversion to YAML.
///
/// TODO: Fix this once we have a solution for #1464.
struct DescribedInstance {
  StringAttr name;

  /// A comment associated with the interface instance.
  StringAttr description;

  /// The dimensionality of the interface instantiation.
  ArrayAttr dimensions;

  /// The symbol associated with the interface.
  FlatSymbolRefAttr interface;
};

} // namespace yaml
} // namespace

// These macros tell the YAML infrastructure that these are types which can
// show up in vectors and provides implementations of how to serialize these.
// Each of these macros puts the resulting class into the `llvm::yaml` namespace
// (which is why these are outside the `llvm::yaml` namespace below).
LLVM_YAML_IS_SEQUENCE_VECTOR(::yaml::DescribedSignal)
LLVM_YAML_IS_SEQUENCE_VECTOR(::yaml::DescribedInstance)
LLVM_YAML_IS_SEQUENCE_VECTOR(sv::InterfaceOp)

// This `llvm::yaml` namespace contains implementations of classes that enable
// conversion from an `sv::InterfaceOp` to a YAML representation of that
// interface using [LLVM's YAML I/O library](https://llvm.org/docs/YamlIO.html).
namespace llvm {
namespace yaml {

using namespace ::yaml;

/// Convert newlines and comments to remove the comments.  This produces better
/// looking YAML output.  E.g., this will convert the following:
///
///   // foo
///   // bar
///
/// Into the following:
///
///   foo
///   bar
std::string static stripComment(StringRef str) {
  std::string descriptionString;
  llvm::raw_string_ostream stream(descriptionString);
  SmallVector<StringRef> splits;
  str.split(splits, "\n");
  llvm::interleave(
      splits,
      [&](auto substr) {
        substr.consume_front("//");
        stream << substr.drop_while([](auto c) { return c == ' '; });
      },
      [&]() { stream << "\n"; });
  return descriptionString;
}

/// Conversion from a `DescribedSignal` to YAML.  This is
/// implemented using YAML normalization to first convert this to an internal
/// `Field` structure which has a one-to-one mapping to the YAML represntation.
template <>
struct MappingContextTraits<DescribedSignal, Context> {
  /// A one-to-one representation with a YAML representation of a signal/field.
  struct Field {
    /// The name of the field.
    StringRef name;

    /// An optional, textual description of what the field is.
    Optional<std::string> description;

    /// The dimensions of the field.
    SmallVector<unsigned, 2> dimensions;

    /// The width of the underlying type.
    unsigned width;

    /// Construct a `Field` from a `DescribedSignal` (an `sv::InterfaceSignalOp`
    /// with an optional description).
    Field(IO &io, DescribedSignal &op)
        : name(op.signal.sym_nameAttr().getValue()) {

      // Convert the description from a `StringAttr` (which may be null) to an
      // `Optional<StringRef>`.  This aligns exactly with the YAML
      // representation.
      if (op.description)
        description = stripComment(op.description.getValue());

      // Unwrap the type of the field into an array of dimensions and a width.
      // By example, this is going from the following hardware type:
      //
      //     !hw.uarray<1xuarray<2xuarray<3xi8>>>
      //
      // To the following representation:
      //
      //     dimensions: [ 3, 2, 1 ]
      //     width: 8
      //
      // Note that the above is equivalenet to the following Verilog
      // specification.
      //
      //     wire [7:0] foo [2:0][1:0][0:0]
      //
      // Do this by repeatedly unwrapping unpacked array types until you get to
      // the underlying type.  The dimensions need to be reversed as this
      // unwrapping happens in reverse order of the final representation.
      auto tpe = op.signal.type();
      while (auto vector = tpe.dyn_cast<hw::UnpackedArrayType>()) {
        dimensions.push_back(vector.getSize());
        tpe = vector.getElementType();
      }
      dimensions = SmallVector<unsigned>(llvm::reverse(dimensions));

      // The final non-array type must be an integer.  Leave this as an assert
      // with a blind cast because we generated this type in this pass (and we
      // therefore cannot fail this cast).
      assert(tpe.isa<IntegerType>());
      width = tpe.cast<IntegerType>().getWidth();
    }

    /// A no-argument constructor is necessary to work with LLVM's YAML library.
    Field(IO &io){UNIMPLEMENTED_DEFAULT("Field")}

    /// This cannot be denomralized back to an interface op.
    DescribedSignal denormalize(IO &) {
      UNIMPLEMENTED_DENORM("DescribedSignal")
    }
  };

  static void mapping(IO &io, DescribedSignal &op, Context &ctx) {
    MappingNormalization<Field, DescribedSignal> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapOptional("description", keys->description);
    io.mapRequired("dimensions", keys->dimensions);
    io.mapRequired("width", keys->width);
  }
};

/// Conversion from a `DescribedInstance` to YAML.  This is implemented using
/// YAML normalization to first convert the `DescribedInstance` to an internal
/// `Instance` struct which has a one-to-one representation with the final YAML
/// representation.
template <>
struct MappingContextTraits<DescribedInstance, Context> {
  /// A YAML-serializable representation of an interface instantiation.
  struct Instance {
    /// The name of the interface.
    StringRef name;

    /// An optional textual description of the interface.
    Optional<std::string> description = None;

    /// An array describing the dimnensionality of the interface.
    SmallVector<int64_t, 2> dimensions;

    /// The underlying interface.
    FlatSymbolRefAttr interface;

    Instance(IO &io, DescribedInstance &op)
        : name(op.name.getValue()), interface(op.interface) {

      // Convert the description from a `StringAttr` (which may be null) to an
      // `Optional<StringRef>`.  This aligns exactly with the YAML
      // representation.
      if (op.description)
        description = stripComment(op.description.getValue());

      for (auto &d : op.dimensions) {
        auto dimension = d.dyn_cast<IntegerAttr>();
        dimensions.push_back(dimension.getInt());
      }
    }

    Instance(IO &io){UNIMPLEMENTED_DEFAULT("Instance")}

    DescribedInstance denormalize(IO &) {
      UNIMPLEMENTED_DENORM("DescribedInstance")
    }
  };

  static void mapping(IO &io, DescribedInstance &op, Context &ctx) {
    MappingNormalization<Instance, DescribedInstance> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapOptional("description", keys->description);
    io.mapRequired("dimensions", keys->dimensions);
    io.mapRequired("interface", ctx.interfaceMap[keys->interface], ctx);
  }
};

/// Conversion from an `sv::InterfaceOp` to YAML.  This is implemented using
/// YAML normalization to first convert the interface to an internal `Interface`
/// which reformats the Grand Central-generated interface into the YAML format.
template <>
struct MappingContextTraits<sv::InterfaceOp, Context> {
  /// A YAML-serializable representation of an interface.  This consists of
  /// fields (vector or ground types) and nested interfaces.
  struct Interface {
    /// The name of the interface.
    StringRef name;

    /// All ground or vectors that make up the interface.
    std::vector<DescribedSignal> fields;

    /// Instantiations of _other_ interfaces.
    std::vector<DescribedInstance> instances;

    /// Construct an `Interface` from an `sv::InterfaceOp`.  This is tuned to
    /// "parse" the structure of an interface that the Grand Central pass
    /// generates.  The structure of `Field`s and `Instance`s is documented
    /// below.
    ///
    /// A field will look like the following.  The verbatim description is
    /// optional:
    ///
    ///     sv.verbatim "// <description>" {
    ///       firrtl.grandcentral.yaml.type = "description",
    ///       symbols = []}
    ///     sv.interface.signal @<name> : <type>
    ///
    /// An interface instanctiation will look like the following.  The verbatim
    /// description is optional.
    ///
    ///     sv.verbatim "// <description>" {
    ///       firrtl.grandcentral.type = "description",
    ///       symbols = []}
    ///     sv.verbatim "<name> <symbol>();" {
    ///       firrtl.grandcentral.yaml.name = "<name>",
    ///       firrtl.grandcentral.yaml.dimensions = [<first dimension>, ...],
    ///       firrtl.grandcentral.yaml.symbol = @<symbol>,
    ///       firrtl.grandcentral.yaml.type = "instance",
    ///       symbols = []}
    ///
    Interface(IO &io, sv::InterfaceOp &op) : name(op.getName()) {
      // A mutable store of the description.  This occurs in the op _before_ the
      // field or instance, so we need someplace to put it until we use it.
      StringAttr description = {};

      for (auto &op : op.getBodyBlock()->getOperations()) {
        TypeSwitch<Operation *>(&op)
            // A verbatim op is either a description or an interface
            // instantiation.
            .Case<sv::VerbatimOp>([&](sv::VerbatimOp op) {
              auto tpe = op->getAttrOfType<StringAttr>(
                  "firrtl.grandcentral.yaml.type");

              // This is a descripton.  Update the mutable description and
              // continue;
              if (tpe.getValue() == "description") {
                description = op.stringAttr();
                return;
              }

              // This is an unsupported construct. Just drop it.
              if (tpe.getValue() == "unsupported") {
                description = {};
                return;
              }

              // This is an instance of another interface.  Add the symbol to
              // the vector of instances.
              auto name = op->getAttrOfType<StringAttr>(
                  "firrtl.grandcentral.yaml.name");
              auto dimensions = op->getAttrOfType<ArrayAttr>(
                  "firrtl.grandcentral.yaml.dimensions");
              auto symbol = op->getAttrOfType<FlatSymbolRefAttr>(
                  "firrtl.grandcentral.yaml.symbol");
              instances.push_back(
                  DescribedInstance({name, description, dimensions, symbol}));
              description = {};
            })
            // An interface signal op is a field.
            .Case<sv::InterfaceSignalOp>([&](sv::InterfaceSignalOp op) {
              fields.push_back(DescribedSignal({description, op}));
              description = {};
            });
      }
    }

    /// A no-argument constructor is necessary to work with LLVM's YAML library.
    Interface(IO &io){UNIMPLEMENTED_DEFAULT("Interface")}

    /// This cannot be denomralized back to an interface op.
    sv::InterfaceOp denormalize(IO &) {
      UNIMPLEMENTED_DENORM("sv::InterfaceOp")
    }
  };

  static void mapping(IO &io, sv::InterfaceOp &op, Context &ctx) {
    MappingNormalization<Interface, sv::InterfaceOp> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapRequired("fields", keys->fields, ctx);
    io.mapRequired("instances", keys->instances, ctx);
  }
};

} // namespace yaml
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

/// A helper to build verbatim strings with symbol placeholders. Provides a
/// mechanism to snapshot the current string and symbols and restore back to
/// this state after modifications. These snapshots are particularly useful when
/// the string is assembled through hierarchical traversal of some sort, which
/// populates the string with a prefix common to all children of a hierarchy
/// (like the interface field traversal in the `GrandCentralPass`).
///
/// The intended use is as follows:
///
///     void baz(VerbatimBuilder &v) {
///       foo(v.snapshot().append("bar"));
///     }
///
/// The function `baz` takes a snapshot of the current verbatim text `v`, adds
/// "bar" to it and calls `foo` with that appended verbatim text. After the call
/// to `foo` returns, any changes made by `foo` as well as the "bar" are dropped
/// from the verbatim text `v`, as the temporary snapshot goes out of scope.
struct VerbatimBuilder {
  struct Base {
    SmallString<128> string;
    SmallVector<Attribute> symbols;
    VerbatimBuilder builder() { return VerbatimBuilder(*this); }
    operator VerbatimBuilder() { return builder(); }
  };

  /// Constructing a builder will snapshot the `Base` which holds the actual
  /// string and symbols.
  VerbatimBuilder(Base &base)
      : base(base), stringBaseSize(base.string.size()),
        symbolsBaseSize(base.symbols.size()) {}

  /// Destroying a builder will reset the `Base` to the original string and
  /// symbols.
  ~VerbatimBuilder() {
    base.string.resize(stringBaseSize);
    base.symbols.resize(symbolsBaseSize);
  }

  // Disallow copying.
  VerbatimBuilder(const VerbatimBuilder &) = delete;
  VerbatimBuilder &operator=(const VerbatimBuilder &) = delete;

  /// Take a snapshot of the current string and symbols. This returns a new
  /// `VerbatimBuilder` that will reset to the current state of the string once
  /// destroyed.
  VerbatimBuilder snapshot() { return VerbatimBuilder(base); }

  /// Get the current string.
  StringRef getString() const { return base.string; }
  /// Get the current symbols;
  ArrayRef<Attribute> getSymbols() const { return base.symbols; }

  /// Append to the string.
  VerbatimBuilder &append(char c) {
    base.string.push_back(c);
    return *this;
  }

  /// Append to the string.
  VerbatimBuilder &append(const Twine &twine) {
    twine.toVector(base.string);
    return *this;
  }

  /// Append a placeholder and symbol to the string.
  VerbatimBuilder &append(Attribute symbol) {
    unsigned id = base.symbols.size();
    base.symbols.push_back(symbol);
    append("{{" + Twine(id) + "}}");
    return *this;
  }

  VerbatimBuilder &operator+=(char c) { return append(c); }
  VerbatimBuilder &operator+=(const Twine &twine) { return append(twine); }
  VerbatimBuilder &operator+=(Attribute symbol) { return append(symbol); }

private:
  Base &base;
  size_t stringBaseSize;
  size_t symbolsBaseSize;
};

/// A wrapper around a string that is used to encode a type which cannot be
/// represented by an mlir::Type for some reason.  This is currently used to
/// represent either an interface, a n-dimensional vector of interfaces, or a
/// tombstone for an actually unsupported type (e.g., an AugmentedBooleanType).
struct VerbatimType {
  /// The textual representation of the type.
  std::string str;

  /// True if this is a type which must be "instatiated" and requires a trailing
  /// "()".
  bool instantiation;

  /// A vector storing the width of each dimension of the type.
  SmallVector<int32_t, 4> dimensions = {};

  /// Serialize this type to a string.
  std::string toStr(StringRef name) {
    SmallString<64> stringType(str);
    stringType.append(" ");
    stringType.append(name);
    for (auto d : llvm::reverse(dimensions)) {
      stringType.append("[");
      stringType.append(Twine(d).str());
      stringType.append("]");
    }
    if (instantiation)
      stringType.append("()");
    stringType.append(";");
    return std::string(stringType);
  }
};

/// A sum type representing either a type encoded as a string (VerbatimType)
/// or an actual mlir::Type.
typedef std::variant<VerbatimType, Type> TypeSum;

/// Stores the information content of an ExtractGrandCentralAnnotation.
struct ExtractionInfo {
  /// The directory where Grand Central generated collateral (modules,
  /// interfaces, etc.) will be written.
  StringAttr directory = {};

  /// The name of the file where any binds will be written.  This will be placed
  /// in the same output area as normal compilation output, e.g., output
  /// Verilog.  This has no relation to the `directory` member.
  StringAttr bindFilename = {};
};

/// Stores information about the companion module of a GrandCentral view.
struct CompanionInfo {
  StringRef name;

  FModuleOp companion;

  FModuleOp mapping;
};

/// Stores a reference to a ground type and an optional NLA associated with
/// that field.
struct FieldAndNLA {
  FieldRef field;
  FlatSymbolRefAttr nlaSym;
};

/// Generate SystemVerilog interfaces from Grand Central annotations.  This pass
/// roughly works in the following three phases:
///
/// 1. Extraction information is determined.
///
/// 2. The circuit is walked to find all scattered annotations related to Grand
///    Central interfaces.  These are: (a) the parent module, (b) the companion
///    module, and (c) all leaves that are to be connected to the interface.
///
/// 3. The circuit-level Grand Central annotation is walked to both generate and
///    instantiate interfaces and to generate the "mappings" file that produces
///    cross-module references (XMRs) to drive the interface.
struct GrandCentralPass : public GrandCentralBase<GrandCentralPass> {
  void runOnOperation() override;

private:
  /// Optionally build an AugmentedType from an attribute.  Return none if the
  /// attribute is not a dictionary or if it does not match any of the known
  /// templates for AugmentedTypes.
  Optional<Attribute> fromAttr(Attribute attr);

  /// Mapping of ID to leaf ground type and an optional non-local annotation
  /// associated with that ID.
  DenseMap<Attribute, FieldAndNLA> leafMap;

  /// Mapping of ID to parent instance and module.  If this module is the top
  /// module, then the first tuple member will be None.
  DenseMap<Attribute, std::pair<Optional<InstanceOp>, FModuleOp>> parentIDMap;

  /// Mapping of ID to companion module.
  DenseMap<Attribute, CompanionInfo> companionIDMap;

  /// An optional prefix applied to all interfaces in the design.  This is set
  /// based on a PrefixInterfacesAnnotation.
  StringRef interfacePrefix;

  NLATable *nlaTable;

  /// The set of NLAs that are dead after this pass.  These will be removed
  /// before the pass finishes.
  DenseSet<StringAttr> deadNLAs;

  /// The design-under-test (DUT) as determined by the presence of a
  /// "sifive.enterprise.firrtl.MarkDUTAnnotation".  This will be null if no DUT
  /// was found.
  FModuleLike dut;

  /// An optional directory for testbench-related files.  This is null if no
  /// "TestBenchDirAnnotation" is found.
  StringAttr testbenchDir;

  /// Return a string containing the name of an interface.  Apply correct
  /// prefixing from the interfacePrefix and module-level prefix parameter.
  std::string getInterfaceName(StringAttr prefix,
                               AugmentedBundleTypeAttr bundleType) {

    if (prefix)
      return (prefix.getValue() + interfacePrefix +
              bundleType.getDefName().getValue())
          .str();
    return (interfacePrefix + bundleType.getDefName().getValue()).str();
  }

  /// Recursively examine an AugmentedType to populate the "mappings" file
  /// (generate XMRs) for this interface.  This does not build new interfaces.
  bool traverseField(Attribute field, IntegerAttr id, VerbatimBuilder &path);

  /// Recursively examine an AugmentedType to both build new interfaces and
  /// populate a "mappings" file (generate XMRs) using `traverseField`.  Return
  /// the type of the field exmained.
  Optional<TypeSum> computeField(Attribute field, IntegerAttr id,
                                 StringAttr prefix, VerbatimBuilder &path);

  /// Recursively examine an AugmentedBundleType to both build new interfaces
  /// and populate a "mappings" file (generate XMRs).  Return none if the
  /// interface is invalid.
  Optional<sv::InterfaceOp> traverseBundle(AugmentedBundleTypeAttr bundle,
                                           IntegerAttr id, StringAttr prefix,
                                           VerbatimBuilder &path);

  /// Return the module associated with this value.
  FModuleLike getEnclosingModule(Value value, FlatSymbolRefAttr sym = {});

  /// Inforamtion about how the circuit should be extracted.  This will be
  /// non-empty if an extraction annotation is found.
  Optional<ExtractionInfo> maybeExtractInfo = None;

  /// A filename describing where to put a YAML representation of the
  /// interfaces generated by this pass.
  Optional<StringAttr> maybeHierarchyFileYAML = None;

  StringAttr getOutputDirectory() {
    if (maybeExtractInfo.hasValue())
      return maybeExtractInfo.getValue().directory;
    return {};
  }

  /// Store of an instance paths analysis.  This is constructed inside
  /// `runOnOperation`, to work around the deleted copy constructor of
  /// `InstancePathCache`'s internal `BumpPtrAllocator`.
  ///
  /// TODO: Investigate a way to not use a pointer here like how `getNamespace`
  /// works below.
  InstancePathCache *instancePaths = nullptr;

  /// The namespace associated with the circuit.  This is lazily constructed
  /// using `getNamesapce`.
  Optional<CircuitNamespace> circuitNamespace = None;

  /// The module namespaces. These are lazily constructed by
  /// `getModuleNamespace`.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// Return a reference to the circuit namespace.  This will lazily construct a
  /// namespace if one does not exist.
  CircuitNamespace &getNamespace() {
    if (!circuitNamespace)
      circuitNamespace = CircuitNamespace(getOperation());
    return circuitNamespace.getValue();
  }

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  /// A symbol table associated with the circuit.  This is lazily constructed by
  /// `getSymbolTable`.
  Optional<SymbolTable> symbolTable = None;

  /// Return a reference to a circuit-level symbol table.  Lazily construct one
  /// if such a symbol table does not already exist.
  SymbolTable &getSymbolTable() {
    if (!symbolTable)
      symbolTable = SymbolTable(getOperation());
    return symbolTable.getValue();
  }

  // Utility that acts like emitOpError, but does _not_ include a note.  The
  // note in emitOpError includes the entire op which means the **ENTIRE**
  // FIRRTL circuit.  This doesn't communicate anything useful to the user
  // other than flooding their terminal.
  InFlightDiagnostic emitCircuitError(StringRef message = {}) {
    return emitError(getOperation().getLoc(), "'firrtl.circuit' op " + message);
  }

  // Insert comment delimiters ("// ") after newlines in the description string.
  // This is necessary to prevent introducing invalid verbatim Verilog.
  //
  // TODO: Add a comment op and lower the description to that.
  // TODO: Tracking issue: https://github.com/llvm/circt/issues/1677
  std::string cleanupDescription(StringRef description) {
    StringRef head;
    SmallString<64> out;
    do {
      std::tie(head, description) = description.split("\n");
      out.append(head);
      if (!description.empty())
        out.append("\n// ");
    } while (!description.empty());
    return std::string(out);
  }

  /// A store of the YAML representation of interfaces.
  DenseMap<Attribute, sv::InterfaceOp> interfaceMap;

  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);

  /// Returns a port's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(FModuleLike module, size_t portIdx);

  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op);

  /// Obtain an inner reference to a module port, possibly adding an `inner_sym`
  /// to that port.
  hw::InnerRefAttr getInnerRefTo(FModuleLike module, size_t portIdx);
};

} // namespace

//===----------------------------------------------------------------------===//
// Code related to handling Grand Central View annotations
//===----------------------------------------------------------------------===//

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// any annotations it may contain.  This is going to generate two types of
/// annotations:
///   1) Annotations necessary to build interfaces and store them at "~"
///   2) Scattered annotations for how components bind to interfaces
static Optional<DictionaryAttr>
parseAugmentedType(ApplyState &state, DictionaryAttr augmentedType,
                   DictionaryAttr root, StringRef companion, StringAttr name,
                   StringAttr defName, Optional<IntegerAttr> id,
                   Optional<StringAttr>(description), Twine clazz,
                   Twine path = {}) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  /// Optionally unpack a ReferenceTarget encoded as a DictionaryAttr.  Return
  /// either a pair containing the Target string (up to the reference) and an
  /// array of components or none if the input is malformed.  The input
  /// DicionaryAttr encoding is a JSON object of a serialized ReferenceTarget
  /// Scala class.  By example, this is converting:
  ///   ~Foo|Foo>a.b[0]
  /// To:
  ///   {"~Foo|Foo>a", {".b", "[0]"}}
  /// The format of a ReferenceTarget object like:
  ///   circuit: String
  ///   module: String
  ///   path: Seq[(Instance, OfModule)]
  ///   ref: String
  ///   component: Seq[TargetToken]
  auto refToTarget =
      [&](DictionaryAttr refTarget) -> llvm::Optional<std::string> {
    auto circuitAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "circuit", loc, clazz, path);
    auto moduleAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "module", loc, clazz, path);
    auto pathAttr =
        tryGetAs<ArrayAttr>(refTarget, refTarget, "path", loc, clazz, path);
    auto componentAttr = tryGetAs<ArrayAttr>(refTarget, refTarget, "component",
                                             loc, clazz, path);
    if (!circuitAttr || !moduleAttr || !pathAttr || !componentAttr)
      return {};

    // Parse non-local annotations.
    SmallString<32> strpath;
    for (auto p : pathAttr) {
      auto dict = p.dyn_cast_or_null<DictionaryAttr>();
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto instHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_1", loc, clazz, path);
      auto modHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_2", loc, clazz, path);
      if (!instHolder || !modHolder) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto inst = tryGetAs<StringAttr>(instHolder, instHolder, "value", loc,
                                       clazz, path);
      auto mod =
          tryGetAs<StringAttr>(modHolder, modHolder, "value", loc, clazz, path);
      if (!inst || !mod) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      strpath += "/" + inst.getValue().str() + ":" + mod.getValue().str();
    }

    SmallVector<Attribute> componentAttrs;
    SmallString<32> componentStr;
    for (size_t i = 0, e = componentAttr.size(); i != e; ++i) {
      auto cPath = (path + ".component[" + Twine(i) + "]").str();
      auto component = componentAttr[i];
      auto dict = component.dyn_cast_or_null<DictionaryAttr>();
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz + "' with path '" + cPath +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto classAttr =
          tryGetAs<StringAttr>(dict, refTarget, "class", loc, clazz, cPath);
      if (!classAttr)
        return {};

      auto value = dict.get("value");

      // A subfield like "bar" in "~Foo|Foo>foo.bar".
      if (auto field = value.dyn_cast<StringAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Field" &&
               "A StringAttr target token must be found with a subfield target "
               "token.");
        componentStr.append((Twine(".") + field.getValue()).str());
        continue;
      }

      // A subindex like "42" in "~Foo|Foo>foo[42]".
      if (auto index = value.dyn_cast<IntegerAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Index" &&
               "An IntegerAttr target token must be found with a subindex "
               "target token.");
        componentStr.append(
            (Twine("[") + Twine(index.getValue().getZExtValue()) + "]").str());
        continue;
      }

      mlir::emitError(loc,
                      "Annotation '" + clazz + "' with path '" + cPath +
                          ".value has unexpected type (should be StringAttr "
                          "for subfield  or IntegerAttr for subindex).")
              .attachNote()
          << "The value received was: " << value << "\n";
      return {};
    }

    auto refAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "ref", loc, clazz, path);

    return llvm::Optional<std::string>(
        {(Twine("~" + circuitAttr.getValue() + "|" + moduleAttr.getValue() +
                strpath + ">" + refAttr.getValue()) +
          componentStr)
             .str()});
  };

  auto classAttr =
      tryGetAs<StringAttr>(augmentedType, root, "class", loc, clazz, path);
  if (!classAttr)
    return None;
  StringRef classBase = classAttr.getValue();
  if (!classBase.consume_front("sifive.enterprise.grandcentral.Augmented")) {
    mlir::emitError(loc,
                    "the 'class' was expected to start with "
                    "'sifive.enterprise.grandCentral.Augmented*', but was '" +
                        classAttr.getValue() + "' (Did you misspell it?)")
            .attachNote()
        << "see annotation: " << augmentedType;
    return None;
  }

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classBase == "BundleType") {
    defName =
        tryGetAs<StringAttr>(augmentedType, root, "defName", loc, clazz, path);
    if (!defName)
      return None;

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmenetedType
    SmallVector<Attribute> elements;
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    for (size_t i = 0, e = elementsAttr.size(); i != e; ++i) {
      auto field = elementsAttr[i].dyn_cast_or_null<DictionaryAttr>();
      if (!field) {
        mlir::emitError(
            loc,
            "Annotation '" + Twine(clazz) + "' with path '.elements[" +
                Twine(i) +
                "]' contained an unexpected type (expected a DictionaryAttr).")
                .attachNote()
            << "The received element was: " << elementsAttr[i] << "\n";
        return None;
      }
      auto ePath = (path + ".elements[" + Twine(i) + "]").str();
      auto name = tryGetAs<StringAttr>(field, root, "name", loc, clazz, ePath);
      auto tpe =
          tryGetAs<DictionaryAttr>(field, root, "tpe", loc, clazz, ePath);
      Optional<StringAttr> description = None;
      if (auto maybeDescription = field.get("description"))
        description = maybeDescription.cast<StringAttr>();
      auto eltAttr =
          parseAugmentedType(state, tpe, root, companion, name, defName, None,
                             description, clazz, path);
      if (!name || !tpe || !eltAttr)
        return None;

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", maybeDescription.cast<StringAttr>());
      attrs.append("name", name);
      attrs.append("tpe", tpe.getAs<StringAttr>("class"));
      elements.push_back(eltAttr.getValue());
    }
    // Add an annotation that stores information necessary to construct the
    // module for the view.  This needs the name of the module (defName) and the
    // names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    if (id)
      attrs.append("id", id.getValue());
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // An AugmentedGroundType looks like:
  //   "ref": ReferenceTarget
  //   "tpe": GroundType
  // The ReferenceTarget is not serialized to a string.  The GroundType will
  // either be an actual FIRRTL ground type or a GrandCentral uninferred type.
  // This can be ignored for us.
  if (classBase == "GroundType") {
    auto maybeTarget = refToTarget(augmentedType.getAs<DictionaryAttr>("ref"));
    if (!maybeTarget) {
      mlir::emitError(loc, "Failed to parse ReferenceTarget").attachNote()
          << "See the full Annotation here: " << root;
      return None;
    }

    auto id = state.newID();

    // TODO: We don't support non-local annotations, so force this annotation
    // into a local annotation.  This does not properly check that the
    // non-local and local targets are totally equivalent.
    auto target = maybeTarget.getValue();

    NamedAttrList elementIface, elementScattered;

    // Populate the annotation for the interface element.
    elementIface.append("class", classAttr);
    if (description)
      elementIface.append("description", description.getValue());
    elementIface.append("id", id);
    elementIface.append("name", name);
    // Populate an annotation that will be scattered onto the element.
    elementScattered.append("class", classAttr);
    elementScattered.append("id", id);
    // If there are sub-targets, then add these.
    auto targetAttr = StringAttr::get(context, target);
    elementScattered.append("target", targetAttr);

    state.addToWorklistFn(
        DictionaryAttr::getWithSorted(context, elementScattered));

    return DictionaryAttr::getWithSorted(context, elementIface);
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classBase == "VectorType") {
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    SmallVector<Attribute> elements;
    for (auto elt : elementsAttr) {
      auto eltAttr = parseAugmentedType(
          state, elt.cast<DictionaryAttr>(), root, companion, name,
          StringAttr::get(context, ""), id, None, clazz, path);
      if (!eltAttr)
        return None;
      elements.push_back(eltAttr.getValue());
    }
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Any of the following are known and expected, but are legacy AugmentedTypes
  // do not have a target:
  //   - AugmentedStringType
  //   - AugmentedBooleanType
  //   - AugmentedIntegerType
  //   - AugmentedDoubleType
  bool isIgnorable =
      llvm::StringSwitch<bool>(classBase)
          .Cases("StringType", "BooleanType", "IntegerType", "DoubleType", true)
          .Default(false);
  if (isIgnorable)
    return augmentedType;

  // Anything else is unexpected or a user error if they manually wrote
  // annotations.  Print an error and error out.
  mlir::emitError(loc, "found unknown AugmentedType '" + classAttr.getValue() +
                           "' (Did you misspell it?)")
          .attachNote()
      << "see annotation: " << augmentedType;
  return None;
}

LogicalResult circt::firrtl::applyGCTView(const AnnoPathValue &target,
                                          DictionaryAttr anno,
                                          ApplyState &state) {

  auto id = state.newID();
  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();
  NamedAttrList companionAttrs, parentAttrs;
  companionAttrs.append("class", StringAttr::get(context, companionAnnoClass));
  companionAttrs.append("id", id);
  companionAttrs.append("type", StringAttr::get(context, "companion"));
  auto viewAttr =
      tryGetAs<DictionaryAttr>(anno, anno, "view", loc, viewAnnoClass);
  if (!viewAttr)
    return failure();
  auto name = tryGetAs<StringAttr>(anno, anno, "name", loc, viewAnnoClass);
  if (!name)
    return failure();
  companionAttrs.append("name", name);
  auto companionAttr =
      tryGetAs<StringAttr>(anno, anno, "companion", loc, viewAnnoClass);
  if (!companionAttr)
    return failure();
  companionAttrs.append("target", companionAttr);
  state.addToWorklistFn(DictionaryAttr::get(context, companionAttrs));

  auto parentAttr =
      tryGetAs<StringAttr>(anno, anno, "parent", loc, viewAnnoClass);
  if (!parentAttr)
    return failure();
  parentAttrs.append("class", StringAttr::get(context, parentAnnoClass));
  parentAttrs.append("id", id);
  parentAttrs.append("name", name);
  parentAttrs.append("target", parentAttr);
  parentAttrs.append("type", StringAttr::get(context, "parent"));
  state.addToWorklistFn(DictionaryAttr::get(context, parentAttrs));

  auto prunedAttr =
      parseAugmentedType(state, viewAttr, anno, companionAttr.getValue(), name,
                         {}, id, {}, viewAnnoClass, "view");
  if (!prunedAttr)
    return failure();

  AnnotationSet annotations(state.circuit);
  annotations.addAnnotations({prunedAttr.getValue()});
  annotations.applyToOperation(state.circuit);

  return success();
}

//===----------------------------------------------------------------------===//
// GrandCentralPass Implementation
//===----------------------------------------------------------------------===//

Optional<Attribute> GrandCentralPass::fromAttr(Attribute attr) {
  auto dict = attr.dyn_cast<DictionaryAttr>();
  if (!dict) {
    emitCircuitError() << "attribute is not a dictionary: " << attr << "\n";
    return None;
  }

  auto clazz = dict.getAs<StringAttr>("class");
  if (!clazz) {
    emitCircuitError() << "missing 'class' key in " << dict << "\n";
    return None;
  }

  auto classBase = clazz.getValue();
  classBase.consume_front("sifive.enterprise.grandcentral.Augmented");

  if (classBase == "BundleType") {
    if (dict.getAs<StringAttr>("defName") && dict.getAs<ArrayAttr>("elements"))
      return AugmentedBundleTypeAttr::get(&getContext(), dict);
    emitCircuitError() << "has an invalid AugmentedBundleType that does not "
                          "contain 'defName' and 'elements' fields: "
                       << dict;
  } else if (classBase == "VectorType") {
    if (dict.getAs<StringAttr>("name") && dict.getAs<ArrayAttr>("elements"))
      return AugmentedVectorTypeAttr::get(&getContext(), dict);
    emitCircuitError() << "has an invalid AugmentedVectorType that does not "
                          "contain 'name' and 'elements' fields: "
                       << dict;
  } else if (classBase == "GroundType") {
    auto id = dict.getAs<IntegerAttr>("id");
    auto name = dict.getAs<StringAttr>("name");
    if (id && leafMap.count(id) && name)
      return AugmentedGroundTypeAttr::get(&getContext(), dict);
    if (!id || !name)
      emitCircuitError() << "has an invalid AugmentedGroundType that does not "
                            "contain 'id' and 'name' fields:  "
                         << dict;
    if (id && !leafMap.count(id))
      emitCircuitError() << "has an AugmentedGroundType with 'id == "
                         << id.getValue().getZExtValue()
                         << "' that does not have a scattered leaf to connect "
                            "to in the circuit "
                            "(was the leaf deleted or constant prop'd away?)";
  } else if (classBase == "StringType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedStringTypeAttr::get(&getContext(), dict);
  } else if (classBase == "BooleanType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedBooleanTypeAttr::get(&getContext(), dict);
  } else if (classBase == "IntegerType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedIntegerTypeAttr::get(&getContext(), dict);
  } else if (classBase == "DoubleType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedDoubleTypeAttr::get(&getContext(), dict);
  } else if (classBase == "LiteralType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedLiteralTypeAttr::get(&getContext(), dict);
  } else if (classBase == "DeletedType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedDeletedTypeAttr::get(&getContext(), dict);
  } else {
    emitCircuitError() << "has an invalid AugmentedType";
  }
  return None;
}

static StringAttr getModPart(Attribute pathSegment) {
  return TypeSwitch<Attribute, StringAttr>(pathSegment)
      .Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
      .Case<hw::InnerRefAttr>([](auto a) { return a.getModule(); });
}

bool GrandCentralPass::traverseField(Attribute field, IntegerAttr id,
                                     VerbatimBuilder &path) {
  return TypeSwitch<Attribute, bool>(field)
      .Case<AugmentedGroundTypeAttr>([&](AugmentedGroundTypeAttr ground) {
        auto [fieldRef, sym] = leafMap.lookup(ground.getID());
        HierPathOp nla;
        if (sym)
          nla = nlaTable->getNLA(sym.getAttr());
        Value leafValue = fieldRef.getValue();
        unsigned fieldID = fieldRef.getFieldID();
        assert(leafValue && "leafValue not found");

        auto builder =
            OpBuilder::atBlockEnd(companionIDMap.lookup(id).mapping.getBody());

        FIRRTLType tpe = leafValue.getType().cast<FIRRTLType>();

        // If the type is zero-width then do not emit an XMR.
        if (!tpe.getBitWidthOrSentinel())
          return true;

        // Generate the path from the LCA to the module that contains the leaf.
        path += " = ";

        // There are two posisibilites for what this is tapping:
        //   1. This is a constant that will be synced into the mappings file.
        //   2. This is something else and we need an XMR.
        // Handle case (1) here and exit.  Handle case (2) following.
        auto uloc = builder.getUnknownLoc();
        auto driver = getDriverFromConnect(leafValue);
        if (driver) {
          if (auto constant =
                  dyn_cast_or_null<ConstantOp>(driver.getDefiningOp())) {
            path.append(Twine(constant.value().getBitWidth()));
            path += "'h";
            SmallString<32> valueStr;
            constant.value().toStringUnsigned(valueStr, 16);
            path.append(valueStr);
            builder.create<sv::VerbatimOp>(
                uloc,
                StringAttr::get(&getContext(),
                                "assign " + path.getString() + ";"),
                ValueRange{}, ArrayAttr::get(&getContext(), path.getSymbols()));
            return true;
          }
        }

        // Populate a hierarchical path to the leaf.  For an NLA this is just
        // the namepath of the associated hierarchical path.  For a local
        // annotation, this is computed from the instance path.
        SmallVector<Attribute> fullLeafPath;
        if (nla) {
          fullLeafPath.append(nla.namepath().begin(), nla.namepath().end());
        } else {
          FModuleLike enclosing = getEnclosingModule(leafValue, sym);
          auto enclosingPaths = instancePaths->getAbsolutePaths(enclosing);
          assert(enclosingPaths.size() == 1 &&
                 "Unable to handle multiply instantiated companions");
          if (enclosingPaths.size() != 1)
            return false;
          StringAttr root =
              instancePaths->instanceGraph.getTopLevelModule().moduleNameAttr();
          for (auto segment : enclosingPaths[0]) {
            fullLeafPath.push_back(getInnerRefTo(segment));
            root = segment.moduleNameAttr().getAttr();
          }
          fullLeafPath.push_back(FlatSymbolRefAttr::get(root));
        }

        // Compute the lowest common ancestor (LCA) of the leaf path and the
        // parent module.  This enables the generated XMR to be as short as
        // possible while not losing specificity.
        ArrayRef<Attribute> minimalLeafPath(fullLeafPath);
        StringAttr parentNameAttr =
            parentIDMap.lookup(id).second.moduleNameAttr();
        minimalLeafPath = minimalLeafPath.drop_until(
            [&](Attribute attr) { return getModPart(attr) == parentNameAttr; });

        path += FlatSymbolRefAttr::get(getModPart(minimalLeafPath.front()));
        if (minimalLeafPath.size() > 0) {
          for (auto segment : minimalLeafPath.drop_back()) {
            path += ".";
            path += segment;
          }
        }

        // Add the leaf value to the path.
        path += '.';
        if (auto blockArg = leafValue.dyn_cast<BlockArgument>()) {
          auto module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
          path += getInnerRefTo(module, blockArg.getArgNumber());
        } else {
          path += getInnerRefTo(leafValue.getDefiningOp());
        }

        if (fieldID > tpe.getMaxFieldID()) {
          leafValue.getDefiningOp()->emitError()
              << "subannotation with fieldID=" << fieldID
              << " is too large for type " << tpe;
          return false;
        }

        // Construct a path given by fieldID.
        while (fieldID) {
          TypeSwitch<FIRRTLType>(tpe)
              .template Case<FVectorType>([&](FVectorType vector) {
                unsigned index = vector.getIndexForFieldID(fieldID);
                tpe = vector.getElementType();
                fieldID -= vector.getFieldID(index);
                path.append("[" + Twine(index) + "]");
              })
              .template Case<BundleType>([&](BundleType bundle) {
                unsigned index = bundle.getIndexForFieldID(fieldID);
                tpe = bundle.getElementType(index);
                fieldID -= bundle.getFieldID(index);
                // FIXME: Invalid verilog names (e.g. "begin", "reg", .. ) will
                // be renamed at ExportVerilog so the path constructed here
                // might become invalid. We can use an inner name ref to encode
                // a reference to a subfield.
                path.append("." + Twine(bundle.getElement(index).name));
              })
              .Default([&](auto op) {
                llvm_unreachable(
                    "fieldID > maxFieldID case must be already handled");
              });
        }

        // Assemble the verbatim op.
        builder.create<sv::VerbatimOp>(
            uloc,
            StringAttr::get(&getContext(), "assign " + path.getString() + ";"),
            ValueRange{}, ArrayAttr::get(&getContext(), path.getSymbols()));
        ++numXMRs;
        return true;
      })
      .Case<AugmentedVectorTypeAttr>([&](auto vector) {
        bool notFailed = true;
        auto elements = vector.getElements();
        for (size_t i = 0, e = elements.size(); i != e; ++i) {
          auto field = fromAttr(elements[i]);
          if (!field)
            return false;
          notFailed &=
              traverseField(field.getValue(), id,
                            path.snapshot().append("[" + Twine(i) + "]"));
        }
        return notFailed;
      })
      .Case<AugmentedBundleTypeAttr>([&](AugmentedBundleTypeAttr bundle) {
        bool anyFailed = true;
        for (auto element : bundle.getElements()) {
          auto field = fromAttr(element);
          if (!field)
            return false;
          auto name = element.cast<DictionaryAttr>().getAs<StringAttr>("name");
          if (!name)
            name = element.cast<DictionaryAttr>().getAs<StringAttr>("defName");
          anyFailed &=
              traverseField(field.getValue(), id,
                            path.snapshot().append("." + name.getValue()));
        }

        return anyFailed;
      })
      .Case<AugmentedStringTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedBooleanTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedIntegerTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedDoubleTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedLiteralTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedDeletedTypeAttr>([&](auto a) { return false; })
      .Default([](auto a) { return true; });
}

Optional<TypeSum> GrandCentralPass::computeField(Attribute field,
                                                 IntegerAttr id,
                                                 StringAttr prefix,
                                                 VerbatimBuilder &path) {

  auto unsupported = [&](StringRef name, StringRef kind) {
    return VerbatimType({("// <unsupported " + kind + " type>").str(), false});
  };

  return TypeSwitch<Attribute, Optional<TypeSum>>(field)
      .Case<AugmentedGroundTypeAttr>(
          [&](AugmentedGroundTypeAttr ground) -> Optional<TypeSum> {
            // Traverse to generate mappings.
            if (!traverseField(field, id, path))
              return None;
            FieldRef fieldRef = leafMap.lookup(ground.getID()).field;
            auto value = fieldRef.getValue();
            auto fieldID = fieldRef.getFieldID();
            auto tpe = value.getType().cast<FIRRTLType>().getFinalTypeByFieldID(
                fieldID);
            if (!tpe.isGround()) {
              value.getDefiningOp()->emitOpError()
                  << "cannot be added to interface with id '"
                  << id.getValue().getZExtValue()
                  << "' because it is not a ground type";
              return None;
            }
            return TypeSum(IntegerType::get(getOperation().getContext(),
                                            tpe.getBitWidthOrSentinel()));
          })
      .Case<AugmentedVectorTypeAttr>(
          [&](AugmentedVectorTypeAttr vector) -> Optional<TypeSum> {
            auto elements = vector.getElements();
            auto firstElement = fromAttr(elements[0]);
            auto elementType =
                computeField(firstElement.getValue(), id, prefix,
                             path.snapshot().append("[" + Twine(0) + "]"));
            if (!elementType)
              return None;

            for (size_t i = 1, e = elements.size(); i != e; ++i) {
              auto subField = fromAttr(elements[i]);
              if (!subField)
                return None;
              (void)traverseField(subField.getValue(), id,
                                  path.snapshot().append("[" + Twine(i) + "]"));
            }

            if (auto *tpe = std::get_if<Type>(&elementType.getValue()))
              return TypeSum(
                  hw::UnpackedArrayType::get(*tpe, elements.getValue().size()));
            auto str = std::get<VerbatimType>(elementType.getValue());
            str.dimensions.push_back(elements.getValue().size());
            return TypeSum(str);
          })
      .Case<AugmentedBundleTypeAttr>(
          [&](AugmentedBundleTypeAttr bundle) -> TypeSum {
            auto iface = traverseBundle(bundle, id, prefix, path);
            assert(iface && iface.getValue());
            (void)iface;
            return VerbatimType({getInterfaceName(prefix, bundle), true});
          })
      .Case<AugmentedStringTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "string");
      })
      .Case<AugmentedBooleanTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "boolean");
      })
      .Case<AugmentedIntegerTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "integer");
      })
      .Case<AugmentedDoubleTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "double");
      })
      .Case<AugmentedLiteralTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "literal");
      })
      .Case<AugmentedDeletedTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "deleted");
      });
}

/// Traverse an Annotation that is an AugmentedBundleType.  During
/// traversal, construct any discovered SystemVerilog interfaces.  If this
/// is the root interface, instantiate that interface in the parent. Recurse
/// into fields of the AugmentedBundleType to construct nested interfaces
/// and generate stringy-typed SystemVerilog hierarchical references to
/// drive the interface. Returns false on any failure and true on success.
Optional<sv::InterfaceOp>
GrandCentralPass::traverseBundle(AugmentedBundleTypeAttr bundle, IntegerAttr id,
                                 StringAttr prefix, VerbatimBuilder &path) {
  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  sv::InterfaceOp iface;
  builder.setInsertionPointToEnd(getOperation().getBody());
  auto loc = getOperation().getLoc();
  auto iFaceName = getNamespace().newName(getInterfaceName(prefix, bundle));
  iface = builder.create<sv::InterfaceOp>(loc, iFaceName);
  ++numInterfaces;
  if (dut &&
      !instancePaths->instanceGraph.isAncestor(companionIDMap[id].companion,
                                               cast<hw::HWModuleLike>(*dut)) &&
      testbenchDir)
    iface->setAttr("output_file",
                   hw::OutputFileAttr::getFromDirectoryAndFilename(
                       &getContext(), testbenchDir.getValue(),
                       iFaceName + ".sv",
                       /*excludFromFileList=*/true));
  else if (maybeExtractInfo)
    iface->setAttr("output_file",
                   hw::OutputFileAttr::getFromDirectoryAndFilename(
                       &getContext(), getOutputDirectory().getValue(),
                       iFaceName + ".sv",
                       /*excludFromFileList=*/true));

  builder.setInsertionPointToEnd(cast<sv::InterfaceOp>(iface).getBody());

  for (auto element : bundle.getElements()) {
    auto field = fromAttr(element);
    if (!field)
      return None;

    auto name = element.cast<DictionaryAttr>().getAs<StringAttr>("name");
    // auto signalSym = hw::InnerRefAttr::get(iface.sym_nameAttr(), name);
    // TODO: The `append(name.getValue())` in the following should actually be
    // `append(signalSym)`, but this requires that `computeField` and the
    // functions it calls always return a type for which we can construct an
    // `InterfaceSignalOp`. Since nested interface instances are currently
    // busted (due to the interface being a symbol table), this doesn't work at
    // the moment. Passing a `name` works most of the time, but can be brittle
    // if the interface field requires renaming in the output (e.g. due to
    // naming conflicts).
    auto elementType =
        computeField(field.getValue(), id, prefix,
                     path.snapshot().append(".").append(name.getValue()));
    if (!elementType)
      return None;

    auto uloc = builder.getUnknownLoc();
    auto description =
        element.cast<DictionaryAttr>().getAs<StringAttr>("description");
    if (description) {
      auto descriptionOp = builder.create<sv::VerbatimOp>(
          uloc, ("// " + cleanupDescription(description.getValue())));

      // If we need to generate a YAML representation of this interface, then
      // add an attribute indicating that this `sv::VerbatimOp` is actually a
      // description.
      if (maybeHierarchyFileYAML)
        descriptionOp->setAttr("firrtl.grandcentral.yaml.type",
                               builder.getStringAttr("description"));
    }

    if (auto *str = std::get_if<VerbatimType>(&elementType.getValue())) {
      auto instanceOp =
          builder.create<sv::VerbatimOp>(uloc, str->toStr(name.getValue()));

      // If we need to generate a YAML representation of the interface, then add
      // attirbutes that describe what this `sv::VerbatimOp` is.
      if (maybeHierarchyFileYAML) {
        if (str->instantiation)
          instanceOp->setAttr("firrtl.grandcentral.yaml.type",
                              builder.getStringAttr("instance"));
        else
          instanceOp->setAttr("firrtl.grandcentral.yaml.type",
                              builder.getStringAttr("unsupported"));
        instanceOp->setAttr("firrtl.grandcentral.yaml.name", name);
        instanceOp->setAttr("firrtl.grandcentral.yaml.dimensions",
                            builder.getI32ArrayAttr(str->dimensions));
        instanceOp->setAttr(
            "firrtl.grandcentral.yaml.symbol",
            FlatSymbolRefAttr::get(builder.getContext(), str->str));
      }
      continue;
    }

    auto tpe = std::get<Type>(elementType.getValue());
    builder.create<sv::InterfaceSignalOp>(uloc, name.getValue(), tpe);
  }

  interfaceMap[FlatSymbolRefAttr::get(builder.getContext(), iFaceName)] = iface;
  return iface;
}

/// Return the module that is associated with this value.  Use the cached/lazily
/// constructed symbol table to make this fast.
FModuleLike GrandCentralPass::getEnclosingModule(Value value,
                                                 FlatSymbolRefAttr sym) {
  if (auto blockArg = value.dyn_cast<BlockArgument>())
    return cast<FModuleOp>(blockArg.getOwner()->getParentOp());

  auto *op = value.getDefiningOp();
  if (InstanceOp instance = dyn_cast<InstanceOp>(op))
    return getSymbolTable().lookup<FModuleOp>(
        instance.moduleNameAttr().getValue());

  return op->getParentOfType<FModuleOp>();
}

/// This method contains the business logic of this pass.
void GrandCentralPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===- Running Grand Central Views/Interface Pass "
                             "-----------------------------===\n");

  CircuitOp circuitOp = getOperation();

  // Look at the circuit annotaitons to do two things:
  //
  // 1. Determine extraction information (directory and filename).
  // 2. Populate a worklist of all annotations that encode interfaces.
  //
  // Remove annotations encoding interfaces, but leave extraction information as
  // this may be needed by later passes.
  SmallVector<Annotation> worklist;
  bool removalError = false;
  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass("sifive.enterprise.grandcentral.AugmentedBundleType")) {
      worklist.push_back(anno);
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(extractGrandCentralClass)) {
      if (maybeExtractInfo.hasValue()) {
        emitCircuitError("more than one 'ExtractGrandCentralAnnotation' was "
                         "found, but exactly one must be provided");
        removalError = true;
        return false;
      }

      auto directory = anno.getMember<StringAttr>("directory");
      auto filename = anno.getMember<StringAttr>("filename");
      if (!directory || !filename) {
        emitCircuitError()
            << "contained an invalid 'ExtractGrandCentralAnnotation' that does "
               "not contain 'directory' and 'filename' fields: "
            << anno.getDict();
        removalError = true;
        return false;
      }

      maybeExtractInfo = {directory, filename};
      // Do not delete this annotation.  Extraction info may be needed later.
      return false;
    }
    if (anno.isClass("sifive.enterprise.grandcentral."
                     "GrandCentralHierarchyFileAnnotation")) {
      if (maybeHierarchyFileYAML.hasValue()) {
        emitCircuitError("more than one 'GrandCentralHierarchyFileAnnotation' "
                         "was found, but zero or one may be provided");
        removalError = true;
        return false;
      }

      auto filename = anno.getMember<StringAttr>("filename");
      if (!filename) {
        emitCircuitError()
            << "contained an invalid 'GrandCentralHierarchyFileAnnotation' "
               "that does not contain 'directory' and 'filename' fields: "
            << anno.getDict();
        removalError = true;
        return false;
      }

      maybeHierarchyFileYAML = filename;
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(
            "sifive.enterprise.grandcentral.PrefixInterfacesAnnotation")) {
      if (!interfacePrefix.empty()) {
        emitCircuitError("more than one 'PrefixInterfacesAnnotation' was "
                         "found, but zero or one may be provided");
        removalError = true;
        return false;
      }

      auto prefix = anno.getMember<StringAttr>("prefix");
      if (!prefix) {
        emitCircuitError()
            << "contained an invalid 'PrefixInterfacesAnnotation' that does "
               "not contain a 'prefix' field: "
            << anno.getDict();
        removalError = true;
        return false;
      }

      interfacePrefix = prefix.getValue();
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(testbenchDirAnnoClass)) {
      testbenchDir = anno.getMember<StringAttr>("dirname");
      return false;
    }
    return false;
  });

  // Find the DUT if it exists.  This needs to be known before the circuit is
  // walked.
  for (auto mod : circuitOp.getOps<FModuleLike>()) {
    if (!AnnotationSet(mod).hasAnnotation(dutAnnoClass))
      continue;

    // TODO: This check is duplicated multiple places, e.g., in
    // WireDFT.  This should be factored out as part of the annotation
    // lowering pass.
    if (dut) {
      auto diag = emitError(mod.getLoc())
                  << "is marked with a '" << dutAnnoClass << "', but '"
                  << dut.moduleName()
                  << "' also had such an annotation (this should "
                     "be impossible!)";
      diag.attachNote(dut.getLoc()) << "the first DUT was found here";
      removalError = true;
    }
    dut = mod;
  }

  if (removalError)
    return signalPassFailure();

  LLVM_DEBUG({
    llvm::dbgs() << "Extraction Info:\n";
    if (maybeExtractInfo)
      llvm::dbgs() << "  directory: " << maybeExtractInfo.getValue().directory
                   << "\n"
                   << "  filename: " << maybeExtractInfo.getValue().bindFilename
                   << "\n";
    else
      llvm::dbgs() << "  <none>\n";
    llvm::dbgs() << "DUT: ";
    if (dut)
      llvm::dbgs() << dut.moduleName() << "\n";
    else
      llvm::dbgs() << "<none>\n";
    llvm::dbgs()
        << "Prefix Info (from PrefixInterfacesAnnotation):\n"
        << "  prefix: " << interfacePrefix << "\n"
        << "Hierarchy File Info (from GrandCentralHierarchyFileAnnotation):\n"
        << "  filename: ";
    if (maybeHierarchyFileYAML)
      llvm::dbgs() << maybeHierarchyFileYAML.getValue();
    else
      llvm::dbgs() << "<none>";
    llvm::dbgs() << "\n";
  });

  // Exit immediately if no annotations indicative of interfaces that need to be
  // built exist.  However, still generate the YAML file if the annotation for
  // this was passed in because some flows expect this.
  if (worklist.empty()) {
    if (!maybeHierarchyFileYAML)
      return markAllAnalysesPreserved();
    std::string yamlString;
    llvm::raw_string_ostream stream(yamlString);
    ::yaml::Context yamlContext({interfaceMap});
    llvm::yaml::Output yout(stream);
    OpBuilder builder(circuitOp);
    SmallVector<sv::InterfaceOp, 0> interfaceVec;
    yamlize(yout, interfaceVec, true, yamlContext);
    builder.setInsertionPointToStart(circuitOp.getBody());
    builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), yamlString)
        ->setAttr("output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(),
                      maybeHierarchyFileYAML.getValue().getValue(),
                      /*excludFromFileList=*/true));
    LLVM_DEBUG({ llvm::dbgs() << "Generated YAML:" << yamlString << "\n"; });
    return;
  }

  // Setup the builder to create ops _inside the FIRRTL circuit_.  This is
  // necessary because interfaces and interface instances are created.
  // Instances link to their definitions via symbols and we don't want to
  // break this.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());

  // Maybe get an "id" from an Annotation.  Generate error messages on the op if
  // no "id" exists.
  auto getID = [&](Operation *op,
                   Annotation annotation) -> Optional<IntegerAttr> {
    auto id = annotation.getMember<IntegerAttr>("id");
    if (!id) {
      op->emitOpError()
          << "contained a malformed "
             "'sifive.enterprise.grandcentral.AugmentedGroundType' annotation "
             "that did not contain an 'id' field";
      removalError = true;
      return None;
    }
    return Optional(id);
  };

  /// TODO: Handle this differently to allow construction of an optionsl
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());
  instancePaths = &instancePathCache;

  // Maybe return the lone instance of a module.  Generate errors on the op if
  // the module is not instantiated or is multiply instantiated.
  auto exactlyOneInstance = [&](FModuleOp op,
                                StringRef msg) -> Optional<InstanceOp> {
    auto *node = instancePaths->instanceGraph[op];

    switch (node->getNumUses()) {
    case 0:
      op->emitOpError() << "is marked as a GrandCentral '" << msg
                        << "', but is never instantiated";
      return None;
    case 1:
      return cast<InstanceOp>(*(*node->uses().begin())->getInstance());
    default:
      auto diag = op->emitOpError()
                  << "is marked as a GrandCentral '" << msg
                  << "', but it is instantiated more than once";
      for (auto *instance : node->uses())
        diag.attachNote(instance->getInstance()->getLoc())
            << "parent is instantiated here";
      return None;
    }
  };

  nlaTable = &getAnalysis<NLATable>();

  /// Walk the circuit and extract all information related to scattered
  /// Grand Central annotations.  This is used to populate: (1) the
  /// companionIDMap, (2) the parentIDMap, and (3) the leafMap.
  /// Annotations are removed as they are discovered and if they are not
  /// malformed.
  removalError = false;
  auto trueAttr = builder.getBoolAttr(true);
  circuitOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<RegOp, RegResetOp, WireOp, NodeOp>([&](auto op) {
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.AugmentedGroundType"))
              return false;
            auto maybeID = getID(op, annotation);
            if (!maybeID)
              return false;
            auto sym =
                annotation.getMember<FlatSymbolRefAttr>("circt.nonlocal");
            leafMap[maybeID.getValue()] = {
                {op.getResult(), annotation.getFieldID()}, sym};
            if (sym)
              deadNLAs.insert(sym.getAttr());
            ++numAnnosRemoved;
            return true;
          });
        })
        // TODO: Figure out what to do with this.
        .Case<InstanceOp>([&](auto op) {
          AnnotationSet::removePortAnnotations(op, [&](unsigned i,
                                                       Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.AugmentedGroundType"))
              return false;
            op.emitOpError()
                << "is marked as an interface element, but this should be "
                   "impossible due to how the Chisel Grand Central API works";
            removalError = true;
            return false;
          });
        })
        .Case<MemOp>([&](auto op) {
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.AugmentedGroundType"))
              return false;
            op.emitOpError()
                << "is marked as an interface element, but this does not make "
                   "sense (is there a scattering bug or do you have a "
                   "malformed hand-crafted MLIR circuit?)";
            removalError = true;
            return false;
          });
          AnnotationSet::removePortAnnotations(
              op, [&](unsigned i, Annotation annotation) {
                if (!annotation.isClass(
                        "sifive.enterprise.grandcentral.AugmentedGroundType"))
                  return false;
                op.emitOpError()
                    << "has port '" << i
                    << "' marked as an interface element, but this does not "
                       "make sense (is there a scattering bug or do you have a "
                       "malformed hand-crafted MLIR circuit?)";
                removalError = true;
                return false;
              });
        })
        .Case<FModuleOp>([&](FModuleOp op) {
          // Handle annotations on the ports.
          AnnotationSet::removePortAnnotations(
              op, [&](unsigned i, Annotation annotation) {
                if (!annotation.isClass(
                        "sifive.enterprise.grandcentral.AugmentedGroundType"))
                  return false;
                auto maybeID = getID(op, annotation);
                if (!maybeID)
                  return false;
                auto sym =
                    annotation.getMember<FlatSymbolRefAttr>("circt.nonlocal");
                leafMap[maybeID.getValue()] = {
                    {op.getArgument(i), annotation.getFieldID()}, sym};
                if (sym)
                  deadNLAs.insert(sym.getAttr());
                ++numAnnosRemoved;
                return true;
              });

          // Handle annotations on the module.
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            // TODO: Change this to remove the "type" field as all these
            // annotations are specialized in the class with ".parent" or
            // ".companion" suffixes.
            if (!annotation.getClass().startswith(viewAnnoClass))
              return false;
            auto tpe = annotation.getMember<StringAttr>("type");
            auto name = annotation.getMember<StringAttr>("name");
            auto id = annotation.getMember<IntegerAttr>("id");
            if (!tpe) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain a 'type' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }
            if (!id) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain an 'id' field with an 'IntegerAttr' value";
              goto FModuleOp_error;
            }
            if (!name) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain a 'name' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }

            // If this is a companion, then:
            //   1. Insert it into the companion map
            //   2. Create a new mapping module.
            //   3. Instatiate the mapping module in the companion.
            //   4. Check that the companion is instantated exactly once.
            //   5. Set attributes on that lone instance so it will become a
            //      bind if extraction information was provided.  If a DUT is
            //      known, then anything in the test harness will not be
            //      extracted.
            if (tpe.getValue() == "companion") {
              builder.setInsertionPointToEnd(circuitOp.getBody());

              // Create the mapping module.
              auto mappingName =
                  getNamespace().newName(name.getValue() + "_mapping");
              auto mapping = builder.create<FModuleOp>(
                  circuitOp.getLoc(), builder.getStringAttr(mappingName),
                  ArrayRef<PortInfo>());
              if (maybeExtractInfo)
                mapping->setAttr(
                    "output_file",
                    hw::OutputFileAttr::getFromDirectoryAndFilename(
                        &getContext(), getOutputDirectory().getValue(),
                        mapping.getName() + ".sv",
                        /*excludeFromFilelist=*/true));
              companionIDMap[id] = {name.getValue(), op, mapping};

              // Instantiate the mapping module inside the companion.  Keep the
              // instance graph up-to-date with this new instantiation.
              builder.setInsertionPointToEnd(op.getBody());
              instancePaths->instanceGraph[op]->addInstance(
                  builder.create<InstanceOp>(circuitOp.getLoc(), mapping,
                                             mapping.getName()),
                  instancePathCache.instanceGraph.addModule(mapping));

              // Assert that the companion is instantiated once and only once.
              auto instance = exactlyOneInstance(op, "companion");
              if (!instance)
                return false;

              // If no extraction info was provided, exit.  Otherwise, setup the
              // lone instance of the companion to be lowered as a bind.
              if (!maybeExtractInfo) {
                ++numAnnosRemoved;
                return true;
              }

              // If the companion is instantiated above the DUT, then don't
              // extract it.
              if (dut && !instancePaths->instanceGraph.isAncestor(
                             op, cast<hw::HWModuleLike>(*dut))) {
                ++numAnnosRemoved;
                return true;
              }

              instance.getValue()->setAttr("lowerToBind", trueAttr);
              instance.getValue()->setAttr(
                  "output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(),
                      maybeExtractInfo.getValue().bindFilename.getValue(),
                      /*excludeFromFileList=*/true));
              op->setAttr("output_file",
                          hw::OutputFileAttr::getFromDirectoryAndFilename(
                              &getContext(),
                              maybeExtractInfo.getValue().directory.getValue(),
                              op.getName() + ".sv",
                              /*excludeFromFileList=*/true,
                              /*includeReplicatedOps=*/true));

              // Look for any blackboxes instantiated by the companion and mark
              // them for inclusion in the Grand Central extraction directory.
              SmallVector<FModuleOp> modules({op});
              DenseSet<Operation *> bboxes, visited({op, mapping});
              while (!modules.empty()) {
                auto mod = modules.pop_back_val();
                visited.insert(mod);
                for (auto inst : mod.getOps<InstanceOp>()) {
                  Operation *sub =
                      instancePaths->instanceGraph.getReferencedModule(inst);
                  if (visited.count(sub))
                    continue;
                  if (auto subMod = dyn_cast<FModuleOp>(sub)) {
                    modules.push_back(subMod);
                    continue;
                  }
                  auto subExtMod = cast<FExtModuleOp>(sub);
                  for (auto anno : AnnotationSet(subExtMod)) {
                    if (!anno.isClass("firrtl.transforms.BlackBoxInlineAnno") &&
                        !anno.isClass("firrtl.transforms.BlackBoxPathAnno"))
                      continue;
                    if (subExtMod->hasAttr("output_file"))
                      break;
                    subExtMod->setAttr(
                        "output_file",
                        hw::OutputFileAttr::getAsDirectory(
                            &getContext(),
                            maybeExtractInfo.getValue().directory.getValue(),
                            /*excludeFromFileList=*/false));
                    break;
                  }
                }
              }

              ++numAnnosRemoved;
              return true;
            }

            // Insert the parent into the parent map, asserting that the parent
            // is instantiated exatly once.
            if (tpe.getValue() == "parent") {
              // Assert that the parent is instantiated once and only once.
              // Allow for this to be the main module in the circuit.
              Optional<InstanceOp> instance;
              if (op != circuitOp.getMainModule()) {
                instance = exactlyOneInstance(op, "parent");
                if (!instance && circuitOp.getMainModule() != op)
                  return false;
              }

              parentIDMap[id] = {instance, cast<FModuleOp>(op)};
              ++numAnnosRemoved;
              return true;
            }

            op.emitOpError()
                << "has a 'sifive.enterprise.grandcentral.ViewAnnotation' with "
                   "an unknown or malformed 'type' field in annotation: "
                << annotation.getDict();

          FModuleOp_error:
            removalError = true;
            return false;
          });
        });
  });

  if (removalError)
    return signalPassFailure();

  // Check that a parent exists for every companion.
  for (auto a : companionIDMap) {
    if (parentIDMap.count(a.first) == 0) {
      emitCircuitError()
          << "contains a 'companion' with id '"
          << a.first.cast<IntegerAttr>().getValue().getZExtValue()
          << "', but does not contain a GrandCentral 'parent' with the same id";
      return signalPassFailure();
    }
  }

  // Check that a companion exists for every parent.
  for (auto a : parentIDMap) {
    if (companionIDMap.count(a.first) == 0) {
      emitCircuitError()
          << "contains a 'parent' with id '"
          << a.first.cast<IntegerAttr>().getValue().getZExtValue()
          << "', but does not contain a GrandCentral 'companion' with the same "
             "id";
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    // Print out the companion map, parent map, and all leaf values that
    // were discovered.  Sort these by their keys before printing to make
    // this easier to read.
    SmallVector<IntegerAttr> ids;
    auto sort = [&ids]() {
      llvm::sort(ids, [](IntegerAttr a, IntegerAttr b) {
        return a.getValue().getZExtValue() < b.getValue().getZExtValue();
      });
    };
    for (auto tuple : companionIDMap)
      ids.push_back(tuple.first.cast<IntegerAttr>());
    sort();
    llvm::dbgs() << "companionIDMap:\n";
    for (auto id : ids) {
      auto value = companionIDMap.lookup(id);
      llvm::dbgs() << "  - " << id.getValue() << ": "
                   << value.companion.getName() << " -> " << value.name << "\n";
    }
    llvm::dbgs() << "parentIDMap:\n";
    for (auto id : ids) {
      auto value = parentIDMap.lookup(id);
      StringRef name;
      if (value.first)
        name = value.first.getValue().name();
      else
        name = value.second.getName();
      llvm::dbgs() << "  - " << id.getValue() << ": " << name << ":"
                   << value.second.getName() << "\n";
    }
    ids.clear();
    for (auto tuple : leafMap)
      ids.push_back(tuple.first.cast<IntegerAttr>());
    sort();
    llvm::dbgs() << "leafMap:\n";
    for (auto id : ids) {
      auto fieldRef = leafMap.lookup(id).field;
      auto value = fieldRef.getValue();
      auto fieldID = fieldRef.getFieldID();
      if (auto blockArg = value.dyn_cast<BlockArgument>()) {
        FModuleOp module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
        llvm::dbgs() << "  - " << id.getValue() << ": "
                     << module.getName() + ">" +
                            module.getPortName(blockArg.getArgNumber());
        if (fieldID)
          llvm::dbgs() << ", fieldID=" << fieldID;
        llvm::dbgs() << "\n";
      } else {
        llvm::dbgs() << "  - " << id.getValue() << ": "
                     << value.getDefiningOp()
                            ->getAttr("name")
                            .cast<StringAttr>()
                            .getValue();
        if (fieldID)
          llvm::dbgs() << ", fieldID=" << fieldID;
        llvm::dbgs() << "\n";
      }
    }
  });

  // Now, iterate over the worklist of interface-encoding annotations to create
  // the interface and all its sub-interfaces (interfaces that it instantiates),
  // instantiate the top-level interface, and generate a "mappings file" that
  // will use XMRs to drive the interface.  If extraction info is available,
  // then the top-level instantiate interface will be marked for extraction via
  // a SystemVerilog bind.
  SmallVector<sv::InterfaceOp, 2> interfaceVec;
  for (auto anno : worklist) {
    auto bundle = AugmentedBundleTypeAttr::get(&getContext(), anno.getDict());

    // The top-level AugmentedBundleType must have a global ID field so that
    // this can be linked to the parent and companion.
    if (!bundle.isRoot()) {
      emitCircuitError() << "missing 'id' in root-level BundleType: "
                         << anno.getDict() << "\n";
      removalError = true;
      continue;
    }

    // Error if a matching parent or companion do not exist.
    if (parentIDMap.count(bundle.getID()) == 0) {
      emitCircuitError() << "no parent found with 'id' value '"
                         << bundle.getID().getValue().getZExtValue() << "'\n";
      removalError = true;
      continue;
    }
    if (companionIDMap.count(bundle.getID()) == 0) {
      emitCircuitError() << "no companion found with 'id' value '"
                         << bundle.getID().getValue().getZExtValue() << "'\n";
      removalError = true;
      continue;
    }

    // Decide on a symbol name to use for the interface instance. This is needed
    // in `traverseBundle` as a placeholder for the connect operations.
    auto parentModule = parentIDMap.lookup(bundle.getID()).second;
    auto symbolName = getNamespace().newName(
        "__" + companionIDMap.lookup(bundle.getID()).name + "_" +
        getInterfaceName(bundle.getPrefix(), bundle) + "__");

    // Recursively walk the AugmentedBundleType to generate interfaces and XMRs.
    // Error out if this returns None (indicating that the annotation annotation
    // is malformed in some way).  A good error message is generated inside
    // `traverseBundle` or the functions it calls.
    auto instanceSymbol =
        hw::InnerRefAttr::get(SymbolTable::getSymbolName(parentModule),
                              StringAttr::get(&getContext(), symbolName));
    VerbatimBuilder::Base verbatimData;
    VerbatimBuilder verbatim(verbatimData);
    verbatim += instanceSymbol;
    auto iface =
        traverseBundle(bundle, bundle.getID(), bundle.getPrefix(), verbatim);
    if (!iface) {
      removalError = true;
      continue;
    }
    ++numViews;

    interfaceVec.push_back(iface.getValue());

    // Instantiate the interface inside the parent.
    builder.setInsertionPointToEnd(parentModule.getBody());
    auto instance = builder.create<sv::InterfaceInstanceOp>(
        getOperation().getLoc(), iface.getValue().getInterfaceType(),
        companionIDMap.lookup(bundle.getID()).name,
        builder.getStringAttr(symbolName));

    // If no extraction information was present, then just leave the interface
    // instantiated in the parent.  Otherwise, make it a bind.
    if (!maybeExtractInfo)
      continue;

    // If the interface is associated with a companion that is instantiated
    // above the DUT (e.g.., in the test harness), then don't extract it.
    if (dut && !instancePaths->instanceGraph.isAncestor(
                   companionIDMap[bundle.getID()].companion,
                   cast<hw::HWModuleLike>(*dut)))
      continue;

    instance->setAttr("doNotPrint", trueAttr);
    builder.setInsertionPointToStart(
        instance->getParentOfType<CircuitOp>().getBody());
    auto bind = builder.create<sv::BindInterfaceOp>(getOperation().getLoc(),
                                                    instanceSymbol);
    bind->setAttr("output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(),
                      maybeExtractInfo.getValue().bindFilename.getValue(),
                      /*excludeFromFileList=*/true));
  }

  // If a `GrandCentralHierarchyFileAnnotation` was passed in, generate a YAML
  // representation of the interfaces that we produced with the filename that
  // that annotation provided.
  if (maybeHierarchyFileYAML) {
    std::string yamlString;
    llvm::raw_string_ostream stream(yamlString);
    ::yaml::Context yamlContext({interfaceMap});
    llvm::yaml::Output yout(stream);
    yamlize(yout, interfaceVec, true, yamlContext);

    builder.setInsertionPointToStart(circuitOp.getBody());
    builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), yamlString)
        ->setAttr("output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(),
                      maybeHierarchyFileYAML.getValue().getValue(),
                      /*excludFromFileList=*/true));
    LLVM_DEBUG({ llvm::dbgs() << "Generated YAML:" << yamlString << "\n"; });
  }

  // Garbage collect dead NLAs.
  auto symTable = getSymbolTable();
  for (auto &op :
       llvm::make_early_inc_range(circuitOp.getBody()->getOperations())) {

    // Remove NLA operations.
    if (auto nla = dyn_cast<HierPathOp>(op)) {
      if (deadNLAs.count(nla.sym_nameAttr())) {
        nlaTable->erase(nla);
        symTable.erase(nla);
      }
      continue;
    }

    auto fmodule = dyn_cast<FModuleOp>(op);
    if (!fmodule)
      continue;

    auto isDead = [&](Annotation anno) -> bool {
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!sym)
        return false;
      bool remove = deadNLAs.count(sym.getAttr());
      numAnnosRemoved += remove;
      return remove;
    };

    // Visit module bodies to remove any dead NLA breadcrumbs.
    for (auto op : fmodule.getBody()->getOps<InstanceOp>())
      AnnotationSet::removeAnnotations(op, isDead);
  }

  // Signal pass failure if any errors were found while examining circuit
  // annotations.
  if (removalError)
    return signalPassFailure();
  markAnalysesPreserved<NLATable>();
}

StringAttr GrandCentralPass::getOrAddInnerSym(FModuleLike module,
                                              size_t portIdx) {
  auto attr = module.getPortSymbolAttr(portIdx);
  if (attr && !attr.getValue().empty())
    return attr;
  StringRef nameHint = "gct_sym";
  if (auto attr = module.getPortNameAttr(portIdx))
    nameHint = attr.getValue();
  auto name = getModuleNamespace(module).newName(nameHint);
  attr = StringAttr::get(module.getContext(), name);
  module.setPortSymbolAttr(portIdx, attr);
  return attr;
}

hw::InnerRefAttr GrandCentralPass::getInnerRefTo(Operation *op) {
  return ::getInnerRefTo(op, "", [&](FModuleOp mod) -> ModuleNamespace & {
    return getModuleNamespace(mod);
  });
}

hw::InnerRefAttr GrandCentralPass::getInnerRefTo(FModuleLike module,
                                                 size_t portIdx) {
  return hw::InnerRefAttr::get(SymbolTable::getSymbolName(module),
                               getOrAddInnerSym(module, portIdx));
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralPass() {
  return std::make_unique<GrandCentralPass>();
}

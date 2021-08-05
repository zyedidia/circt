#include "Circt.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Combinational, comb,
                                      circt::comb::CombDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HW, hw, circt::hw::HWDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SV, sv, circt::sv::SVDialect)

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Circt Tcl Type support functions.
////////////////////////////////////////////////////////////////////////////////

static int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

static void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto *op = unwrap((MlirOperation){obj->internalRep.otherValuePtr});
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

static void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  dup->internalRep.otherValuePtr = src->internalRep.otherValuePtr;
}

static void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
}

////////////////////////////////////////////////////////////////////////////////
// Core Entry and Exit points.
////////////////////////////////////////////////////////////////////////////////

/// Called at the end of the interpreter, unloads circt.
static void Circt_Cleanup(ClientData data) {
  mlir::MLIRContext *context = (mlir::MLIRContext *)data;
  delete context;
}

/// Initialize the tcl binding.  Register types, provide packages, and
/// initialize Circt.
int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }

  // Initialize CIRCT
  mlir::MLIRContext *context = new mlir::MLIRContext;

  context->loadDialect<circt::hw::HWDialect, circt::comb::CombDialect,
                       circt::sv::SVDialect>();

  // Register Tcl types
  Tcl_ObjType *operationType = new Tcl_ObjType;
  operationType->name = "MlirOperation";
  operationType->setFromAnyProc = operationTypeSetFromAnyProc;
  operationType->updateStringProc = operationTypeUpdateStringProc;
  operationType->dupIntRepProc = operationTypeDupIntRepProc;
  operationType->freeIntRepProc = operationTypeFreeIntRepProc;
  Tcl_RegisterObjType(operationType);

  // Register package
  if (Tcl_PkgProvide(interp, "Circt", "1.0") == TCL_ERROR) {
    return TCL_ERROR;
  }

  // Register commands.  The mlir context is stored with the command structure.
  Tcl_CreateObjCommand(interp, "circt", circtCmd, (ClientData)context,
                       Circt_Cleanup);
  return TCL_OK;
}
}

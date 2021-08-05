#include <stdlib.h>
#include <tcl.h>

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

int circtCmd(ClientData cdata, Tcl_Interp *interp, int objc,
             Tcl_Obj *const objv[]);

void Circt_Cleanup(ClientData data) {
  mlir::MLIRContext *context = (mlir::MLIRContext *)data;
  delete context;
}

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }

  // Initialize CIRCT
  mlir::MLIRContext *context = new mlir::MLIRContext;

  context->loadDialect<circt::hw::HWDialect, circt::comb::CombDialect,
                       circt::sv::SVDialect>();

  // Register types

  Tcl_ObjType *operationType = new Tcl_ObjType;
  operationType->name = "MlirOperation";
  operationType->setFromAnyProc = operationTypeSetFromAnyProc;
  operationType->updateStringProc = operationTypeUpdateStringProc;
  operationType->dupIntRepProc = operationTypeDupIntRepProc;
  operationType->freeIntRepProc = operationTypeFreeIntRepProc;
  Tcl_RegisterObjType(operationType);

  Tcl_ObjType *moduleType = new Tcl_ObjType;
  moduleType->name = "MlirModule";
  moduleType->setFromAnyProc = moduleTypeSetFromAnyProc;
  moduleType->updateStringProc = moduleTypeUpdateStringProc;
  moduleType->dupIntRepProc = moduleTypeDupIntRepProc;
  moduleType->freeIntRepProc = moduleTypeFreeIntRepProc;
  Tcl_RegisterObjType(moduleType);

  // Register package
  if (Tcl_PkgProvide(interp, "Circt", "1.0") == TCL_ERROR) {
    return TCL_ERROR;
  }

  // Register commands
  Tcl_CreateObjCommand(interp, "circt", circtCmd, (ClientData)context,
                       Circt_Cleanup);
  return TCL_OK;
}
}

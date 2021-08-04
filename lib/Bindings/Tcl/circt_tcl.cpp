#include <stdlib.h>
#include <tcl.h>

#include "Circt.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/CAPI/Registration.h"

int dumpMLIR(ClientData cdata, Tcl_Interp *interp, int objc,
                    Tcl_Obj *const objv[]) {
  if (objc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, "something");
    return TCL_ERROR;
  }

mlir::ModuleOp op = unwrap((MlirModule){objv[1]->internalRep.otherValuePtr});
  op.dump();
  return TCL_OK;
}

  int loadFirMlirFile(ClientData cdata, Tcl_Interp * interp, int objc,
                      Tcl_Obj *const objv[]) {
    if (objc != 3) {
      Tcl_WrongNumArgs(interp, 1, objv, "something");
      return TCL_ERROR;
    }

    mlir::MLIRContext context;

    context.loadDialect<circt::hw::HWDialect, circt::comb::CombDialect,
                        circt::sv::SVDialect>();

    std::string errorMessage;
    int fileNameLen = 0;
    char *fileName = Tcl_GetStringFromObj(objv[2], &fileNameLen);
    auto input = mlir::openInputFile(fileName, &errorMessage);

    if (!input) {
      return TCL_ERROR;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

    mlir::OwningModuleRef module;
    int fileTypeLen = 0;
    char *fileType = Tcl_GetStringFromObj(objv[1], &fileTypeLen);
    if (!strcmp(fileType, "MLIR")) {
      module = mlir::parseSourceFile(sourceMgr, &context);
    } else if (!strcmp(fileType, "FIR")) {
      // TODO
      return TCL_ERROR;
    } else {
      return TCL_ERROR;
    }

    if (!module) {
      return TCL_ERROR;
    }

    MlirModule m = wrap(module.release());

    auto *obj = Tcl_NewObj();
    obj->typePtr = Tcl_GetObjType("MlirModule");
    obj->internalRep.otherValuePtr = (void*)m.ptr;
    obj->length = 0;
    obj->bytes = nullptr;
    Tcl_SetObjResult(interp, obj);

    return TCL_OK;
  }

  MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Combinational, comb,
                                        circt::comb::CombDialect)
  MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HW, hw, circt::hw::HWDialect)
  MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SV, sv, circt::sv::SVDialect)

  extern "C" {

  int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
      return TCL_ERROR;
    }

    // Initialize CIRCT

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
    Tcl_CreateObjCommand(interp, "loadCirctFile", loadFirMlirFile, NULL, NULL);
    Tcl_CreateObjCommand(interp, "dump", dumpMLIR, NULL, NULL);
    return TCL_OK;
  }
  }

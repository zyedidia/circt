#include "Circt.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

/// Return an error string to tcl.
static int returnErrorStr(Tcl_Interp *interp, const char *error) {
  Tcl_SetObjResult(interp, Tcl_NewStringObj(error, -1));
  return TCL_ERROR;
}

/// Return the circt object as a string.  We don't keep circt objects as strings
/// normally as that would be expensive to maintain, thus there is an explicit
/// conversion to string.
static int circtCmdStringify(mlir::MLIRContext *context, Tcl_Interp *interp,
                         int objc, Tcl_Obj *const objv[]) {
  if (objc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, "circt print  object");
    return TCL_ERROR;
  }

  auto *opholder = objv[2];
  if (Tcl_ConvertToType(interp, opholder, Tcl_GetObjType("MlirOperation")) !=
      TCL_OK)
    return returnErrorStr(interp, "Not an MlirOperation");
  
  auto *op = unwrap((MlirOperation){opholder->internalRep.otherValuePtr});
  std::string str;
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  Tcl_Obj* strobj = Tcl_NewStringObj(str.c_str(), -1);
  Tcl_SetObjResult(interp, strobj);
   return TCL_OK;
}

/// Load an ir file into an object.
static int circtCmdLoad(mlir::MLIRContext *context, Tcl_Interp *interp,
                        int objc, Tcl_Obj *const objv[]) {
  if (objc != 4) {
    Tcl_WrongNumArgs(interp, 1, objv, "circt load [mlir|fir] filename");
    return TCL_ERROR;
  }

  std::string errorMessage;
  int fileNameLen = 0;
  char *fileName = Tcl_GetStringFromObj(objv[3], &fileNameLen);
  auto input = mlir::openInputFile(fileName, &errorMessage);

  if (!input)
    return returnErrorStr(interp, "Cannot open input");

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  mlir::OwningModuleRef module;
  int fileTypeLen = 0;
  char *fileType = Tcl_GetStringFromObj(objv[2], &fileTypeLen);
  if (!strcmp(fileType, "MLIR")) {
    module = mlir::parseSourceFile(sourceMgr, context);
  } else if (!strcmp(fileType, "FIR")) {
    // TODO
    return returnErrorStr(interp, "Can't handle FIR type");
  } else {
    return returnErrorStr(interp, "Unknown file type");
  }

  if (!module)
    return returnErrorStr(interp, "Failed to load module");

  MlirOperation m = wrap(module.release().getOperation());

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirOperation");
  obj->internalRep.otherValuePtr = (void *)m.ptr;
  obj->length = 0;
  obj->bytes = nullptr;
  Tcl_SetObjResult(interp, obj);

  return TCL_OK;
}

/// circtCmd implements the circt command which has these subcommands:
/// * load
/// * print
extern "C" int circtCmd(ClientData cdata, Tcl_Interp *interp, int objc,
                        Tcl_Obj *const objv[]) {
  mlir::MLIRContext *context = (mlir::MLIRContext *)cdata;
  if (objc < 2) {
    Tcl_WrongNumArgs(interp, 1, objv, "circt [print|load]");
    return TCL_ERROR;
  }

  int cmdLen = 0;
  char *cmd = Tcl_GetStringFromObj(objv[1], &cmdLen);
  if (!strcmp(cmd, "load")) {
    return circtCmdLoad(context, interp, objc, objv);
  } else if (!strcmp(cmd, "stringify")) {
    return circtCmdStringify(context, interp, objc, objv);
  }
  return returnErrorStr(interp, "Uknown subcommand");
}

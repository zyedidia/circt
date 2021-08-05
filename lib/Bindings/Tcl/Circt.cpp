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

int returnErrorStr(Tcl_Interp *interp, const char *error) {
  Tcl_SetObjResult(interp, Tcl_NewStringObj(error, -1));
  return TCL_ERROR;
}

int operationTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void operationTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto *op = unwrap((MlirOperation){obj->internalRep.otherValuePtr});
  llvm::raw_string_ostream stream(str);
  op->print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

void operationTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = unwrap((MlirOperation){src->internalRep.otherValuePtr})->clone();
  dup->internalRep.otherValuePtr = wrap(op).ptr;
}

void operationTypeFreeIntRepProc(Tcl_Obj *obj) {
  auto *op = unwrap((MlirOperation){obj->internalRep.otherValuePtr});
  op->erase();
}

int moduleTypeSetFromAnyProc(Tcl_Interp *interp, Tcl_Obj *obj) {
  return TCL_ERROR;
}

void moduleTypeUpdateStringProc(Tcl_Obj *obj) {
  std::string str;
  auto op = unwrap((MlirModule){obj->internalRep.otherValuePtr});
  llvm::raw_string_ostream stream(str);
  op.print(stream);
  obj->length = str.length();
  obj->bytes = Tcl_Alloc(obj->length);
  memcpy(obj->bytes, str.c_str(), obj->length);
  obj->bytes[obj->length] = '\0';
}

void moduleTypeDupIntRepProc(Tcl_Obj *src, Tcl_Obj *dup) {
  auto *op = unwrap((MlirModule){src->internalRep.otherValuePtr})->clone();
  dup->internalRep.otherValuePtr = wrap(op).ptr;
}

void moduleTypeFreeIntRepProc(Tcl_Obj *obj) {
  auto op = unwrap((MlirModule){obj->internalRep.otherValuePtr});
  op.erase();
}
static int circtCmdPrint(mlir::MLIRContext *context, Tcl_Interp *interp,
                         int objc, Tcl_Obj *const objv[]) {
  return TCL_ERROR;
}

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

  MlirModule m = wrap(module.release());

  auto *obj = Tcl_NewObj();
  obj->typePtr = Tcl_GetObjType("MlirModule");
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
  } else if (!strcmp(cmd, "print")) {
    return circtCmdPrint(context, interp, objc, objv);
  }
  return returnErrorStr(interp, "Uknown subcommand");
}

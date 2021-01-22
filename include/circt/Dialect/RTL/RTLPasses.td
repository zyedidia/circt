//===-- RTLPasses.td - RTL pass definition file ------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLPASSES_TD
#define CIRCT_DIALECT_RTL_RTLPASSES_TD

include "mlir/Pass/PassBase.td"

def BlackboxCallout: Pass<"blackbox-callout", "rtl::RTLExternModuleOp"> {
  let summary = "Call external program to deal with blackboxes in RTL.";
  let constructor = "circt::rtl::createBlackboxCalloutPass()";

  let options = [
      Option<"genString", "bbc-generator", "std::string", "\".*\"", "Generator name pattern for blackbox callout">,
      Option<"execString", "bbc-executable", "std::string", "", "Program to call for blackbox callout">,
      Option<"useJSON", "bbc-json", "bool", "false", "Pass JSON with blackbox callout">
  ];
}

#endif // CIRCT_DIALECT_RTL_RTLPASSES_TD

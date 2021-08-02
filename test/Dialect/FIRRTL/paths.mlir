// RUN: circt-opt  %s | FileCheck %s

firrtl.circuit "TopLevel" {

// This represents the non-local annotation path "TopLevel|TopLevel/inst1:Simple>x"

// CHECK-LABEL: firrtl.module @Simple
  firrtl.module @Simple(in %clk: !firrtl.clock, in %pathpass: !firrtl.path) {
    %x = firrtl.reg %clk paths %pathpass {name = "x"} : !firrtl.bundle<a: uint<1>>
    %y = firrtl.reg %clk {name = "y"} : !firrtl.bundle<a: uint<1>>
  }

// CHECK-LABEL: firrtl.module @TopLevel
  firrtl.module @TopLevel(in %clk: !firrtl.clock) {
      %0 = firrtl.path.anchor {annotations = [{a = "a"}]}
      %1 = firrtl.invalidvalue : !firrtl.path
      %clk1, %path1 = firrtl.instance @Simple {name = "inst1"} : !firrtl.clock, !firrtl.path
      %clk2, %path2 = firrtl.instance @Simple {name = "inst2"} : !firrtl.clock, !firrtl.path
      firrtl.connect %clk1, %clk : !firrtl.clock, !firrtl.clock
      firrtl.connect %clk2, %clk : !firrtl.clock, !firrtl.clock
      firrtl.connect %path1, %0 : !firrtl.path, !firrtl.path
      firrtl.connect %path2, %1 : !firrtl.path, !firrtl.path
  }

}


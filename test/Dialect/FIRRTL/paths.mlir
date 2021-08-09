// RUN: circt-opt  %s | FileCheck %s

firrtl.circuit "TopLevel" {

// This represents the non-local annotation path "TopLevel|TopLevel/inst1:Simple>x"

firrtl.extmodule @ext()

// CHECK-LABEL: firrtl.module @Simple
  firrtl.module @Simple(in %clk: !firrtl.clock, in %reset : !firrtl.reset, in %val : !firrtl.uint<4>, in %pathpass: !firrtl.path) {
    %x1 = firrtl.reg %clk paths %pathpass {name = "x1"} : !firrtl.uint<4>
    %x2 = firrtl.reg %clk {name = "x2"} : !firrtl.uint<4>

    %y1 = firrtl.regreset %clk, %reset, %val paths %pathpass {name = "y1"} : !firrtl.reset, !firrtl.uint<4>, !firrtl.uint<4>
    %y2 = firrtl.regreset %clk, %reset, %val {name = "y2"} : !firrtl.reset, !firrtl.uint<4>, !firrtl.uint<4>

    %z1 = firrtl.wire paths %pathpass {name = "z1"} : !firrtl.uint<4>
    %z2 = firrtl.wire {name = "z2"} : !firrtl.uint<4>

    %a1 = firrtl.node %z1 paths %pathpass {name = "a1"} : !firrtl.uint<4>
    %a2 = firrtl.node %z2 {name = "a2"} : !firrtl.uint<4>

    firrtl.instance @ext paths %pathpass {name = "i1"}
    firrtl.instance @ext {name = "i2"}

    %m1 = firrtl.mem paths %pathpass Undefined {depth = 1 : i64, name = "m1", portNames = ["write0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>
    %m2 = firrtl.mem Undefined {depth = 1 : i64, name = "m2", portNames = ["write0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: sint<8>, mask: uint<1>>

  }

// CHECK-LABEL: firrtl.module @TopLevel
  firrtl.module @TopLevel(in %clk: !firrtl.clock, in %reset : !firrtl.reset, in %val : !firrtl.uint<4>) {
      %0 = firrtl.path.anchor {annotations = [{a = "a"}]}
      %1 = firrtl.path.anchor @pathname {annotations = [{a = "a"}]}
      %2 = firrtl.invalidvalue : !firrtl.path
      %clk1, %res1, %v1, %path1 = firrtl.instance @Simple {name = "inst1"} : !firrtl.clock, !firrtl.reset, !firrtl.uint<4>, !firrtl.path
      %clk2, %res2, %v2, %path2 = firrtl.instance @Simple {name = "inst2"} : !firrtl.clock, !firrtl.reset, !firrtl.uint<4>, !firrtl.path
      %clk3, %res3, %v3, %path3 = firrtl.instance @Simple {name = "inst3"} : !firrtl.clock, !firrtl.reset, !firrtl.uint<4>, !firrtl.path
      firrtl.connect %clk1, %clk : !firrtl.clock, !firrtl.clock
      firrtl.connect %clk2, %clk : !firrtl.clock, !firrtl.clock
      firrtl.connect %clk3, %clk : !firrtl.clock, !firrtl.clock
      firrtl.connect %res1, %reset : !firrtl.reset, !firrtl.reset
      firrtl.connect %res2, %reset : !firrtl.reset, !firrtl.reset
      firrtl.connect %res3, %reset : !firrtl.reset, !firrtl.reset
      firrtl.connect %v1, %val : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %v2, %val : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %v3, %val : !firrtl.uint<4>, !firrtl.uint<4>
      firrtl.connect %path1, %0 : !firrtl.path, !firrtl.path
      firrtl.connect %path2, %1 : !firrtl.path, !firrtl.path
      firrtl.connect %path3, %1 : !firrtl.path, !firrtl.path
  }

}


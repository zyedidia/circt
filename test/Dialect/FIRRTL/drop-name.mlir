// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-drop-name))' %s | FileCheck %s --check-prefixes=CHECK,DEAD
// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-drop-name{drop-only-dead-names=false}))' %s | FileCheck %s --check-prefixes=CHECK,ALL

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK-NEXT:  %a = firrtl.wire droppable_name  : !firrtl.uint<1>
    // CHECK-NEXT:  %b = firrtl.reg droppable_name %clock  : !firrtl.uint<1>
    // CHECK-NEXT:  %c = firrtl.regreset droppable_name %clock, %a, %b  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:  %d = firrtl.node droppable_name %c  : !firrtl.uint<1>
    // DEAD-NEXT:  %e = firrtl.node %reset : !firrtl.uint<1>
    // ALL-NEXT:   %e = firrtl.node droppable_name %reset : !firrtl.uint<1>

    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.reg %clock : !firrtl.uint<1>
    %c = firrtl.regreset %clock, %a, %b : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %d = firrtl.node %c : !firrtl.uint<1>
    %e = firrtl.node %reset : !firrtl.uint<1>
    firrtl.strictconnect %out, %e : !firrtl.uint<1>
  }
}

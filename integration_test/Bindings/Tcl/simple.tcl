load libcirct-tcl.so Circt

set circuit [circt load "MLIR" "../../../test/Dialect/SV/basic.mlir"];
puts [circt stringify $circuit]
puts $circuit

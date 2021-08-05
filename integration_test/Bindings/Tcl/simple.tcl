load libcirct-tcl.so Circt

set circuit [circt load "MLIR" "/home/andrew/src/circt/out.mlir"];
puts [circt stringify $circuit]


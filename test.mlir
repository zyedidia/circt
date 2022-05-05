hw.module(%in: i3) {
  %bundleOut_0_d_bits_opcode = sv.wire sym @__SiFive_TLFIFOFixer_8__bundleOut_0_d_bits_opcode  : !hw.inout<i3>
  sv.assign %bundleOut_0_d_bits_opcode, %in : i32
}

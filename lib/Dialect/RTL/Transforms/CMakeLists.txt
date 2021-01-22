file(GLOB globbed *.cpp)

add_circt_dialect_library(CIRCTRTLTransforms
  ${globbed}

  DEPENDS
  MLIRRTLPassIncGen

  LINK_LIBS PUBLIC
  CIRCTRTL
  MLIREDSC
  MLIRIR
  MLIRStandard
  MLIRTransformUtils
)

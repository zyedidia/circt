add_circt_dialect_library(
  CIRCTDomain
  DomainDialect.cpp
  DomainOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/HW

  DEPENDS
  MLIRDomainIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  MLIRIR
)

// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: domain.container "ClockDomain"(id = "clock_domain_1")
domain.container "ClockDomain"(id = "clock_domain_1") {
  // CHECK: domain.property "frequency", 100 : i64
  domain.property "frequency", 100 : i64
}

// CHECK: domain.container "ClockDomain"(id = "clock_domain_2")
domain.container "ClockDomain"(id = "clock_domain_2") {
  // CHECK: domain.property "frequency", 200 : i64
  domain.property "frequency", 200 : i64
}

// CHECK: domain.container "ClockDomains"(id = "clock_domains")
domain.container "ClockDomains"(id = "clock_domains") {
  // CHECK: domain.reference "clock_domain_1"
  domain.reference "clock_domain_1"
  // CHECK: domain.reference "clock_domain_2"
  domain.reference "clock_domain_2"
}

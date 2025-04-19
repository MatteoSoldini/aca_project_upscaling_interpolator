//===- vector_scaler_mul.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>

extern "C" {

void vector_scalar_mul_aie_scalar(int32_t *a, int32_t *c, int32_t *factor,
                                  int32_t N) {
  event0();
  /*
  The size of the vector depends on the type. For example, the standard vector register in AIE2 is 512 bits.
  For int16_t, that means we can store 32 of them in 1x 512b vector register.
  Extending this to the other supported data types, we have the following abbreviated table:
  Data type	Vector size
    int32_t   	16
    int16_t   	32
    int8_t   	  64
    int4_t   	  128

  Note that if the listed data types * vector size ends up being larger than 512-bits,
  that just means it's stored in 2+ vector registers instead of just one.
  */

  constexpr int vec_factor = 16; // try to increase this value and check performance :D - my best was 64
  int32_t *__restrict pA1 = a;
  int32_t *__restrict pC1 = c;
  const int F = N / vec_factor;
  int32_t fac = *factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, )
  {
      aie::vector<int32_t, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::accum<acc64, vec_factor> cout = aie::mul(A0, fac);
      aie::store_v(pC1, cout.template to_vector<int32_t>(0));
      pC1 += vec_factor;
  }
  event1();
}
} // extern "C"

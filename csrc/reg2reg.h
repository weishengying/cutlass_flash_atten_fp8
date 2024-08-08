#pragma once

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include <cute/tensor.hpp>

// Reshape Utility for converting the layout from accumulator of GEMM-I
// to Operand A of GEMM-II.
struct ReshapeTStoTP {
  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &&tC, FragmentQ &&tQ) {

    // get the layout of one row of Q.
    auto layoutQRow = make_ordered_layout(tQ(_, 0, _).layout());
    // get the layout of  M dimension of C.
    auto layoutCM = get<1>(tC.layout());
    return make_layout(get<0>(layoutQRow), layoutCM, get<1>(layoutQRow));
  }
};

// Need this register byte permute/shuffle to match register layout of
// (FP8 downcasted) accumulator of GEMM-I to FP8 operand A of GEMM-II.
struct ReorgCFp8toAFp8{
  int selectorEx0;
  int selectorEx1;  
  int selectorEx4;
  int selectorEx5;
  int upper_map[4] = {0,3,1,2};
  int lower_map[4] = {1,2,0,3};
  
  
CUTLASS_DEVICE ReorgCFp8toAFp8() {
  int laneId = cutlass::canonical_lane_idx();
  
   if (laneId % 4 == 0 || laneId % 4 == 3) {
     selectorEx0 = 0x3210;
     selectorEx1 = 0x7654;
     selectorEx4 = 0x5410;
	   selectorEx5 = 0x7632;
   } else {
     selectorEx0 = 0x7654;
     selectorEx1 = 0x3210;
     selectorEx4 = 0x1054;
	   selectorEx5 = 0x3276;
   }  
   
}

template <typename Fragment>
CUTLASS_DEVICE auto operator()(Fragment &accum) {

  using namespace cute;  
  //   ((_2,_2),_1,_8):((_1,_2),_0,_4)
  // 0,1 | 4,5
  // 2,3 | 6,7

  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.

  auto data = accum.data();
  int n = 0;

#pragma unroll
  for (int i = 0; i < MT; ++i) {

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    // for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {
    for (int k = 0; k < NT / 2; ++k) {

      auto upper = *reinterpret_cast<uint32_t*>(&data[n]); //前4个value FP8,正好组成一个int32
      auto lower = *reinterpret_cast<uint32_t*>(&data[n+4]); //后四个value
      
      auto upper0 = __byte_perm(upper, lower, selectorEx0);
      auto lower0 = __byte_perm(upper, lower, selectorEx1);      
      upper0 = __shfl_sync(uint32_t(-1),upper0, upper_map[threadIdx.x%4],4);
      lower0 = __shfl_sync(uint32_t(-1),lower0, lower_map[threadIdx.x%4],4);
  
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
      data_32bit[0] = __byte_perm(upper0, lower0, selectorEx4);
      data_32bit[1] = __byte_perm(upper0, lower0, selectorEx5);
      n += 8;
    }
  }
}
};

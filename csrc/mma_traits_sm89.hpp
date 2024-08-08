#pragma once

#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

#include <cute/numeric/integer_subbyte.hpp>

#include <cutlass/numeric_types.h>

namespace cute
{

namespace {

// (T32,V1) -> (M8,N8)
using SM80_8x4      = Layout<Shape <Shape < _4,_8>,_1>,
                             Stride<Stride< _8,_1>,_0>>;
// (T32,V2) -> (M8,N8)
using SM80_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
                             Stride<Stride<_16,_1>,_8>>;
// (T32,V4) -> (M8,N16)
using SM80_8x16_Row = Layout<Shape <Shape < _4,_8>,_4>,
                             Stride<Stride<_32,_1>,_8>>;
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_E4M3_TN>
{
     using ValTypeD = float;
     using ValTypeA = cutlass::float_e4m3_t;
     using ValTypeB = cutlass::float_e4m3_t;
     using ValTypeC = float;

     using Shape_MNK = Shape<_16,_8,_32>;
     using ThrID   = Layout<_32>;
     using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
     Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
     using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
     Stride<Stride<_32,_1>, Stride<_8,_128>>>;
     using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32F8F8F32_E5M2_TN>
{
    using ValTypeD = float;
    using ValTypeA = cutlass::float_e5m2_t;
    using ValTypeB = cutlass::float_e5m2_t;
    using ValTypeC = float;

    using Shape_MNK = Shape<_16,_8,_32>;
    using ThrID   = Layout<_32>;
    using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
    Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
    using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
    Stride<Stride<_32,_1>, Stride<_8,_128>>>;
    using CLayout = SM80_16x8_Row;
};

}
#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

#include "mma_sm89.hpp"
#include "mma_traits_sm89.hpp"
using namespace cute;


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type>
struct Flash_kernel_traits {

    using Element = elem_type;
    static constexpr bool Has_cp_async = true;

    using ElementAccum = float;
    using index_t = uint32_t;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e5m2_t>,
        MMA_Atom<SM89_16x8x32_F32F8F8F32_E5M2_TN>,
        MMA_Atom<SM89_16x8x32_F32F8F8F32_E4M3_TN>
    >;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;

    // TODO: review
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _32>>;
        
    // using TiledMma = TiledMMA< //(16*kNWarps, 16, 32)
    //     typename Base::MMA_Atom_Arch,
    //     Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
    //     Tile<Int<16 * kNWarps>,
    //         Layout<Shape <_2,_4,_2>, 
    //                 Stride<_1,_4,_2>>, // Permutation on N, size 16                     
    //         _32>>;


    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_16, Int<kBlockKSmem>>, // (16, 64)
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));


    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    // output 从 regis copy --> SMEM
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;

    static constexpr int kSmemQCount = size(SmemLayoutQ{});
    static constexpr int kSmemKVCount = size(SmemLayoutK{}) + size(SmemLayoutV{});
    static constexpr int kSmemQSize = kSmemQCount * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(Element);
    // TODO:
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");


    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 16 vals per read
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 16 vals per store
};

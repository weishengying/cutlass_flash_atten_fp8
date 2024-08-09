#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

#include "static_switch.h"
#include "kernel_traits.h"
#include "flash.h"
#include "utils.h"
#include "reg2reg.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // NOTE: 根据 mma_tile 的示意图来确定每个线程处理的是第几个 token

    // NOTE:
    // 计算thread的处理范围, mask掉超出范围的部分

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2; //根据 mma 指令集的特性，每行 8 个元素由 4 个线程处理

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
    // (2, nrow), 2*8 for each
    const int group_stride = kNWarps * 16;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        // 根据定义的 mma 指令, 一行4个线程处理 8 个value
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            // j用于计算value 1和value 2对应col
            // col_idx最终表示当前thread所处理的value的列号
            const int col_idx = col_idx_base + j;

            // mask掉scores中(QK后的结果)超出范围的部分
            // 列号和行号对比

            // Without the "make_coord" we get wrong results
            // for nrow(2, MMA_M)
            #pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

              #pragma unroll
              for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                if (col_idx > row_idx) {
                  tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                }
              }

            }

        }
    }
}

// NOTE: A矩阵已经在寄存器中的gemm封装
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    // NOTE: 符合M N K描述: A[M, K] @ B[N, K] = C[M, N]
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    // NOTE: retile 成拷贝需要的大小
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// copy from S to D with tiled_copy
// TODO: 需要支持causal模式的的跳过拷贝
template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // TODO: 原版处这里identity_MN是用来跳过大块的block的, predicate用于跳过block内的拷贝
        // TODO: 添加predicate逻辑, 用于跳过无用拷贝
        // if (get<0>(identity_MN(0, m, 0)) < max_MN)
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        }
    }
}


template <typename ToType, typename Fragment>
inline __device__ auto convert_float32_to_fp8(Fragment const &acc_fp32) {
  Tensor acc_fp8 = make_tensor<ToType>(shape(acc_fp32));
  using convert_type = std::conditional_t<
                            std::is_same_v<ToType, cutlass::float_e5m2_t>,
                            __nv_fp8x2_e5m2,
                            __nv_fp8x2_e4m3
                        >;
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp8x2 = recast<convert_type>(acc_fp8);
    for (int i = 0; i < size(acc_fp32x2); ++i) { 
      acc_fp8x2(i) = convert_type(acc_fp32x2(i)); 
    }
  }
  return acc_fp8;
}


// TODO:
// https://github.com/NVIDIA/cutlass/issues/802
// TODO: convert出来后数据是否在寄存器?
// template <typename Fragment>
// inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
//   Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
//   {
//     Tensor acc_fp32x2 = recast< float2>(acc_fp32);
//     Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
//     for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
//   }
//   return acc_fp16;
// }

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
        }
    }
}



// Convert acc_layout from (MMA=(2,2), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

// scores:((2, MMA_M),(2, MMA_N))，经过了 causal 之后的 Q_i 和 k_j^T 的乘积，
// scores_max:(2 * MMA_N), rowmax 的结果
// scores_sum:(2 * MMA_N)， rowsum 的结果
// acc_o:((2, 2),(MMA_M, MMA_N))， 最后的计算结果
template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    if (Is_first) {
        // NOTE: 第一次softmax不需要rescale, 只需要记录 Sij(kblockM, kblockN) 的 rowmax 和 rowsum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // 记录上一次的 rowmax
        Tensor scores_max_prev = make_fragment_like(scores_max); // 相当于公式中的 m_i^{j-1}
        cute::copy(scores_max, scores_max_prev);
        // NOTE: 计算最新的 max 
        // reduce_max包含步:
        //  1. 求当前thread内max: 遍历
        //  2. reduce thread间的max: 使用线程数洗牌指令做 all reduce，每个线程都获得了最大值
        reduce_max</*zero_init=*/false>(scores, scores_max); // scores_max 变成最新的最大值，相当于公式中的 m_i^{j}
        // Reshape acc_o from ((2,2), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) { // 遍历每一行
            // NOTE: 辅助变量: 当前行max
            float scores_max_cur = scores_max(mi); // 当前行的最大值
            // NOTE: 计算上一次 score_sum 的 rescale 值
            float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2); // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}.
            scores_sum(mi) *= scores_scale; // 想当于公式中的  e^{m_i^{j-1} - m_i^{j}}l_i^{j-1}
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; } // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        }
        // NOTE: Apply the exp to all the elements with new max value， 这里相当于论文公式里的 P_i^_j
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        Tensor scores_sum_cur = make_fragment_like(scores_sum);  // l_i^{j} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        // NOTE: 累计求和
        reduce_sum(scores, scores_sum_cur); // rowsum(P_i^_j)
        // NOTE: 新分母累加到旧分母
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); } // l{ij} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1} + rowsum(P_i^_j)
    }
};

} // namespace flash

void set_params_fprop(Flash_fwd_params &params,

                      // device pointers
                      const torch::Tensor q,
                      const torch::Tensor k,
                      const torch::Tensor v,
                      torch::Tensor out,

                      void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal) {

  memset(&params, 0, sizeof(params));

  params.bs = q.size(0);
  params.head = q.size(1);
  params.q_seqlen = q.size(2);
  params.dim = q.size(3);

  params.k_head = k.size(1);
  params.k_seqlen = k.size(2);

  params.bs_stride = q.stride(0);
  params.head_stride = q.stride(1);
  params.seqlen_stride = q.stride(2);
  params.dim_stride = q.stride(3);

  params.softmax_scale = softmax_scale;
  // TODO: 使用log2做scale
  params.softmax_scale_log2 = softmax_scale * M_LOG2E;
  params.is_causal = is_causal;
  params.is_fp8_e5m2 = q.dtype() == torch::kFloat8_e5m2;

  // LogSumExp save for backward
  params.softmax_lse_ptr = softmax_lse_d;

  // TODO: get ptr
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.out_ptr = out.data_ptr();
}


// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  // TODO: Aligned的话smem的计算是否有问题
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, bool Is_causal=false, typename Params>
__global__ void flash_attention_v2_cutlass_kernel(const Params params) {

  using namespace cute;

  // m block index
  const int m_block = blockIdx.x;

  // bs * head
  const int base_id = blockIdx.y;
  // The thread index.
  const int tidx = threadIdx.x;

  // TODO: 传入泛型
  // NOTE: 小技巧
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using index_t = typename Kernel_traits::index_t;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;


  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  // Shared memory.
  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  const int bs_head_offset = base_id * params.head_stride;

  // TODO: base offset for MHA
  // NOTE: convert C pointer to Tensor for convenience
  Tensor Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
      make_shape(params.q_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor K = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset),
      make_shape(params.k_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor V = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset),
      make_shape(Int<kHeadDim>{}, params.k_seqlen),
      make_stride(params.k_seqlen, Int<1>{}));
  Tensor O = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset),
      make_shape(params.q_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));

  
  // 加载Q, K, V分块
  // (kBlockM, kHeadDim, num_tile_n)
  Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
  // NOTE: loading流水线, 初次加载所需K, V
  Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_, 0)); // 这里注意 V，因为已经转置了

  // 获取MMA抽象
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{}); //(kBlockM, kHeadDim)
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{}); //(kBlockN, kHeadDim)
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{}); //(kHeadDim, kBlockN)

  // NOTE: copy抽象
  // NOTE: QKV gmem -> smem拷贝的抽象
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  // NOTE: 定义gmem -> smem拷贝的src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


  // NOTE: 定义smem -> reg拷贝的dst
  // partition_fragment与partition类似, 只是返回的是寄存器表示
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K) =  ((4, 2, 2), MMA_M, MMA_N)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K) =  ((4, 2), MMA_N, MMA_K)
  Tensor tOrV  = thr_mma.partition_fragment_B(sV);                           // (MMA,MMA_K,MMA_N) =  ((2, 2), MMA_M, MMA_N)
 
  // NOTE: 准备拷贝Q, K, V到 reg 的copy对象 (smem --> reg)
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ); // tSsQ --> tSrQ

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK); // tSrK --> tSsK
  

  // TODO: 拷贝时转置
  // NOTE: smem->reg拷贝Vt
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsV = smem_thr_copy_V.partition_S(sV);
  // NOTE: 命名规则, t表示to, s/g表示位置(smem, gmem)
  // 从smem加载时做retiling
  // tKgK表示gmem中的K, 用作gmem->smem的src
  // tKsK表示smem中的K, 用作gmem->smem的dst
  // tSsK表示smem中的K, 用作smem->reg的src

  // 流水线加载初始Q, K
  // 加载Q到smem
  flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // 加载K到smem
  flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  // 开始执行异步拷贝
  cute::cp_async_fence();

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{}); // （MMA, MMA_M, MMA_K)

  // NOTE: K, V分块的数量: 处理的区间
  const int n_block_min = 0;
  // NOTE: 1. mask between N BLOCKs if is causal mode
  int seqlen_start = m_block * kBlockM;
  int seqlen_end = (m_block + 1) * kBlockM;
  int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN); 

  // NOTE: 需要记录的max
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{}); // （2 * MMA_M）

  // NOTE: 需要记录的 softmax 分母
  Tensor scores_sum = make_fragment_like(scores_max); // （2 * MMA_M）

  clear(rAccOut); // （MMA=(2,2), MMA_M, MMA_K), 初始化为 0 

  for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
    auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); //(MMA=(2,2), MMA_M, MMA_N)
    clear(rAccScore); // 初始化为 0

    // 等待Q, K的gmem -> smem拷贝完成, 即Q, K就绪
    // wait<0>表示等待还剩0个未完成
    flash::cp_async_wait<0>();
    __syncthreads();

    // gemm的同时异步加载V
    gV = local_tile(V, make_tile(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_,  nbi));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    // 异步加载V到smem
    flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // 发起异步拷贝
    cute::cp_async_fence();

    // O = Q@K.T
    // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );

    // NOTE: Convert from layout C to layout A;  (MMA=(2, 2),(MMA_M, MMA_N)) --> ((2, MMA_M),(2, MMA_N))
    Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. mask within N BLOCKs
    if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
    }
  
    // NOTE: 等待V加载完成, 为下个K加载准备初始状态
    flash::cp_async_wait<0>();
    __syncthreads();
    
    // advance K
    if (nbi != n_block_max - 1) {
      gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      cute::cp_async_fence();
    }
    
    // 计算softmax
    // scores:((2, MMA_M),(2, MMA_N)), Q_i * K_j^T 的值
    // scores_max:(2 * MMA_N)
    // scores_sum:(2 * MMA_N)
    // rAccOut:((2, 2),(MMA_M, MMA_N))，相当于 O_i
    nbi == 0 ? flash::softmax_rescale_o<true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
      flash::softmax_rescale_o<false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);


    // 计算完成后， scores 相当于公式中的 P_i^j
    // 实际执行 P_i^j @ V
    // (score AKA rAccScore): exp(QK[M, N] - m_i^j) @ V[N, dim]
    // NOTE: DABC: F32F8F8F32, convert D type(F32) to A type(F8)
    Tensor rP = flash::convert_float32_to_fp8<Element>(rAccScore);
    auto reg2reg = ReorgCFp8toAFp8();
    reg2reg(rP);
    // NOTE: Convert from layout C to layout A;  (MMA=(2, 2), MMA_M, MMA_N)) --> (MMA=(4, 2, 2), MMA_M, MMA_N))
    // 这里由于写死了 kBlockM, KBlockN 都为64，所以第一个 gemm 的输出总是为(64，64), 因此根据 MMA 指令集的特性，可以容易算出 MMA_M = 1, MMA_N = 2
    //（我暂时不知道怎么写 layout 除法（针对静态shape），所以目前固定写为下面这种形式）
    auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, _1, _2>>{};
    Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
    // rAccOut:((2, 2),(MMA_M, MMA_N))
    flash::gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  } 
  
  
  // Epilogue

  // NOTE: 最后统一除上分母部分
  // Reshape acc_o from ((2,2), 1, 8) to (nrow=(2, 1), ncol=(2, 8))
  // AKA reshape to (nrow, ncol) but with specific MMA layout
  Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
  // for row
  #pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    float scale = inv_sum;
    // for col
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // Convert acc_o from fp32 to fp8
  Tensor rO = flash::convert_float32_to_fp8<Element>(rAccOut);
  // 复用sQ的smem做sO的拷出
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)

  // Partition sO to match the accumulator partitioning
  // TODO: review
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // NOTE: 先拷贝到smem
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // 创建到smem -> gmem的拷贝
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();

  // NOTE:: 再拷贝到gmem

  // TODO: review, 这里两个copy的作用
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  flash::copy(gmem_tiled_copy_O, tOrO, tOgO);
  
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  // TODO: check if works: default stream = 0
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;

  const int num_m_block =
      (params.q_seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

  dim3 grid(num_m_block, params.bs * params.head, 1);
  dim3 block(Kernel_traits::kNThreads);

  int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_v2_cutlass_kernel<Kernel_traits, Is_causal, Flash_fwd_params>;
  // NOTE: smem过大时需要设置
  if (smem_size >= 48 * 1024) {
      CUDA_ERROR_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  // TODO: stream
  kernel<<<grid, block, smem_size>>>(params);
}

template<typename T, int Headdim>
void run_flash_fwd_(Flash_fwd_params &params, cudaStream_t stream);

// TODO: 挨个写出特化, 目前使用通用模板
// 如, run_flash_fwd_hdim32用于特化hdim=32
// 这样做可以根据实际情况微调kBlockN和kBlockM的组合, 也可以加速编译
template<typename T, int Headdim>
void run_flash_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/128, /*kBlockN_=*/128, /*kNWarps_=*/4, T>, Is_causal>(params, stream);

        // TODO: kBlockM, kBlockN的组合
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/64, /*kBlockN_=*/64, /*kNWarps_=*/4, T>, Is_causal>(params, stream);
    });
}

// entry point of flash attention
void run_flash_attn_cutlass(Flash_fwd_params &params, cudaStream_t stream) {
    // FP8_SWITCH yield elem_type namespace
    FP8_SWITCH(params.is_fp8_e5m2, [&] {
        // FWD_HEADDIM_SWITCH yield kHeadDim constexpr
        FWD_HEADDIM_SWITCH(params.dim, [&] {
            run_flash_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, bool is_causal = false, float softmax_scale=1) {

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);
  // dim
  int dim = q.size(3);
  auto opts = q.options();
  // auto out = torch::empty_like(q, opts.dtype(torch::kFloat16));
  auto out = torch::empty_like(q);
  Flash_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      nullptr, softmax_scale, is_causal);

  run_flash_attn_cutlass(params, 0);

  // Wait until kernel finish.
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return {out};
}




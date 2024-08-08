#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
namespace cute {


// 添加FP8相关的SM89指令支持
// MMA 16x8x32 TN
struct SM89_16x8x32_F32F8F8F32_E4M3_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void
    fma(float    & d0, float      & d1, float      & d2, float      & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
        asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));

    }
};

struct SM89_16x8x32_F32F8F8F32_E5M2_TN
{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void
    fma(float    & d0, float      & d1, float      & d2, float      & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
// #if defined(CUTE_ARCH_MMA_SM89_ENABLED)
        asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
        "r"(b0),  "r"(b1),
        "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
// #else
//         CUTE_RUNTIME_ASSERT("Attempting to use SM89_16x8x32_S32S8S8S32_TN without CUTE_ARCH_MMA_SM89_ENABLED");
// #endif
    }
};

}
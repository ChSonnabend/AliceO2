// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionKernelMacros.h
/// \author David Rohr

// clang-format off
#ifndef O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H
#define O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H

#include "GPUDefMacros.h"

#define GPUCA_M_KRNL_TEMPLATE_B(a, b, ...) a, a::b
#define GPUCA_M_KRNL_TEMPLATE_A(...) GPUCA_M_KRNL_TEMPLATE_B(__VA_ARGS__, defaultKernel)
#define GPUCA_M_KRNL_TEMPLATE(...) GPUCA_M_KRNL_TEMPLATE_A(GPUCA_M_STRIP(__VA_ARGS__))

#define GPUCA_M_KRNL_NUM_B(a, b, ...) a::b
#define GPUCA_M_KRNL_NUM_A(...) GPUCA_M_KRNL_NUM_B(__VA_ARGS__, defaultKernel)
#define GPUCA_M_KRNL_NUM(...) GPUCA_M_KRNL_NUM_A(GPUCA_M_STRIP(__VA_ARGS__))

#define GPUCA_M_KRNL_NAME_B0(a, b, ...) GPUCA_M_CAT3(a, _, b)
#define GPUCA_M_KRNL_NAME_B1(a) a
#define GPUCA_M_KRNL_NAME_A(...) GPUCA_M_CAT(GPUCA_M_KRNL_NAME_B, GPUCA_M_SINGLEOPT(__VA_ARGS__))(__VA_ARGS__)
#define GPUCA_M_KRNL_NAME(...) GPUCA_M_KRNL_NAME_A(GPUCA_M_STRIP(__VA_ARGS__))

#if defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_HOSTONLY)
#ifndef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(...)
#endif
#ifndef GPUCA_KRNL_CUSTOM
#define GPUCA_KRNL_CUSTOM(...)
#endif
#define GPUCA_KRNL_REG_EXTRREG(...) GPUCA_M_STRIP(__VA_ARGS__)
#define GPUCA_KRNL_CUSTOM_EXTRREG(MODE, ...) GPUCA_ATTRRES_XCUSTOM(MODE, __VA_ARGS__)
#define GPUCA_KRNL_NONE_EXTRREG(MODE, ...) GPUCA_ATTRRES_XNONE(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_REG(MODE, reg, num, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_REG, MODE))(num) GPUCA_ATTRRES_XREG (MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_CUSTOM(MODE, custom, args, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_KRNL_CUSTOM, MODE))(args) GPUCA_ATTRRES_XCUSTOM(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_NONE(MODE, none, ...) GPUCA_ATTRRES_XNONE(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_(MODE, ...)
#define GPUCA_ATTRRES_XNONE(MODE, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_XCUSTOM(MODE, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES_XREG(MODE, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(MODE, __VA_ARGS__)
#define GPUCA_ATTRRES(MODE, ...) GPUCA_M_EXPAND(GPUCA_M_CAT(GPUCA_ATTRRES_, GPUCA_M_FIRST(__VA_ARGS__)))(MODE, __VA_ARGS__)

// GPU Kernel entry point
#define GPUCA_KRNLGPU_DEF(x_class, x_attributes, x_arguments, ...) \
  GPUg() void GPUCA_ATTRRES(, GPUCA_M_STRIP(x_attributes)) GPUCA_M_CAT(krnl_, GPUCA_M_KRNL_NAME(x_class))(GPUCA_CONSMEM_PTR int32_t _iSector_internal GPUCA_M_STRIP(x_arguments))

#ifdef GPUCA_KRNL_DEFONLY
#define GPUCA_KRNLGPU(...) GPUCA_KRNLGPU_DEF(__VA_ARGS__);
#else
#define GPUCA_KRNLGPU(x_class, x_attributes, x_arguments, x_forward, ...) \
  GPUCA_KRNLGPU_DEF(x_class, x_attributes, x_arguments, x_forward, __VA_ARGS__) \
  { \
    GPUshared() typename GPUCA_M_STRIP_FIRST(x_class)::GPUSharedMemory smem; \
    GPUCA_M_STRIP_FIRST(x_class)::template Thread<GPUCA_M_KRNL_NUM(x_class)>(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, GPUCA_M_STRIP_FIRST(x_class)::Processor(GPUCA_CONSMEM)[_iSector_internal] GPUCA_M_STRIP(x_forward)); \
  }
#endif

// GPU Host wrappers for kernel
#define GPUCA_KRNL_HOST(x_class, ...) \
  GPUCA_KRNLGPU(x_class, __VA_ARGS__) \
  template <> class GPUCA_KRNL_BACKEND_CLASS::backendInternal<GPUCA_M_KRNL_TEMPLATE(x_class)> { \
   public: \
    template <typename T, typename... Args> \
    static inline void runKernelBackendMacro(const krnlSetupTime& _xyz, T* me, const Args&... args) \
    { \
      auto& x = _xyz.x; \
      auto& y = _xyz.y; \
      GPUCA_KRNL_CALL(x_class, __VA_ARGS__) \
    } \
  };

#define GPUCA_KRNL_PROP(x_class, x_attributes) \
  template <> gpu_reconstruction_kernels::krnlProperties GPUCA_KRNL_BACKEND_CLASS::getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>() { \
    gpu_reconstruction_kernels::krnlProperties ret = gpu_reconstruction_kernels::krnlProperties{GPUCA_ATTRRES(_EXTRREG, GPUCA_M_STRIP(x_attributes))}; \
    return ret.nThreads > 0 ? ret : gpu_reconstruction_kernels::krnlProperties{(int32_t)mThreadCount}; \
  }

#endif // GPUCA_GPUCODE

#define GPUCA_KRNL_LB(x_class, x_attributes, ...) GPUCA_KRNL(x_class, (REG, (GPUCA_M_CAT(GPUCA_LB_, GPUCA_M_KRNL_NAME(x_class))), GPUCA_M_STRIP(x_attributes)), __VA_ARGS__)

#endif // O2_GPU_GPURECONSTRUCTIONKERNELMACROS_H
// clang-format on

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

/// \file GPUDefParametersRuntime.h
/// \author David Rohr

#ifndef GPUDEFPARAMETERSRUNTIME_H
#define GPUDEFPARAMETERSRUNTIME_H

namespace o2::gpu
{
struct GPUDefParameters {  // clang-format off
  int32_t par_LB_maxThreads[$<LIST:LENGTH,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>>] = {};
  int32_t par_LB_minBlocks[$<LIST:LENGTH,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>>] = {};
  int32_t par_LB_forceBlocks[$<LIST:LENGTH,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>>] = {};
  $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_PARAMS>,PREPEND,int32_t par_>,APPEND, = 0>,$<SEMICOLON>
  >;
  $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_STRING_PARAMS>,PREPEND,char par_>,APPEND,[128] = "">,$<SEMICOLON>
  >;
};  // clang-format on
}  // namespace o2::gpu

#endif  // GPUDEFPARAMETERSRUNTIME_H

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

/// \file GPUReconstructionProcessingKernels.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONPROCESSINGKERNELS_H
#define GPURECONSTRUCTIONPROCESSINGKERNELS_H

#include "GPUReconstructionProcessing.h"
#include "GPUReconstructionKernelIncludes.h"

namespace o2::gpu
{

#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-lambda-capture" // this is not alway captured below
#endif

template <class T, class S>
void GPUReconstructionProcessing::KernelInterface<T, S>::runKernelVirtual(const int num, const void* args)
{
  switch (num) { // clang-format off
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types, x_num)                             \
    case x_num: {                                                                                             \
      const auto& args2 = *(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>*)args; \
      std::apply([this, &args2](auto&... vals) { ((T*)this)->template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(args2.s, vals...); }, args2.v); \
      break;                                                                                                  \
    }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  } // clang-format on
}

#pragma GCC diagnostic push

} // namespace o2::gpu

#endif // GPURECONSTRUCTIONPROCESSINGKERNELS_H

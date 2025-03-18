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

/// \file GPUCommonHelpers.h
/// \author David Rohr

// GPUChkErr and GPUChkErrI will both check x for an error, using the loaded backend of GPUReconstruction (requiring GPUReconstruction.h to be included by the user).
// In case of an error, it will print out the corresponding CUDA / HIP / OpenCL error code
// GPUChkErr will download GPUReconstruction error values from GPU, print them, and terminate the application with an exception if an error occured.
// GPUChkErrI will return 0 or 1, depending on whether an error has occurred.
// These Macros must be called ona GPUReconstruction instance.
// The GPUChkErrS and GPUChkErrSI are similar but static, without required GPUReconstruction instance.
// Examples:
// if (mRec->GPUChkErrI(cudaMalloc(...))) { exit(1); }
// gpuRecObj.GPUChkErr(cudaMalloc(...));
// if (GPUChkErrSI(cudaMalloc(..))) { exit(1); }

#ifndef GPUCOMMONHELPERS_H
#define GPUCOMMONHELPERS_H

// Please #include "GPUReconstruction.h" in your code, if you use these 2!
#define GPUChkErr(x) GPUChkErrA(x, __FILE__, __LINE__, true)
#define GPUChkErrI(x) GPUChkErrA(x, __FILE__, __LINE__, false)
#define GPUChkErrS(x) o2::gpu::internal::GPUReconstructionChkErr(x, __FILE__, __LINE__, true)
#define GPUChkErrSI(x) o2::gpu::internal::GPUReconstructionChkErr(x, __FILE__, __LINE__, false)

#include "GPUCommonDef.h"
#include "GPUCommonLogger.h"
#include <cstdint>

namespace o2::gpu::internal
{
#define GPUCOMMON_INTERNAL_CAT_A(a, b, c) a##b##c
#define GPUCOMMON_INTERNAL_CAT(...) GPUCOMMON_INTERNAL_CAT_A(__VA_ARGS__)
extern int32_t GPUCOMMON_INTERNAL_CAT(GPUReconstruction, GPUCA_GPUTYPE, ChkErr)(const int64_t error, const char* file, int32_t line);
inline int32_t GPUReconstructionCPUChkErr(const int64_t error, const char* file, int32_t line)
{
  if (error) {
    LOGF(error, "GPUCommon Error Code %ld (%s:%d)", (long)error, file, line);
  }
  return error != 0;
}
static inline int32_t GPUReconstructionChkErr(const int64_t error, const char* file, int32_t line, bool failOnError)
{
  int32_t retVal = error && GPUCOMMON_INTERNAL_CAT(GPUReconstruction, GPUCA_GPUTYPE, ChkErr)(error, file, line);
  if (retVal && failOnError) {
    throw std::runtime_error("GPU API Call Failure");
  }
  return error;
}
#undef GPUCOMMON_INTERNAL_CAT_A
#undef GPUCOMMON_INTERNAL_CAT
} // namespace o2::gpu::internal

#endif

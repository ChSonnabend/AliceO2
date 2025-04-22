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

/// \file GPUReconstructionOCLKernels.cxx
/// \author David Rohr

#include "GPUReconstructionOCLIncludesHost.h"
#include "GPUReconstructionKernelIncludes.h"

#include "GPUReconstructionOCLKernelsSpecialize.inc"
#include "GPUReconstructionProcessingKernels.inc"
template void GPUReconstructionProcessing::KernelInterface<GPUReconstructionOCL, GPUReconstructionDeviceBase>::runKernelVirtual(const int num, const void* args);

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionOCL::runKernelBackend(const krnlSetupTime& _xyz, const Args&... args)
{
  cl_kernel k = getKernelObject<cl_kernel, T, I>();
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  auto& z = _xyz.z;
  GPUChkErr(OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.index, args...));

  cl_event ev;
  cl_event* evr;
  bool tmpEvent = false;
  if (z.ev == nullptr && GetProcessingSettings().deviceTimers && GetProcessingSettings().debugLevel > 0) {
    evr = &ev;
    tmpEvent = true;
  } else {
    evr = (cl_event*)z.ev;
  }
  GPUChkErr(clExecuteKernelA(mInternals->command_queue[x.stream], k, x.nThreads, x.nThreads * x.nBlocks, evr, (cl_event*)z.evList, z.nEvents));
  if (GetProcessingSettings().deviceTimers && GetProcessingSettings().debugLevel > 0) {
    cl_ulong time_start, time_end;
    GPUChkErr(clWaitForEvents(1, evr));
    GPUChkErr(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
    GPUChkErr(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
    _xyz.t = (time_end - time_start) * 1.e-9f;
    if (tmpEvent) {
      GPUChkErr(clReleaseEvent(ev));
    }
  }
}

template <class T, int32_t I>
int32_t GPUReconstructionOCL::AddKernel()
{
  std::string name(GetKernelName<T, I>());
  std::string kname("krnl_" + name);

  cl_int ocl_error;
  cl_kernel krnl = clCreateKernel(mInternals->program, kname.c_str(), &ocl_error);
  if (GPUChkErrI(ocl_error)) {
    GPUError("Error creating OPENCL Kernel: %s", name.c_str());
    return 1;
  }
  mInternals->kernels.emplace_back(krnl);
  return 0;
}

template <class S, class T, int32_t I>
S& GPUReconstructionOCL::getKernelObject()
{
  return mInternals->kernels[GetKernelNum<T, I>()];
}

int32_t GPUReconstructionOCL::AddKernels()
{
#define GPUCA_KRNL(x_class, ...)                     \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>()) { \
    return 1;                                        \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  return 0;
}

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

template <>
inline void GPUReconstructionOCLBackend::runKernelBackendInternal<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  cl_int4 val0 = {0, 0, 0, 0};
  GPUFailedMsg(clEnqueueFillBuffer(mInternals->command_queue[_xyz.x.stream], mInternals->mem_gpu, &val0, sizeof(val0), (char*)ptr - (char*)mDeviceMemoryBase, (size + sizeof(val0) - 1) & ~(sizeof(val0) - 1), _xyz.z.evList == nullptr ? 0 : _xyz.z.nEvents, _xyz.z.evList->getEventList<cl_event>(), _xyz.z.ev->getEventList<cl_event>()));
}

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionOCLBackend::runKernelBackendInternal(const krnlSetupTime& _xyz, const Args&... args)
{
  cl_kernel k = getKernelObject<cl_kernel, T, I>();
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  auto& z = _xyz.z;
  GPUFailedMsg(OCLsetKernelParameters(k, mInternals->mem_gpu, mInternals->mem_constant, y.index, args...));

  cl_event ev;
  cl_event* evr;
  bool tmpEvent = false;
  if (z.ev == nullptr && mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0) {
    evr = &ev;
    tmpEvent = true;
  } else {
    evr = (cl_event*)z.ev;
  }
  GPUFailedMsg(clExecuteKernelA(mInternals->command_queue[x.stream], k, x.nThreads, x.nThreads * x.nBlocks, evr, (cl_event*)z.evList, z.nEvents));
  if (mProcessingSettings.deviceTimers && mProcessingSettings.debugLevel > 0) {
    cl_ulong time_start, time_end;
    GPUFailedMsg(clWaitForEvents(1, evr));
    GPUFailedMsg(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
    GPUFailedMsg(clGetEventProfilingInfo(*evr, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
    _xyz.t = (time_end - time_start) * 1.e-9f;
    if (tmpEvent) {
      GPUFailedMsg(clReleaseEvent(ev));
    }
  }
}

template <class T, int32_t I, typename... Args>
void GPUReconstructionOCLBackend::runKernelBackend(const krnlSetupArgs<T, I, Args...>& args)
{
  std::apply([this, &args](auto&... vals) { runKernelBackendInternal<T, I, Args...>(args.s, vals...); }, args.v);
}

template <class T, int32_t I>
inline uint32_t GPUReconstructionOCLBackend::FindKernel()
{
  std::string name(GetKernelName<T, I>());

  for (uint32_t k = 0; k < mInternals->kernels.size(); k++) {
    if (mInternals->kernels[k].second == name) {
      return (k);
    }
  }
  GPUError("Could not find OpenCL kernel %s", name.c_str());
  throw ::std::runtime_error("Requested unsupported OpenCL kernel");
}

template <class T, int32_t I>
int32_t GPUReconstructionOCLBackend::AddKernel()
{
  std::string name(GetKernelName<T, I>());
  std::string kname("krnl_" + name);

  cl_int ocl_error;
  cl_kernel krnl = clCreateKernel(mInternals->program, kname.c_str(), &ocl_error);
  if (GPUFailedMsgI(ocl_error)) {
    GPUError("Error creating OPENCL Kernel: %s", name.c_str());
    return 1;
  }
  mInternals->kernels.emplace_back(krnl, name);
  return 0;
}

template <class S, class T, int32_t I>
S& GPUReconstructionOCLBackend::getKernelObject()
{
  static uint32_t krnl = FindKernel<T, I>();
  return mInternals->kernels[krnl].first;
}

int32_t GPUReconstructionOCLBackend::AddKernels()
{
#define GPUCA_KRNL(x_class, ...)                     \
  if (AddKernel<GPUCA_M_KRNL_TEMPLATE(x_class)>()) { \
    return 1;                                        \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
  return 0;
}

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types) \
  GPUCA_KRNL_PROP(x_class, x_attributes)                                   \
  template void GPUReconstructionOCLBackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args);
#define GPUCA_KRNL_BACKEND_CLASS GPUReconstructionOCLBackend
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

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

/// \file GPUReconstructionOCL.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONOCL_H
#define GPURECONSTRUCTIONOCL_H

#include "GPUReconstructionDeviceBase.h"

#ifdef _WIN32
extern "C" __declspec(dllexport) o2::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#else
extern "C" o2::gpu::GPUReconstruction* GPUReconstruction_Create_OCL(const o2::gpu::GPUSettingsDeviceBackend& cfg);
#endif

namespace o2::gpu
{
struct GPUReconstructionOCLInternals;

class GPUReconstructionOCLBackend : public GPUReconstructionDeviceBase
{
 public:
  ~GPUReconstructionOCLBackend() override;

 protected:
  GPUReconstructionOCLBackend(const GPUSettingsDeviceBackend& cfg);

  int32_t InitDevice_Runtime() override;
  int32_t ExitDevice_Runtime() override;
  void UpdateAutomaticProcessingSettings() override;

  int32_t GPUFailedMsgAI(const int64_t error, const char* file, int32_t line);
  void GPUFailedMsgA(const int64_t error, const char* file, int32_t line);

  void SynchronizeGPU() override;
  int32_t DoStuckProtection(int32_t stream, deviceEvent event) override;
  int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false) override;
  void SynchronizeStream(int32_t stream) override;
  void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) override;
  void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) override;
  bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) override;

  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) override;
  void ReleaseEvent(deviceEvent ev) override;
  void RecordMarker(deviceEvent* ev, int32_t stream) override;

  virtual bool ContextForAllPlatforms() { return false; }

  template <class T, int32_t I = 0>
  int32_t AddKernel(bool multi = false);
  template <class T, int32_t I = 0>
  uint32_t FindKernel(int32_t num);
  template <typename K, typename... Args>
  int32_t runKernelBackendInternal(const krnlSetupTime& _xyz, K& k, const Args&... args);
  template <class T, int32_t I = 0>
  gpu_reconstruction_kernels::krnlProperties getKernelPropertiesBackend();

  GPUReconstructionOCLInternals* mInternals;

  template <class T, int32_t I = 0, typename... Args>
  int32_t runKernelBackend(const krnlSetupArgs<T, I, Args...>& args);
  template <class S, class T, int32_t I, bool MULTI>
  S& getKernelObject();

  int32_t GetOCLPrograms();
  bool CheckPlatform(uint32_t i);
};

using GPUReconstructionOCL = GPUReconstructionKernels<GPUReconstructionOCLBackend>;
} // namespace o2::gpu

#endif

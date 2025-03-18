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

/// \file GPUReconstructionCPU.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONICPU_H
#define GPURECONSTRUCTIONICPU_H

#include "GPUReconstructionProcessing.h"
#include "GPUConstantMem.h"
#include <stdexcept>
#include <vector>

#include "GPUGeneralKernels.h"
#include "GPUReconstructionKernelIncludes.h"
#include "GPUReconstructionKernels.h"

namespace o2::gpu
{

class GPUReconstructionCPUBackend : public GPUReconstructionProcessing
{
 public:
  ~GPUReconstructionCPUBackend() override = default;

 protected:
  GPUReconstructionCPUBackend(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionProcessing(cfg) {}
  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackend(const gpu_reconstruction_kernels::krnlSetupArgs<T, I, Args...>& args);
  template <class T, int32_t I = 0, typename... Args>
  void runKernelBackendInternal(const gpu_reconstruction_kernels::krnlSetupTime& _xyz, const Args&... args);
  template <class T, int32_t I>
  gpu_reconstruction_kernels::krnlProperties getKernelPropertiesBackend();
};

class GPUReconstructionCPU : public GPUReconstructionKernels<GPUReconstructionCPUBackend>
{
  friend GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg);
  friend class GPUChain;

 public:
  ~GPUReconstructionCPU() override;
  static constexpr krnlRunRange krnlRunRangeNone{0};
  static constexpr krnlEvent krnlEventNone = krnlEvent{nullptr, nullptr, 0};

  template <class S, int32_t I = 0, typename... Args>
  void runKernel(krnlSetup&& setup, Args&&... args);
  template <class S, int32_t I = 0>
  const gpu_reconstruction_kernels::krnlProperties getKernelProperties()
  {
    return getKernelPropertiesImpl(gpu_reconstruction_kernels::classArgument<S, I>());
  }

  virtual int32_t GPUDebug(const char* state = "UNKNOWN", int32_t stream = -1, bool force = false);
  int32_t GPUStuck() { return mGPUStuck; }
  void ResetDeviceProcessorTypes();

  int32_t RunChains() override;

  void UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, int32_t stream = -1);

 protected:
  struct GPUProcessorProcessors : public GPUProcessor {
    GPUConstantMem* mProcessorsProc = nullptr;
    void* SetPointersDeviceProcessor(void* mem);
    int16_t mMemoryResProcessors = -1;
  };

  GPUReconstructionCPU(const GPUSettingsDeviceBackend& cfg) : GPUReconstructionKernels(cfg) {}

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types)                                                                                                                   \
  inline void runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>, bool cpuFallback, double& timer, krnlSetup&& setup GPUCA_M_STRIP(x_arguments)) \
  {                                                                                                                                                                                          \
    if (cpuFallback) {                                                                                                                                                                       \
      GPUReconstructionCPU::runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));                  \
    } else {                                                                                                                                                                                 \
      runKernelImpl(krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>(setup.x, setup.y, setup.z, timer GPUCA_M_STRIP(x_forward)));                                        \
    }                                                                                                                                                                                        \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

  int32_t registerMemoryForGPU_internal(const void* ptr, size_t size) override { return 0; }
  int32_t unregisterMemoryForGPU_internal(const void* ptr) override { return 0; }

  virtual void SynchronizeStream(int32_t stream) {}
  virtual void SynchronizeEvents(deviceEvent* evList, int32_t nEvents = 1) {}
  virtual void StreamWaitForEvents(int32_t stream, deviceEvent* evList, int32_t nEvents = 1) {}
  virtual bool IsEventDone(deviceEvent* evList, int32_t nEvents = 1) { return true; }
  virtual void RecordMarker(deviceEvent* ev, int32_t stream) {}
  virtual void SynchronizeGPU() {}
  virtual void ReleaseEvent(deviceEvent ev) {}

  size_t TransferMemoryResourceToGPU(GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, true, res->Ptr(), res->PtrDevice()); }
  size_t TransferMemoryResourceToHost(GPUMemoryResource* res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryInternal(res, stream, ev, evList, nEvents, false, res->PtrDevice(), res->Ptr()); }
  size_t TransferMemoryResourcesToGPU(GPUProcessor* proc, int32_t stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, true); }
  size_t TransferMemoryResourcesToHost(GPUProcessor* proc, int32_t stream = -1, bool all = false) { return TransferMemoryResourcesHelper(proc, stream, all, false); }
  size_t TransferMemoryResourceLinkToGPU(int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryResourceToGPU(&mMemoryResources[res], stream, ev, evList, nEvents); }
  size_t TransferMemoryResourceLinkToHost(int16_t res, int32_t stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1) { return TransferMemoryResourceToHost(&mMemoryResources[res], stream, ev, evList, nEvents); }
  virtual size_t GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1);
  virtual size_t GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int32_t nEvents = 1);
  size_t WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream = -1, deviceEvent* ev = nullptr) override;
  virtual size_t TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst);

  int32_t InitDevice() override;
  int32_t ExitDevice() override;
  int32_t GetThread();

  virtual int32_t PrepareTextures() { return 0; }
  virtual int32_t DoStuckProtection(int32_t stream, deviceEvent event) { return 0; }

  // Pointers to tracker classes
  GPUProcessorProcessors mProcShadow; // Host copy of tracker objects that will be used on the GPU
  GPUConstantMem*& mProcessorsShadow = mProcShadow.mProcessorsProc;

  uint32_t mBlockCount = 1;
  uint32_t mThreadCount = 1;
  uint32_t mWarpSize = 1;

 private:
  size_t TransferMemoryResourcesHelper(GPUProcessor* proc, int32_t stream, bool all, bool toGPU);
};

template <class S, int32_t I, typename... Args>
inline void GPUReconstructionCPU::runKernel(krnlSetup&& setup, Args&&... args)
{
  HighResTimer* t = nullptr;
  GPUDataTypes::RecoStep myStep = S::GetRecoStep() == GPUDataTypes::RecoStep::NoRecoStep ? setup.x.step : S::GetRecoStep();
  if (myStep == GPUDataTypes::RecoStep::NoRecoStep) {
    throw std::runtime_error("Failure running general kernel without defining RecoStep");
  }
  int32_t cpuFallback = IsGPU() ? (setup.x.device == krnlDeviceType::CPU ? 2 : (mRecoSteps.stepsGPUMask & myStep) != myStep) : 0;
  uint32_t& nThreads = setup.x.nThreads;
  uint32_t& nBlocks = setup.x.nBlocks;
  const uint32_t stream = setup.x.stream;
  auto prop = getKernelProperties<S, I>();
  const int32_t autoThreads = cpuFallback ? 1 : prop.nThreads;
  const int32_t autoBlocks = cpuFallback ? 1 : (prop.forceBlocks ? prop.forceBlocks : (prop.minBlocks * mBlockCount));
  if (nBlocks == (uint32_t)-1) {
    nBlocks = (nThreads + autoThreads - 1) / autoThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-2) {
    nBlocks = nThreads;
    nThreads = autoThreads;
  } else if (nBlocks == (uint32_t)-3) {
    nBlocks = autoBlocks;
    nThreads = autoThreads;
  } else if ((int32_t)nThreads < 0) {
    nThreads = cpuFallback ? 1 : -nThreads;
  }
  if (nThreads > GPUCA_MAX_THREADS) {
    throw std::runtime_error("GPUCA_MAX_THREADS exceeded");
  }
  if (mProcessingSettings.debugLevel >= 3) {
    GPUInfo("Running kernel %s (Stream %d, Index %d, Grid %d/%d) on %s", GetKernelName<S, I>(), stream, setup.y.index, nBlocks, nThreads, cpuFallback == 2 ? "CPU (forced)" : cpuFallback ? "CPU (fallback)" : mDeviceName.c_str());
  }
  if (nThreads == 0 || nBlocks == 0) {
    return;
  }
  if (mProcessingSettings.debugLevel >= 1) {
    t = &getKernelTimer<S, I>(myStep, !IsGPU() || cpuFallback ? getHostThreadIndex() : stream);
    if ((!mProcessingSettings.deviceTimers || !IsGPU() || cpuFallback) && (mNActiveThreadsOuterLoop < 2 || getHostThreadIndex() == 0)) {
      t->Start();
    }
  }
  double deviceTimerTime = 0.;
  runKernelImplWrapper(gpu_reconstruction_kernels::classArgument<S, I>(), cpuFallback, deviceTimerTime, std::forward<krnlSetup&&>(setup), std::forward<Args>(args)...);
  if (GPUDebug(GetKernelName<S, I>(), stream, mProcessingSettings.serializeGPU & 1)) {
    throw std::runtime_error("kernel failure");
  }
  if (mProcessingSettings.debugLevel >= 1) {
    if (t) {
      if (deviceTimerTime != 0.) {
        t->AddTime(deviceTimerTime);
        if (t->IsRunning()) {
          t->Abort();
        }
      } else if (t->IsRunning()) {
        t->Stop();
      }
    }
    if (CheckErrorCodes(cpuFallback) && !mProcessingSettings.ignoreNonFatalGPUErrors) {
      throw std::runtime_error("kernel error code");
    }
  }
}

} // namespace o2::gpu

#endif

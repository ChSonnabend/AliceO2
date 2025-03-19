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

/// \file GPUReconstructionCPU.cxx
/// \author David Rohr

#include "GPUReconstructionCPU.h"
#include "GPUReconstructionIncludes.h"
#include "GPUReconstructionThreading.h"
#include "GPUChain.h"

#include "GPUTPCClusterData.h"
#include "GPUTPCSectorOutCluster.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUMemoryResource.h"
#include "GPUConstantMem.h"
#include "GPUMemorySizeScalers.h"
#include <atomic>
#include <ctime>

#define GPUCA_LOGGING_PRINTF
#include "GPULogging.h"

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace o2::gpu;
using namespace o2::gpu::gpu_reconstruction_kernels;

constexpr GPUReconstructionCPU::krnlRunRange GPUReconstructionCPU::krnlRunRangeNone;
constexpr GPUReconstructionCPU::krnlEvent GPUReconstructionCPU::krnlEventNone;

GPUReconstruction* GPUReconstruction::GPUReconstruction_Create_CPU(const GPUSettingsDeviceBackend& cfg) { return new GPUReconstructionCPU(cfg); }

GPUReconstructionCPU::~GPUReconstructionCPU()
{
  Exit(); // Needs to be identical to GPU backend bahavior in order to avoid calling abstract methods later in the destructor
}

template <class T, int32_t I, typename... Args>
inline void GPUReconstructionCPUBackend::runKernelBackendInternal(const krnlSetupTime& _xyz, const Args&... args)
{
  auto& x = _xyz.x;
  auto& y = _xyz.y;
  if (x.device == krnlDeviceType::Device) {
    throw std::runtime_error("Cannot run device kernel on host");
  }
  if (x.nThreads != 1) {
    throw std::runtime_error("Cannot run device kernel on host with nThreads != 1");
  }
  int32_t nThreads = getNKernelHostThreads(false);
  if (nThreads > 1) {
    if (mProcessingSettings.debugLevel >= 5) {
      printf("Running %d Threads\n", nThreads);
    }
    tbb::this_task_arena::isolate([&] {
      mThreading->activeThreads->execute([&] {
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, x.nBlocks, 1), [&](const tbb::blocked_range<uint32_t>& r) {
          typename T::GPUSharedMemory smem;
          for (uint32_t iB = r.begin(); iB < r.end(); iB++) {
            T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.index], args...);
          }
        });
      });
    });
  } else {
    for (uint32_t iB = 0; iB < x.nBlocks; iB++) {
      typename T::GPUSharedMemory smem;
      T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Processor(*mHostConstantMem)[y.index], args...);
    }
  }
}

template <>
inline void GPUReconstructionCPUBackend::runKernelBackendInternal<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  int32_t nnThreads = std::max<int32_t>(1, std::min<int32_t>(size / (16 * 1024 * 1024), getNKernelHostThreads(true)));
  if (nnThreads > 1) {
    tbb::parallel_for(0, nnThreads, [&](int iThread) {
      size_t threadSize = size / nnThreads;
      if (threadSize % 4096) {
        threadSize += 4096 - threadSize % 4096;
      }
      size_t offset = threadSize * iThread;
      size_t mySize = std::min<size_t>(threadSize, size - offset);
      if (mySize) {
        memset((char*)ptr + offset, 0, mySize);
      } // clang-format off
    }, tbb::static_partitioner()); // clang-format on
  } else {
    memset(ptr, 0, size);
  }
}

template <class T, int32_t I, typename... Args>
void GPUReconstructionCPUBackend::runKernelBackend(const krnlSetupArgs<T, I, Args...>& args)
{
#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-lambda-capture" // this is not alway captured below
#endif
  std::apply([this, &args](auto&... vals) { runKernelBackendInternal<T, I, Args...>(args.s, vals...); }, args.v);
#pragma GCC diagnostic push
}

template <class T, int32_t I>
krnlProperties GPUReconstructionCPUBackend::getKernelPropertiesBackend()
{
  return krnlProperties{1, 1};
}

#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types)                                                                                                       \
  template void GPUReconstructionCPUBackend::runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args); \
  template krnlProperties GPUReconstructionCPUBackend::getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>();
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

size_t GPUReconstructionCPU::TransferMemoryInternal(GPUMemoryResource* res, int32_t stream, deviceEvent* ev, deviceEvent* evList, int32_t nEvents, bool toGPU, const void* src, void* dst) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpy(void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents) { return 0; }
size_t GPUReconstructionCPU::GPUMemCpyAlways(bool onGpu, void* dst, const void* src, size_t size, int32_t stream, int32_t toGPU, deviceEvent* ev, deviceEvent* evList, int32_t nEvents)
{
  memcpy(dst, src, size);
  return 0;
}
size_t GPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int32_t stream, deviceEvent* ev) { return 0; }
int32_t GPUReconstructionCPU::GPUDebug(const char* state, int32_t stream, bool force) { return 0; }
size_t GPUReconstructionCPU::TransferMemoryResourcesHelper(GPUProcessor* proc, int32_t stream, bool all, bool toGPU)
{
  int32_t inc = toGPU ? GPUMemoryResource::MEMORY_INPUT_FLAG : GPUMemoryResource::MEMORY_OUTPUT_FLAG;
  int32_t exc = toGPU ? GPUMemoryResource::MEMORY_OUTPUT_FLAG : GPUMemoryResource::MEMORY_INPUT_FLAG;
  size_t n = 0;
  for (uint32_t i = 0; i < mMemoryResources.size(); i++) {
    GPUMemoryResource& res = mMemoryResources[i];
    if (res.mPtr == nullptr) {
      continue;
    }
    if (proc && res.mProcessor != proc) {
      continue;
    }
    if (!(res.mType & GPUMemoryResource::MEMORY_GPU) || (res.mType & GPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) {
      continue;
    }
    if (!mProcessingSettings.keepAllMemory && !all && (res.mType & exc) && !(res.mType & inc)) {
      continue;
    }
    if (toGPU) {
      n += TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
    } else {
      n += TransferMemoryResourceToHost(&mMemoryResources[i], stream);
    }
  }
  return n;
}

int32_t GPUReconstructionCPU::GetThread()
{
// Get Thread ID
#if defined(__APPLE__)
  return (0); // syscall is deprecated on MacOS..., only needed for GPU support which we don't do on Mac anyway
#elif defined(_WIN32)
  return ((int32_t)(size_t)GetCurrentThread());
#else
  return ((int32_t)syscall(SYS_gettid));
#endif
}

int32_t GPUReconstructionCPU::InitDevice()
{
  mActiveHostKernelThreads = mMaxHostThreads;
  mThreading->activeThreads = std::make_unique<tbb::task_arena>(mActiveHostKernelThreads);
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      if (mDeviceMemorySize > mHostMemorySize) {
        mHostMemorySize = mDeviceMemorySize;
      }
      mHostMemoryBase = operator new(mHostMemorySize GPUCA_OPERATOR_NEW_ALIGNMENT);
    }
    mHostMemoryPermanent = mHostMemoryBase;
    ClearAllocatedMemory();
  }
  if (mProcessingSettings.inKernelParallel) {
    mBlockCount = mMaxHostThreads;
  }
  mProcShadow.mProcessorsProc = processors();
  return 0;
}

int32_t GPUReconstructionCPU::ExitDevice()
{
  if (mProcessingSettings.memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    if (mMaster == nullptr) {
      operator delete(mHostMemoryBase GPUCA_OPERATOR_NEW_ALIGNMENT);
    }
    mHostMemoryPool = mHostMemoryBase = mHostMemoryPoolEnd = mHostMemoryPermanent = nullptr;
    mHostMemorySize = 0;
  }
  return 0;
}

int32_t GPUReconstructionCPU::RunChains()
{
  mMemoryScalers->temporaryFactor = 1.;
  mStatNEvents++;
  mNEventsProcessed++;

  if (mProcessingSettings.debugLevel >= 3 || mProcessingSettings.allocDebugLevel) {
    printf("Allocated memory when starting processing %34s", "");
    PrintMemoryOverview();
  }
  mTimerTotal.Start();
  const std::clock_t cpuTimerStart = std::clock();
  if (mProcessingSettings.doublePipeline) {
    int32_t retVal = EnqueuePipeline();
    if (retVal) {
      return retVal;
    }
  } else {
    if (mSlaves.size() || mMaster) {
      WriteConstantParams(); // Reinitialize // TODO: Get this in sync with GPUChainTracking::DoQueuedUpdates, and consider the doublePipeline
    }
    for (uint32_t i = 0; i < mChains.size(); i++) {
      int32_t retVal = mChains[i]->RunChain();
      if (retVal) {
        return retVal;
      }
    }
    if (GetProcessingSettings().tpcFreeAllocatedMemoryAfterProcessing) {
      ClearAllocatedMemory();
    }
  }
  mTimerTotal.Stop();
  mStatCPUTime += (double)(std::clock() - cpuTimerStart) / CLOCKS_PER_SEC;
  if (mProcessingSettings.debugLevel >= 3 || mProcessingSettings.allocDebugLevel) {
    printf("Allocated memory when ending processing %36s", "");
    PrintMemoryOverview();
  }

  mStatWallTime = (mTimerTotal.GetElapsedTime() * 1000000. / mStatNEvents);
  std::string nEventReport;
  if (GetProcessingSettings().debugLevel >= 0 && mStatNEvents > 1) {
    nEventReport += "   (avergage of " + std::to_string(mStatNEvents) + " runs)";
  }
  double kernelTotal = 0;
  std::vector<double> kernelStepTimes(GPUDataTypes::N_RECO_STEPS, 0.);

  if (GetProcessingSettings().debugLevel >= 1) {
    for (uint32_t i = 0; i < mTimers.size(); i++) {
      double time = 0;
      if (mTimers[i] == nullptr) {
        continue;
      }
      for (int32_t j = 0; j < mTimers[i]->num; j++) {
        HighResTimer& timer = mTimers[i]->timer[j];
        time += timer.GetElapsedTime();
        if (mProcessingSettings.resetTimers) {
          timer.Reset();
        }
      }

      uint32_t type = mTimers[i]->type;
      if (type == 0) {
        kernelTotal += time;
        int32_t stepNum = getRecoStepNum(mTimers[i]->step);
        kernelStepTimes[stepNum] += time;
      }
      char bandwidth[256] = "";
      if (mTimers[i]->memSize && mStatNEvents && time != 0.) {
        snprintf(bandwidth, 256, " (%8.3f GB/s - %'14zu bytes - %'14zu per call)", mTimers[i]->memSize / time * 1e-9, mTimers[i]->memSize / mStatNEvents, mTimers[i]->memSize / mStatNEvents / mTimers[i]->count);
      }
      printf("Execution Time: Task (%c %8ux): %50s Time: %'10.0f us%s\n", type == 0 ? 'K' : 'C', mTimers[i]->count, mTimers[i]->name.c_str(), time * 1000000 / mStatNEvents, bandwidth);
      if (mProcessingSettings.resetTimers) {
        mTimers[i]->count = 0;
        mTimers[i]->memSize = 0;
      }
    }
  }
  if (GetProcessingSettings().recoTaskTiming) {
    for (int32_t i = 0; i < GPUDataTypes::N_RECO_STEPS; i++) {
      if (kernelStepTimes[i] != 0. || mTimersRecoSteps[i].timerTotal.GetElapsedTime() != 0.) {
        printf("Execution Time: Step              : %11s %38s Time: %'10.0f us %64s ( Total Time : %'14.0f us, CPU Time : %'14.0f us, %'7.2fx )\n", "Tasks",
               GPUDataTypes::RECO_STEP_NAMES[i], kernelStepTimes[i] * 1000000 / mStatNEvents, "", mTimersRecoSteps[i].timerTotal.GetElapsedTime() * 1000000 / mStatNEvents, mTimersRecoSteps[i].timerCPU * 1000000 / mStatNEvents, mTimersRecoSteps[i].timerCPU / mTimersRecoSteps[i].timerTotal.GetElapsedTime());
      }
      if (mTimersRecoSteps[i].bytesToGPU) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10.0f us (%8.3f GB/s - %'14zu bytes - %'14zu per call)\n", mTimersRecoSteps[i].countToGPU, "DMA to GPU", GPUDataTypes::RECO_STEP_NAMES[i], mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1000000 / mStatNEvents,
               mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].timerToGPU.GetElapsedTime() * 1e-9, mTimersRecoSteps[i].bytesToGPU / mStatNEvents, mTimersRecoSteps[i].bytesToGPU / mTimersRecoSteps[i].countToGPU);
      }
      if (mTimersRecoSteps[i].bytesToHost) {
        printf("Execution Time: Step (D %8ux): %11s %38s Time: %'10.0f us (%8.3f GB/s - %'14zu bytes - %'14zu per call)\n", mTimersRecoSteps[i].countToHost, "DMA to Host", GPUDataTypes::RECO_STEP_NAMES[i], mTimersRecoSteps[i].timerToHost.GetElapsedTime() * 1000000 / mStatNEvents,
               mTimersRecoSteps[i].bytesToHost / mTimersRecoSteps[i].timerToHost.GetElapsedTime() * 1e-9, mTimersRecoSteps[i].bytesToHost / mStatNEvents, mTimersRecoSteps[i].bytesToHost / mTimersRecoSteps[i].countToHost);
      }
      if (mProcessingSettings.resetTimers) {
        mTimersRecoSteps[i].bytesToGPU = mTimersRecoSteps[i].bytesToHost = 0;
        mTimersRecoSteps[i].timerToGPU.Reset();
        mTimersRecoSteps[i].timerToHost.Reset();
        mTimersRecoSteps[i].timerTotal.Reset();
        mTimersRecoSteps[i].timerCPU = 0;
        mTimersRecoSteps[i].countToGPU = 0;
        mTimersRecoSteps[i].countToHost = 0;
      }
    }
    for (int32_t i = 0; i < GPUDataTypes::N_GENERAL_STEPS; i++) {
      if (mTimersGeneralSteps[i].GetElapsedTime() != 0.) {
        printf("Execution Time: General Step      : %50s Time: %'10.0f us\n", GPUDataTypes::GENERAL_STEP_NAMES[i], mTimersGeneralSteps[i].GetElapsedTime() * 1000000 / mStatNEvents);
      }
    }
    if (GetProcessingSettings().debugLevel >= 1) {
      mStatKernelTime = kernelTotal * 1000000 / mStatNEvents;
      printf("Execution Time: Total   : %50s Time: %'10.0f us%s\n", "Total Kernel", mStatKernelTime, nEventReport.c_str());
    }
    printf("Execution Time: Total   : %50s Time: %'10.0f us ( CPU Time : %'10.0f us, %7.2fx ) %s\n", "Total Wall", mStatWallTime, mStatCPUTime * 1000000 / mStatNEvents, mStatCPUTime / mTimerTotal.GetElapsedTime(), nEventReport.c_str());
  } else if (GetProcessingSettings().debugLevel >= 0) {
    GPUInfo("Total Wall Time: %10.0f us%s", mStatWallTime, nEventReport.c_str());
  }
  if (mProcessingSettings.resetTimers) {
    mStatNEvents = 0;
    mStatCPUTime = 0;
    mTimerTotal.Reset();
  }

  return 0;
}

void GPUReconstructionCPU::ResetDeviceProcessorTypes()
{
  for (uint32_t i = 0; i < mProcessors.size(); i++) {
    if (mProcessors[i].proc->mGPUProcessorType != GPUProcessor::PROCESSOR_TYPE_DEVICE && mProcessors[i].proc->mLinkedProcessor) {
      mProcessors[i].proc->mLinkedProcessor->InitGPUProcessor(this, GPUProcessor::PROCESSOR_TYPE_DEVICE);
    }
  }
}

void GPUReconstructionCPU::UpdateParamOccupancyMap(const uint32_t* mapHost, const uint32_t* mapGPU, uint32_t occupancyTotal, int32_t stream)
{
  param().occupancyMap = mapHost;
  param().occupancyTotal = occupancyTotal;
  if (IsGPU()) {
    if (!((size_t)&param().occupancyTotal - (size_t)&param().occupancyMap == sizeof(param().occupancyMap) && sizeof(param().occupancyMap) == sizeof(size_t) && sizeof(param().occupancyTotal) < sizeof(size_t))) {
      throw std::runtime_error("occupancy data not consecutive in GPUParam");
    }
    const auto threadContext = GetThreadContext();
    size_t tmp[2] = {(size_t)mapGPU, 0};
    memcpy(&tmp[1], &occupancyTotal, sizeof(occupancyTotal));
    WriteToConstantMemory((char*)&processors()->param.occupancyMap - (char*)processors(), &tmp, sizeof(param().occupancyMap) + sizeof(param().occupancyTotal), stream);
  }
}

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
///
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include <unistd.h>
#include <thread>
#include <fmt/format.h>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"
#include "GPUCommonHelpers.h"

#ifdef ITS_MEASURE_GPU_TIME
#define START_GPU_STREAM_TIMER(stream, name)        \
  cudaEvent_t event_start, event_stop;              \
  GPUChkErrS(cudaEventCreate(&event_start));        \
  GPUChkErrS(cudaEventCreate(&event_stop));         \
  GPUChkErrS(cudaEventRecord(event_start, stream)); \
  const std::string task_name = name;

#define STOP_GPU_STREAM_TIMER(stream)                                                \
  GPUChkErrS(cudaEventRecord(event_stop, stream));                                   \
  GPUChkErrS(cudaEventSynchronize(event_stop));                                      \
  float ms;                                                                          \
  GPUChkErrS(cudaEventElapsedTime(&ms, event_start, event_stop));                    \
  std::cout << "Elapsed time for " << task_name << ": " << ms << " ms" << std::endl; \
  GPUChkErrS(cudaEventDestroy(event_start));                                         \
  GPUChkErrS(cudaEventDestroy(event_stop));
#else
#define START_GPU_STREAM_TIMER(stream, name)
#define STOP_GPU_STREAM_TIMER(stream)
#endif

namespace o2
{
namespace its
{
using constants::GB;
using constants::MB;

namespace gpu
{
class Stream final
{
 public:
  Stream();
  ~Stream();

  [[nodiscard]] const cudaStream_t& get() const;

 private:
  cudaStream_t mStream;
};

Stream::Stream()
{
  GPUChkErrS(cudaStreamCreate(&mStream));
}

Stream::~Stream()
{
  GPUChkErrS(cudaStreamDestroy(mStream));
}

const cudaStream_t& Stream::get() const
{
  return mStream;
}

void* DefaultGPUAllocator::allocate(size_t size)
{
  LOGP(fatal, "Called DefaultGPUAllocator::allocate with size {}", size);
  return nullptr; // to be implemented
}

template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  mIsGPU = true;
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU() = default;

template <int nLayers>
void TimeFrameGPU<nLayers>::allocMemAsync(void** ptr, size_t size, Stream* strPtr, bool extAllocator)
{
  if (extAllocator) {
    *ptr = mAllocator->allocate(size);
  } else {
    LOGP(debug, "Calling default CUDA allocator");
    GPUChkErrS(cudaMallocAsync(reinterpret_cast<void**>(ptr), size, strPtr->get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::setDevicePropagator(const o2::base::PropagatorImpl<float>* propagator)
{
  mPropagatorDevice = propagator;
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadIndexTableUtils(const int iteration)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading indextable utils");
  if (!iteration) {
    LOGP(debug, "gpu-allocation: allocating IndexTableUtils buffer, for {} MB.", sizeof(IndexTableUtils) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtils), nullptr, getExtAllocator());
  }
  LOGP(debug, "gpu-transfer: loading IndexTableUtils object, for {} MB.", sizeof(IndexTableUtils) / MB);
  GPUChkErrS(cudaMemcpyAsync(mIndexTableUtilsDevice, &mIndexTableUtils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUnsortedClustersDevice(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading unsorted clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading {} unsorted clusters on layer {}, for {} MB.", mUnsortedClusters[iLayer].size(), iLayer, mUnsortedClusters[iLayer].size() * sizeof(Cluster) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[iLayer]), mUnsortedClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
      GPUChkErrS(cudaHostRegister(mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mUnsortedClustersDevice[iLayer], mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
    GPUChkErrS(cudaHostRegister(mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mUnsortedClustersDeviceArray, mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersDevice(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading sorted clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading {} clusters on layer {}, for {} MB.", mClusters[iLayer].size(), iLayer, mClusters[iLayer].size() * sizeof(Cluster) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[iLayer]), mClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
      GPUChkErrS(cudaHostRegister(mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mClustersDevice[iLayer], mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
    GPUChkErrS(cudaHostRegister(mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mClustersDeviceArray, mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersIndexTables(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading sorted clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading clusters indextable for layer {} with {} elements, for {} MB.", iLayer, mIndexTables[iLayer].size(), mIndexTables[iLayer].size() * sizeof(int) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mClustersIndexTablesDevice[iLayer]), mIndexTables[iLayer].size() * sizeof(int), nullptr, getExtAllocator());
      GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDevice[iLayer], mIndexTables[iLayer].data(), mIndexTables[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mClustersIndexTablesDeviceArray), nLayers * sizeof(int), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDeviceArray, mClustersIndexTablesDevice.data(), nLayers * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createUsedClustersDevice(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating used clusters flags");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: creating {} used clusters flags on layer {}, for {} MB.", mUsedClusters[iLayer].size(), iLayer, mUsedClusters[iLayer].size() * sizeof(unsigned char) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mUsedClustersDevice[iLayer]), mUsedClusters[iLayer].size() * sizeof(unsigned char), nullptr, getExtAllocator());
      GPUChkErrS(cudaMemsetAsync(mUsedClustersDevice[iLayer], 0, mUsedClusters[iLayer].size() * sizeof(unsigned char), mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mUsedClustersDeviceArray), nLayers * sizeof(unsigned char*), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDeviceArray, mUsedClustersDevice.data(), nLayers * sizeof(unsigned char*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUsedClustersDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading used clusters flags");
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} used clusters flags on layer {}, for {} MB.", mUsedClusters[iLayer].size(), iLayer, mClusters[iLayer].size() * sizeof(unsigned char) / MB);
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDevice[iLayer], mUsedClusters[iLayer].data(), mUsedClusters[iLayer].size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadROframeClustersDevice(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading ROframe clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading {} ROframe clusters info on layer {}, for {} MB.", mROFramesClusters[iLayer].size(), iLayer, mROFramesClusters[iLayer].size() * sizeof(int) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mROFramesClustersDevice[iLayer]), mROFramesClusters[iLayer].size() * sizeof(int), nullptr, getExtAllocator());
      GPUChkErrS(cudaMemcpyAsync(mROFramesClustersDevice[iLayer], mROFramesClusters[iLayer].data(), mROFramesClusters[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mROFrameClustersDeviceArray), nLayers * sizeof(int*), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mROFrameClustersDeviceArray, mROFramesClustersDevice.data(), nLayers * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackingFrameInfoDevice(const int iteration)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading trackingframeinfo");
  if (!iteration) {
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      LOGP(debug, "gpu-transfer: loading {} tfinfo on layer {}, for {} MB.", mTrackingFrameInfo[iLayer].size(), iLayer, mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[iLayer]), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), nullptr, getExtAllocator());
      GPUChkErrS(cudaHostRegister(mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDevice[iLayer], mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    }
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), nLayers * sizeof(TrackingFrameInfo*), nullptr, getExtAllocator());
    GPUChkErrS(cudaHostRegister(mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDeviceArray, mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadMultiplicityCutMask(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading multiplicity cut mask");
    LOGP(debug, "gpu-transfer: loading multiplicity cut mask with {} elements, for {} MB.", mMultiplicityCutMask.size(), mMultiplicityCutMask.size() * sizeof(bool) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mMultMaskDevice), mMultiplicityCutMask.size() * sizeof(uint8_t), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mMultMaskDevice, mMultiplicityCutMask.data(), mMultiplicityCutMask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadVertices(const int iteration)
{
  if (!iteration) {
    START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading seeding vertices");
    LOGP(debug, "gpu-transfer: loading {} ROframes vertices, for {} MB.", mROFramesPV.size(), mROFramesPV.size() * sizeof(int) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mROFramesPVDevice), mROFramesPV.size() * sizeof(int), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mROFramesPVDevice, mROFramesPV.data(), mROFramesPV.size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    LOGP(debug, "gpu-transfer: loading {} seeding vertices, for {} MB.", mPrimaryVertices.size(), mPrimaryVertices.size() * sizeof(Vertex) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mPrimaryVerticesDevice), mPrimaryVertices.size() * sizeof(Vertex), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mPrimaryVerticesDevice, mPrimaryVertices.data(), mPrimaryVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
    STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsLUTDevice(const int iteration)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating tracklets LUTs");
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    if (!iteration) {
      LOGP(debug, "gpu-transfer: creating tracklets LUT for {} elements on layer {}, for {} MB.", mClusters[iLayer].size() + 1, iLayer, (mClusters[iLayer].size() + 1) * sizeof(int) / MB);
      allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDevice[iLayer]), (mClusters[iLayer].size() + 1) * sizeof(int), nullptr, getExtAllocator());
    }
    GPUChkErrS(cudaMemsetAsync(mTrackletsLUTDevice[iLayer], 0, (mClusters[iLayer].size() + 1) * sizeof(int), mGpuStreams[0]->get()));
  }
  if (!iteration) {
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDeviceArray), (nLayers - 1) * sizeof(int*), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), mTrackletsLUTDevice.size() * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsBuffers()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating cells buffers");
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    mNTracklets[iLayer] = 0;
    GPUChkErrS(cudaMemcpyAsync(&mNTracklets[iLayer], mTrackletsLUTDevice[iLayer] + mClusters[iLayer].size(), sizeof(int), cudaMemcpyDeviceToHost));
    LOGP(debug, "gpu-transfer: creating tracklets buffer for {} elements on layer {}, for {} MB.", mNTracklets[iLayer], iLayer, mNTracklets[iLayer] * sizeof(Tracklet) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[iLayer]), mNTracklets[iLayer] * sizeof(Tracklet), nullptr, getExtAllocator());
  }
  allocMemAsync(reinterpret_cast<void**>(&mTrackletsDeviceArray), (nLayers - 1) * sizeof(Tracklet*), nullptr, getExtAllocator());
  GPUChkErrS(cudaHostRegister(mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mTrackletsDeviceArray, mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading tracklets");
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} tracklets on layer {}, for {} MB.", mTracklets[iLayer].size(), iLayer, mTracklets[iLayer].size() * sizeof(Tracklet) / MB);
    GPUChkErrS(cudaHostRegister(mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackletsDevice[iLayer], mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsLUTDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading tracklets");
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading tracklets LUT for {} elements on layer {}, for {} MB", mTrackletsLookupTable[iLayer].size(), iLayer + 1, mTrackletsLookupTable[iLayer].size() * sizeof(int) / MB);
    GPUChkErrS(cudaHostRegister(mTrackletsLookupTable[iLayer].data(), mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDevice[iLayer + 1], mTrackletsLookupTable[iLayer].data(), mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  GPUChkErrS(cudaHostRegister(mTrackletsLUTDevice.data(), (nLayers - 1) * sizeof(int*), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), (nLayers - 1) * sizeof(int*), cudaMemcpyHostToDevice));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursIndexTablesDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating cells neighbours");
  // Here we do also the creation of the CellsDeviceArray, as the cells buffers are populated separately in the previous steps.
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), nullptr, getExtAllocator());
  GPUChkErrS(cudaHostRegister(mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading neighbours LUT for {} elements on layer {}, for {} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeed) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (mNCells[iLayer] + 1) * sizeof(int), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (mNCells[iLayer] + 1) * sizeof(int), mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "reserving neighboursLUT");
  LOGP(debug, "gpu-allocation: reserving neighbours LUT for {} elements on layer {} , for {} MB.", nCells + 1, layer, (nCells + 1) * sizeof(int) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursLUTDevice[layer]), (nCells + 1) * sizeof(int), nullptr, getExtAllocator()); // We need one element more to move exc -> inc
  GPUChkErrS(cudaMemsetAsync(mNeighboursLUTDevice[layer], 0, (nCells + 1) * sizeof(int), mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading cell seeds");
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} cell seeds on layer {}, for {} MB.", mCells[iLayer].size(), iLayer, mCells[iLayer].size() * sizeof(CellSeed) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), mCells[iLayer].size() * sizeof(CellSeed), nullptr, getExtAllocator());
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (mCells[iLayer].size() + 1) * sizeof(int), nullptr, getExtAllocator()); // accessory for the neigh. finding.
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[0]->get()));
    GPUChkErrS(cudaMemcpyAsync(mCellsDevice[iLayer], mCells[iLayer].data(), mCells[iLayer].size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), nullptr, getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsLUTDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating cells LUTs");
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: creating cell LUT for {} elements on layer {}, for {} MB.", mNTracklets[iLayer] + 1, iLayer, (mNTracklets[iLayer] + 1) * sizeof(int) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDevice[iLayer]), (mNTracklets[iLayer] + 1) * sizeof(int), nullptr, getExtAllocator());
    GPUChkErrS(cudaMemsetAsync(mCellsLUTDevice[iLayer], 0, (mNTracklets[iLayer] + 1) * sizeof(int), mGpuStreams[0]->get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDeviceArray), (nLayers - 2) * sizeof(int*), nullptr, getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mCellsLUTDeviceArray, mCellsLUTDevice.data(), mCellsLUTDevice.size() * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsBuffers(const int layer)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "creating cells buffers");
  mNCells[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNCells[layer], mCellsLUTDevice[layer] + mNTracklets[layer], sizeof(int), cudaMemcpyDeviceToHost));
  LOGP(debug, "gpu-transfer: creating cell buffer for {} elements on layer {}, for {} MB.", mNCells[layer], layer, mNCells[layer] * sizeof(CellSeed) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[layer]), mNCells[layer] * sizeof(CellSeed), nullptr, getExtAllocator());

  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsLUTDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading cells LUTs");
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading cell LUT for {} elements on layer {}, for {} MB.", mCellsLookupTable[iLayer].size(), iLayer, mCellsLookupTable[iLayer].size() * sizeof(int) / MB);
    GPUChkErrS(cudaHostRegister(mCellsLookupTable[iLayer].data(), mCellsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mCellsLUTDevice[iLayer + 1], mCellsLookupTable[iLayer].data(), mCellsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadRoadsDevice()
{
  LOGP(debug, "gpu-transfer: loading {} roads, for {} MB.", mRoads.size(), mRoads.size() * sizeof(Road<nLayers - 2>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mRoadsDevice), mRoads.size() * sizeof(Road<nLayers - 2>), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaHostRegister(mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mRoadsDevice, mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackSeedsDevice(std::vector<CellSeed>& seeds)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "loading track seeds");
  LOGP(debug, "gpu-transfer: loading {} track seeds, for {} MB.", seeds.size(), seeds.size() * sizeof(CellSeed) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeed), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaHostRegister(seeds.data(), seeds.size() * sizeof(CellSeed), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int layer, const unsigned int nNeighbours)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "reserving neighbours");
  LOGP(debug, "gpu-allocation: reserving {} neighbours (pairs), for {} MB.", nNeighbours, nNeighbours * sizeof(gpuPair<int, int>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), nNeighbours * sizeof(gpuPair<int, int>), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, nNeighbours * sizeof(gpuPair<int, int>), mGpuStreams[0]->get()));
  LOGP(debug, "gpu-allocation: reserving {} neighbours, for {} MB.", nNeighbours, nNeighbours * sizeof(gpuPair<int, int>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), nNeighbours * sizeof(int), mGpuStreams[0], getExtAllocator());
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int layer, std::vector<std::pair<int, int>>& neighbours)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "reserving neighbours");
  mCellsNeighbours[layer].clear();
  mCellsNeighbours[layer].resize(neighbours.size());
  LOGP(debug, "gpu-allocation: reserving {} neighbours (pairs), for {} MB.", neighbours.size(), neighbours.size() * sizeof(gpuPair<int, int>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), neighbours.size() * sizeof(gpuPair<int, int>), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, neighbours.size() * sizeof(gpuPair<int, int>), mGpuStreams[0]->get()));
  LOGP(debug, "gpu-allocation: reserving {} neighbours, for {} MB.", neighbours.size(), neighbours.size() * sizeof(gpuPair<int, int>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), neighbours.size() * sizeof(int), mGpuStreams[0], getExtAllocator());
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDeviceArray()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "reserving neighbours");
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDeviceArray), (nLayers - 2) * sizeof(int*), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mNeighboursDeviceArray, mNeighboursDevice.data(), (nLayers - 2) * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice(std::vector<CellSeed>& seeds)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "reserving tracks");
  mTrackITSExt.clear();
  mTrackITSExt.resize(seeds.size());
  LOGP(debug, "gpu-allocation: reserving {} tracks, for {} MB.", seeds.size(), seeds.size() * sizeof(o2::its::TrackITSExt) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackITSExtDevice), seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0], getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mTrackITSExtDevice, 0, seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0]->get()));
  GPUChkErrS(cudaHostRegister(mTrackITSExt.data(), seeds.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "downloading cells");
  for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: downloading {} cells on layer: {}, for {} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeed) / MB);
    mCells[iLayer].resize(mNCells[iLayer]);
    GPUChkErrS(cudaMemcpyAsync(mCells[iLayer].data(), mCellsDevice[iLayer], mNCells[iLayer] * sizeof(CellSeed), cudaMemcpyDeviceToHost, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsLUTDevice()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "downloading cell luts");
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    LOGP(debug, "gpu-transfer: downloading cells lut on layer {} for {} elements", iLayer, (mNTracklets[iLayer + 1] + 1));
    mCellsLookupTable[iLayer].resize(mNTracklets[iLayer + 1] + 1);
    GPUChkErrS(cudaMemcpyAsync(mCellsLookupTable[iLayer].data(), mCellsLUTDevice[iLayer + 1], (mNTracklets[iLayer + 1] + 1) * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[0]->get()));
  }
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsNeighboursDevice(std::vector<std::vector<std::pair<int, int>>>& neighbours, const int layer)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), fmt::format("downloading neighbours from layer {}", layer));
  LOGP(debug, "gpu-transfer: downloading {} neighbours, for {} MB.", neighbours[layer].size(), neighbours[layer].size() * sizeof(std::pair<int, int>) / MB);
  // TODO: something less dangerous than assuming the same memory layout of std::pair and gpuPair... or not? :)
  GPUChkErrS(cudaMemcpyAsync(neighbours[layer].data(), mNeighbourPairsDevice[layer], neighbours[layer].size() * sizeof(gpuPair<int, int>), cudaMemcpyDeviceToHost, mGpuStreams[0]->get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadNeighboursLUTDevice(std::vector<int>& lut, const int layer)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), fmt::format("downloading neighbours LUT from layer {}", layer));
  LOGP(debug, "gpu-transfer: downloading neighbours LUT for {} elements on layer {}, for {} MB.", lut.size(), layer, lut.size() * sizeof(int) / MB);
  GPUChkErrS(cudaMemcpyAsync(lut.data(), mNeighboursLUTDevice[layer], lut.size() * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[0]->get()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadTrackITSExtDevice(std::vector<CellSeed>& seeds)
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "downloading tracks");
  LOGP(debug, "gpu-transfer: downloading {} tracks, for {} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / MB);
  GPUChkErrS(cudaMemcpyAsync(mTrackITSExt.data(), mTrackITSExtDevice, seeds.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost, mGpuStreams[0]->get()));
  GPUChkErrS(cudaHostUnregister(mTrackITSExt.data()));
  GPUChkErrS(cudaHostUnregister(seeds.data()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterRest()
{
  START_GPU_STREAM_TIMER(mGpuStreams[0]->get(), "unregistering rest of the host memory");
  LOGP(debug, "unregistering rest of the host memory...");
  GPUChkErrS(cudaHostUnregister(mCellsDevice.data()));
  GPUChkErrS(cudaHostUnregister(mTrackletsDevice.data()));
  STOP_GPU_STREAM_TIMER(mGpuStreams[0]->get());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    GPUChkErrS(cudaHostUnregister(mUnsortedClusters[iLayer].data()));
    GPUChkErrS(cudaHostUnregister(mClusters[iLayer].data()));
    GPUChkErrS(cudaHostUnregister(mTrackingFrameInfo[iLayer].data()));
  }
  GPUChkErrS(cudaHostUnregister(mTrackingFrameInfoDevice.data()));
  GPUChkErrS(cudaHostUnregister(mUnsortedClustersDevice.data()));
  GPUChkErrS(cudaHostUnregister(mClustersDevice.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       IndexTableUtils* utils,
                                       const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(mGpuParams.nTimeFrameChunks);
  for (auto& str : mGpuStreams) {
    str = new Stream();
  }
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
}

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2

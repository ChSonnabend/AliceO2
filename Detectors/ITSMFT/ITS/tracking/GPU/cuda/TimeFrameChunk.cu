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

#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TracerGPU.h"

#include "ITStrackingGPU/TimeFrameChunk.h"

#include <unistd.h>
#include <thread>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

namespace o2::its
{
using constants::GB;
using constants::MB;
namespace gpu
{
using utils::checkGPUError;

template <int nLayers>
GpuTimeFrameChunk<nLayers>::~GpuTimeFrameChunk()
{
  if (mAllocated) {
    for (int i = 0; i < nLayers; ++i) {
      checkGPUError(cudaFree(mClustersDevice[i]));
      // checkGPUError(cudaFree(mTrackingFrameInfoDevice[i]));
      checkGPUError(cudaFree(mClusterExternalIndicesDevice[i]));
      checkGPUError(cudaFree(mIndexTablesDevice[i]));
      if (i < nLayers - 1) {
        checkGPUError(cudaFree(mTrackletsDevice[i]));
        checkGPUError(cudaFree(mTrackletsLookupTablesDevice[i]));
        if (i < nLayers - 2) {
          checkGPUError(cudaFree(mCellsDevice[i]));
          checkGPUError(cudaFree(mCellsLookupTablesDevice[i]));
          checkGPUError(cudaFree(mRoadsLookupTablesDevice[i]));
          if (i < nLayers - 3) {
            checkGPUError(cudaFree(mNeighboursCellLookupTablesDevice[i]));
            checkGPUError(cudaFree(mNeighboursCellDevice[i]));
          }
        }
      }
    }
    // checkGPUError(cudaFree(mRoadsDevice));
    checkGPUError(cudaFree(mCUBTmpBufferDevice));
    checkGPUError(cudaFree(mFoundTrackletsDevice));
    checkGPUError(cudaFree(mNFoundCellsDevice));
    checkGPUError(cudaFree(mCellsDeviceArray));
    checkGPUError(cudaFree(mNeighboursCellDeviceArray));
    checkGPUError(cudaFree(mNeighboursCellLookupTablesDeviceArray));
  }
}

template <int nLayers>
void GpuTimeFrameChunk<nLayers>::allocate(const size_t nrof, Stream& stream)
{
  RANGE("device_partition_allocation", 2);
  mNRof = nrof;
  // for (int i = 0; i < nLayers; ++i) {
  //   static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //   // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //   static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //   static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof, &stream, true);
  //   if (i < nLayers - 1) {
  //     static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //     static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //     if (i < nLayers - 2) {
  //       static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->validatedTrackletsCapacity * nrof, &stream, true);
  //       static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(CellSeed) * mTFGPUParams->maxNeighboursSize * nrof, &stream, true);
  //       static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mRoadsLookupTablesDevice[i]), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, &stream, true);
  //       if (i < nLayers - 3) {
  //         static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mNeighboursCellLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, &stream, true);
  //         static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mNeighboursCellDevice[i])), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, &stream, true);
  //       }
  //       if (i < 2) {
  //         static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&(mNTrackletsPerClusterDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  //       }
  //     }
  //   }
  // }
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mCUBTmpBufferDevice), mTFGPUParams->tmpCUBBufferSize * nrof, &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mLinesDevice), sizeof(Line) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mNFoundLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mNExclusiveFoundLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof + 1, &stream, true); // + 1 for cub::DeviceScan::ExclusiveSum, to cover cases where we have maximum number of clusters per ROF
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mUsedTrackletsDevice), sizeof(unsigned char) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mClusteredLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mTFGPUParams->maxTrackletsPerCluster * nrof, &stream, true);
  // // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mRoadsDevice), sizeof(Road<nLayers - 2>) * mTFGPUParams->maxRoadPerRofSize * nrof, &stream, true);

  // /// Invariant allocations
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mFoundTrackletsDevice), (nLayers - 1) * sizeof(int) * nrof, &stream, true); // No need to reset, we always read it after writing
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mNFoundCellsDevice), (nLayers - 2) * sizeof(int) * nrof, &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mNeighboursCellDeviceArray), (nLayers - 3) * sizeof(int*), &stream, true);
  // static_cast<TimeFrameGPU<nLayers>*>(mTimeFramePtr)->allocMemAsync(reinterpret_cast<void**>(&mNeighboursCellLookupTablesDeviceArray), (nLayers - 3) * sizeof(int*), &stream, true);

  // /// Copy pointers of allocated memory to regrouping arrays
  // checkGPUError(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, stream.get()));
  // checkGPUError(cudaMemcpyAsync(mNeighboursCellDeviceArray, mNeighboursCellDevice.data(), (nLayers - 3) * sizeof(int*), cudaMemcpyHostToDevice, stream.get()));
  // checkGPUError(cudaMemcpyAsync(mNeighboursCellLookupTablesDeviceArray, mNeighboursCellLookupTablesDevice.data(), (nLayers - 3) * sizeof(int*), cudaMemcpyHostToDevice, stream.get()));

  mAllocated = true;
}

template <int nLayers>
void GpuTimeFrameChunk<nLayers>::reset(const Task task, Stream& stream)
{
  RANGE("buffer_reset", 0);
  // if ((bool)task) { // Vertexer-only initialisation (cannot be constexpr: due to the presence of gpu raw calls can't be put in header)
  //   for (int i = 0; i < 2; i++) {
  //     auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
  //     auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof;
  //     thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
  //     checkGPUError(cudaMemsetAsync(mNTrackletsPerClusterDevice[i], 0, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
  //   }
  //   checkGPUError(cudaMemsetAsync(mUsedTrackletsDevice, false, sizeof(unsigned char) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
  //   checkGPUError(cudaMemsetAsync(mClusteredLinesDevice, -1, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mTFGPUParams->maxTrackletsPerCluster * mNRof, stream.get()));
  // } else {
  //   for (int i = 0; i < nLayers; ++i) {
  //     if (i < nLayers - 1) {
  //       checkGPUError(cudaMemsetAsync(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
  //       auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
  //       auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof;
  //       thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
  //       if (i < nLayers - 2) {
  //         checkGPUError(cudaMemsetAsync(mCellsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->cellsLUTsize * mNRof, stream.get()));
  //         checkGPUError(cudaMemsetAsync(mRoadsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
  //         if (i < nLayers - 3) {
  //           checkGPUError(cudaMemsetAsync(mNeighboursCellLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
  //           checkGPUError(cudaMemsetAsync(mNeighboursCellDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
  //         }
  //       }
  //     }
  //   }
  //   checkGPUError(cudaMemsetAsync(mNFoundCellsDevice, 0, (nLayers - 2) * sizeof(int), stream.get()));
  // }
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(const int nrof, const TimeFrameGPUParameters& config)
{
  size_t rofsize = nLayers * sizeof(int); // number of clusters per ROF
  // rofsize += nLayers * sizeof(Cluster) * config.clustersPerROfCapacity;                                        // clusters
  // rofsize += nLayers * sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity;                              // tracking frame info
  // rofsize += nLayers * sizeof(int) * config.clustersPerROfCapacity;                                            // external cluster indices
  // rofsize += nLayers * sizeof(int) * (256 * 128 + 1);                                                          // index tables
  // rofsize += (nLayers - 1) * sizeof(int) * config.clustersPerROfCapacity;                                      // tracklets lookup tables
  // rofsize += (nLayers - 1) * sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity; // tracklets
  // rofsize += 2 * sizeof(int) * config.clustersPerROfCapacity;                                                  // tracklets found per cluster (vertexer)
  // rofsize += sizeof(unsigned char) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;            // used tracklets (vertexer)
  // rofsize += (nLayers - 2) * sizeof(int) * config.validatedTrackletsCapacity;                                  // cells lookup tables
  // rofsize += (nLayers - 2) * sizeof(CellSeed) * config.maxNeighboursSize;                                      // cells
  // rofsize += (nLayers - 3) * sizeof(int) * config.maxNeighboursSize;                                           // cell neighbours lookup tables
  // rofsize += (nLayers - 3) * sizeof(int) * config.maxNeighboursSize;                                           // cell neighbours
  // rofsize += sizeof(Road<nLayers - 2>) * config.maxRoadPerRofSize;                                             // roads
  // rofsize += (nLayers - 2) * sizeof(int) * config.maxNeighboursSize;                                           // road LUT
  // rofsize += sizeof(Line) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;                     // lines
  // rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines
  // rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines exclusive sum
  // rofsize += sizeof(int) * config.clustersPerROfCapacity * config.maxTrackletsPerCluster;                      // lines used in clusterlines

  // rofsize += (nLayers - 1) * sizeof(int); // total found tracklets
  // rofsize += (nLayers - 2) * sizeof(int); // total found cells

  return rofsize * nrof;
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(const TimeFrameGPUParameters& config)
{
  size_t total = config.tmpCUBBufferSize;                  // CUB tmp buffers
  total += sizeof(gpu::StaticTrackingParameters<nLayers>); // static parameters loaded once
  return total;
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeRofPerChunk(const TimeFrameGPUParameters& config, const size_t m)
{
  return (m * GB / (float)(config.nTimeFrameChunks) - GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(config)) / (float)GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(1, config);
}

/// Interface
template <int nLayers>
Cluster* GpuTimeFrameChunk<nLayers>::getDeviceClusters(const int layer)
{
  return mClustersDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceClusterExternalIndices(const int layer)
{
  return mClusterExternalIndicesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceIndexTables(const int layer)
{
  return mIndexTablesDevice[layer];
}

template <int nLayers>
Tracklet* GpuTimeFrameChunk<nLayers>::getDeviceTracklets(const int layer)
{
  return mTrackletsDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceTrackletsLookupTables(const int layer)
{
  return mTrackletsLookupTablesDevice[layer];
}

template <int nLayers>
CellSeed* GpuTimeFrameChunk<nLayers>::getDeviceCells(const int layer)
{
  return mCellsDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellsLookupTables(const int layer)
{
  return mCellsLookupTablesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellNeigboursLookupTables(const int layer)
{
  return mNeighboursCellLookupTablesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellNeighbours(const int layer)
{
  return mNeighboursCellDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceRoadsLookupTables(const int layer)
{
  return mRoadsLookupTablesDevice[layer];
}

// Load data
template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::loadDataOnDevice(const size_t startRof, const size_t maxRof, const int maxLayers, Stream& stream)
{
  RANGE("load_clusters_data", 5);
  // auto nRofs = std::min(maxRof - startRof, mNRof);
  // mNPopulatedRof = mTimeFramePtr->getNClustersROFrange(startRof, nRofs, 0).size();
  // for (int i = 0; i < maxLayers; ++i) {
  //   mHostClusters[i] = mTimeFramePtr->getClustersPerROFrange(startRof, nRofs, i);
  //   mHostIndexTables[i] = mTimeFramePtr->getIndexTablePerROFrange(startRof, nRofs, i);
  //   if (mHostClusters[i].size() > mTFGPUParams->clustersPerROfCapacity * nRofs) {
  //     LOGP(warning, "Clusters on layer {} exceed the expected value, resizing to config value: {}, will lose information!", i, mTFGPUParams->clustersPerROfCapacity * nRofs);
  //   }
  //   checkGPUError(cudaMemcpyAsync(mClustersDevice[i],
  //                                 mHostClusters[i].data(),
  //                                 (int)std::min(mHostClusters[i].size(), mTFGPUParams->clustersPerROfCapacity * nRofs) * sizeof(Cluster),
  //                                 cudaMemcpyHostToDevice, stream.get()));
  //   if (mHostIndexTables[i].data()) {
  //     checkGPUError(cudaMemcpyAsync(mIndexTablesDevice[i],
  //                                   mHostIndexTables[i].data(),
  //                                   mHostIndexTables[i].size() * sizeof(int),
  //                                   cudaMemcpyHostToDevice, stream.get()));
  //   }
  // }
  return mNPopulatedRof; // return the number of ROFs we loaded the data for.
}
template class GpuTimeFrameChunk<7>;
} // namespace gpu
} // namespace o2::its
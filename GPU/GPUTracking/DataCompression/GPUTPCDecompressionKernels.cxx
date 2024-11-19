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

/// \file GPUTPCDecompressionKernels.cxx
/// \author Gabriele Cimador

#include "GPUTPCDecompressionKernels.h"
#include "GPULogging.h"
#include "GPUConstantMem.h"
#include "GPUTPCCompressionTrackModel.h"
#include "GPUCommonAlgorithm.h"
#include "TPCClusterDecompressionCore.inc"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step0attached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t trackStart, int32_t trackEnd)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  const GPUParam& GPUrestrict() param = processors.param;

  const uint32_t maxTime = (param.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;

  for (int32_t i = trackStart + get_global_id(0); i < trackEnd; i += get_global_size(0)) {
    uint32_t offset = decompressor.mAttachedClustersOffsets[i];
    TPCClusterDecompressionCore::decompressTrack(cmprClusters, param, maxTime, i, offset, decompressor);
  }
}

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::step1unattached>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors, int32_t sliceStart, int32_t nSlices)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  ClusterNative* GPUrestrict() clusterBuffer = decompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  uint32_t* offsets = decompressor.mUnattachedClustersOffsets;
  for (int32_t i = get_global_id(0); i < GPUCA_ROW_COUNT * nSlices; i += get_global_size(0)) {
    uint32_t iRow = i % GPUCA_ROW_COUNT;
    uint32_t iSlice = sliceStart + (i / GPUCA_ROW_COUNT);
    const uint32_t linearIndex = iSlice * GPUCA_ROW_COUNT + iRow;
    uint32_t tmpBufferIndex = computeLinearTmpBufferIndex(iSlice, iRow, decompressor.mMaxNativeClustersPerBuffer);
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[iSlice][iRow];
    if (decompressor.mNativeClustersIndex[linearIndex] != 0) {
      decompressorMemcpyBasic(buffer, decompressor.mTmpNativeClusters + tmpBufferIndex, decompressor.mNativeClustersIndex[linearIndex]);
    }
    ClusterNative* clout = buffer + decompressor.mNativeClustersIndex[linearIndex];
    uint32_t end = offsets[linearIndex] + ((linearIndex >= decompressor.mInputGPU.nSliceRows) ? 0 : decompressor.mInputGPU.nSliceRowClusters[linearIndex]);
    TPCClusterDecompressionCore::decompressHits(cmprClusters, offsets[linearIndex], end, clout);
    if (processors.param.rec.tpc.clustersShiftTimebins != 0.f) {
      for (uint32_t k = 0; k < outputAccess->nClusters[iSlice][iRow]; k++) {
        auto& cl = buffer[k];
        float t = cl.getTime() + processors.param.rec.tpc.clustersShiftTimebins;
        if (t < 0) {
          t = 0;
        }
        if (processors.param.continuousMaxTimeBin > 0 && t > processors.param.continuousMaxTimeBin) {
          t = processors.param.continuousMaxTimeBin;
        }
        cl.setTime(t);
      }
    }
  }
}

template <typename T>
GPUdi() void GPUTPCDecompressionKernels::decompressorMemcpyBasic(T* GPUrestrict() dst, const T* GPUrestrict() src, uint32_t size)
{
  for (uint32_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

template <>
GPUdii() void GPUTPCDecompressionUtilKernels::Thread<GPUTPCDecompressionUtilKernels::sortPerSectorRow>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  ClusterNative* GPUrestrict() clusterBuffer = processors.tpcDecompressor.mNativeClustersBuffer;
  const ClusterNativeAccess* outputAccess = processors.ioPtrs.clustersNative;
  for (uint32_t i = get_global_id(0); i < GPUCA_NSLICES * GPUCA_ROW_COUNT; i += get_global_size(0)) {
    uint32_t slice = i / GPUCA_ROW_COUNT;
    uint32_t row = i % GPUCA_ROW_COUNT;
    ClusterNative* buffer = clusterBuffer + outputAccess->clusterOffset[slice][row];
    GPUCommonAlgorithm::sort(buffer, buffer + outputAccess->nClusters[slice][row]);
  }
}

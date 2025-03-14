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

/// \file GPUTPCNNClusterizer.cxx
/// \author Christian Sonnabend

#include "GPUReconstruction.h"
#include "ML/3rdparty/GPUORTFloat16.h"
#include "GPUTPCNNClusterizer.h"

using namespace o2::gpu;

void GPUTPCNNClusterizer::InitializeProcessor() {}

void GPUTPCNNClusterizer::SetMaxData(const GPUTrackingInOutPointers& io) {}

void* GPUTPCNNClusterizer::setIOPointers(void* mem)
{
  if (nnClusterizerDtype == 0 && nnClusterizerElementSize > 0) {
    computePointerWithAlignment(mem, inputData16, nnClusterizerBatchedMode * nnClusterizerElementSize);
  } else if (nnClusterizerDtype == 1 && nnClusterizerElementSize > 0) {
    computePointerWithAlignment(mem, inputData32, nnClusterizerBatchedMode * nnClusterizerElementSize);
  }
  computePointerWithAlignment(mem, peakPositions, nnClusterizerBatchedMode);
  computePointerWithAlignment(mem, clusterFlags, 2 * nnClusterizerBatchedMode);
  computePointerWithAlignment(mem, centralCharges, nnClusterizerBatchedMode);
  computePointerWithAlignment(mem, outputDataClass, nnClusterizerTotalClusters);
  if (nnClusterizerModelClassNumOutputNodes > 0) {
    computePointerWithAlignment(mem, modelProbabilities, nnClusterizerBatchedMode * nnClusterizerModelClassNumOutputNodes);
  }
  if (!nnClusterizerUseCfRegression) {
    if (nnClusterizerModelReg1NumOutputNodes > 0) {
      computePointerWithAlignment(mem, outputDataReg1, nnClusterizerBatchedMode * nnClusterizerModelReg1NumOutputNodes);
    }
    if (nnClusterizerModelReg2NumOutputNodes > 0) {
      computePointerWithAlignment(mem, outputDataReg2, nnClusterizerBatchedMode * nnClusterizerModelReg2NumOutputNodes);
    }
  }
  return mem;
}

void GPUTPCNNClusterizer::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizer::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(mISector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}

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
#include "GPUSettings.h"

using namespace o2::gpu;

void GPUTPCNNClusterizer::InitializeProcessor() {}

void GPUTPCNNClusterizer::SetMaxData(const GPUTrackingInOutPointers& io) {}

void* GPUTPCNNClusterizer::setIOPointers(void* mem)
{
  if (nnClusterizerBatchedMode > 0) {
    if (nnInferenceInputDType == 0 && nnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, inputData_16, nnClusterizerBatchedMode * nnClusterizerElementSize);
    } else if (nnInferenceInputDType == 1 && nnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, inputData_32, nnClusterizerBatchedMode * nnClusterizerElementSize);
    }
    computePointerWithAlignment(mem, clusterFlags, 2 * nnClusterizerBatchedMode);

    if (nnInferenceOutputDType == 0 && nnClusterizerElementSize > 0) {
      if (nnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, modelProbabilities_16, nnClusterizerBatchedMode * nnClusterizerModelClassNumOutputNodes);
      }
      if (!nnClusterizerUseCfRegression) {
        if (nnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, outputDataReg1_16, nnClusterizerBatchedMode * nnClusterizerModelReg1NumOutputNodes);
        }
        if (nnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, outputDataReg2_16, nnClusterizerBatchedMode * nnClusterizerModelReg2NumOutputNodes);
        }
      }
    } else if (nnInferenceOutputDType == 1 && nnClusterizerElementSize > 0) {
      if (nnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, modelProbabilities_32, nnClusterizerBatchedMode * nnClusterizerModelClassNumOutputNodes);
      }
      if (!nnClusterizerUseCfRegression) {
        if (nnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, outputDataReg1_32, nnClusterizerBatchedMode * nnClusterizerModelReg1NumOutputNodes);
        }
        if (nnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, outputDataReg2_32, nnClusterizerBatchedMode * nnClusterizerModelReg2NumOutputNodes);
        }
      }
    }
    if (!nnClusterizerUseCfRegression) {
      if (nnClusterizerModelReg1NumOutputNodes > 0) {
        computePointerWithAlignment(mem, outputDataReg1_32, nnClusterizerBatchedMode * nnClusterizerModelReg1NumOutputNodes);
      }
      if (nnClusterizerModelReg2NumOutputNodes > 0) {
        computePointerWithAlignment(mem, outputDataReg2_32, nnClusterizerBatchedMode * nnClusterizerModelReg2NumOutputNodes);
      }
    }
  }
  if (nnClusterizerTotalClusters > 0) {
    computePointerWithAlignment(mem, outputDataClass, nnClusterizerTotalClusters);
  }
  if (nnClusterizerTotalClusters > 0) {
    computePointerWithAlignment(mem, outputDataClass, nnClusterizerTotalClusters);
  }
  return mem;
}

// std::vector<int32_t> GPUTPCNNClusterizer::pointerSizes() {
//   std::vector<int32_t> sizes(7, -1);
//   if (nnClusterizerBatchedMode > 0) {
//     if (nnInferenceInputDType == 0 && nnClusterizerElementSize > 0) {
//       sizes[0] = nnClusterizerBatchedMode * nnClusterizerElementSize; // inputData16
//     } else if (nnInferenceInputDType == 1 && nnClusterizerElementSize > 0) {
//       sizes[1] = nnClusterizerBatchedMode * nnClusterizerElementSize; // inputData32
//     }
//     sizes[2] = 2 * nnClusterizerBatchedMode; // clusterFlags
//     if (nnClusterizerModelClassNumOutputNodes > 0) {
//       sizes[3] = nnClusterizerBatchedMode * nnClusterizerModelClassNumOutputNodes; // modelProbabilities
//     }
//     if (!nnClusterizerUseCfRegression) {
//       if (nnClusterizerModelReg1NumOutputNodes > 0) {
//         sizes[4] = nnClusterizerBatchedMode * nnClusterizerModelReg1NumOutputNodes; // outputDataReg1
//       }
//       if (nnClusterizerModelReg2NumOutputNodes > 0) {
//         sizes[5] = nnClusterizerBatchedMode * nnClusterizerModelReg2NumOutputNodes; // outputDataReg2
//       }
//     }
//   }
//   if (nnClusterizerTotalClusters > 0) {
//     sizes[6] = nnClusterizerTotalClusters; // outputDataClass
//   }
//   return sizes;
// }

void GPUTPCNNClusterizer::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizer::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(mISector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}

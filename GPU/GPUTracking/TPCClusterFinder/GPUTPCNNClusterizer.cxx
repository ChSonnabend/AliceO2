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
  if (mNnClusterizerBatchedMode > 0) {
    if (mNnInferenceInputDType == 0 && mNnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, mInputData_16, mNnClusterizerBatchedMode * mNnClusterizerElementSize);
    } else if (mNnInferenceInputDType == 1 && mNnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, mInputData_32, mNnClusterizerBatchedMode * mNnClusterizerElementSize);
    }
    computePointerWithAlignment(mem, mClusterFlags, 2 * mNnClusterizerBatchedMode);

    if (mNnInferenceOutputDType == 0 && mNnClusterizerElementSize > 0) {
      if (mNnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, mModelProbabilities_16, mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes);
      }
      if (!mNnClusterizerUseCfRegression) {
        if (mNnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg1_16, mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes);
        }
        if (mNnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg2_16, mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes);
        }
      }
    } else if (mNnInferenceOutputDType == 1 && mNnClusterizerElementSize > 0) {
      if (mNnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, mModelProbabilities_32, mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes);
      }
      if (!mNnClusterizerUseCfRegression) {
        if (mNnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg1_32, mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes);
        }
        if (mNnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg2_32, mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes);
        }
      }
    }
  }
  if (mNnClusterizerTotalClusters > 0) {
    computePointerWithAlignment(mem, mOutputDataClass, mNnClusterizerTotalClusters);
  }
  return mem;
}

// std::vector<int32_t> GPUTPCNNClusterizer::pointerSizes() {
//   std::vector<int32_t> sizes(7, -1);
//   if (mNnClusterizerBatchedMode > 0) {
//     if (mNnInferenceInputDType == 0 && mNnClusterizerElementSize > 0) {
//       sizes[0] = mNnClusterizerBatchedMode * mNnClusterizerElementSize; // inputData16
//     } else if (mNnInferenceInputDType == 1 && mNnClusterizerElementSize > 0) {
//       sizes[1] = mNnClusterizerBatchedMode * mNnClusterizerElementSize; // inputData32
//     }
//     sizes[2] = 2 * mNnClusterizerBatchedMode; // mClusterFlags
//     if (mNnClusterizerModelClassNumOutputNodes > 0) {
//       sizes[3] = mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes; // modelProbabilities
//     }
//     if (!mNnClusterizerUseCfRegression) {
//       if (mNnClusterizerModelReg1NumOutputNodes > 0) {
//         sizes[4] = mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes; // outputDataReg1
//       }
//       if (mNnClusterizerModelReg2NumOutputNodes > 0) {
//         sizes[5] = mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes; // outputDataReg2
//       }
//     }
//   }
//   if (mNnClusterizerTotalClusters > 0) {
//     sizes[6] = mNnClusterizerTotalClusters; // mOutputDataClass
//   }
//   return sizes;
// }

void GPUTPCNNClusterizer::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizer::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(mISector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}

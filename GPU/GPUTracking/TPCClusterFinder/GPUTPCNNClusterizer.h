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

/// \file GPUTPCNNClusterizer.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZER_H
#define O2_GPUTPCNNCLUSTERIZER_H

#include "CfChargePos.h"
#include "GPUProcessor.h"

namespace o2::OrtDataType
{
struct Float16_t;
}

namespace o2::gpu
{

class GPUTPCNNClusterizer : public GPUProcessor
{
 public:
  GPUTPCNNClusterizer() = default;
  void* setIOPointers(void*);
  void RegisterMemoryAllocation();
  void InitializeProcessor();
  void SetMaxData(const GPUTrackingInOutPointers&);

  // Neural network clusterization

  int mNnClusterizerSizeInputRow = 3;
  int mNnClusterizerSizeInputPad = 3;
  int mNnClusterizerSizeInputTime = 3;
  int mNnClusterizerElementSize = -1;
  bool mNnClusterizerAddIndexData = true;
  float mNnClassThreshold = 0.01;
  bool mNnSigmoidTrafoClassThreshold = 1;
  int mNnClusterizerUseCfRegression = 0;
  int mNnClusterizerBatchedMode = 1;
  int mNnClusterizerTotalClusters = 1;
  int mNnClusterizerVerbosity = 0;
  int mNnClusterizerBoundaryFillValue = -1;
  int mNnClusterizerModelClassNumOutputNodes = -1;
  int mNnClusterizerModelReg1NumOutputNodes = -1;
  int mNnClusterizerModelReg2NumOutputNodes = -1;
  int mNnInferenceInputDType = 0;  // 0: float16, 1: float32
  int mNnInferenceOutputDType = 0; // 0: float16, 1: float32
  int mISector = -1;
  int mDeviceId = -1;

  // Memory allocation for neural network

  bool* mClusterFlags = nullptr; // mSplitInTime, mSplitInPad. Techincally both flags are set in the same way -> ClusterAccumulator.cx=nullptr
  int* mOutputDataClass = nullptr;

  // FP32
  float* mInputData_32 = nullptr;
  float* mModelProbabilities_32 = nullptr;
  float* mOutputDataReg1_32 = nullptr;
  float* mOutputDataReg2_32 = nullptr;

  // FP16
  OrtDataType::Float16_t* mInputData_16 = nullptr;
  OrtDataType::Float16_t* mModelProbabilities_16 = nullptr;
  OrtDataType::Float16_t* mOutputDataReg1_16 = nullptr;
  OrtDataType::Float16_t* mOutputDataReg2_16 = nullptr;

  int16_t mMemoryId = -1;
}; // class GPUTPCNNClusterizer

} // namespace o2::gpu

#endif

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

  int nnClusterizerSizeInputRow = 3;
  int nnClusterizerSizeInputPad = 3;
  int nnClusterizerSizeInputTime = 3;
  int nnClusterizerElementSize = -1;
  bool nnClusterizerAddIndexData = true;
  float nnClassThreshold = 0.01;
  bool nnSigmoidTrafoClassThreshold = 1;
  int nnClusterizerUseCfRegression = 0;
  int nnClusterizerBatchedMode = 1;
  int nnClusterizerTotalClusters = 1;
  int nnClusterizerVerbosity = 0;
  int nnClusterizerBoundaryFillValue = -1;
  int nnClusterizerModelClassNumOutputNodes = -1;
  int nnClusterizerModelReg1NumOutputNodes = -1;
  int nnClusterizerModelReg2NumOutputNodes = -1;
  int nnInferenceInputDType = 0;  // 0: float16, 1: float32
  int nnInferenceOutputDType = 0; // 0: float16, 1: float32
  int mISector = -1;
  int deviceId = -1;

  // Memory allocation for neural network

  bool* clusterFlags = nullptr; // mSplitInTime, mSplitInPad. Techincally both flags are set in the same way -> ClusterAccumulator.cx=nullptr
  int* outputDataClass = nullptr;

  // FP32
  float* inputData_32 = nullptr;
  float* modelProbabilities_32 = nullptr;
  float* outputDataReg1_32 = nullptr;
  float* outputDataReg2_32 = nullptr;

  // FP16
  OrtDataType::Float16_t* inputData_16 = nullptr;
  OrtDataType::Float16_t* modelProbabilities_16 = nullptr;
  OrtDataType::Float16_t* outputDataReg1_16 = nullptr;
  OrtDataType::Float16_t* outputDataReg2_16 = nullptr;

  int16_t mMemoryId = -1;
}; // class GPUTPCNNClusterizer

} // namespace o2::gpu

#endif

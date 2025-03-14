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

#include "ChargePos.h"
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
  float nnClassThreshold = 0.16;
  bool nnSigmoidTrafoClassThreshold = 1;
  int nnClusterizerUseCfRegression = 0;
  int nnClusterizerBatchedMode = 1;
  int nnClusterizerTotalClusters = 1;
  int nnClusterizerVerbosity = 0;
  int nnClusterizerBoundaryFillValue = -1;
  int nnClusterizerDumpDigits = 0;
  int nnClusterizerApplyCfDeconvolution = 0;
  int nnClusterizerModelClassNumOutputNodes = -1;
  int nnClusterizerModelReg1NumOutputNodes = -1;
  int nnClusterizerModelReg2NumOutputNodes = -1;
  int nnClusterizerDtype = 0; // 0: float16, 1: float32
  int mISector = -1;

  // Memory allocation for neural network
  uint class2_elements = 0;
  float* inputData32 = nullptr;
  OrtDataType::Float16_t* inputData16 = nullptr;
  float* outputDataClass = nullptr;
  float* modelProbabilities = nullptr;
  float* outputDataReg1 = nullptr;
  float* outputDataReg2 = nullptr;

  ChargePos* peakPositions = nullptr;
  bool* clusterFlags = nullptr; // mSplitInTime, mSplitInPad. Techincally both flags are set in the same way -> ClusterAccumulator.cx=nullptrx
  float* centralCharges = nullptr;
  int16_t mMemoryId = -1;
}; // class GPUTPCNNClusterizer

} // namespace o2::gpu

#endif

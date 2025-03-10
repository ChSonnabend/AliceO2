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

/// \file GPUTPCNNClusterizerInternals.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizerInternals.h"

using namespace o2::gpu;

GPUTPCNNClusterizerInternals::GPUTPCNNClusterizerInternals(GPUSettingsProcessing settings, processorType& clusterer)
{
  clusterer_internal = &clusterer;
  GPUSettingsProcessingNNclusterizer nn_settings = settings.nn;
  OrtOptions = {{"model-path", nn_settings.nnClassificationPath},
                {"device", nn_settings.nnInferenceDevice},
                {"device-id", std::to_string(nn_settings.nnInferenceDeviceId)},
                {"allocate-device-memory", std::to_string(nn_settings.nnInferenceAllocateDevMem)},
                {"dtype", nn_settings.nnInferenceDtype},
                {"intra-op-num-threads", std::to_string(nn_settings.nnInferenceThreadsPerNN)},
                {"enable-optimizations", std::to_string(nn_settings.nnInferenceEnableOrtOptimization)},
                {"enable-profiling", std::to_string(nn_settings.nnInferenceOrtProfiling)},
                {"profiling-output-path", nn_settings.nnInferenceOrtProfilingPath},
                {"logging-level", std::to_string(nn_settings.nnInferenceVerbosity)}};
  sector = clusterer.mISector;

  model_class.init(OrtOptions);
  reg_model_paths = splitString(nn_settings.nnRegressionPath, ":");

  if (!nn_settings.nnClusterizerUseCfRegression) {
    if (model_class.getNumOutputNodes()[0][1] == 1 || reg_model_paths.size() == 1) {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelClassNumOutputNodes = model_class.getNumOutputNodes()[0][1];
    } else {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
      OrtOptions["model-path"] = reg_model_paths[1];
      model_reg_2.init(OrtOptions);
      clusterer.nnClusterizerModelReg2NumOutputNodes = model_reg_2.getNumOutputNodes()[0][1];
    }
  }
}

void* GPUTPCNNClusterizerInternals::setIOPointers(void* mem)
{
  if (clusterer_internal->nnClusterizerDtype == 0) {
    computePointerWithAlignment(mem, clusterer_internal->inputData16, clusterer_internal->nnClusterizerCurrentSize * clusterer_internal->nnClusterizerElementSize);
  } else if (clusterer_internal->nnClusterizerDtype == 1) {
    computePointerWithAlignment(mem, clusterer_internal->inputData32, clusterer_internal->nnClusterizerCurrentSize * clusterer_internal->nnClusterizerElementSize);
  }
  computePointerWithAlignment(mem, clusterer_internal->outputDataClass, clusterer_internal->nnClusterizerCurrentSize);
  computePointerWithAlignment(mem, clusterer_internal->modelProbabilities, clusterer_internal->nnClusterizerCurrentSize * clusterer_internal->nnClusterizerModelClassNumOutputNodes);
  computePointerWithAlignment(mem, clusterer_internal->outputDataReg1, clusterer_internal->nnClusterizerCurrentSize * clusterer_internal->nnClusterizerModelReg1NumOutputNodes);
  computePointerWithAlignment(mem, clusterer_internal->outputDataReg2, clusterer_internal->nnClusterizerCurrentSize * clusterer_internal->nnClusterizerModelReg2NumOutputNodes);
  computePointerWithAlignment(mem, clusterer_internal->peakPositions, clusterer_internal->nnClusterizerCurrentSize);
  computePointerWithAlignment(mem, clusterer_internal->clusterFlags, 2 * clusterer_internal->nnClusterizerCurrentSize);
  computePointerWithAlignment(mem, clusterer_internal->centralCharges, clusterer_internal->nnClusterizerCurrentSize);

  return mem;
}

void GPUTPCNNClusterizerInternals::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizerInternals::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(sector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}

// Apply the neural network to the input data. Note: These are not GPU kernels. We let ONNX take care of that
void GPUTPCNNClusterizerInternals::inferenceNetworkClass(processorType& clusterer, int8_t dtype, uint batch_idx)
{
  if (dtype == 0) {
    model_class.inference<OrtDataType::Float16_t, float>(clusterer.inputData16 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.modelProbabilities);
  } else {
    model_class.inference<float, float>(clusterer.inputData32 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.modelProbabilities);
  }
}

void GPUTPCNNClusterizerInternals::inferenceNetworkReg1(processorType& clusterer, int8_t dtype, uint batch_idx)
{
  if (dtype == 0) {
    model_reg_1.inference<OrtDataType::Float16_t, float>(clusterer.inputData16 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.outputDataReg1);
  } else {
    model_reg_1.inference<float, float>(clusterer.inputData32 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.outputDataReg1);
  }
}

void GPUTPCNNClusterizerInternals::inferenceNetworkReg2(processorType& clusterer, int8_t dtype, uint batch_idx)
{
  if (dtype == 0) {
    model_reg_2.inference<OrtDataType::Float16_t, float>(clusterer.inputData16 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.outputDataReg2);
  } else {
    model_reg_2.inference<float, float>(clusterer.inputData32 + batch_idx, clusterer.nnClusterizerCurrentSize * clusterer.nnClusterizerElementSize, clusterer.outputDataReg2);
  }
}
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
/// \file     model.cxx
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class with functions for ONNX model applications
///

// ONNX includes
#include "ML/onnx_interface.h"

namespace o2
{

namespace ml
{

std::string OnnxModel::printShape(const std::vector<int64_t>& v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

void OnnxModel::init(std::string localPath, bool enableOptimizations, int threads)
{

  LOG(info) << "--- ONNX-ML model ---";
  modelPath = localPath;
  activeThreads = threads;

  /// Enableing optimizations
  if (enableOptimizations) {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  }

  mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx-model");
  mSession = std::make_shared<Ort::Experimental::Session>(*mEnv, modelPath, sessionOptions);

  mInputNames = mSession->GetInputNames();
  mInputShapes = mSession->GetInputShapes();
  mOutputNames = mSession->GetOutputNames();
  mOutputShapes = mSession->GetOutputShapes();

  LOG(info) << "Input Nodes:";
  for (size_t i = 0; i < mInputNames.size(); i++) {
    LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
  }

  LOG(info) << "Output Nodes:";
  for (size_t i = 0; i < mOutputNames.size(); i++) {
    LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
  }
  
  LOG(info) << "--- Model initialized! ---";
}

float* OnnxModel::inference(std::vector<Ort::Value> input, int device_id)
{

  // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sessionOptions, device_id));

  try {
    auto outputTensors = mSession->Run(mInputNames, input, mOutputNames);
    float* outputValues = outputTensors[0].GetTensorMutableData<float>();
    return outputValues;
  } catch (const Ort::Exception& exception) {
    LOG(error) << "Error running model inference: " << exception.what();
  }
  return nullptr;
}

float* OnnxModel::inference(std::vector<float> input, int device_id)
{

  // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sessionOptions, device_id));

  int64_t size = input.size();
  assert(size % mInputShapes[0][1] == 0);
  std::vector<int64_t> inputShape{size / mInputShapes[0][1], mInputShapes[0][1]};
  std::vector<Ort::Value> inputTensors;
  inputTensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(input.data(), size, inputShape));
  try {
    auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);
    float* outputValues = outputTensors[0].GetTensorMutableData<float>();
    return outputValues;
  } catch (const Ort::Exception& exception) {
    LOG(error) << "Error running model inference: " << exception.what();
  }
  return nullptr;
}

template<class T>
float* OnnxModel::inference(T input, int size)
{

  std::vector<std::vector<int64_t>> inputShape = mInputShapes;
  inputShape[0][0] = size;
  auto inputTensors = Ort::Experimental::Value::CreateTensor<float>(&input, size, inputShape);
  try {
    auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);
    float* outputValues = outputTensors[0].template GetTensorMutableData<float>();
    return outputValues;
  } catch (const Ort::Exception& exception) {
    LOG(error) << "Error running model inference: " << exception.what();
  }
  return nullptr;
}

void OnnxModel::setActiveThreads(int threads)
{
  activeThreads = threads;
}

} // namespace gpu

} // namespace GPUCA_NAMESPACE

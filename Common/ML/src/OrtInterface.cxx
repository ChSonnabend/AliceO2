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

/// \file     OrtInterface.cxx
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
/// \brief    A header library for loading ONNX models and inferencing them on CPU and GPU

#include "ML/OrtInterface.h"
#include "ML/3rdparty/GPUORTFloat16.h"

// ONNX includes
#include <onnxruntime_cxx_api.h>

namespace o2
{

namespace ml
{

struct OrtModel::OrtVariables { // The actual implementation is hidden in the .cxx file
  // ORT runtime objects
  Ort::RunOptions runOptions;
  std::shared_ptr<Ort::Env> env = nullptr;
  std::shared_ptr<Ort::Session> session = nullptr; ///< ONNX session
  Ort::SessionOptions sessionOptions;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
  std::unique_ptr<Ort::IoBinding> ioBinding = nullptr;
};

// General purpose
void OrtModel::initOptions(std::unordered_map<std::string, std::string> optionsMap)
{
  pImplOrt = new OrtVariables();

  // Load from options map
  if (!optionsMap.contains("model-path")) {
    LOG(fatal) << "(ORT) Model path cannot be empty!";
  }

  if (!optionsMap["model-path"].empty()) {
    modelPath = optionsMap["model-path"];
    deviceType = (optionsMap.contains("device-type") ? optionsMap["device-type"] : "CPU");
    deviceId = (optionsMap.contains("device-id") ? std::stoi(optionsMap["device-id"]) : -1);
    allocateDeviceMemory = (optionsMap.contains("allocate-device-memory") ? std::stoi(optionsMap["allocate-device-memory"]) : 0);
    intraOpNumThreads = (optionsMap.contains("intra-op-num-threads") ? std::stoi(optionsMap["intra-op-num-threads"]) : 0);
    interOpNumThreads = (optionsMap.contains("inter-op-num-threads") ? std::stoi(optionsMap["inter-op-num-threads"]) : 0);
    loggingLevel = (optionsMap.contains("logging-level") ? std::stoi(optionsMap["logging-level"]) : 0);
    enableProfiling = (optionsMap.contains("enable-profiling") ? std::stoi(optionsMap["enable-profiling"]) : 0);
    enableOptimizations = (optionsMap.contains("enable-optimizations") ? std::stoi(optionsMap["enable-optimizations"]) : 0);
    envName = (optionsMap.contains("onnx-environment-name") ? optionsMap["onnx-environment-name"] : "onnx_model_inference");

    if (deviceType == "CPU") {
      (pImplOrt->sessionOptions).SetIntraOpNumThreads(intraOpNumThreads);
      (pImplOrt->sessionOptions).SetInterOpNumThreads(interOpNumThreads);
      if (intraOpNumThreads > 1 || interOpNumThreads > 1) {
        (pImplOrt->sessionOptions).SetExecutionMode(ExecutionMode::ORT_PARALLEL);
      } else if (intraOpNumThreads == 1) {
        (pImplOrt->sessionOptions).SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      }
      if (loggingLevel < 2) {
        LOG(info) << "(ORT) CPU execution provider set with " << intraOpNumThreads << " (intraOpNumThreads) and " << interOpNumThreads << " (interOpNumThreads) threads";
      }
    }

    // OrtROCMProviderOptions rocm_options{};
    // (pImplOrt->sessionOptions).AppendExecutionProvider_ROCM(rocm_options);

    (pImplOrt->sessionOptions).DisableMemPattern();
    (pImplOrt->sessionOptions).DisableCpuMemArena();

    if (enableProfiling) {
      if (optionsMap.contains("profiling-output-path")) {
        (pImplOrt->sessionOptions).EnableProfiling((optionsMap["profiling-output-path"] + "/ORT_LOG_").c_str());
      } else {
        LOG(warning) << "(ORT) If profiling is enabled, optionsMap[\"profiling-output-path\"] should be set. Disabling profiling for now.";
        (pImplOrt->sessionOptions).DisableProfiling();
      }
    } else {
      (pImplOrt->sessionOptions).DisableProfiling();
    }

    (pImplOrt->sessionOptions).SetGraphOptimizationLevel(GraphOptimizationLevel(enableOptimizations));
    (pImplOrt->sessionOptions).SetLogSeverityLevel(OrtLoggingLevel(loggingLevel));

    mInitialized = true;
  } else {
    LOG(fatal) << "(ORT) Model path cannot be empty!";
  }
}

void OrtModel::initEnvironment()
{
  pImplOrt->env = std::make_shared<Ort::Env>(
    // tp_options,
    OrtLoggingLevel(loggingLevel),
    (envName.empty() ? "ORT" : envName.c_str()),
    // Integrate ORT logging into Fairlogger
    [](void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message) {
      if (severity == ORT_LOGGING_LEVEL_VERBOSE) {
        LOG(debug) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      } else if (severity == ORT_LOGGING_LEVEL_INFO) {
        LOG(info) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      } else if (severity == ORT_LOGGING_LEVEL_WARNING) {
        LOG(warning) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      } else if (severity == ORT_LOGGING_LEVEL_ERROR) {
        LOG(error) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      } else if (severity == ORT_LOGGING_LEVEL_FATAL) {
        LOG(fatal) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      } else {
        LOG(info) << "(ORT) [" << logid << "|" << category << "|" << code_location << "]: " << message;
      }
    },
    (void*)3);
  (pImplOrt->env)->DisableTelemetryEvents(); // Disable telemetry events
}

void OrtModel::initSession()
{
  if (allocateDeviceMemory) {
    memoryOnDevice(deviceId);
  }
  pImplOrt->session = std::make_shared<Ort::Session>(*pImplOrt->env, modelPath.c_str(), pImplOrt->sessionOptions);
  pImplOrt->ioBinding = std::make_unique<Ort::IoBinding>(*pImplOrt->session);

  setIO();

  if (loggingLevel < 2) {
    LOG(info) << "(ORT) Model loaded successfully! (inputs: " << printShape(mInputShapes, mInputNames) << ", outputs: " << printShape(mOutputShapes, mInputNames) << ")";
  }
}

void OrtModel::memoryOnDevice(int32_t deviceIndex)
{
#if (defined(ORT_ROCM_BUILD) || defined(ORT_MIGRAPHX_BUILD) || defined(ORT_CUDA_BUILD) || defined(ORT_TENSORRT_BUILD))
  if (deviceIndex >= 0) {
    (pImplOrt->runOptions).AddConfigEntry("disable_synchronize_execution_providers", "1");
    (pImplOrt->sessionOptions).AddConfigEntry("session.use_device_allocator_for_initializers", "1"); // See kOrtSessionOptionsUseDeviceAllocatorForInitializers, https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    (pImplOrt->sessionOptions).AddConfigEntry("session.use_env_allocators", "1");                    // This should enable to use the volatile memory allocation defined in O2/GPU/GPUTracking/TPCClusterFinder/GPUTPCNNClusterizerHost.cxx; not working yet: ONNX still assigns new memory at init time
    (pImplOrt->sessionOptions).AddConfigEntry("session_options.enable_cpu_mem_arena", "0");          // This should enable to use the volatile memory allocation defined in O2/GPU/GPUTracking/TPCClusterFinder/GPUTPCNNClusterizerHost.cxx; not working yet: ONNX still assigns new memory at init time
    // Arena memory shrinkage comes at performance cost
    /// For now prefer to use single allocation, enabled by O2/GPU/GPUTracking/Base/cuda/GPUReconstructionCUDA.cu -> SetONNXGPUStream -> rocm_options.arena_extend_strategy = 0;
    // (pImplOrt->runOptions).AddConfigEntry("memory.enable_memory_arena_shrinkage", ("gpu:" + std::to_string(deviceIndex)).c_str()); // See kOrtRunOptionsConfigEnableMemoryArenaShrinkage, https://github.com/microsoft/onnxruntime/blob/90c263f471bbce724e77d8e62831d3a9fa838b2f/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h#L27

    std::string dev_mem_str = "";
    if (deviceType == "ROCM") {
      dev_mem_str = "Hip";
    }
    if (deviceType == "CUDA") {
      dev_mem_str = "Cuda";
    }
    pImplOrt->memoryInfo = Ort::MemoryInfo(dev_mem_str.c_str(), OrtAllocatorType::OrtDeviceAllocator, deviceIndex, OrtMemType::OrtMemTypeDefault);
    if (loggingLevel < 2) {
      LOG(info) << "(ORT) Memory info set to on-device memory for device type " << deviceType << " with ID " << deviceIndex << " and pImplOrt pointer " << pImplOrt;
    }
  }
#endif
}

void OrtModel::resetSession()
{
  pImplOrt->session = std::make_shared<Ort::Session>(*(pImplOrt->env), modelPath.c_str(), pImplOrt->sessionOptions);
}

// Getters
Ort::SessionOptions* OrtModel::getSessionOptions()
{
  return &pImplOrt->sessionOptions;
}

Ort::MemoryInfo* OrtModel::getMemoryInfo()
{
  return &pImplOrt->memoryInfo;
}

Ort::Env* OrtModel::getEnv()
{
  return (pImplOrt->env).get();
}

template <class I, class O>
std::vector<O> OrtModel::v2v(std::vector<I>& input, bool clearInput)
{
  if constexpr (std::is_same_v<I, O>) {
    return input;
  } else {
    std::vector<O> output(input.size());
    std::transform(std::begin(input), std::end(input), std::begin(output), [](I f) { return O(f); });
    if (clearInput) {
      input.clear();
    }
    return output;
  }
}

void OrtModel::setIO()
{
  for (size_t i = 0; i < (pImplOrt->session)->GetInputCount(); ++i) {
    mInputNames.push_back((pImplOrt->session)->GetInputNameAllocated(i, pImplOrt->allocator).get());
  }
  for (size_t i = 0; i < (pImplOrt->session)->GetInputCount(); ++i) {
    mInputShapes.emplace_back((pImplOrt->session)->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < (pImplOrt->session)->GetOutputCount(); ++i) {
    mOutputNames.push_back((pImplOrt->session)->GetOutputNameAllocated(i, pImplOrt->allocator).get());
  }
  for (size_t i = 0; i < (pImplOrt->session)->GetOutputCount(); ++i) {
    mOutputShapes.emplace_back((pImplOrt->session)->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  inputNamesChar.resize(mInputNames.size(), nullptr);
  std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
                 [&](const std::string& str) { return str.c_str(); });
  outputNamesChar.resize(mOutputNames.size(), nullptr);
  std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
                 [&](const std::string& str) { return str.c_str(); });

  inputShapesCopy = mInputShapes;
  outputShapesCopy = mOutputShapes;
  inputSizePerNode.resize(mInputShapes.size(), 1);
  outputSizePerNode.resize(mOutputShapes.size(), 1);
  mInputsTotal = 1;
  for (size_t i = 0; i < mInputShapes.size(); ++i) {
    if (mInputShapes[i].size() > 0) {
      for (size_t j = 1; j < mInputShapes[i].size(); ++j) {
        if (mInputShapes[i][j] > 0) {
          mInputsTotal *= mInputShapes[i][j];
          inputSizePerNode[i] *= mInputShapes[i][j];
        }
      }
    }
  }
  mOutputsTotal = 1;
  for (size_t i = 0; i < mOutputShapes.size(); ++i) {
    if (mOutputShapes[i].size() > 0) {
      for (size_t j = 1; j < mOutputShapes[i].size(); ++j) {
        if (mOutputShapes[i][j] > 0) {
          mOutputsTotal *= mOutputShapes[i][j];
          outputSizePerNode[i] *= mOutputShapes[i][j];
        }
      }
    }
  }
}

void OrtModel::setEnv(Ort::Env* env)
{
  pImplOrt->env = std::shared_ptr<Ort::Env>(env);
}

// Inference
template <class I, class O>
std::vector<O> OrtModel::inference(std::vector<I>& input)
{
  std::vector<int64_t> inputShape = mInputShapes[0];
  inputShape[0] = input.size();
  for (size_t i = 1; i < mInputShapes[0].size(); ++i) {
    inputShape[0] /= mInputShapes[0][i];
  }
  std::vector<Ort::Value> inputTensor;
  if constexpr (std::is_same_v<I, OrtDataType::Float16_t>) {
    inputTensor.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(pImplOrt->memoryInfo, reinterpret_cast<Ort::Float16_t*>(input.data()), input.size(), inputShape.data(), inputShape.size()));
  } else {
    inputTensor.emplace_back(Ort::Value::CreateTensor<I>(pImplOrt->memoryInfo, input.data(), input.size(), inputShape.data(), inputShape.size()));
  }
  // input.clear();
  auto outputTensors = (pImplOrt->session)->Run(pImplOrt->runOptions, inputNamesChar.data(), inputTensor.data(), inputTensor.size(), outputNamesChar.data(), outputNamesChar.size());
  O* outputValues = outputTensors[0].template GetTensorMutableData<O>();
  std::vector<O> outputValuesVec{outputValues, outputValues + inputShape[0] * mOutputShapes[0][1]};
  outputTensors.clear();
  return outputValuesVec;
}

template std::vector<float> OrtModel::inference<float, float>(std::vector<float>&);
template std::vector<float> OrtModel::inference<OrtDataType::Float16_t, float>(std::vector<OrtDataType::Float16_t>&);
template std::vector<OrtDataType::Float16_t> OrtModel::inference<OrtDataType::Float16_t, OrtDataType::Float16_t>(std::vector<OrtDataType::Float16_t>&);

template std::vector<int8_t> OrtModel::inference<int8_t, int8_t>(std::vector<int8_t>&);

template std::vector<int8_t> OrtModel::inference<float, int8_t>(std::vector<float>&);

template <class I, class O>
void OrtModel::inference(I* input, int64_t input_size, O* output)
{
  // std::vector<std::string> providers = Ort::GetAvailableProviders();
  // for (const auto& provider : providers) {
  //     LOG(info) << "Available Execution Provider: " << provider;
  // }
  std::vector<int64_t> inputShape{input_size, (int64_t)mInputShapes[0][1]};
  Ort::Value inputTensor = Ort::Value(nullptr);
  if constexpr (std::is_same_v<I, OrtDataType::Float16_t>) {
    inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(pImplOrt->memoryInfo, reinterpret_cast<Ort::Float16_t*>(input), input_size * mInputShapes[0][1], inputShape.data(), inputShape.size());
  } else {
    inputTensor = Ort::Value::CreateTensor<I>(pImplOrt->memoryInfo, input, input_size * mInputShapes[0][1], inputShape.data(), inputShape.size());
  }
  (pImplOrt->ioBinding)->BindInput(mInputNames[0].c_str(), inputTensor);

  std::vector<int64_t> outputShape{input_size, mOutputShapes[0][1]};
  Ort::Value outputTensor = Ort::Value(nullptr);
  if constexpr (std::is_same_v<O, OrtDataType::Float16_t>) {
    outputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(pImplOrt->memoryInfo, reinterpret_cast<Ort::Float16_t*>(output), input_size * mOutputShapes[0][1], outputShape.data(), outputShape.size());
  } else {
    outputTensor = Ort::Value::CreateTensor<O>(pImplOrt->memoryInfo, output, input_size * mOutputShapes[0][1], outputShape.data(), outputShape.size());
  }
  (pImplOrt->ioBinding)->BindOutput(mOutputNames[0].c_str(), outputTensor);
  (pImplOrt->session)->Run(pImplOrt->runOptions, *pImplOrt->ioBinding);
}

template void OrtModel::inference<OrtDataType::Float16_t, OrtDataType::Float16_t>(OrtDataType::Float16_t*, int64_t, OrtDataType::Float16_t*);
template void OrtModel::inference<OrtDataType::Float16_t, float>(OrtDataType::Float16_t*, int64_t, float*);
template void OrtModel::inference<float, OrtDataType::Float16_t>(float*, int64_t, OrtDataType::Float16_t*);
template void OrtModel::inference<float, float>(float*, int64_t, float*);

template <class I, class O>
void OrtModel::inference(I** input, int64_t input_size, O* output)
{
  std::vector<Ort::Value> inputTensors(inputShapesCopy.size());

  for (size_t i = 0; i < inputShapesCopy.size(); ++i) {

    inputShapesCopy[i][0] = input_size;  // batch-size
    outputShapesCopy[i][0] = input_size; // batch-size

    if constexpr (std::is_same_v<I, OrtDataType::Float16_t>) {
      inputTensors[i] = Ort::Value::CreateTensor<Ort::Float16_t>(
        pImplOrt->memoryInfo,
        reinterpret_cast<Ort::Float16_t*>(input[i]),
        inputSizePerNode[i] * input_size,
        inputShapesCopy[i].data(),
        inputShapesCopy[i].size());
    } else {
      inputTensors[i] = Ort::Value::CreateTensor<I>(
        pImplOrt->memoryInfo,
        input[i],
        inputSizePerNode[i] * input_size,
        inputShapesCopy[i].data(),
        inputShapesCopy[i].size());
    }
  }

  Ort::Value outputTensor = Ort::Value(nullptr);
  if constexpr (std::is_same_v<O, OrtDataType::Float16_t>) {
    outputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
      pImplOrt->memoryInfo,
      reinterpret_cast<Ort::Float16_t*>(output),
      outputSizePerNode[0] * input_size, // assumes that there is only one output node
      outputShapesCopy[0].data(),
      outputShapesCopy[0].size());
  } else {
    outputTensor = Ort::Value::CreateTensor<O>(
      pImplOrt->memoryInfo,
      output,
      outputSizePerNode[0] * input_size, // assumes that there is only one output node
      outputShapesCopy[0].data(),
      outputShapesCopy[0].size());
  }

  // === Run inference ===
  pImplOrt->session->Run(
    pImplOrt->runOptions,
    inputNamesChar.data(),
    inputTensors.data(),
    inputNamesChar.size(),
    outputNamesChar.data(),
    &outputTensor,
    outputNamesChar.size());
}

template void OrtModel::inference<OrtDataType::Float16_t, OrtDataType::Float16_t>(OrtDataType::Float16_t**, int64_t, OrtDataType::Float16_t*);
template void OrtModel::inference<OrtDataType::Float16_t, float>(OrtDataType::Float16_t**, int64_t, float*);
template void OrtModel::inference<float, OrtDataType::Float16_t>(float**, int64_t, OrtDataType::Float16_t*);
template void OrtModel::inference<float, float>(float**, int64_t, float*);

template <class I, class O>
std::vector<O> OrtModel::inference(std::vector<std::vector<I>>& inputs)
{
  std::vector<Ort::Value> input_tensors;

  for (size_t i = 0; i < inputs.size(); ++i) {

    inputShapesCopy[i][0] = inputs[i].size() / inputSizePerNode[i]; // batch-size

    if constexpr (std::is_same_v<I, OrtDataType::Float16_t>) {
      input_tensors.emplace_back(
        Ort::Value::CreateTensor<Ort::Float16_t>(
          pImplOrt->memoryInfo,
          reinterpret_cast<Ort::Float16_t*>(inputs[i].data()),
          inputSizePerNode[i] * inputShapesCopy[i][0],
          inputShapesCopy[i].data(),
          inputShapesCopy[i].size()));
    } else {
      input_tensors.emplace_back(
        Ort::Value::CreateTensor<I>(
          pImplOrt->memoryInfo,
          inputs[i].data(),
          inputSizePerNode[i] * inputShapesCopy[i][0],
          inputShapesCopy[i].data(),
          inputShapesCopy[i].size()));
    }
  }

  int32_t totalOutputSize = mOutputsTotal * inputShapesCopy[0][0];

  // === Run inference ===
  auto output_tensors = pImplOrt->session->Run(
    pImplOrt->runOptions,
    inputNamesChar.data(),
    input_tensors.data(),
    input_tensors.size(),
    outputNamesChar.data(),
    outputNamesChar.size());

  // === Extract output values ===
  O* output_data = output_tensors[0].template GetTensorMutableData<O>();
  std::vector<O> output_vec(output_data, output_data + totalOutputSize);
  output_tensors.clear();
  return output_vec;
}

template std::vector<float> OrtModel::inference<float, float>(std::vector<std::vector<float>>&);
template std::vector<OrtDataType::Float16_t> OrtModel::inference<OrtDataType::Float16_t, OrtDataType::Float16_t>(std::vector<std::vector<OrtDataType::Float16_t>>&);

// Release session
void OrtModel::release(bool profilingEnabled)
{
  // if (profilingEnabled) {
  //   pImplOrt->session->EndProfiling();
  // }
  LOG(info) << "(ORT) Size of pImplOrt: " << sizeof(*pImplOrt) << " bytes";
}

// private
std::string OrtModel::printShape(const std::vector<int64_t>& v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) {
    ss << v[i] << "x";
  }
  ss << v[v.size() - 1];
  return ss.str();
}

std::string OrtModel::printShape(const std::vector<std::vector<int64_t>>& v, std::vector<std::string>& n)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size(); i++) {
    ss << n[i] << " -> (";
    for (size_t j = 0; j < v[i].size() - 1; j++) {
      ss << v[i][j] << "x";
    }
    ss << v[i][v[i].size() - 1] << "); ";
  }
  return ss.str();
}

} // namespace ml

} // namespace o2

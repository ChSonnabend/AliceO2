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

/// \file     OrtInterface.h
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
/// \brief    A header library for loading ONNX models and inferencing them on CPU and GPU

#ifndef O2_ML_ORTINTERFACE_H
#define O2_ML_ORTINTERFACE_H

// C++ and system includes
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>

// O2 includes
#include "Framework/Logger.h"

namespace Ort
{
struct SessionOptions;
struct MemoryInfo;
struct Env;
} // namespace Ort

namespace o2
{

namespace ml
{

class OrtModel
{

 public:
  // Constructors & destructors
  OrtModel() = default;
  OrtModel(std::unordered_map<std::string, std::string> optionsMap) { init(optionsMap); }
  void init(std::unordered_map<std::string, std::string> optionsMap)
  {
    initOptions(optionsMap);
    initEnvironment();
  }
  virtual ~OrtModel() = default;

  // General purpose
  void initOptions(std::unordered_map<std::string, std::string> optionsMap);
  void initEnvironment();
  void initSession();
  void memoryOnDevice(int32_t = 0);
  bool isInitialized() { return mInitialized; }
  void resetSession();

  // Getters
  std::vector<std::vector<int64_t>> getNumInputNodes() const { return mInputShapes; }
  std::vector<std::vector<int64_t>> getNumOutputNodes() const { return mOutputShapes; }
  std::vector<std::string> getInputNames() const { return mInputNames; }
  std::vector<std::string> getOutputNames() const { return mOutputNames; }
  Ort::SessionOptions* getSessionOptions();
  Ort::MemoryInfo* getMemoryInfo();
  Ort::Env* getEnv();
  int32_t getIntraOpNumThreads() const { return intraOpNumThreads; }
  int32_t getInterOpNumThreads() const { return interOpNumThreads; }

  // Setters
  void setDeviceId(int32_t id) { deviceId = id; }
  void setIO();
  void setActiveThreads(int threads) { intraOpNumThreads = threads; }
  void setIntraOpNumThreads(int threads)
  {
    if (deviceType == "CPU") {
      intraOpNumThreads = threads;
    }
  }
  void setInterOpNumThreads(int threads)
  {
    if (deviceType == "CPU") {
      interOpNumThreads = threads;
    }
  }
  void setEnv(Ort::Env*);

  // Conversion
  template <class I, class O>
  std::vector<O> v2v(std::vector<I>&, bool = true);

  // Inferencing
  template <class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. OrtDataType::Float16_t from O2/Common/ML/include/ML/GPUORTFloat16.h
  std::vector<O> inference(std::vector<I>&);

  template <class I, class O>
  std::vector<O> inference(std::vector<std::vector<I>>&);

  template <class I, class O>
  void inference(I*, int64_t, O*);

  template <class I, class O>
  void inference(I**, int64_t, O*);

  void release(bool = false);

 private:
  // ORT variables -> need to be hidden as pImpl
  struct OrtVariables;
  OrtVariables* pImplOrt;

  // Input & Output specifications of the loaded network
  std::vector<const char*> inputNamesChar, outputNamesChar;
  std::vector<std::string> mInputNames, mOutputNames;
  std::vector<std::vector<int64_t>> mInputShapes, mOutputShapes, inputShapesCopy, outputShapesCopy; // Input shapes
  std::vector<int64_t> inputSizePerNode, outputSizePerNode;                                         // Output shapes
  int32_t mInputsTotal = 0, mOutputsTotal = 0;                                                      // Total number of inputs and outputs

  // Environment settings
  bool mInitialized = false;
  std::string modelPath, envName = "", deviceType = "CPU", thread_affinity = ""; // device options should be cpu, rocm, migraphx, cuda
  int32_t intraOpNumThreads = 1, interOpNumThreads = 1, deviceId = -1, enableProfiling = 0, loggingLevel = 0, allocateDeviceMemory = 0, enableOptimizations = 0;

  std::string printShape(const std::vector<int64_t>&);
  std::string printShape(const std::vector<std::vector<int64_t>>&, std::vector<std::string>&);
};

} // namespace ml

} // namespace o2

#endif // O2_ML_ORTINTERFACE_H

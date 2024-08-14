#include <iostream>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <boost/thread.hpp>
#include <stdlib.h>
#include <unordered_map>
#include <regex>
#include <chrono>
#include <thread>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <chrono>

#include "Algorithm/RangeTokenizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "Headers/DataHeader.h"

#include "Steer/MCKinematicsReader.h"

#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "DataFormatsTPC/Defs.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"

#include "DetectorsRaw/HBFUtils.h"

using namespace o2;
using namespace o2::tpc;
using namespace o2::framework;

namespace o2
{
namespace tpc
{
class onnxGPUinference : public Task
{
  public:

    onnxGPUinference(std::unordered_map<std::string, std::string> options_map) : env(ORT_LOGGING_LEVEL_WARNING, "onnx_model_inference") {
        model_path = options_map["path"];
        device = options_map["device"];
        dtype = options_map["dtype"];
        std::stringstream(options_map["device-id"]) >> device_id;
        std::stringstream(options_map["num-iter"]) >> test_size_iter;
        std::stringstream(options_map["size-tensor"]) >> test_size_tensor;
        std::stringstream(options_map["measure-cycle"]) >> epochs_measure;

        LOG(info) << "Options loaded";

        // Set the environment variable to use ROCm execution provider
        if(device=="GPU"){
          Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(session_options, device_id));
          LOG(info) << "ROCM execution provider set";
        } else if(device=="CPU"){
          LOG(info) << "CPU execution";
        } else {
          LOG(fatal) << "Device not recognized";
        }
        // std::vector<std::string> providers = session.GetProviders();
        // for (const auto& provider : providers) {
        //   LOG(info) << "Using execution provider: " << provider << std::endl;
        // }
        
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session.reset(new Ort::Session{env, model_path.c_str(), session_options});
        LOG(info) << "Session created";

        LOG(info) << "Number of iterations: " << test_size_iter << ", size of the test tensor: " << test_size_tensor << ", measuring every " << epochs_measure << " cycles";
      
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            mInputNames.push_back(session->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            mInputShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            mOutputNames.push_back(session->GetOutputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            mOutputShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        LOG(info) << "Initializing ONNX names and sizes";
        inputNamesChar.resize(mInputNames.size(), nullptr);
        std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
            [&](const std::string& str) { return str.c_str(); });
        outputNamesChar.resize(mOutputNames.size(), nullptr);
        std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
            [&](const std::string& str) { return str.c_str(); });

        // Print names
        LOG(info) << "Input Nodes:";
        for (size_t i = 0; i < mInputNames.size(); i++) {
          LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
        }

        LOG(info) << "Output Nodes:";
        for (size_t i = 0; i < mOutputNames.size(); i++) {
          LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
        }
    };

    void runONNXGPUModel(std::vector<Ort::Value>& input) {
        auto outputTensors = session->Run(runOptions, inputNamesChar.data(), input.data(), 1, outputNamesChar.data(), outputNamesChar.size());
    };

    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {
        double time = 0;

        LOG(info) << "Preparing input data";
        // Prepare input data
        std::vector<int64_t> inputShape{test_size_tensor, mInputShapes[0][1]};

        LOG(info) << "Creating memory info";
        // std::string device_type_onnx;
        // if(device=="CPU"){
        //   device_type_onnx = "Cpu";
        // } else if(device=="GPU"){
        //   device_type_onnx = "Rocm";
        // } else {
        //   LOG(fatal) << "Device not recognized";
        // }
        Ort::MemoryInfo mem_info("Cpu", OrtAllocatorType::OrtArenaAllocator, device_id, OrtMemType::OrtMemTypeDefault);

        LOG(info) << "Creating ONNX tensor";
        std::vector<Ort::Value> input_tensor;
        if(dtype=="FP16"){
          std::vector<Ort::Float16_t> input_data(mInputShapes[0][1] * test_size_tensor, (Ort::Float16_t)1.f);  // Example input
          input_tensor.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(mem_info, input_data.data(), input_data.size(), inputShape.data(), inputShape.size())); 
        } else {
          std::vector<float> input_data(mInputShapes[0][1] * test_size_tensor, 1.0f);  // Example input
          input_tensor.emplace_back(Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(), inputShape.data(), inputShape.size())); 
        }

        LOG(info) << "Starting inference";
        for(int i = 0; i < test_size_iter; i++){
          auto start_network_eval = std::chrono::high_resolution_clock::now();
          runONNXGPUModel(input_tensor);
          // std::vector<float> output = model.inference(test);
          auto end_network_eval = std::chrono::high_resolution_clock::now();
          time += std::chrono::duration<double, std::ratio<1, (unsigned long)1e9>>(end_network_eval - start_network_eval).count();
          if((i % epochs_measure == 0) && (i != 0)){
              time /= 1e9;
              LOG(info) << "Total time: " << time << "s. Timing: " << uint64_t((double)test_size_tensor*epochs_measure/time) << " elements / s";
              time = 0;
          }
        }

        // for(auto out : output){
        //   LOG(info) << "Test output: " << out;
        // }
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

  private:
    
    std::vector<char> model_buffer;
    std::string model_path, device, dtype;
    int device_id;
    size_t test_size_iter, test_size_tensor, epochs_measure;
    
    Ort::RunOptions runOptions;
    Ort::Env env;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> inputNamesChar, outputNamesChar;
    std::vector<std::string> mInputNames;
    std::vector<std::vector<int64_t>> mInputShapes;
    std::vector<std::string> mOutputNames;
    std::vector<std::vector<int64_t>> mOutputShapes;

    std::string printShape(const std::vector<int64_t>& v)
    {
      std::stringstream ss("");
      for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
      ss << v[v.size() - 1];
      return ss.str();
    };
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to ONNX model"}},
    {"device", VariantType::String, "CPU", {"Device on which the ONNX model is run"}},
    {"device-id", VariantType::Int, 0, {"Device ID on which the ONNX model is run"}},
    {"dtype", VariantType::String, "-", {"Dtype in which the ONNX model is run (FP16 or FP32)"}},
    {"size-tensor", VariantType::Int, 100000, {"Size tensor"}},
    {"num-iter", VariantType::Int, 100, {"Number of iterations"}},
    {"measure-cycle", VariantType::Int, 10, {"Epochs in which to measure"}},
  };
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec testProcess(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

  // A copy of the global workflow options from customize() to pass to the task
  std::unordered_map<std::string, std::string> options_map{
    {"path", cfgc.options().get<std::string>("path")},
    {"device", cfgc.options().get<std::string>("device")},
    {"device-id", std::to_string(cfgc.options().get<int>("device-id"))},
    {"dtype", cfgc.options().get<std::string>("dtype")},
    {"size-tensor", std::to_string(cfgc.options().get<int>("size-tensor"))},
    {"num-iter", std::to_string(cfgc.options().get<int>("num-iter"))},
    {"measure-cycle", std::to_string(cfgc.options().get<int>("measure-cycle"))},
  };

  return DataProcessorSpec{
    "test-onnx-gpu",
    inputs,
    outputs,
    adaptFromTask<onnxGPUinference>(options_map),
    Options{
      {"somethingElse", VariantType::String, "-", {"Something else"}}
    }
  };
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  WorkflowSpec specs;

  static std::vector<InputSpec> inputs;
  static std::vector<OutputSpec> outputs;

  specs.push_back(testProcess(cfgc, inputs, outputs));

  return specs;
}
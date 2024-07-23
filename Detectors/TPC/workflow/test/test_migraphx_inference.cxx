#define __HIP_PLATFORM_AMD__

#include <iostream>
#include <vector>
#include <fstream>
#include <migraphx/migraphx.hpp>

#ifdef ClassDef
#undef ClassDef
#endif

#ifdef TreeRef
#undef TreeRef
#endif

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
class testMIGraphX : public Task
{
  public:

    testMIGraphX(std::unordered_map<std::string, std::string> options_map){
      model_path = options_map["path"];
      device = options_map["device"];
      dtype = options_map["dtype"];
      std::stringstream(options_map["num-iter"]) >> test_size_iter;
      std::stringstream(options_map["size-tensor"]) >> test_size_tensor;
      std::stringstream(options_map["measure-cycle"]) >> epochs_measure;

      LOG(info) << "Number of iterations: " << test_size_iter << ", size of the test tensor: " << test_size_tensor << ", measuring every " << epochs_measure << " cycles";
    };

    // Helper function to read a file into a vector of chars
    void read_file(const std::string& file_name)
    {
        std::ifstream file(file_name, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_name);
        }
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        model_buffer = std::vector<char>(size);
        if (!file.read(model_buffer.data(), size)) {
            throw std::runtime_error("Failed to read file: " + file_name);
        }
        LOG(info) << "Model loaded successfully";
    };

    void runMIGraphXModel() {
      // Load the ONNX model
      // read_file(model_path);

      // hipChooseDevice(0);

      // Create MIGraphX program
      migraphx::onnx_options onnx_opts;
      migraphx::program p = migraphx::parse_onnx(model_path.c_str());

      p.print();

      LOG(info) << "Model parsed, buffer filled.";

      LOG(info) << "Test";
      migraphx::compile_options comp_opts;
      LOG(info) << "Test";
      comp_opts.set_offload_copy();
      p.compile(migraphx::target("gpu"), comp_opts);

      p.print();

      LOG(info) << "Model compiled. GPU acquired.";

      // Prepare input data
      auto param_shapes = p.get_parameter_shapes();
      auto input        = param_shapes.names().front();

      LOG(info) << "Model input shape: " << param_shapes["input"].bytes();

      std::vector<float> input_data(param_shapes["input"].bytes() / sizeof(float), 1.0f); // Example input data
      migraphx::argument input_arg = migraphx::argument(param_shapes["input"], input_data.data());

      // Create parameter map
      migraphx::program_parameters prog_params;
      prog_params.add(input, migraphx::argument(param_shapes[input], input_data.data()));

      // Execute the program
      auto outputs = p.eval(prog_params);

      // Extract the output data
      float* results = reinterpret_cast<float*>(outputs[0].data());
    };

    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {

        double time = 0;

        for(int i = 0; i < test_size_iter; i++){
          auto start_network_eval = std::chrono::high_resolution_clock::now();
          runMIGraphXModel();
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
    size_t test_size_iter, test_size_tensor, epochs_measure;
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to PyTorch model"}},
    {"device", VariantType::String, "-", {"Device on which the PyTorch model is run"}},
    {"dtype", VariantType::String, "-", {"Dtype in which the PyTorch model is run (FP16 or FP32)"}},
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
    {"dtype", cfgc.options().get<std::string>("dtype")},
    {"size-tensor", std::to_string(cfgc.options().get<int>("size-tensor"))},
    {"num-iter", std::to_string(cfgc.options().get<int>("num-iter"))},
    {"measure-cycle", std::to_string(cfgc.options().get<int>("measure-cycle"))},
  };

  return DataProcessorSpec{
    "test-migraphx",
    inputs,
    outputs,
    adaptFromTask<testMIGraphX>(options_map),
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

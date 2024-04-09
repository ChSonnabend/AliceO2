#include "ML/torch_interface.h"

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
class testTorch : public Task
{
  public:
    // testTorch(std::unordered_map<std::string, std::string> options_map){};
    testTorch(std::unordered_map<std::string, std::string> options_map){
      model_path = options_map["path"];
      device = options_map["device"];
      dtype = options_map["dtype"];

      model.printAvailDevices();
      if(device.find(std::string("-")) == std::string::npos) {
        model.setDevice(false, device);
      } else {
        model.setDevice(true, "cpu");
      }
      model.load(model_path);
      model.printModel();
      if(dtype.find(std::string("half")) != std::string::npos || dtype.find(std::string("FP16")) != std::string::npos) {
        model.setDType(torch::kFloat16);
      } else {
        model.setDType(torch::kFloat32);
      }
    };
    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {
      
      size_t test_size_iter = 100, test_size_tensor = 10000, epochs_measure = 10;
      double time = 0;
      
      for(int i = 0; i < test_size_iter; i++){
        // std::vector<std::vector<float>> test(test_size_tensor);
        // for(int j = 0; j < test_size_tensor; j++){
        //   test[j] = std::vector<float>(7*7*7, 1.f);
        // }
        torch::Tensor test = torch::ones((10000,7*7*7));
        auto start_network_eval = std::chrono::high_resolution_clock::now();
        auto output = model.inference(test);
        // std::vector<float> output = model.inference(test);
        auto end_network_eval = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<double, std::ratio<1, (unsigned long)1e9>>(end_network_eval - start_network_eval).count();
        if((i % epochs_measure == 0) && (i != 0)){
          LOG(info) << "Timing: " << int(test_size_tensor*epochs_measure/(time/1e9)) << " elements / s";
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
    std::string model_path, device, dtype;
    o2::ml::TorchModel model;
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to PyTorch model"}},
    {"device", VariantType::String, "-", {"Device on which the PyTorch model is run"}},
    {"dtype", VariantType::String, "-", {"Dtype in which the PyTorch model is run (FP16 or FP32)"}}
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
  };

  return DataProcessorSpec{
    "test-torch",
    inputs,
    outputs,
    adaptFromTask<testTorch>(options_map),
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

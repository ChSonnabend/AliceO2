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

      model.printAvailDevices();
      if(device.find(std::string("-")) == std::string::npos) {
        model.setDevice(false, device);
      } else {
        model.setDevice(true, "cpu");
      }
      
      model.load(model_path);
      model.printModel();
    };
    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {
      std::vector<std::vector<float>> test(1);
      test[0] = std::vector<float>(7*7*7, 1.f);
      std::vector<float> output = model.inference(test);
      for(auto out : output){
        LOG(info) << "Test output: " << out;
      }
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
  private:
    std::string model_path, device;
    o2::ml::TorchModel model;
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to PyTorch model"}},
    {"device", VariantType::String, "-", {"Device on which the PyTorch model is run"}}
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

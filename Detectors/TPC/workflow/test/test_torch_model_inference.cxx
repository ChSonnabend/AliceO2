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

// #ifdef ClassDef
// #undef ClassDef
// #endif

// #ifdef TreeRef
// #undef TreeRef
// #endif

// #include "ML/torch_interface.h"

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;
using namespace boost;

namespace o2
{
namespace tpc
{
class testTorch : public Task
{
  public:
    testTorch(std::unordered_map<std::string, std::string> options_map){};
    // testTorch(std::unordered_map<std::string, std::string> options_map){
    //   model_path = options_map["path"];
    //   o2::ml::TorchModel model;
    //   model.init(model_path);
    //   model.printModel();
    // };
    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
  private:
    std::string model_path;
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to PyTorch model"}}
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
  };

  return DataProcessorSpec{
    "test-torch",
    inputs,
    outputs,
    adaptFromTask<testTorch>(options_map),
    Options{
      {"somethingElse", VariantType::String, "", {"Something else"}}
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
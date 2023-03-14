#include "CCDB/CcdbApi.h"

#include "DetectorsRaw/HBFUtils.h"

#include "DataFormatsTPC/WorkflowHelper.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"

#include "ML/onnx_interface.h"

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;
using namespace o2::ml;

class OnnxInference : public Task
{
 public:
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int verbose = 0;
  std::string localpath = "";
};

void OnnxInference::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  localpath = ic.options().get<std::string>("path");
}

void OnnxInference::run(ProcessingContext& pc)
{
  OnnxModel network;
  network.init(localpath);

  std::vector<float> dummyInput = {1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.};

  auto start_network_eval = std::chrono::high_resolution_clock::now();
  float* output_network = network.inference(dummyInput);
  auto stop_network_eval = std::chrono::high_resolution_clock::now();

  if (verbose > 0) {
    std::cout << "Network eval duration: " << std::chrono::duration<float, std::ratio<1, 1000000000>>(stop_network_eval - start_network_eval).count() / 2;
    std::cout << "dummy output: " << output_network[0] << ", " << output_network[1] << ", " << output_network[2] << ", " << output_network[3] << std::endl;
  }
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

#include "Framework/runDataProcessing.h"

DataProcessorSpec testOnnx()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tpc-mc-labels",
      inputs,
      outputs,
      adaptFromTask<OnnxInference>(),
      Options{
        {"verbose", VariantType::Int, 0, {"Verbosity level"}},
        {"path", VariantType::String, "", {"Path to local ONNX model"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{testOnnx()};
}
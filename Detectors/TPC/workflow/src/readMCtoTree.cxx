#include "DataFormatsTPC/WorkflowHelper.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2;
using namespace o2::framework;

class readMCtruth : public Task
{
    public:
        void init(InitContext& ic) final;
        void run(ProcessingContext& pc) final;
    private:
        std::string infile = "tpcdigits.root";
        std::string outfile = "mclabels.root";
};

void readMCtruth::init(InitContext& ic){
    infile = ic.options().get<std::string>("input-file");
    outfile = ic.options().get<std::string>("output-file");
}

void readMCtruth::process(ProcessingContext& pc){

    // Reading...
    TFile fin(infile.c_str(), "OPEN");
    LOG(info) << "Opened infile...";
    auto treein = (TTree*)fin.Get("o2sim");
    LOG(info) << "Read the tree...";
    dataformats::IOMCTruthContainerView* io = nullptr;
    auto cont = o2::dataformats::MCLabelIOHelper::loadFromTTree(treein, "Labels", 0);
    LOG(info) << "Container initialized and Labels loaded...";

    // Writing...
    TFile fout(outfile.c_str(), "RECREATE");
    LOG(info) << "Writing to outfile...";
    TTree treeout("o2sim", "o2sim");
    auto br = treeout.Branch("Labels", &io, 32000, 2);
    LOG(info) << "Writing labels";
    treeout.Fill();
    treeout.Write();
    fout.Close();
    LOG(info) << "Done!";
}

#include "Framework/runDataProcessing.h"

DataProcessorSpec readMonteCarloLabels()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tpc-mc-labels",
    inputs,
    outputs,
    adaptFromTask<readMCtruth>(),
    Options{
      {"input-file", VariantType::String, "tpcdigits.root", {"Input file name"}},
      {"output-file", VariantType::String, "mclabels.root", {"Input file name"}}
    }};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
    return WorkflowSpec{readMonteCarloLabels()};
}
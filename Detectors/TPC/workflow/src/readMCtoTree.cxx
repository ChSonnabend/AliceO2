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
#include "Framework/ControlService.h"

using namespace o2;
using namespace o2::framework;

class readMCtruth : public Task
{
 public:
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int sectorsRead = 0;
  std::string infile = "tpcdigits.root";
  std::string outfile = "mclabels.root";
  int verbose = 0;
};

void readMCtruth::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  infile = ic.options().get<std::string>("input-file");
  outfile = ic.options().get<std::string>("output-file");
}

void readMCtruth::run(ProcessingContext& pc)
{
  TFile* digitFile = TFile::Open(infile.c_str());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  o2::dataformats::IOMCTruthContainerView* plabels[36] = {0};
  std::vector<o2::tpc::Digit>* digits[36] = {0};
  int nBranches = 0;
  bool mcPresent = false, perSector = false;

  if (digitTree->GetBranch("TPPCDigit")) {
    LOG(info) << "Joint digit branch is found";
    nBranches = 1;
    digitTree->SetBranchAddress("TPCDigit", &digits[0]);
    if (digitTree->GetBranch("TPCDigitMCTruth")) {
      mcPresent = true;
      digitTree->SetBranchAddress("TPCDigitMCTruth", &plabels[0]);
    }
  } else {
    nBranches = 36;
    perSector = true;
    for (int i = 0; i < 36; i++) {
      std::string digBName = fmt::format("TPCDigit_{:d}", i).c_str();
      if (digitTree->GetBranch(digBName.c_str())) {
        digitTree->SetBranchAddress(digBName.c_str(), &digits[i]);
        std::string digMCBName = fmt::format("TPCDigitMCTruth_{:d}", i).c_str();
        if (digitTree->GetBranch(digMCBName.c_str())) {
          mcPresent = true;
          digitTree->SetBranchAddress(digMCBName.c_str(), &plabels[i]);
        }
      }
    }
  }

  TFile outputFile(outfile.c_str(), "RECREATE");
  TTree* mcTree = new TTree("mcLabels", "MC tree");

  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels[36];

  for (int iEvent = 0; iEvent < digitTree->GetEntriesFast(); ++iEvent) {
    if (verbose > 0) {
      std::cout << "Processing event: " << iEvent;
    }
    digitTree->GetEntry(iEvent);
    for (int ib = 0; ib < nBranches; ib++) {
      if (plabels[ib]) {
        plabels[ib]->copyandflatten(labels[ib]);
        delete plabels[ib];
        plabels[ib] = nullptr;
      }
    }

    for (int ib = 0; ib < nBranches; ib++) {
      if (!digits[ib]) {
        continue;
      }
      if (verbose > 0) {
        std::cout << "Filling branch " << (fmt::format("mc_labels_sector_{:d}", ib)).c_str() << "\n";
      }
      int nd = digits[ib]->size();
      int val = 0;
      mcTree->Branch(fmt::format("mc_labels_sector_{:d}", ib).c_str(), &val, fmt::format("mc_labels_sector_{:d}/I", ib).c_str());
      for (int idig = 0; idig < nd; idig++) {
        const auto& digit = (*digits[ib])[idig];
        o2::MCCompLabel lab;
        if (mcPresent) {
          lab = labels[ib].getLabels(idig)[0];
        }
        if (verbose > 1) {
          std::cout << "Digit " << digit << " from " << lab << "; is noise: " << (lab.isNoise() ? "TRUE" : "FALSE") << "; is valid: " << (lab.isValid() ? "TRUE" : "FALSE") << "\n";
        }
        val = (lab.isValid() ? 1 : 0);
        mcTree->Fill();
      }
      if (verbose > 0) {
        std::cout << "Filled branch " << (fmt::format("mc_labels_sector_{:d}", ib)).c_str() << "\n";
      }
    }
  }
  outputFile.Write();
  outputFile.Close();

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
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
      {"verbose", VariantType::Int, 0, {"Verbosity level"}},
      {"input-file", VariantType::String, "tpcdigits.root", {"Input file name"}},
      {"output-file", VariantType::String, "mclabels.root", {"Input file name"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}
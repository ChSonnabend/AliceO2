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
  int verbose = 0;
  std::string mode = "digits";
  std::string inFileDigits = "tpcdigits.root";
  std::string outFileDigits = "MClabelsDigits.root";
  std::string inFileTracks = "tpctracks.root";
  std::string outFileTracks = "MClabelsTracks.root";
};

void readMCtruth::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  outFileDigits = ic.options().get<std::string>("outfile-digits");
  inFileTracks = ic.options().get<std::string>("infile-tracks");
  outFileTracks = ic.options().get<std::string>("outfile-tracks");
}

void readMCtruth::run(ProcessingContext& pc)
{
  if (mode.find(std::string("digits")) != std::string::npos) {
    TFile* digitFile = TFile::Open(inFileDigits.c_str());
    TTree* digitTree = (TTree*)digitFile->Get("o2sim");

    o2::dataformats::IOMCTruthContainerView* plabels[36] = {0};
    std::vector<o2::tpc::Digit>* digits[36] = {0};
    int nBranches = 0;
    bool mcPresent = false, perSector = false;

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

    TFile outputFile(outFileDigits.c_str(), "RECREATE");
    TTree* mcTree = new TTree("mcLabels", "MC tree");

    Int_t sector, row, pad, qed, validity, fakehit;
    mcTree->Branch("sector", &sector);
    mcTree->Branch("row", &row);
    mcTree->Branch("pad", &pad);
    mcTree->Branch("isQED", &qed);
    mcTree->Branch("isValid", &validity);
    mcTree->Branch("isFake", &fakehit);

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

        for (int idig = 0; idig < nd; idig++) {
          const auto& digit = (*digits[ib])[idig];
          o2::MCCompLabel lab;
          if (mcPresent) {
            lab = labels[ib].getLabels(idig)[0];
          }
          if (verbose > 1) {
            std::cout << "Digit " << digit << " from " << lab << "; is noise: " << (lab.isNoise() ? "TRUE" : "FALSE") << "; is valid: " << (lab.isValid() ? "TRUE" : "FALSE") << "\n";
          }
          sector = ib;
          row = digit.getRow();
          pad = digit.getPad();
          qed = (lab.isQED() ? 1 : 0);
          fakehit = (lab.isFake() ? 1 : 0);
          validity = (lab.isValid() ? 1 : 0);
          mcTree->Fill();
        }
        if (verbose > 0) {
          std::cout << "Filled branch " << (fmt::format("mc_labels_sector_{:d}", ib)).c_str() << "\n";
        }
      }
    }
    mcTree->Write();
    delete mcTree;
    outputFile.Close();
  }
  if (mode.find(std::string("tracks")) != std::string::npos) {
    std::cout << "The 'tracks' functionality is not implemented yet..."
              << "\n";
  };

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
      {"mode", VariantType::String, "digits,tracks", {"Mode for running over tracks-file or digits-file: digits or tracks."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"outfile-digits", VariantType::String, "MClabelsDigits.root", {"Output file name with mc labels (digits)"}},
      {"infile-tracks", VariantType::String, "tpctracks.root", {"Input file name (tracks)"}},
      {"outfile-tracks", VariantType::String, "MClabelsTracks.root", {"Output file name with mc labels (tracks)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}
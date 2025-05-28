#include "TPCWorkflow/QaIdealClusters.h"

namespace o2::tpc {
    class digitCombiner : public Task {
        public:
            digitCombiner() = default;
            ~digitCombiner() = default;

            void readDigits(std::string, std::vector<std::vector<o2::tpc::Digit>>&, uint&);
            void writeToRootFile();

            void init(InitContext&) final;
            void run(ProcessingContext&) final;

        private:
            std::vector<std::vector<o2::tpc::Digit>> digitBuffer;
            std::vector<std::string> inputFiles;
            std::string outputFile;
            bool shiftTimeBins = true;
    };

void digitCombiner::readDigits(std::string file, std::vector<std::vector<o2::tpc::Digit>>& globalBuffer, uint& timeShift) {
    LOG(info) << "Reading digits from file: " << file;

    // reading in the raw digit information
    TFile* digitFile = TFile::Open(file.c_str());
    TTree* digitTree = (TTree*)digitFile->Get("o2sim");

    std::vector<o2::tpc::Digit>* digits = nullptr;

    uint maxtime = 0;
    for(int sector = 0; sector < 36; ++sector) {
        std::string branch_name = fmt::format("TPCDigit_{:d}", sector);
        digitTree->SetBranchAddress(branch_name.c_str(), &digits);
        digitTree->GetEntry(0);

        LOG(info) << "--> Processing sector: " << sector << " with " << (digits ? digits->size() : 0) << " digits";
        uint startIndex = globalBuffer[sector].size();
        globalBuffer[sector].resize(globalBuffer[sector].size() + digits->size());
        for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
            const auto& digit = (*digits)[i_digit];
            o2::tpc::Digit newDigit(digit.getCRU(), digit.getChargeFloat(), digit.getRow(), digit.getPad(), digit.getTimeStamp() + timeShift);
            globalBuffer[sector][startIndex] = newDigit;
            if (newDigit.getTimeStamp() > maxtime) {
                maxtime = newDigit.getTimeStamp();
            }
            startIndex++;
        }
    }
    timeShift = maxtime;
    LOG(info) << "Max time stamp after reading file: " << maxtime;
}

void digitCombiner::init(InitContext& ic) {
    std::string ifiles = ic.options().get<std::string>("input-files");
    std::string ofiles = ic.options().get<std::string>("output-file");
    shiftTimeBins = ic.options().get<int>("shift-time-bins");
    inputFiles = o2::utils::Str::tokenize(ifiles, ';');
    outputFile = ofiles;
}

void digitCombiner::writeToRootFile() {
    // Create output ROOT file
    TFile* outFile = TFile::Open(outputFile.c_str(), "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        throw std::runtime_error("Could not create output ROOT file: " + outputFile);
    }

    // Create TTree
    TTree* outTree = new TTree("o2sim", "Combined TPC Digits");

    // Prepare branch containers
    std::vector<std::vector<o2::tpc::Digit>*> branchDigits(36, nullptr);

    // Create branches for each sector
    LOG(info) << "Writing to output file: " << outputFile;
    for (int sector = 0; sector < 36; ++sector) {
        LOG(info) << "--> Creating branch for sector: " << sector;
        branchDigits[sector] = new std::vector<o2::tpc::Digit>();
        std::string branch_name = fmt::format("TPCDigit_{:d}", sector);
        outTree->Branch(branch_name.c_str(), &branchDigits[sector]);
    }

    // Fill the branch containers with the digitBuffer content
    for (int sector = 0; sector < 36; ++sector) {
        LOG(info) << "--> Filling branch for sector: " << sector;
        *(branchDigits[sector]) = digitBuffer[sector];
    }

    // Fill the tree (single entry, as in readDigits)
    outTree->Fill();

    // Write and clean up
    outFile->cd();
    outTree->Write();
    outFile->Close();

    // Clean up dynamically allocated vectors
    for (auto ptr : branchDigits) {
        delete ptr;
    }
}

void digitCombiner::run(ProcessingContext& pc) {
    digitBuffer.resize(36); // 36 sectors

    uint timeShift = 0;
    for (const auto& file : inputFiles) {
        readDigits(file, digitBuffer, timeShift);
        if (!shiftTimeBins) {
            LOG(warn) << "Not shifting time bins, digits may overlap.";
            timeShift = 0; // Reset time shift for the next file
        }
    }

    writeToRootFile();
}

};

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec processDigitCombiner(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{
  return DataProcessorSpec{
    "tpc-digits-combine",
    inputs,
    outputs,
    adaptFromTask<digitCombiner>(),
    Options{
        {"input-files", VariantType::String, "", {"Semicolon-separated list of digit files"}},
        {"output-file", VariantType::String, "", {"Output digit file"}},
        {"shift-time-bins", VariantType::Int, 1, {"Boolean - 1: Auto shift time-bins, 0: Combine digits without overwriting timestamp -> Digtis will not be shifted and can overlap"}}
    }};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  static std::vector<InputSpec> inputs;
  static std::vector<OutputSpec> outputs;
  specs.push_back(processDigitCombiner(cfgc, inputs, outputs));
  return specs;
}

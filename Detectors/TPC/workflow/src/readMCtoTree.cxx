#include "Headers/DataHeader.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"

#include "TFile.h"
#include "TTree.h"

using namespace o2;
using namespace o2::tpc;
using namespace o2::framework;

class readMCtruth : public Task
{
 public:
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int verbose = 0;
  std::string mode = "digits,native,tracks";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileTracks = "tpctracks.root";
};

void readMCtruth::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");
  inFileTracks = ic.options().get<std::string>("infile-tracks");
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

    TFile outputFile("mclabels_digits.root", "RECREATE");
    TTree* mcTree = new TTree("mcLabelsDigits", "MC tree");

    Int_t sector, row, pad, qed, validity, fakehit;
    mcTree->Branch("digits_sector", &sector);
    mcTree->Branch("digits_row", &row);
    mcTree->Branch("digits_pad", &pad);
    mcTree->Branch("digits_isQED", &qed);
    mcTree->Branch("digits_isValid", &validity);
    mcTree->Branch("digits_isFake", &fakehit);

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
          std::cout << "Filling digits for sector " << ib << "\n";
        }
        int nd = digits[ib]->size();

        for (int idig = 0; idig < nd; idig++) {
          const auto& digit = (*digits[ib])[idig];
          o2::MCCompLabel lab;
          if (mcPresent) {
            lab = labels[ib].getLabels(idig)[0];
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
          std::cout << "Filled digits for sector " << ib << "\n";
        }
      }
    }
    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      std::cout << "TPC digit reader done!\n";
    }
  }

  if (mode.find(std::string("native")) != std::string::npos) {

    // From O2/Detectors/TPC/qc/macro/runClusters.C

    ClusterNativeHelper::Reader tpcClusterReader;
    tpcClusterReader.init(inFileNative.c_str());

    ClusterNativeAccess clusterIndex;
    std::unique_ptr<ClusterNative[]> clusterBuffer;
    memset(&clusterIndex, 0, sizeof(clusterIndex));
    o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;

    qc::Clusters clusters;

    TFile outputFile("mclabels_native.root", "RECREATE");
    TTree* mcTree = new TTree("mcLabelsNative", "MC tree");

    Int_t sector, row, nclusters, nevent;
    Float_t pad, ctime, sigmapad, sigmatime, qmax, qtot;
    mcTree->Branch("native_sector", &sector);
    mcTree->Branch("native_row", &row);
    mcTree->Branch("native_pad", &pad);
    mcTree->Branch("native_time", &ctime);
    mcTree->Branch("native_sigmapad", &sigmapad);
    mcTree->Branch("native_sigmatime", &sigmatime);
    mcTree->Branch("native_event", &nevent);
    mcTree->Branch("native_nclusters", &nclusters);
    mcTree->Branch("native_qmax", &qmax);
    mcTree->Branch("native_qtot", &qtot);

    for (int i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
      if(verbose>0){
        std::cout << "Event " << i << "\n";
      }
      nevent = i;
      tpcClusterReader.read(i);
      tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
      size_t iClusters = 0;
      for (int isector = 0; isector < o2::tpc::constants::MAXSECTOR; ++isector) {
        for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
          const int nClusters = clusterIndex.nClusters[isector][irow];
          sector = isector;
          row = irow;
          nclusters = nClusters;
          if (!nClusters) {
            continue;
          }
          for (int icl = 0; icl < nClusters; ++icl) {
            const auto& cl = *(clusterIndex.clusters[isector][irow] + icl);
            clusters.processCluster(cl, Sector(isector), irow);
            ctime = cl.getTime();
            pad = cl.getPad();
            qmax = cl.getQmax();
            qtot = cl.getQtot();
            sigmapad = cl.getSigmaPad();
            sigmatime = cl.getSigmaTime();
            mcTree->Fill();
            ++iClusters;
          }
        }
      }
    }

    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      std::cout << "TPC native reader done!\n";
    }
  };

  if (mode.find(std::string("tracks")) != std::string::npos) {

    auto inputFile = TFile::Open(inFileTracks.c_str());
    auto tracksTree = (TTree*)inputFile->Get("tpcrec");
    if (tracksTree == nullptr) {
      std::cout << "Error getting tree\n";
      return;
    }
    std::vector<TrackTPC>* tpcTracks = nullptr;
    tracksTree->SetBranchAddress("TPCTracks", &tpcTracks);

    std::vector<o2::tpc::TrackTPC>* mTracks = nullptr;
    tracksTree->SetBranchAddress("TPCTracks", &mTracks);

    std::vector<o2::tpc::TPCClRefElem>* mClusRefTracks = nullptr;
    tracksTree->SetBranchAddress("ClusRefs", &mClusRefTracks);

    std::vector<o2::MCCompLabel>* mMCTruthTracks = nullptr;
    tracksTree->SetBranchAddress("TPCTracksMCTruth", &mMCTruthTracks);

    std::vector<o2::tpc::TrackTPC> mTracksOut;
    std::vector<o2::tpc::TPCClRefElem> mClusRefTracksOut;
    std::vector<o2::MCCompLabel> mMCTruthTracksOut;
    mTracksOut.swap(*mTracks);
    mClusRefTracksOut.swap(*mClusRefTracks);
    mMCTruthTracksOut.swap(*mMCTruthTracks);

    // Accumulating the cluster references and tracks
    for (int iev = 0; iev < tracksTree->GetEntries(); iev++) {
      tracksTree->GetEntry(tracksTree->GetReadEntry() + 1 + iev);
      uint32_t shift = mClusRefTracks->size();

      auto clBegin = mClusRefTracks->begin();
      auto clEnd = mClusRefTracks->end();
      std::copy(clBegin, clEnd, std::back_inserter(mClusRefTracksOut));

      auto trBegin = mTracks->begin();
      auto trEnd = mTracks->end();
      if (shift) {
        for (auto tr = trBegin; tr != trEnd; tr++) {
          tr->shiftFirstClusterRef(shift);
        }
      }
      std::copy(trBegin, trEnd, std::back_inserter(mTracksOut));
      std::copy(mMCTruthTracks->begin(), mMCTruthTracks->end(), std::back_inserter(mMCTruthTracksOut));
    }

    if(verbose > 0){
      std::cout << "Found " << mTracksOut.size() << " tracks! Processing...\n";
    }

    tracksTree->GetEntry(tracksTree->GetReadEntry() + 1);
    using TrackTunePar = o2::globaltracking::TrackTuneParams;
    const auto& trackTune = TrackTunePar::Instance();

    if (trackTune.sourceLevelTPC &&
        (trackTune.useTPCInnerCorr || trackTune.useTPCOuterCorr ||
         trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable || trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable)) {
      for (auto trc : *mTracks) {
        if (trackTune.useTPCInnerCorr) {
          trc.updateParams(trackTune.tpcParInner);
        }
        if (trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable) {
          trc.updateCov(trackTune.tpcCovInner, trackTune.tpcCovInnerType);
        }
        if (trackTune.useTPCOuterCorr) {
          trc.getParamOut().updateParams(trackTune.tpcParOuter);
        }
        if (trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable) {
          trc.getParamOut().updateCov(trackTune.tpcCovOuter, trackTune.tpcCovOuterType);
        }
      }
    }

    TFile outputFile("mclabels_tracks.root", "RECREATE");
    TTree* mcTree = new TTree("mcLabelsTracks", "MC tree");

    uint8_t sectorIndex, rowIndex;
    uint32_t clusterIndex, trackCount = 0;
    // std::array<bool, maxRows> clMap{}, shMap{};

    mcTree->Branch("tracks_sector", &sectorIndex);
    mcTree->Branch("tracks_row", &rowIndex);
    mcTree->Branch("tracks_clusterIdx", &clusterIndex);
    mcTree->Branch("tracks_count", &trackCount);

    for (auto track : mTracksOut) {
      for (int i = 0; i < track.getNClusterReferences(); i++) {
        o2::tpc::TrackTPC::getClusterReference(mClusRefTracksOut, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
        mcTree->Fill();
      }
      trackCount++;
    }

    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      std::cout << "TPC tracks reader done!\n";
    }

    // std::cout << "The 'tracks' functionality is not implemented yet..."
    //           << "\n";
    //
    // // /data.local1/csonnab/MyO2/O2/Detectors/TPC/qc/macro/runPID.C
    // auto file = TFile::Open(inputFileName.data());
    // std::vector<TrackTPC>* tpcTracks = nullptr;
    // tree->SetBranchAddress("TPCTracks", &tpcTracks);
    //
    // // /data.local1/csonnab/MyO2/O2/Detectors/StrangenessTracking/macros/XiTrackingStudy.C
    // auto treeTPC = (TTree*)fTPC->Get("tpcrec");
    // std::vector<o2::MCCompLabel>* labTPCvec = nullptr;
    // treeTPC->SetBranchAddress("TPCTracksMCTruth", &labTPCvec);
    //
    // for(int frame = 0; frame < treeTPC->GetEntriesFast(); frame++){
    //   treeTPC->GetEvent(frame)
    // }
    //
    // // /data.local1/csonnab/MyO2/O2/Detectors/AOD/src/StandaloneAODProducer.cxx
    // auto tpctracks = fetchTracks<o2::tpc::TrackTPC>(inFileTracks, "tpcrec", "TPCTracks");
    // LOG(info) << "FOUND " << tpctracks->size() << " TPC tracks";
    //
    // track = &((*tpctracks)[trackindex.getIndex()]);
    //
    // std::array<float, 3> pxpypz;
    //       track->getPxPyPzGlo(pxpypz);
    //       trackCursor(0, index, 0 /* CORRECT THIS */, track->getX(), track->getAlpha(), track->getY(), track->getZ(), track->getSnp(), track->getTgl(),
    //                   track->getPt() /*CHECK!!*/);
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
      {"mode", VariantType::String, "digits,native,tracks", {"Mode for running over tracks-file or digits-file: digits, native and/or tracks."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"infile-tracks", VariantType::String, "tpctracks.root", {"Input file name (tracks)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}
#include "DetectorsRaw/HBFUtils.h"

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

#include "Headers/DataHeader.h"

#include "Steer/MCKinematicsReader.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;

class readMCtruth : public Task
{
 public:
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int verbose = 0;
  std::string mode = "digits,native,tracks,digitizer";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileTracks = "tpctracks.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";
};

void readMCtruth::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");
  inFileTracks = ic.options().get<std::string>("infile-tracks");
  inFileKinematics = ic.options().get<std::string>("infile-kinematics");
  inFileDigitizer = ic.options().get<std::string>("infile-digitizer");
}

void readMCtruth::run(ProcessingContext& pc)
{

  // Digits --> Raw information about sector, row, pad, time, charge
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
        LOG(info) << "Processing event: " << iEvent;
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
          LOG(info) << "Filling digits for sector " << ib;
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
      }
    }
    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      LOG(info) << "TPC digit reader done!";
    }
  }


  // Native clusters --> Found cluster centers from the clusterizer
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
        LOG(info) << "Event " << i;
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
      LOG(info) << "TPC native reader done!";
    }
  };


  // Tracks --> Reconstructed tracks from the native clusters
  if (mode.find(std::string("tracks")) != std::string::npos) {

    auto inputFile = TFile::Open(inFileTracks.c_str());
    auto tracksTree = (TTree*)inputFile->Get("tpcrec");
    if (tracksTree == nullptr) {
      LOG(error) << "Error getting tree!";
      return;
    }

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
      LOG(info) << "Found " << mTracksOut.size() << " tracks! Processing...";
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

    uint8_t sectorIndex, rowIndex, side, state;
    uint32_t clusterIndex, trackCount = 0;
    // std::array<bool, maxRows> clMap{}, shMap{};

    mcTree->Branch("tracks_sector", &sectorIndex);
    mcTree->Branch("tracks_row", &rowIndex);
    mcTree->Branch("tracks_clusterIdx", &clusterIndex);
    mcTree->Branch("tracks_side", &side); // A = 0, C=1, both=2
    mcTree->Branch("tracks_count", &trackCount); // Index to keep track in file
    mcTree->Branch("tracks_state", &state); // State of the particle: 0 = valid, 1 = something is wrong

    for (auto track : mTracksOut) {
      for (int i = 0; i < track.getNClusterReferences(); i++) {
        if(track.hasASideClustersOnly()){
          side=0;
        }
        else if(track.hasCSideClustersOnly()){
          side=1;
        }
        else if(track.hasBothSidesClusters()){
          side=2;
        }
        else{
          side=3;
        }

        if(track.isValid()){
          state=0;
        }
        else{
          state=1;
        }

        o2::tpc::TrackTPC::getClusterReference(mClusRefTracksOut, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
        mcTree->Fill();
      }
      trackCount++;
    }

    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      LOG(info) << "TPC tracks reader done!";
    }
  };


  // Kinematics --> Raw information about the tracks including MC
  if (mode.find(std::string("kinematics")) != std::string::npos) {

    // std::vector<mcInfo_t> mMCInfos;
    // std::vector<GPUTPCMCInfoCol> mMCInfosCol;

    o2::steer::MCKinematicsReader mcReader(inFileKinematics.c_str());
    int nSimEvents = mcReader.getNEvents(0);
    // mMCInfos.resize(nSimEvents);
    std::vector<int> refId;

    auto dc = o2::steer::DigitizationContext::loadFromFile(inFileKinematics.c_str());
    auto evrec = dc->getEventRecords();

    TFile outputFile("mclabels_kinematics.root", "RECREATE");
    TTree* mcTree = new TTree("mcLabelsKinematics", "MC tree");

    float_t xpos, ypos, zpos, px, py, pz;
    int32_t idx;

    mcTree->Branch("kinematics_x", &xpos);
    mcTree->Branch("kinematics_y", &ypos);
    mcTree->Branch("kinematics_z", &zpos);
    mcTree->Branch("kinematics_px", &px);
    mcTree->Branch("kinematics_py", &py);
    mcTree->Branch("kinematics_pz", &pz);
    mcTree->Branch("kinematics_idx", &idx);

    // mMCInfosCol.resize(nSimEvents);
    for (int i = 0; i < nSimEvents; i++) {
      auto ir = evrec[i];
      auto ir0 = o2::raw::HBFUtils::Instance().getFirstIRofTF(ir);
      float timebin = (float)ir.differenceInBC(ir0) / o2::tpc::constants::LHCBCPERTIMEBIN;

      const std::vector<o2::MCTrack>& tracks = mcReader.getTracks(0, i);
      const std::vector<o2::TrackReference>& trackRefs = mcReader.getTrackRefsByEvent(0, i);

      refId.resize(tracks.size());
      std::fill(refId.begin(), refId.end(), -1);
      for (unsigned int j = 0; j < trackRefs.size(); j++) {
        if (trackRefs[j].getDetectorId() == o2::detectors::DetID::TPC) {
          int trkId = trackRefs[j].getTrackID();
          if (refId[trkId] == -1) {
            refId[trkId] = j;
          }
        }
      }

      // mMCInfosCol[i].first = mMCInfos.size();
      // mMCInfosCol[i].num = tracks.size();
      // mMCInfos.resize(mMCInfos.size() + tracks.size());
      for (unsigned int j = 0; j < tracks.size(); j++) {
        if (refId[j] >= 0) {
          const auto& trkRef = trackRefs[refId[j]];
          xpos = trkRef.X();
          ypos = trkRef.Y();
          zpos = trkRef.Z();
          px = trkRef.Px();
          py = trkRef.Py();
          pz = trkRef.Pz();
          // info.genRadius = std::sqrt(trk.GetStartVertexCoordinatesX() * trk.GetStartVertexCoordinatesX() + trk.GetStartVertexCoordinatesY() * trk.GetStartVertexCoordinatesY() + trk.GetStartVertexCoordinatesZ() * trk.GetStartVertexCoordinatesZ());
        } else {
          xpos = ypos = zpos = px = py = pz = 0;
          // info.genRadius = 0;
        }
        mcTree->Fill();
      }
    }

    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    if (verbose > 0) {
      LOG(info) << "TPC kinematics reader done!";
    }

  };

  // Digitizer -> Cluster-information based on MC labels, see O2/Detectors/TPC/simulation/src/Digitizer.cxx, O2/Steer/DigitizerWorkflow/src/TPCDigitizerSpec.cxx
  if (mode.find(std::string("digitizer")) != std::string::npos) {

    auto inputFile = TFile::Open(inFileDigitizer.c_str());

    TFile outputFile("mclabels_digits_raw.root", "RECREATE");
    TTree* mcTree = new TTree("mcLabelsDigitizer", "MC tree");

    // int sec, row, maxp, maxt;
    float sec, row, maxp, maxt, cogp, cogt, cogq, maxq;

    mcTree->Branch("digitizer_sector", &sec);
    mcTree->Branch("digitizer_row", &row);
    mcTree->Branch("digitizer_cogpad", &cogp);
    mcTree->Branch("digitizer_cogtime", &cogt);
    mcTree->Branch("digitizer_cogq", &cogq);
    mcTree->Branch("digitizer_maxpad", &maxp);
    mcTree->Branch("digitizer_maxtime", &maxt);
    mcTree->Branch("digitizer_maxq", &maxq);

    int current_sector = 0;
    for(auto&& keyAsObj : *(inputFile->GetListOfKeys())){

      auto key = (TKey*) keyAsObj;
      if(verbose>0){
        LOG(info) << "Processing sector " << current_sector << " (" << key->GetName() << ") ...";
      }

      auto digitizerSector = (TTree*)inputFile->Get(key->GetName());
      digitizerSector->SetBranchAddress("cluster_sector", &sec);
      digitizerSector->SetBranchAddress("cluster_row", &row);
      digitizerSector->SetBranchAddress("cluster_cog_pad", &cogp);
      digitizerSector->SetBranchAddress("cluster_cog_time", &cogt);
      digitizerSector->SetBranchAddress("cluster_cog_q", &cogq);
      digitizerSector->SetBranchAddress("cluster_max_pad", &maxp);
      digitizerSector->SetBranchAddress("cluster_max_time", &maxt);
      digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
      
      for(int i=0; i<digitizerSector->GetEntries(); i++){
        digitizerSector->GetEntry(i);
        mcTree->Write();
      }
      current_sector++;
    }

    inputFile->Close();
    mcTree->Write();
    delete mcTree;
    outputFile.Close();
    
    if (verbose > 0) {
      LOG(info) << "TPC digitizer reader done!";
    }

  }


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
      {"mode", VariantType::String, "digits,native,tracks,digitizer", {"Mode for running over tracks-file or digits-file: digits, native, tracks, kinematics and/or digitizer."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"infile-tracks", VariantType::String, "tpctracks.root", {"Input file name (tracks)"}},
      {"infile-kinematics", VariantType::String, "collisioncontext.root", {"Input file name (kinematics)"}},
      {"infile-digitizer", VariantType::String, "mclabels_digitizer.root", {"Input file name (digitizer)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}
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
  std::string mode = "digits,native,tracks,ideal_clusterizer";
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

    Int_t sector, row, pad, time;
    mcTree->Branch("digits_sector", &sector);
    mcTree->Branch("digits_row", &row);
    mcTree->Branch("digits_pad", &pad);
    mcTree->Branch("digits_time", &time);

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
          time = digit.getTimeStamp();
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
  if (mode.find(std::string("ideal_clusterizer")) != std::string::npos) {

    int sec, row, maxp, maxt, pcount, lab;
    float cogp, cogt, cogq, maxq;
    long elements = 0;

    std::vector<int> sectors, rows, maxps, maxts, point_count, mclabels;
    std::vector<float> cogps, cogts, cogqs, maxqs;

    for(int i = 0; i<36; i++){

      if(verbose>0){
        LOG(info) << "Processing ideal clusterizer, sector " << i << " ...";
      }
      std::stringstream tmp_file;
      tmp_file << "mclabels_digitizer_" << i << ".root";
      auto inputFile = TFile::Open(tmp_file.str().c_str());
      std::stringstream tmp_sec;
      tmp_sec << "sector_" << i;
      auto digitizerSector = (TTree*)inputFile->Get(tmp_sec.str().c_str());

      digitizerSector->SetBranchAddress("cluster_sector", &sec);
      digitizerSector->SetBranchAddress("cluster_row", &row);
      digitizerSector->SetBranchAddress("cluster_cog_pad", &cogp);
      digitizerSector->SetBranchAddress("cluster_cog_time", &cogt);
      digitizerSector->SetBranchAddress("cluster_cog_q", &cogq);
      digitizerSector->SetBranchAddress("cluster_max_pad", &maxp);
      digitizerSector->SetBranchAddress("cluster_max_time", &maxt);
      digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
      digitizerSector->SetBranchAddress("cluster_points", &pcount);

      for(int j=0; j<digitizerSector->GetEntries(); j++){
        try{
          digitizerSector->GetEntry(j);
          sectors.push_back(sec);
          rows.push_back(row);
          maxps.push_back(maxp);
          maxts.push_back(maxt);
          cogps.push_back(cogp);
          cogts.push_back(cogt);
          cogqs.push_back(cogq);
          maxqs.push_back(maxq);
          point_count.push_back(pcount);
          elements++;
        }
        catch(...){
          LOG(info) << "(Digitizer) Problem occured in sector " << i;
        }
      }

      inputFile->Close();

    }

    TFile* outputFileIdeal = new TFile("mclabels_ideal_clusters.root", "RECREATE");
    TTree* mcTreeIdeal = new TTree("mcLabelsDigitizer", "MC tree");

    mcTreeIdeal->Branch("digitizer_sector", &sec);
    mcTreeIdeal->Branch("digitizer_row", &row);
    mcTreeIdeal->Branch("digitizer_cogpad", &cogp);
    mcTreeIdeal->Branch("digitizer_cogtime", &cogt);
    mcTreeIdeal->Branch("digitizer_cogq", &cogq);
    mcTreeIdeal->Branch("digitizer_maxpad", &maxp);
    mcTreeIdeal->Branch("digitizer_maxtime", &maxt);
    mcTreeIdeal->Branch("digitizer_maxq", &maxq);

    for(int i = 0; i<elements; i++){
      sec = sectors[i];
      row = rows[i];
      maxp = maxps[i];
      maxt = maxts[i];
      cogp = cogps[i];
      cogt = cogts[i];
      cogq = cogqs[i];
      maxq = maxqs[i];
      mcTreeIdeal->Fill();
    }
    
    mcTreeIdeal->Write();
    delete mcTreeIdeal;
    outputFileIdeal->Close();
    delete outputFileIdeal;

    if (verbose > 0) {
      LOG(info) << "TPC ideal clusterizer reader done!";
    }

    sectors.clear(); rows.clear(); maxps.clear(); maxts.clear(); elements=0;


    // Full digits with mclabels
    // for(int i = 0; i<36; i++){
// 
    //   if(verbose>0){
    //     LOG(info) << "Processing ideal clusters, no selection and CoG, sector " << i << " ...";
    //   }
    //   std::stringstream tmp_file;
    //   tmp_file << "mclabels_ideal_full_" << i << ".root";
    //   auto inputFile = TFile::Open(tmp_file.str().c_str());
    //   std::stringstream tmp_sec;
    //   tmp_sec << "sector_" << i;
    //   auto digitsMcIdeal = (TTree*)inputFile->Get(tmp_sec.str().c_str());
// 
    //   digitsMcIdeal->SetBranchAddress("cluster_sector", &sec);
    //   digitsMcIdeal->SetBranchAddress("cluster_row", &row);
    //   digitsMcIdeal->SetBranchAddress("cluster_pad", &maxp);
    //   digitsMcIdeal->SetBranchAddress("cluster_time", &maxt);
    //   digitsMcIdeal->SetBranchAddress("cluster_label", &lab);
    //   digitsMcIdeal->SetBranchAddress("cluster_q", &maxq);
// 
    //   for(int j=0; j<digitsMcIdeal->GetEntries(); j++){
    //     try{
    //       digitsMcIdeal->GetEntry(j);
    //       sectors.push_back(sec);
    //       rows.push_back(row);
    //       maxps.push_back(maxp);
    //       maxts.push_back(maxt);
    //       maxqs.push_back(maxq);
    //       mclabels.push_back(lab);
    //       elements++;
    //     }
    //     catch(...){
    //       LOG(info) << "(Ideal full) Problem occured in sector " << i;
    //     }
    //   }
// 
    //   inputFile->Close();
// 
    // }
// 
    // TFile* outputFileIdealFull = new TFile("mclabels_ideal_digits.root", "RECREATE");
    // TTree* mcTreeIdealFull = new TTree("mcLabelsDigitizer", "MC tree");
// 
    // mcTreeIdealFull->SetBranchAddress("mcfull_sector", &sec);
    // mcTreeIdealFull->SetBranchAddress("mcfull_row", &row);
    // mcTreeIdealFull->SetBranchAddress("mcfull_pad", &maxp);
    // mcTreeIdealFull->SetBranchAddress("mcfull_time", &maxt);
    // mcTreeIdealFull->SetBranchAddress("mcfull_q", &maxq);
    // mcTreeIdealFull->SetBranchAddress("mcfull_label", &lab);
// 
    // for(int i = 0; i<elements; i++){
    //   sec = sectors[i];
    //   row = rows[i];
    //   maxp = maxps[i];
    //   maxt = maxts[i];
    //   maxq = maxqs[i];
    //   lab = mclabels[i];
    //   mcTreeIdealFull->Fill();
    // }
    // 
    // mcTreeIdealFull->Write();
    // delete mcTreeIdealFull;
    // outputFileIdealFull->Close();
    // delete outputFileIdealFull;
// 
    // if (verbose > 0) {
    //   LOG(info) << "TPC full ideal mc digits reader done!";
    // }
// 
    // sectors.clear(); rows.clear(); maxps.clear(); maxts.clear(); elements=0;
    

    // Cluster maxima found by the clusterizer
    for(int i = 0; i<36; i++){
      if(verbose>0){
        LOG(info) << "Processing clusterizer maxima, sector " << i << " ...";
      }
      std::stringstream tmp_file;
      tmp_file << "mclabels_clusterizer_sector_" << i << ".root";
      auto inputFile = TFile::Open(tmp_file.str().c_str());
      std::stringstream tmp_sec;
      auto digitizerSector = (TTree*)inputFile->Get("mcLabelsClusterizer");

      digitizerSector->SetBranchAddress("clusterizer_sector", &sec);
      digitizerSector->SetBranchAddress("clusterizer_row", &row);
      digitizerSector->SetBranchAddress("clusterizer_pad", &maxp);
      digitizerSector->SetBranchAddress("clusterizer_time", &maxt);

      for(int j=0; j<digitizerSector->GetEntries(); j++){
        try{
          digitizerSector->GetEntry(j);
          sectors.push_back(sec);
          rows.push_back(row);
          maxps.push_back(maxp);
          maxts.push_back(maxt);
          elements++;
        }
        catch(...){
          LOG(info) << "(Clustermax) Problem occured in sector " << i;
        }
      }

      inputFile->Close();

    }

    TFile* outputFileMax = new TFile("mclabels_clusterizer_max.root", "RECREATE");
    TTree* mcTreeMax = new TTree("mcLabelsClusterizer", "MC tree");

    mcTreeMax->Branch("clusterizer_sector", &sec);
    mcTreeMax->Branch("clusterizer_row", &row);
    mcTreeMax->Branch("clusterizer_maxpad", &maxp);
    mcTreeMax->Branch("clusterizer_maxtime", &maxt);

    for(int i = 0; i<elements; i++){
      sec = sectors[i];
      row = rows[i];
      maxp = maxps[i];
      maxt = maxts[i];
      mcTreeMax->Fill();
    }
    
    mcTreeMax->Write();
    delete mcTreeMax;
    outputFileMax->Close();
    delete outputFileMax;


    if (verbose > 0) {
      LOG(info) << "TPC cluster_max reader done!";
    }
  }

  // Creator for training_data
//  if (mode.find(std::string("training_data")) != std::string::npos) {
//
//    TFile* digitFile = TFile::Open(inFileDigits.c_str());
//    TTree* digitTree = (TTree*)digitFile->Get("o2sim");
//
//    float full_data[2][36][152][155][8000];
//
//    std::vector<std::string> labels = {"mCRU", "mRow", "mPad", "mTimeStamp", "mCharge"};
//
//    for(int i = 0; i<36; i++){
//      int sec, row, pad, time;
//      float charge;
//      std::string leafPath = fmt::format("TPCDigit_{:d}/", i);
//      digitTree->SetBranchAddress((leafPath + labels[0]).c_str(), &sec); // sec is actually saved as the CRU
//      digitTree->SetBranchAddress((leafPath + labels[1]).c_str(), &row);
//      digitTree->SetBranchAddress((leafPath + labels[2]).c_str(), &pad);
//      digitTree->SetBranchAddress((leafPath + labels[3]).c_str(), &time);
//      digitTree->SetBranchAddress((leafPath + labels[4]).c_str(), &charge);
//
//      for(int elem = 0; elem < digitTree->GetEntriesFast(); elem++){
//        digitTree->GetEntry(elem);
//        full_data[0][int(sec/10)][row][pad+6][time+6] = charge;
//      }
//    }
//    digitFile->Close();
//
//
//
//    int sec, row, maxp, maxt, pcount, index_1=0;
//    float cogp, cogt, cogq, maxq;
//    long elements = 0;
//
//    std::vector<int> sectors, rows, maxps, maxts, point_count;
//    std::vector<float> cogps, cogts, cogqs, maxqs;
//
//    for(int i = 0; i<36; i++){
//
//      if(verbose>0){
//        LOG(info) << "Processing ideal clusterizer, sector " << i << " ...";
//      }
//      std::stringstream tmp_file;
//      tmp_file << "mclabels_digitizer_" << i << ".root";
//      auto inputFile = TFile::Open(tmp_file.str().c_str());
//      std::stringstream tmp_sec;
//      tmp_sec << "sector_" << i;
//      auto digitizerSector = (TTree*)inputFile->Get(tmp_sec.str().c_str());
//
//      digitizerSector->SetBranchAddress("cluster_sector", &sec);
//      digitizerSector->SetBranchAddress("cluster_row", &row);
//      digitizerSector->SetBranchAddress("cluster_cog_pad", &cogp);
//      digitizerSector->SetBranchAddress("cluster_cog_time", &cogt);
//      digitizerSector->SetBranchAddress("cluster_cog_q", &cogq);
//      digitizerSector->SetBranchAddress("cluster_max_pad", &maxp);
//      digitizerSector->SetBranchAddress("cluster_max_time", &maxt);
//      digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
//      digitizerSector->SetBranchAddress("cluster_points", &pcount);
//
//      for(int j=0; j<digitizerSector->GetEntries(); j++){
//        digitizerSector->GetEntry(j);
//        sectors.push_back(sec);
//        rows.push_back(row);
//        maxps.push_back(maxp);
//        maxts.push_back(maxt);
//        cogps.push_back(cogp);
//        cogts.push_back(cogt);
//        cogqs.push_back(cogq);
//        maxqs.push_back(maxq);
//        point_count.push_back(pcount);
//        index_1++;
//
//        full_data[1][sec][row][maxp+6][maxt+6] = index_1;
//      }
//      inputFile->Close();
//    }
//
//
//    /// Create output grid of 11x11x(number of maxima)
//
//    unsigned long countermax = 0;
//    std::vector<std::vector<int>> grid_max;
//    std::vector<std::vector<float>> output_grid;
//    for(int i = 0; i<36; i++){
//      int sec, row, pad, time;
//      if(verbose>0){
//        LOG(info) << "Processing clusterizer maxima, sector " << i << " ...";
//      }
//      std::stringstream tmp_file;
//      tmp_file << "mclabels_clusterizer_sector_" << i << ".root";
//      auto inputFile = TFile::Open(tmp_file.str().c_str());
//      std::stringstream tmp_sec;
//      auto clustermaxSector = (TTree*)inputFile->Get("mcLabelsClusterizer");
//
//      clustermaxSector->SetBranchAddress("clusterizer_sector", &sec);
//      clustermaxSector->SetBranchAddress("clusterizer_row", &row);
//      clustermaxSector->SetBranchAddress("clusterizer_pad", &pad);
//      clustermaxSector->SetBranchAddress("clusterizer_time", &time);
//
//      for(int j=0; j<clustermaxSector->GetEntries(); j++){
//        clustermaxSector->GetEntry(j);
//        grid_max.push_back(std::vector<int>{sec,row,pad,time});
//        output_grid.push_back(std::vector<float>(121,0));
//        for(int g1=-5; g1<6; g1++){
//          for(int g2=-5; g2<6; g2++){
//            output_grid[j][11*g1 + g2] = full_data[0][sec][row][pad+g1+6][time+g2+6];
//          }
//        }
//        countermax++;
//      }
//      inputFile->Close();
//    }
//
//
//    TFile* outputFileTr = new TFile("training_data.root", "RECREATE");
//    TTree* mcTreeTr = new TTree("training_data", "MC tree");
//
//    std::vector<float> grid;
//
//    mcTreeTr->Branch("training_sector", &sec);
//    mcTreeTr->Branch("training_row", &row);
//    mcTreeTr->Branch("training_cog_pad", &cogp);
//    mcTreeTr->Branch("training_cog_time", &cogt);
//    mcTreeTr->Branch("training_cog_q", &cogq);
//    mcTreeTr->Branch("training_max_pad", &maxp);
//    mcTreeTr->Branch("training_max_time", &maxt);
//    mcTreeTr->Branch("training_max_q", &maxq);
//    for(int i = 0; i<121; i++){
//      std::stringstream tmp_branch;
//      tmp_branch << "grid_pad_" << int(i/11) << "_time_" << int(i%11) << ".root";
//      mcTreeTr->Branch(tmp_branch.str().c_str(), &(grid[i]));
//    }
//
//    int idx_1 = 0;
//    for(int i = 0; i<countermax; i++){
//      for(int j = 0; j<121; j++){
//        grid[i] = output_grid[j][i];
//      }
//      idx_1 = full_data[1][grid_max[i][0]][grid_max[i][1]][grid_max[i][2]+6][grid_max[i][3]+6];
//      if(idx_1!=0){
//        sec = sectors[idx_1];
//        row = rows[idx_1];
//        maxp = maxps[idx_1];
//        maxt = maxts[idx_1];
//        cogp = cogps[idx_1];
//        cogt = cogts[idx_1];
//        cogq = cogqs[idx_1];
//        maxq = maxqs[idx_1];
//      }
//      else{
//        sec = -1;
//        row = -1;
//        maxp = -1;
//        maxt = -1;
//        cogp = -1;
//        cogt = -1;
//        cogq = -1;
//        maxq = -1;
//      }
//      mcTreeTr->Fill();
//    }
//  }

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
      {"mode", VariantType::String, "digits,native,ideal_clusterizer", {"Mode for running over tracks-file or digits-file: digits, native, tracks, kinematics and/or digitizer."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"infile-tracks", VariantType::String, "tpctracks.root", {"Input file name (tracks)"}},
      {"infile-kinematics", VariantType::String, "collisioncontext.root", {"Input file name (kinematics)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}
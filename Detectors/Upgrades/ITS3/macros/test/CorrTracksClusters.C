// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TROOT.h>
#include "TEfficiency.h"
#include <TFile.h>
#include <TH2F.h>
#include <TTree.h>

#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITS/TrackITS.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/TrackReference.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITS3Base/SegmentationMosaix.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#endif

using namespace std;
using namespace o2::itsmft;
using namespace o2::its;
using SegmentationIB = o2::its3::SegmentationMosaix;
using SegmentationOB = o2::itsmft::SegmentationAlpide;
static constexpr int kNLayer = 7;
static constexpr int INVALID_INT = -99;
static constexpr float INVALID_FLOAT = -99.f;

//______________________________________________________________________________
// ParticleInfo structure
struct ParticleInfo {
  int event{};
  int pdg{};
  float pt{};
  float recpt{};
  float eta{};
  float phi{};
  float pvx{};
  float pvy{};
  float pvz{};
  float dcaxy{};
  float dcaz{};
  int mother{};
  int first{};
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = false;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  std::array<int, kNLayer> clusterSize;
  std::array<int, kNLayer> clusterPattern;
  std::array<float, kNLayer> clusterLocX;
  std::array<float, kNLayer> clusterLocZ;
  std::array<float, kNLayer> hitLocX;
  std::array<float, kNLayer> hitLocY;
  std::array<float, kNLayer> hitLocZ;
  o2::its::TrackITS track;
  ParticleInfo()
  {
    clusterSize.fill(INVALID_INT);
    clusterPattern.fill(INVALID_INT);
    clusterLocX.fill(INVALID_FLOAT);
    clusterLocZ.fill(INVALID_FLOAT);
    hitLocX.fill(INVALID_FLOAT);
    hitLocY.fill(INVALID_FLOAT);
    hitLocZ.fill(INVALID_FLOAT);
  }
};

//______________________________________________________________________________
// Convert curved local coordinates to flat coordinates
void CurvedLocalToFlat(o2::math_utils::Point3D<float>& point, const SegmentationIB& seg)
{
  float xFlat = 0.f, yFlat = 0.f;
  seg.curvedToFlat(point.X(), point.Y(), xFlat, yFlat);
  point.SetXYZ(xFlat, yFlat, point.Z());
}

//______________________________________________________________________________
// Resolve pattern from patternID and iterator
bool resolvePattern(const o2::itsmft::CompClusterExt& cluster,
                    decltype(std::declval<std::vector<unsigned char>>().cbegin())& pattIt,
                    const o2::its3::TopologyDictionary& dict,
                    bool isIB,
                    o2::itsmft::ClusterPattern& pattOut)
{
  auto pattID = cluster.getPatternID();
  if (pattID != o2::itsmft::CompCluster::InvalidPatternID) {
    if (!dict.getSize(true) && !dict.getSize(false)) {
      LOGP(error, "Encountered non-invalid pattern ID {} but dictionary is missing!", pattID);
      return false;
    }
    if (dict.isGroup(pattID, isIB)) {
      pattOut.acquirePattern(pattIt);
    } else {
      pattOut = dict.getPattern(pattID, isIB);
    }
  } else {
    pattOut.acquirePattern(pattIt);
  }
  return true;
}

//______________________________________________________________________________
// Function to analyze reconstructed tracks
void analyzeRecoTracks(TTree* recTree,
                       const std::vector<o2::its::TrackITS>* recArr,
                       const std::vector<o2::MCCompLabel>* trkLabArr,
                       std::vector<std::vector<ParticleInfo>>& info,
                       float bz,
                       ULong_t& unaccounted,
                       ULong_t& good,
                       ULong_t& fakes,
                       ULong_t& total)
{
  unaccounted = good = fakes = total = 0;
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // reco tracks frames
    if (recTree->GetEvent(frame) == 0)
      continue;
    total += trkLabArr->size();
    for (unsigned int iTrack = 0; iTrack < trkLabArr->size(); ++iTrack) {
      auto lab = trkLabArr->at(iTrack);
      if (!lab.isSet()) {
        unaccounted++;
        continue;
      }
      int trackID, evID, srcID;
      bool fake;
      lab.get(trackID, evID, srcID, fake);
      if (evID < 0 || evID >= (int)info.size()) {
        unaccounted++;
        continue;
      }
      if (trackID < 0 || trackID >= (int)info[evID].size()) {
        unaccounted++;
        continue;
      }
      info[evID][trackID].isReco += !fake;
      info[evID][trackID].isFake += fake;
      if (recArr->at(iTrack).isBetter(info[evID][trackID].track, 1.e9)) {
        info[evID][trackID].storedStatus = fake;
        info[evID][trackID].track = recArr->at(iTrack);
        float ip[2]{0., 0.};
        info[evID][trackID].track.getImpactParams(info[evID][trackID].pvx,
                                                  info[evID][trackID].pvy,
                                                  info[evID][trackID].pvz, bz, ip);
        info[evID][trackID].dcaxy = ip[0];
        info[evID][trackID].dcaz = ip[1];
        info[evID][trackID].recpt = info[evID][trackID].track.getPt();
      }
      fakes += static_cast<ULong_t>(fake);
      good += static_cast<ULong_t>(!fake);
    }
  }
  LOGP(info, "** Some statistics:");
  LOGP(info, "\t- Total number of tracks: {}", total);
  LOGP(info, "\t- Total number of tracks not corresponding to particles: {} ({:.2f}%)", unaccounted, unaccounted * 100. / total);
  LOGP(info, "\t- Total number of fakes: {} ({:.2f}%)", fakes, fakes * 100. / total);
  LOGP(info, "\t- Total number of good: {} ({:.2f}%)", good, good * 100. / total);
}

//______________________________________________________________________________
// Read and map hit information from hitTree
void mapHitsForMCEvents(TTree* hitTree,
                        std::vector<std::vector<o2::itsmft::Hit>*>& hitVecPool,
                        std::vector<std::unordered_map<uint64_t, int>>& mc2hitVec,
                        const std::vector<int>& mcEvMin,
                        const std::vector<int>& mcEvMax,
                        size_t nROFRec)
{
  for (unsigned int irof = 0; irof < nROFRec; irof++) {
    for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
      if (!hitVecPool[im]) {
        hitTree->SetBranchAddress("IT3Hit", &hitVecPool[im]);
        hitTree->GetEntry(im);
        auto& mc2hit = mc2hitVec[im];
        const auto* hitArray = hitVecPool[im];
        for (int ih = hitArray->size(); ih--;) {
          const auto& hit = (*hitArray)[ih];
          uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
          mc2hit.emplace(key, ih);
        }
      }
    }
  }
}

//______________________________________________________________________________
// Load geometry and magnetic field information
void loadGeometryAndField(const std::string& magfile, const std::string& inputGeom, float& bz, o2::its::GeometryTGeo*& gman)
{
  o2::base::Propagator::initFieldFromGRP(magfile);
  bz = o2::base::Propagator::Instance()->getNominalBz();
  o2::base::GeometryManager::loadGeometry(inputGeom);
  gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                 o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G));
}

//______________________________________________________________________________
// Load topology dictionary
void loadTopologyDictionary(const std::string& dictfile, o2::its3::TopologyDictionary& dict)
{
  std::ifstream iofile(dictfile);
  if (iofile.good()) {
    LOG(info) << "Running with dictionary: " << dictfile;
    dict.readFromFile(dictfile);
  } else {
    LOG(info) << "Dictionary file not found: " << dictfile;
  }
}

//______________________________________________________________________________
// Build ROF
void buildMcEvRangePerROF(const std::vector<o2::itsmft::MC2ROFRecord>& mc2rofVec,
                          size_t nROFRec,
                          std::vector<int>& mcEvMin,
                          std::vector<int>& mcEvMax)
{
  for (size_t imc = 0; imc < mc2rofVec.size(); ++imc) {
    const auto& mc2rof = mc2rofVec[imc];
    if (mc2rof.rofRecordID < 0)
      continue;
    for (size_t i = mc2rof.minROF; i <= mc2rof.maxROF; ++i) {
      if (i >= nROFRec)
        continue;
      mcEvMin[i] = std::min(mcEvMin[i], static_cast<int>(imc));
      mcEvMax[i] = std::max(mcEvMax[i], static_cast<int>(imc));
    }
  }
}

//______________________________________________________________________________
// Load Hits data
void prepareHitAccess(const std::string& hitfile,
                      TTree*& hitTree,
                      std::vector<std::vector<o2::itsmft::Hit>*>& hitVecPool,
                      std::vector<std::unordered_map<uint64_t, int>>& mc2hitVec)
{
  TFile* fHit = TFile::Open(hitfile.data());
  hitTree = (TTree*)fHit->Get("o2sim");
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);
}

void loadCluster(const std::string& clusfile,
                 TTree*& clusTree,
                 std::vector<o2::itsmft::CompClusterExt>*& clusArr,
                 o2::dataformats::MCTruthContainer<o2::MCCompLabel>*& clusLabArr,
                 std::vector<o2::itsmft::MC2ROFRecord>& mc2rofVec,
                 std::vector<unsigned char>*& patternsPtr,
                 std::vector<o2::itsmft::ROFRecord>& rofRecVec)
{
  // Open file and let it persist
  TFile* fileC = TFile::Open(clusfile.data());
  // Get tree
  clusTree = dynamic_cast<TTree*>(fileC->Get("o2sim"));
  // Cluster array
  clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  // MC truth
  clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
  clusTree->SetBranchAddress("ITSClusterPatt", &patternsPtr);
  // ROF records
  std::vector<o2::itsmft::ROFRecord>* rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);
  // MC2ROF mapping
  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofVecP = &mc2rofVec;
  clusTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  clusTree->GetEntry(0);
  // After setting all branch addresses, trigger preload of the first entr
}

//______________________________________________________________________________
// Load Reconstructed Tracks data
void loadRecoTracks(const std::string& tracfile,
                    TTree*& recTree,
                    std::vector<o2::its::TrackITS>*& recArr,
                    std::vector<o2::MCCompLabel>*& trkLabArr)
{
  TFile* fTrk = TFile::Open(tracfile.data());
  recTree = (TTree*)fTrk->Get("o2sim");
  recTree->SetBranchAddress("ITSTrack", &recArr);
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);
}

//______________________________________________________________________________
// Load MC Track information
void loadMCTrackInfo(const std::string& kinefile,
                     std::vector<std::vector<ParticleInfo>>& info,
                     std::vector<o2::MCTrack>*& mcArr,
                     o2::dataformats::MCEventHeader*& mcEvent,
                     TTree*& mcTree)
{
  TFile* kineFile = TFile::Open(kinefile.data());
  mcTree = (TTree*)kineFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0);
  mcTree->SetBranchStatus("MCTrack*", 1);
  mcTree->SetBranchStatus("MCEventHeader*", 1);
  mcTree->SetBranchAddress("MCTrack", &mcArr);
  mcTree->SetBranchAddress("MCEventHeader.", &mcEvent);

  int nev = mcTree->GetEntriesFast();
  info.resize(nev);
  for (int n = 0; n < nev; n++) {
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    for (unsigned int mcI = 0; mcI < mcArr->size(); ++mcI) {
      auto part = mcArr->at(mcI);
      info[n][mcI].pvx = mcEvent->GetX();
      info[n][mcI].pvy = mcEvent->GetY();
      info[n][mcI].pvz = mcEvent->GetZ();
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
    }
  }
}

//______________________________________________________________________________
// Main function CorrTracksClusters
void CorrTracksClusters(const std::string& tracfile = "o2trac_its.root",
                        const std::string& clusfile = "o2clus_its.root",
                        const std::string& kinefile = "o2sim_Kine.root",
                        const std::string& magfile = "o2sim_grp.root",
                        const std::string& hitfile = "o2sim_HitsIT3.root",
                        const std::string& dictfile = "IT3dictionary.root",
                        const std::string& inputGeom = "",
                        bool batch = false)
{
  gROOT->SetBatch(batch);

  // Geo and Field
  LOGP(info, "Geo and Field loading");
  float bz{0.f};
  o2::its::GeometryTGeo* gman = nullptr;
  loadGeometryAndField(magfile, inputGeom, bz, gman);
  LOGP(info, "Finished Geo and Field loading");

  // MC tracks
  LOGP(info, "MC Track Info loading");
  std::vector<o2::MCTrack>* mcArr = nullptr;
  o2::dataformats::MCEventHeader* mcEvent = nullptr;
  TTree* mcTree = nullptr;
  std::vector<std::vector<ParticleInfo>> info;
  loadMCTrackInfo(kinefile, info, mcArr, mcEvent, mcTree);
  LOGP(info, "Finished MC Track Info loading");

  // Reconstructed tracks
  LOGP(info, "Reco Tracks loading");
  TTree* recTree = nullptr;
  std::vector<o2::its::TrackITS>* recArr = nullptr;
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  loadRecoTracks(tracfile, recTree, recArr, trkLabArr);
  LOGP(info, "Finished Reco Tracks loading");

  // Run analyzeRecoTracks
  LOGP(info, "Track analysis (analyzeRecoTracks)");
  ULong_t unaccounted{0}, good{0}, fakes{0}, total{0};
  analyzeRecoTracks(recTree, recArr, trkLabArr, info, bz, unaccounted, good, fakes, total);
  LOGP(info, "Finished track analysis (analyzeRecoTracks)");

  // Topology dictionary
  LOGP(info, "Topology Dictionary loading");
  o2::its3::TopologyDictionary dict;
  loadTopologyDictionary(dictfile, dict);
  LOGP(info, "Finished Topology Dictionary loading");

  // Clusters
  LOGP(info, "Cluster Data loading");
  TTree* clusTree = nullptr;
  std::vector<o2::itsmft::CompClusterExt>* clusArr = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<unsigned char>* patternsPtr = nullptr;
  std::vector<o2::itsmft::MC2ROFRecord> mc2rofVec;
  std::vector<o2::itsmft::ROFRecord> rofRecVec;
  loadCluster(clusfile, clusTree, clusArr, clusLabArr, mc2rofVec, patternsPtr, rofRecVec);
  LOGP(info, "Finished Cluster Data loading");
  // clusTree->GetEntry(0);

  // Hits
  LOGP(info, "Hits loading");
  TTree* hitTree = nullptr;
  std::vector<std::vector<o2::itsmft::Hit>*> hitVecPool;
  std::vector<std::unordered_map<uint64_t, int>> mc2hitVec;
  prepareHitAccess(hitfile, hitTree, hitVecPool, mc2hitVec);
  LOGP(info, "Finished Hits loading");

  // Build min and max MC events used by each ROF
  LOGP(info, "Building MC event ranges");
  std::vector<int> mcEvMin, mcEvMax;
  mcEvMin.assign(rofRecVec.size(), hitTree->GetEntries());
  mcEvMax.assign(rofRecVec.size(), -1);
  buildMcEvRangePerROF(mc2rofVec, rofRecVec.size(), mcEvMin, mcEvMax);
  LOGP(info, "Initial MC event ranges built");
  unsigned int nROFRec = rofRecVec.size();

  // Map hits for MC events
  LOGP(info, "Map hits for MC events");
  mapHitsForMCEvents(hitTree, hitVecPool, mc2hitVec, mcEvMin, mcEvMax, nROFRec);
  LOGP(info, "Mapped hits for MC events");

  // Run cluster particle matching
  auto pattIt = patternsPtr->cbegin();
  for (unsigned int iClus = 0; iClus < clusArr->size(); ++iClus) {
    auto lab = (clusLabArr->getLabels(iClus))[0];
    const auto& c = (*clusArr)[iClus];
    // Ensure pattIt is advanced even if cluster is skipped
    if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect()) {
      if (c.getPatternID() == CompCluster::InvalidPatternID) {
        o2::itsmft::ClusterPattern::skipPattern(pattIt);
      }
      continue;
    }

    int trackID{0}, evID{0}, srcID{0};
    bool fake{false};
    lab.get(trackID, evID, srcID, fake);
    if (evID < 0 || static_cast<size_t>(evID) >= info.size() || trackID < 0 || static_cast<size_t>(trackID) >= info[evID].size()) {
      if (c.getPatternID() == CompCluster::InvalidPatternID) {
        o2::itsmft::ClusterPattern::skipPattern(pattIt);
      }
      continue;
    }
    UShort_t chipID = c.getSensorID();
    int layer = gman->getLayer(chipID);
    bool isIB = layer < 3;
    info[evID][trackID].clusters |= 1 << layer;

    o2::math_utils::Point3D<float> clusterPos;
    int clusterSize;
    auto pattID = c.getPatternID();
    o2::itsmft::ClusterPattern patt;
    if (!resolvePattern(c, pattIt, dict, isIB, patt)) {
      continue;
    }
    clusterSize = patt.getNPixels();
    clusterPos = dict.getClusterCoordinates(c, patt, false);

    if (isIB) {
      CurvedLocalToFlat(clusterPos, SegmentationIB(layer));
    }

    info[evID][trackID].clusterSize[layer] = clusterSize;
    info[evID][trackID].clusterPattern[layer] = pattID;
    info[evID][trackID].clusterLocX[layer] = clusterPos.X();
    info[evID][trackID].clusterLocZ[layer] = clusterPos.Z();

    const auto& mc2hit = mc2hitVec[lab.getEventID()];
    const auto* hitArray = hitVecPool[lab.getEventID()];
    uint64_t key = (uint64_t(trackID) << 32) + c.getSensorID();
    auto hitIt = mc2hit.find(key);
    if (hitIt == mc2hit.end())
      continue;
    const auto& hit = (*hitArray)[hitIt->second];

    auto hitLocSta = gman->getMatrixL2G(chipID) ^ hit.GetPosStart();
    auto hitLocEnd = gman->getMatrixL2G(chipID) ^ hit.GetPos();

    if (isIB) {
      CurvedLocalToFlat(hitLocSta, SegmentationIB(layer));
      CurvedLocalToFlat(hitLocEnd, SegmentationIB(layer));
      info[evID][trackID].hitLocX[layer] = 0.5f * (hitLocSta.X() + hitLocEnd.X());
      info[evID][trackID].hitLocY[layer] = 0.5f * (hitLocSta.Y() + hitLocEnd.Y());
      info[evID][trackID].hitLocZ[layer] = 0.5f * (hitLocSta.Z() + hitLocEnd.Z());
    } else {
      auto x0 = hitLocSta.X(), dx = hitLocEnd.X() - x0;
      auto y0 = hitLocSta.Y(), dy = hitLocEnd.Y() - y0;
      auto z0 = hitLocSta.Z(), dz = hitLocEnd.Z() - z0;
      auto r = (0.5f * (SegmentationOB::SensorLayerThickness - SegmentationOB::SensorLayerThicknessEff) - y0) / dy;
      info[evID][trackID].hitLocX[layer] = x0 + r * dx;
      info[evID][trackID].hitLocY[layer] = y0 + r * dy;
      info[evID][trackID].hitLocZ[layer] = z0 + r * dz;
    }
  }

  LOGP(info, "Finished cluster-to-particle matching");

  // The following part generates statistical histograms and outputs a TTree
  int nb = 100;
  double xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
  double a = std::log(ptcuth / ptcutl) / nb;
  for (int i = 0; i <= nb; ++i) {
    xbins[i] = ptcutl * std::exp(i * a);
  }
  auto* h_pt_num = new TH1D("h_pt_num", ";#it{p}_{T} (GeV/#it{c});Number of tracks", nb, xbins);
  auto* h_pt_den = new TH1D("h_pt_den", ";#it{p}_{T} (GeV/#it{c});Number of generated primary particles", nb, xbins);
  auto* h_pt_eff = new TEfficiency("h_pt_eff", "Tracking Efficiency;#it{p}_{T} (GeV/#it{c});Eff.", nb, xbins);

  auto* h_eta_num = new TH1D("h_eta_num", ";#it{#eta};Number of tracks", 60, -3, 3);
  auto* h_eta_den = new TH1D("h_eta_den", ";#it{#eta};Number of generated particles", 60, -3, 3);
  auto* h_eta_eff = new TEfficiency("h_eta_eff", "Tracking Efficiency;#it{#eta};Eff.", 60, -3, 3);

  auto* h_phi_num = new TH1D("h_phi_num", ";#varphi;Number of tracks", 360, 0., 2 * TMath::Pi());
  auto* h_phi_den = new TH1D("h_phi_den", ";#varphi;Number of generated particles", 360, 0., 2 * TMath::Pi());
  auto* h_phi_eff = new TEfficiency("h_phi_eff", "Tracking Efficiency;#varphi;Eff.", 360, 0., 2 * TMath::Pi());

  auto* h_pt_fake = new TH1D("h_pt_fake", ";#it{p}_{T} (GeV/#it{c});Number of fake tracks", nb, xbins);
  auto* h_pt_multifake = new TH1D("h_pt_multifake", ";#it{p}_{T} (GeV/#it{c});Number of multifake tracks", nb, xbins);
  auto* h_pt_clones = new TH1D("h_pt_clones", ";#it{p}_{T} (GeV/#it{c});Number of cloned tracks", nb, xbins);
  auto* h_dcaxy_vs_pt = new TH2D("h_dcaxy_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{xy} (#mum)", nb, xbins, 2000, -500., 500.);
  auto* h_dcaxy_vs_eta = new TH2D("h_dcaxy_vs_eta", ";#it{#eta};DCA_{xy} (#mum)", 60, -3, 3, 2000, -500., 500.);
  auto* h_dcaxy_vs_phi = new TH2D("h_dcaxy_vs_phi", ";#varphi;DCA_{xy} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);
  auto* h_dcaz_vs_pt = new TH2D("h_dcaz_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{z} (#mum)", nb, xbins, 2000, -500., 500.);
  auto* h_dcaz_vs_eta = new TH2D("h_dcaz_vs_eta", ";#it{#eta};DCA_{z} (#mum)", 60, -3, 3, 2000, -500., 500.);
  auto* h_dcaz_vs_phi = new TH2D("h_dcaz_vs_phi", ";#varphi;DCA_{z} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);
  auto* h_chi2 = new TH2D("h_chi2", ";#it{p}_{T} (GeV/#it{c});#chi^{2};Number of tracks", nb, xbins, 200, 0., 100.);

  for (auto& evInfo : info) {
    for (auto& part : evInfo) {
      if ((part.clusters & 0x7f) != 0x7f) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters
        // != 0x1f << 2 && part.clusters != 0x0f && part.clusters != 0x0f << 1
        // && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        continue;
      }
      if (!part.isPrimary) {
        continue;
      }

      h_pt_den->Fill(part.pt);
      h_eta_den->Fill(part.eta);
      h_phi_den->Fill(part.phi);

      if (part.isReco != 0u) {
        h_pt_num->Fill(part.pt);
        h_eta_num->Fill(part.eta);
        h_phi_num->Fill(part.phi);
        if (std::abs(part.eta) < 0.5) {
          h_dcaxy_vs_pt->Fill(part.pt, part.dcaxy * 10000);
          h_dcaz_vs_pt->Fill(part.pt, part.dcaz * 10000);
        }
        h_dcaz_vs_eta->Fill(part.eta, part.dcaz * 10000);
        h_dcaxy_vs_eta->Fill(part.eta, part.dcaxy * 10000);
        h_dcaxy_vs_phi->Fill(part.phi, part.dcaxy * 10000);
        h_dcaz_vs_phi->Fill(part.phi, part.dcaz * 10000);

        h_chi2->Fill(part.pt, part.track.getChi2());

        if (part.isReco > 1) {
          for (int _i{0}; _i < part.isReco - 1; ++_i) {
            h_pt_clones->Fill(part.pt);
          }
        }
      }
      if (part.isFake != 0u) {
        h_pt_fake->Fill(part.pt);
        if (part.isFake > 1) {
          for (int _i{0}; _i < part.isFake - 1; ++_i) {
            h_pt_multifake->Fill(part.pt);
          }
        }
      }
    }
  }

  LOGP(info, "Streaming output TTree to file");
  TFile file("CorrTracksClusters.root", "recreate");
  TTree tree("ParticleInfo", "ParticleInfo");
  ParticleInfo pInfo;
  tree.Branch("particle", &pInfo);
  for (auto& event : info) {
    for (auto& part : event) {
      int nCl{0};
      for (unsigned int bit{0}; bit < sizeof(pInfo.clusters) * 8; ++bit) {
        nCl += bool(part.clusters & (1 << bit));
      }
      if (nCl < 3) {
        continue;
      }
      pInfo = part;
      tree.Fill();
    }
  }
  tree.Write();
  h_pt_num->Write();
  h_eta_num->Write();
  h_phi_num->Write();
  h_pt_den->Write();
  h_eta_den->Write();
  h_phi_den->Write();
  h_pt_multifake->Write();
  h_pt_fake->Write();
  h_dcaxy_vs_pt->Write();
  h_dcaz_vs_pt->Write();
  h_dcaxy_vs_eta->Write();
  h_dcaxy_vs_phi->Write();
  h_dcaz_vs_eta->Write();
  h_dcaz_vs_phi->Write();
  h_pt_clones->Write();
  h_chi2->Write();

  h_pt_eff->SetTotalHistogram(*h_pt_den, "");
  h_pt_eff->SetPassedHistogram(*h_pt_num, "");
  h_pt_eff->SetTitle("Tracking Efficiency;#it{p}_{T} (GeV/#it{c});Eff.");
  h_pt_eff->Write();

  h_phi_eff->SetTotalHistogram(*h_phi_den, "");
  h_phi_eff->SetPassedHistogram(*h_phi_num, "");
  h_phi_eff->SetTitle("Tracking Efficiency;#varphi;Eff.");
  h_phi_eff->Write();

  h_eta_eff->SetTotalHistogram(*h_eta_den, "");
  h_eta_eff->SetPassedHistogram(*h_eta_num, "");
  h_eta_eff->SetTitle("Tracking Efficiency;#it{#eta};Eff.");
  h_eta_eff->Write();

  file.Close();
  LOGP(info, "Finished streaming output TTree to file");
  LOGP(info, "done.");
}

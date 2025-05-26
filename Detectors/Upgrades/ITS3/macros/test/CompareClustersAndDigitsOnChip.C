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

/// \file CompareClustersAndDigitsOnChip.C
/// \brief Macro to compare ITS3 clusters and digits on a pixel array,

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TNtuple.h>
#include <TROOT.h>
#include <TString.h>
#include <TArrow.h>
#include <TStyle.h>
#include <TTree.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <map>
#endif

#define ENABLE_UPGRADES
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITS3Base/SegmentationMosaix.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"

struct Data {
  TH2F* pixelArray;
  TGraph* hitS;
  TGraph* hitM;
  TGraph* hitE;
  TGraph* clusS;
  TGraph* cog;
  TLegend* leg;
  std::vector<TBox*>* vClusBox;
  void clear()
  {
    delete pixelArray;
    delete hitS;
    delete hitM;
    delete hitE;
    delete clusS;
    delete cog;
    delete leg;
    for (auto& b : *vClusBox) {
      delete b;
    }
    delete vClusBox;
  }
};

void CompareClustersAndDigitsOnChip(std::string clusfile = "o2clus_its.root",
                                    std::string digifile = "it3digits.root",
                                    std::string dictfile = "",
                                    std::string hitfile = "o2sim_HitsIT3.root",
                                    std::string inputGeom = "o2sim_geometry.root",
                                    bool batch = true)
{
  TH1::AddDirectory(kFALSE);
  gROOT->SetBatch(batch);
  gStyle->SetPalette(kRainBow);
  gStyle->SetOptStat(0);

  using namespace o2::base;
  using namespace o2::its;
  using o2::itsmft::Hit;
  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::CompClusterExt;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector
  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  std::array<o2::its3::SegmentationMosaix, 3> mMosaixSegmentations{0, 1, 2};

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                                                 o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms
  const int nChips = gman->getNumberOfChips();

  LOGP(info, "Total number of chips is {} in ITS3 (IB and OB)", nChips);

  // Create all plots
  LOGP(info, "Selecting chips to be visualised");
  std::set<int> selectedChips;
  std::map<std::string, std::vector<int>> chipGroups;

  for (int chipID{0}; chipID < nChips; ++chipID) {
    TString tpath = gman->getMatrixPath(chipID);
    std::string path = tpath.Data();

    std::vector<std::string> tokens;
    std::istringstream iss(path);
    std::string token;
    while (std::getline(iss, token, '/')) {
      if (!token.empty()) {
        tokens.push_back(token);
      }
    }

    std::string segmentName, staveName, carbonFormName;
    for (const auto& t : tokens) {
      if (t.find("ITS3Segment") != std::string::npos)
        segmentName = t;
      if (t.find("ITSUStave") != std::string::npos)
        staveName = t;
      if (t.find("ITS3CarbonForm") != std::string::npos)
        carbonFormName = t;
    }

    std::string groupKey;
    if (!segmentName.empty()) {
      groupKey = segmentName + "_" + carbonFormName;
    } else if (!staveName.empty()) {
      groupKey = staveName;
    } else {
      continue;
    }

    chipGroups[groupKey].push_back(chipID);
  }

  LOGP(info, "From each IB Segment or OB Stave, 10 chipIDs are uniformly selected");
  LOGP(info, "Selected chipID: ");
  for (auto& [groupName, ids] : chipGroups) {
    std::vector<int> sampled;
    if (ids.size() <= 10) {
      for (auto id : ids) {
        selectedChips.insert(id);
        sampled.push_back(id);
      }
    } else {
      for (int i{0}; i < 10; ++i) {
        int idx = i * (ids.size() - 1) / 9; // 9 intervals for 10 points
        int id = ids[idx];
        if (selectedChips.insert(id).second) {
          sampled.push_back(id);
        }
      }
    }

    std::ostringstream oss;
    std::string topOrBot = "N/A";
    std::smatch match;
    std::regex rgxSegment(R"(Segment(\d+)_(\d+)_ITS3CarbonForm\d+_(\d+))");
    std::regex rgxStave(R"(Stave(\d+)_(\d+))");
    if (std::regex_search(groupName, match, rgxSegment)) {
      int layer = std::stoi(match[1]);
      int segment = std::stoi(match[2]);
      int carbonForm = std::stoi(match[3]);
      topOrBot = (carbonForm == 0 ? "TOP" : "BOT");
      oss << topOrBot << " segment " << segment << " at layer " << layer << ": ";
    } else if (std::regex_search(groupName, match, rgxStave)) {
      int layer = std::stoi(match[1]);
      int stave = std::stoi(match[2]);
      oss << "Stave " << stave << " at layer " << layer << ": ";
    } else {
      LOGP(error, "Cannot select the correct chipID in OB or IB");
      return;
    }
    for (auto id : sampled) {
      oss << id << " ";
    }
    LOG(info) << oss.str();
  }
  LOGP(info, "{} selected chips will be visualized and analyzed.", chipGroups.size());

  // Hits
  TFile fileH(hitfile.data());
  auto* hitTree = dynamic_cast<TTree*>(fileH.Get("o2sim"));
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("IT3Hit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("IT3Digit", &digArr);
  o2::dataformats::IOMCTruthContainerView* plabels = nullptr;
  digTree->SetBranchAddress("IT3DigitMCTruth", &plabels);

  // Clusters
  TFile fileC(clusfile.data());
  auto* clusTree = dynamic_cast<TTree*>(fileC.Get("o2sim"));
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch != nullptr) {
    pattBranch->SetAddress(&patternsPtr);
  }

  // Topology dictionary
  o2::its3::TopologyDictionary dict;
  bool hasAvailableDict = false;
  if (!dictfile.empty()) {
    std::ifstream file(dictfile.c_str());
    if (file.good()) {
      LOGP(info, "Running with external topology dictionary: {}", dictfile);
      dict.readFromFile(dictfile);
      LOGP(info, "The IB dictionary size is {}, and the OB dictionary size is {}", dict.getSize(true), dict.getSize(false));
      hasAvailableDict = dict.getSize(true) != 0 && dict.getSize(false) != 0;
      if (hasAvailableDict) {
        LOGP(info, "Dictionaries is vaild.");
      } else {
        LOGP(info, "Dictionaries is NOT vaild!");
      }
    } else {
      LOGP(info, "Cannot open dictionary file: {}. Running without external dictionary!", dictfile);
      dictfile = "";
    }
  } else {
    LOGP(info, "Running without external topology dictionary!");
  }

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if ((hitTree != nullptr) && (clusTree->GetBranch("ITSClusterMCTruth") != nullptr)) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  }

  clusTree->GetEntry(0);
  unsigned int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);

  // Build min and max MC events used by each ROF
  for (int imc = mc2rofVec.size(); imc--;) {
    const auto& mc2rof = mc2rofVec[imc];
    if (mc2rof.rofRecordID < 0) {
      continue; // this MC event did not contribute to any ROF
    }
    for (unsigned int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
      unsigned int irof = mc2rof.rofRecordID + irfd;
      if (irof >= nROFRec) {
        LOGP(error, "ROF = {} from MC2ROF record is >= N ROFs = {}", irof, nROFRec);
      }
      if (mcEvMin[irof] > imc) {
        mcEvMin[irof] = imc;
      }
      if (mcEvMax[irof] < imc) {
        mcEvMax[irof] = imc;
      }
    }
  }

  // Create all plots
  LOGP(info, "Creating plots");
  std::unordered_map<int, Data> data;
  auto initData = [&](int chipID, Data& dat) {
    if (dat.pixelArray)
      return;

    int nCol{0}, nRow{0};
    float lengthPixArr{0}, widthPixArr{0};
    bool isIB = o2::its3::constants::detID::isDetITS3(chipID);
    int layer = gman->getLayer(chipID);
    if (isIB) {
      nCol = o2::its3::SegmentationMosaix::NCols;
      nRow = o2::its3::SegmentationMosaix::NRows;
      lengthPixArr = o2::its3::constants::pixelarray::pixels::mosaix::pitchZ * nCol;
      widthPixArr = o2::its3::constants::pixelarray::pixels::mosaix::pitchX * nRow;
    } else {
      nCol = o2::itsmft::SegmentationAlpide::NCols;
      nRow = o2::itsmft::SegmentationAlpide::NRows;
      lengthPixArr = o2::itsmft::SegmentationAlpide::PitchCol * nCol;
      widthPixArr = o2::itsmft::SegmentationAlpide::PitchRow * nRow;
    }

    dat.pixelArray = new TH2F(Form("histSensor_%d", chipID), Form("SensorID=%d;z(cm);x(cm)", chipID),
                              nCol, -0.5 * lengthPixArr, 0.5 * lengthPixArr,
                              nRow, -0.5 * widthPixArr, 0.5 * widthPixArr);
    dat.hitS = new TGraph();
    dat.hitS->SetMarkerStyle(kFullTriangleDown);
    dat.hitS->SetMarkerColor(kGreen);
    dat.hitM = new TGraph();
    dat.hitM->SetMarkerStyle(kFullCircle);
    dat.hitM->SetMarkerColor(kGreen + 3);
    dat.hitE = new TGraph();
    dat.hitE->SetMarkerStyle(kFullTriangleUp);
    dat.hitE->SetMarkerColor(kGreen + 5);
    dat.clusS = new TGraph();
    dat.clusS->SetMarkerStyle(kFullSquare);
    dat.clusS->SetMarkerColor(kBlue);
    dat.cog = new TGraph();
    dat.cog->SetMarkerStyle(kFullDiamond);
    dat.cog->SetMarkerColor(kRed);
    dat.leg = new TLegend(0.7, 0.7, 0.92, 0.92);
    dat.leg->AddEntry(dat.hitS, "Hit Start");
    dat.leg->AddEntry(dat.hitM, "Hit Middle");
    dat.leg->AddEntry(dat.hitE, "Hit End");
    dat.leg->AddEntry(dat.clusS, "Cluster Start");
    dat.leg->AddEntry(dat.cog, "Cluster COG");
    dat.vClusBox = new std::vector<TBox*>;
  };

  LOGP(info, "Filling digits");
  for (int iDigit{0}; digTree->LoadTree(iDigit) >= 0; ++iDigit) {
    digTree->GetEntry(iDigit);
    for (const auto& digit : *digArr) {
      const auto chipID = digit.getChipIndex();
      if (!selectedChips.count(chipID))
        continue;
      const auto layer = gman->getLayer(chipID);
      bool isIB = layer < 3;
      float locDigiX{0}, locDigiZ{0};
      if (isIB) {
        mMosaixSegmentations[layer].detectorToLocal(digit.getRow(), digit.getColumn(), locDigiX, locDigiZ);
      } else {
        o2::itsmft::SegmentationAlpide::detectorToLocal(digit.getRow(), digit.getColumn(), locDigiX, locDigiZ);
      }
      auto& dat = data[chipID];
      initData(chipID, dat);
      data[chipID].pixelArray->Fill(locDigiZ, locDigiX);
    }
  }

  LOGP(info, "Building min and max MC events used by each ROF, total ROFs {}", nROFRec);
  auto pattIt = patternsPtr->cbegin();
  bool isAllPattIDInvaild{true};
  for (unsigned int irof{0}; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
    // >> read and map MC events contributing to this ROF
    for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
      if (hitVecPool[im] == nullptr) {
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

    // Clusters in this ROF
    for (int icl{0}; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
      const auto& cluster = (*clusArr)[clEntry];
      const auto chipID = cluster.getSensorID();
      if (!selectedChips.count(chipID)) {
        // Even if not selected, advance pattIt if patternID is InvalidPatternID
        if (cluster.getPatternID() == o2::itsmft::CompCluster::InvalidPatternID) {
          o2::itsmft::ClusterPattern::skipPattern(pattIt);
        }
        continue;
      }
      const auto pattID = cluster.getPatternID();
      const bool isIB = o2::its3::constants::detID::isDetITS3(chipID);
      const auto layer = gman->getLayer(chipID);
      auto& dat = data[chipID];
      initData(chipID, dat);
      o2::itsmft::ClusterPattern pattern;
      // Pattern extraction
      if (cluster.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID) {
        isAllPattIDInvaild = false;
        if (!hasAvailableDict) {
          LOGP(error, "Encountered pattern ID {}, which is not equal to the invalid pattern ID {}", cluster.getPatternID(), o2::itsmft::CompCluster::InvalidPatternID);
          LOGP(error, "Clusters have already been generated with a dictionary which was not provided properly!");
          return;
        }
        if (dict.isGroup(cluster.getPatternID(), isIB)) {
          pattern.acquirePattern(pattIt);
        } else {
          pattern = dict.getPattern(cluster.getPatternID(), isIB);
        }
      } else {
        pattern.acquirePattern(pattIt);
      }

      // Hits
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];
      if (!lab.isValid())
        continue;
      const int trID = lab.getTrackID();
      const auto& mc2hit = mc2hitVec[lab.getEventID()];
      const auto* hitArray = hitVecPool[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end())
        continue;
      o2::math_utils::Point3D<float> locHMiddle;
      const auto& hit = (*hitArray)[hitEntry->second];
      auto locHEnd = gman->getMatrixL2G(chipID) ^ (hit.GetPos());
      auto locHStart = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      if (isIB) {
        float xFlat{0.}, yFlat{0.};
        mMosaixSegmentations[layer].curvedToFlat(locHEnd.X(), locHEnd.Y(), xFlat, yFlat);
        locHEnd.SetXYZ(xFlat, yFlat, locHEnd.Z());
        mMosaixSegmentations[layer].curvedToFlat(locHStart.X(), locHStart.Y(), xFlat, yFlat);
        locHStart.SetXYZ(xFlat, yFlat, locHStart.Z());
      }
      locHMiddle.SetXYZ(0.5f * (locHEnd.X() + locHStart.X()),
                        0.5f * (locHEnd.Y() + locHStart.Y()),
                        0.5f * (locHEnd.Z() + locHStart.Z()));
      data[chipID].hitS->AddPoint(locHStart.Z(), locHStart.X());
      data[chipID].hitM->AddPoint(locHMiddle.Z(), locHMiddle.X());
      data[chipID].hitE->AddPoint(locHEnd.Z(), locHEnd.X());

      // Cluster Start point
      float locCluX{0}, locCluZ{0};
      if (isIB) {
        mMosaixSegmentations[layer].detectorToLocal(cluster.getRow(), cluster.getCol(), locCluX, locCluZ);
      } else {
        o2::itsmft::SegmentationAlpide::detectorToLocal(cluster.getRow(), cluster.getCol(), locCluX, locCluZ);
      }
      data[chipID].clusS->AddPoint(locCluZ, locCluX);

      // COG
      o2::math_utils::Point3D<float> locCOG;
      // Cluster COG using dictionary (if available)
      if (hasAvailableDict && (pattID != o2::itsmft::CompCluster::InvalidPatternID && !dict.isGroup(pattID, isIB))) {
        locCOG = dict.getClusterCoordinates(cluster);
      } else {
        if (isIB) {
          locCOG = o2::its3::TopologyDictionary::getClusterCoordinates(cluster, pattern, false);
        } else {
          locCOG = o2::itsmft::TopologyDictionary::getClusterCoordinates(cluster, pattern, false);
        }
      }
      if (isIB) {
        float flatX{0}, flatY{0};
        mMosaixSegmentations[layer].curvedToFlat(locCOG.X(), locCOG.Y(), flatX, flatY);
        locCOG.SetCoordinates(flatX, flatY, locCOG.Z());
      }
      data[chipID].cog->AddPoint(locCOG.Z(), locCOG.X());

      // Cluster Box using dictionary if available, otherwise use raw pattern
      float lowLeftX{0}, lowLeftZ{0}, topRightX{0}, topRightZ{0};
      // Use dictionary-based cluster box
      if (isIB) {
        mMosaixSegmentations[layer].detectorToLocal(cluster.getRow(), cluster.getCol(), lowLeftX, lowLeftZ);
        mMosaixSegmentations[layer].detectorToLocal(cluster.getRow() + pattern.getRowSpan() - 1,
                                                    cluster.getCol() + pattern.getColumnSpan() - 1,
                                                    topRightX, topRightZ);
        lowLeftX += 0.5 * o2::its3::constants::pixelarray::pixels::mosaix::pitchX;
        lowLeftZ -= 0.5 * o2::its3::constants::pixelarray::pixels::mosaix::pitchZ;
        topRightX -= 0.5 * o2::its3::constants::pixelarray::pixels::mosaix::pitchX;
        topRightZ += 0.5 * o2::its3::constants::pixelarray::pixels::mosaix::pitchZ;
      } else {
        o2::itsmft::SegmentationAlpide::detectorToLocal(cluster.getRow(), cluster.getCol(), lowLeftX, lowLeftZ);
        o2::itsmft::SegmentationAlpide::detectorToLocal(cluster.getRow() + pattern.getRowSpan() - 1,
                                                        cluster.getCol() + pattern.getColumnSpan() - 1,
                                                        topRightX, topRightZ);
        lowLeftX += 0.5 * o2::itsmft::SegmentationAlpide::PitchRow;
        lowLeftZ -= 0.5 * o2::itsmft::SegmentationAlpide::PitchCol;
        topRightX -= 0.5 * o2::itsmft::SegmentationAlpide::PitchRow;
        topRightZ += 0.5 * o2::itsmft::SegmentationAlpide::PitchCol;
      }
      auto clusBox = new TBox(lowLeftZ, lowLeftX, topRightZ, topRightX);
      clusBox->SetFillColorAlpha(0, 0);
      clusBox->SetFillStyle(0);
      clusBox->SetLineWidth(4);
      clusBox->SetLineColor(kBlack);
      data[chipID].vClusBox->push_back(clusBox);
    }
  }

  if (isAllPattIDInvaild) {
    LOGP(info, "Verified input cluster file was generated w/o topology dictionary");
    if (!dictfile.empty()) {
      LOGP(error, "Non-dictionary cluster file processed by external dictionary! Please adjust input.");
      return;
    }
  }

  LOGP(info, "Writing to root file");
  double x1, y1, x2, y2;
  auto oFileOut = TFile::Open("CompareClustersAndDigitsOnChip.root", "RECREATE");
  oFileOut->cd();
  for (int chipID{0}; chipID < nChips; chipID++) {
    if (!selectedChips.count(chipID))
      continue;
    auto& dat = data[chipID];
    TString tpath = gman->getMatrixPath(chipID);
    const std::string cpath{tpath.Data() + 39, tpath.Data() + tpath.Length()};
    const std::filesystem::path p{cpath};
    std::string nestedDir = p.parent_path().string();
    TDirectory* currentDir = oFileOut;
    std::istringstream iss(nestedDir);
    std::string token;
    while (std::getline(iss, token, '/')) {
      if (token.empty())
        continue;
      TDirectory* nextDir = currentDir->GetDirectory(token.c_str());
      if (!nextDir) {
        nextDir = currentDir->mkdir(token.c_str());
      }
      if (!nextDir) {
        LOGP(error, "Cannot create subdirectory: %s", token.c_str());
        break;
      }
      currentDir = nextDir;
      currentDir->cd();
    }
    if (!currentDir) {
      LOGP(error, "Failed to create nested directory for chip %d", chipID);
      continue;
    }

    auto canv = new TCanvas(Form("%s_%d", p.filename().c_str(), chipID));
    canv->SetTitle(Form("%s_%d", p.filename().c_str(), chipID));
    canv->cd();
    gPad->SetGrid(1, 1);
    dat.pixelArray->Draw("colz");
    dat.hitS->Draw("p;same");
    dat.hitM->Draw("p;same");
    dat.hitE->Draw("p;same");
    auto arr = new TArrow();
    arr->SetArrowSize(0.01);
    for (int i{0}; i < dat.hitS->GetN(); ++i) {
      dat.hitS->GetPoint(i, x1, y1);
      dat.hitE->GetPoint(i, x2, y2);
      arr->DrawArrow(x1, y1, x2, y2);
    }
    dat.clusS->Draw("p;same");
    if (dat.cog->GetN() != 0)
      dat.cog->Draw("p;same");
    for (const auto& clusBox : *dat.vClusBox) {
      clusBox->Draw();
    }
    dat.leg->Draw();
    canv->SetEditable(false);

    currentDir->WriteTObject(canv, canv->GetName());
    dat.clear();
    delete canv;
    delete arr;
    printf("\rWriting chip %05d", chipID);
  }
  printf("\n");
  oFileOut->Write();
  oFileOut->Close();
  LOGP(info, "Finished writing selected chip visualizations.");
}
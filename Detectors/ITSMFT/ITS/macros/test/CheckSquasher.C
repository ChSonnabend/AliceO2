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
#include <DataFormatsITSMFT/ROFRecord.h>
#include <DataFormatsITSMFT/CompCluster.h>
#include <ITSBase/GeometryTGeo.h>
#include <Framework/Logger.h>
#include <DataFormatsITSMFT/TopologyDictionary.h>
#include <DetectorsCommonDataFormats/DetectorNameConf.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <ITSMFTReconstruction/ChipMappingITS.h>
#include <CCDB/BasicCCDBManager.h>

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TColor.h>
#include <TPad.h>

#include <vector>
#include <gsl/gsl>
#endif

static bool invPal = false;
void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern>& pattVec,
                        std::vector<o2::itsmft::CompClusterExt>* ITSclus,
                        std::vector<unsigned char>* ITSpatt,
                        o2::itsmft::TopologyDictionary& mdict);

void drawClustersInChipInRof(const uint chipId,
                             gsl::span<const o2::itsmft::CompClusterExt> clustersInFrame,
                             gsl::span<const o2::itsmft::ClusterPattern> patternsInFrame,
                             TH2D* hHitMapsVsFrame,
                             TH2D* hHitMapSuperimposed = nullptr);

void drawClustersInStaveInRof(const int staveId,
                              const int layerId,
                              gsl::span<const o2::itsmft::CompClusterExt> clustersInFrame,
                              gsl::span<const o2::itsmft::ClusterPattern> patternsInFrame,
                              TH2D* hHitMapsVsFrame);

void CheckSquasher1(const uint chipId = 0, const uint startingROF = 0, const uint nRofs = 3, bool showSuperimposed = false, const string fname = "o2clus_its.root")
{
  if (!invPal) {
    TColor::InvertPalette();
    invPal = true;
  }

  gStyle->SetOptStat(0);
  // Geometry
  auto& cc = o2::ccdb::BasicCCDBManager::instance();
  cc.setTimestamp(o2::ccdb::getCurrentTimestamp());
  auto* gman = cc.get<o2::its::GeometryTGeo>("ITS/Config/Geometry");
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  // Topology dictionary
  auto mdict = cc.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");
  auto fITSclus = TFile::Open(fname.data(), "r");
  auto treeITSclus = (TTree*)fITSclus->Get("o2sim");

  std::vector<o2::itsmft::CompClusterExt>* ITSclus = nullptr;
  std::vector<o2::itsmft::ROFRecord>* ITSrof = nullptr;
  std::vector<unsigned char>* ITSpatt = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;

  treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
  treeITSclus->SetBranchAddress("ITSClustersROF", &ITSrof);
  treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
  // treeITSclus->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());
  std::vector<TH2D*> hHitMapsVsFrame(nRofs);
  TH2D* hHitMapSuperimposed = nullptr;
  hHitMapSuperimposed = new TH2D(Form("chip%i_superimposed", chipId), Form("chip %i superimposed; ; ; Counts", chipId), 1024, -0.5, 1023.5, 512, -0.5, 511.5);

  treeITSclus->GetEvent(0);
  LOGP(info, "there are {} rofs in this TF", ITSrof->size());

  // Get patterns
  std::vector<o2::itsmft::ClusterPattern> pattVec;
  getClusterPatterns(pattVec, ITSclus, ITSpatt, *mdict);

  for (unsigned int iR{0}; iR < nRofs; iR++) {
    LOGP(info, "===============\n \tProcessing rof {} \n\t===============", iR + startingROF);
    hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof %i; ; ; Counts", chipId, startingROF + iR), 1024, -0.5, 1023.5, 512, -0.5, 511.5);

    // work on data
    const auto& rof = (*ITSrof)[startingROF + iR];
    auto clustersInFrame = rof.getROFData(*ITSclus);
    auto patternsInFrame = rof.getROFData(pattVec);
    drawClustersInChipInRof(chipId, clustersInFrame, patternsInFrame, hHitMapsVsFrame[iR], hHitMapSuperimposed);
  }
  auto canvas = new TCanvas(Form("chip%d", chipId), Form("chip%d", chipId), nRofs * 1000, 600);
  auto canvasSuperimposition = new TCanvas(Form("chip%d_superimposed", chipId), Form("chip%d_superimposed", chipId), 600, 600);

  canvas->Divide(nRofs, 1);
  for (unsigned int i{0}; i < nRofs; ++i) {
    canvas->cd(i + 1);
    gPad->SetGridx();
    gPad->SetGridy();
    hHitMapsVsFrame[i]->Draw("colz");
  }
  canvasSuperimposition->cd();
  gPad->SetGridx();
  gPad->SetGridy();
  if (showSuperimposed) {
    hHitMapSuperimposed->Draw("colz");
  } else {
    delete canvasSuperimposition;
  }
}

void CheckSquasher(const int staveId, const uint layerId, const uint startingROF = 0, const uint nRofs = 3, const string fname = "o2clus_its.root")
{
  std::array<int, 3> staves{12, 14, 16};
  if (!invPal) {
    TColor::InvertPalette();
    invPal = true;
  }

  gStyle->SetOptStat(0);
  // Geometry
  auto& cc = o2::ccdb::BasicCCDBManager::instance();
  cc.setTimestamp(o2::ccdb::getCurrentTimestamp());
  auto* gman = cc.get<o2::its::GeometryTGeo>("ITS/Config/Geometry");
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  // Topology dictionary
  auto mdict = cc.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");
  auto fITSclus = TFile::Open(fname.data(), "r");
  auto treeITSclus = (TTree*)fITSclus->Get("o2sim");

  std::vector<o2::itsmft::CompClusterExt>* ITSclus = nullptr;
  std::vector<o2::itsmft::ROFRecord>* ITSrof = nullptr;
  std::vector<unsigned char>* ITSpatt = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;

  treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
  treeITSclus->SetBranchAddress("ITSClustersROF", &ITSrof);
  treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);

  auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());
  std::vector<TH2D*> hHitMapsVsFrame(nRofs);
  TH2D* hHitMapSuperimposed = nullptr;
  std::vector<TH2D*> hHitMapStaves;
  hHitMapStaves.resize(staveId < 0 ? staves[layerId] : 1);
  for (int iStave{0}; iStave < (staveId < 0 ? staves[layerId] : 1); ++iStave) {
    hHitMapStaves[staveId < 0 ? iStave : 0] = new TH2D(Form("stave_%d", staveId < 0 ? iStave : staveId), "", 1024 * 9, -0.5, 1023.5 * 9, 512, -0.5, 511.5);
  }

  treeITSclus->GetEvent(0);
  LOGP(info, "there are {} rofs in this TF", ITSrof->size());

  // Get patterns
  std::vector<o2::itsmft::ClusterPattern> pattVec;
  getClusterPatterns(pattVec, ITSclus, ITSpatt, *mdict);

  for (unsigned int iR{0}; iR < nRofs; iR++) {
    LOGP(info, " ===============\n \tProcessing rof {} \n\t===============", iR + startingROF);
    // work on data
    const auto& rof = (*ITSrof)[startingROF + iR];
    auto clustersInFrame = rof.getROFData(*ITSclus);
    auto patternsInFrame = rof.getROFData(pattVec);
    for (int iStave{0}; iStave < (staveId < 0 ? staves[layerId] : 1); ++iStave) {
      drawClustersInStaveInRof(staveId < 0 ? iStave : staveId, layerId, clustersInFrame, patternsInFrame, hHitMapStaves[staveId < 0 ? iStave : 0]);
    }
  }

  auto canvas = new TCanvas(Form("stave%d", staveId), Form("stave%d", staveId), 9 * 1000, (staveId < 0 ? staves[layerId] : 1) * 1000);
  if (staveId < 0) {
    canvas->Divide(1, staves[layerId]);
  }
  for (int iStave{0}; iStave < (staveId < 0 ? staves[layerId] : 1); ++iStave) {
    if (staveId < 0) {
      canvas->cd(iStave + 1);
    }
    gPad->SetGridx();
    gPad->SetGridy();
    hHitMapStaves[staveId < 0 ? iStave : 0]->Draw("colz");
  }
}

void getClusterPatterns(std::vector<o2::itsmft::ClusterPattern>& pattVec, std::vector<o2::itsmft::CompClusterExt>* ITSclus, std::vector<unsigned char>* ITSpatt, o2::itsmft::TopologyDictionary& mdict)
{
  pattVec.reserve(ITSclus->size());
  auto pattIt = ITSpatt->cbegin();

  for (unsigned int iClus{0}; iClus < ITSclus->size(); ++iClus) {
    auto& clus = (*ITSclus)[iClus];

    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {

      npix = mdict.getNpixels(pattID);
      patt = mdict.getPattern(pattID);
    }

    pattVec.push_back(patt);
  }
}

void drawClustersInChipInRof(const uint chipId,
                             gsl::span<const o2::itsmft::CompClusterExt> clustersInFrame,
                             gsl::span<const o2::itsmft::ClusterPattern> patternsInFrame,
                             TH2D* hHitMapsVsFrame,
                             TH2D* hHitMapSuperimposed)
{
  for (unsigned int clusInd{0}; clusInd < clustersInFrame.size(); clusInd++) {
    const auto& clus = clustersInFrame[clusInd];
    auto sID = clus.getSensorID();

    if (sID == chipId) {
      LOGP(info, "Processing cluster {}", clusInd);
      clus.print();

      // extract pattern info
      auto col = clus.getCol();
      auto row = clus.getRow();

      std::cout << patternsInFrame[clusInd];

      std::cout << std::endl;
      int ic = 0, ir = 0;

      auto colSpan = patternsInFrame[clusInd].getColumnSpan();
      auto rowSpan = patternsInFrame[clusInd].getRowSpan();
      auto nBits = rowSpan * colSpan;

      for (int i = 2; i < patternsInFrame[clusInd].getUsedBytes() + 2; i++) {
        unsigned char tempChar = patternsInFrame[clusInd].getByte(i);
        int s = 128; // 0b10000000
        while (s > 0) {
          if ((tempChar & s) != 0) // checking active pixels
          {
            hHitMapsVsFrame->Fill(col + ic, row + ir);
            hHitMapSuperimposed == nullptr ?: hHitMapSuperimposed->Fill(col + ic, row + ir);
          }
          ic++;
          s >>= 1;
          if ((ir + 1) * ic == nBits) {
            break;
          }
          if (ic == colSpan) {
            ic = 0;
            ir++;
          }
          if ((ir + 1) * ic == nBits) {
            break;
          }
        }
      }
    }
  }
}

void drawClustersInStaveInRof(const int staveId,
                              const int layerId,
                              gsl::span<const o2::itsmft::CompClusterExt> clustersInFrame,
                              gsl::span<const o2::itsmft::ClusterPattern> patternsInFrame,
                              TH2D* hHitMapsVsFrame)
{
  o2::itsmft::ChipMappingITS chipMapping;
  int lay, sta, ssta, mod, chipInMod;
  std::array<int, 9> sensorIDs; // sIDs in the staves of the IB.
  for (unsigned int clusInd{0}; clusInd < clustersInFrame.size(); clusInd++) {
    const auto& clus = clustersInFrame[clusInd];
    auto sID = clus.getSensorID();

    chipMapping.expandChipInfoHW(sID, lay, sta, ssta, mod, chipInMod);

    if (sta == staveId && lay == layerId) {
      // extract pattern info
      auto col = clus.getCol();
      auto row = clus.getRow();

      int ic = 0, ir = 0;

      auto colSpan = patternsInFrame[clusInd].getColumnSpan();
      auto rowSpan = patternsInFrame[clusInd].getRowSpan();
      auto nBits = rowSpan * colSpan;

      for (int i = 2; i < patternsInFrame[clusInd].getUsedBytes() + 2; i++) {
        unsigned char tempChar = patternsInFrame[clusInd].getByte(i);
        int s = 128; // 0b10000000
        while (s > 0) {
          if ((tempChar & s) != 0) // checking active pixels
          {
            // Normalize the chip ID to be within the range [0, 8]
            int normalizedChipID = sID % 9; // Hardcode for the ITS IB
            double x = col + ic + normalizedChipID * 1024;
            double y = row + ir;
            hHitMapsVsFrame->Fill(x, y);
          }
          ic++;
          s >>= 1;
          if ((ir + 1) * ic == nBits) {
            break;
          }
          if (ic == colSpan) {
            ic = 0;
            ir++;
          }
          if ((ir + 1) * ic == nBits) {
            break;
          }
        }
      }
    }
  }
}
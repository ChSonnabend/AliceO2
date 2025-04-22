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

#include "Rtypes.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TGeoCompositeShape.h"
#include "TSystem.h"
#include "TGLViewer.h"
#include "TMath.h"

#include "TFile.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TRandom3.h"
#include "TLine.h"
#include "TArc.h"

#include <iostream>
#include <fmt/format.h>

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Cartesian.h"

#include "ITS3Base/SpecsV2.h"
#include "ITS3Base/SegmentationMosaix.h"
#include "ITSBase/GeometryTGeo.h"

#endif

using gITS = o2::its::GeometryTGeo;

void CheckMosaixSegment(bool isTestDetectorToLocal = false,
                        bool isTestFlatToCurved = false,
                        bool isTestLocalToGlobal = false)
{
  using namespace o2::its3;
  static constexpr unsigned int mNCols{SegmentationMosaix::NCols};
  static constexpr unsigned int mNRows{SegmentationMosaix::NRows};
  static constexpr unsigned int nPixels{mNCols * mNRows};
  std::array<SegmentationMosaix, 3> mMosaixSegmentations{0, 1, 2};

  if (isTestDetectorToLocal || isTestFlatToCurved) {
    namespace cp = constants::pixelarray;
    TH2I* h_row_col = new TH2I("h_row_col", "rows and cols sown;row;col", mNRows, 0, mNRows, mNCols, 0, mNCols);
    TH2D* h_xLocal_zLocal = new TH2D("h_xLocal_zLocal", "x and z from rows and cols;xLocal;zLocal", mNRows, -cp::length / 2, cp::length / 2, mNCols, -cp::width / 2, cp::width / 2);
    TH2I* h_row_col_translate = new TH2I("h_row_col_translate", "rows and cols from x and z;row;col", mNRows, 0, mNRows, mNCols, 0, mNCols);
    TGraph* g_row_xLocal = new TGraph();
    g_row_xLocal->SetMarkerStyle(20);
    g_row_xLocal->SetMarkerSize(0.2);
    TGraph* g_col_zLocal = new TGraph();
    g_col_zLocal->SetMarkerStyle(20);
    g_col_zLocal->SetMarkerSize(0.2);
    TGraph* g_row_xLocal_translate = new TGraph();
    g_row_xLocal_translate->SetMarkerStyle(20);
    g_row_xLocal_translate->SetMarkerSize(0.2);
    TGraph* g_col_zLocal_translate = new TGraph();
    g_col_zLocal_translate->SetMarkerStyle(20);

    SegmentationMosaix seg(0);
    int nPoint = 0;
    for (UInt_t i = 0; i < mNRows; ++i) {
      for (UInt_t j = 0; j < mNCols; ++j) {
        float xLocal = -1;
        float zLocal = -1;
        int row_trans = -1;
        int col_trans = -1;
        seg.detectorToLocal(i, j, xLocal, zLocal);
        seg.localToDetector(xLocal, zLocal, row_trans, col_trans);
        g_row_xLocal->SetPoint(nPoint, i, xLocal);
        g_col_zLocal->SetPoint(nPoint, j, zLocal);
        g_row_xLocal_translate->SetPoint(nPoint, xLocal, row_trans);
        g_col_zLocal_translate->SetPoint(nPoint++, zLocal, col_trans);

        bool pattern = ((i >= 50 && i <= 100) || (i >= 250 && i <= 350)) && ((j >= 30 && j <= 70) || (j >= 100 && j <= 120));
        if (pattern) {
          h_row_col->Fill(i, j);
          h_xLocal_zLocal->Fill(xLocal, zLocal);
          h_row_col_translate->Fill(row_trans, col_trans);
        }
      }
    }
    TCanvas* c1 = new TCanvas("c1", "c1", 1200, 400);
    gStyle->SetPadLeftMargin(0.15);
    // gStyle->SetPalette(kCMYK);
    c1->Divide(3, 1);
    c1->cd(1);
    h_row_col->Draw("colz");
    c1->cd(2);
    h_xLocal_zLocal->Draw("colz");
    c1->cd(3);
    h_row_col_translate->Draw("colz");

    TCanvas* c2 = new TCanvas("c2", "c2", 1600, 400);
    c2->Divide(4, 1);
    c2->cd(1);
    g_row_xLocal->SetTitle("xLocal vs row;row;xLocal");
    g_row_xLocal->Draw("same ap");
    c2->cd(2);
    g_col_zLocal->SetTitle("zLocal vs col;col;zLocal");
    g_col_zLocal->Draw("same ap");
    c2->cd(3);
    g_row_xLocal_translate->SetTitle("row_translate vs xLocal;xLocal;row_translate");
    g_row_xLocal_translate->Draw("same ap");
    c2->cd(4);
    g_col_zLocal_translate->SetTitle("col_translate vs zLocal;zLocal;col_translate");
    g_col_zLocal_translate->Draw("same ap");
  }

  if (isTestLocalToGlobal) {
    o2::base::GeometryManager::loadGeometry();
    namespace cp = constants::pixelarray;
    TH2D* h_xCurved_yCurved = new TH2D("h_xCurved_yCurved", "from flat to curved;x;y", 200, -1, 4, 200, -2, 3);
    TH2D* h_xFlat_yFlat = new TH2D("h_xFlat_yFlat", "from curved to flat ;x;y", 200, -1, 4, 200, -2, 3);
    TH2D* h_xGlobal_yGlobal = new TH2D("h_xGlobal_yGlobal", ";xGlobel;yGlobal", 5000, -constants::radii[2], +constants::radii[2], 5000, -constants::radii[2], +constants::radii[2]);
    TH2D* h_zGlobal_xGlobal = new TH2D("h_zGlobal_xGlobal", ";zGlobel;xGlobal", 50, -constants::segment::width, +constants::segment::width, 1000, -constants::radii[2] * 1.2, +constants::radii[2] * 1.2);
    TH2D* h_zGlobal_yGlobal = new TH2D("h_zGlobal_yGlobal", ";zGlobel;yGlobal", 50, -constants::segment::width, +constants::segment::width, 1000, -constants::radii[2] * 1.2, +constants::radii[2] * 1.2);

    for (unsigned int iLayer{}; iLayer < 3; ++iLayer) {
      // for (unsigned int iLayer{}; iLayer < constants::nLayers; ++iLayer) {
      for (unsigned int iCarbonForm{0}; iCarbonForm < 2; ++iCarbonForm) {
        // No Loop for chip = carbonform id
        for (unsigned int iSegment{0}; iSegment < constants::nSegments[iLayer]; ++iSegment) {
          for (unsigned int iRSU{0}; iRSU < 12; ++iRSU) {
            for (unsigned int iTile{0}; iTile < 12; ++iTile) {
              TString path{"/TOP_1/"};
              path += Form("%s_0/", gITS::getITS3LayerPattern(iLayer));
              path += Form("%s_%d/", gITS::getITS3CarbonFormPattern(iLayer), iCarbonForm);
              path += Form("%s_0/", gITS::getITS3ChipPattern(iLayer));
              path += Form("%s_%d/", gITS::getITS3SegmentPattern(iLayer), iSegment);
              path += Form("%s_%d/", gITS::getITS3RSUPattern(iLayer), iRSU);
              path += Form("%s_%d/", gITS::getITS3TilePattern(iLayer), iTile);
              if (!gGeoManager->CheckPath(path.Data())) {
                std::cerr << path << std::endl;
              }
              gGeoManager->cd(path.Data());
              auto pixelArray = gGeoManager->GetCurrentVolume();
              // Get the current matrix
              TGeoHMatrix* matrix = gGeoManager->GetCurrentMatrix();
              for (UInt_t row = 0; row < mNRows; row++) {
                for (UInt_t col = 0; col < mNCols; col++) {
                  float xLocal = 0;
                  float zLocal = 0;
                  float xCurved = 0;
                  float yCurved = 0;
                  float xLocal_translate = 0;
                  float yLocal_translate = 0;

                  mMosaixSegmentations[iLayer].detectorToLocal(row, col, xLocal, zLocal);
                  mMosaixSegmentations[iLayer].flatToCurved(xLocal, 0., xCurved, yCurved);
                  double posLocal[3] = {xCurved, yCurved, zLocal};
                  double posGlobal[3] = {0, 0, 0};
                  mMosaixSegmentations[iLayer].curvedToFlat(xCurved, yCurved, xLocal_translate, yLocal_translate);
                  matrix->LocalToMaster(posLocal, posGlobal);

                  h_xCurved_yCurved->Fill(xLocal, 0);
                  h_xCurved_yCurved->Fill(xCurved, yCurved);
                  h_xFlat_yFlat->Fill(xCurved, yCurved);
                  h_xFlat_yFlat->Fill(xLocal_translate, 0);
                  h_xGlobal_yGlobal->Fill(posGlobal[0], posGlobal[1]);
                  h_zGlobal_xGlobal->Fill(posGlobal[2], posGlobal[0]);
                  h_zGlobal_yGlobal->Fill(posGlobal[2], posGlobal[1]);
                }
              }
            }
          }
        }
      }
    }

    TArc* arc[3];
    h_xCurved_yCurved->Draw("colz");
    for (int i = 0; i < 3; i++) {
      arc[i] = new TArc(-0, 0, constants::radiiOuter[i], -5, 40);
      arc[i]->SetFillStyle(0);
    }

    TCanvas* c2 = new TCanvas("c2", "c2", 400, 400);
    c2->Divide(2, 1);
    c2->cd(1);
    TLine* line = new TLine(h_xCurved_yCurved->GetXaxis()->GetXmin(), 0, h_xCurved_yCurved->GetXaxis()->GetXmax(), 0);
    line->Draw("same");
    h_xCurved_yCurved->Draw("colz");
    for (int i = 0; i < 3; i++)
      arc[i]->Draw("same only");
    c2->cd(2);
    line->Draw("same");
    h_xFlat_yFlat->Draw("colz");
    for (int i = 0; i < 3; i++)
      arc[i]->Draw("same only");

    TCanvas* c3 = new TCanvas("c3", "c3", 800, 800);
    c3->Divide(2, 2);
    c3->cd(1);
    // Draw a cross in the middle
    h_xGlobal_yGlobal->Draw("colz");
    TLine* line1 = new TLine(h_xGlobal_yGlobal->GetXaxis()->GetXmin(), 0, h_xGlobal_yGlobal->GetXaxis()->GetXmax(), 0);
    TLine* line2 = new TLine(0, h_xGlobal_yGlobal->GetYaxis()->GetXmin(), 0, h_xGlobal_yGlobal->GetYaxis()->GetXmax());
    line1->Draw("same");
    line2->Draw("same");
    c3->cd(2);
    // Draw a horizontal line in the middle
    h_zGlobal_xGlobal->Draw("colz");
    TLine* line3 = new TLine(h_zGlobal_xGlobal->GetXaxis()->GetXmin(), 0, h_zGlobal_xGlobal->GetXaxis()->GetXmax(), 0);
    line3->Draw("same");
    c3->cd(3);
    // Draw a vertical line in the middle
    h_zGlobal_yGlobal->Draw("colz");
    TLine* line4 = new TLine(h_zGlobal_yGlobal->GetXaxis()->GetXmin(), 0, h_zGlobal_yGlobal->GetXaxis()->GetXmax(), 0);
    line4->Draw("same");
  }
}

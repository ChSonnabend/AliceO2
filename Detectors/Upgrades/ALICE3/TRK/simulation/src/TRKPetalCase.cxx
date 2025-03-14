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

#include "TRKSimulation/TRKPetalCase.h"
#include "TRKBase/GeometryTGeo.h"
#include <DetectorsBase/MaterialManager.h>

#include "Framework/Logger.h"

#include "TGeoTube.h"
#include "TGeoMatrix.h"
#include "TGeoCompositeShape.h"
#include "TGeoVolume.h"
#include "TString.h"
#include "TMath.h"

namespace o2
{
namespace trk
{
TRKPetalCase::TRKPetalCase(Int_t number, TGeoVolume* motherVolume, Bool_t irisOpen) : mPetalCaseNumber(number), mOpenState(irisOpen)
{

  mWallThickness = .15e-1; // cm // Assume all the walls have the same thickness for now.
  mRIn = 0.48;             // cm
  mROut = 3;               // cm
  mRInOpenState = 1.5;     // cm
  mPetalCaseLength = 70.;  // cm

  // Calculate angular coverages of azimuthal part of wall (equivalent to that of the sensitive volumes)
  mAngularCoverageAzimuthalWall = (0.25 * (2 * mRIn * TMath::Pi()) - 2 * mWallThickness) / mRIn;
  mAngularCoverageRadialWall = mWallThickness / mRIn;
  mToDeg = 180 / TMath::Pi();

  // Calculate the center of the petal (x_c, y_c, z_c) based on whether it is open or not
  mZPos = 0;
  if (mOpenState) {
    Double_t rHalfPetal = 0.5 * (mRIn + mROut);
    Double_t rOpenStateCenter = TMath::Sqrt(rHalfPetal * rHalfPetal + mRInOpenState * mRInOpenState);
    mXPos = rOpenStateCenter * TMath::Cos(0.25 * TMath::Pi() + (mPetalCaseNumber - 1) * 0.5 * TMath::Pi());
    mYPos = rOpenStateCenter * TMath::Sin(0.25 * TMath::Pi() + (mPetalCaseNumber - 1) * 0.5 * TMath::Pi());
  } else {
    mXPos = 0.;
    mYPos = 0.;
  }

  // Make the petal case
  constructCase(motherVolume);
  // Make coldplate
  constructColdPlate(motherVolume);
  // Add the detection petals (quarter disks and barrel layers)
  constructDetectionPetals(motherVolume);
}

TString TRKPetalCase::getFullName()
{
  TString fullCompositeName = Form("PETALCASE%d_FULLCOMPOSITE", mPetalCaseNumber);
  TGeoCompositeShape* fullCompositeShape = new TGeoCompositeShape(fullCompositeName, mFullCompositeFormula);
  return fullCompositeName;
}

void TRKPetalCase::constructCase(TGeoVolume* motherVolume)
{

  // Petal case parts in TGeoTubeSeg
  mInnerAzimuthalWall = new TGeoTubeSeg(Form("PETAL%d_INNER_AZIMUTHAL_WALL", mPetalCaseNumber), mRIn, mRIn + mWallThickness, mPetalCaseLength / 2., -0.5 * mAngularCoverageAzimuthalWall * mToDeg, 0.5 * mAngularCoverageAzimuthalWall * mToDeg);
  mOuterAzimuthalWall = new TGeoTubeSeg(Form("PETAL%d_OUTER_AZIMUTHAL_WALL", mPetalCaseNumber), mROut, mROut + mWallThickness, mPetalCaseLength / 2., -0.5 * mAngularCoverageAzimuthalWall * mToDeg, 0.5 * mAngularCoverageAzimuthalWall * mToDeg);
  mRadialWall = new TGeoTubeSeg(Form("PETAL%d_RADIAL_WALL", mPetalCaseNumber), mRIn, mROut + mWallThickness, mPetalCaseLength / 2., -0.5 * mAngularCoverageRadialWall * mToDeg, 0.5 * mAngularCoverageRadialWall * mToDeg);
  mForwardWall = new TGeoTubeSeg(Form("PETAL%d_FORWARD_WALL", mPetalCaseNumber), mRIn, mROut + mWallThickness, mWallThickness / 2., -0.5 * (mAngularCoverageAzimuthalWall + 2 * mAngularCoverageRadialWall) * mToDeg, 0.5 * (mAngularCoverageAzimuthalWall + 2 * mAngularCoverageRadialWall) * mToDeg);

  // Rotate to correct section : 0-3
  mAzimuthalWallRot = new TGeoRotation((TString)Form("PETAL%d_AZIMUTHAL_WALL_ROT", mPetalCaseNumber), (mPetalCaseNumber * 0.5 * TMath::Pi() + 0.5 * mAngularCoverageAzimuthalWall + mAngularCoverageRadialWall) * mToDeg, 0., 0.);
  mAzimuthalWallRot->RegisterYourself();
  mRadialWall1Rot = new TGeoRotation((TString)Form("PETAL%d_RADIAL_WALL1_ROT", mPetalCaseNumber), (mPetalCaseNumber * 0.5 * TMath::Pi() + 0.5 * mAngularCoverageRadialWall) * mToDeg, 0., 0.);
  mRadialWall1Rot->RegisterYourself();
  mRadialWall2Rot = new TGeoRotation((TString)Form("PETAL%d_RADIAL_WALL2_ROT", mPetalCaseNumber), (mPetalCaseNumber * 0.5 * TMath::Pi() + mAngularCoverageAzimuthalWall + 1.5 * mAngularCoverageRadialWall) * mToDeg, 0., 0.);
  mRadialWall2Rot->RegisterYourself();

  // Place to correct position (open or closed)
  mAzimuthalWallCombiTrans = new TGeoCombiTrans((TString)Form("PETAL%d_AZIMUTHAL_WALL_COMBITRANS", mPetalCaseNumber), mXPos, mYPos, mZPos, mAzimuthalWallRot);
  mAzimuthalWallCombiTrans->RegisterYourself();
  mRadialWall1CombiTrans = new TGeoCombiTrans((TString)Form("PETAL%d_RADIAL_WALL1_COMBITRANS", mPetalCaseNumber), mXPos, mYPos, mZPos, mRadialWall1Rot);
  mRadialWall1CombiTrans->RegisterYourself();
  mRadialWall2CombiTrans = new TGeoCombiTrans((TString)Form("PETAL%d_RADIAL_WALL2_COMBITRANS", mPetalCaseNumber), mXPos, mYPos, mZPos, mRadialWall2Rot);
  mRadialWall2CombiTrans->RegisterYourself();
  mForwardWall1CombiTrans = new TGeoCombiTrans((TString)Form("PETAL%d_FORWARD_WALL1_COMBITRANS", mPetalCaseNumber), mXPos, mYPos, (mPetalCaseLength + mWallThickness) / 2., mAzimuthalWallRot);
  mForwardWall1CombiTrans->RegisterYourself();
  mForwardWall2CombiTrans = new TGeoCombiTrans((TString)Form("PETAL%d_FORWARD_WALL2_COMBITRANS", mPetalCaseNumber), mXPos, mYPos, -(mPetalCaseLength + mWallThickness) / 2., mAzimuthalWallRot);
  mForwardWall2CombiTrans->RegisterYourself();

  TString petalCaseCompositeFormula = (TString)Form("PETAL%d_INNER_AZIMUTHAL_WALL:PETAL%d_AZIMUTHAL_WALL_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber) + (TString)Form("+PETAL%d_OUTER_AZIMUTHAL_WALL:PETAL%d_AZIMUTHAL_WALL_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber) + (TString)Form("+PETAL%d_RADIAL_WALL:PETAL%d_RADIAL_WALL1_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber) + (TString)Form("+PETAL%d_RADIAL_WALL:PETAL%d_RADIAL_WALL2_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber) + (TString)Form("+PETAL%d_FORWARD_WALL:PETAL%d_FORWARD_WALL1_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber) + (TString)Form("+PETAL%d_FORWARD_WALL:PETAL%d_FORWARD_WALL2_COMBITRANS", mPetalCaseNumber, mPetalCaseNumber);

  TGeoCompositeShape* petalCaseComposite = new TGeoCompositeShape((TString)Form("PETALCASE%dsh", mPetalCaseNumber), petalCaseCompositeFormula);
  mFullCompositeFormula = petalCaseComposite->GetName();
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* kMedBe = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_BERYLLIUM");

  mPetalCaseName = Form("PETALCASE%d", mPetalCaseNumber);
  mPetalCaseVolume = new TGeoVolume(mPetalCaseName, petalCaseComposite, kMedBe);
  mPetalCaseVolume->SetVisibility(1);
  mPetalCaseVolume->SetLineColor(kGray);

  LOGP(info, "Creating IRIS Tracker vacuum petal case {}", mPetalCaseNumber);
  LOGP(info, "Inserting {} in {} ", mPetalCaseVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(mPetalCaseVolume, 1, nullptr);
}

void TRKPetalCase::constructColdPlate(TGeoVolume* motherVolume)
{
  Double_t coldPlateRadius = 2.6;     // cm
  Double_t coldPlateThickness = 0.15; // cm
  Double_t coldPlateLength = 50.;     // cm

  mColdPlate = new TGeoTubeSeg((TString)Form("PETAL%d_COLDPLATE", mPetalCaseNumber), coldPlateRadius, coldPlateRadius + coldPlateThickness, coldPlateLength / 2., -0.5 * mAngularCoverageAzimuthalWall * mToDeg, 0.5 * mAngularCoverageAzimuthalWall * mToDeg);
  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* medCeramic = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_CERAMIC");
  mColdPlateVolume = new TGeoVolume(Form("COLDPLATE%d", mPetalCaseNumber), mColdPlate, medCeramic);

  TString coldPlateCompositeFormula = mColdPlate->GetName();
  coldPlateCompositeFormula += ":";
  coldPlateCompositeFormula += mAzimuthalWallCombiTrans->GetName();
  addToPetalCaseComposite(coldPlateCompositeFormula);

  mColdPlateVolume->SetVisibility(1);
  mColdPlateVolume->SetLineColor(kGray);

  LOGP(info, "Creating cold plate service");
  LOGP(info, "Inserting {} in {} ", mColdPlateVolume->GetName(), motherVolume->GetName());
  motherVolume->AddNode(mColdPlateVolume, 1, mAzimuthalWallCombiTrans);
}

void TRKPetalCase::constructDetectionPetals(TGeoVolume* motherVolume)
{
  // Add petal layers
  // layerNumber, layerName, rIn, angularCoverage, zLength, layerx2X0
  mPetalLayers.emplace_back(0, Form("%s_LAYER%d", mPetalCaseName.Data(), 0), 0.5f, mAngularCoverageAzimuthalWall, 50.f, 1.e-3);
  mPetalLayers.emplace_back(1, Form("%s_LAYER%d", mPetalCaseName.Data(), 1), 1.2f, mAngularCoverageAzimuthalWall, 50.f, 1.e-3);
  mPetalLayers.emplace_back(2, Form("%s_LAYER%d", mPetalCaseName.Data(), 2), 2.5f, mAngularCoverageAzimuthalWall, 50.f, 1.e-3);
  for (Int_t i = 0; i < mPetalLayers.size(); ++i) {
    mPetalLayers[i].createLayer(motherVolume, mAzimuthalWallCombiTrans);
  }

  // Add petal disks
  // diskNumber, diskName, zPos, rIn, rOut, angularCoverage, diskx2X0
  mPetalDisks.emplace_back(0, Form("%s_DISK%d", mPetalCaseName.Data(), 0), 26., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  mPetalDisks.emplace_back(1, Form("%s_DISK%d", mPetalCaseName.Data(), 1), 30., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  mPetalDisks.emplace_back(2, Form("%s_DISK%d", mPetalCaseName.Data(), 2), 34., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  mPetalDisks.emplace_back(3, Form("%s_DISK%d", mPetalCaseName.Data(), 3), -26., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  mPetalDisks.emplace_back(4, Form("%s_DISK%d", mPetalCaseName.Data(), 4), -30., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  mPetalDisks.emplace_back(5, Form("%s_DISK%d", mPetalCaseName.Data(), 5), -34., .5, 2.5, mAngularCoverageAzimuthalWall, 1.e-3);
  for (Int_t i = 0; i < mPetalDisks.size(); ++i) {
    mPetalDisks[i].createDisk(motherVolume, mAzimuthalWallCombiTrans);
  }

  addDetectionPetelsToFullComposite();
}

void TRKPetalCase::addDetectionPetelsToFullComposite()
{
  for (Int_t i = 0; i < mPetalLayers.size(); ++i) {
    Double_t zLength = mPetalLayers[i].getZLength();
    Double_t rIn = mPetalLayers[i].getInnerRadius();
    Double_t thickness = mPetalLayers[i].getChipThickness();
    Double_t angularCoverage = mPetalLayers[i].getAngularCoverage();
    TGeoTubeSeg* layerForExcavation = new TGeoTubeSeg(Form("PETALCASE%d_EXCAVATIONLAYER%d", mPetalCaseNumber, i), rIn, rIn + thickness, zLength / 2., -0.5 * angularCoverage * mToDeg, 0.5 * angularCoverage * mToDeg);

    TString layerForExcavationCompositeFormula = layerForExcavation->GetName();
    layerForExcavationCompositeFormula += ":";
    layerForExcavationCompositeFormula += mAzimuthalWallCombiTrans->GetName();
    addToPetalCaseComposite(layerForExcavationCompositeFormula);
  }

  for (Int_t i = 0; i < mPetalDisks.size(); ++i) {
    Double_t zPos = mPetalDisks[i].getZ();
    Double_t rIn = mPetalDisks[i].getInnerRadius();
    Double_t rOut = mPetalDisks[i].getOuterRadius();
    Double_t thickness = mPetalDisks[i].getThickness();
    Double_t angularCoverage = mPetalDisks[i].getAngularCoverage();
    TGeoTubeSeg* diskForExcavation = new TGeoTubeSeg(Form("PETALCASE%d_EXCAVATIONDISK%d", mPetalCaseNumber, i), rIn, rOut, thickness / 2., -0.5 * angularCoverage * mToDeg, 0.5 * angularCoverage * mToDeg);
    TGeoCombiTrans* diskForExcavationCombiTrans = new TGeoCombiTrans(*(mAzimuthalWallCombiTrans->MakeClone())); // Copy from petal case
    diskForExcavationCombiTrans->SetName((TString)Form("PETALCASE%d_EXCAVATIONDISK%d_COMBITRANS", mPetalCaseNumber, i));
    diskForExcavationCombiTrans->SetDz(zPos); // Overwrite z location
    diskForExcavationCombiTrans->RegisterYourself();

    TString diskForExcavationCompositeFormula = diskForExcavation->GetName();
    diskForExcavationCompositeFormula += ":";
    diskForExcavationCompositeFormula += diskForExcavationCombiTrans->GetName();
    addToPetalCaseComposite(diskForExcavationCompositeFormula);
  }
}

// ClassImp(TRKPetalCase);
} // namespace trk
} // namespace o2
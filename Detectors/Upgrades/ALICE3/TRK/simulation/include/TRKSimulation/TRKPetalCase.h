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

#ifndef ALICEO2_TRK_PETALCASE_H
#define ALICEO2_TRK_PETALCASE_H

#include <Rtypes.h>

#include "TRKSimulation/TRKPetalLayer.h"
#include "TRKSimulation/TRKPetalDisk.h"
#include "TGeoCompositeShape.h"

namespace o2
{
namespace trk
{
class TRKPetalCase
{
 public:
  TRKPetalCase() = default;
  TRKPetalCase(Int_t number, TGeoVolume* motherVolume, Bool_t irisOpen);
  ~TRKPetalCase() = default;

  // Sensitive volume list
  std::vector<TRKPetalLayer> mPetalLayers;
  std::vector<TRKPetalDisk> mPetalDisks;

  auto getPetalCaseName() { return mPetalCaseName; }
  TString getFullName();

 private:
  void constructCase(TGeoVolume* motherVolume);
  void constructColdPlate(TGeoVolume* motherVolume);
  void constructDetectionPetals(TGeoVolume* motherVolume);
  void addDetectionPetelsToFullComposite();

  void addToPetalCaseComposite(TString shape) { mFullCompositeFormula += ("+" + shape); }

  Int_t mPetalCaseNumber; // Used to determine rotation and position. 0-3
  Bool_t mOpenState;      // At injection energy, the iris tracker is in the open position. During stable beams, it is closed

  TString mPetalCaseName;
  TString mFullCompositeFormula; // Used to excavate the petal and all its components from the vacuum

  // Center position of the petal case. 0,0,0 at stable beams (a.k.a. closed state)
  Double_t mXPos, mYPos, mZPos;

  Double_t mWallThickness;   // cm // Assume all the walls have the same thickness for now
  Double_t mRIn;             // cm
  Double_t mROut;            // cm
  Double_t mRInOpenState;    // cm
  Double_t mPetalCaseLength; // cm

  Double_t mAngularCoverageAzimuthalWall; // Rad // Angular coverage of azimuthal part of wall (equivalent to that of the sensitive volumes)
  Double_t mAngularCoverageRadialWall;    // Rad // Angular coverage of radial part of wall
  Double_t mToDeg;

  // Petal case parts -> In one composite shape
  TGeoTubeSeg* mInnerAzimuthalWall;
  TGeoTubeSeg* mOuterAzimuthalWall;
  TGeoTubeSeg* mRadialWall;
  TGeoTubeSeg* mForwardWall;

  TGeoRotation* mAzimuthalWallRot;
  TGeoRotation* mRadialWall1Rot;
  TGeoRotation* mRadialWall2Rot;

  TGeoCombiTrans* mAzimuthalWallCombiTrans;
  TGeoCombiTrans* mRadialWall1CombiTrans;
  TGeoCombiTrans* mRadialWall2CombiTrans;
  TGeoCombiTrans* mForwardWall1CombiTrans;
  TGeoCombiTrans* mForwardWall2CombiTrans;

  TGeoVolume* mPetalCaseVolume;

  // Cold plate
  TGeoTubeSeg* mColdPlate;
  TGeoVolume* mColdPlateVolume;

  ClassDef(TRKPetalCase, 1);
};

} // namespace trk
} // namespace o2
#endif // ALICEO2_TRK_PETALCASE_H
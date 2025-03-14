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

/// \file TRKPetalDisk.cxx
/// \brief Implementation of the TRKPetalDisk class

#include "TRKSimulation/TRKPetalDisk.h"
#include "TRKBase/GeometryTGeo.h"

#include <fairlogger/Logger.h> // for LOG

#include "TGeoManager.h"        // for TGeoManager, gGeoManager
#include "TGeoMatrix.h"         // for TGeoCombiTrans, TGeoRotation, etc
#include "TGeoTube.h"           // for TGeoTube, TGeoTubeSeg
#include "TGeoVolume.h"         // for TGeoVolume, TGeoVolumeAssembly
#include "TGeoCompositeShape.h" // for TGeoCompositeShape
#include "TMathBase.h"          // for Abs
#include "TMath.h"              // for Sin, RadToDeg, DegToRad, Cos, Tan, etc
#include "TGeoTube.h"

#include <cstdio> // for snprintf

namespace o2
{
namespace trk
{

TRKPetalDisk::TRKPetalDisk(Int_t diskNumber, std::string diskName, Float_t z, Float_t rIn, Float_t rOut, Float_t angularCoverage, Float_t Diskx2X0)
{
  // Creates a simple parametrized petal disk
  mDiskNumber = diskNumber;
  mDiskName = diskName;
  mZ = z;
  mAngularCoverage = angularCoverage;
  mx2X0 = Diskx2X0;
  mInnerRadius = rIn;
  mOuterRadius = rOut;
  Float_t Si_X0 = 9.5;
  mChipThickness = Diskx2X0 * Si_X0;

  LOG(info) << "Creating TRK Disk " << mDiskNumber;
  LOG(info) << "   Using silicon X0 = " << Si_X0 << " to emulate disk radiation length.";
  LOG(info) << "   Disk z = " << mZ << " ; R_in = " << mInnerRadius << " ; R_out = " << mOuterRadius << " ; x2X0 = " << mx2X0 << " ; ChipThickness = " << mChipThickness;
}

void TRKPetalDisk::createDisk(TGeoVolume* motherVolume, TGeoCombiTrans* combiTrans)
{
  // Create tube, set sensitive volume, add to mother volume
  Double_t toDeg = 180 / TMath::Pi();
  std::string chipName = mDiskName + "_" + o2::trk::GeometryTGeo::getTRKChipPattern() + std::to_string(mDiskNumber),
              sensName = mDiskName + "_" + Form("%s%d", GeometryTGeo::getTRKSensorPattern(), mDiskNumber);

  mSensorName = sensName;

  TGeoTubeSeg* sensor = new TGeoTubeSeg(mInnerRadius, mOuterRadius, mChipThickness / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);
  TGeoTubeSeg* chip = new TGeoTubeSeg(mInnerRadius, mOuterRadius, mChipThickness / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);
  TGeoTubeSeg* disk = new TGeoTubeSeg(mInnerRadius, mOuterRadius, mChipThickness / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);

  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kYellow);
  TGeoVolume* diskVol = new TGeoVolume(mDiskName.c_str(), disk, medAir);
  diskVol->SetLineColor(kYellow);

  LOG(info) << "Inserting " << sensVol->GetName() << " inside " << chipVol->GetName();
  chipVol->AddNode(sensVol, 1, nullptr);

  LOG(info) << "Inserting " << chipVol->GetName() << " inside " << diskVol->GetName();
  diskVol->AddNode(chipVol, 1, nullptr);

  // Finally put everything in the mother volume
  TGeoCombiTrans* fwdPetalCombiTrans = new TGeoCombiTrans(*(combiTrans->MakeClone())); // Copy from petal case
  fwdPetalCombiTrans->SetDz(mZ);                                                       // Overwrite z location
  fwdPetalCombiTrans->RegisterYourself();

  LOG(info) << "Inserting " << diskVol->GetName() << " inside " << motherVolume->GetName();
  motherVolume->AddNode(diskVol, 1, fwdPetalCombiTrans);
}
// ClassImp(TRKPetalLayer);

} // namespace trk
} // namespace o2
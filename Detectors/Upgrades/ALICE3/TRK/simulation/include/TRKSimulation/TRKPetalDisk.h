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

/// \file TRKPetalDisk.h
/// \brief Definition of the TRKPetalDisk class

#ifndef ALICEO2_TRK_PETAL_DISK_H_
#define ALICEO2_TRK_PETAL_DISK_H_

#include "TGeoManager.h"       // for gGeoManager
#include "Rtypes.h"            // for Double_t, Int_t, Bool_t, etc
#include <fairlogger/Logger.h> // for LOG

namespace o2
{
namespace trk
{

/// This class defines the Geometry for the TRK Disk TGeo.
class TRKPetalDisk
{
 public:
  TRKPetalDisk() = default;
  TRKPetalDisk(Int_t diskNumber, std::string diskName, Float_t z, Float_t rIn, Float_t rOut, Float_t angularCoverage, Float_t Diskx2X0);
  ~TRKPetalDisk() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getOuterRadius() const { return mOuterRadius; }
  auto getThickness() const { return mChipThickness; }
  auto getAngularCoverage() const { return mAngularCoverage; }
  auto getZ() const { return mZ; }
  auto getx2X0() const { return mx2X0; }
  auto getName() const { return mDiskName; }
  auto getSensorName() const { return mSensorName; }

  /// Creates the actual Disk and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  void createDisk(TGeoVolume* motherVolume, TGeoCombiTrans* combiTrans);

 private:
  Int_t mDiskNumber = -1; ///< Current disk number
  std::string mDiskName;  ///< Current disk name
  std::string mSensorName;
  Double_t mInnerRadius; ///< Inner radius of this disk
  Double_t mOuterRadius; ///< Outer radius of this disk
  Double_t mAngularCoverage;
  Double_t mZ;             ///< Z position of the disk
  Double_t mChipThickness; ///< Chip thickness
  Double_t mx2X0;          ///< Disk material budget x/X0

  ClassDef(TRKPetalDisk, 1);
};
} // namespace trk
} // namespace o2

#endif // ALICEO2_TRK_PETAL_DISK_H

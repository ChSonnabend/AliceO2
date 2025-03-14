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

#ifndef ALICEO2_TRK_PETAL_LAYER_H
#define ALICEO2_TRK_PETAL_LAYER_H

#include "TGeoManager.h"
#include <Rtypes.h>
#include "TGeoTube.h"

#include "TRKBase/TRKBaseParam.h"

namespace o2
{
namespace trk
{
class TRKPetalLayer
{
 public:
  TRKPetalLayer() = default;
  TRKPetalLayer(Int_t layerNumber, std::string layerName, Float_t rIn, Float_t angularCoverage, Float_t zLength, Float_t layerX2X0);
  ~TRKPetalLayer() = default;

  auto getInnerRadius() const { return mInnerRadius; }
  auto getAngularCoverage() const { return mAngularCoverage; }
  auto getZLength() { return mZ; }
  auto getx2X0() const { return mX2X0; }
  auto getChipThickness() const { return mChipThickness; }
  auto getNumber() const { return mLayerNumber; }
  auto getName() const { return mLayerName; }
  auto getSensorName() const { return mSensorName; }

  void createLayer(TGeoVolume* motherVolume, TGeoCombiTrans* combiTrans);

 private:
  Int_t mLayerNumber;
  std::string mLayerName;
  std::string mSensorName;
  Float_t mInnerRadius;
  Float_t mZ;
  Float_t mX2X0;
  Float_t mChipThickness;
  Float_t mModuleWidth;     // u.m. = cm
  Float_t mAngularCoverage; // rad

  TGeoTubeSeg* mLayer;

  ClassDef(TRKPetalLayer, 1);
};

} // namespace trk
} // namespace o2
#endif // ALICEO2_TRK_PETAL_LAYER_H
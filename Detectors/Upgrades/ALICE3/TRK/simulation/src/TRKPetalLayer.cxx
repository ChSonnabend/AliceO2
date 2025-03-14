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

#include "TRKSimulation/TRKPetalLayer.h"
#include "TRKBase/GeometryTGeo.h"

#include "Framework/Logger.h"

#include "TGeoTube.h"
#include "TGeoBBox.h"
#include "TGeoVolume.h"
#include "TGeoTube.h"
#include "TGeoMatrix.h"

#include "TMath.h"

namespace o2
{
namespace trk
{
TRKPetalLayer::TRKPetalLayer(Int_t layerNumber, std::string layerName, Float_t rIn, Float_t angularCoverage, Float_t zLength, Float_t layerX2X0)
  : mLayerNumber(layerNumber), mLayerName(layerName), mInnerRadius(rIn), mAngularCoverage(angularCoverage), mZ(zLength), mX2X0(layerX2X0), mModuleWidth(4.54)
{
  Float_t Si_X0 = 9.5f;
  mChipThickness = mX2X0 * Si_X0;
  LOGP(info, "Creating layer: id: {} rInner: {} thickness: {} zLength: {} x2X0: {}", mLayerNumber, mInnerRadius, mChipThickness, mZ, mX2X0);
}

void TRKPetalLayer::createLayer(TGeoVolume* motherVolume, TGeoCombiTrans* combiTrans)
{
  TGeoMedium* medSi = gGeoManager->GetMedium("TRK_SILICON$");
  TGeoMedium* medAir = gGeoManager->GetMedium("TRK_AIR$");

  std::string staveName = mLayerName + "_" + o2::trk::GeometryTGeo::getTRKStavePattern() + std::to_string(mLayerNumber),
              chipName = mLayerName + "_" + o2::trk::GeometryTGeo::getTRKChipPattern() + std::to_string(mLayerNumber),
              sensName = mLayerName + "_" + Form("%s%d", GeometryTGeo::getTRKSensorPattern(), mLayerNumber);

  mSensorName = sensName;

  Double_t toDeg = 180 / TMath::Pi();
  mLayer = new TGeoTubeSeg(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);
  TGeoVolume* layerVol = new TGeoVolume(mLayerName.c_str(), mLayer, medAir);
  layerVol->SetLineColor(kYellow);

  TGeoTubeSeg* stave = new TGeoTubeSeg(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);
  TGeoTubeSeg* chip = new TGeoTubeSeg(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);
  TGeoTubeSeg* sensor = new TGeoTubeSeg(mInnerRadius, mInnerRadius + mChipThickness, mZ / 2., -0.5 * mAngularCoverage * toDeg, 0.5 * mAngularCoverage * toDeg);

  TGeoVolume* sensVol = new TGeoVolume(sensName.c_str(), sensor, medSi);
  sensVol->SetLineColor(kYellow);
  TGeoVolume* chipVol = new TGeoVolume(chipName.c_str(), chip, medSi);
  chipVol->SetLineColor(kYellow);
  TGeoVolume* staveVol = new TGeoVolume(staveName.c_str(), stave, medSi);
  staveVol->SetLineColor(kYellow);

  LOGP(info, "Inserting {} in {} ", sensVol->GetName(), chipVol->GetName());
  chipVol->AddNode(sensVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", chipVol->GetName(), staveVol->GetName());
  staveVol->AddNode(chipVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", staveVol->GetName(), layerVol->GetName());
  layerVol->AddNode(staveVol, 1, nullptr);

  LOGP(info, "Inserting {} in {} ", layerVol->GetName(), motherVolume->GetName());
  motherVolume->AddNode(layerVol, 1, combiTrans);
}
// ClassImp(TRKPetalLayer);

} // namespace trk
} // namespace o2
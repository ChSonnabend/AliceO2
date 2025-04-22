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

/// \file CreateLUT.C
/// \author David Rohr

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TSystem.h>
#include "DetectorsBase/MatLayerCylSet.h"
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUChainTrackingGetters.inc"

using namespace o2::gpu;

void createLUT()
{
  o2::base::MatLayerCylSet* lut = o2::base::MatLayerCylSet::loadFromFile("matbud.root");
  gSystem->Load("libO2GPUTracking");
  GPUReconstruction* rec = GPUReconstruction::CreateInstance(GPUReconstruction::DeviceType::CPU);
  GPUChainTracking* chain = rec->AddChain<GPUChainTracking>();
  chain->SetMatLUT(lut);
  rec->DumpSettings();
}

#endif

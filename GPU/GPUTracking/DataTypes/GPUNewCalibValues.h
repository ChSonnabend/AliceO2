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

/// \file GPUNewCalibValues.h
/// \author David Rohr

#ifndef GPUNEWCALIBVALUES_H
#define GPUNEWCALIBVALUES_H

#include "GPUCommonDef.h"

namespace o2::gpu
{

struct GPUNewCalibValues {
  bool newSolenoidField = false;
  bool newContinuousMaxTimeBin = false;
  bool newTPCTimeBinCut = false;
  float solenoidField = 0.f;
  uint32_t continuousMaxTimeBin = 0;
  int32_t tpcTimeBinCut = 0;

  void updateFrom(const GPUNewCalibValues* from);
};

} // namespace o2::gpu

#endif

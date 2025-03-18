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

#include "TRKSimulation/DPLDigitizerParam.h"

namespace o2
{
namespace trk
{
// this makes sure that the constructor of the parameters is statically called
// so that these params are part of the parameter database
static auto& sDigitizerParamITS = o2::trk::DPLDigitizerParam<o2::detectors::DetID::TRK>::Instance();
static auto& sDigitizerParamMFT = o2::trk::DPLDigitizerParam<o2::detectors::DetID::FT3>::Instance();
} // namespace trk
} // namespace o2

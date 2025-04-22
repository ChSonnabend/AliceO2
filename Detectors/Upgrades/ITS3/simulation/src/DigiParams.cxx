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

/// \file DigiParams.cxx
/// \brief Implementation of the ITS3 digitization steering params

#include "Framework/Logger.h"
#include "ITS3Simulation/DigiParams.h"

ClassImp(o2::its3::DigiParams);

namespace o2::its3
{

void DigiParams::print() const
{
  // print settings
  LOGF(info, "ITS3 DigiParams settings:");
  LOGF(info, "Continuous readout             : %s", isContinuous() ? "ON" : "OFF");
  LOGF(info, "Readout Frame Length(ns)       : %f", getROFrameLength());
  LOGF(info, "Strobe delay (ns)              : %f", getStrobeDelay());
  LOGF(info, "Strobe length (ns)             : %f", getStrobeLength());
  LOGF(info, "Threshold (N electrons)        : %d", getChargeThreshold());
  LOGF(info, "Min N electrons to account     : %d", getMinChargeToAccount());
  LOGF(info, "Number of charge sharing steps : %d", getNSimSteps());
  LOGF(info, "ELoss to N electrons factor    : %e", getEnergyToNElectrons());
  LOGF(info, "Noise level per pixel          : %e", getNoisePerPixel());
  LOGF(info, "Charge time-response:\n");
  getSignalShape().print();
}

} // namespace o2::its3

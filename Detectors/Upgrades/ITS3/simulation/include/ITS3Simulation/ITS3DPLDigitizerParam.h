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

#ifndef ALICEO2_ITS3DPLDIGITIZERPARAM_H_
#define ALICEO2_ITS3DPLDIGITIZERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::its3
{

struct ITS3DPLDigitizerParam : public o2::conf::ConfigurableParamHelper<ITS3DPLDigitizerParam> {
  float IBNoisePerPixel = 1.e-8; ///< MOSAIX Noise per channel
  int IBChargeThreshold = 150;   ///< charge threshold in Nelectrons for IB
  int IBMinChargeToAccount = 15; ///< minimum charge contribution to account for IB
  int nIBSimSteps = 18;          ///< number of steps in response for IB

  O2ParamDef(ITS3DPLDigitizerParam, "ITS3DPLDigitizerParam");
};

} // namespace o2::its3

#endif
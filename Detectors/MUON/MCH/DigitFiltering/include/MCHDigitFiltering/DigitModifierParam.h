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

#ifndef O2_MCH_DIGITFILTERING_DIGIT_MODIFIER_PARAM_H_
#define O2_MCH_DIGITFILTERING_DIGIT_MODIFIER_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

/**
 * @class DigitModifierParam
 * @brief Configurable parameters for the digit updating
 */
struct DigitModifierParam : public o2::conf::ConfigurableParamHelper<DigitModifierParam> {

  bool updateST1 = false; ///< whether or not to modify ST1 digits
  bool updateST2 = false; ///< whether or not to modify ST2 digits

  O2ParamDef(DigitModifierParam, "MCHDigitModifier");
};

} // namespace o2::mch

#endif

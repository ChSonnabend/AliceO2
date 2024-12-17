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

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_GENERATORPYTHIA8PARAM_H_
#define ALICEO2_EVENTGEN_GENERATORPYTHIA8PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to configure the settings of
 ** the GeneratorPythia8 event generator
 **/
struct Pythia8GenConfig {
  std::string config = "";
  std::string hooksFileName = "";
  std::string hooksFuncName = "";
  bool includePartonEvent = false; // whether to keep the event before hadronization
  std::string particleFilter = ""; // user particle filter
  int verbose = 0;                 // verbose control (if > 0 may show more info messages about what is going on)
};

// construct a configurable param singleton out of the Pythia8GenConfig struct
struct GeneratorPythia8Param : public o2::conf::ConfigurableParamPromoter<GeneratorPythia8Param, Pythia8GenConfig> {
  O2ParamDef(GeneratorPythia8Param, "GeneratorPythia8");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORPYTHIA8PARAM_H_

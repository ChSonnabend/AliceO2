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

/// \author M. Giacalone - October 2024

#ifndef ALICEO2_EVENTGEN_GENERATORHYBRID_H_
#define ALICEO2_EVENTGEN_GENERATORHYBRID_H_

#include "Generators/Generator.h"
#include "Generators/BoxGenerator.h"
#include <Generators/GeneratorPythia8.h>
#include <Generators/GeneratorHepMC.h>
#include <Generators/GeneratorFromFile.h>
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "Generators/GeneratorHybridParam.h"
#include "Generators/GeneratorHepMCParam.h"
#include "Generators/GeneratorPythia8Param.h"
#include "Generators/GeneratorFileOrCmdParam.h"
#include "Generators/GeneratorFromO2KineParam.h"
#include "Generators/GeneratorExternalParam.h"
#include <TRandom3.h>
#include "CommonUtils/ConfigurationMacroHelper.h"
#include "FairGenerator.h"
#include <DetectorsBase/Stack.h>
#include <SimConfig/SimConfig.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>
#include "TBufferJSON.h"

namespace o2
{
namespace eventgen
{

class GeneratorHybrid : public Generator
{

 public:
  GeneratorHybrid() = default;
  GeneratorHybrid(const std::string& inputgens);
  ~GeneratorHybrid() = default;

  Bool_t Init() override;
  Bool_t generateEvent() override;
  Bool_t importParticles() override;

  Bool_t parseJSON(const std::string& path);
  template <typename T>
  std::string jsonValueToString(const T& value);

 private:
  o2::eventgen::Generator* currentgen = nullptr;
  std::vector<std::unique_ptr<o2::eventgen::Generator>> gens;
  const std::vector<std::string> generatorNames = {"extkinO2", "boxgen", "external", "hepmc", "pythia8", "pythia8pp", "pythia8hi", "pythia8hf", "pythia8powheg"};
  std::vector<std::string> mInputGens;
  std::vector<std::string> mGens;
  std::vector<std::string> mConfigs;
  std::vector<std::string> mConfsPythia8;

  // Parameters configurations
  std::vector<std::unique_ptr<o2::eventgen::BoxGenConfig>> mBoxGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::Pythia8GenConfig>> mPythia8GenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::O2KineGenConfig>> mO2KineGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::ExternalGenConfig>> mExternalGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::FileOrCmdGenConfig>> mFileOrCmdGenConfigs;
  std::vector<std::unique_ptr<o2::eventgen::HepMCGenConfig>> mHepMCGenConfigs;

  bool mRandomize = false;
  std::vector<int> mFractions;
  int mseqCounter = 0;
  int mCurrentFraction = 0;
  int mIndex = 0;
};

} // namespace eventgen
} // namespace o2

#endif

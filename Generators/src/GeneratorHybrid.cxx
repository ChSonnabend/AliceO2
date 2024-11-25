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

#include "Generators/GeneratorHybrid.h"
#include <fairlogger/Logger.h>
#include <algorithm>

namespace o2
{
namespace eventgen
{
GeneratorHybrid::GeneratorHybrid(const std::string& inputgens)
{
  if (!parseJSON(inputgens)) {
    LOG(fatal) << "Failed to parse JSON configuration from input generators";
    exit(1);
  }
  mRandomize = GeneratorHybridParam::Instance().randomize;
  if (mConfigs.size() != mInputGens.size()) {
    LOG(fatal) << "Number of configurations does not match the number of generators";
    exit(1);
  }
  if (mConfigs.size() == 0) {
    for (auto gen : mInputGens) {
      mConfigs.push_back("");
    }
  }
  int index = 0;
  if (!mRandomize) {
    if (mFractions.size() != mInputGens.size()) {
      LOG(fatal) << "Number of fractions does not match the number of generators";
      return;
    }
    // Check if all elements of mFractions are 0
    if (std::all_of(mFractions.begin(), mFractions.end(), [](int i) { return i == 0; })) {
      LOG(fatal) << "All fractions provided are 0, no simulation will be performed";
      return;
    }
  }
  for (auto gen : mInputGens) {
    // Search if the generator name is inside generatorNames (which is a vector of strings)
    LOG(info) << "Checking if generator " << gen << " is in the list of available generators \n";
    if (std::find(generatorNames.begin(), generatorNames.end(), gen) != generatorNames.end()) {
      LOG(info) << "Found generator " << gen << " in the list of available generators \n";
      if (gen.compare("boxgen") == 0) {
        if (mConfigs[index].compare("") == 0) {
          gens.push_back(std::make_unique<o2::eventgen::BoxGenerator>());
        } else {
          // Get the index of boxgen configuration
          int confBoxIndex = std::stoi(mConfigs[index].substr(7));
          gens.push_back(std::make_unique<o2::eventgen::BoxGenerator>(*mBoxGenConfigs[confBoxIndex]));
        }
        mGens.push_back(gen);
      } else if (gen.compare(0, 7, "pythia8") == 0) {
        // Check if mConfigs[index] contains pythia8_ and a number
        if (mConfigs[index].compare("") == 0) {
          auto pars = Pythia8GenConfig();
          gens.push_back(std::make_unique<o2::eventgen::GeneratorPythia8>(pars));
        } else {
          // Get the index of pythia8 configuration
          int confPythia8Index = std::stoi(mConfigs[index].substr(8));
          gens.push_back(std::make_unique<o2::eventgen::GeneratorPythia8>(*mPythia8GenConfigs[confPythia8Index]));
        }
        mConfsPythia8.push_back(mConfigs[index]);
        mGens.push_back(gen);
      } else if (gen.compare("extkinO2") == 0) {
        int confO2KineIndex = std::stoi(mConfigs[index].substr(9));
        gens.push_back(std::make_unique<o2::eventgen::GeneratorFromO2Kine>(*mO2KineGenConfigs[confO2KineIndex]));
        mGens.push_back(gen);
      } else if (gen.compare("external") == 0) {
        int confextIndex = std::stoi(mConfigs[index].substr(9));
        auto& extgen_filename = mExternalGenConfigs[confextIndex]->fileName;
        auto& extgen_func = mExternalGenConfigs[confextIndex]->funcName;
        auto extGen = std::unique_ptr<o2::eventgen::Generator>(o2::conf::GetFromMacro<o2::eventgen::Generator*>(extgen_filename, extgen_func, "FairGenerator*", "extgen"));
        if (!extGen) {
          LOG(fatal) << "Failed to load external generator from " << extgen_filename << " with function " << extgen_func;
          exit(1);
        }
        gens.push_back(std::move(extGen));
        mGens.push_back(gen);
      } else if (gen.compare("hepmc") == 0) {
        int confHepMCIndex = std::stoi(mConfigs[index].substr(6));
        gens.push_back(std::make_unique<o2::eventgen::GeneratorHepMC>());
        auto& globalConfig = o2::conf::SimConfig::Instance();
        dynamic_cast<o2::eventgen::GeneratorHepMC*>(gens.back().get())->setup(*mFileOrCmdGenConfigs[confHepMCIndex], *mHepMCGenConfigs[confHepMCIndex], globalConfig);
        mGens.push_back(gen);
      }
    } else {
      LOG(fatal) << "Generator " << gen << " not found in the list of available generators \n";
      exit(1);
    }
    index++;
  }
}

Bool_t GeneratorHybrid::Init()
{
  // init all sub-gens
  int count = 0;
  for (auto& gen : mGens) {
    if (gen == "pythia8pp") {
      auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_inel.cfg";
      LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
      dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
    } else if (gen == "pythia8hf") {
      auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hf.cfg";
      LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
      dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
    } else if (gen == "pythia8hi") {
      auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_hi.cfg";
      LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
      dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
    } else if (gen == "pythia8powheg") {
      auto config = std::string(std::getenv("O2_ROOT")) + "/share/Generators/egconfig/pythia8_powheg.cfg";
      LOG(info) << "Setting \'Pythia8\' base configuration: " << config << std::endl;
      dynamic_cast<o2::eventgen::GeneratorPythia8*>(gens[count].get())->setConfig(config);
    }
    gens[count]->Init();
    addSubGenerator(count, gen);
    count++;
  }
  return Generator::Init();
}

Bool_t GeneratorHybrid::generateEvent()
{
  // Order randomisation or sequence of generators
  // following provided fractions, if not generators are used in proper sequence
  if (mRandomize) {
    mIndex = gRandom->Integer(mGens.size());
  } else {
    while (mFractions[mCurrentFraction] == 0 || mseqCounter == mFractions[mCurrentFraction]) {
      if (mFractions[mCurrentFraction] != 0) {
        mseqCounter = 0;
      }
      mCurrentFraction = (mCurrentFraction + 1) % mFractions.size();
    }
    mIndex = mCurrentFraction;
  }
  if (mConfigs[mIndex].compare("") == 0) {
    LOG(info) << "GeneratorHybrid: generating event with generator " << mGens[mIndex];
  } else {
    LOG(info) << "GeneratorHybrid: generating event with generator " << mConfigs[mIndex];
  }
  gens[mIndex]->clearParticles(); // clear container of this class
  gens[mIndex]->generateEvent();
  // notify the sub event generator
  notifySubGenerator(mIndex);
  mseqCounter++;
  return true;
}

Bool_t GeneratorHybrid::importParticles()
{
  mParticles.clear(); // clear container of mother class
  gens[mIndex]->importParticles();
  std::copy(gens[mIndex]->getParticles().begin(), gens[mIndex]->getParticles().end(), std::back_insert_iterator(mParticles));

  // we need to fix particles statuses --> need to enforce this on the importParticles level of individual generators
  for (auto& p : mParticles) {
    auto st = o2::mcgenstatus::MCGenStatusEncoding(p.GetStatusCode(), p.GetStatusCode()).fullEncoding;
    p.SetStatusCode(st);
    p.SetBit(ParticleStatus::kToBeDone, true);
  }

  return true;
}

template <typename T>
std::string GeneratorHybrid::jsonValueToString(const T& value)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value.Accept(writer);
  return buffer.GetString();
}

Bool_t GeneratorHybrid::parseJSON(const std::string& path)
{
  // Parse JSON file to build map
  std::ifstream fileStream(path, std::ios::in);
  if (!fileStream.is_open()) {
    LOG(error) << "Cannot open " << path;
    return false;
  }
  rapidjson::IStreamWrapper isw(fileStream);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError()) {
    LOG(error) << "Error parsing provided json file " << path;
    LOG(error) << "  - Error -> " << rapidjson::GetParseError_En(doc.GetParseError());
    return false;
  }

  // Put the generator names in mInputGens
  if (doc.HasMember("generators")) {
    const auto& gens = doc["generators"];
    for (const auto& gen : gens.GetArray()) {
      // push in mInputGens the "name" of the generator
      std::string name = gen["name"].GetString();
      mInputGens.push_back(name);
      if (gen.HasMember("config")) {
        if (name == "boxgen") {
          const auto& boxconf = gen["config"];
          auto boxConfig = TBufferJSON::FromJSON<o2::eventgen::BoxGenConfig>(jsonValueToString(boxconf).c_str());
          mBoxGenConfigs.push_back(std::move(boxConfig));
          mConfigs.push_back("boxgen_" + std::to_string(mBoxGenConfigs.size() - 1));
          continue;
        } else if (name == "pythia8") {
          const auto& pythia8conf = gen["config"];
          auto pythia8Config = TBufferJSON::FromJSON<o2::eventgen::Pythia8GenConfig>(jsonValueToString(pythia8conf).c_str());
          mPythia8GenConfigs.push_back(std::move(pythia8Config));
          mConfigs.push_back("pythia8_" + std::to_string(mPythia8GenConfigs.size() - 1));
          continue;
        } else if (name == "extkinO2") {
          const auto& o2kineconf = gen["config"];
          auto o2kineConfig = TBufferJSON::FromJSON<o2::eventgen::O2KineGenConfig>(jsonValueToString(o2kineconf).c_str());
          mO2KineGenConfigs.push_back(std::move(o2kineConfig));
          mConfigs.push_back("extkinO2_" + std::to_string(mO2KineGenConfigs.size() - 1));
          continue;
        } else if (name == "external") {
          const auto& extconf = gen["config"];
          auto extConfig = TBufferJSON::FromJSON<o2::eventgen::ExternalGenConfig>(jsonValueToString(extconf).c_str());
          mExternalGenConfigs.push_back(std::move(extConfig));
          mConfigs.push_back("external_" + std::to_string(mExternalGenConfigs.size() - 1));
          continue;
        } else if (name == "hepmc") {
          const auto& genconf = gen["config"];
          const auto& cmdconf = genconf["configcmd"];
          const auto& hepmcconf = genconf["confighepmc"];
          auto cmdConfig = TBufferJSON::FromJSON<o2::eventgen::FileOrCmdGenConfig>(jsonValueToString(cmdconf).c_str());
          auto hepmcConfig = TBufferJSON::FromJSON<o2::eventgen::HepMCGenConfig>(jsonValueToString(hepmcconf).c_str());
          mFileOrCmdGenConfigs.push_back(std::move(cmdConfig));
          mHepMCGenConfigs.push_back(std::move(hepmcConfig));
          mConfigs.push_back("hepmc_" + std::to_string(mFileOrCmdGenConfigs.size() - 1));
          continue;
        } else {
          mConfigs.push_back("");
        }
      } else {
        if (name == "boxgen" || name == "pythia8" || name == "extkinO2" || name == "external" || name == "hepmc") {
          LOG(fatal) << "No configuration provided for generator " << name;
          return false;
        } else {
          mConfigs.push_back("");
        }
      }
    }
  }

  // Get fractions and put them in mFractions
  if (doc.HasMember("fractions")) {
    const auto& fractions = doc["fractions"];
    for (const auto& frac : fractions.GetArray()) {
      mFractions.push_back(frac.GetInt());
    }
  } else {
    // Set fractions to unity for all generators in case they are not provided
    const auto& gens = doc["generators"];
    for (const auto& gen : gens.GetArray()) {
      mFractions.push_back(1);
    }
  }
  return true;
}

} // namespace eventgen
} // namespace o2

ClassImp(o2::eventgen::GeneratorHybrid);
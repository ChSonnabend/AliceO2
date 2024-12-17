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

#define BOOST_TEST_MODULE Test GeneratorPythia8Param class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <CommonUtils/ConfigurableParam.h>
#include <Generators/GeneratorPythia8Param.h>
#include <boost/property_tree/ptree.hpp>
#include "CCDB/BasicCCDBManager.h"
#include <Generators/GeneratorFromFile.h>
#include <iostream>
#include <filesystem>
#include <unistd.h>

// Tests various aspects of the
// ConfigurableParamPromoter class, which is used to promote
// Pythia8GenConfig to a configurable param
BOOST_AUTO_TEST_CASE(pythia8_Pythia8GenConfig)
{
  o2::conf::ConfigurableParam::updateFromString(
    "GeneratorPythia8.config=Foo;GeneratorPythia8.includePartonEvent=true");

  using o2::eventgen::GeneratorPythia8Param;

  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().config, std::string("Foo"));
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().includePartonEvent, true);

  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().includePartonEvent, o2::conf::ConfigurableParam::getValueAs<bool>("GeneratorPythia8.includePartonEvent"));
  // setValue - getValue
  o2::conf::ConfigurableParam::setValue("GeneratorPythia8.config", "Baz");
  BOOST_CHECK_EQUAL(o2::conf::ConfigurableParam::getValueAs<std::string>("GeneratorPythia8.config"), std::string("Baz"));
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().config, std::string("Baz"));

  // member provenance
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().getMemberProvenance("config"), o2::conf::ConfigurableParam::EParamProvenance::kRT);
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().getMemberProvenance("verbose"), o2::conf::ConfigurableParam::EParamProvenance::kCODE);

  // config detach
  auto config_copy = GeneratorPythia8Param::Instance().detach();
  BOOST_CHECK_EQUAL(config_copy.config, std::string("Baz"));
  BOOST_CHECK_EQUAL(config_copy.includePartonEvent, true);

  // file IO
  TFile tmp_file("GeneratorParamConfig_tmp.root", "RECREATE");

  GeneratorPythia8Param::Instance().serializeTo(&tmp_file);
  // modify the instance to some intermediate fluent value
  o2::conf::ConfigurableParam::setValue("GeneratorPythia8.includePartonEvent", "0");
  BOOST_CHECK_EQUAL(config_copy.includePartonEvent, true);
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().includePartonEvent, false);
  tmp_file.Close();

  // read back
  TFile tmp_file2("GeneratorParamConfig_tmp.root", "READ");
  const_cast<GeneratorPythia8Param&>(GeneratorPythia8Param::Instance()).initFrom(&tmp_file2);
  BOOST_CHECK_EQUAL(GeneratorPythia8Param::Instance().includePartonEvent, true);
  tmp_file2.Close();

  // CCDB IO
  std::string ccdbUrl = "http://ccdb-test.cern.ch:8080";
  bool hostReachable = false;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::string pathA = "/Generators/UnitTest/Pythia8/GeneratorPythia8Param";
  std::map<std::string, std::string> md;
  long start = 1000, stop = 2000;
  api.storeAsTFileAny(&GeneratorPythia8Param::Instance(), pathA, md, start, stop);

  // modify the instance to some intermediate fluent value
  o2::conf::ConfigurableParam::setValue("GeneratorPythia8.includePartonEvent", "0");

  auto returnedobj = api.retrieveFromTFileAny<o2::eventgen::GeneratorPythia8Param>(pathA, md, (start + stop) / 2);
  GeneratorPythia8Param::Instance().printKeyValues();
};

BOOST_AUTO_TEST_CASE(EventPool_Alien_Path)
{
  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = "alien:///alice/cern.ch/user/s/swenzel/selfjobs/evtpool_pythia8pp_test-20241126-152715";
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() > 0);
};

BOOST_AUTO_TEST_CASE(EventPool_Alien_File)
{
  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = "alien:///alice/cern.ch/user/s/swenzel/selfjobs/evtpool_pythia8pp_test-20241126-152715/001/evtpool.root";
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == 1);
};

BOOST_AUTO_TEST_CASE(EventPool_Alien_WrongFileName)
{
  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = "alien:///foo_123";
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == 0);
};

BOOST_AUTO_TEST_CASE(EventPool_Local_Path)
{
  namespace fs = std::filesystem;

  // we need to create some local tmp files that mimick the event pool
  // this is a helper to do this
  auto createPoolFiles = [](const fs::path& tmpDir, int numFiles) {
    for (int i = 0; i < numFiles; ++i) {
      // Generate a unique file name
      fs::path fileDir = tmpDir / std::to_string(i);
      fs::path filePath = fileDir / o2::eventgen::GeneratorFromEventPool::eventpool_filename;
      fs::create_directory(fileDir);
      // Create and close the file (touch)
      std::ofstream file(filePath);
      file.close();
    }
  };

  // Seed for randomness
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  // process id
  auto proc = getpid();

  // Create a random directory in the system temp directory
  fs::path tmpDir = fs::temp_directory_path() / ("eventpool_test_" + std::to_string(proc) + "_" + std::to_string(std::rand()));
  fs::create_directory(tmpDir);
  constexpr int numfiles = 11;
  createPoolFiles(tmpDir, numfiles);

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = tmpDir.string();
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == numfiles);

  // remove the files
  if (fs::exists(tmpDir)) {
    fs::remove_all(tmpDir); // Remove all files and the directory
  }
};

BOOST_AUTO_TEST_CASE(EventPool_Local_RootFile)
{
  namespace fs = std::filesystem;

  // we need to create a fake local root file in the right format
  // Seed for randomness
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  // process id
  auto proc = getpid();
  // Create a random directory in the system temp directory
  fs::path tmpDir = fs::temp_directory_path() / ("eventpool_testlocalrootfile_" + std::to_string(proc) + "_" + std::to_string(std::rand()));
  //
  fs::path filePath = tmpDir / o2::eventgen::GeneratorFromEventPool::eventpool_filename;
  fs::create_directory(tmpDir);
  // Create and close the file (touch); needs to be a ROOT file so using TFile
  TFile file(filePath.string().c_str(), "CREATE");
  file.Close();

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = tmpDir.string() + "/evtpool.root";
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == 1);

  // remove the files
  if (fs::exists(tmpDir)) {
    fs::remove_all(tmpDir); // Remove all files and the directory
  }
};

BOOST_AUTO_TEST_CASE(EventPool_Local_ListFile)
{
  // test reading list of files from a (txt) file
  // create this txt file on the fly

  namespace fs = std::filesystem;

  std::srand(static_cast<unsigned>(std::time(nullptr)));
  // process id
  auto proc = getpid();
  // Create a random directory in the system temp directory
  fs::path tmpDir = fs::temp_directory_path() / ("eventpool_testlocallistfile_" + std::to_string(proc) + "_" + std::to_string(std::rand()));
  fs::create_directory(tmpDir);

  std::ofstream file(tmpDir / std::string("filelist.txt"));

  constexpr int numfiles = 11;
  for (int i = 0; i < numfiles; ++i) {
    // Generate a unique file name
    fs::path filePath = fs::path(std::string("alien:///foo")) / std::to_string(i) / o2::eventgen::GeneratorFromEventPool::eventpool_filename;
    file << filePath.string() << "\n";
  }
  file.close();

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = tmpDir.string() + std::string("/filelist.txt");
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == numfiles);

  // remove the files
  if (fs::exists(tmpDir)) {
    fs::remove_all(tmpDir); // Remove all files and the directory
  }
};

BOOST_AUTO_TEST_CASE(EventPool_Local_WrongPath)
{
  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = "/tmp/MyEvtPool/filelist_DOESNOTEXIST.txt";
  o2::eventgen::GeneratorFromEventPool gen(config);
  auto files = gen.setupFileUniverse(config.eventPoolPath);
  BOOST_CHECK(files.size() == 0);
};
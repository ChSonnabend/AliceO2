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

/// \file CreateCTPConfig.C
/// \brief create CTP config, test it and add to database
/// \author Roman Lietava

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
int CheckCTPConfig(std::string cfgRun3str = "/home/rl/backup24/runs/559781.rcfg2", int writeToFile = 0)
{
  //
  // run3 config
  //
  if (cfgRun3str.find(".rcfg") == std::string::npos) {
    std::cout << "No file name:" << cfgRun3str << std::endl;
    return 1;
  } else {
    std::string filename = cfgRun3str;
    std::ifstream in;
    in.open(filename);
    if (!in) {
      std::cout << "Can not open file:" << filename << std::endl;
      return 2;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    cfgRun3str = buffer.str();
  }
  //
  CTPConfiguration ctpcfg;
  int ret = ctpcfg.loadConfigurationRun3(cfgRun3str);
  ctpcfg.printStream(std::cout);
  std::cout << "CTP config done" << std::endl;
  // ctpcfg.checkConfigConsistency();
  auto ctpclasses = ctpcfg.getCTPClasses();
  for (auto const& cls : ctpclasses) {
    std::cout << cls.descriptor->name << ":" << std::hex << cls.descriptor->getInputsMask() << std::endl;
  }
  return ret;
}

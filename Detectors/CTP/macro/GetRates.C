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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsCTP/Configuration.h>
#include "CTPWorkflowScalers/ctpCCDBManager.h"
#include "Framework/Logger.h"
#endif
using namespace o2::ctp;

void GetRates(int run = 559617)
{
  uint64_t inputmaskCum = 0, classmackCum = 0;
  int ntrigSel = 0;

  auto& cmb = o2::ccdb::BasicCCDBManager::instance();
  auto ctpcfg = cmb.getSpecificForRun<o2::ctp::CTPConfiguration>("CTP/Config/Config", run);
  if (!ctpcfg) {
    LOGP(error, "Can not get config for run {}", run);
    return;
  }
  CTPConfiguration ctpconfig;
  ctpconfig.loadConfigurationRun3(ctpcfg->getConfigString());
  ctpconfig.printStream(std::cout);
  auto& triggerclasses = ctpconfig.getCTPClasses();
  LOGP(info, "Found {} trigger classes", triggerclasses.size());
  int indexInList = 0;
  for (const auto& trgclass : triggerclasses) {
    uint64_t inputmask = 0;
    if (trgclass.descriptor != nullptr) {
      inputmask = trgclass.descriptor->getInputsMask();
      // LOGP(info, "inputmask: {:#x}", inputmask);
    }
    trgclass.printStream(std::cout);
    //    std::cout << indexInList << ": " << trgclass.name << ", input mask 0x" << std::hex << inputmask << ", class mask 0x" << trgclass.classMask << std::dec << std::endl;
    indexInList++;
    if (trgclass.cluster->getClusterDetNames().find("TRD") != std::string::npos || trgclass.cluster->getClusterDetNames().find("trd") != std::string::npos) {
      LOGP(info, "Found TRD trigger cluster, class mask: {:#x}, input mask: {:#x}", trgclass.classMask, inputmask);
      inputmaskCum |= inputmask;
      classmackCum |= trgclass.classMask;
      ntrigSel++;
    }
  }

  LOGP(info, "Found {} triggers with TRD: classMasks: {:#x}  inputMasks: {:#x}", ntrigSel, classmackCum, inputmaskCum);
}

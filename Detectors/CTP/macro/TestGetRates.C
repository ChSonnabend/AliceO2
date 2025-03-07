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
#include <DataFormatsCTP/CTPRateFetcher.h>
#endif
using namespace o2::ctp;

void TestGetRates(int runN = 0)
{
  std::vector<int> runs;
  std::vector<std::string> codes = {"T0VTX", "T0VTX", "ZNChadronic", "ZNChadronic", "T0VTX"};
  if (runN == 0) {
    runs = {529066, 539218, 544013, 544518, 557251};
  } else {
    runs.push_back(runN);
  }
  auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
  int i = 0;
  for (auto const& runNumber : runs) {
    // Opening run
    std::pair<int64_t, int64_t> pp = ccdb.getRunDuration(runNumber);
    long ts = pp.first + 60;
    // std::cout << "Run duration:" << pp.first << " " << pp.second << std::endl;
    std::cout << "===> RUN:" << runNumber << " duration:" << (pp.second - pp.first) / 1000. << std::endl;

    CTPRateFetcher fetcher;
    fetcher.setupRun(runNumber, &ccdb, ts, 1);
    fetcher.setOrbit(1);
    std::array<double, 3> rates;
    fetcher.getRates(rates, &ccdb, runNumber, codes[i]);
    std::cout << "Start:" << rates[0] << " End:" << rates[1] << " Middle:" << rates[2] << " code:" << codes[i] << std::endl;
    double lumi1 = fetcher.getLumi(&ccdb, runNumber, codes[i], 0);
    double lumi2 = fetcher.getLumi(&ccdb, runNumber, codes[i], 1);
    std::cout << " Lumi NO pile up corr:" << lumi1 << " Lumi with pile upcorr:" << lumi2 << " code:" << codes[i] << std::endl;
    i++;
  }
}

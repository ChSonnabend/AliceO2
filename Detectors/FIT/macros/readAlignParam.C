// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  readAlignParam.C
/// \brief ROOT macro for reading geometry alignment parameters
///
/// \author Andreas Molander <andreas.molander@cern.ch>

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/BasicCCDBManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

#include <string>
#include <vector>

#endif

int readAlignParam(const std::string& detectorName = "FT0",
                   long timestamp = -1,
                   const std::string& ccdbUrl = "https://alice-ccdb.cern.ch")
{
  o2::ccdb::BasicCCDBManager& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
  ccdbManager.setURL(ccdbUrl);
  ccdbManager.setTimestamp(timestamp);

  const o2::detectors::DetID detID(detectorName.c_str());
  const std::string alignmentPath = o2::base::DetectorNameConf::getAlignmentPath(detID);
  const auto alignments = ccdbManager.get<std::vector<o2::detectors::AlignParam>>(alignmentPath);

  if (!alignments) {
    std::cerr << "No alignment parameters found at " << alignmentPath << std::endl;
    return 1;
  }

  for (auto alignment : *alignments) {
    alignment.print();
  }

  return 0;
}
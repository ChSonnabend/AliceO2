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

#include "DataFormatsFIT/LookUpTable.h"
#include "CCDB/BasicCCDBManager.h"
#include <unordered_map>
using namespace o2::fit;
template <typename MapEntryCRU2ModuleType, typename MapEntryPM2ChannelID>
void LookupTableBase<MapEntryCRU2ModuleType, MapEntryPM2ChannelID>::initCCDB(const std::string& urlCCDB, const std::string& pathToStorageInCCDB, long timestamp)
{

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(urlCCDB);
  mVecEntryFEE = *(mgr.getForTimeStamp<LookupTableBase<MapEntryCRU2ModuleType, MapEntryPM2ChannelID>::Table_t>(pathToStorageInCCDB, timestamp));
  prepareLUT();
}
template class o2::fit::LookupTableBase<std::unordered_map<EntryCRU, EModuleType, HasherCRU, ComparerCRU>,
                                        std::unordered_map<EntryPM, int, HasherPM, ComparerPM>>;

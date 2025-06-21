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

#include <format>
#include "ITStracking/Constants.h"

#include "ITStracking/Configuration.h"

namespace o2::its
{

std::string asString(TrackingMode mode)
{
  switch (mode) {
    case TrackingMode::Sync:
      return "sync";
    case TrackingMode::Async:
      return "async";
    case TrackingMode::Cosmics:
      return "cosmics";
    case TrackingMode::Unset:
      return "unset";
  }
  return "unknown";
}

std::ostream& operator<<(std::ostream& os, TrackingMode v)
{
  os << asString(v);
  return os;
}

std::string TrackingParameters::asString() const
{
  std::string str = std::format("NZb:{} NPhB:{} NROFIt:{} PerVtx:{} DropFail:{} ClSh:{} TtklMinPt:{:.2f} MinCl:{}",
                                ZBins, PhiBins, nROFsPerIterations, PerPrimaryVertexProcessing, DropTFUponFailure, ClusterSharing, TrackletMinPt, MinTrackLength);
  bool first = true;
  for (int il = NLayers; il >= MinTrackLength; il--) {
    int slot = NLayers - il;
    if (slot < (int)MinPt.size() && MinPt[slot] > 0) {
      if (first) {
        first = false;
        str += " MinPt: ";
      }
      str += std::format("L{}:{:.2f} ", il, MinPt[slot]);
    }
  }
  str += " SystErrY/Z:";
  for (size_t i = 0; i < SystErrorY2.size(); i++) {
    str += std::format("{:.2e}/{:.2e} ", SystErrorY2[i], SystErrorZ2[i]);
  }
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / constants::GB);
  }
  return str;
}

std::string VertexingParameters::asString() const
{
  std::string str = std::format("NZb:{} NPhB:{} DRof:{} ClsCont:{} MaxTrkltCls:{} ZCut:{} PhCut:{}", ZBins, PhiBins, deltaRof, clusterContributorsCut, maxTrackletsPerCluster, zCut, phiCut);
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / constants::GB);
  }
  return str;
}

} // namespace o2::its

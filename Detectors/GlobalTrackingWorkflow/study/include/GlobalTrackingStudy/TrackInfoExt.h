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

// class for extended Track info (for debugging)

#ifndef ALICEO2_TRINFOEXT_H
#define ALICEO2_TRINFOEXT_H

#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace dataformats
{

struct TrackInfoExt {
  o2::track::TrackParCov track;
  DCA dca{};
  DCA dcaTPC{};
  VtxTrackIndex gid;
  MatchInfoTOF infoTOF;
  std::array<float, 3> innerTPCPos{};  // innermost cluster position at assigned time
  std::array<float, 3> innerTPCPos0{}; // innermost cluster position at nominal time0
  float ttime = 0;
  float ttimeE = 0;
  float xmin = 0;
  float chi2ITSTPC = 0.f;
  float q2ptITS = 0.f;
  float q2ptTPC = 0.f;
  float q2ptITSTPC = 0.f;
  float q2ptITSTPCTRD = 0.f;
  uint16_t nClTPC = 0;
  uint16_t nClTPCShared = 0;
  uint8_t pattITS = 0;
  uint8_t nClITS = 0;
  uint8_t rowMinTPC = 0;
  uint8_t padFromEdge = -1;
  uint8_t rowMaxTPC = 0;
  uint8_t rowCountTPC = 0;

  float getTPCInX() const { return innerTPCPos[0]; }
  float getTPCInY() const { return innerTPCPos[1]; }
  float getTPCInZ() const { return innerTPCPos[2]; }
  float getTPCInX0() const { return innerTPCPos0[0]; }
  float getTPCInY0() const { return innerTPCPos0[1]; }
  float getTPCInZ0() const { return innerTPCPos0[2]; }

  ClassDefNV(TrackInfoExt, 5);
};

} // namespace dataformats
} // namespace o2

#endif

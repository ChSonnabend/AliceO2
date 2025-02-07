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

/// \file GPUTRDTrackerDebug.h
/// \brief For performance analysis + error parametrization of the TRD tracker

/// \author Ole Schmidt

#ifndef GPUTRDTRACKERDEBUG_H
#define GPUTRDTRACKERDEBUG_H

#if defined(ENABLE_GPUTRDDEBUG) && 0

// could implement debug code, as we had for AliRoot

#else

namespace o2
{
namespace gpu
{

template <class T>
class GPUTRDTrackerDebug
{
 public:
  GPUd() void CreateStreamer() {}
  GPUd() void ExpandVectors() {}
  GPUd() void Reset() {}

  // general information
  GPUd() void SetGeneralInfo(int32_t iEv, int32_t nTPCtracks, int32_t iTrk, float pt) {}

  // track parameters
  GPUd() void SetTrackParameter(const T& trk, int32_t ly) {}
  GPUd() void SetTrackParameterNoUp(const T& trk, int32_t ly) {}
  GPUd() void SetTrack(const T& trk) {}

  // tracklet parameters
  GPUd() void SetRawTrackletPosition(const float fX, const float fY, const float fZ, int32_t ly) {}
  GPUd() void SetCorrectedTrackletPosition(const float* fYZ, int32_t ly) {}
  GPUd() void SetTrackletCovariance(const float* fCov, int32_t ly) {}
  GPUd() void SetTrackletProperties(const float dy, const int32_t det, int32_t ly) {}

  // update information
  GPUd() void SetChi2Update(float chi2, int32_t ly) {}
  GPUd() void SetChi2YZPhiUpdate(float chi2, int32_t ly) {}

  // other infos
  GPUd() void SetRoad(float roadY, float roadZ, int32_t ly) {}
  GPUd() void SetFindable(bool* findable) {}
  GPUd() void Output() {}
};
#if !defined(GPUCA_GPUCODE) || defined(GPUCA_GPUCODE_DEVICE) // FIXME: DR: WORKAROUND to avoid CUDA bug creating host symbols for device code.
template class GPUTRDTrackerDebug<GPUTRDTrackGPU>;
#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
template class GPUTRDTrackerDebug<GPUTRDTrack>;
#endif
#endif
} // namespace gpu
} // namespace o2

#endif
#endif // GPUTRDTRACKERDEBUG_H

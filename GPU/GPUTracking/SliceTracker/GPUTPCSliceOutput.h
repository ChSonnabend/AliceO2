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

/// \file GPUTPCSliceOutput.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCSLICEOUTPUT_H
#define GPUTPCSLICEOUTPUT_H

#include "GPUTPCDef.h"
#include "GPUTPCTrack.h"

namespace o2
{
namespace gpu
{
struct GPUOutputControl;

/**
 * @class GPUTPCSliceOutput
 *
 * GPUTPCSliceOutput class is used to store the output of GPUTPCTracker{Component}
 * and transport the output to GPUTPCGBMerger{Component}
 *
 * The class contains all the necessary information about TPC tracks, reconstructed in one slice.
 * This includes the reconstructed track parameters and some compressed information
 * about the assigned clusters: clusterId, position and amplitude.
 *
 */
class GPUTPCSliceOutput
{
 public:
  GPUhd() uint32_t NTracks() const
  {
    return mNTracks;
  }
  GPUhd() uint32_t NLocalTracks() const { return mNLocalTracks; }
  GPUhd() uint32_t NTrackClusters() const { return mNTrackClusters; }
  GPUhd() const GPUTPCTrack* GetFirstTrack() const
  {
    return (const GPUTPCTrack*)((const char*)this + sizeof(*this));
  }
  GPUhd() GPUTPCTrack* FirstTrack()
  {
    return (GPUTPCTrack*)((char*)this + sizeof(*this));
  }
  GPUhd() size_t Size() const
  {
    return (mMemorySize);
  }

  static uint32_t EstimateSize(uint32_t nOfTracks, uint32_t nOfTrackClusters);
  static void Allocate(GPUTPCSliceOutput*& ptrOutput, int32_t nTracks, int32_t nTrackHits, GPUOutputControl* outputControl, void*& internalMemory);

  GPUhd() void SetNTracks(uint32_t v) { mNTracks = v; }
  GPUhd() void SetNLocalTracks(uint32_t v) { mNLocalTracks = v; }
  GPUhd() void SetNTrackClusters(uint32_t v) { mNTrackClusters = v; }

 private:
  GPUTPCSliceOutput() = delete;                                    // NOLINT: Must be private or ROOT tries to use them!
  ~GPUTPCSliceOutput() = delete;                                   // NOLINT
  GPUTPCSliceOutput(const GPUTPCSliceOutput&) = delete;            // NOLINT
  GPUTPCSliceOutput& operator=(const GPUTPCSliceOutput&) = delete; // NOLINT

  GPUhd() void SetMemorySize(size_t val) { mMemorySize = val; }

  uint32_t mNTracks; // number of reconstructed tracks
  uint32_t mNLocalTracks;
  uint32_t mNTrackClusters;     // total number of track clusters
  size_t mMemorySize;           // Amount of memory really used
};
} // namespace gpu
} // namespace o2
#endif

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

/// \file ClusterAccumulator.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CLUSTER_ACCUMULATOR_H
#define O2_GPU_CLUSTER_ACCUMULATOR_H

#include "clusterFinderDefs.h"
#include "PackedCharge.h"
#include "Array2D.h"

namespace o2
{

namespace tpc
{
struct ClusterNative;
}

namespace gpu
{

struct ChargePos;
struct GPUParam;
class GPUTPCGeometry;

class ClusterAccumulator
{

 public:
  GPUd() tpccf::Charge updateInner(PackedCharge, tpccf::Delta2);
  GPUd() tpccf::Charge updateOuter(PackedCharge, tpccf::Delta2);

  GPUd() void setFull(float qtot, float padMean, float padSigma, float timeMean, float timeSigma, uint8_t splitInPad, uint8_t splitInTime)
  {
    mQtot = qtot;
    mPadMean = padMean;
    mPadSigma = padSigma;
    mTimeMean = timeMean;
    mTimeSigma = timeSigma;
    mSplitInPad = splitInPad;
    mSplitInTime = splitInTime;
  }

  GPUd() void finalize(const ChargePos&, const tpccf::Charge, tpccf::TPCTime);
  GPUd() bool toNative(const ChargePos&, const tpccf::Charge, tpc::ClusterNative&, const GPUParam&, const Array2D<PackedCharge>&);

 private:
  float mQtot = 0;
  float mPadMean = 0;
  float mPadSigma = 0;
  float mTimeMean = 0;
  float mTimeSigma = 0;
  uint8_t mSplitInTime = 0;
  uint8_t mSplitInPad = 0;

  GPUd() void update(tpccf::Charge, tpccf::Delta2);
};

} // namespace gpu
} // namespace o2

#endif

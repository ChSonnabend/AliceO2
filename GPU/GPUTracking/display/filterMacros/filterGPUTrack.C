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

#include "GPUO2Interface.h"
#include "GPUTPCGMMergedTrack.h"

namespace o2::gpu
{
struct GPUConstantMem;
};

using namespace o2::gpu;

void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
{
  for (uint32_t i = 0; i < filter->size(); i++) {
    auto& trk = ioPtrs->mergedTracks[i];
    (*filter)[i] = fabsf(trk.GetParam().GetQPt()) < 1.0f;
  }
}

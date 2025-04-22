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

/// \file GPUKernelClassesFwd.h
/// \author David Rohr

#ifndef GPUKERNELCLASSESFWDN_H
#define GPUKERNELCLASSESFWDN_H

#include "GPUTRDDef.h"

namespace o2::gpu
{
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward, x_types, ...) class GPUCA_M_FIRST(GPUCA_M_STRIP(x_class));
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL

struct GPUTPCClusterOccupancyMapBin;
namespace gputpcgmmergertypes
{
struct GPUTPCGMBorderRange;
}
struct GPUTPCLinearLabels;
struct CfChargePos;
} // namespace o2::gpu

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

#endif

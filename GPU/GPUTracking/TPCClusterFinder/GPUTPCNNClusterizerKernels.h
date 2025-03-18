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

/// \file GPUTPCNNClusterizerKernels.h
/// \author Christian Sonnabend

#ifndef O2_GPU_NN_CLUSTERIZER_H
#define O2_GPU_NN_CLUSTERIZER_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "GPUTPCNNClusterizer.h"

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

namespace o2::gpu
{

class ClusterAccumulator;
class MCLabelAccumulator;

class GPUTPCNNClusterizerKernels : public GPUKernelTemplate
{
 public:
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCNNClusterizerKernels);
  struct GPUSharedMemory {
    // Regular cluster finder
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
    uint8_t innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
  };

  GPUhdi() constexpr static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  enum K : int32_t {
    runCfClusterizer = 0,
    fillInputNN = 1,
    determineClass1Labels = 2,
    determineClass2Labels = 3,
    publishClass1Regression = 4,
    publishClass2Regression = 5,
    removeAllSplitFlags = 6,
    setDeconvolutionFlags = 7,
    publishDeconvolutionFlags = 8
  };

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t, int32_t, int32_t, int32_t, GPUSharedMemory&, processorType&, uint8_t = 0, int8_t = 0, int8_t = 0, uint = 0, Args...);

 private:
  static GPUd() void fillInputData(int32_t, int32_t, int32_t, int32_t, processorType&, uint8_t, int8_t, uint);
  static GPUd() void publishClustersReg1(uint, GPUSharedMemory&, processorType&, uint8_t, int8_t, int8_t, uint);
  static GPUd() void publishClustersReg2(uint, GPUSharedMemory&, processorType&, uint8_t, int8_t, int8_t, uint);

  static GPUd() int padOffset(int, int);
  static GPUd() int rowOffset(int, int);
  static GPUd() bool isBoundary(int, int, int, const GPUTPCGeometry&);

  static GPUdi() uint8_t countPeaksInner(uint16_t, const uint8_t*, uint8_t*);
  static GPUdi() uint8_t countPeaksOuter(uint16_t, uint8_t, const uint8_t*);
  static GPUd() bool isBoundary(int, int, int);
};

} // namespace o2::gpu

#endif

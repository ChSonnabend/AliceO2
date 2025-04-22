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

/// \file dumpGPUDefParam.C
/// \author David Rohr

// Run e.g. as:
// ROOT_INCLUDE_PATH="`pwd`/include" root -l -q -b src/GPU/GPUTracking/Standalone/tools/dumpGPUDefParam.C'()'

// Logic for testing to load the default parameters
/* #define GPUCA_GPUCODE
#define GPUCA_GPUTYPE_AMPERE
#define GPUCA_MAXN 40
#define GPUCA_ROW_COUNT 152
#define GPUCA_TPC_COMP_CHUNK_SIZE 1024
#include "GPUDefParametersConstants.h"
#include "GPUDefParametersDefaults.h" */

// Alternatively, logic to load file that sets GPUDefParameters
#include "testParam.h"

#include "GPUDefParametersLoad.inc"
void dumpGPUDefParam()
{
  auto param = o2::gpu::internal::GPUDefParametersLoad();
  printf("Loaded params:\n%s", o2::gpu::internal::GPUDefParametersExport(param, false).c_str());
  FILE* fp = fopen("parameters.out", "w+b");
  fwrite(&param, 1, sizeof(param), fp);
  fclose(fp);
}

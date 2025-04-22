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

/// \file GPUReconstructionOCLKernelsSpecialize.inc
/// \author David Rohr

template <>
inline void GPUReconstructionOCL::runKernelBackend<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  cl_int4 val0 = {0, 0, 0, 0};
  GPUChkErr(clEnqueueFillBuffer(mInternals->command_queue[_xyz.x.stream], mInternals->mem_gpu, &val0, sizeof(val0), (char*)ptr - (char*)mDeviceMemoryBase, (size + sizeof(val0) - 1) & ~(sizeof(val0) - 1), _xyz.z.evList == nullptr ? 0 : _xyz.z.nEvents, _xyz.z.evList->getEventList<cl_event>(), _xyz.z.ev->getEventList<cl_event>()));
}

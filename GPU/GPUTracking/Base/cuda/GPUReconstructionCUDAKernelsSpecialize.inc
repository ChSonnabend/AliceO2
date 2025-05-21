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

/// \file GPUReconstructionCUDAKernelsSpecialize.inc
/// \author David Rohr

#if defined(GPUCA_SPECIALIZE_THRUST_SORTS) && !defined(GPUCA_GPUCODE_COMPILEKERNELS)

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerMergeBorders, 3>(const krnlSetupTime& _xyz, GPUTPCGMBorderRange* const& range, int32_t const& N, int32_t const& cmpMax)
{
  if (cmpMax) {
    GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, range, N, MergeBorderTracks_compMax());
  } else {
    GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, range, N, MergeBorderTracks_compMin());
  }
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerSortTracks, 0>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackOrderProcess(), processors()->tpcMerger.NMergedTracks(), GPUTPCGMMergerSortTracks_comp(mProcessorsShadow->tpcMerger.MergedTracks()));
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerSortTracksQPt, 0>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackSort(), processors()->tpcMerger.NMergedTracks(), GPUTPCGMMergerSortTracksQPt_comp(mProcessorsShadow->tpcMerger.MergedTracks()));
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerMergeLoopers, 1>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.LooperCandidates(), processors()->tpcMerger.Memory()->nLooperMatchCandidates, GPUTPCGMMergerMergeLoopers_comp());
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMO2Output, GPUTPCGMO2Output::sort>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackSortO2(), processors()->tpcMerger.NOutputTracksTPCO2(), GPUTPCGMO2OutputSort_comp());
}
#endif // GPUCA_SPECIALIZE_THRUST_SORTS

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  GPUChkErr(cudaMemsetAsync(ptr, 0, size, mInternals->Streams[_xyz.x.stream]));
}

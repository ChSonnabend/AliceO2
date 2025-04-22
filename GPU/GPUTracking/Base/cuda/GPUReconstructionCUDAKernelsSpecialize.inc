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

namespace o2::gpu::internal
{
namespace // anonymous
{
struct MergeBorderTracks_compMax {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMax != b.fMax) ? (a.fMax < b.fMax) : (a.fId < b.fId), a.fMax < b.fMax);
  }
};
struct MergeBorderTracks_compMin {
  GPUd() bool operator()(const GPUTPCGMBorderRange& a, const GPUTPCGMBorderRange& b)
  {
    return GPUCA_DETERMINISTIC_CODE((a.fMin != b.fMin) ? (a.fMin < b.fMin) : (a.fId < b.fId), a.fMin < b.fMin);
  }
};

struct GPUTPCGMMergerSortTracks_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracks_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    if (a.CCE() != b.CCE()) {
      return a.CCE() > b.CCE();
    }
    if (a.Legs() != b.Legs()) {
      return a.Legs() > b.Legs();
    }
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (a.NClusters() != b.NClusters()) {
        return a.NClusters() > b.NClusters();
      } if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return aa > bb;
    , // !GPUCA_DETERMINISTIC_CODE
      return a.NClusters() > b.NClusters();
    ) // clang-format on
  }
};

struct GPUTPCGMMergerSortTracksQPt_comp {
  const GPUTPCGMMergedTrack* const mCmp;
  GPUhd() GPUTPCGMMergerSortTracksQPt_comp(GPUTPCGMMergedTrack* cmp) : mCmp(cmp) {}
  GPUd() bool operator()(const int32_t aa, const int32_t bb)
  {
    const GPUTPCGMMergedTrack& GPUrestrict() a = mCmp[aa];
    const GPUTPCGMMergedTrack& GPUrestrict() b = mCmp[bb];
    GPUCA_DETERMINISTIC_CODE( // clang-format off
      if (CAMath::Abs(a.GetParam().GetQPt()) != CAMath::Abs(b.GetParam().GetQPt())) {
        return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
      } if (a.GetParam().GetY() != b.GetParam().GetY()) {
        return a.GetParam().GetY() > b.GetParam().GetY();
      }
      return a.GetParam().GetZ() > b.GetParam().GetZ();
    , // !GPUCA_DETERMINISTIC_CODE
      return CAMath::Abs(a.GetParam().GetQPt()) > CAMath::Abs(b.GetParam().GetQPt());
    ) // clang-format on
  }
};

struct GPUTPCGMMergerMergeLoopers_comp {
  GPUd() bool operator()(const MergeLooperParam& a, const MergeLooperParam& b)
  {
    return CAMath::Abs(a.refz) < CAMath::Abs(b.refz);
  }
};

struct GPUTPCGMO2OutputSort_comp {
  GPUd() bool operator()(const GPUTPCGMMerger::tmpSort& a, const GPUTPCGMMerger::tmpSort& b)
  {
    return (a.y > b.y);
  }
};

} // anonymous namespace
} // namespace o2::gpu::internal

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
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackOrderProcess(), processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracks_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
}

template <>
inline void GPUCA_M_CAT(GPUReconstruction, GPUCA_GPUTYPE)::runKernelBackendTimed<GPUTPCGMMergerSortTracksQPt, 0>(const krnlSetupTime& _xyz)
{
  GPUCommonAlgorithm::sortOnDevice(this, _xyz.x.stream, mProcessorsShadow->tpcMerger.TrackSort(), processors()->tpcMerger.NOutputTracks(), GPUTPCGMMergerSortTracksQPt_comp(mProcessorsShadow->tpcMerger.OutputTracks()));
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

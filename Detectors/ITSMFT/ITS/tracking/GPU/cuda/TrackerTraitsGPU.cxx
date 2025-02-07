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
///

#include <array>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "DataFormatsITS/TrackITS.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStracking/TrackingConfigParam.h"
namespace o2::its
{
constexpr int UnusedIndex{-1};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialise(iteration, mTrkParams[iteration], nLayers);
  mTimeFrameGPU->loadClustersDevice(iteration);
  mTimeFrameGPU->loadUnsortedClustersDevice(iteration);
  mTimeFrameGPU->loadClustersIndexTables(iteration);
  mTimeFrameGPU->loadTrackingFrameInfoDevice(iteration);
  mTimeFrameGPU->loadMultiplicityCutMask(iteration);
  mTimeFrameGPU->loadVertices(iteration);
  mTimeFrameGPU->loadROframeClustersDevice(iteration);
  mTimeFrameGPU->createUsedClustersDevice(iteration);
  mTimeFrameGPU->loadIndexTableUtils(iteration);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerTracklets(const int iteration, int, int)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerCells(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::extendTracks(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::setBz(float bz)
{
  mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfTracklets() const
{
  return std::accumulate(mTimeFrameGPU->getNTracklets().begin(), mTimeFrameGPU->getNTracklets().end(), 0);
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

////////////////////////////////////////////////////////////////////////////////
// Hybrid tracking
template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeTrackletsHybrid(const int iteration, int iROFslice, int iVertex)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  mTimeFrameGPU->createTrackletsLUTDevice(iteration);

  const Vertex diamondVert({mTrkParams[iteration].Diamond[0], mTrkParams[iteration].Diamond[1], mTrkParams[iteration].Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  int startROF{mTrkParams[iteration].nROFsPerIterations > 0 ? iROFslice * mTrkParams[iteration].nROFsPerIterations : 0};
  int endROF{o2::gpu::CAMath::Min(mTrkParams[iteration].nROFsPerIterations > 0 ? (iROFslice + 1) * mTrkParams[iteration].nROFsPerIterations + mTrkParams[iteration].DeltaROF : mTimeFrameGPU->getNrof(), mTimeFrameGPU->getNrof())};

  countTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                       mTimeFrameGPU->getDeviceMultCutMask(),
                                       startROF,
                                       endROF,
                                       mTimeFrameGPU->getNrof(),
                                       mTrkParams[iteration].DeltaROF,
                                       iVertex,
                                       mTimeFrameGPU->getDeviceVertices(),
                                       mTimeFrameGPU->getDeviceROFramesPV(),
                                       mTimeFrameGPU->getPrimaryVerticesNum(),
                                       mTimeFrameGPU->getDeviceArrayClusters(),
                                       mTimeFrameGPU->getClusterSizes(),
                                       mTimeFrameGPU->getDeviceROframeClusters(),
                                       mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                       mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                       mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                       mTimeFrameGPU->getDeviceTrackletsLUTs(), // Required for the exclusive sums
                                       iteration,
                                       mTrkParams[iteration].NSigmaCut,
                                       mTimeFrameGPU->getPhiCuts(),
                                       mTrkParams[iteration].PVres,
                                       mTimeFrameGPU->getMinRs(),
                                       mTimeFrameGPU->getMaxRs(),
                                       mTimeFrameGPU->getPositionResolutions(),
                                       mTrkParams[iteration].LayerRadii,
                                       mTimeFrameGPU->getMSangles(),
                                       conf.nBlocks,
                                       conf.nThreads);
  mTimeFrameGPU->createTrackletsBuffers();
  computeTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceMultCutMask(),
                                         startROF,
                                         endROF,
                                         mTimeFrameGPU->getNrof(),
                                         mTrkParams[iteration].DeltaROF,
                                         iVertex,
                                         mTimeFrameGPU->getDeviceVertices(),
                                         mTimeFrameGPU->getDeviceROFramesPV(),
                                         mTimeFrameGPU->getPrimaryVerticesNum(),
                                         mTimeFrameGPU->getDeviceArrayClusters(),
                                         mTimeFrameGPU->getClusterSizes(),
                                         mTimeFrameGPU->getDeviceROframeClusters(),
                                         mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                         mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                         mTimeFrameGPU->getDeviceArrayTracklets(),
                                         mTimeFrameGPU->getDeviceTracklet(),
                                         mTimeFrameGPU->getNTracklets(),
                                         mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                         mTimeFrameGPU->getDeviceTrackletsLUTs(),
                                         iteration,
                                         mTrkParams[iteration].NSigmaCut,
                                         mTimeFrameGPU->getPhiCuts(),
                                         mTrkParams[iteration].PVres,
                                         mTimeFrameGPU->getMinRs(),
                                         mTimeFrameGPU->getMaxRs(),
                                         mTimeFrameGPU->getPositionResolutions(),
                                         mTrkParams[iteration].LayerRadii,
                                         mTimeFrameGPU->getMSangles(),
                                         conf.nBlocks,
                                         conf.nThreads);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeCellsHybrid(const int iteration)
{
  mTimeFrameGPU->createCellsLUTDevice();
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  for (int iLayer = 0; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
    if (!mTimeFrameGPU->getNTracklets()[iLayer + 1] || !mTimeFrameGPU->getNTracklets()[iLayer]) {
      continue;
    }
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[iLayer])};
    countCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                      mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                      mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                      mTimeFrameGPU->getDeviceArrayTracklets(),
                      mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                      mTimeFrameGPU->getNTracklets()[iLayer],
                      iLayer,
                      nullptr,
                      mTimeFrameGPU->getDeviceArrayCellsLUT(),
                      mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                      mBz,
                      mTrkParams[iteration].MaxChi2ClusterAttachment,
                      mTrkParams[iteration].CellDeltaTanLambdaSigma,
                      mTrkParams[iteration].NSigmaCut,
                      conf.nBlocks,
                      conf.nThreads);
    mTimeFrameGPU->createCellsBuffers(iLayer);
    computeCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                        mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                        mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                        mTimeFrameGPU->getDeviceArrayTracklets(),
                        mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                        mTimeFrameGPU->getNTracklets()[iLayer],
                        iLayer,
                        mTimeFrameGPU->getDeviceCells()[iLayer],
                        mTimeFrameGPU->getDeviceArrayCellsLUT(),
                        mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                        mBz,
                        mTrkParams[iteration].MaxChi2ClusterAttachment,
                        mTrkParams[iteration].CellDeltaTanLambdaSigma,
                        mTrkParams[iteration].NSigmaCut,
                        conf.nBlocks,
                        conf.nThreads);
  }
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighboursHybrid(const int iteration)
{
  mTimeFrameGPU->createNeighboursIndexTablesDevice();
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad() - 1; ++iLayer) {
    const int nextLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer + 1])};

    if (!nextLayerCellsNum) {
      continue;
    }

    mTimeFrameGPU->createNeighboursLUTDevice(iLayer, nextLayerCellsNum);
    unsigned int nNeigh = countCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                                     mTimeFrameGPU->getDeviceNeighboursLUT(iLayer), // LUT is initialised here.
                                                     mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                                     mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                                     mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                                     mTrkParams[0].MaxChi2ClusterAttachment,
                                                     mBz,
                                                     iLayer,
                                                     mTimeFrameGPU->getNCells()[iLayer],
                                                     nextLayerCellsNum,
                                                     1e2,
                                                     conf.nBlocks,
                                                     conf.nThreads);

    mTimeFrameGPU->createNeighboursDevice(iLayer, nNeigh);

    computeCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                 mTimeFrameGPU->getDeviceNeighboursLUT(iLayer),
                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                 mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                 mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                 mTrkParams[0].MaxChi2ClusterAttachment,
                                 mBz,
                                 iLayer,
                                 mTimeFrameGPU->getNCells()[iLayer],
                                 nextLayerCellsNum,
                                 1e2,
                                 conf.nBlocks,
                                 conf.nThreads);

    filterCellNeighboursHandler(mTimeFrameGPU->getCellsNeighbours()[iLayer],
                                mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                mTimeFrameGPU->getDeviceNeighbours(iLayer),
                                nNeigh);
  }
  mTimeFrameGPU->createNeighboursDeviceArray();
  mTimeFrameGPU->unregisterRest();
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    std::vector<CellSeed> trackSeeds;
    for (int startLayer{mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      if ((mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }
      std::vector<int> lastCellId, updatedCellId;
      std::vector<CellSeed> lastCellSeed, updatedCellSeed;

      processNeighboursHandler<nLayers>(startLayer,
                                        startLevel,
                                        mTimeFrameGPU->getDeviceArrayCells(),
                                        mTimeFrameGPU->getDeviceCells()[startLayer],
                                        mTimeFrameGPU->getArrayNCells(),
                                        mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                        mTimeFrameGPU->getDeviceNeighboursAll(),
                                        mTimeFrameGPU->getDeviceNeighboursLUTs(),
                                        mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                        trackSeeds,
                                        mBz,
                                        mTrkParams[0].MaxChi2ClusterAttachment,
                                        mTrkParams[0].MaxChi2NDF,
                                        mTimeFrameGPU->getDevicePropagator(),
                                        mCorrType,
                                        conf.nBlocks,
                                        conf.nThreads);
    }
    // fixme: I don't want to move tracks back and forth, but I need a way to use a thrust::allocator that is aware of our managed memory.
    if (!trackSeeds.size()) {
      LOGP(info, "No track seeds found, skipping track finding");
      continue;
    }
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds);
    mTimeFrameGPU->loadTrackSeedsDevice(trackSeeds);

    trackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),             // CellSeed* trackSeeds
                     mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo** foundTrackingFrameInfo
                     mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt* tracks
                     mTrkParams[iteration].MinPt,                      // std::vector<float>& minPtsHost,
                     trackSeeds.size(),                                // const size_t nSeeds
                     mBz,                                              // const float Bz
                     startLevel,                                       // const int startLevel,
                     mTrkParams[0].MaxChi2ClusterAttachment,           // float maxChi2ClusterAttachment
                     mTrkParams[0].MaxChi2NDF,                         // float maxChi2NDF
                     mTimeFrameGPU->getDevicePropagator(),             // const o2::base::Propagator* propagator
                     mCorrType,                                        // o2::base::PropagatorImpl<float>::MatCorrType
                     conf.nBlocks,
                     conf.nThreads);

    mTimeFrameGPU->downloadTrackITSExtDevice(trackSeeds);

    auto& tracks = mTimeFrameGPU->getTrackITSExt();

    for (auto& track : tracks) {
      if (!track.getChi2()) {
        continue; // this is to skip the unset tracks that are put at the beginning of the vector by the sorting. To see if this can be optimised.
      }
      int nShared = 0;
      bool isFirstShared{false};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == UnusedIndex) {
          continue;
        }
        nShared += int(mTimeFrameGPU->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
        isFirstShared |= !iLayer && mTimeFrameGPU->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
      }

      if (nShared > mTrkParams[0].ClusterSharing) {
        continue;
      }

      std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == UnusedIndex) {
          continue;
        }
        mTimeFrameGPU->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        int currentROF = mTimeFrameGPU->getClusterROF(iLayer, track.getClusterIndex(iLayer));
        for (int iR{0}; iR < 3; ++iR) {
          if (rofs[iR] == INT_MAX) {
            rofs[iR] = currentROF;
          }
          if (rofs[iR] == currentROF) {
            break;
          }
        }
      }
      if (rofs[2] != INT_MAX) {
        continue;
      }
      if (rofs[1] != INT_MAX) {
        track.setNextROFbit();
      }
      mTimeFrameGPU->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
    }
    mTimeFrameGPU->loadUsedClustersDevice();
  }
  if (iteration == mTrkParams.size() - 1) {
    mTimeFrameGPU->unregisterHostMemory(0);
  }
};

template class TrackerTraitsGPU<7>;
} // namespace o2::its

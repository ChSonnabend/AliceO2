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

#include <cuda_runtime.h>
#include <array>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <thread>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/Vertex.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

#ifdef GPUCA_NO_FAST_MATH
#define GPU_BLOCKS 1
#define GPU_THREADS 1
#else
#define GPU_BLOCKS 99999
#define GPU_THREADS 99999
#endif

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

#define gpuCheckError(x)                \
  {                                     \
    gpuAssert((x), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    LOGF(error, "GPUassert: %s %s %d", cudaGetErrorString(code), file, line);
    if (abort) {
      throw std::runtime_error("GPU assert failed.");
    }
  }
}

namespace o2::its
{
using namespace constants::its2;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

GPUd() float Sq(float v)
{
  return v * v;
}

namespace gpu
{

GPUd() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                              const o2::its::IndexTableUtils& utils,
                              const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::CAMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = (maxdeltaphi > constants::math::Pi) ? 0.f : currentCluster.phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::CAMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = (maxdeltaphi > constants::math::Pi) ? constants::math::TwoPi : currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::CAMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::CAMath::Min(ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float bz,
                     const TrackingFrameInfo** tfInfos,
                     const o2::base::Propagator* prop,
                     o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = tfInfos[iLayer][track.getClusterIndex(iLayer)];
    if (!track.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
      return false;
    }

    if (!prop->propagateToX(track,
                            trackingHit.xTrackingFrame,
                            bz,
                            o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                            o2::base::PropagatorImpl<float>::MAX_STEP,
                            matCorrType)) {
      return false;
    }

    if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
      const float xx0 = (iLayer > 2) ? 1.e-2f : 5.e-3f; // Rough layer thickness
      constexpr float radiationLength = 9.36f;          // Radiation length of Si [cm]
      constexpr float density = 2.33f;                  // Density of Si [g/cm^3]
      if (!track.correctForMaterial(xx0, xx0 * radiationLength * density, true)) {
        return false;
      }
    }

    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};

    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    nCl++;
  }
  return o2::gpu::CAMath::Abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
}

GPUd() o2::track::TrackParCov buildTrackSeed(const Cluster& cluster1,
                                             const Cluster& cluster2,
                                             const TrackingFrameInfo& tf3,
                                             const float bz)
{
  const float ca = o2::gpu::CAMath::Cos(tf3.alphaTrackingFrame), sa = o2::gpu::CAMath::Sin(tf3.alphaTrackingFrame);
  const float x1 = cluster1.xCoordinate * ca + cluster1.yCoordinate * sa;
  const float y1 = -cluster1.xCoordinate * sa + cluster1.yCoordinate * ca;
  const float z1 = cluster1.zCoordinate;
  const float x2 = cluster2.xCoordinate * ca + cluster2.yCoordinate * sa;
  const float y2 = -cluster2.xCoordinate * sa + cluster2.yCoordinate * ca;
  const float z2 = cluster2.zCoordinate;
  const float x3 = tf3.xTrackingFrame;
  const float y3 = tf3.positionTrackingFrame[0];
  const float z3 = tf3.positionTrackingFrame[1];

  const bool zeroField{o2::gpu::CAMath::Abs(bz) < o2::constants::math::Almost0};
  const float tgp = zeroField ? o2::gpu::CAMath::ATan2(y3 - y1, x3 - x1) : 1.f;
  const float crv = zeroField ? 1.f : math_utils::computeCurvature(x3, y3, x2, y2, x1, y1);
  const float snp = zeroField ? tgp / o2::gpu::CAMath::Sqrt(1.f + tgp * tgp) : crv * (x3 - math_utils::computeCurvatureCentreX(x3, y3, x2, y2, x1, y1));
  const float tgl12 = math_utils::computeTanDipAngle(x1, y1, x2, y2, z1, z2);
  const float tgl23 = math_utils::computeTanDipAngle(x2, y2, x3, y3, z2, z3);
  const float q2pt = zeroField ? 1.f / o2::track::kMostProbablePt : crv / (bz * o2::constants::math::B2C);
  const float q2pt2 = crv * crv;
  const float sg2q2pt = o2::track::kC1Pt2max * (q2pt2 > 0.0005 ? (q2pt2 < 1 ? q2pt2 : 1) : 0.0005);
  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            {y3, z3, snp, 0.5f * (tgl12 + tgl23), q2pt},
                            {tf3.covarianceTrackingFrame[0],
                             tf3.covarianceTrackingFrame[1], tf3.covarianceTrackingFrame[2],
                             0.f, 0.f, track::kCSnp2max,
                             0.f, 0.f, 0.f, track::kCTgl2max,
                             0.f, 0.f, 0.f, 0.f, sg2q2pt});
}

struct sort_tracklets {
  GPUhd() bool operator()(const Tracklet& a, const Tracklet& b) { return a.firstClusterIndex < b.firstClusterIndex || (a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex < b.secondClusterIndex); }
};

struct equal_tracklets {
  GPUhd() bool operator()(const Tracklet& a, const Tracklet& b) { return a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex == b.secondClusterIndex; }
};

template <typename T1, typename T2>
struct pair_to_first : public thrust::unary_function<gpuPair<T1, T2>, T1> {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.first;
  }
};

template <typename T1, typename T2>
struct pair_to_second : public thrust::unary_function<gpuPair<T1, T2>, T2> {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.second;
  }
};

template <typename T1, typename T2>
struct is_invalid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return p.first == -1 && p.second == -1;
  }
};

template <typename T1, typename T2>
struct is_valid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return !(p.first == -1 && p.second == -1);
  }
};

struct seed_selector {
  float maxQ2Pt;
  float maxChi2;

  GPUhd() seed_selector(float maxQ2Pt, float maxChi2) : maxQ2Pt(maxQ2Pt), maxChi2(maxChi2) {}
  GPUhd() bool operator()(const CellSeed& seed) const
  {
    return !(seed.getQ2Pt() > maxQ2Pt || seed.getChi2() > maxChi2);
  }
};

struct compare_track_chi2 {
  GPUhd() bool operator()(const TrackITSExt& a, const TrackITSExt& b) const
  {
    return a.getChi2() < b.getChi2();
  }
};

GPUd() gpuSpan<const Vertex> getPrimaryVertices(const int rof,
                                                const int* roframesPV,
                                                const int nROF,
                                                const uint8_t* mask,
                                                const Vertex* vertices)
{
  const int start_pv_id = roframesPV[rof];
  const int stop_rof = rof >= nROF - 1 ? nROF : rof + 1;
  size_t delta = mask[rof] ? roframesPV[stop_rof] - start_pv_id : 0; // return empty span if ROF is excluded
  return gpuSpan<const Vertex>(&vertices[start_pv_id], delta);
};

GPUd() gpuSpan<const Cluster> getClustersOnLayer(const int rof,
                                                 const int totROFs,
                                                 const int layer,
                                                 const int** roframesClus,
                                                 const Cluster** clusters)
{
  if (rof < 0 || rof >= totROFs) {
    return gpuSpan<const Cluster>();
  }
  const int start_clus_id{roframesClus[layer][rof]};
  const int stop_rof = rof >= totROFs - 1 ? totROFs : rof + 1;
  const unsigned int delta = roframesClus[layer][stop_rof] - start_clus_id;
  return gpuSpan<const Cluster>(&(clusters[layer][start_clus_id]), delta);
}

template <int nLayers>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const float* minPts,
  const unsigned int nSeeds,
  const float bz,
  const int startLevel,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {
    auto& seed = trackSeeds[iCurrentTrackSeedIndex];

    TrackITSExt temporaryTrack{seed};

    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    int* clusters = seed.getClusters();
    for (int iL{0}; iL < 7; ++iL) {
      temporaryTrack.setExternalClusterIndex(iL, clusters[iL], clusters[iL] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack,               // TrackITSExt& track,
                               0,                            // int lastLayer,
                               nLayers,                      // int firstLayer,
                               1,                            // int firstCluster,
                               maxChi2ClusterAttachment,     // float maxChi2ClusterAttachment,
                               maxChi2NDF,                   // float maxChi2NDF,
                               o2::constants::math::VeryBig, // float maxQoverPt,
                               0,                            // nCl,
                               bz,                           // float bz,
                               foundTrackingFrameInfo,       // TrackingFrameInfo** trackingFrameInfo,
                               propagator,                   // const o2::base::Propagator* propagator,
                               matCorrType);                 // o2::base::PropagatorF::MatCorrType matCorrType
    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);

    fitSuccess = fitTrack(temporaryTrack,           // TrackITSExt& track,
                          nLayers - 1,              // int lastLayer,
                          -1,                       // int firstLayer,
                          -1,                       // int firstCluster,
                          maxChi2ClusterAttachment, // float maxChi2ClusterAttachment,
                          maxChi2NDF,               // float maxChi2NDF,
                          50.f,                     // float maxQoverPt,
                          0,                        // nCl,
                          bz,                       // float bz,
                          foundTrackingFrameInfo,   // TrackingFrameInfo** trackingFrameInfo,
                          propagator,               // const o2::base::Propagator* propagator,
                          matCorrType);             // o2::base::PropagatorF::MatCorrType matCorrType
    if (!fitSuccess || temporaryTrack.getPt() < minPts[nLayers - temporaryTrack.getNClusters()]) {
      continue;
    }
    tracks[iCurrentTrackSeedIndex] = temporaryTrack;
  }
}

template <bool initRun, int nLayers = 7> // Version for new tracker to supersede the old one
GPUg() void computeLayerCellNeighboursKernel(
  CellSeed** cellSeedArray,
  int* neighboursLUT,
  int* neighboursIndexTable,
  int** cellsLUTs,
  gpuPair<int, int>* cellNeighbours,
  const float maxChi2ClusterAttachment,
  const float bz,
  const int layerIndex,
  const unsigned int nCells,
  const int maxCellNeighbours = 1e2)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells; iCurrentCellIndex += blockDim.x * gridDim.x) {
    const auto& currentCellSeed{cellSeedArray[layerIndex][iCurrentCellIndex]};
    const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsLUTs[layerIndex + 1][nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsLUTs[layerIndex + 1][nextLayerTrackletIndex + 1]};
    int foundNeighbours{0};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      CellSeed nextCellSeed{cellSeedArray[layerIndex + 1][iNextCell]};      // Copy
      if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex) { // Check if cells share the same tracklet
        break;
      }
      if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
          !nextCellSeed.propagateTo(currentCellSeed.getX(), bz)) {
        continue;
      }
      float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed);
      if (chi2 > maxChi2ClusterAttachment) /// TODO: switch to the chi2 wrt cluster to avoid correlation
      {
        continue;
      }
      if constexpr (initRun) {
        atomicAdd(neighboursLUT + iNextCell, 1);
        foundNeighbours++;
        neighboursIndexTable[iCurrentCellIndex]++;
      } else {
        cellNeighbours[neighboursIndexTable[iCurrentCellIndex] + foundNeighbours] = {iCurrentCellIndex, iNextCell};
        foundNeighbours++;
        // FIXME: this is prone to race conditions: check on level is not atomic
        const int currentCellLevel{currentCellSeed.getLevel()};
        if (currentCellLevel >= nextCellSeed.getLevel()) {
          cellSeedArray[layerIndex + 1][iNextCell].setLevel(currentCellLevel + 1);
        }
      }
    }
  }
}

template <bool initRun, int nLayers = 7>
GPUg() void computeLayerCellsKernel(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTrackletsCurrent,
  const int layer,
  CellSeed* cells,
  int** cellsLUTs,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut)
{
  constexpr float radl = 9.36f;                                                           // Radiation length of Si [cm].
  constexpr float rho = 2.33f;                                                            // Density of Si [g/cm^3].
  constexpr float layerxX0[7] = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f}; // Hardcoded here for the moment.
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    const Tracklet& currentTracklet = tracklets[layer][iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsLUT[layer + 1][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[layer + 1][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[layer + 1][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[layer + 1][iNextTrackletIndex];
      const float deltaTanLambda{o2::gpu::CAMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};

      if (deltaTanLambda / cellDeltaTanLambdaSigma < nSigmaCut) {
        const int clusId[3]{
          sortedClusters[layer][currentTracklet.firstClusterIndex].clusterId,
          sortedClusters[layer + 1][nextTracklet.firstClusterIndex].clusterId,
          sortedClusters[layer + 2][nextTracklet.secondClusterIndex].clusterId};

        const auto& cluster1_glo = unsortedClusters[layer][clusId[0]];
        const auto& cluster2_glo = unsortedClusters[layer + 1][clusId[1]];
        const auto& cluster3_tf = tfInfo[layer + 2][clusId[2]];
        auto track{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf, bz)};
        float chi2{0.f};
        bool good{false};
        for (int iC{2}; iC--;) {
          const TrackingFrameInfo& trackingHit = tfInfo[layer + iC][clusId[iC]];
          if (!track.rotate(trackingHit.alphaTrackingFrame)) {
            break;
          }
          if (!track.propagateTo(trackingHit.xTrackingFrame, bz)) {
            break;
          }

          if (!track.correctForMaterial(layerxX0[layer + iC], layerxX0[layer] * radl * rho, true)) {
            break;
          }

          const auto predChi2{track.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
          if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
            break;
          }
          if (!iC && predChi2 > maxChi2ClusterAttachment) {
            break;
          }
          good = !iC;
          chi2 += predChi2;
        }
        if (!good) {
          continue;
        }
        if constexpr (!initRun) {
          new (cells + cellsLUTs[layer][iCurrentTrackletIndex] + foundCells) CellSeed{layer, clusId[0], clusId[1], clusId[2], iCurrentTrackletIndex, iNextTrackletIndex, track, chi2};
        }
        ++foundCells;
        if constexpr (initRun) {
          cellsLUTs[layer][iCurrentTrackletIndex] = foundCells;
        }
      }
    }
  }
}

template <bool initRun = true, int nLayers = 7>
GPUg() void computeLayerTrackletsMultiROFKernel(
  const IndexTableUtils* utils,
  const uint8_t* multMask,
  const int layerIndex,
  const int startROF,
  const int endROF,
  const int totalROFs,
  const int deltaROF,
  const Vertex* vertices,
  const int* rofPV,
  const int nVertices,
  const int vertexId,
  const Cluster** clusters,           // Input data rof0
  const int** ROFClusters,            // Number of clusters on layers per ROF
  const unsigned char** usedClusters, // Used clusters
  const int** indexTables,            // Input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  Tracklet** tracklets,               // Output data
  int** trackletsLUT,
  const int iteration,
  const float NSigmaCut,
  const float phiCut,
  const float resolutionPV,
  const float minR,
  const float maxR,
  const float positionResolution,
  const float meanDeltaR = -42.f,
  const float MSAngle = -42.f)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  for (unsigned int iROF{blockIdx.x}; iROF < endROF - startROF; iROF += gridDim.x) {
    const short rof0 = iROF + startROF;
    auto primaryVertices = getPrimaryVertices(rof0, rofPV, totalROFs, multMask, vertices);
    const auto startVtx{vertexId >= 0 ? vertexId : 0};
    const auto endVtx{vertexId >= 0 ? o2::gpu::CAMath::Min(vertexId + 1, static_cast<int>(primaryVertices.size())) : static_cast<int>(primaryVertices.size())};
    const short minROF = o2::gpu::CAMath::Max(startROF, static_cast<int>(rof0 - deltaROF));
    const short maxROF = o2::gpu::CAMath::Min(endROF - 1, static_cast<int>(rof0 + deltaROF));
    auto clustersCurrentLayer = getClustersOnLayer(rof0, totalROFs, layerIndex, ROFClusters, clusters);
    if (clustersCurrentLayer.empty()) {
      continue;
    }

    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < clustersCurrentLayer.size(); currentClusterIndex += blockDim.x) {
      unsigned int storedTracklets{0};
      auto currentCluster{clustersCurrentLayer[currentClusterIndex]};
      const int currentSortedIndex{ROFClusters[layerIndex][rof0] + currentClusterIndex};
      if (usedClusters[layerIndex][currentCluster.clusterId]) {
        continue;
      }

      const float inverseR0{1.f / currentCluster.radius};
      for (int iV{startVtx}; iV < endVtx; ++iV) {
        auto& primaryVertex{primaryVertices[iV]};
        if (primaryVertex.isFlagSet(2) && iteration != 3) {
          continue;
        }
        const float resolution = o2::gpu::CAMath::Sqrt(Sq(resolutionPV) / primaryVertex.getNContributors() + Sq(positionResolution));
        const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
        const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
        const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
        const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
        const float sigmaZ{o2::gpu::CAMath::Sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * MSAngle))};
        const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * NSigmaCut, phiCut)};
        if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
          continue;
        }
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }

        const int tableSize{phiBins * zBins + 1};
        for (short rof1{minROF}; rof1 <= maxROF; ++rof1) {
          auto clustersNextLayer = getClustersOnLayer(rof1, totalROFs, layerIndex + 1, ROFClusters, clusters);
          if (clustersNextLayer.empty()) {
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % phiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTables[layerIndex + 1][(rof1 - startROF) * tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTables[layerIndex + 1][(rof1 - startROF) * tableSize + maxBinIndex];
            for (int nextClusterIndex{firstRowClusterIndex}; nextClusterIndex < maxRowClusterIndex; ++nextClusterIndex) {
              if (nextClusterIndex >= clustersNextLayer.size()) {
                break;
              }
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClusters[layerIndex + 1][nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::CAMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::CAMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
              const int nextSortedIndex{ROFClusters[layerIndex + 1][rof1] + nextClusterIndex};
              if (deltaZ / sigmaZ < NSigmaCut && (deltaPhi < phiCut || o2::gpu::CAMath::Abs(deltaPhi - constants::math::TwoPi) < phiCut)) {
                if constexpr (initRun) {
                  trackletsLUT[layerIndex][currentSortedIndex]++; // we need l0 as well for usual exclusive sums.
                } else {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                  new (tracklets[layerIndex] + trackletsLUT[layerIndex][currentSortedIndex] + storedTracklets) Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, rof0, rof1};
                }
                ++storedTracklets;
              }
            }
          }
        }
      }
    }
  }
}

template <int nLayers = 7>
GPUg() void compileTrackletsLookupTableKernel(const Tracklet* tracklets,
                                              int* trackletsLookUpTable,
                                              const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    atomicAdd(&trackletsLookUpTable[tracklets[currentTrackletIndex].firstClusterIndex], 1);
  }
}

template <bool dryRun, bool debug = false, int nLayers = 7>
GPUg() void processNeighboursKernel(const int layer,
                                    const int level,
                                    CellSeed** allCellSeeds,
                                    CellSeed* currentCellSeeds,
                                    const int* currentCellIds,
                                    const unsigned int nCurrentCells,
                                    CellSeed* updatedCellSeeds,
                                    int* updatedCellsIds,
                                    int* foundSeedsTable,               // auxiliary only in GPU code to compute the number of cells per iteration
                                    const unsigned char** usedClusters, // Used clusters
                                    int* neighbours,
                                    int* neighboursLUT,
                                    const TrackingFrameInfo** foundTrackingFrameInfo,
                                    const float bz,
                                    const float maxChi2ClusterAttachment,
                                    const o2::base::Propagator* propagator,
                                    const o2::base::PropagatorF::MatCorrType matCorrType)
{
  constexpr float radl = 9.36f;                                                           // Radiation length of Si [cm].
  constexpr float rho = 2.33f;                                                            // Density of Si [g/cm^3].
  constexpr float layerxX0[7] = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f}; // Hardcoded here for the moment.
  for (unsigned int iCurrentCell = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCell < nCurrentCells; iCurrentCell += blockDim.x * gridDim.x) {
    int foundSeeds{0};
    const auto& currentCell{currentCellSeeds[iCurrentCell]};
    if (currentCell.getLevel() != level) {
      continue;
    }
    if (currentCellIds == nullptr && (usedClusters[layer][currentCell.getFirstClusterIndex()] ||
                                      usedClusters[layer + 1][currentCell.getSecondClusterIndex()] ||
                                      usedClusters[layer + 2][currentCell.getThirdClusterIndex()])) {
      continue;
    }
    const int cellId = currentCellIds == nullptr ? iCurrentCell : currentCellIds[iCurrentCell];

    const int startNeighbourId{cellId ? neighboursLUT[cellId - 1] : 0};
    const int endNeighbourId{neighboursLUT[cellId]};

    for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
      const int neighbourCellId = neighbours[iNeighbourCell];
      const CellSeed& neighbourCell = allCellSeeds[layer - 1][neighbourCellId];

      if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
        continue;
      }
      if (usedClusters[layer - 1][neighbourCell.getFirstClusterIndex()]) {
        continue;
      }
      if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
        continue;
      }
      CellSeed seed{currentCell};
      auto& trHit = foundTrackingFrameInfo[layer - 1][neighbourCell.getFirstClusterIndex()];

      if (!seed.rotate(trHit.alphaTrackingFrame)) {
        continue;
      }

      if (!propagator->propagateToX(seed, trHit.xTrackingFrame, bz, o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, matCorrType)) {
        continue;
      }

      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!seed.correctForMaterial(layerxX0[layer - 1], layerxX0[layer - 1] * radl * rho, true)) {
          continue;
        }
      }

      auto predChi2{seed.getPredictedChi2Quiet(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)};
      if ((predChi2 > maxChi2ClusterAttachment) || predChi2 < 0.f) {
        continue;
      }
      seed.setChi2(seed.getChi2() + predChi2);
      if (!seed.o2::track::TrackParCov::update(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)) {
        continue;
      }
      seed.getClusters()[layer - 1] = neighbourCell.getFirstClusterIndex();
      seed.setLevel(neighbourCell.getLevel());
      seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
      seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
      if constexpr (dryRun) {
        foundSeedsTable[iCurrentCell]++;
      } else {
        updatedCellsIds[foundSeedsTable[iCurrentCell] + foundSeeds] = neighbourCellId;
        updatedCellSeeds[foundSeedsTable[iCurrentCell] + foundSeeds] = seed;
      }
      foundSeeds++;
    }
  }
}

/////////////////////////////////////////
// Debug Kernels
/////////////////////////////////////////

template <typename T>
GPUd() void pPointer(T* ptr)
{
  printf("[%p]\t", ptr);
}

template <typename... Args>
GPUg() void printPointersKernel(std::tuple<Args...> args)
{
  auto print_all = [&](auto... ptrs) {
    (pPointer(ptrs), ...);
  };
  std::apply(print_all, args);
}

template <typename T>
struct trackletSortEmptyFunctor : public thrust::binary_function<T, T, bool> {
  GPUhd() bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs.firstClusterIndex > rhs.firstClusterIndex;
  }
};

template <typename T>
struct trackletSortIndexFunctor : public thrust::binary_function<T, T, bool> {
  GPUhd() bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs.firstClusterIndex < rhs.firstClusterIndex || (lhs.firstClusterIndex == rhs.firstClusterIndex && lhs.secondClusterIndex < rhs.secondClusterIndex);
  }
};

GPUg() void printBufferLayerOnThread(const int layer, const int* v, unsigned int size, const int len = 150, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (int i{0}; i < size; ++i) {
      if (!(i % len)) {
        printf("\n layer %d: ===> %d/%d\t", layer, i, (int)size);
      }
      printf("%d\t", v[i]);
    }
    printf("\n");
  }
}

GPUg() void printMatrixRow(const int row, int** mat, const unsigned int rowLength, const int len = 150, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (int i{0}; i < rowLength; ++i) {
      if (!(i % len)) {
        printf("\n row %d: ===> %d/%d\t", row, i, (int)rowLength);
      }
      printf("%d\t", mat[row][i]);
    }
    printf("\n");
  }
}

GPUg() void printBufferPointersLayerOnThread(const int layer, void** v, unsigned int size, const int len = 150, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (int i{0}; i < size; ++i) {
      if (!(i % len)) {
        printf("\n layer %d: ===> %d/%d\t", layer, i, (int)size);
      }
      printf("%p\t", (void*)v[i]);
    }
    printf("\n");
  }
}

GPUg() void printVertices(const Vertex* v, unsigned int size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vertices: \n");
    for (int i{0}; i < size; ++i) {
      printf("\tx=%f y=%f z=%f\n", v[i].getX(), v[i].getY(), v[i].getZ());
    }
  }
}

GPUg() void printNeighbours(const gpuPair<int, int>* neighbours,
                            const int* nNeighboursIndexTable,
                            const unsigned int nCells,
                            const unsigned int tId = 0)
{
  for (unsigned int iNeighbour{0}; iNeighbour < nNeighboursIndexTable[nCells]; ++iNeighbour) {
    if (threadIdx.x == tId) {
      printf("%d -> %d\n", neighbours[iNeighbour].first, neighbours[iNeighbour].second);
    }
  }
}

GPUg() void printTrackletsLUTPerROF(const int layerId,
                                    const int** ROFClusters,
                                    int** luts,
                                    const int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    for (auto rofId{0}; rofId < 2304; ++rofId) {
      int nClus = ROFClusters[layerId][rofId + 1] - ROFClusters[layerId][rofId];
      if (!nClus) {
        continue;
      }
      printf("rof: %d (%d) ==> ", rofId, nClus);

      for (int iC{0}; iC < nClus; ++iC) {
        int nT = luts[layerId][ROFClusters[layerId][rofId] + iC];
        printf("%d\t", nT);
      }
      printf("\n");
    }
  }
}

GPUg() void printCellSeeds(CellSeed* seed, int nCells, const unsigned int tId = 0)
{
  for (unsigned int iCell{0}; iCell < nCells; ++iCell) {
    if (threadIdx.x == tId) {
      seed[iCell].printCell();
    }
  }
}
} // namespace gpu

template <int nLayers>
void countTrackletsInROFsHandler(const IndexTableUtils* utils,
                                 const uint8_t* multMask,
                                 const int startROF,
                                 const int endROF,
                                 const int maxROF,
                                 const int deltaROF,
                                 const int vertexId,
                                 const Vertex* vertices,
                                 const int* rofPV,
                                 const int nVertices,
                                 const Cluster** clusters,
                                 std::vector<unsigned int> nClusters,
                                 const int** ROFClusters,
                                 const unsigned char** usedClusters,
                                 const int** clustersIndexTables,
                                 int** trackletsLUTs,
                                 gsl::span<int*> trackletsLUTsHost,
                                 const int iteration,
                                 const float NSigmaCut,
                                 std::vector<float>& phiCuts,
                                 const float resolutionPV,
                                 std::vector<float>& minRs,
                                 std::vector<float>& maxRs,
                                 std::vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 std::vector<float>& mulScatAng,
                                 const int nBlocks,
                                 const int nThreads)
{
  for (int iLayer = 0; iLayer < nLayers - 1; ++iLayer) {
    gpu::computeLayerTrackletsMultiROFKernel<true><<<nBlocks, nThreads>>>(
      utils,
      multMask,
      iLayer,
      startROF,
      endROF,
      maxROF,
      deltaROF,
      vertices,
      rofPV,
      nVertices,
      vertexId,
      clusters,
      ROFClusters,
      usedClusters,
      clustersIndexTables,
      nullptr,
      trackletsLUTs,
      iteration,
      NSigmaCut,
      phiCuts[iLayer],
      resolutionPV,
      minRs[iLayer + 1],
      maxRs[iLayer + 1],
      resolutions[iLayer],
      radii[iLayer + 1] - radii[iLayer],
      mulScatAng[iLayer]);
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,            // d_temp_storage
                                                temp_storage_bytes,        // temp_storage_bytes
                                                trackletsLUTsHost[iLayer], // d_in
                                                trackletsLUTsHost[iLayer], // d_out
                                                nClusters[iLayer] + 1,     // num_items
                                                0));                       // NOLINT: this is the offset of the sum, not a pointer
    discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,            // d_temp_storage
                                                temp_storage_bytes,        // temp_storage_bytes
                                                trackletsLUTsHost[iLayer], // d_in
                                                trackletsLUTsHost[iLayer], // d_out
                                                nClusters[iLayer] + 1,     // num_items
                                                0));                       // NOLINT: this is the offset of the sum, not a pointer
    gpuCheckError(cudaFree(d_temp_storage));
  }
}

template <int nLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils* utils,
                                   const uint8_t* multMask,
                                   const int startROF,
                                   const int endROF,
                                   const int maxROF,
                                   const int deltaROF,
                                   const int vertexId,
                                   const Vertex* vertices,
                                   const int* rofPV,
                                   const int nVertices,
                                   const Cluster** clusters,
                                   std::vector<unsigned int> nClusters,
                                   const int** ROFClusters,
                                   const unsigned char** usedClusters,
                                   const int** clustersIndexTables,
                                   Tracklet** tracklets,
                                   gsl::span<Tracklet*> spanTracklets,
                                   gsl::span<int> nTracklets,
                                   int** trackletsLUTs,
                                   gsl::span<int*> trackletsLUTsHost,
                                   const int iteration,
                                   const float NSigmaCut,
                                   std::vector<float>& phiCuts,
                                   const float resolutionPV,
                                   std::vector<float>& minRs,
                                   std::vector<float>& maxRs,
                                   std::vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   std::vector<float>& mulScatAng,
                                   const int nBlocks,
                                   const int nThreads)
{
  for (int iLayer = 0; iLayer < nLayers - 1; ++iLayer) {
    gpu::computeLayerTrackletsMultiROFKernel<false><<<nBlocks, nThreads>>>(utils,
                                                                           multMask,
                                                                           iLayer,
                                                                           startROF,
                                                                           endROF,
                                                                           maxROF,
                                                                           deltaROF,
                                                                           vertices,
                                                                           rofPV,
                                                                           nVertices,
                                                                           vertexId,
                                                                           clusters,
                                                                           ROFClusters,
                                                                           usedClusters,
                                                                           clustersIndexTables,
                                                                           tracklets,
                                                                           trackletsLUTs,
                                                                           iteration,
                                                                           NSigmaCut,
                                                                           phiCuts[iLayer],
                                                                           resolutionPV,
                                                                           minRs[iLayer + 1],
                                                                           maxRs[iLayer + 1],
                                                                           resolutions[iLayer],
                                                                           radii[iLayer + 1] - radii[iLayer],
                                                                           mulScatAng[iLayer]);
    thrust::device_ptr<Tracklet> tracklets_ptr(spanTracklets[iLayer]);
    thrust::sort(thrust::device, tracklets_ptr, tracklets_ptr + nTracklets[iLayer], gpu::sort_tracklets());
    auto unique_end = thrust::unique(thrust::device, tracklets_ptr, tracklets_ptr + nTracklets[iLayer], gpu::equal_tracklets());
    nTracklets[iLayer] = unique_end - tracklets_ptr;
    if (iLayer > 0) {
      gpuCheckError(cudaMemset(trackletsLUTsHost[iLayer], 0, nClusters[iLayer] * sizeof(int)));
      gpu::compileTrackletsLookupTableKernel<<<nBlocks, nThreads>>>(spanTracklets[iLayer], trackletsLUTsHost[iLayer], nTracklets[iLayer]);
      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,            // d_temp_storage
                                                  temp_storage_bytes,        // temp_storage_bytes
                                                  trackletsLUTsHost[iLayer], // d_in
                                                  trackletsLUTsHost[iLayer], // d_out
                                                  nClusters[iLayer] + 1,     // num_items
                                                  0));                       // NOLINT: this is the offset of the sum, not a pointer
      discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,            // d_temp_storage
                                                  temp_storage_bytes,        // temp_storage_bytes
                                                  trackletsLUTsHost[iLayer], // d_in
                                                  trackletsLUTsHost[iLayer], // d_out
                                                  nClusters[iLayer] + 1,     // num_items
                                                  0));                       // NOLINT: this is the offset of the sum, not a pointer
      gpuCheckError(cudaFree(d_temp_storage));
    }
  }
}

void countCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int layer,
  CellSeed* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  const int nBlocks,
  const int nThreads)
{
  gpu::computeLayerCellsKernel<true><<<nBlocks, nThreads>>>(
    sortedClusters,           // const Cluster**
    unsortedClusters,         // const Cluster**
    tfInfo,                   // const TrackingFrameInfo**
    tracklets,                // const Tracklets**
    trackletsLUT,             // const int**
    nTracklets,               // const int
    layer,                    // const int
    cells,                    // CellSeed*
    cellsLUTsArrayDevice,     // int**
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,     // d_temp_storage
                                              temp_storage_bytes, // temp_storage_bytes
                                              cellsLUTsHost,      // d_in
                                              cellsLUTsHost,      // d_out
                                              nTracklets + 1,     // num_items
                                              0));                // NOLINT: this is the offset of the sum, not a pointer
  discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,     // d_temp_storage
                                              temp_storage_bytes, // temp_storage_bytes
                                              cellsLUTsHost,      // d_in
                                              cellsLUTsHost,      // d_out
                                              nTracklets + 1,     // num_items
                                              0));                // NOLINT: this is the offset of the sum, not a pointer
  gpuCheckError(cudaFree(d_temp_storage));
}

void computeCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int layer,
  CellSeed* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  const int nBlocks,
  const int nThreads)
{
  gpu::computeLayerCellsKernel<false><<<nBlocks, nThreads>>>(
    sortedClusters,           // const Cluster**
    unsortedClusters,         // const Cluster**
    tfInfo,                   // const TrackingFrameInfo**
    tracklets,                // const Tracklets**
    trackletsLUT,             // const int**
    nTracklets,               // const int
    layer,                    // const int
    cells,                    // CellSeed*
    cellsLUTsArrayDevice,     // int**
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
}

void countCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                int* neighboursLUT,
                                int** cellsLUTs,
                                gpuPair<int, int>* cellNeighbours,
                                int* neighboursIndexTable,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const int layerIndex,
                                const unsigned int nCells,
                                const unsigned int nCellsNext,
                                const int maxCellNeighbours,
                                const int nBlocks,
                                const int nThreads)
{
  gpu::computeLayerCellNeighboursKernel<true><<<nBlocks, nThreads>>>(
    cellsLayersDevice,
    neighboursLUT,
    neighboursIndexTable,
    cellsLUTs,
    cellNeighbours,
    maxChi2ClusterAttachment,
    bz,
    layerIndex,
    nCells,
    maxCellNeighbours);
  // gpuCheckError(cudaPeekAtLastError());
  // gpuCheckError(cudaDeviceSynchronize());
  void *d_temp_storage = nullptr, *d_temp_storage_2 = nullptr;
  size_t temp_storage_bytes = 0, temp_storage_bytes_2 = 0;
  gpuCheckError(cub::DeviceScan::InclusiveSum(d_temp_storage,     // d_temp_storage
                                              temp_storage_bytes, // temp_storage_bytes
                                              neighboursLUT,      // d_in
                                              neighboursLUT,      // d_out
                                              nCellsNext));       // num_items

  discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  gpuCheckError(cub::DeviceScan::InclusiveSum(d_temp_storage,       // d_temp_storage
                                              temp_storage_bytes,   // temp_storage_bytes
                                              neighboursLUT,        // d_in
                                              neighboursLUT,        // d_out
                                              nCellsNext));         // num_items
  gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage_2,     // d_temp_storage
                                              temp_storage_bytes_2, // temp_storage_bytes
                                              neighboursIndexTable, // d_in
                                              neighboursIndexTable, // d_out
                                              nCells + 1,           // num_items
                                              0));                  // NOLINT: this is the offset of the sum, not a pointer
  discardResult(cudaMalloc(&d_temp_storage_2, temp_storage_bytes_2));
  gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage_2,     // d_temp_storage
                                              temp_storage_bytes_2, // temp_storage_bytes
                                              neighboursIndexTable, // d_in
                                              neighboursIndexTable, // d_out
                                              nCells + 1,           // num_items
                                              0));                  // NOLINT: this is the offset of the sum, not a pointer
  gpuCheckError(cudaFree(d_temp_storage));
  gpuCheckError(cudaFree(d_temp_storage_2));
  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}

void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                  int* neighboursLUT,
                                  int** cellsLUTs,
                                  gpuPair<int, int>* cellNeighbours,
                                  int* neighboursIndexTable,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const int layerIndex,
                                  const unsigned int nCells,
                                  const unsigned int nCellsNext,
                                  const int maxCellNeighbours,
                                  const int nBlocks,
                                  const int nThreads)
{

  gpu::computeLayerCellNeighboursKernel<false><<<o2::gpu::CAMath::Min(nBlocks, GPU_BLOCKS),
                                                 o2::gpu::CAMath::Min(nThreads, GPU_THREADS)>>>(
    cellsLayersDevice,
    neighboursLUT,
    neighboursIndexTable,
    cellsLUTs,
    cellNeighbours,
    maxChi2ClusterAttachment,
    bz,
    layerIndex,
    nCells,
    maxCellNeighbours);
  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}

int filterCellNeighboursHandler(std::vector<int>& neighHost, // TODO: eventually remove this!
                                gpuPair<int, int>* cellNeighbourPairs,
                                int* cellNeighbours,
                                unsigned int nNeigh)
{
  thrust::device_ptr<gpuPair<int, int>> neighVectorPairs(cellNeighbourPairs);
  thrust::device_ptr<int> validNeighs(cellNeighbours);
  thrust::device_vector<int> keys(nNeigh); // TODO: externally allocate.
  thrust::device_vector<int> vals(nNeigh); // TODO: externally allocate.
  thrust::copy(thrust::make_transform_iterator(neighVectorPairs, gpu::pair_to_second<int, int>()),
               thrust::make_transform_iterator(neighVectorPairs + nNeigh, gpu::pair_to_second<int, int>()),
               keys.begin());
  thrust::sequence(vals.begin(), vals.end());
  thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
  thrust::device_vector<gpuPair<int, int>> sortedNeigh(nNeigh);
  thrust::copy(thrust::make_permutation_iterator(neighVectorPairs, vals.begin()),
               thrust::make_permutation_iterator(neighVectorPairs, vals.end()),
               sortedNeigh.begin());
  discardResult(cudaDeviceSynchronize());
  auto trimmedBegin = thrust::find_if(sortedNeigh.begin(), sortedNeigh.end(), gpu::is_valid_pair<int, int>()); // trim leading -1s
  auto trimmedSize = sortedNeigh.end() - trimmedBegin;
  neighHost.resize(trimmedSize);
  thrust::transform(trimmedBegin, sortedNeigh.end(), validNeighs, gpu::pair_to_first<int, int>());
  gpuCheckError(cudaMemcpy(neighHost.data(), cellNeighbours, trimmedSize * sizeof(int), cudaMemcpyDeviceToHost));

  return trimmedSize;
}

template <int nLayers>
void processNeighboursHandler(const int startLayer,
                              const int startLevel,
                              CellSeed** allCellSeeds,
                              CellSeed* currentCellSeeds,
                              std::array<int, nLayers - 2>& nCells,
                              const unsigned char** usedClusters,
                              std::array<int*, nLayers - 2>& neighbours,
                              gsl::span<int*> neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              std::vector<CellSeed>& seedsHost,
                              const float bz,
                              const float maxChi2ClusterAttachment,
                              const float maxChi2NDF,
                              const o2::base::Propagator* propagator,
                              const o2::base::PropagatorF::MatCorrType matCorrType,
                              const int nBlocks,
                              const int nThreads)
{
  thrust::device_vector<int> foundSeedsTable(nCells[startLayer] + 1); // Shortcut: device_vector skips central memory management, we are relying on the contingency. TODO: fix this.
                                                                      // thrust::device_vector<int> lastCellIds(lastCellIdHost);
                                                                      // thrust::device_vector<CellSeed> lastCellSeed(lastCellSeedHost);
  thrust::device_vector<int> lastCellId, updatedCellId;
  thrust::device_vector<CellSeed> lastCellSeed, updatedCellSeed;
  gpu::processNeighboursKernel<true><<<nBlocks, nThreads>>>(startLayer,
                                                            startLevel,
                                                            allCellSeeds,
                                                            currentCellSeeds,
                                                            nullptr,
                                                            nCells[startLayer],
                                                            nullptr,
                                                            nullptr,
                                                            thrust::raw_pointer_cast(&foundSeedsTable[0]),
                                                            usedClusters,
                                                            neighbours[startLayer - 1],
                                                            neighboursDeviceLUTs[startLayer - 1],
                                                            foundTrackingFrameInfo,
                                                            bz,
                                                            maxChi2ClusterAttachment,
                                                            propagator,
                                                            matCorrType);
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  gpuCheckError(cub::DeviceScan::ExclusiveSum(nullptr,                                       // d_temp_storage
                                              temp_storage_bytes,                            // temp_storage_bytes
                                              thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_in
                                              thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_out
                                              nCells[startLayer] + 1,                        // num_items
                                              0));                                           // NOLINT: this is the offset of the sum, not a pointer
  discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,                                // d_temp_storage
                                              temp_storage_bytes,                            // temp_storage_bytes
                                              thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_in
                                              thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_out
                                              nCells[startLayer] + 1,                        // num_items
                                              0));                                           // NOLINT: this is the offset of the sum, not a pointer

  updatedCellId.resize(foundSeedsTable.back());
  updatedCellSeed.resize(foundSeedsTable.back());

  gpu::processNeighboursKernel<false><<<nBlocks, nThreads>>>(startLayer,
                                                             startLevel,
                                                             allCellSeeds,
                                                             currentCellSeeds,
                                                             nullptr,
                                                             nCells[startLayer],
                                                             thrust::raw_pointer_cast(&updatedCellSeed[0]),
                                                             thrust::raw_pointer_cast(&updatedCellId[0]),
                                                             thrust::raw_pointer_cast(&foundSeedsTable[0]),
                                                             usedClusters,
                                                             neighbours[startLayer - 1],
                                                             neighboursDeviceLUTs[startLayer - 1],
                                                             foundTrackingFrameInfo,
                                                             bz,
                                                             maxChi2ClusterAttachment,
                                                             propagator,
                                                             matCorrType);
  auto t1 = updatedCellSeed.size();
  gpuCheckError(cudaFree(d_temp_storage));
  int level = startLevel;
  for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
    temp_storage_bytes = 0;
    lastCellSeed.swap(updatedCellSeed);
    lastCellId.swap(updatedCellId);
    thrust::device_vector<CellSeed>().swap(updatedCellSeed);
    thrust::device_vector<int>().swap(updatedCellId);
    auto lastCellSeedSize{lastCellSeed.size()};
    foundSeedsTable.resize(nCells[iLayer] + 1);
    thrust::fill(foundSeedsTable.begin(), foundSeedsTable.end(), 0);
    --level;
    gpu::processNeighboursKernel<true><<<nBlocks, nThreads>>>(iLayer,
                                                              level,
                                                              allCellSeeds,
                                                              thrust::raw_pointer_cast(&lastCellSeed[0]),
                                                              thrust::raw_pointer_cast(&lastCellId[0]),
                                                              lastCellSeedSize,
                                                              nullptr,
                                                              nullptr,
                                                              thrust::raw_pointer_cast(&foundSeedsTable[0]),
                                                              usedClusters,
                                                              neighbours[iLayer - 1],
                                                              neighboursDeviceLUTs[iLayer - 1],
                                                              foundTrackingFrameInfo,
                                                              bz,
                                                              maxChi2ClusterAttachment,
                                                              propagator,
                                                              matCorrType);
    gpuCheckError(cub::DeviceScan::ExclusiveSum(nullptr,                                       // d_temp_storage
                                                temp_storage_bytes,                            // temp_storage_bytes
                                                thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_in
                                                thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_out
                                                nCells[iLayer] + 1,                            // num_items
                                                0));                                           // NOLINT: this is the offset of the sum, not a pointer
    discardResult(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    gpuCheckError(cub::DeviceScan::ExclusiveSum(d_temp_storage,                                // d_temp_storage
                                                temp_storage_bytes,                            // temp_storage_bytes
                                                thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_in
                                                thrust::raw_pointer_cast(&foundSeedsTable[0]), // d_out
                                                nCells[iLayer] + 1,                            // num_items
                                                0));                                           // NOLINT: this is the offset of the sum, not a pointer
    auto foundSeeds{foundSeedsTable.back()};
    updatedCellId.resize(foundSeeds);
    thrust::fill(updatedCellId.begin(), updatedCellId.end(), 0);
    updatedCellSeed.resize(foundSeeds);
    thrust::fill(updatedCellSeed.begin(), updatedCellSeed.end(), CellSeed());

    gpu::processNeighboursKernel<false><<<nBlocks, nThreads>>>(iLayer,
                                                               level,
                                                               allCellSeeds,
                                                               thrust::raw_pointer_cast(&lastCellSeed[0]),
                                                               thrust::raw_pointer_cast(&lastCellId[0]),
                                                               lastCellSeedSize,
                                                               thrust::raw_pointer_cast(&updatedCellSeed[0]),
                                                               thrust::raw_pointer_cast(&updatedCellId[0]),
                                                               thrust::raw_pointer_cast(&foundSeedsTable[0]),
                                                               usedClusters,
                                                               neighbours[iLayer - 1],
                                                               neighboursDeviceLUTs[iLayer - 1],
                                                               foundTrackingFrameInfo,
                                                               bz,
                                                               maxChi2ClusterAttachment,
                                                               propagator,
                                                               matCorrType);
    gpuCheckError(cudaFree(d_temp_storage));
  }
  thrust::device_vector<CellSeed> outSeeds(updatedCellSeed.size());
  auto end = thrust::copy_if(updatedCellSeed.begin(), updatedCellSeed.end(), outSeeds.begin(), gpu::seed_selector(1.e3, maxChi2NDF * ((startLevel + 2) * 2 - 5)));
  auto s{end - outSeeds.begin()};
  std::vector<CellSeed> outSeedsHost(s);
  thrust::copy(updatedCellSeed.begin(), updatedCellSeed.begin() + s, outSeedsHost.begin());
  seedsHost.insert(seedsHost.end(), outSeedsHost.begin(), outSeedsHost.end());
}

void trackSeedHandler(CellSeed* trackSeeds,
                      const TrackingFrameInfo** foundTrackingFrameInfo,
                      o2::its::TrackITSExt* tracks,
                      std::vector<float>& minPtsHost,
                      const unsigned int nSeeds,
                      const float bz,
                      const int startLevel,
                      float maxChi2ClusterAttachment,
                      float maxChi2NDF,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType,
                      const int nBlocks,
                      const int nThreads)
{
  thrust::device_vector<float> minPts(minPtsHost);
  gpu::fitTrackSeedsKernel<<<nBlocks, nThreads>>>(
    trackSeeds,                           // CellSeed*
    foundTrackingFrameInfo,               // TrackingFrameInfo**
    tracks,                               // TrackITSExt*
    thrust::raw_pointer_cast(&minPts[0]), // const float* minPts,
    nSeeds,                               // const unsigned int
    bz,                                   // const float
    startLevel,                           // const int
    maxChi2ClusterAttachment,             // float
    maxChi2NDF,                           // float
    propagator,                           // const o2::base::Propagator*
    matCorrType);                         // o2::base::PropagatorF::MatCorrType
  thrust::device_ptr<o2::its::TrackITSExt> tr_ptr(tracks);

  thrust::sort(tr_ptr, tr_ptr + nSeeds, gpu::compare_track_chi2());
  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}

template void countTrackletsInROFsHandler<7>(const IndexTableUtils* utils,
                                             const uint8_t* multMask,
                                             const int startROF,
                                             const int endROF,
                                             const int maxROF,
                                             const int deltaROF,
                                             const int vertexId,
                                             const Vertex* vertices,
                                             const int* rofPV,
                                             const int nVertices,
                                             const Cluster** clusters,
                                             std::vector<unsigned int> nClusters,
                                             const int** ROFClusters,
                                             const unsigned char** usedClusters,
                                             const int** clustersIndexTables,
                                             int** trackletsLUTs,
                                             gsl::span<int*> trackletsLUTsHost,
                                             const int iteration,
                                             const float NSigmaCut,
                                             std::vector<float>& phiCuts,
                                             const float resolutionPV,
                                             std::vector<float>& minRs,
                                             std::vector<float>& maxRs,
                                             std::vector<float>& resolutions,
                                             std::vector<float>& radii,
                                             std::vector<float>& mulScatAng,
                                             const int nBlocks,
                                             const int nThreads);

template void computeTrackletsInROFsHandler<7>(const IndexTableUtils* utils,
                                               const uint8_t* multMask,
                                               const int startROF,
                                               const int endROF,
                                               const int maxROF,
                                               const int deltaROF,
                                               const int vertexId,
                                               const Vertex* vertices,
                                               const int* rofPV,
                                               const int nVertices,
                                               const Cluster** clusters,
                                               std::vector<unsigned int> nClusters,
                                               const int** ROFClusters,
                                               const unsigned char** usedClusters,
                                               const int** clustersIndexTables,
                                               Tracklet** tracklets,
                                               gsl::span<Tracklet*> spanTracklets,
                                               gsl::span<int> nTracklets,
                                               int** trackletsLUTs,
                                               gsl::span<int*> trackletsLUTsHost,
                                               const int iteration,
                                               const float NSigmaCut,
                                               std::vector<float>& phiCuts,
                                               const float resolutionPV,
                                               std::vector<float>& minRs,
                                               std::vector<float>& maxRs,
                                               std::vector<float>& resolutions,
                                               std::vector<float>& radii,
                                               std::vector<float>& mulScatAng,
                                               const int nBlocks,
                                               const int nThreads);

template void processNeighboursHandler<7>(const int startLayer,
                                          const int startLevel,
                                          CellSeed** allCellSeeds,
                                          CellSeed* currentCellSeeds,
                                          std::array<int, 5>& nCells,
                                          const unsigned char** usedClusters,
                                          std::array<int*, 5>& neighbours,
                                          gsl::span<int*> neighboursDeviceLUTs,
                                          const TrackingFrameInfo** foundTrackingFrameInfo,
                                          std::vector<CellSeed>& seedsHost,
                                          const float bz,
                                          const float maxChi2ClusterAttachment,
                                          const float maxChi2NDF,
                                          const o2::base::Propagator* propagator,
                                          const o2::base::PropagatorF::MatCorrType matCorrType,
                                          const int nBlocks,
                                          const int nThreads);
} // namespace o2::its
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

namespace gpu
{
GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float Bz,
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
                            Bz,
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
  return o2::gpu::GPUCommonMath::Abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
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

  const bool zeroField{o2::gpu::GPUCommonMath::Abs(bz) < o2::constants::math::Almost0};
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

template <int nLayers>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const unsigned int nSeeds,
  const float Bz,
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
                               Bz,                           // float Bz,
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
                          Bz,                       // float Bz,
                          foundTrackingFrameInfo,   // TrackingFrameInfo** trackingFrameInfo,
                          propagator,               // const o2::base::Propagator* propagator,
                          matCorrType);             // o2::base::PropagatorF::MatCorrType matCorrType
    if (!fitSuccess) {
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
          // atomicExch(cellSeedArray[layerIndex + 1][iNextCell].getLevelPtr(), currentCellLevel + 1); // Update level on corresponding cell
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
  const Tracklet** tracklets,
  const int** trackletsLUT,
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
    const int nextLayerFirstTrackletIndex{trackletsLUT[layer][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[layer][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[layer + 1][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[layer + 1][iNextTrackletIndex];
      const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};

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

/////////////////////////////////////////
// Debug Kernels
/////////////////////////////////////////
GPUd() const int4 getBinsRect(const Cluster& currentCluster, const int layerIndex,
                              const o2::its::IndexTableUtils& utils,
                              const float z1, const float z2, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

GPUhd() float Sq(float q)
{
  return q * q;
}

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

// Functors to sort tracklets
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

// Print layer buffer
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

// Dump vertices
GPUg() void printVertices(const Vertex* v, unsigned int size, const unsigned int tId = 0)
{
  if (blockIdx.x * blockDim.x + threadIdx.x == tId) {
    printf("vertices: ");
    for (int i{0}; i < size; ++i) {
      printf("x=%f y=%f z=%f\n", v[i].getX(), v[i].getY(), v[i].getZ());
    }
  }
}

// Dump tracklets
GPUg() void printTracklets(const Tracklet* t,
                           const int offset,
                           const int startRof,
                           const int nrof,
                           const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
                           const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
                           const int maxClustersPerRof = 5e2,
                           const int maxTrackletsPerCluster = 50,
                           const unsigned int tId = 0)
{
  if (threadIdx.x == tId) {
    auto offsetCurrent{roFrameClustersCurrentLayer[offset]};
    auto offsetNext{roFrameClustersNextLayer[offset]};
    auto offsetChunk{(startRof - offset) * maxClustersPerRof * maxTrackletsPerCluster};
    for (int i{offsetChunk}; i < offsetChunk + nrof * maxClustersPerRof * maxTrackletsPerCluster; ++i) {
      if (t[i].firstClusterIndex != -1) {
        t[i].dump(offsetCurrent, offsetNext);
      }
    }
  }
}

GPUg() void printTrackletsNotStrided(const Tracklet* t,
                                     const int offset,
                                     const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
                                     const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
                                     const int ntracklets,
                                     const unsigned int tId = 0)
{
  if (threadIdx.x == tId) {
    auto offsetCurrent{roFrameClustersCurrentLayer[offset]};
    auto offsetNext{roFrameClustersNextLayer[offset]};
    for (int i{0}; i < ntracklets; ++i) {
      t[i].dump(offsetCurrent, offsetNext);
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

// Compute the tracklets for a given layer
template <int nLayers = 7>
GPUg() void computeLayerTrackletsKernelSingleRof(
  const short rof0,
  const short maxRofs,
  const int layerIndex,
  const Cluster* clustersCurrentLayer,        // input data rof0
  const Cluster* clustersNextLayer,           // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const int* indexTable,                      // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const int* roFrameClusters,                 // input data O(1)
  const int* roFrameClustersNext,             // input data O(1)
  const unsigned char* usedClustersLayer,     // input data rof0
  const unsigned char* usedClustersNextLayer, // input data rof1
  const Vertex* vertices,                     // input data
  int* trackletsLookUpTable,                  // output data
  Tracklet* tracklets,                        // output data
  const int nVertices,
  const int currentLayerClustersSize,
  const float phiCut,
  const float minR,
  const float maxR,
  const float meanDeltaR,
  const float positionResolution,
  const float mSAngle,
  const StaticTrackingParameters<nLayers>* trkPars,
  const IndexTableUtils* utils,
  const unsigned int maxTrackletsPerCluster = 50)
{
  for (int currentClusterIndex = blockIdx.x * blockDim.x + threadIdx.x; currentClusterIndex < currentLayerClustersSize; currentClusterIndex += blockDim.x * gridDim.x) {
    unsigned int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[currentClusterIndex]};
    const int currentSortedIndex{roFrameClusters[rof0] + currentClusterIndex};
    if (usedClustersLayer[currentSortedIndex]) {
      continue;
    }
    short minRof = (rof0 >= trkPars->DeltaROF) ? rof0 - trkPars->DeltaROF : 0;
    short maxRof = (rof0 == static_cast<short>(maxRofs - trkPars->DeltaROF)) ? rof0 : rof0 + trkPars->DeltaROF;
    const float inverseR0{1.f / currentCluster.radius};
    for (int iPrimaryVertex{0}; iPrimaryVertex < nVertices; iPrimaryVertex++) {
      const auto& primaryVertex{vertices[iPrimaryVertex]};
      if (primaryVertex.getX() == 0.f && primaryVertex.getY() == 0.f && primaryVertex.getZ() == 0.f) {
        continue;
      }
      const float resolution{o2::gpu::GPUCommonMath::Sqrt(Sq(trkPars->PVres) / primaryVertex.getNContributors() + Sq(positionResolution))};
      const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
      const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
      const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
      const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
      const float sigmaZ{o2::gpu::CAMath::Sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

      const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * trkPars->NSigmaCut, phiCut)};
      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += trkPars->PhiBins;
      }
      constexpr int tableSize{256 * 128 + 1}; // hardcoded for the time being

      for (short rof1{minRof}; rof1 <= maxRof; ++rof1) {
        if (!(roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1])) { // number of clusters on next layer > 0
          continue;
        }
        for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
          int iPhiBin = (selectedBinsRect.y + iPhiCount) % trkPars->PhiBins;
          const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
          const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
          const int firstRowClusterIndex = indexTable[rof1 * tableSize + firstBinIndex];
          const int maxRowClusterIndex = indexTable[rof1 * tableSize + maxBinIndex];
          for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
            if (iNextCluster >= (roFrameClustersNext[rof1 + 1] - roFrameClustersNext[rof1])) {
              break;
            }
            const Cluster& nextCluster{getPtrFromRuler<Cluster>(rof1, clustersNextLayer, roFrameClustersNext)[iNextCluster]};
            if (usedClustersNextLayer[nextCluster.clusterId]) {
              continue;
            }
            const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
            const float deltaZ{o2::gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};

            if (deltaZ / sigmaZ < trkPars->NSigmaCut && (deltaPhi < phiCut || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < phiCut)) {
              trackletsLookUpTable[currentSortedIndex]++; // Race-condition safe
              const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
              const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
              const unsigned int stride{currentClusterIndex * maxTrackletsPerCluster};
              new (tracklets + stride + storedTracklets) Tracklet{currentSortedIndex, roFrameClustersNext[rof1] + iNextCluster, tanL, phi, rof0, rof1};
              ++storedTracklets;
            }
          }
        }
      }
    }
    // if (storedTracklets > maxTrackletsPerCluster) {
    //   printf("its-gpu-tracklet finder: found more tracklets per clusters (%d) than maximum set (%d), check the configuration!\n", maxTrackletsPerCluster, storedTracklets);
    // }
  }
}

template <int nLayers = 7>
GPUg() void compileTrackletsLookupTableKernel(const Tracklet* tracklets,
                                              int* trackletsLookUpTable,
                                              const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    auto& tracklet{tracklets[currentTrackletIndex]};
    if (tracklet.firstClusterIndex >= 0) {
      atomicAdd(trackletsLookUpTable + tracklet.firstClusterIndex, 1);
    }
  }
}

template <int nLayers = 7>
GPUg() void computeLayerTrackletsKernelMultipleRof(
  const int layerIndex,
  const int iteration,
  const unsigned int startRofId,
  const unsigned int rofSize,
  const int maxRofs,
  const Cluster* clustersCurrentLayer,        // input data rof0
  const Cluster* clustersNextLayer,           // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const int* roFrameClustersCurrentLayer,     // Number of clusters on layer 0 per ROF
  const int* roFrameClustersNextLayer,        // Number of clusters on layer 1 per ROF
  const int* indexTablesNext,                 // input data rof0-delta <rof0< rof0+delta (up to 3 rofs)
  const unsigned char* usedClustersLayer,     // input data rof0
  const unsigned char* usedClustersNextLayer, // input data rof1
  Tracklet* tracklets,                        // output data
  const Vertex* vertices,
  const int* nVertices,
  const float phiCut,
  const float minR,
  const float maxR,
  const float meanDeltaR,
  const float positionResolution,
  const float mSAngle,
  const StaticTrackingParameters<nLayers>* trkPars,
  const IndexTableUtils* utils,
  const unsigned int maxClustersPerRof = 5e2,
  const unsigned int maxTrackletsPerCluster = 50)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  for (unsigned int iRof{blockIdx.x}; iRof < rofSize; iRof += gridDim.x) {
    auto rof0 = iRof + startRofId;
    auto nClustersCurrentLayerRof = o2::gpu::GPUCommonMath::Min(roFrameClustersCurrentLayer[rof0 + 1] - roFrameClustersCurrentLayer[rof0], (int)maxClustersPerRof);
    // if (nClustersCurrentLayerRof > maxClustersPerRof) {
    //   printf("its-gpu-tracklet finder: on layer %d found more clusters per ROF (%d) than maximum set (%d), check the configuration!\n", layerIndex, nClustersCurrentLayerRof, maxClustersPerRof);
    // }
    auto* clustersCurrentLayerRof = clustersCurrentLayer + (roFrameClustersCurrentLayer[rof0] - roFrameClustersCurrentLayer[startRofId]);
    auto nVerticesRof0 = nVertices[rof0 + 1] - nVertices[rof0];
    auto trackletsRof0 = tracklets + maxTrackletsPerCluster * maxClustersPerRof * iRof;
    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < nClustersCurrentLayerRof; currentClusterIndex += blockDim.x) {
      unsigned int storedTracklets{0};
      const Cluster& currentCluster{clustersCurrentLayerRof[currentClusterIndex]};
      const int currentSortedIndex{roFrameClustersCurrentLayer[rof0] + currentClusterIndex};
      const int currentSortedIndexChunk{currentSortedIndex - roFrameClustersCurrentLayer[startRofId]};
      if (usedClustersLayer[currentSortedIndex]) {
        continue;
      }

      int minRof = (rof0 >= trkPars->DeltaROF) ? rof0 - trkPars->DeltaROF : 0;
      int maxRof = (rof0 == maxRofs - trkPars->DeltaROF) ? rof0 : rof0 + trkPars->DeltaROF; // works with delta = {0, 1}
      const float inverseR0{1.f / currentCluster.radius};

      for (int iPrimaryVertex{0}; iPrimaryVertex < nVerticesRof0; iPrimaryVertex++) {
        const auto& primaryVertex{vertices[nVertices[rof0] + iPrimaryVertex]};
        const float resolution{o2::gpu::GPUCommonMath::Sqrt(Sq(trkPars->PVres) / primaryVertex.getNContributors() + Sq(positionResolution))};
        const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
        const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
        const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
        const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
        const float sigmaZ{o2::gpu::CAMath::Sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * mSAngle))};

        const int4 selectedBinsRect{getBinsRect(currentCluster, layerIndex, *utils, zAtRmin, zAtRmax, sigmaZ * trkPars->NSigmaCut, phiCut)};

        if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
          continue;
        }
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
        if (phiBinsNum < 0) {
          phiBinsNum += trkPars->PhiBins;
        }
        const int tableSize{phiBins * zBins + 1};
        for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
          auto nClustersNext{roFrameClustersNextLayer[rof1 + 1] - roFrameClustersNextLayer[rof1]};
          if (!nClustersNext) { // number of clusters on next layer > 0
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % trkPars->PhiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTablesNext[(rof1 - startRofId) * tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTablesNext[(rof1 - startRofId) * tableSize + maxBinIndex];
            for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
              if (iNextCluster >= nClustersNext) {
                break;
              }
              auto nextClusterIndex{roFrameClustersNextLayer[rof1] - roFrameClustersNextLayer[startRofId] + iNextCluster};
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClustersNextLayer[nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};

              if ((deltaZ / sigmaZ < trkPars->NSigmaCut && (deltaPhi < phiCut || o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < phiCut))) {
                const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                const unsigned int stride{currentClusterIndex * maxTrackletsPerCluster};
                if (storedTracklets < maxTrackletsPerCluster) {
                  new (trackletsRof0 + stride + storedTracklets) Tracklet{currentSortedIndexChunk, nextClusterIndex, tanL, phi, static_cast<short>(rof0), static_cast<short>(rof1)};
                }
                // else {
                // printf("its-gpu-tracklet-finder: on rof %d layer: %d: found more tracklets (%d) than maximum allowed per cluster. This is lossy!\n", rof0, layerIndex, storedTracklets);
                // }
                ++storedTracklets;
              }
            }
          }
        }
      }
    }
  }
}

// Decrease LUT entries corresponding to duplicated tracklets. NB: duplicate tracklets are removed separately (see const Tracklets*).
GPUg() void removeDuplicateTrackletsEntriesLUTKernel(
  int* trackletsLookUpTable,
  const Tracklet* tracklets,
  const int* nTracklets,
  const int layerIndex)
{
  int id0{-1}, id1{-1};
  for (int iTracklet{0}; iTracklet < nTracklets[layerIndex]; ++iTracklet) {
    auto& trk = tracklets[iTracklet];
    if (trk.firstClusterIndex == id0 && trk.secondClusterIndex == id1) {
      trackletsLookUpTable[id0]--;
    } else {
      id0 = trk.firstClusterIndex;
      id1 = trk.secondClusterIndex;
    }
  }
}

} // namespace gpu

void countCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  const Tracklet** tracklets,
  const int** trackletsLUT,
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
  // gpu::printBufferLayerOnThread<<<1, 1>>>(layer, cellsLUTsHost, nTracklets + 1);
  gpuCheckError(cudaFree(d_temp_storage));
}

void computeCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  const Tracklet** tracklets,
  const int** trackletsLUT,
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
  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
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

  gpu::computeLayerCellNeighboursKernel<false><<<o2::gpu::GPUCommonMath::Min(nBlocks, GPU_BLOCKS),
                                                 o2::gpu::GPUCommonMath::Min(nThreads, GPU_THREADS)>>>(
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

void filterCellNeighboursHandler(std::vector<int>& neighHost,
                                 gpuPair<int, int>* cellNeighbours,
                                 unsigned int nNeigh)
{
  thrust::device_ptr<gpuPair<int, int>> neighVector(cellNeighbours);
  thrust::device_vector<int> keys(nNeigh); // TODO: externally allocate.
  thrust::device_vector<int> vals(nNeigh); // TODO: externally allocate.
  thrust::copy(thrust::make_transform_iterator(neighVector, gpu::pair_to_second<int, int>()),
               thrust::make_transform_iterator(neighVector + nNeigh, gpu::pair_to_second<int, int>()),
               keys.begin());
  thrust::sequence(vals.begin(), vals.end());
  thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
  thrust::device_vector<gpuPair<int, int>> sortedNeigh(nNeigh);
  thrust::copy(thrust::make_permutation_iterator(neighVector, vals.begin()),
               thrust::make_permutation_iterator(neighVector, vals.end()),
               sortedNeigh.begin());
  discardResult(cudaDeviceSynchronize());
  auto trimmedBegin = thrust::find_if(sortedNeigh.begin(), sortedNeigh.end(), gpu::is_valid_pair<int, int>()); // trim leading -1s
  auto trimmedSize = sortedNeigh.end() - trimmedBegin;
  thrust::device_vector<int> validNeigh(trimmedSize);
  neighHost.resize(trimmedSize);
  thrust::transform(trimmedBegin, sortedNeigh.end(), validNeigh.begin(), gpu::pair_to_first<int, int>());
  gpuCheckError(cudaMemcpy(neighHost.data(), thrust::raw_pointer_cast(validNeigh.data()), trimmedSize * sizeof(int), cudaMemcpyDeviceToHost));
}

void trackSeedHandler(CellSeed* trackSeeds,
                      const TrackingFrameInfo** foundTrackingFrameInfo,
                      o2::its::TrackITSExt* tracks,
                      const unsigned int nSeeds,
                      const float Bz,
                      const int startLevel,
                      float maxChi2ClusterAttachment,
                      float maxChi2NDF,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType,
                      const int nBlocks,
                      const int nThreads)
{
  gpu::fitTrackSeedsKernel<<<nBlocks, nThreads>>>(
    trackSeeds,               // CellSeed*
    foundTrackingFrameInfo,   // TrackingFrameInfo**
    tracks,                   // TrackITSExt*
    nSeeds,                   // const unsigned int
    Bz,                       // const float
    startLevel,               // const int
    maxChi2ClusterAttachment, // float
    maxChi2NDF,               // float
    propagator,               // const o2::base::Propagator*
    matCorrType);             // o2::base::PropagatorF::MatCorrType

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}
} // namespace o2::its

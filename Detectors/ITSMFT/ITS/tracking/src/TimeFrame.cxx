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
/// \file TimeFrame.cxx
/// \brief
///

#include "ITStracking/TimeFrame.h"
#include "ITStracking/MathUtils.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/TrackingConfigParam.h"

#include <iostream>

namespace
{
struct ClusterHelper {
  float phi;
  float r;
  int bin;
  int ind;
};

inline float MSangle(float mass, float p, float xX0)
{
  float beta = p / o2::gpu::CAMath::Hypot(mass, p);
  return 0.0136f * o2::gpu::CAMath::Sqrt(xX0) * (1.f + 0.038f * o2::gpu::CAMath::Log(xX0)) / (beta * p);
}

inline float Sq(float v)
{
  return v * v;
}

} // namespace

namespace o2::its
{

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

template <int nLayers>
TimeFrame<nLayers>::TimeFrame()
{
  resetVectors();
}

template <int nLayers>
TimeFrame<nLayers>::~TimeFrame()
{
  resetVectors();
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVertices(const bounded_vector<Vertex>& vertices)
{
  for (const auto& vertex : vertices) {
    mPrimaryVertices.emplace_back(vertex);
    if (!isBeamPositionOverridden) {
      const int w{vertex.getNContributors()};
      mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.getX() * w) / (mBeamPosWeight + w);
      mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.getY() * w) / (mBeamPosWeight + w);
      mBeamPosWeight += w;
    }
  }
  mROFramesPV.push_back(mPrimaryVertices.size());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVertices(const bounded_vector<Vertex>& vertices, const int rofId, const int iteration)
{
  addPrimaryVertices(gsl::span<const Vertex>(vertices), rofId, iteration);
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesLabels(bounded_vector<std::pair<MCCompLabel, float>>& labels)
{
  mVerticesMCRecInfo.insert(mVerticesMCRecInfo.end(), labels.begin(), labels.end());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesInROF(const bounded_vector<Vertex>& vertices, const int rofId, const int iteration)
{
  mPrimaryVertices.insert(mPrimaryVertices.begin() + mROFramesPV[rofId], vertices.begin(), vertices.end());
  for (int i = rofId + 1; i < mROFramesPV.size(); ++i) {
    mROFramesPV[i] += vertices.size();
  }
  mTotVertPerIteration[iteration] += vertices.size();
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesLabelsInROF(const bounded_vector<std::pair<MCCompLabel, float>>& labels, const int rofId)
{
  mVerticesMCRecInfo.insert(mVerticesMCRecInfo.begin() + mROFramesPV[rofId], labels.begin(), labels.end());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVertices(const gsl::span<const Vertex>& vertices, const int rofId, const int iteration)
{
  bounded_vector<Vertex> futureVertices(mMemoryPool.get());
  for (const auto& vertex : vertices) {
    if (vertex.getTimeStamp().getTimeStamp() < rofId) { // put a copy in the past
      insertPastVertex(vertex, iteration);
    } else {
      if (vertex.getTimeStamp().getTimeStamp() > rofId) { // or put a copy in the future
        futureVertices.emplace_back(vertex);
      }
    }
    mPrimaryVertices.emplace_back(vertex); // put a copy in the present
    mTotVertPerIteration[iteration]++;
    if (!isBeamPositionOverridden) { // beam position is updated only at first occurrence of the vertex. A bit sketchy if we have past/future vertices, it should not impact too much.
      const int w{vertex.getNContributors()};
      mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.getX() * w) / (mBeamPosWeight + w);
      mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.getY() * w) / (mBeamPosWeight + w);
      mBeamPosWeight += w;
    }
  }
  mROFramesPV.push_back(mPrimaryVertices.size()); // current rof must have number of vertices up to present
  for (auto& vertex : futureVertices) {
    mPrimaryVertices.emplace_back(vertex);
    mTotVertPerIteration[iteration]++;
  }
}

template <int nLayers>
int TimeFrame<nLayers>::loadROFrameData(gsl::span<o2::itsmft::ROFRecord> rofs,
                                        gsl::span<const itsmft::CompClusterExt> clusters,
                                        gsl::span<const unsigned char>::iterator& pattIt,
                                        const itsmft::TopologyDictionary* dict,
                                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    deepVectorClear(mUnsortedClusters[iLayer], mMemoryPool.get());
    deepVectorClear(mTrackingFrameInfo[iLayer], mMemoryPool.get());
    deepVectorClear(mClusterExternalIndices[iLayer], mMemoryPool.get());
    clearResizeBoundedVector(mROFramesClusters[iLayer], 1, mMemoryPool.get(), 0);

    if (iLayer < 2) {
      deepVectorClear(mTrackletsIndexROF[iLayer], mMemoryPool.get());
      deepVectorClear(mNTrackletsPerCluster[iLayer], mMemoryPool.get());
      deepVectorClear(mNTrackletsPerClusterSum[iLayer], mMemoryPool.get());
    }
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  mNrof = 0;
  clearResizeBoundedVector(mClusterSize, clusters.size(), mMemoryPool.get());
  for (auto& rof : rofs) {
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      auto& c = clusters[clusterId];

      int layer = geom->getLayer(c.getSensorID());

      auto pattID = c.getPatternID();
      o2::math_utils::Point3D<float> locXYZ;
      float sigmaY2 = DefClusError2Row, sigmaZ2 = DefClusError2Col, sigmaYZ = 0; // Dummy COG errors (about half pixel size)
      unsigned int clusterSize{0};
      if (pattID != itsmft::CompCluster::InvalidPatternID) {
        sigmaY2 = dict->getErr2X(pattID);
        sigmaZ2 = dict->getErr2Z(pattID);
        if (!dict->isGroup(pattID)) {
          locXYZ = dict->getClusterCoordinates(c);
          clusterSize = dict->getNpixels(pattID);
        } else {
          o2::itsmft::ClusterPattern patt(pattIt);
          locXYZ = dict->getClusterCoordinates(c, patt);
          clusterSize = patt.getNPixels();
        }
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict->getClusterCoordinates(c, patt, false);
        clusterSize = patt.getNPixels();
      }
      mClusterSize.push_back(std::clamp(clusterSize, 0u, 255u));
      auto sensorID = c.getSensorID();
      // Inverse transformation to the local --> tracking
      auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
      // Transformation to the local --> global
      auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

      addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                  std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                  std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

      /// Rotate to the global frame
      addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), mUnsortedClusters[layer].size());
      addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
      mROFramesClusters[iL].push_back(mUnsortedClusters[iL].size());
    }
    mNrof++;
  }

  for (auto i = 0; i < mNTrackletsPerCluster.size(); ++i) {
    mNTrackletsPerCluster[i].resize(mUnsortedClusters[1].size());
    mNTrackletsPerClusterSum[i].resize(mUnsortedClusters[1].size() + 1); // Exc sum "prepends" a 0
  }

  if (mcLabels != nullptr) {
    mClusterLabels = mcLabels;
  }

  return mNrof;
}

template <int nLayers>
void TimeFrame<nLayers>::prepareClusters(const TrackingParameters& trkParam, const int maxLayers)
{
  bounded_vector<ClusterHelper> cHelper(mMemoryPool.get());
  bounded_vector<int> clsPerBin(trkParam.PhiBins * trkParam.ZBins, 0, mMemoryPool.get());
  for (int rof{0}; rof < mNrof; ++rof) {
    if ((int)mMultiplicityCutMask.size() == mNrof && !mMultiplicityCutMask[rof]) {
      continue;
    }
    for (int iLayer{0}; iLayer < std::min(trkParam.NLayers, maxLayers); ++iLayer) {
      std::fill(clsPerBin.begin(), clsPerBin.end(), 0);
      const auto unsortedClusters{getUnsortedClustersOnLayer(rof, iLayer)};
      const int clustersNum{static_cast<int>(unsortedClusters.size())};

      deepVectorClear(cHelper);
      cHelper.resize(clustersNum);

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {

        const Cluster& c = unsortedClusters[iCluster];
        ClusterHelper& h = cHelper[iCluster];
        float x = c.xCoordinate - mBeamPos[0];
        float y = c.yCoordinate - mBeamPos[1];
        const float& z = c.zCoordinate;
        float phi = math_utils::computePhi(x, y);
        int zBin{mIndexTableUtils.getZBinIndex(iLayer, z)};
        if (zBin < 0) {
          zBin = 0;
          mBogusClusters[iLayer]++;
        } else if (zBin >= trkParam.ZBins) {
          zBin = trkParam.ZBins - 1;
          mBogusClusters[iLayer]++;
        }
        int bin = mIndexTableUtils.getBinIndex(zBin, mIndexTableUtils.getPhiBinIndex(phi));
        h.phi = phi;
        h.r = math_utils::hypot(x, y);
        mMinR[iLayer] = o2::gpu::GPUCommonMath::Min(h.r, mMinR[iLayer]);
        mMaxR[iLayer] = o2::gpu::GPUCommonMath::Max(h.r, mMaxR[iLayer]);
        h.bin = bin;
        h.ind = clsPerBin[bin]++;
      }
      bounded_vector<int> lutPerBin(clsPerBin.size(), 0, mMemoryPool.get());
      lutPerBin[0] = 0;
      for (unsigned int iB{1}; iB < lutPerBin.size(); ++iB) {
        lutPerBin[iB] = lutPerBin[iB - 1] + clsPerBin[iB - 1];
      }

      auto clusters2beSorted{getClustersOnLayer(rof, iLayer)};
      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        const ClusterHelper& h = cHelper[iCluster];

        Cluster& c = clusters2beSorted[lutPerBin[h.bin] + h.ind];
        c = unsortedClusters[iCluster];
        c.phi = h.phi;
        c.radius = h.r;
        c.indexTableBinIndex = h.bin;
      }
      for (unsigned int iB{0}; iB < clsPerBin.size(); ++iB) {
        mIndexTables[iLayer][rof * (trkParam.ZBins * trkParam.PhiBins + 1) + iB] = lutPerBin[iB];
      }
      for (auto iB{clsPerBin.size()}; iB < (trkParam.ZBins * trkParam.PhiBins + 1); iB++) {
        mIndexTables[iLayer][rof * (trkParam.ZBins * trkParam.PhiBins + 1) + iB] = clustersNum;
      }
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::initialise(const int iteration, const TrackingParameters& trkParam, const int maxLayers, bool resetVertices)
{
  if (iteration == 0) {
    if (maxLayers < trkParam.NLayers && resetVertices) {
      resetRofPV();
      deepVectorClear(mTotVertPerIteration);
    }
    deepVectorClear(mTracks);
    deepVectorClear(mTracksLabel);
    deepVectorClear(mLines);
    deepVectorClear(mLinesLabels);
    if (resetVertices) {
      deepVectorClear(mVerticesMCRecInfo);
    }
    clearResizeBoundedVector(mTracks, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mTracksLabel, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mLinesLabels, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mCells, trkParam.CellsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mCellsLookupTable, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellsNeighbours, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellsNeighboursLUT, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellLabels, trkParam.CellsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mTracklets, std::min(trkParam.TrackletsPerRoad(), maxLayers - 1), mMemoryPool.get());
    clearResizeBoundedVector(mTrackletLabels, trkParam.TrackletsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mTrackletsLookupTable, trkParam.TrackletsPerRoad(), mMemoryPool.get());
    mIndexTableUtils.setTrackingParameters(trkParam);
    clearResizeBoundedVector(mPositionResolution, trkParam.NLayers, mMemoryPool.get());
    clearResizeBoundedVector(mBogusClusters, trkParam.NLayers, mMemoryPool.get());
    deepVectorClear(mTrackletClusters);
    for (unsigned int iLayer{0}; iLayer < std::min((int)mClusters.size(), maxLayers); ++iLayer) {
      clearResizeBoundedVector(mClusters[iLayer], mUnsortedClusters[iLayer].size(), mMemoryPool.get());
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), mMemoryPool.get());
      mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt(0.5 * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer]) + trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]);
    }
    clearResizeBoundedArray(mIndexTables, mNrof * (trkParam.ZBins * trkParam.PhiBins + 1), mMemoryPool.get());
    clearResizeBoundedVector(mLines, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mTrackletClusters, mNrof, mMemoryPool.get());

    for (int iLayer{0}; iLayer < trkParam.NLayers; ++iLayer) {
      if (trkParam.SystErrorY2[iLayer] > 0.f || trkParam.SystErrorZ2[iLayer] > 0.f) {
        for (auto& tfInfo : mTrackingFrameInfo[iLayer]) {
          /// Account for alignment systematics in the cluster covariance matrix
          tfInfo.covarianceTrackingFrame[0] += trkParam.SystErrorY2[iLayer];
          tfInfo.covarianceTrackingFrame[2] += trkParam.SystErrorZ2[iLayer];
        }
      }
    }
  }
  mNTrackletsPerROF.resize(2);
  for (auto& v : mNTrackletsPerROF) {
    v = bounded_vector<int>(mNrof + 1, 0, mMemoryPool.get());
  }
  if (iteration == 0 || iteration == 3) {
    prepareClusters(trkParam, maxLayers);
  }
  mTotalTracklets = {0, 0};
  if (maxLayers < trkParam.NLayers) { // Vertexer only, but in both iterations
    for (size_t iLayer{0}; iLayer < maxLayers; ++iLayer) {
      deepVectorClear(mUsedClusters[iLayer]);
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), mMemoryPool.get());
    }
  }

  mTotVertPerIteration.resize(1 + iteration);
  mNoVertexROF = 0;
  deepVectorClear(mRoads);
  deepVectorClear(mRoadLabels);

  mMSangles.resize(trkParam.NLayers);
  mPhiCuts.resize(mClusters.size() - 1, 0.f);

  float oneOverR{0.001f * 0.3f * std::abs(mBz) / trkParam.TrackletMinPt};
  for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
    mMSangles[iLayer] = MSangle(0.14f, trkParam.TrackletMinPt, trkParam.LayerxX0[iLayer]);
    mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt(0.5f * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer]) + trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]);
    if (iLayer < mClusters.size() - 1) {
      const float& r1 = trkParam.LayerRadii[iLayer];
      const float& r2 = trkParam.LayerRadii[iLayer + 1];
      const float res1 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer]);
      const float res2 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer + 1]);
      const float cosTheta1half = o2::gpu::CAMath::Sqrt(1.f - Sq(0.5f * r1 * oneOverR));
      const float cosTheta2half = o2::gpu::CAMath::Sqrt(1.f - Sq(0.5f * r2 * oneOverR));
      float x = r2 * cosTheta1half - r1 * cosTheta2half;
      float delta = o2::gpu::CAMath::Sqrt(1. / (1.f - 0.25f * Sq(x * oneOverR)) * (Sq(0.25f * r1 * r2 * Sq(oneOverR) / cosTheta2half + cosTheta1half) * Sq(res1) + Sq(0.25f * r1 * r2 * Sq(oneOverR) / cosTheta1half + cosTheta2half) * Sq(res2)));
      mPhiCuts[iLayer] = std::min(o2::gpu::CAMath::ASin(0.5f * x * oneOverR) + 2.f * mMSangles[iLayer] + delta, constants::math::Pi * 0.5f);
    }
  }

  for (int iLayer{0}; iLayer < std::min((int)mTracklets.size(), maxLayers); ++iLayer) {
    deepVectorClear(mTracklets[iLayer]);
    deepVectorClear(mTrackletLabels[iLayer]);
    if (iLayer < (int)mCells.size()) {
      deepVectorClear(mCells[iLayer]);
      deepVectorClear(mTrackletsLookupTable[iLayer]);
      mTrackletsLookupTable[iLayer].resize(mClusters[iLayer + 1].size(), 0);
      deepVectorClear(mCellLabels[iLayer]);
    }

    if (iLayer < (int)mCells.size() - 1) {
      deepVectorClear(mCellsLookupTable[iLayer]);
      deepVectorClear(mCellsNeighbours[iLayer]);
      deepVectorClear(mCellsNeighboursLUT[iLayer]);
    }
  }
}

template <int nLayers>
unsigned long TimeFrame<nLayers>::getArtefactsMemory() const
{
  unsigned long size{0};
  for (auto& trkl : mTracklets) {
    size += sizeof(Tracklet) * trkl.size();
  }
  for (auto& cells : mCells) {
    size += sizeof(CellSeed) * cells.size();
  }
  for (auto& cellsN : mCellsNeighbours) {
    size += sizeof(int) * cellsN.size();
  }
  return size + sizeof(Road<nLayers - 2>) * mRoads.size();
}

template <int nLayers>
void TimeFrame<nLayers>::printArtefactsMemory() const
{
  LOGP(info, "TimeFrame: Artefacts occupy {:.2f} MB", getArtefactsMemory() / constants::MB);
}

template <int nLayers>
void TimeFrame<nLayers>::fillPrimaryVerticesXandAlpha()
{
  if (mPValphaX.size()) {
    mPValphaX.clear();
  }
  mPValphaX.reserve(mPrimaryVertices.size());
  for (auto& pv : mPrimaryVertices) {
    mPValphaX.emplace_back(std::array<float, 2>{o2::gpu::CAMath::Hypot(pv.getX(), pv.getY()), math_utils::computePhi(pv.getX(), pv.getY())});
  }
}

template <int nLayers>
void TimeFrame<nLayers>::computeTrackletsPerROFScans()
{
  for (ushort iLayer = 0; iLayer < 2; ++iLayer) {
    for (unsigned int iRof{0}; iRof < mNrof; ++iRof) {
      if (mMultiplicityCutMask[iRof]) {
        mTotalTracklets[iLayer] += mNTrackletsPerROF[iLayer][iRof];
      }
    }
    std::exclusive_scan(mNTrackletsPerROF[iLayer].begin(), mNTrackletsPerROF[iLayer].end(), mNTrackletsPerROF[iLayer].begin(), 0);
    std::exclusive_scan(mNTrackletsPerCluster[iLayer].begin(), mNTrackletsPerCluster[iLayer].end(), mNTrackletsPerClusterSum[iLayer].begin(), 0);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::checkTrackletLUTs()
{
  for (uint32_t iLayer{0}; iLayer < getTracklets().size(); ++iLayer) {
    int prev{-1};
    int count{0};
    for (uint32_t iTracklet{0}; iTracklet < getTracklets()[iLayer].size(); ++iTracklet) {
      auto& trk = getTracklets()[iLayer][iTracklet];
      int currentId{trk.firstClusterIndex};
      if (currentId < prev) {
        std::cout << "First Cluster Index not increasing monotonically on L:T:ID:Prev " << iLayer << "\t" << iTracklet << "\t" << currentId << "\t" << prev << std::endl;
      } else if (currentId == prev) {
        count++;
      } else {
        if (iLayer > 0) {
          auto& lut{getTrackletsLookupTable()[iLayer - 1]};
          if (count != lut[prev + 1] - lut[prev]) {
            std::cout << "LUT count broken " << iLayer - 1 << "\t" << prev << "\t" << count << "\t" << lut[prev + 1] << "\t" << lut[prev] << std::endl;
          }
        }
        count = 1;
      }
      prev = currentId;
      if (iLayer > 0) {
        auto& lut{getTrackletsLookupTable()[iLayer - 1]};
        if (iTracklet >= (uint32_t)(lut[currentId + 1]) || iTracklet < (uint32_t)(lut[currentId])) {
          std::cout << "LUT broken: " << iLayer - 1 << "\t" << currentId << "\t" << iTracklet << std::endl;
        }
      }
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::resetVectors()
{
  mMinR.fill(10000.);
  mMaxR.fill(-1.);
  for (int iLayers{nLayers}; iLayers--;) {
    mClusters[iLayers].clear();
    mUnsortedClusters[iLayers].clear();
    mTrackingFrameInfo[iLayers].clear();
    mClusterExternalIndices[iLayers].clear();
    mUsedClusters[iLayers].clear();
    mROFramesClusters[iLayers].clear();
    mNClustersPerROF[iLayers].clear();
  }
  for (int i{2}; i--;) {
    mTrackletsIndexROF[i].clear();
  }
}

template <int nLayers>
void TimeFrame<nLayers>::resetTracklets()
{
  for (auto& trkl : mTracklets) {
    deepVectorClear(trkl);
  }
  deepVectorClear(mTrackletsLookupTable);
}

template <int nLayers>
void TimeFrame<nLayers>::printTrackletLUTonLayer(int i)
{
  std::cout << "--------" << std::endl
            << "Tracklet LUT " << i << std::endl;
  for (int j : mTrackletsLookupTable[i]) {
    std::cout << j << "\t";
  }
  std::cout << "\n--------" << std::endl
            << std::endl;
}

template <int nLayers>
void TimeFrame<nLayers>::printCellLUTonLayer(int i)
{
  std::cout << "--------" << std::endl
            << "Cell LUT " << i << std::endl;
  for (int j : mCellsLookupTable[i]) {
    std::cout << j << "\t";
  }
  std::cout << "\n--------" << std::endl
            << std::endl;
}

template <int nLayers>
void TimeFrame<nLayers>::printTrackletLUTs()
{
  for (unsigned int i{0}; i < mTrackletsLookupTable.size(); ++i) {
    printTrackletLUTonLayer(i);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printCellLUTs()
{
  for (unsigned int i{0}; i < mCellsLookupTable.size(); ++i) {
    printCellLUTonLayer(i);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printVertices()
{
  std::cout << "Vertices in ROF (nROF = " << mNrof << ", lut size = " << mROFramesPV.size() << ")" << std::endl;
  for (unsigned int iR{0}; iR < mROFramesPV.size(); ++iR) {
    std::cout << mROFramesPV[iR] << "\t";
  }
  std::cout << "\n\n Vertices:" << std::endl;
  for (unsigned int iV{0}; iV < mPrimaryVertices.size(); ++iV) {
    std::cout << mPrimaryVertices[iV].getX() << "\t" << mPrimaryVertices[iV].getY() << "\t" << mPrimaryVertices[iV].getZ() << std::endl;
  }
  std::cout << "--------" << std::endl;
}

template <int nLayers>
void TimeFrame<nLayers>::printROFoffsets()
{
  std::cout << "--------" << std::endl;
  for (unsigned int iLayer{0}; iLayer < mROFramesClusters.size(); ++iLayer) {
    std::cout << "Layer " << iLayer << std::endl;
    for (auto value : mROFramesClusters[iLayer]) {
      std::cout << value << "\t";
    }
    std::cout << std::endl;
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printNClsPerROF()
{
  std::cout << "--------" << std::endl;
  for (unsigned int iLayer{0}; iLayer < mNClustersPerROF.size(); ++iLayer) {
    std::cout << "Layer " << iLayer << std::endl;
    for (auto& value : mNClustersPerROF[iLayer]) {
      std::cout << value << "\t";
    }
    std::cout << std::endl;
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printSliceInfo(const int startROF, const int sliceSize)
{
  std::cout << "Dumping slice of " << sliceSize << " rofs:" << std::endl;
  for (int iROF{startROF}; iROF < startROF + sliceSize; ++iROF) {
    std::cout << "ROF " << iROF << " dump:" << std::endl;
    for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
      std::cout << "Layer " << iLayer << " has: " << getClustersOnLayer(iROF, iLayer).size() << " clusters." << std::endl;
    }
    std::cout << "Number of seeding vertices: " << getPrimaryVertices(iROF).size() << std::endl;
    int iVertex{0};
    for (auto& v : getPrimaryVertices(iROF)) {
      std::cout << "\t vertex " << iVertex++ << ": x=" << v.getX() << " " << " y=" << v.getY() << " z=" << v.getZ() << " has " << v.getNContributors() << " contributors." << std::endl;
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::setMemoryPool(std::shared_ptr<BoundedMemoryResource>& pool)
{
  wipe();
  mMemoryPool = pool;

  auto initVector = [&]<typename T>(bounded_vector<T>& vec) {
    auto alloc = vec.get_allocator().resource();
    if (alloc != mMemoryPool.get()) {
      vec = bounded_vector<T>(mMemoryPool.get());
    }
  };
  auto initArrays = [&]<typename T, size_t S>(std::array<bounded_vector<T>, S>& arr) {
    for (size_t i{0}; i < S; ++i) {
      auto alloc = arr[i].get_allocator().resource();
      if (alloc != mMemoryPool.get()) {
        arr[i] = bounded_vector<T>(mMemoryPool.get());
      }
    }
  };
  auto initVectors = [&]<typename T>(std::vector<bounded_vector<T>>& vec) {
    for (size_t i{0}; i < vec.size(); ++i) {
      auto alloc = vec[i].get_allocator().resource();
      if (alloc != mMemoryPool.get()) {
        vec[i] = bounded_vector<T>(mMemoryPool.get());
      }
    }
  };

  initVector(mTotVertPerIteration);
  initVector(mPrimaryVertices);
  initVector(mROFramesPV);
  initArrays(mClusters);
  initArrays(mTrackingFrameInfo);
  initArrays(mClusterExternalIndices);
  initArrays(mROFramesClusters);
  initArrays(mNTrackletsPerCluster);
  initArrays(mNTrackletsPerClusterSum);
  initArrays(mNClustersPerROF);
  initArrays(mIndexTables);
  initArrays(mUsedClusters);
  initArrays(mUnsortedClusters);
  initVector(mROFramesPV);
  initVector(mPrimaryVertices);
  initVector(mRoads);
  initVector(mRoadLabels);
  initVector(mMSangles);
  initVector(mPhiCuts);
  initVector(mPositionResolution);
  initVector(mClusterSize);
  initVector(mPValphaX);
  initVector(mBogusClusters);
  initArrays(mTrackletsIndexROF);
  initVectors(mTracks);
  initVectors(mTracklets);
  initVectors(mCells);
  initVectors(mCellSeeds);
  initVectors(mCellSeedsChi2);
  initVectors(mCellsNeighbours);
  initVectors(mCellsLookupTable);
}

template <int nLayers>
void TimeFrame<nLayers>::wipe()
{
  deepVectorClear(mUnsortedClusters);
  deepVectorClear(mTracks);
  deepVectorClear(mTracklets);
  deepVectorClear(mCells);
  deepVectorClear(mCellSeeds);
  deepVectorClear(mCellSeedsChi2);
  deepVectorClear(mRoads);
  deepVectorClear(mCellsNeighbours);
  deepVectorClear(mCellsLookupTable);
  deepVectorClear(mTotVertPerIteration);
  deepVectorClear(mPrimaryVertices);
  deepVectorClear(mROFramesPV);
  deepVectorClear(mClusters);
  deepVectorClear(mTrackingFrameInfo);
  deepVectorClear(mClusterExternalIndices);
  deepVectorClear(mROFramesClusters);
  deepVectorClear(mNTrackletsPerCluster);
  deepVectorClear(mNTrackletsPerClusterSum);
  deepVectorClear(mNClustersPerROF);
  deepVectorClear(mIndexTables);
  deepVectorClear(mUsedClusters);
  deepVectorClear(mUnsortedClusters);
  deepVectorClear(mROFramesPV);
  deepVectorClear(mPrimaryVertices);
  deepVectorClear(mRoads);
  deepVectorClear(mRoadLabels);
  deepVectorClear(mMSangles);
  deepVectorClear(mPhiCuts);
  deepVectorClear(mPositionResolution);
  deepVectorClear(mClusterSize);
  deepVectorClear(mPValphaX);
  deepVectorClear(mBogusClusters);
  deepVectorClear(mTrackletsIndexROF);
}

template class TimeFrame<7>;

} // namespace o2::its

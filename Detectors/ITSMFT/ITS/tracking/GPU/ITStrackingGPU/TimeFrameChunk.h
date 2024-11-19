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

#ifndef TRACKINGITSGPU_INCLUDE_TIMEFRAMECHUNKGPU_H
#define TRACKINGITSGPU_INCLUDE_TIMEFRAMECHUNKGPU_H

#include "ITStracking/Configuration.h"
#include "ITStracking/TimeFrame.h"

#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/Array.h"
#include "ITStrackingGPU/Vector.h"
#include "ITStrackingGPU/Stream.h"

#include <gsl/gsl>

namespace o2::its::gpu
{
template <int nLayers>
struct StaticTrackingParameters {
  StaticTrackingParameters<nLayers>& operator=(const StaticTrackingParameters<nLayers>& t) = default;
  void set(const TrackingParameters& pars)
  {
    ClusterSharing = pars.ClusterSharing;
    MinTrackLength = pars.MinTrackLength;
    NSigmaCut = pars.NSigmaCut;
    PVres = pars.PVres;
    DeltaROF = pars.DeltaROF;
    ZBins = pars.ZBins;
    PhiBins = pars.PhiBins;
    CellDeltaTanLambdaSigma = pars.CellDeltaTanLambdaSigma;
  }

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = nLayers;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  int DeltaROF = 0;
  int ZBins{256};
  int PhiBins{128};

  /// Cell finding cuts
  float CellDeltaTanLambdaSigma = 0.007f;
};

template <int nLayers>
class GpuTimeFrameChunk
{
 public:
  static size_t computeScalingSizeBytes(const int, const TimeFrameGPUParameters&);
  static size_t computeFixedSizeBytes(const TimeFrameGPUParameters&);
  static size_t computeRofPerChunk(const TimeFrameGPUParameters&, const size_t);

  GpuTimeFrameChunk() = delete;
  GpuTimeFrameChunk(o2::its::TimeFrame* tf, TimeFrameGPUParameters& conf)
  {
    mTimeFramePtr = tf;
    mTFGPUParams = &conf;
  }
  ~GpuTimeFrameChunk();

  /// Most relevant operations
  void allocate(const size_t, Stream&);
  void reset(const Task, Stream&);
  size_t loadDataOnDevice(const size_t, const size_t, const int, Stream&);

  /// Interface
  Cluster* getDeviceClusters(const int);
  int* getDeviceClusterExternalIndices(const int);
  int* getDeviceIndexTables(const int);
  Tracklet* getDeviceTracklets(const int);
  int* getDeviceTrackletsLookupTables(const int);
  CellSeed* getDeviceCells(const int);
  int* getDeviceCellsLookupTables(const int);
  int* getDeviceRoadsLookupTables(const int);
  TimeFrameGPUParameters* getTimeFrameGPUParameters() const { return mTFGPUParams; }

  int* getDeviceCUBTmpBuffer() { return mCUBTmpBufferDevice; }
  int* getDeviceFoundTracklets() { return mFoundTrackletsDevice; }
  int* getDeviceNFoundCells() { return mNFoundCellsDevice; }
  int* getDeviceCellNeigboursLookupTables(const int);
  int* getDeviceCellNeighbours(const int);
  CellSeed** getDeviceArrayCells() const { return mCellsDeviceArray; }
  int** getDeviceArrayNeighboursCell() const { return mNeighboursCellDeviceArray; }
  int** getDeviceArrayNeighboursCellLUT() const { return mNeighboursCellLookupTablesDeviceArray; }

  /// Vertexer only
  int* getDeviceNTrackletCluster(const int combid) { return mNTrackletsPerClusterDevice[combid]; }
  Line* getDeviceLines() { return mLinesDevice; };
  int* getDeviceNFoundLines() { return mNFoundLinesDevice; }
  int* getDeviceNExclusiveFoundLines() { return mNExclusiveFoundLinesDevice; }
  unsigned char* getDeviceUsedTracklets() { return mUsedTrackletsDevice; }
  int* getDeviceClusteredLines() { return mClusteredLinesDevice; }
  size_t getNPopulatedRof() const { return mNPopulatedRof; }

 private:
  /// Host
  std::array<gsl::span<const Cluster>, nLayers> mHostClusters;
  std::array<gsl::span<const int>, nLayers> mHostIndexTables;

  /// Device
  std::array<Cluster*, nLayers> mClustersDevice;
  std::array<int*, nLayers> mClusterExternalIndicesDevice;
  std::array<int*, nLayers> mIndexTablesDevice;
  std::array<Tracklet*, nLayers - 1> mTrackletsDevice;
  std::array<int*, nLayers - 1> mTrackletsLookupTablesDevice;
  std::array<CellSeed*, nLayers - 2> mCellsDevice;
  // Road<nLayers - 2>* mRoadsDevice;
  std::array<int*, nLayers - 2> mCellsLookupTablesDevice;
  std::array<int*, nLayers - 3> mNeighboursCellDevice;
  std::array<int*, nLayers - 3> mNeighboursCellLookupTablesDevice;
  std::array<int*, nLayers - 2> mRoadsLookupTablesDevice;

  // These are to make them accessible using layer index
  CellSeed** mCellsDeviceArray;
  int** mNeighboursCellDeviceArray;
  int** mNeighboursCellLookupTablesDeviceArray;

  // Small accessory buffers
  int* mCUBTmpBufferDevice;
  int* mFoundTrackletsDevice;
  int* mNFoundCellsDevice;

  /// Vertexer only
  Line* mLinesDevice;
  int* mNFoundLinesDevice;
  int* mNExclusiveFoundLinesDevice;
  unsigned char* mUsedTrackletsDevice;
  std::array<int*, 2> mNTrackletsPerClusterDevice;
  int* mClusteredLinesDevice;

  /// State and configuration
  bool mAllocated = false;
  size_t mNRof = 0;
  size_t mNPopulatedRof = 0;
  o2::its::TimeFrame* mTimeFramePtr = nullptr;
  TimeFrameGPUParameters* mTFGPUParams = nullptr;
};
} // namespace o2::its::gpu
#endif
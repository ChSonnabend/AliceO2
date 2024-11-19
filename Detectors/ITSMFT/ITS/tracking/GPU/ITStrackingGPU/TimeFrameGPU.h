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

#ifndef TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H
#define TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"

#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/Array.h"
#include "ITStrackingGPU/Vector.h"
#include "ITStrackingGPU/Stream.h"
#include "ITStrackingGPU/TimeFrameChunk.h"

#include <gsl/gsl>

namespace o2
{
namespace its
{
namespace gpu
{

class DefaultGPUAllocator : public ExternalAllocator
{
  void* allocate(size_t size) override;
};

template <int nLayers = 7>
class TimeFrameGPU : public TimeFrame
{
  friend class GpuTimeFrameChunk<nLayers>;

 public:
  TimeFrameGPU();
  ~TimeFrameGPU();

  /// Most relevant operations
  void registerHostMemory(const int);
  void unregisterHostMemory(const int);
  void initialise(const int, const TrackingParameters&, const int, IndexTableUtils* utils = nullptr, const TimeFrameGPUParameters* pars = nullptr);
  void initDevice(IndexTableUtils*, const TrackingParameters& trkParam, const TimeFrameGPUParameters&, const int, const int);
  void initDeviceSAFitting();
  void loadTrackingFrameInfoDevice(const int);
  void loadUnsortedClustersDevice(const int);
  void loadClustersDevice(const int);
  void loadTrackletsDevice();
  void loadTrackletsLUTDevice();
  void loadCellsDevice();
  void loadCellsLUTDevice();
  void loadTrackSeedsDevice();
  void loadTrackSeedsChi2Device();
  void loadRoadsDevice();
  void loadTrackSeedsDevice(std::vector<CellSeed>&);
  void createCellsBuffers(const int);
  void createCellsDevice();
  void createCellsLUTDevice();
  void createNeighboursDevice();
  void createNeighboursDevice(const unsigned int& layer, std::vector<std::pair<int, int>>& neighbours);
  void createNeighboursLUTDevice(const int, const unsigned int);
  void createTrackITSExtDevice(std::vector<CellSeed>&);
  void downloadTrackITSExtDevice(std::vector<CellSeed>&);
  void downloadCellsNeighboursDevice(std::vector<std::vector<std::pair<int, int>>>&, const int);
  void downloadNeighboursLUTDevice(std::vector<int>&, const int);
  void downloadCellsDevice();
  void downloadCellsLUTDevice();
  void unregisterRest();
  void initDeviceChunks(const int, const int);
  template <Task task>
  size_t loadChunkData(const size_t, const size_t, const size_t);
  size_t getNChunks() const { return mMemChunks.size(); }
  GpuTimeFrameChunk<nLayers>& getChunk(const int chunk) { return mMemChunks[chunk]; }
  Stream& getStream(const size_t stream) { return mGpuStreams[stream]; }
  void wipe(const int);

  /// interface
  int getNClustersInRofSpan(const int, const int, const int) const;
  IndexTableUtils* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }
  int* getDeviceROFramesClusters(const int layer) { return mROFramesClustersDevice[layer]; }
  std::vector<std::vector<Vertex>>& getVerticesInChunks() { return mVerticesInChunks; }
  std::vector<std::vector<int>>& getNVerticesInChunks() { return mNVerticesInChunks; }
  std::vector<o2::its::TrackITSExt>& getTrackITSExt() { return mTrackITSExt; }
  std::vector<std::vector<o2::MCCompLabel>>& getLabelsInChunks() { return mLabelsInChunks; }
  int getNAllocatedROFs() const { return mNrof; } // Allocated means maximum nROF for each chunk while populated is the number of loaded ones.
  StaticTrackingParameters<nLayers>* getDeviceTrackingParameters() { return mTrackingParamsDevice; }
  Vertex* getDeviceVertices() { return mVerticesDevice; }
  int* getDeviceROFramesPV() { return mROFramesPVDevice; }
  unsigned char* getDeviceUsedClusters(const int);
  const o2::base::Propagator* getChainPropagator();

  // Hybrid
  Road<nLayers - 2>* getDeviceRoads() { return mRoadsDevice; }
  TrackITSExt* getDeviceTrackITSExt() { return mTrackITSExtDevice; }
  int* getDeviceNeighboursLUT(const int layer) { return mNeighboursLUTDevice[layer]; }
  gpuPair<int, int>* getDeviceNeighbours(const int layer) { return mNeighboursDevice[layer]; }
  TrackingFrameInfo* getDeviceTrackingFrameInfo(const int);
  const TrackingFrameInfo** getDeviceArrayTrackingFrameInfo() const { return mTrackingFrameInfoDeviceArray; }
  const Cluster** getDeviceArrayClusters() const { return mClustersDeviceArray; }
  const Cluster** getDeviceArrayUnsortedClusters() const { return mUnsortedClustersDeviceArray; }
  const Tracklet** getDeviceArrayTracklets() const { return mTrackletsDeviceArray; }
  const int** getDeviceArrayTrackletsLUT() const { return mTrackletsLUTDeviceArray; }
  int** getDeviceArrayCellsLUT() const { return mCellsLUTDeviceArray; }
  int** getDeviceArrayNeighboursCellLUT() const { return mNeighboursCellLUTDeviceArray; }
  CellSeed** getDeviceArrayCells() const { return mCellsDeviceArray; }
  CellSeed* getDeviceTrackSeeds() { return mTrackSeedsDevice; }
  o2::track::TrackParCovF** getDeviceArrayTrackSeeds() { return mCellSeedsDeviceArray; }
  float** getDeviceArrayTrackSeedsChi2() { return mCellSeedsChi2DeviceArray; }
  int* getDeviceNeighboursIndexTables(const int layer) { return mNeighboursIndexTablesDevice[layer]; }

  void setDevicePropagator(const o2::base::PropagatorImpl<float>*) override;

  // Host-specific getters
  gsl::span<int> getHostNTracklets(const int chunkId);
  gsl::span<int> getHostNCells(const int chunkId);

  // Host-available device getters
  gsl::span<int*> getDeviceCellLUTs() { return mCellsLUTDevice; }
  gsl::span<CellSeed*> getDeviceCells() { return mCellsDevice; }
  gsl::span<int, nLayers - 2> getNCellsDevice() { return mNCells; }

 private:
  void allocMemAsync(void**, size_t, Stream*, bool); // Abstract owned and unowned memory allocations
  bool mHostRegistered = false;
  std::vector<GpuTimeFrameChunk<nLayers>> mMemChunks;
  TimeFrameGPUParameters mGpuParams;
  StaticTrackingParameters<nLayers> mStaticTrackingParams;

  // Host-available device buffer sizes
  std::array<int, nLayers - 2> mNCells;

  // Device pointers
  StaticTrackingParameters<nLayers>* mTrackingParamsDevice;
  IndexTableUtils* mIndexTableUtilsDevice;
  std::array<int*, nLayers> mROFramesClustersDevice;
  std::array<unsigned char*, nLayers> mUsedClustersDevice;
  Vertex* mVerticesDevice;
  int* mROFramesPVDevice;

  // Hybrid pref
  std::array<Cluster*, nLayers> mClustersDevice;
  std::array<Cluster*, nLayers> mUnsortedClustersDevice;
  const Cluster** mClustersDeviceArray;
  const Cluster** mUnsortedClustersDeviceArray;
  std::array<Tracklet*, nLayers - 1> mTrackletsDevice;
  const Tracklet** mTrackletsDeviceArray;
  const int** mTrackletsLUTDeviceArray;
  std::array<int*, nLayers - 2> mTrackletsLUTDevice;
  std::array<int*, nLayers - 2> mCellsLUTDevice;
  std::array<int*, nLayers - 3> mNeighboursLUTDevice;

  int** mCellsLUTDeviceArray;
  int** mNeighboursCellDeviceArray;
  int** mNeighboursCellLUTDeviceArray;
  std::array<CellSeed*, nLayers - 2> mCellsDevice;
  std::array<int*, nLayers - 2> mNeighboursIndexTablesDevice;
  CellSeed* mTrackSeedsDevice;
  CellSeed** mCellsDeviceArray;
  std::array<o2::track::TrackParCovF*, nLayers - 2> mCellSeedsDevice;
  o2::track::TrackParCovF** mCellSeedsDeviceArray;
  std::array<float*, nLayers - 2> mCellSeedsChi2Device;
  float** mCellSeedsChi2DeviceArray;

  Road<nLayers - 2>* mRoadsDevice;
  TrackITSExt* mTrackITSExtDevice;
  std::array<gpuPair<int, int>*, nLayers - 2> mNeighboursDevice;
  std::array<TrackingFrameInfo*, nLayers> mTrackingFrameInfoDevice;
  const TrackingFrameInfo** mTrackingFrameInfoDeviceArray;

  // State
  std::vector<Stream> mGpuStreams;
  size_t mAvailMemGB;
  bool mFirstInit = true;

  // Output
  std::vector<std::vector<Vertex>> mVerticesInChunks;
  std::vector<std::vector<int>> mNVerticesInChunks;
  std::vector<std::vector<o2::MCCompLabel>> mLabelsInChunks;

  // Host memory used only in GPU tracking
  std::vector<int> mHostNTracklets;
  std::vector<int> mHostNCells;

  // Temporary buffer for storing output tracks from GPU tracking
  std::vector<TrackITSExt> mTrackITSExt;
};

template <int nLayers>
template <Task task>
size_t TimeFrameGPU<nLayers>::loadChunkData(const size_t chunk, const size_t offset, const size_t maxRofs) // offset: readout frame to start from, maxRofs: to manage boundaries
{
  size_t nRof{0};

  mMemChunks[chunk].reset(task, mGpuStreams[chunk]); // Reset chunks memory
  if constexpr ((bool)task) {
    nRof = mMemChunks[chunk].loadDataOnDevice(offset, maxRofs, 3, mGpuStreams[chunk]);
  } else {
    nRof = mMemChunks[chunk].loadDataOnDevice(offset, maxRofs, nLayers, mGpuStreams[chunk]);
  }
  LOGP(debug, "In chunk {}: loaded {} readout frames starting from {}", chunk, nRof, offset);
  return nRof;
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNClustersInRofSpan(const int rofIdstart, const int rofSpanSize, const int layerId) const
{
  return static_cast<int>(mROFramesClusters[layerId][(rofIdstart + rofSpanSize) < mROFramesClusters.size() ? rofIdstart + rofSpanSize : mROFramesClusters.size() - 1] - mROFramesClusters[layerId][rofIdstart]);
}
} // namespace gpu
} // namespace its
} // namespace o2
#endif

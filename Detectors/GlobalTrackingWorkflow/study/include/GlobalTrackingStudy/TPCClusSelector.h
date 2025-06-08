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

// helper class for TPC clusters selection

#ifndef ALICEO2_TPCCLUSSELECTOR_H
#define ALICEO2_TPCCLUSSELECTOR_H

#include <vector>
#include <array>
#include <Rtypes.h>

namespace o2::tpc
{
class ClusterNativeAccess;

class TPCClusSelector
{
  // helper to select TPC cluster matching to certain timebin and optionally pads range
  // example of usage:
  /*
    TPCClusSelector clSel;
    o2::tpc::ClusterNativeHelper::Reader tcpClusterReader;
    tcpClusterReader.init(native_clusters_file.c_str());
    o2::tpc::ClusterNativeAccess tpcClusterIdxStruct;
    std::unique_ptr<o2::tpc::ClusterNative[]> tpcClusterBuffer;       ///< buffer for clusters in tpcClusterIdxStruct
    o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer tpcClusterMCBuffer;  ///< buffer for mc labels

    tcpClusterReader.read(iTF);
    tcpClusterReader.fillIndex(tpcClusterIdxStruct, tpcClusterBuffer, tpcClusterMCBuffer);

    clSel.fill(tpcClusterIdxStruct); // Create sorted index
    // to get i-th cluster in orderer timebins:
    const auto& clus = tpcClusterIdxStruct.clusters[sector][row][  clSel.getIndex(sector, row, i)];

    // to get sorted indices range of clusters in the tbmin:tbmax range
    auto rng = clSel.findClustersRange(sector, row, tbmin, tbmax, tpcClusterIdxStruct);
    if (rng.first>rng.second) { // nothing is found }
    const auto& cln = tpcClusterIdxStruct.clusters[sector][row][clSel.getIndex(sector, row, rng.first )]; /...

    // to get number of clusters in tbmin:tbmax, padmin:padmax range (and optionally get the list)
    std::vector<int> cllist; // optional list
    int nfnd = clSel.findClustersEntries(sector, row, tbmin, tbmax, padmin, padmax, tpcClusterIdxStruct, &cllist);
    for (int i=0;i<nfnd;i++) {
      const auto& cln = tpcClusterIdxStruct.clusters[sector][row][cllist[i]]; /...  direct indices!
    }
   */

 public:
  void clear()
  {
    for (auto& s : mSectors)
      s.clear();
  }
  size_t getIndex(int sec, int row, uint32_t icl) const { return mSectors[sec].rows[row][icl]; }

  std::pair<int, int> findClustersRange(int sec, int row, float tbmin, float tbmax, const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct);
  int findClustersEntries(int sec, int row, float tbmin, float tbmax, float padmin, float padmax, const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct, std::vector<int>* clIDDirect = nullptr);
  void fill(const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct);

  int getNThreads() const { return mNThreads; }
  void setNThreads(int n);

 private:
  struct Sector {
    static constexpr int NRows = 152;
    std::array<std::vector<uint16_t>, NRows> rows;
    void clear()
    {
      for (auto& r : rows)
        r.clear();
    }
  };

  static constexpr int NSectors = 36;
  std::array<Sector, NSectors> mSectors{};
  int mNThreads = 1;

  ClassDefNV(TPCClusSelector, 1);
};

} // namespace o2::tpc

#endif

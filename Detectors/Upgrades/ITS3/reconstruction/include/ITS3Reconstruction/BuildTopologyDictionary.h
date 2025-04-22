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

/// \file TopologyDictionary.h
/// \brief Definition of the BuildTopologyDictionary class for ITS3

#ifndef ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H

#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

namespace o2::its3
{

class BuildTopologyDictionary
{
  using TopoInfo = std::unordered_map<long unsigned, itsmft::TopologyInfo>;
  using TopoStat = std::map<long unsigned, itsmft::TopoStat>;
  using TopoFreq = std::vector<std::pair<unsigned long, unsigned long>>;

 public:
  static constexpr float IgnoreVal = 999.;
  void accountTopology(const itsmft::ClusterTopology& cluster, bool IB, float dX = IgnoreVal, float dZ = IgnoreVal);
  void setNCommon(unsigned int nCommon, bool IB); // set number of common topologies
  void setThreshold(double thr, bool IB);
  void setThresholdCumulative(double cumulative, bool IB); // Considering the integral
  void groupRareTopologies();
  void printDictionary(const std::string& fname);
  void printDictionaryBinary(const std::string& fname);
  void saveDictionaryRoot(const std::string& fname);

  [[nodiscard]] unsigned int getTotClusters(bool IB) const { return (IB) ? mTotClustersIB : mTotClustersOB; }
  [[nodiscard]] unsigned int getNotInGroups(bool IB) const { return (IB) ? mNCommonTopologiesIB : mNCommonTopologiesOB; }
  [[nodiscard]] const TopologyDictionary& getDictionary() const { return mDictionary; }

  friend std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& BD);

 private:
  void accountTopologyImpl(const itsmft::ClusterTopology& cluster, TopoInfo& tinfo, TopoStat& tstat, unsigned int& ntot, float sigmaX, float sigmaZ, float dX, float dZ);
  void setNCommonImpl(unsigned int ncom, TopoFreq& tfreq, TopoStat& tstat, unsigned int& ncommon, unsigned int ntot);
  void setThresholdImpl(double thr, TopoFreq& tfreq, TopoInfo& tinfo, TopoStat& tstat, unsigned int& ncommon, double& freqthres, unsigned int ntot);
  void setThresholdCumulativeImpl(double cumulative, TopoFreq& tfreq, unsigned int& ncommon, double& freqthres, unsigned int ntot);
  void groupRareTopologiesImpl(TopoFreq& tfreq, TopoInfo& tinfo, TopoStat& tstat, unsigned int& ncommon, double& freqthres, TopologyDictionaryData& data, unsigned int ntot);

  TopologyDictionary mDictionary; ///< Dictionary of topologies
  unsigned int mTotClustersIB{0};
  unsigned int mTotClustersOB{0};
  unsigned int mNCommonTopologiesIB{0};
  unsigned int mNCommonTopologiesOB{0};
  double mFrequencyThresholdIB{0.};
  double mFrequencyThresholdOB{0.};
  TopoInfo mMapInfoIB;
  TopoInfo mMapInfoOB;
  TopoStat mTopologyMapIB;       //! IB Temporary map of type <hash, TopStat>
  TopoStat mTopologyMapOB;       //! OB Temporary map of type <hash, TopStat>
  TopoFreq mTopologyFrequencyIB; //! IB <freq,hash>, needed to define threshold
  TopoFreq mTopologyFrequencyOB; //! OB <freq,hash>, needed to define threshold

  ClassDefNV(BuildTopologyDictionary, 3);
};
} // namespace o2::its3

#endif

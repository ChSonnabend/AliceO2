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

/// \file TopologyDictionary.cxx

#include "ITS3Reconstruction/BuildTopologyDictionary.h"
#include "ITS3Reconstruction/LookUp.h"
#include "DataFormatsITSMFT/CompCluster.h"

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITS3Base/SegmentationMosaix.h"

#include "TFile.h"

ClassImp(o2::its3::BuildTopologyDictionary);

namespace o2::its3
{
void BuildTopologyDictionary::accountTopology(const itsmft::ClusterTopology& cluster, bool IB, float dX, float dZ)
{
  accountTopologyImpl(cluster,
                      ((IB) ? mMapInfoIB : mMapInfoOB),
                      ((IB) ? mTopologyMapIB : mTopologyMapOB),
                      ((IB) ? mTotClustersIB : mTotClustersOB),
                      ((IB) ? SegmentationMosaix::PitchRow : itsmft::SegmentationAlpide::PitchRow),
                      ((IB) ? SegmentationMosaix::PitchCol : itsmft::SegmentationAlpide::PitchCol),
                      dX, dZ);
}

void BuildTopologyDictionary::accountTopologyImpl(const itsmft::ClusterTopology& cluster, TopoInfo& tinfo, TopoStat& tstat, unsigned int& tot, float sigmaX, float sigmaZ, float dX, float dZ)
{
  ++tot;
  bool useDf = dX < IgnoreVal / 2; // we may need to account the frequency but to not update the centroid

  auto& topoStat = tstat[cluster.getHash()];
  topoStat.countsTotal++;
  if (topoStat.countsTotal == 1) { // a new topology is inserted
    topoStat.topology = cluster;
    //___________________DEFINING_TOPOLOGY_CHARACTERISTICS__________________
    itsmft::TopologyInfo topInf;
    topInf.mPattern.setPattern(cluster.getPattern().data());
    topInf.mSizeX = cluster.getRowSpan();
    topInf.mSizeZ = cluster.getColumnSpan();
    //__________________COG_Determination_____________
    topInf.mNpixels = cluster.getClusterPattern().getCOG(topInf.mCOGx, topInf.mCOGz);
    if (useDf) {
      topInf.mXmean = dX;
      topInf.mZmean = dZ;
      topoStat.countsWithBias = 1;
    } else { // assign expected sigmas from the pixel X, Z sizes
      topInf.mXsigma2 = sigmaX * sigmaX / 12.f / (float)std::min(10, topInf.mSizeX);
      topInf.mZsigma2 = sigmaZ * sigmaZ / 12.f / (float)std::min(10, topInf.mSizeZ);
    }
    tinfo.emplace(cluster.getHash(), topInf);
  } else {
    if (useDf) {
      auto num = topoStat.countsWithBias++;
      auto ind = tinfo.find(cluster.getHash());
      float tmpxMean = ind->second.mXmean;
      float newxMean = ind->second.mXmean = ((tmpxMean)*num + dX) / (num + 1);
      float tmpxSigma2 = ind->second.mXsigma2;
      ind->second.mXsigma2 = (num * tmpxSigma2 + (dX - tmpxMean) * (dX - newxMean)) / (num + 1); // online variance algorithm
      float tmpzMean = ind->second.mZmean;
      float newzMean = ind->second.mZmean = ((tmpzMean)*num + dZ) / (num + 1);
      float tmpzSigma2 = ind->second.mZsigma2;
      ind->second.mZsigma2 = (num * tmpzSigma2 + (dZ - tmpzMean) * (dZ - newzMean)) / (num + 1); // online variance algorithm
    }
  }
}

void BuildTopologyDictionary::setNCommon(unsigned int nCommon, bool IB)
{
  mDictionary.resetMaps(IB);

  auto& freqTopo = ((IB) ? mTopologyFrequencyIB : mTopologyFrequencyOB);
  auto& freqThres = ((IB) ? mFrequencyThresholdIB : mFrequencyThresholdOB);
  auto& comTopo = ((IB) ? mNCommonTopologiesIB : mNCommonTopologiesOB);
  auto ntot = ((IB) ? mTotClustersIB : mTotClustersOB);

  setNCommonImpl(nCommon,
                 freqTopo,
                 ((IB) ? mTopologyMapIB : mTopologyMapOB),
                 comTopo,
                 ntot);
  // Recaculate also the threshold
  freqThres = ((double)freqTopo[comTopo - 1].first) / ntot;
}

void BuildTopologyDictionary::setNCommonImpl(unsigned int ncom, TopoFreq& tfreq, TopoStat& tstat, unsigned int& ncommon, unsigned int ntot)
{
  if (ncom >= itsmft::CompCluster::InvalidPatternID) {
    LOGP(warning, "Redefining nCommon from {} to {} to be below InvalidPatternID", ncom, itsmft::CompCluster::InvalidPatternID - 1);
    ncom = itsmft::CompCluster::InvalidPatternID - 1;
  }
  tfreq.clear();
  for (auto&& p : tstat) { // p os pair<ulong,TopoStat>
    tfreq.emplace_back(p.second.countsTotal, p.first);
  }
  std::sort(tfreq.begin(), tfreq.end(),
            [](const std::pair<unsigned long, unsigned long>& couple1,
               const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
  ncommon = ncom;
}

void BuildTopologyDictionary::setThreshold(double thr, bool IB)
{
  mDictionary.resetMaps(IB);
  setThresholdImpl(thr,
                   ((IB) ? mTopologyFrequencyIB : mTopologyFrequencyOB),
                   ((IB) ? mMapInfoIB : mMapInfoOB),
                   ((IB) ? mTopologyMapIB : mTopologyMapOB),
                   ((IB) ? mNCommonTopologiesIB : mNCommonTopologiesOB),
                   ((IB) ? mFrequencyThresholdIB : mFrequencyThresholdOB),
                   ((IB) ? mTotClustersIB : mTotClustersOB));
}

void BuildTopologyDictionary::setThresholdImpl(double thr, TopoFreq& tfreq, TopoInfo& tinfo, TopoStat& tstat, unsigned int& ncommon, double& freqthres, unsigned int ntot)
{
  setNCommonImpl(0, tfreq, tstat, ncommon, ntot);
  freqthres = thr;
  for (const auto& q : tfreq) {
    if (((double)q.first) / ntot > thr) {
      ++ncommon;
    } else {
      break;
    }
  }
  if (ncommon >= itsmft::CompCluster::InvalidPatternID) {
    freqthres = ((double)tfreq[itsmft::CompCluster::InvalidPatternID - 1].first) / ntot;
    LOGP(warning, "Redefining prob. threshold from {} to {} to be below InvalidPatternID (was {})", thr, freqthres, ntot);
    ncommon = itsmft::CompCluster::InvalidPatternID - 1;
  }
}

void BuildTopologyDictionary::setThresholdCumulative(double cumulative, bool IB)
{
  if (cumulative <= 0. || cumulative >= 1.) {
    cumulative = 0.99;
  }

  auto& freqTopo = ((IB) ? mTopologyFrequencyIB : mTopologyFrequencyOB);
  auto& freqThres = ((IB) ? mFrequencyThresholdIB : mFrequencyThresholdOB);
  auto& statTopo = ((IB) ? mTopologyMapIB : mTopologyMapOB);
  auto& comTopo = ((IB) ? mNCommonTopologiesIB : mNCommonTopologiesOB);
  auto ntot = ((IB) ? mTotClustersIB : mTotClustersOB);

  mDictionary.resetMaps(IB);
  setNCommonImpl(0, freqTopo, statTopo, comTopo, ntot);
  setThresholdCumulativeImpl(cumulative, freqTopo, comTopo, freqThres, ntot);
}

void BuildTopologyDictionary::setThresholdCumulativeImpl(double cumulative, TopoFreq& tfreq, unsigned int& ncommon, double& freqthres, unsigned int ntot)
{
  double totFreq = 0.;
  for (auto& q : tfreq) {
    totFreq += ((double)(q.first)) / ntot;
    if (totFreq < cumulative) {
      ++ncommon;
      if (ncommon >= itsmft::CompCluster::InvalidPatternID) {
        totFreq -= ((double)(q.first)) / ntot;
        --ncommon;
        LOGP(warning, "Redefining cumulative threshould from {} to {} to be below InvalidPatternID)", cumulative, totFreq);
      }
    } else {
      break;
    }
  }
  freqthres = ((double)(tfreq[--ncommon].first)) / ntot;
  while (std::fabs(((double)tfreq[ncommon--].first) / ntot - freqthres) < 1.e-15) {
  }
  freqthres = ((double)tfreq[ncommon++].first) / ntot;
}

void BuildTopologyDictionary::groupRareTopologies()
{
  LOG(info) << "Dictionary finalisation";
  LOG(info) << "Number of IB clusters: " << mTotClustersIB;
  LOG(info) << "Number of OB clusters: " << mTotClustersOB;

  groupRareTopologiesImpl(mTopologyFrequencyIB, mMapInfoIB, mTopologyMapIB, mNCommonTopologiesIB, mFrequencyThresholdIB, mDictionary.mDataIB, mNCommonTopologiesIB);
  groupRareTopologiesImpl(mTopologyFrequencyOB, mMapInfoOB, mTopologyMapOB, mNCommonTopologiesOB, mFrequencyThresholdOB, mDictionary.mDataOB, mNCommonTopologiesOB);

  LOG(info) << "Dictionay finalised";
  LOG(info) << "IB:";
  mDictionary.mDataIB.print();
  LOG(info) << "OB:";
  mDictionary.mDataOB.print();
}

void BuildTopologyDictionary::groupRareTopologiesImpl(TopoFreq& tfreq, TopoInfo& tinfo, TopoStat& tstat, unsigned int& ncommon, double& freqthres, TopologyDictionaryData& data, unsigned int ntot)
{
  double totFreq = 0.;
  for (unsigned int j = 0; j < ncommon; j++) {
    itsmft::GroupStruct gr;
    gr.mHash = tfreq[j].second;
    gr.mFrequency = ((double)(tfreq[j].first)) / ntot;
    totFreq += gr.mFrequency;
    // rough estimation for the error considering a uniform distribution
    const auto& topo = tinfo.find(gr.mHash)->second;
    gr.mErrX = std::sqrt(topo.mXsigma2);
    gr.mErrZ = std::sqrt(topo.mZsigma2);
    gr.mErr2X = topo.mXsigma2;
    gr.mErr2Z = topo.mZsigma2;
    gr.mXCOG = -1 * topo.mCOGx;
    gr.mZCOG = topo.mCOGz;
    gr.mNpixels = topo.mNpixels;
    gr.mPattern = topo.mPattern;
    gr.mIsGroup = false;
    data.mVectorOfIDs.push_back(gr);
    if (j == int(itsmft::CompCluster::InvalidPatternID - 1)) {
      LOGP(warning, "Limiting N unique topologies to {}, threshold freq. to {}, cumulative freq. to {} to be below InvalidPatternID", j, gr.mFrequency, totFreq);
      ncommon = j;
      freqthres = gr.mFrequency;
      break;
    }
  }
  // groupRareTopologies based on binning over number of rows and columns (TopologyDictionary::NumberOfRowClasses *
  // NumberOfColClasses)

  std::unordered_map<int, std::pair<itsmft::GroupStruct, unsigned long>> tmp_GroupMap; //<group ID, <Group struct, counts>>

  int grNum = 0;
  int rowBinEdge = 0;
  int colBinEdge = 0;
  for (int iRowClass = 0; iRowClass < its3::TopologyDictionary::MaxNumberOfRowClasses; iRowClass++) {
    for (int iColClass = 0; iColClass < its3::TopologyDictionary::MaxNumberOfColClasses; iColClass++) {
      rowBinEdge = (iRowClass + 1) * its3::TopologyDictionary::RowClassSpan;
      colBinEdge = (iColClass + 1) * its3::TopologyDictionary::ColClassSpan;
      grNum = its3::LookUp::groupFinder(rowBinEdge, colBinEdge);
      // Create a structure for a group of rare topologies
      itsmft::GroupStruct gr;
      gr.mHash = (((unsigned long)(grNum)) << 32) & 0xffffffff00000000;
      gr.mErrX = its3::TopologyDictionary::RowClassSpan / std::sqrt(12.f * (float)std::min(10, rowBinEdge));
      gr.mErrZ = its3::TopologyDictionary::ColClassSpan / std::sqrt(12.f * (float)std::min(10, colBinEdge));
      gr.mErr2X = gr.mErrX * gr.mErrX;
      gr.mErr2Z = gr.mErrZ * gr.mErrZ;
      gr.mXCOG = 0;
      gr.mZCOG = 0;
      gr.mNpixels = rowBinEdge * colBinEdge;
      gr.mIsGroup = true;
      gr.mFrequency = 0.;
      /// A dummy pattern with all fired pixels in the bounding box is assigned to groups of rare topologies.
      unsigned char dummyPattern[itsmft::ClusterPattern::kExtendedPatternBytes] = {0};
      dummyPattern[0] = (unsigned char)rowBinEdge;
      dummyPattern[1] = (unsigned char)colBinEdge;
      int nBits = rowBinEdge * colBinEdge;
      int nBytes = nBits / 8;
      for (int iB = 2; iB < nBytes + 2; iB++) {
        dummyPattern[iB] = (unsigned char)255;
      }
      int residualBits = nBits % 8;
      if (residualBits != 0) {
        unsigned char tempChar = 0;
        while (residualBits > 0) {
          residualBits--;
          tempChar |= 1 << (7 - residualBits);
        }
        dummyPattern[nBytes + 2] = tempChar;
      }
      gr.mPattern.setPattern(dummyPattern);
      // Filling the map for groups
      tmp_GroupMap[grNum] = std::make_pair(gr, 0);
    }
  }
  int rs{}, cs{}, index{};

  // Updating the counts for the groups of rare topologies
  for (auto j{ncommon}; j < tfreq.size(); j++) {
    unsigned long hash1 = tfreq[j].second;
    rs = tstat.find(hash1)->second.topology.getRowSpan();
    cs = tstat.find(hash1)->second.topology.getColumnSpan();
    index = its3::LookUp::groupFinder(rs, cs);
    tmp_GroupMap[index].second += tfreq[j].first;
  }

  for (auto&& p : tmp_GroupMap) {
    itsmft::GroupStruct& group = p.second.first;
    group.mFrequency = ((double)p.second.second) / ntot;
    data.mVectorOfIDs.push_back(group);
  }

  // Sorting the dictionary preserving all unique topologies
  std::sort(data.mVectorOfIDs.begin(), data.mVectorOfIDs.end(), [](const itsmft::GroupStruct& a, const itsmft::GroupStruct& b) {
    return (!a.mIsGroup) && b.mIsGroup ? true : (a.mIsGroup && (!b.mIsGroup) ? false : (a.mFrequency > b.mFrequency));
  });
  if (data.mVectorOfIDs.size() >= itsmft::CompCluster::InvalidPatternID - 1) {
    LOGP(warning, "Max allowed {} patterns is reached, stopping", itsmft::CompCluster::InvalidPatternID - 1);
    data.mVectorOfIDs.resize(itsmft::CompCluster::InvalidPatternID - 1);
  }
  // Sorting the dictionary to final form
  std::sort(data.mVectorOfIDs.begin(), data.mVectorOfIDs.end(), [](const itsmft::GroupStruct& a, const itsmft::GroupStruct& b) { return a.mFrequency > b.mFrequency; });
  // Creating the map for common topologies
  for (int iKey = 0; iKey < data.mVectorOfIDs.size(); iKey++) {
    itsmft::GroupStruct& gr = data.mVectorOfIDs[iKey];
    if (!gr.mIsGroup) {
      data.mCommonMap.emplace(gr.mHash, iKey);
      if (gr.mPattern.getUsedBytes() == 1) {
        data.mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.getByte(2)] = iKey;
      }
    } else {
      data.mGroupMap.emplace((int)(gr.mHash >> 32) & 0x00000000ffffffff, iKey);
    }
  }
}

std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& DB)
{
  os << "--- InnerBarrel\n";
  for (unsigned int i = 0; i < DB.mNCommonTopologiesIB; i++) {
    const unsigned long& hash = DB.mTopologyFrequencyIB[i].second;
    os << "Hash: " << hash << '\n';
    os << "counts: " << DB.mTopologyMapIB.find(hash)->second.countsTotal;
    os << " (with bias provided: " << DB.mTopologyMapIB.find(hash)->second.countsWithBias << ")" << '\n';
    os << "sigmaX: " << std::sqrt(DB.mMapInfoIB.find(hash)->second.mXsigma2) << '\n';
    os << "sigmaZ: " << std::sqrt(DB.mMapInfoIB.find(hash)->second.mZsigma2) << '\n';
    os << DB.mTopologyMapIB.find(hash)->second.topology;
  }
  os << "--- OuterBarrel\n";
  for (unsigned int i = 0; i < DB.mNCommonTopologiesOB; i++) {
    const unsigned long& hash = DB.mTopologyFrequencyOB[i].second;
    os << "Hash: " << hash << '\n';
    os << "counts: " << DB.mTopologyMapOB.find(hash)->second.countsTotal;
    os << " (with bias provided: " << DB.mTopologyMapOB.find(hash)->second.countsWithBias << ")" << '\n';
    os << "sigmaX: " << std::sqrt(DB.mMapInfoOB.find(hash)->second.mXsigma2) << '\n';
    os << "sigmaZ: " << std::sqrt(DB.mMapInfoOB.find(hash)->second.mZsigma2) << '\n';
    os << DB.mTopologyMapOB.find(hash)->second.topology;
  }
  return os;
}

void BuildTopologyDictionary::printDictionary(const std::string& fname)
{
  LOG(info) << "Saving the the dictionary in binary format: ";
  std::ofstream out(fname);
  out << mDictionary;
  out.close();
  LOG(info) << " `-> done!";
}

void BuildTopologyDictionary::printDictionaryBinary(const std::string& fname)
{
  LOG(info) << "Printing the dictionary: ";
  std::ofstream out(fname);
  mDictionary.writeBinaryFile(fname);
  out.close();
  LOG(info) << " `-> done!";
}

void BuildTopologyDictionary::saveDictionaryRoot(const std::string& fname)
{
  LOG(info) << "Saving the the dictionary in a ROOT file: " << fname;
  TFile output(fname.c_str(), "recreate");
  output.WriteObjectAny(&mDictionary, mDictionary.Class(), "ccdb_object");
  output.Close();
  LOG(info) << " `-> done!";
}

} // namespace o2::its3

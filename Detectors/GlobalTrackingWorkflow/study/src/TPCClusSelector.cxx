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
#include "GlobalTrackingStudy/TPCClusSelector.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "Framework/Logger.h"
#include <numeric>
#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::tpc;

void TPCClusSelector::setNThreads(int n)
{
#ifndef WITH_OPENMP
  if (n > 1) {
    LOGP(warn, "No OpenMP");
  }
  n = 1;
#endif
  mNThreads = n;
}

std::pair<int, int> TPCClusSelector::findClustersRange(int sec, int row, float tbmin, float tbmax, const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct)
{
  // find sorted indices of clusters in the [tbmin:tbmax] range, if not found, return {-1,-2}
  const auto& vidx = mSectors[sec].rows[row];
  const auto* clarr = tpcClusterIdxStruct.clusters[sec][row];
  // use binary search to find 1st cluster with time >= tb
  int ncl = vidx.size(), left = 0, right = ncl;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (clarr[vidx[mid]].getTime() < tbmin) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (left == ncl || clarr[vidx[left]].getTime() > tbmax) {
    return {-1, -2}; // all clusters have time < tbmin or no clusters in the range [tbmin:tbmax]
  }
  int idmin = left, idmax = left, idtst = idmin;
  // look at smaller times
  while (++idtst < ncl && clarr[vidx[idtst]].getTime() <= tbmax) {
    idmax = idtst;
  }
  return {idmin, idmax};
}

int TPCClusSelector::findClustersEntries(int sec, int row, float tbmin, float tbmax, float padmin, float padmax, const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct, std::vector<int>* clIDDirect)
{
  // find direct cluster indices for tbmin:tbmas / padmin/padmax range, fill clIDDirect vector if provided
  const auto& vidx = mSectors[sec].rows[row];
  const auto* clarr = tpcClusterIdxStruct.clusters[sec][row];
  // use binary search to find 1st cluster with time >= tb
  int ncl = vidx.size(), left = 0, right = ncl;
  if (clIDDirect) {
    clIDDirect->clear();
  }
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (clarr[vidx[mid]].getTime() < tbmin) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (left == ncl || clarr[vidx[left]].getTime() > tbmax) {
    return 0; // all clusters have time < tbmin or no clusters in the range [tbmin:tbmax]
  }
  int nclf = 0;
  while (left < ncl) {
    const auto& cl = clarr[vidx[left]];
    if (cl.getTime() > tbmax) {
      break;
    }
    if (cl.getPad() >= padmin && cl.getPad() <= padmax) {
      nclf++;
      if (clIDDirect) {
        clIDDirect->push_back(vidx[left]);
      }
    }
  }
  return nclf;
}

void TPCClusSelector::fill(const o2::tpc::ClusterNativeAccess& tpcClusterIdxStruct)
{
  for (int is = 0; is < NSectors; is++) {
    auto& sect = mSectors[is];
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
    for (int ir = 0; ir < Sector::NRows; ir++) {
      size_t ncl = tpcClusterIdxStruct.nClusters[is][ir];
      if (ncl >= 0xffff) {
        LOGP(error, "Row {} of sector {} has {} clusters, truncating to {}", ir, is, ncl, int(0xffff));
        ncl = 0xffff;
      }
      auto& rowidx = sect.rows[ir];
      rowidx.resize(ncl);
      std::iota(rowidx.begin(), rowidx.end(), 0);
      const auto* clus = tpcClusterIdxStruct.clusters[is][ir]; // C array of clusters
      std::sort(rowidx.begin(), rowidx.end(), [&](size_t a, size_t b) { return clus[a].getTime() < clus[b].getTime(); });
    }
  }
}

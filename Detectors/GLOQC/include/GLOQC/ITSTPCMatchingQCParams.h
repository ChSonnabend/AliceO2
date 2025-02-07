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

/// \author Chiara.Zampolli@cern.ch

#ifndef ALICEO2_ITSTPCMATCHINGQC_PARAMS_H
#define ALICEO2_ITSTPCMATCHINGQC_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::gloqc
{

// There are configurable params for TPC-ITS matching
struct ITSTPCMatchingQCParams : public o2::conf::ConfigurableParamHelper<ITSTPCMatchingQCParams> {

  int nBinsPt = 100;
  float minPtITSCut = 0.1;
  float etaITSCut = 1.4;
  int32_t minNITSClustersCut = 0;
  float maxChi2PerClusterITS = 1e10;
  float minPtTPCCut = 0.1;
  float etaTPCCut = 1.4;
  int32_t minNTPCClustersCut = 60;
  float minDCACut = 100.;
  float minDCACutY = 10.;
  float minPtCut = 0.1;
  float maxPtCut = 20;
  float etaCut = 1.4;
  float etaNo0Cut = 0.05;
  float cutK0Mass = 0.05f;
  float maxEtaK0 = 0.8f;
  float K0Scaling = 1.f;
  float minTPCOccpp = 0.f;
  float maxTPCOccpp = 1.e6;
  int nBinsTPCOccpp = 6;
  float minTPCOccPbPb = 0.f;
  float maxTPCOccPbPb = 8.e6;
  int nBinsTPCOccPbPb = 8;
  float maxK0DCA = 0.01;
  float minK0CosPA = 0.995;

  O2ParamDef(ITSTPCMatchingQCParams, "ITSTPCMatchingQC");
};

} // namespace o2::gloqc
  // end namespace o2

#endif

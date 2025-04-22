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

/// \file   NeuralNetworkClusterizer.h
/// \brief  Fetching neural networks for clusterization from CCDB
/// \author Christian Sonnabend

#ifndef AliceO2_TPC_NeuralNetworkClusterizer_h
#define AliceO2_TPC_NeuralNetworkClusterizer_h

#include "CCDB/CcdbApi.h"

namespace o2::tpc
{

class NeuralNetworkClusterizer
{
 public:
  NeuralNetworkClusterizer() = default;
  void initCcdbApi(std::string url);
  void loadIndividualFromCCDB(std::map<std::string, std::string> settings);

 private:
  o2::ccdb::CcdbApi ccdbApi;
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
};

} // namespace o2::tpc
#endif

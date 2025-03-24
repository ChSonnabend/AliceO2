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

/// \file GPUTPCNNClusterizerHost.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZERHOST_H
#define O2_GPUTPCNNCLUSTERIZERHOST_H

#include <string>
#include <unordered_map>
#include <vector>
#include "ML/OrtInterface.h"

using namespace o2::ml;

namespace o2::OrtDataType
{
struct Float16_t;
}

namespace o2::gpu
{

class GPUTPCNNClusterizer;
struct GPUSettingsProcessingNNclusterizer;

class GPUTPCNNClusterizerHost
{
 public:
  GPUTPCNNClusterizerHost() = default;
  GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer&);
  void init(const GPUSettingsProcessingNNclusterizer&);
  void loadFromCCDB(std::map<std::string, std::string>);

  void initClusterizer(const GPUSettingsProcessingNNclusterizer&, GPUTPCNNClusterizer&);

  void networkInference(o2::ml::OrtModel model, GPUTPCNNClusterizer& clusterer, size_t size, float* output, int32_t dtype);

  void digitWriter(GPUTPCClusterFinder&, std::string);
  void combineDigitFiles(int);

  std::unordered_map<std::string, std::string> OrtOptions;
  o2::ml::OrtModel model_class, model_reg_1, model_reg_2; // For splitting clusters

 private:
  // Avoid including CommonUtils/StringUtils.h
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
  // std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
  // {
  //   std::vector<std::string> tokens;
  //   std::size_t pos = 0;
  //   std::size_t found;
  //   while ((found = input.find(delimiter, pos)) != std::string::npos) {
  //     tokens.push_back(input.substr(pos, found - pos));
  //     pos = found + delimiter.length();
  //   }
  //   tokens.push_back(input.substr(pos));
  //   return tokens;
  // }
}; // class GPUTPCNNClusterizerHost

} // namespace o2::gpu

#endif

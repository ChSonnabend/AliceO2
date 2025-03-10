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

/// \file GPUTPCNNClusterizerInternals.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZERINTERNALS_H
#define O2_GPUTPCNNCLUSTERIZERINTERNALS_H

#include "ML/OrtInterface.h"
#include "ChargePos.h"
#include "GPUReconstruction.h"
#include "GPUProcessor.h"
#include "GPUTPCClusterFinder.h"
#include "GPUHostDataTypes.h"

using namespace o2::ml;

namespace o2::gpu
{

class GPUTPCNNClusterizerInternals : public GPUProcessor
{
 public:
  typedef GPUTPCClusterFinder processorType;
  GPUTPCNNClusterizerInternals() = default;
  GPUTPCNNClusterizerInternals(GPUSettingsProcessing, processorType&);
  void* setIOPointers(void*);
  void RegisterMemoryAllocation();
  void inferenceNetworkClass(processorType&, int8_t, uint);
  void inferenceNetworkReg1(processorType&, int8_t, uint);
  void inferenceNetworkReg2(processorType&, int8_t, uint);

  std::unordered_map<std::string, std::string> OrtOptions;
  o2::ml::OrtModel model_class, model_reg_1, model_reg_2; // For splitting clusters
  std::vector<std::string> reg_model_paths;

 private:
  processorType* clusterer_internal;
  int sector = -1;
  int16_t mMemoryId = -1;

  // Avoid including CommonUtils/StringUtils.h
  std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
  {
    std::vector<std::string> tokens;
    std::size_t pos = 0;
    std::size_t found;

    while ((found = input.find(delimiter, pos)) != std::string::npos) {
      tokens.push_back(input.substr(pos, found - pos));
      pos = found + delimiter.length();
    }
    tokens.push_back(input.substr(pos));

    return tokens;
  }
}; // class GPUTPCNNClusterizerInternals

} // namespace o2::gpu

#endif
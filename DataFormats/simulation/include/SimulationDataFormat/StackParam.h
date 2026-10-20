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

#ifndef ALICEO2_SIMDATAFORMAT_STACKPARAM_H_
#define ALICEO2_SIMDATAFORMAT_STACKPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace sim
{

// configuration parameters for simulation Stack
struct StackParam : public o2::conf::ConfigurableParamHelper<StackParam> {
  bool storeSecondaries = true;
  bool pruneKine = true;
  std::string transportPrimary = "all";
  std::string transportPrimaryFileName = "";
  std::string transportPrimaryFuncName = "";
  bool transportPrimaryInvert = false;
  // simnet.birth.v1: 34 raw birth features. The ONNX graph selects its inputs,
  // embeds preprocessing/domain guards and returns [batch,1] class-1 probability.
  // Class 1 means no recorded hits in the entire subtree. Invalid scores keep tracks.
  std::string transportPrimaryOnnxCCDBPath = "";
  // Local deployment graph, mutually exclusive with the CCDB path.
  std::string transportPrimaryOnnxModelFile = "";
  // Record birth scores in MCTrack and transport every track for unbiased truth.
  // Requires pruneKine=false and storeSecondaries=true. No threshold is needed.
  bool transportPrimaryOnnxShadow = false;
  // Parallel simulation: one event-merged CSV per output prefix. Empty disables.
  // Only active in shadow mode; "auto" gives <prefix>_transport_scores.csv.
  std::string transportPrimaryOnnxCsv = "auto";
  // Export input roots only; subtree truth still uses every transport descendant.
  bool transportPrimaryOnnxCsvRootsOnly = false;
  std::string transportPrimaryOnnxModelLabel = ""; // short model identifier in CSV rows
  // Double preserves validation cuts just above tied float32 scores.
  double transportPrimaryOnnxThreshold = -1.; // require explicit configuration
  int transportPrimaryOnnxOutputIndex = 0;
  bool transportPrimaryOnnxApplySigmoid = false; // probability already in graph
  // Saved roots-v3 models only saw simulation roots, including injected tracks.
  // Enable only for a model trained/validated on transport secondaries too.
  bool transportPrimaryOnnxSecondaries = false;

  // boilerplate stuff + make principal key "Stack"
  O2ParamDef(StackParam, "Stack");
};

} // namespace sim
} // namespace o2

#endif // ALICEO2_SIMDATAFORMAT_STACKPARAM_H_

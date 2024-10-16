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

/// \file GPUO2InterfaceConfiguration.h
/// \author David Rohr

#ifndef GPUO2INTERFACECONFIGURATION_H
#define GPUO2INTERFACECONFIGURATION_H

#ifndef GPUCA_HAVE_O2HEADERS
#define GPUCA_HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <gsl/gsl>
#include "GPUSettings.h"
#include "GPUDataTypes.h"
#include "GPUHostDataTypes.h"
#include "GPUOutputControl.h"
#include "DataFormatsTPC/Constants.h"

class TH1F;
class TH1D;
class TH2F;
class TGraphAsymmErrors;

namespace o2
{
namespace tpc
{
class TrackTPC;
class Digit;
} // namespace tpc
namespace gpu
{
class TPCFastTransform;
class GPUReconstruction;
struct GPUSettingsO2;

struct GPUInterfaceQAOutputs {
  const std::vector<TH1F>* hist1 = nullptr;
  const std::vector<TH2F>* hist2 = nullptr;
  const std::vector<TH1D>* hist3 = nullptr;
  const std::vector<TGraphAsymmErrors>* hist4 = nullptr;
  bool newQAHistsCreated = false;
};

struct GPUInterfaceOutputs : public GPUTrackingOutputs {
  GPUInterfaceQAOutputs qa;
};

struct GPUInterfaceInputUpdate {
  std::function<void(GPUTrackingInOutPointers*& data, GPUInterfaceOutputs*& outputs)> callback; // Callback which provides final data ptrs / outputRegions after Clusterization stage
  std::function<void()> notifyCallback;                                                         // Callback called to notify that Clusterization state has finished without update
};

// Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration {
  GPUO2InterfaceConfiguration() = default;
  ~GPUO2InterfaceConfiguration() = default;
  GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;

  // Settings for the Interface class
  struct GPUInterfaceSettings {
    bool outputToExternalBuffers = false;
    // These constants affect GPU memory allocation only and do not limit the CPU processing
    uint64_t maxTPCZS = 8192ul * 1024 * 1024;
    uint32_t maxTPCHits = 1024 * 1024 * 1024;
    uint32_t maxTRDTracklets = 128 * 1024;
    uint32_t maxITSTracks = 96 * 1024;
  };

  GPUSettingsDeviceBackend configDeviceBackend;
  GPUSettingsProcessing configProcessing;
  GPUSettingsGRP configGRP;
  GPUSettingsRec configReconstruction;
  GPUSettingsDisplay configDisplay;
  GPUSettingsQA configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  GPUCalibObjectsConst configCalib;

  GPUSettingsO2 ReadConfigurableParam();
  static GPUSettingsO2 ReadConfigurableParam(GPUO2InterfaceConfiguration& obj);
  void PrintParam();

 private:
  void PrintParam_internal();
};

} // namespace gpu
} // namespace o2

#endif

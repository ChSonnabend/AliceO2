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

#include "Framework/DeviceSpec.h"
class ImVec2;

namespace o2::framework
{
class DeviceMetricsInfo;
class DeviceInfo;
class DataProcessingStates;

namespace gui
{
static constexpr int WND = 16;
/// View of the DataRelayer metrics for a given DeviceInfo
void displayDataRelayer(DeviceMetricsInfo const& metrics, DeviceInfo const& info, DeviceSpec const& spec, DataProcessingStates const&, ImVec2 const& size, int& v);

} // namespace gui
} // namespace o2::framework

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

#ifndef ALICE_O2_EVENTVISUALISATION_CONFIG_H
#define ALICE_O2_EVENTVISUALISATION_CONFIG_H
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::event_visualisation
{
struct EveConfParam : o2::conf::ConfigurableParamHelper<EveConfParam> {
  float EMCCellTimeMax = 100.f;  // Max. EMCAL cell time (in ns)
  float EMCCellEnergyMin = 0.3f; // Min. EMCAL cell energy (in GeV)

  float timeMinMax[2] = {-999.f, -999.f}; // if min<max (microseconds), show only track in this range of every TF

  int maxFiles = 150;     // maximum number of json files in folder
  int maxTracks = -1;     // maximum number of track stored in json file (-1 means no limit)
  int maxBytes = 3000000; // number of bytes stored in time interval which stops producing new data file (-1 means no limit)

  int maxPVs = -1;                              // if > 0: enable PV mode and limit max number of vertices
  int minTracks = -1;                           // don't create file if less than the specified number of all tracks is present
  int minITSTracks = -1;                        // don't create file if less than the specified number of ITS tracks is present
  int onlyNthEvent = -1;                        // process only every nth event (if > 1)
  float PVXYZMin[3] = {-999.f, -999.f, -999.f}; // primary vertex min x,y,z
  float PVXYZMax[3] = {999.f, 999.f, 999.f};    // primary vertex max x,y,z
  float TPCOnlyMaxDCARZ[2] = {-999.f, -999.f};  // max DCA r,z (if positive) for TPC only tracks in the primary vertex mode
  float TPCEtaAbsMax = -1.f;                    // if positive, show TPC tracks only below this abs eta

  bool PVMode = false;         // produce jsons with individual primary vertices, not total time frame data
  bool PVTriggersMode = false; // instead of drawing vertices with tracks (and maybe calorimeter triggers), draw vertices with calorimeter triggers (and maybe tracks)"
  bool trackSorting = false;   // sort track by track time before applying filters
  bool filterITSROF = false;   // don't display tracks outside ITS readout frame
  bool calibrateEMC = true;    // apply on-the-fly EMCAL calibration

  O2ParamDef(EveConfParam, "eveconf");
};
} // namespace o2::event_visualisation

#endif

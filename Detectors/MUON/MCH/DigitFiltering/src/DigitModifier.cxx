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

#include "MCHDigitFiltering/DigitModifier.h"

#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include <fmt/format.h>
#include <functional>
#include <array>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <limits>

namespace
{
using PadRemappingTable = std::unordered_map<int, int>;
using PadRemappingTableWithLimits = std::pair<PadRemappingTable, std::pair<int, int>>;
using PadRemappingTablesForDE = std::vector<PadRemappingTableWithLimits>;
using PadRemappingTables = std::unordered_map<int, PadRemappingTablesForDE>;

// utility function that updates a digit with a given pad remapping table
bool updateDigitMapping(o2::mch::Digit& digit, const PadRemappingTables& padsRemapping)
{
  int deId = digit.getDetID();
  // check if the current DE is included in the pad remapping table
  auto padsRemappingForDE = padsRemapping.find(deId);
  if (padsRemappingForDE == padsRemapping.end()) {
    return false;
  }

  // find the remapping table that contains this padId, if existing
  int padId = digit.getPadID();
  for (auto& padsRemappingForDS : padsRemappingForDE->second) {
    if (padId < padsRemappingForDS.second.first || padId > padsRemappingForDS.second.second) {
      continue;
    }

    auto padIDRemapped = padsRemappingForDS.first.find(digit.getPadID());
    if (padIDRemapped == padsRemappingForDS.first.end()) {
      continue;
    }

    // update the digit
    digit.setPadID(padIDRemapped->second);
    return true;
  }
  return false;
}

/** Initialization of the pad remapping table for Station 1 DEs
 *  See https://its.cern.ch/jira/browse/MCH-4 for detals
 */
void initST1PadsRemappingTable(PadRemappingTables& fullTable)
{
  std::array<int, 8> deToRemap{100, 101, 102, 103, 200, 201, 202, 203};
  std::array<int, 7> dsToRemap{1, 27, 53, 79, 105, 131, 157};

  std::vector<int> newToOld(64);
  newToOld[0] = 55;
  newToOld[1] = 1;
  newToOld[2] = 11;
  newToOld[3] = 48;
  newToOld[4] = 4;
  newToOld[5] = 52;
  newToOld[6] = 12;
  newToOld[7] = 61;
  newToOld[8] = 59;
  newToOld[9] = 9;
  newToOld[10] = 10;
  newToOld[11] = 17;
  newToOld[12] = 5;
  newToOld[13] = 36;
  newToOld[14] = 57;
  newToOld[15] = 13;
  newToOld[16] = 21;
  newToOld[17] = 23;
  newToOld[18] = 34;
  newToOld[19] = 58;
  newToOld[20] = 20;
  newToOld[21] = 62;
  newToOld[22] = 43;
  newToOld[23] = 24;
  newToOld[24] = 38;
  newToOld[25] = 49;
  newToOld[26] = 26;
  newToOld[27] = 47;
  newToOld[28] = 50;
  newToOld[29] = 41;
  newToOld[30] = 31;
  newToOld[31] = 53;
  newToOld[32] = 32;
  newToOld[33] = 15;
  newToOld[34] = 33;
  newToOld[35] = 42;
  newToOld[36] = 3;
  newToOld[37] = 18;
  newToOld[38] = 37;
  newToOld[39] = 40;
  newToOld[40] = 30;
  newToOld[41] = 39;
  newToOld[42] = 46;
  newToOld[43] = 22;
  newToOld[44] = 35;
  newToOld[45] = 45;
  newToOld[46] = 0;
  newToOld[47] = 25;
  newToOld[48] = 51;
  newToOld[49] = 27;
  newToOld[50] = 28;
  newToOld[51] = 44;
  newToOld[52] = 6;
  newToOld[53] = 29;
  newToOld[54] = 2;
  newToOld[55] = 56;
  newToOld[56] = 19;
  newToOld[57] = 60;
  newToOld[58] = 54;
  newToOld[59] = 16;
  newToOld[60] = 8;
  newToOld[61] = 14;
  newToOld[62] = 7;
  newToOld[63] = 63;

  for (auto deId : deToRemap) {

    // create an empty table, or reset the existing one
    fullTable[deId] = PadRemappingTablesForDE();
    // get a reference to the table for the current DE
    auto& tableForDE = fullTable[deId];

    const o2::mch::mapping::Segmentation& segment = o2::mch::mapping::segmentation(deId);
    for (auto dsId : dsToRemap) {
      // add an empty table for the currend DS board
      auto& tableForDSWithLimits = tableForDE.emplace_back();
      auto& tableForDS = tableForDSWithLimits.first;

      int padIdMin = std::numeric_limits<int>::max();
      int padIdMax = -1;
      for (int channel = 0; channel < 64; channel++) {
        // get the pad ID associated to the channel in the new mapping
        // this IS NOT the pad that originally fired
        int padId = segment.findPadByFEE(dsId, channel);
        // get the corresponding channel number in the old mapping
        // this IS the electronic channel that originally fired
        int channelInOldMapping = newToOld[channel];
        // get the pad ID associated to the fired channel in the new mapping
        int padIdRemapped = segment.findPadByFEE(dsId, channelInOldMapping);
        // update the pad remapping table
        tableForDS[padId] = padIdRemapped;

        padIdMin = std::min(padIdMin, padId);
        padIdMax = std::max(padIdMax, padId);
      }

      tableForDSWithLimits.second.first = padIdMin;
      tableForDSWithLimits.second.second = padIdMax;
    }
  }
}

o2::mch::DigitModifier createST1MappingCorrector(int runNumber)
{
  static PadRemappingTables padsRemapping;

  constexpr int lastRunToBeFixed = 560402;
  // ST2 mapping needs to be corrected only for data collected up to the end of 2024 Pb-Pb
  if (runNumber > lastRunToBeFixed) {
    // do not modify digits collected after 2024 Pb-Pb
    return {};
  }

  if (padsRemapping.empty()) {
    initST1PadsRemappingTable(padsRemapping);
  }

  return [](o2::mch::Digit& digit) {
    updateDigitMapping(digit, padsRemapping);
  };
}

/** Initialization of the pad remapping table for Station 2 DEs
 *  See https://its.cern.ch/jira/browse/MCH-5 for details
 */
void initST2PadsRemappingTable(PadRemappingTables& fullTable)
{
  // Remapping of ST2 DS boards near the rounded part
  std::array<int, 8> deToRemap{300, 301, 302, 303, 400, 401, 402, 403};
  std::array<int, 5> dsToRemap{99, 100, 101, 102, 103};

  for (auto deId : deToRemap) {

    // create an empty table, or reset the existing one
    fullTable[deId] = PadRemappingTablesForDE();
    // get a reference to the table for the current DE
    auto& tableForDE = fullTable[deId];

    const o2::mch::mapping::Segmentation& segment = o2::mch::mapping::segmentation(deId);
    for (auto dsId : dsToRemap) {

      auto& tableForDSWithLimits = tableForDE.emplace_back();
      auto& tableForDS = tableForDSWithLimits.first;

      // double loop on DS channels
      // 1. find the minimum pad index of the DS board
      int padIdMin = -1;
      int channelForPadIdMin = -1;
      for (int channel = 0; channel < 64; channel++) {
        auto padId = segment.findPadByFEE(dsId, int(channel));
        if (padId < 0) {
          // this should never occur in this specific case, as all channels of this group of boards
          // is connected to pads, hence we rise an exception
          throw std::out_of_range(fmt::format("Unknown padId for DE{} DS{} channel {}", deId, dsId, channel));
        }
        if (padIdMin < 0 || padId < padIdMin) {
          padIdMin = padId;
          channelForPadIdMin = channel;
        }
      }

      int padIdMax = -1;
      // 2. build the re-mapping table
      for (int channel = 0; channel < 64; channel++) {
        auto padId = segment.findPadByFEE(dsId, int(channel));
        if (padId < padIdMin) {
          // something is wrong here...
          continue;
        }

        // update maximum padId value
        padIdMax = std::max(padIdMax, padId);

        int padIdInDS = padId - padIdMin;
        int padColumn = padIdInDS / 16;
        int padRow = padIdInDS % 16;

        int padIdRemapped = -1;

        switch (padColumn) {
          case 0:
            // shift right by 3 columns
            padIdRemapped = padId + 16 * 3;
            break;
          case 1:
            // shift right by 1 column
            padIdRemapped = padId + 16;
            break;
          case 2:
            // shift left by 1 column
            padIdRemapped = padId - 16;
            break;
          case 3:
            // shift left by 3 columns
            padIdRemapped = padId - 16 * 3;
            break;
        }

        // padsRemapping[deId][padId] = padIdRemapped;
        tableForDS[padId] = padIdRemapped;
      }

      tableForDSWithLimits.second.first = padIdMin;
      tableForDSWithLimits.second.second = padIdMax;
    }
  }
}

o2::mch::DigitModifier createST2MappingCorrector(int runNumber)
{
  // static std::unordered_map<int, std::unordered_map<int, int>> padsRemapping;
  static PadRemappingTables padsRemapping;

  constexpr int lastRunToBeFixed = 560402;
  // ST2 mapping needs to be corrected only for data collected up to the end of 2024 Pb-Pb
  if (runNumber > lastRunToBeFixed) {
    // do not modify digits collected after 2024 Pb-Pb
    return {};
  }

  if (padsRemapping.empty()) {
    initST2PadsRemappingTable(padsRemapping);
  }

  return [](o2::mch::Digit& digit) {
    updateDigitMapping(digit, padsRemapping);
  };
}
} // namespace

namespace o2::mch
{
DigitModifier createDigitModifier(int runNumber,
                                  bool updateST1,
                                  bool updateST2)
{
  DigitModifier modifierST1 = updateST1 ? createST1MappingCorrector(runNumber) : DigitModifier{};
  DigitModifier modifierST2 = updateST2 ? createST2MappingCorrector(runNumber) : DigitModifier{};

  if (modifierST1 || modifierST2) {
    return [modifierST1, modifierST2](Digit& digit) {
      // the ST1/ST2 modifiers are mutually exclusive, depending on the DeID associated to the digit
      auto detID = digit.getDetID();
      if (modifierST1 && detID >= 100 && detID < 300) {
        modifierST1(digit);
      }
      if (modifierST2 && detID >= 300 && detID < 500) {
        modifierST2(digit);
      }
    };
  } else {
    // return an empty function if none of the modifiers is set
    return {};
  }
}

} // namespace o2::mch

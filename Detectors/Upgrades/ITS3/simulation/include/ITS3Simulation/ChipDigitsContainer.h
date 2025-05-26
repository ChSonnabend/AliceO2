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

#ifndef ALICEO2_ITS3_CHIPDIGITSCONTAINER_
#define ALICEO2_ITS3_CHIPDIGITSCONTAINER_

#include "ITSMFTBase/SegmentationAlpide.h"        // Base class in o2::itsmft namespace
#include "ITSMFTSimulation/ChipDigitsContainer.h" // Base class in o2::itsmft namespace
#include "ITS3Base/SegmentationMosaix.h"          // OB segmentation implementation
#include "ITS3Base/SpecsV2.h"                     // Provides SpecsV2::isDetITS3() interface
#include "ITS3Simulation/DigiParams.h"            // ITS3-specific DigiParams interface
#include <TRandom.h>

namespace o2::its3
{

class ChipDigitsContainer : public o2::itsmft::ChipDigitsContainer
{
 private:
  bool innerBarrel; ///< true if the chip belongs to the inner barrel (IB), false if outer barrel (OB)
  int maxRows;      ///< maximum number of rows
  int maxCols;      ///< maximum number of columns

 public:
  explicit ChipDigitsContainer(UShort_t idx = 0);

  using SegmentationIB = SegmentationMosaix;
  using SegmentationOB = o2::itsmft::SegmentationAlpide;

  /// Returns whether the chip is in the inner barrel (IB)
  void setChipIndex(UShort_t idx)
  {
    o2::itsmft::ChipDigitsContainer::setChipIndex(idx);
    innerBarrel = constants::detID::isDetITS3(getChipIndex());
    maxRows = innerBarrel ? SegmentationIB::NRows : SegmentationOB::NRows;
    maxCols = innerBarrel ? SegmentationIB::NCols : SegmentationOB::NCols;
  }

  int getMaxRows() const { return maxRows; }
  int getMaxCols() const { return maxCols; }
  bool isIB() const;
  /// Adds noise digits, deleted the one using the itsmft::DigiParams interface
  void addNoise(UInt_t rofMin, UInt_t rofMax, const o2::itsmft::DigiParams* params, int maxRows = o2::itsmft::SegmentationAlpide::NRows, int maxCols = o2::itsmft::SegmentationAlpide::NCols) = delete;
  void addNoise(UInt_t rofMin, UInt_t rofMax, const o2::its3::DigiParams* params);

  ClassDefNV(ChipDigitsContainer, 1);
};

} // namespace o2::its3

#endif // ALICEO2_ITS3_CHIPDIGITSCONTAINER_
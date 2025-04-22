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

/// \file Cluster.cxx
/// \brief Implementation of the ITSMFT cluster

#include "DataFormatsITSMFT/CompCluster.h"
#include <cassert>
#include <iostream>
#include <format>

using namespace o2::itsmft;

std::ostream& operator<<(std::ostream& stream, const CompCluster& cl)
{
  stream << cl.asString();
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CompClusterExt& cl)
{
  stream << cl.asString();
  return stream;
}

std::string CompCluster::asString() const
{
  return std::format(" row: {:4d} col: {:4d} pattID: {:4d} [flag: {:1d}]", getRow(), getCol(), getPatternID(), getFlag());
}

std::string CompClusterExt::asString() const
{
  return std::format(" chip: {:5d} row: {:4d} col: {:4d} pattID: {:4d} [flag: {:1d}]", getChipID(), getRow(), getCol(), getPatternID(), getFlag());
}

//______________________________________________________________________________
void CompCluster::print() const
{
  // print itself
  std::cout << *this << "\n";
}

//______________________________________________________________________________
void CompClusterExt::print() const
{
  // print itself
  std::cout << *this << "\n";
}

//______________________________________________________________________________
void CompCluster::sanityCheck()
{
  // check self-consistency
  static_assert(NBitsRow + NBitsCol + NBitsPattID + 1 < 8 * sizeof(mData), "mData is too short to fit all fields");
}

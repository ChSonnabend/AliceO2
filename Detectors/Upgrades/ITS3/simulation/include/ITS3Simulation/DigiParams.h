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

#ifndef ITS3_DIGIPARAMS_H
#define ITS3_DIGIPARAMS_H

#include "ITSMFTSimulation/DigiParams.h"

namespace o2::its3
{

class DigiParams final : public o2::itsmft::DigiParams
{
 public:
  const o2::itsmft::AlpideSimResponse* getAlpSimResponse() const = delete;
  void setAlpSimResponse(const o2::itsmft::AlpideSimResponse* par) = delete;

  const o2::itsmft::AlpideSimResponse* getOBSimResponse() const { return mOBSimResponse; }
  void setOBSimResponse(const o2::itsmft::AlpideSimResponse* response) { mOBSimResponse = response; }

  const o2::itsmft::AlpideSimResponse* getIBSimResponse() const { return mIBSimResponse; }
  void setIBSimResponse(const o2::itsmft::AlpideSimResponse* response) { mIBSimResponse = response; }

  bool hasResponseFunctions() const { return mIBSimResponse != nullptr && mOBSimResponse != nullptr; }

  void print() const final;

 private:
  const o2::itsmft::AlpideSimResponse* mOBSimResponse = nullptr; //!< pointer to external response
  const o2::itsmft::AlpideSimResponse* mIBSimResponse = nullptr; //!< pointer to external response

  ClassDef(DigiParams, 1);
};

} // namespace o2::its3

#endif

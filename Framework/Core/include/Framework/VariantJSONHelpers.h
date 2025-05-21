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
#ifndef FRAMEWORK_VARIANTJSONHELPERS_H
#define FRAMEWORK_VARIANTJSONHELPERS_H

#include "Framework/Variant.h"

#include <rapidjson/reader.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/error/en.h>

#include <iosfwd>

namespace o2::framework
{
struct VariantJSONHelpers {
  template <VariantType V>
  static Variant read(std::istream& s);

  static void write(std::ostream& o, Variant const& v);
};
} // namespace o2::framework

#endif // FRAMEWORK_VARIANTJSONHELPERS_H

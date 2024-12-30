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

/*
 Due to the root bug https://github.com/root-project/root/issues/17216
 we cannot safely use std::pair<std::int64_t,...> since it is saved in the
 root file as long int, on the MacOS considered to be different from int64_t or
 UInt64_t. Thererefor, we define out own O2LongInt and make sure that it is at
 least 8 bytes long.
*/

#ifndef O2_FRAMEWORK_O2LONGINT_H_
#define O2_FRAMEWORK_O2LONGINT_H_

namespace o2
{

static_assert(sizeof(long int) >= 8, "long int on this machine is < 8 bytes.");

using O2LongInt = long int;
using O2LongUInt = unsigned long int;

} // namespace o2
#endif // O2_FRAMEWORK_O2LONGINT_H_

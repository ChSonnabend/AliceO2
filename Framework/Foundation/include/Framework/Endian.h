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

#ifndef O2_FRAMEWORK_ENDIAN_H_
#define O2_FRAMEWORK_ENDIAN_H_

#include <cstdint>
#include <concepts>
#include <cstring>
// Lookup file for __BYTE_ORDER
#ifdef __APPLE__
#include <machine/endian.h>
#define swap16_ ntohs
#define swap32_ ntohl
#define swap64_ ntohll
#else
#include <endian.h>
#define swap16_ be16toh
#define swap32_ be32toh
#define ntohll be64toh
#define htonll htobe64
#define swap64_ ntohll
#endif
#define O2_HOST_BYTE_ORDER __BYTE_ORDER
#define O2_BIG_ENDIAN __BIG_ENDIAN
#define O2_LITTLE_ENDIAN __LITTLE_ENDIAN

inline uint16_t doSwap(std::same_as<uint16_t> auto x)
{
  return swap16_(x);
}

inline uint32_t doSwap(std::same_as<uint32_t> auto x)
{
  return swap32_(x);
}

inline uint64_t doSwap(std::same_as<uint64_t> auto x)
{
  return swap64_(x);
}

template <typename T>
inline void doSwapCopy_(void* dest, void* source, int size) noexcept
{
  auto tdest = static_cast<T*>(dest);
  auto tsrc = static_cast<T*>(source);
  for (auto i = 0; i < size; ++i) {
    tdest[i] = doSwap<T>(tsrc[i]);
  }
}

inline void swapCopy(unsigned char* dest, char* source, int size, int typeSize) noexcept
{
  switch (typeSize) {
    case 1:
      return (void)std::memcpy(dest, source, size);
    case 2:
      return doSwapCopy_<uint16_t>(dest, source, size);
    case 4:
      return doSwapCopy_<uint32_t>(dest, source, size);
    case 8:
      return doSwapCopy_<uint64_t>(dest, source, size);
  }
}
#endif // O2_FRAMEWORK_ENDIAN_H_

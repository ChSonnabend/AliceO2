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
#ifndef O2_FRAMEWORK_ENUM_BIT_OPERATORS_H_
#define O2_FRAMEWORK_ENUM_BIT_OPERATORS_H_

#include <type_traits>

#define O2_DEFINE_ENUM_BIT_OPERATORS(enum_t)             \
  constexpr auto operator|(enum_t lhs, enum_t rhs)       \
  {                                                      \
    return static_cast<enum_t>(                          \
      static_cast<std::underlying_type_t<enum_t>>(lhs) | \
      static_cast<std::underlying_type_t<enum_t>>(rhs)); \
  }                                                      \
                                                         \
  constexpr auto operator&(enum_t lhs, enum_t rhs)       \
  {                                                      \
    return static_cast<enum_t>(                          \
      static_cast<std::underlying_type_t<enum_t>>(lhs) & \
      static_cast<std::underlying_type_t<enum_t>>(rhs)); \
  }                                                      \
                                                         \
  constexpr auto operator^(enum_t lhs, enum_t rhs)       \
  {                                                      \
    return static_cast<enum_t>(                          \
      static_cast<std::underlying_type_t<enum_t>>(lhs) ^ \
      static_cast<std::underlying_type_t<enum_t>>(rhs)); \
  }                                                      \
                                                         \
  constexpr auto operator~(enum_t op)                    \
  {                                                      \
    return static_cast<enum_t>(                          \
      ~static_cast<std::underlying_type_t<enum_t>>(op)); \
  }                                                      \
                                                         \
  constexpr auto& operator|=(enum_t& lhs, enum_t rhs)    \
  {                                                      \
    lhs = lhs | rhs;                                     \
    return lhs;                                          \
  }                                                      \
                                                         \
  constexpr auto& operator&=(enum_t& lhs, enum_t rhs)    \
  {                                                      \
    lhs = lhs & rhs;                                     \
    return lhs;                                          \
  }                                                      \
                                                         \
  constexpr enum_t& operator^=(enum_t& lhs, enum_t rhs)  \
  {                                                      \
    lhs = lhs ^ rhs;                                     \
    return lhs;                                          \
  }

#define O2_ENUM_TEST_BIT(mask, value) ((mask & value) == value)
#define O2_ENUM_SET_BIT(bit) ((1 << bit))
#define O2_ENUM_ANY_BIT(enum) ((static_cast<std::underlying_type_t<decltype(enum)>>(enum) != 0))

#endif

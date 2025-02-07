# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE GPUUtils)

set(HDRS_CINT FlatObject.h
              GPUCommonBitSet.h
)

set(HDRS_INSTALL
)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ../GPUTracking/utils/EmptyFile.cxx
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon ROOT::RIO)
  target_include_directories(${targetName}
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                    $<INSTALL_INTERFACE:include/GPU>)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT}
                            LINKDEF GPUUtilsLinkDef.h)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB GPUCA_TPC_GEOMETRY_O2)

  install(FILES ${HDRS_CINT} ${HDRS_INSTALL} DESTINATION include/GPU)
endif()

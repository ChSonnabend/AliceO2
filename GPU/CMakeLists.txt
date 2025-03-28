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

# Subdirectories will be compiled with O2 / Standalone To simplify the
# CMake, variables are defined for Sources / Headers first. Then, the actual
# CMake build scripts use these variables.

add_subdirectory(Common)
add_subdirectory(Utils)
add_subdirectory(TPCFastTransformation)
add_subdirectory(GPUTracking)
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  add_subdirectory(GPUbenchmark)
  add_subdirectory(Workflow)
endif()

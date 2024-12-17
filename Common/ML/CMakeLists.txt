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

# Pass ORT variables as a preprocessor definition
if(DEFINED ENV{ORT_ROCM_BUILD})
    add_compile_definitions(ORT_ROCM_BUILD=$ENV{ORT_ROCM_BUILD})
endif()
if(DEFINED ENV{ORT_CUDA_BUILD})
    add_compile_definitions(ORT_CUDA_BUILD=$ENV{ORT_CUDA_BUILD})
endif()
if(DEFINED ENV{ORT_MIGRAPHX_BUILD})
    add_compile_definitions(ORT_MIGRAPHX_BUILD=$ENV{ORT_MIGRAPHX_BUILD})
endif()
if(DEFINED ENV{ORT_TENSORRT_BUILD})
    add_compile_definitions(ORT_TENSORRT_BUILD=$ENV{ORT_TENSORRT_BUILD})
endif()

o2_add_library(ML
               SOURCES src/OrtInterface.cxx
               TARGETVARNAME targetName
               PRIVATE_LINK_LIBRARIES O2::Framework ONNXRuntime::ONNXRuntime)

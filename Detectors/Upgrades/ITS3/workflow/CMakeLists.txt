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

# add_compile_options(-O0 -g -fPIC -fno-omit-frame-pointer)

o2_add_library(ITS3Workflow
               SOURCES src/DigitReaderSpec.cxx
                       src/DigitWriterSpec.cxx
                       src/RecoWorkflow.cxx
                       src/ClustererSpec.cxx
                       src/TrackerSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::DataFormatsITSMFT
                                     O2::SimulationDataFormat
                                     O2::ITS3Simulation
                                     O2::ITStracking
                                     O2::ITSMFTReconstruction
                                     O2::ITS3Reconstruction
                                     O2::ITS3TrackingInterface
                                     O2::ITSWorkflow
                                     O2::GPUTracking
                                     O2::ITSBase)

o2_add_executable(reco-workflow
                  SOURCES src/its3-reco-workflow.cxx
                  COMPONENT_NAME its3
                  PUBLIC_LINK_LIBRARIES O2::ITS3Workflow)

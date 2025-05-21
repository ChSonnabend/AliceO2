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

o2_add_library(ITS3Simulation
               SOURCES src/ITS3Layer.cxx
                       src/ITS3Services.cxx
                       src/DescriptorInnerBarrelITS3.cxx
                       src/Digitizer.cxx
                       src/DigiParams.cxx
                       src/ITS3DPLDigitizerParam.cxx
                       src/ChipDigitsContainer.cxx
                       src/ChipSimResponse.cxx
               PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                     O2::ITSBase O2::ITSMFTSimulation O2::ITSMFTBase
                                     ROOT::Physics)

o2_target_root_dictionary(ITS3Simulation
                          HEADERS include/ITS3Simulation/ITS3Layer.h
                                  include/ITS3Simulation/ITS3Services.h
                                  include/ITS3Simulation/DescriptorInnerBarrelITS3.h
                                  include/ITS3Simulation/Digitizer.h
                                  include/ITS3Simulation/DigiParams.h
                                  include/ITS3Simulation/ITS3DPLDigitizerParam.h
                                  include/ITS3Simulation/ChipDigitsContainer.h
                                  include/ITS3Simulation/ChipSimResponse.h
                                  )

o2_data_file(COPY data  DESTINATION Detectors/ITS3/simulation)

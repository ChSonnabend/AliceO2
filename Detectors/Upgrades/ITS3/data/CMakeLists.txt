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

set(APTS_RESPONSE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/APTSResponseData.root")

add_custom_command(
  OUTPUT ${APTS_RESPONSE_OUTPUT}
  COMMAND ${CMAKE_BINARY_DIR}/stage/bin/o2-alpide-response-generator
    -c APTS
    -i ${ITSRESPONSE_DIR}/response/ITS3ChipResponseData/AptsResponseData/
    -o ${CMAKE_CURRENT_BINARY_DIR}/
  DEPENDS GenerateAlpideResponse
          ${ITSRESPONSE_DIR}/response/ITS3ChipResponseData/AptsResponseData/
  COMMENT "Generating APTSResponseData.root"
  VERBATIM
)

add_custom_target(
  GenerateAPTSResponse ALL
  DEPENDS ${APTS_RESPONSE_OUTPUT}
)

install(
  FILES ${APTS_RESPONSE_OUTPUT}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Detectors/Upgrades/ITS3/data/ITS3ChipResponseData/"
)

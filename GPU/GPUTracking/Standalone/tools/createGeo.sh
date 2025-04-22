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

#!/bin/bash

PATH=$PATH:/usr/share/Modules/bin/:/home/qon/alice/alibuild
export ALIBUILD_WORK_DIR="$HOME/alice/sw"
eval "`alienv shell-helper`"
alienv load O2/latest

o2-sim -n 1

export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/home/qon/alice/GPU/Common/:/home/qon/alice/GPU/GPUTracking/Base:/home/qon/alice/GPU/GPUTracking/SectorTracker:/home/qon/alice/GPU/GPUTracking/Merger:/home/qon/alice/GPU/GPUTracking/TRDTracking
root -l -q -b createGeo.C+

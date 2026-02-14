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

#ifndef O2_CALIBRATION_FT0_EVENTS_PER_BC_CALIBRATOR_H
#define O2_CALIBRATION_FT0_EVENTS_PER_BC_CALIBRATOR_H

#include "Framework/runDataProcessing.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include "DataFormatsFT0/Digit.h"
#include "FT0Calibration/EventsPerBcCalibrator.h"

namespace o2::calibration
{
class FT0EventsPerBcProcessor final : public o2::framework::Task
{
 public:
  FT0EventsPerBcProcessor(std::shared_ptr<o2::base::GRPGeomRequest> request) : mCCDBRequest(request) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    if (ic.options().hasOption("slot-len-sec")) {
      mSlotLenSec = ic.options().get<uint32_t>("slot-len-sec");
    }
    if (ic.options().hasOption("one-object-per-run")) {
      mOneObjectPerRun = ic.options().get<bool>("one-object-per-run");
    }
    if (ic.options().hasOption("min-entries-number")) {
      mMinNumberOfEntries = ic.options().get<uint32_t>("min-entries-number");
    }
    if (ic.options().hasOption("min-ampl-side-a")) {
      mMinAmplitudeSideA = ic.options().get<int32_t>("min-ampl-side-a");
    }
    if (ic.options().hasOption("min-ampl-side-c")) {
      mMinAmplitudeSideC = ic.options().get<int32_t>("min-ampl-side-c");
    }

    mCalibrator = std::make_unique<o2::ft0::EventsPerBcCalibrator>(mMinNumberOfEntries, mMinAmplitudeSideA, mMinAmplitudeSideC);

    if (mOneObjectPerRun) {
      LOG(info) << "Only one object will be created at the end of run";
      mCalibrator->setUpdateAtTheEndOfRunOnly();
    }
    if (mOneObjectPerRun == false) {
      LOG(info) << "Defined slot interval to " << mSlotLenSec << " seconds";
      mCalibrator->setSlotLengthInSeconds(mSlotLenSec);
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    if (digits.size() == 0) {
      return;
    }
    mCalibrator->process(digits);
    if (mOneObjectPerRun == false) {
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Received end-of-stream, checking for slot to finalize...";
    mCalibrator->checkSlotsToFinalize();
    sendOutput(ec.outputs());
    mCalibrator->initOutput();
  }

  void sendOutput(o2::framework::DataAllocator& output)
  {
    using o2::framework::Output;
    const auto& tvxHists = mCalibrator->getTvxPerBc();
    auto& infos = mCalibrator->getTvxPerBcCcdbInfo();
    for (unsigned int idx = 0; idx < tvxHists.size(); idx++) {
      auto& info = infos[idx];
      const auto& payload = tvxHists[idx];

      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, info.get());
      LOG(info) << "Sending object " << info->getPath() << "/" << info->getFileName() << " of size " << image->size()
                << " bytes, valid for " << info->getStartValidityTimestamp() << " : " << info->getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "EventsPerBc", idx}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EventsPerBc", idx}, *info.get());
    }

    if (tvxHists.size()) {
      mCalibrator->initOutput();
    }
  }

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  std::unique_ptr<o2::ft0::EventsPerBcCalibrator> mCalibrator;
  bool mOneObjectPerRun;
  uint32_t mSlotLenSec;
  uint32_t mMinNumberOfEntries;
  int32_t mMinAmplitudeSideA;
  int32_t mMinAmplitudeSideC;
};
} // namespace o2::calibration
#endif
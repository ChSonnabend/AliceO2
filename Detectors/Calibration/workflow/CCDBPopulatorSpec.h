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

#ifndef O2_CALIBRATION_CCDBPOPULATOR_H
#define O2_CALIBRATION_CCDBPOPULATOR_H

/// @file   CCDBPopulator.h
/// @brief  device to populate CCDB

#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include <unordered_map>
#include <chrono>
#include <vector>
#include <utility>
#include <map>

namespace o2
{
namespace calibration
{

class CCDBPopulator : public o2::framework::Task
{
 public:
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbApi = o2::ccdb::CcdbApi;

  using BLOB = std::vector<char>;
  using TBLOB = std::pair<long, BLOB>; // pair of creation time and object to upload
  using OBJCACHE = std::map<CcdbObjectInfo, TBLOB>;

  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void stop() final;

  void checkCache(long delay);
  void doUpload(const CcdbObjectInfo& wrp, const gsl::span<const char>& pld, bool cached = false);
  void logAsNeeded(long nowMS, const std::string& path, std::string& msg);

 private:
  CcdbApi mAPI;
  long mThrottlingDelayMS = 0;  // LOG(important) at most once per this period for given path
  int mOrderingLatencyMS = -1;  // if >0, bufferize and upload if no object with smaller SOV was received in this time interval in ms
  bool mFatalOnFailure = true;  // produce fatal on failed upload
  bool mValidateUpload = false; // validate upload by querying its headers
  bool mEnded = false;
  std::unordered_map<std::string, std::pair<long, int>> mThrottling;
  std::unordered_map<std::string, OBJCACHE> mOrdCache;
  std::int64_t mSSpecMin = -1;                             // min subspec to accept
  std::int64_t mSSpecMax = -1;                             // max subspec to accept
  std::string mCCDBpath = "http://ccdb-test.cern.ch:8080"; // CCDB path
  int mRunNoFromDH = 0;
  std::string mRunNoStr = {};
};

void CCDBPopulator::init(o2::framework::InitContext& ic)
{
  mCCDBpath = ic.options().get<std::string>("ccdb-path");
  mSSpecMin = ic.options().get<std::int64_t>("sspec-min");
  mSSpecMax = ic.options().get<std::int64_t>("sspec-max");
  mFatalOnFailure = ic.options().get<bool>("fatal-on-failure");
  mValidateUpload = ic.options().get<bool>("validate-upload");
  mThrottlingDelayMS = ic.options().get<std::int64_t>("throttling-delay");
  mOrderingLatencyMS = ic.options().get<int>("ordering-latency");
  mAPI.init(mCCDBpath);
}

void CCDBPopulator::run(o2::framework::ProcessingContext& pc)
{
  int nSlots = pc.inputs().getNofParts(0);
  if (nSlots != pc.inputs().getNofParts(1)) {
    LOGP(alarm, "Number of slots={} in part0 is different from that ({}) in part1", nSlots, pc.inputs().getNofParts(1));
    return;
  } else if (nSlots == 0) {
    LOG(alarm) << "0 slots received";
    return;
  }
  mRunNoFromDH = pc.services().get<o2::framework::TimingInfo>().runNumber;
  if (mRunNoFromDH > 0) {
    mRunNoStr = std::to_string(mRunNoFromDH);
  }
  auto nowMS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  for (int isl = 0; isl < nSlots; isl++) {
    auto refWrp = pc.inputs().get("clbWrapper", isl);
    auto refPld = pc.inputs().get("clbPayload", isl);
    if (!o2::framework::DataRefUtils::isValid(refWrp)) {
      LOGP(alarm, "Wrapper is not valid for slot {}", isl);
      continue;
    }
    if (!o2::framework::DataRefUtils::isValid(refPld)) {
      LOGP(alarm, "Payload is not valid for slot {}", isl);
      continue;
    }
    if (mSSpecMin >= 0 && mSSpecMin <= mSSpecMax) { // there is a selection
      auto ss = std::int64_t(o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(refWrp)->subSpecification);
      if (ss < mSSpecMin || ss > mSSpecMax) {
        continue;
      }
    }
    const auto wrp = pc.inputs().get<CcdbObjectInfo*>(refWrp);
    const auto pld = pc.inputs().get<gsl::span<char>>(refPld); // this is actually an image of TMemFile
    if (!wrp) {
      LOGP(alarm, "No CcdbObjectInfo info for {} at slot {}",
           o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(refWrp)->dataDescription.as<std::string>(), isl);
      continue;
    }
    if (mOrderingLatencyMS <= 0) { // ordering is not requested
      doUpload(*wrp, pld);
    } else {
      auto& pathCache = mOrdCache[wrp->getPath()];
      auto stt = pathCache.emplace(*wrp, std::make_pair(nowMS, std::vector<char>(pld.size())));
      if (stt.second) { // insertion success
        stt.first->second.second.assign(pld.begin(), pld.end());
        std::string msg = fmt::format("Bufferizing for ordering ccdb object {}/{} of size {} valid for {} : {}",
                                      wrp->getPath(), wrp->getFileName(), pld.size(), wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp());
        logAsNeeded(nowMS, wrp->getPath(), msg);
      } else {
        bool v = stt.first != pathCache.end();
        LOGP(error, "failed to bufferize a {} object with SOV={}/EOV={} received at {}, conflicting with previously bufferized one SOV={}/EOV={} received at {}",
             wrp->getPath(), wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp(), nowMS,
             v ? std::to_string(stt.first->first.getStartValidityTimestamp()) : std::string{"N/A"},
             v ? std::to_string(stt.first->first.getEndValidityTimestamp()) : std::string{"N/A"},
             v ? std::to_string(stt.first->second.first) : std::string{"N/A"});
      }
    }
  }
  if (mOrderingLatencyMS > 0) {
    checkCache(mOrderingLatencyMS);
  }
}

void CCDBPopulator::checkCache(long delay)
{
  // check if some entries in cache are ripe enough to upload
  auto nowMS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  for (auto& pathCache : mOrdCache) { // loop over paths
    if (delay < 0 && pathCache.second.size()) {
      LOGP(important, "Uploading {} cached objects for path {}", pathCache.second.size(), pathCache.first);
    }
    for (auto it = pathCache.second.begin(); it != pathCache.second.end();) { // loop over objects of the path
      if (nowMS - it->second.first > delay) {
        doUpload(it->first, {it->second.second.data(), it->second.second.size()}, true);
        it = pathCache.second.erase(it);
      } else {
        break;
      }
    }
  }
}

void CCDBPopulator::doUpload(const CcdbObjectInfo& wrp, const gsl::span<const char>& pld, bool cached)
{
  std::string msg = fmt::format("Storing in ccdb {}{}/{} of size {} valid for {} : {}", cached ? "cached " : "", wrp.getPath(), wrp.getFileName(), pld.size(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
  auto uploadTS = o2::ccdb::getCurrentTimestamp();
  logAsNeeded(uploadTS, wrp.getPath(), msg);
  std::map<std::string, std::string> metadata;
  const auto* md = &wrp.getMetaData();
  if (mRunNoFromDH > 0 && md->find(o2::base::NameConf::CCDBRunTag.data()) == md->end()) { // if valid run number is provided and it is not filled in the metadata, add it to the clone
    metadata = *md;                                                                       // clone since the md from the message is const
    metadata[o2::base::NameConf::CCDBRunTag.data()] = mRunNoStr;
    md = &metadata;
  }
  int res = mAPI.storeAsBinaryFile(&pld[0], pld.size(), wrp.getFileName(), wrp.getObjectType(), wrp.getPath(), *md, wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
  if (res) {
    if (mFatalOnFailure) {
      LOGP(fatal, "failed on uploading to {} / {} for [{}:{}]", mAPI.getURL(), wrp.getPath(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
    } else {
      LOGP(error, "failed on uploading to {} / {} for [{}:{}]", mAPI.getURL(), wrp.getPath(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
    }
  }
  // if requested, make sure that the new object can be queried
  if (mValidateUpload || wrp.getValidateUpload()) {
    constexpr long MAXDESYNC = 3;
    auto headers = mAPI.retrieveHeaders(wrp.getPath(), {}, wrp.getStartValidityTimestamp() + (wrp.getEndValidityTimestamp() - wrp.getStartValidityTimestamp()) / 2);
    if (headers.empty() ||
        std::atol(headers["Created"].c_str()) < uploadTS - MAXDESYNC ||
        std::atol(headers["Valid-From"].c_str()) != wrp.getStartValidityTimestamp() ||
        std::atol(headers["Valid-Until"].c_str()) != wrp.getEndValidityTimestamp()) {
      if (mFatalOnFailure) {
        LOGP(fatal, "Failed to validate upload to {} / {} for [{}:{}]", mAPI.getURL(), wrp.getPath(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
      } else {
        LOGP(error, "Failed to validate upload to {} / {} for [{}:{}]", mAPI.getURL(), wrp.getPath(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
      }
    } else {
      LOGP(important, "Validated upload to {} / {} for [{}:{}]", mAPI.getURL(), wrp.getPath(), wrp.getStartValidityTimestamp(), wrp.getEndValidityTimestamp());
    }
  }
}

void CCDBPopulator::logAsNeeded(long nowMS, const std::string& path, std::string& msg)
{
  auto& lastLog = mThrottling[path];
  if (lastLog.first + mThrottlingDelayMS < nowMS) {
    if (lastLog.second) {
      msg += fmt::format(" ({} uploads were logged as INFO)", lastLog.second);
      lastLog.second = 0;
    }
    lastLog.first = nowMS;
    LOG(important) << msg;
  } else {
    lastLog.second++;
    LOG(info) << msg;
  }
}

void CCDBPopulator::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mEnded) {
    return;
  }
  mEnded = true;
  LOG(info) << "EndOfStream received";
  if (mOrderingLatencyMS > 0) {
    checkCache(-mOrderingLatencyMS); // force
  }
}

void CCDBPopulator::stop()
{
  if (mEnded) {
    return;
  }
  mEnded = true;
  LOG(info) << "Forced stop";
  if (mOrderingLatencyMS > 0) {
    checkCache(-mOrderingLatencyMS); // force
  }
}

} // namespace calibration

namespace framework
{

DataProcessorSpec getCCDBPopulatorDeviceSpec(const std::string& defCCDB, const std::string& nameExt)
{
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs = {{"clbPayload", "CLP", Lifetime::Sporadic}, {"clbWrapper", "CLW", Lifetime::Sporadic}};
  std::string devName = "ccdb-populator";
  devName += nameExt;
  return DataProcessorSpec{
    devName,
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::calibration::CCDBPopulator>()},
    Options{
      {"ccdb-path", VariantType::String, defCCDB, {"Path to CCDB"}},
      {"sspec-min", VariantType::Int64, -1L, {"min subspec to accept"}},
      {"sspec-max", VariantType::Int64, -1L, {"max subspec to accept"}},
      {"ordering-latency", VariantType::Int, -1, {"if enabled (positive) bufferize object and upload it if no object with smaller SOV received in given waiting time (ms)"}},
      {"throttling-delay", VariantType::Int64, 300000L, {"produce important type log at most once per this period in ms for each CCDB path"}},
      {"validate-upload", VariantType::Bool, false, {"valider upload by querying its headers"}},
      {"fatal-on-failure", VariantType::Bool, false, {"do not produce fatal on failed upload"}}}};
}

} // namespace framework
} // namespace o2

#endif

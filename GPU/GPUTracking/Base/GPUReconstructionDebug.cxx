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

/// \file GPUReconstructionDebug.cxx
/// \author David Rohr

#include "GPUReconstruction.h"
#include "GPULogging.h"
#include "GPUSettings.h"

#include <csignal>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <filesystem>
#include <chrono>
#include <format>

using namespace o2::gpu;

struct GPUReconstruction::debugInternal {
  std::function<void(int32_t, siginfo_t*, void*)> signalCallback;
  std::function<void()> debugCallback = nullptr;
  std::function<void()> reinstallCallback = nullptr;
  std::unordered_map<int32_t, struct sigaction> oldActions;
  size_t debugCount = 0;
  static void globalCallback(int32_t signal, siginfo_t* info, void* ucontext)
  {
    GPUReconstruction::mDebugData->signalCallback(signal, info, ucontext);
  }
};

std::unique_ptr<GPUReconstruction::debugInternal> GPUReconstruction::mDebugData;

void GPUReconstruction::debugInit()
{
  if (GetProcessingSettings().debugOnFailure) {
    static std::mutex initMutex;
    {
      std::lock_guard<std::mutex> guard(initMutex);
      if (mDebugData) {
        GPUFatal("Error handlers for debug dumps already set, cannot set them again");
      }
      mDebugData = std::make_unique<debugInternal>();
    }
    mDebugEnabled = true;
    if ((GetProcessingSettings().debugOnFailure & 1) || (GetProcessingSettings().debugOnFailure & 2)) {
      struct sigaction sa, oldsa;
      memset(&sa, 0, sizeof(sa));
      sa.sa_sigaction = GPUReconstruction::debugInternal::globalCallback;
      sa.sa_flags = SA_SIGINFO;
      uint32_t mask = GetProcessingSettings().debugOnFailureSignalMask == (uint32_t)-1 ? ((1 << SIGINT) | (1 << SIGABRT) | (1 << SIGBUS) | (1 << SIGTERM) | (1 << SIGSEGV)) : GetProcessingSettings().debugOnFailureSignalMask;
      if (mask) {
        for (uint32_t i = 0; i < sizeof(mask) * 8; i++) {
          if (mask & (1 << i)) {
            if (sigaction(i, &sa, &oldsa)) {
              GPUFatal("Error installing signal handler for error dump on signal %d", i);
            }
            mDebugData->oldActions.emplace(i, oldsa);
          }
        }
      }

      mDebugData->signalCallback = [this, &oldActions = mDebugData->oldActions, myAction = std::move(sa)](int32_t signal, siginfo_t* info, void* ucontext) {
        static std::mutex callbackMutex;
        std::lock_guard<std::mutex> guard(callbackMutex);
        if (mDebugData->debugCallback) {
          GPUInfo("Running debug callback for signal %d", signal);
          mDebugData->debugCallback();
          mDebugData->debugCount++;
        }
        mDebugData->debugCallback = nullptr;
        if (!GetProcessingSettings().debugOnFailureNoForwardSignal) {
          sigaction(signal, &oldActions[signal], nullptr);
          raise(signal);
          mDebugData->reinstallCallback = [signal, myAction]() { sigaction(signal, &myAction, nullptr); };
        }
      };
    }
  }
}

void GPUReconstruction::debugExit()
{
  if (!mDebugEnabled) {
    return;
  }
  if (mDebugData) {
    for (auto& it : mDebugData->oldActions) {
      if (sigaction(it.first, &it.second, nullptr)) {
        GPUFatal("Error restoring signal handler for signal %d", it.first);
      }
    }
  }
  mDebugEnabled = false;
}

void GPUReconstruction::setDebugDumpCallback(std::function<void()>&& callback)
{
  if (mMaster) {
    if (mDebugData->reinstallCallback) {
      mDebugData->reinstallCallback();
      mDebugData->reinstallCallback = nullptr;
    }
    mMaster->setDebugDumpCallback(std::move(callback));
  } else if (mDebugEnabled && mDebugData) {
    mDebugData->debugCallback = callback;
  }
}

std::string GPUReconstruction::getDebugFolder(const std::string& prefix)
{
  const std::filesystem::path target_dir = GetProcessingSettings().debugOnFailureDirectory;

  std::size_t total_size = 0;
  std::size_t subfolder_count = 0;

  if (!std::filesystem::exists(target_dir) || !std::filesystem::is_directory(target_dir)) {
    GPUError("Invalid debugOnFailureDirectory %s", GetProcessingSettings().debugOnFailureDirectory.c_str());
    return "";
  }

  for (const auto& entry : std::filesystem::directory_iterator(target_dir)) {
    if (entry.is_directory()) {
      subfolder_count++;

      for (const auto& subentry : std::filesystem::directory_iterator(entry.path())) {
        if (subentry.is_regular_file()) {
          std::error_code ec;
          auto size = std::filesystem::file_size(subentry.path(), ec);
          if (!ec) {
            total_size += size;
          }
        }
      }
    }
  }

  if ((GetProcessingSettings().debugOnFailureMaxFiles && subfolder_count >= GetProcessingSettings().debugOnFailureMaxFiles) || (GetProcessingSettings().debugOnFailureMaxSize && (total_size >> 30) >= GetProcessingSettings().debugOnFailureMaxSize)) {
    GPUError("Cannot store debug dump files, target storage exceeded: %zu dumps, %zu bytes", subfolder_count, total_size);
    return "";
  }

  auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::ostringstream dateTime;
  dateTime << std::put_time(std::localtime(&currentTime), "%Y-%m-%d_%H-%M-%S");

  int32_t attempt = 0;
  std::string outname;
  while (true) {
    if (attempt++ >= 512) {
      GPUError("Error creating debug dump folder");
      return "";
    }

    outname = GetProcessingSettings().debugOnFailureDirectory + "/debug_" + prefix + (prefix == "" ? "" : "_") + dateTime.str() + "_" + std::to_string(attempt);
    std::error_code ec;
    bool created = std::filesystem::create_directory(outname, ec);
    if (!ec && created) {
      break;
    }
  }

  GPUInfo("Debug dump to %s", outname.c_str());
  return outname;
}

bool GPUReconstruction::triggerDebugDump()
{
  if (mMaster) {
    return mMaster->triggerDebugDump();
  } else if (mDebugEnabled && mDebugData && mDebugData->debugCallback) {
    GPUInfo("Running triggered debug callback");
    mDebugData->debugCallback();
    mDebugData->debugCount++;
    mDebugData->debugCallback = nullptr;
    return true;
  }
  return false;
}

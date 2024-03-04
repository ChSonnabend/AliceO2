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

///
/// \file     model.h
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class for PyTorch models
///

#ifndef GPU_ML_TORCH_INTERFACE_H
#define GPU_ML_TORCH_INTERFACE_H

// Torch
#include <torch/script.h>
#include <torch/torch.h>

// System includes
#include <iostream>
#include <math.h>
#include <cmath>
#include <memory>
#include <chrono>
#include <thread>

#include "Framework/Logger.h"

namespace o2
{

namespace ml
{

class TorchModel
{

 public:
  TorchModel() = default;
  ~TorchModel() = default;

  // Inferencing
  void init(const std::string, const bool = true);

  // Loggers & Printers
  void printAvailDevices();
  void printModel();

  // Getters
  torch::Device getDevice();
  
  // Setters
  void setDevice(const bool = 1, const torch::Device = torch::kCPU);

 private:
  std::string modelpath;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;
};

} // namespace ml

} // namespace o2

#endif // GPU_ML_TORCH_INTERFACE_H
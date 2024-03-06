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

#include "ML/torch_interface.h"

namespace o2
{

namespace ml
{

void TorchModel::init(const std::string filepath, const bool autodetect){
    setDevice(autodetect, torch::kCPU);
    modelpath = filepath;
    model = torch::jit::load(filepath, device);
    LOG(info) << "(TORCH) Model " << filepath << " loaded";
}

void TorchModel::printModel(){
    LOG(info) << "(TORCH) --- Model ---";
    model.dump(false, false, false);
}

void TorchModel::printAvailDevices(){
    LOG(info) << "(TORCH) --- Printing available devices ---";
    // // Print available GPUs
    // int num_gpus = torch::cuda::device_count();
    // LOG(info) << "Available GPUs:";
    // for (int i = 0; i < num_gpus; ++i) {
    //     torch::cuda::device(i);
    //     auto gpu = torch::cuda::current_device();
    //     LOG(info) << "GPU " << i << ": " << gpu.name();
    //     LOG(info) << "    Device ID: " << gpu.index();
    //     LOG(info) << "    Compute Capability: " << gpu.major() << "." << gpu.minor();
    //     LOG(info) << "    Total Memory: " << gpu.total_memory() << " bytes";
    // }
// 
    // // Print CPU specifications
    // LOG(info) << "CPU Specifications:";
    // LOG(info) << "    Number of threads: " << torch::get_num_threads();
}

// Getters
torch::Device TorchModel::getDevice(){
    return device;
}

// Setters
void TorchModel::setDevice(const bool autodetect, const torch::Device dev){
    std::string string_device = "CPU";
    if(autodetect) {
        LOG(info) << "(TORCH) Device auto-detection enabled!";
        if (torch::cuda::is_available()) {
            LOG(info) << "(TORCH) GPU detected on system";
            // int dev_id = torch::cuda::current_device();
            // LOG(info) << "Found device: " << dev_id << ", name: " << torch::cuda::get_device_name(dev_id);
            device = torch::kCUDA;
            at::Device d(at::kCUDA);
            auto *g = c10::impl::getDeviceGuardImpl(d.type());
            LOG(info) << "(TORCH) Device: " << g->getDevice();
            string_device = "GPU";
        } else {
            LOG(info) << "(TORCH) No GPU detected";
            device = dev;
        }
    } else {
        LOG(info) << "(TORCH) Device auto-detection disabled!";
        if(dev == torch::kCUDA){
            if(torch::cuda::is_available()){
                LOG(info) << "(TORCH) GPU requested as device and found";
                device = torch::kCUDA;
                at::Device d(at::kCUDA);
                auto *g = c10::impl::getDeviceGuardImpl(d.type());
                LOG(info) << "(TORCH) Device: " << g->getDevice();
                string_device = "GPU";
            } else {
                LOG(info) << "(TORCH) GPU requested as device but not found";
                string_device = "CPU";
                device = torch::kCPU;
            }
        } else {
            LOG(info) << "(TORCH) CPU requested as device";
            device = torch::kCPU;
        }
    }
    LOG(info) << "(TORCH) Device set to " << string_device;
}

}
}
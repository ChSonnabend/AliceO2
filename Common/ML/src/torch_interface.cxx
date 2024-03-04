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
    std::cout << "(TORCH) Model " << filepath << " loaded" << std::endl;
}

void TorchModel::printModel(){
    std::cout << "--- Model ---" << std::endl;
    model.dump(false, false, false);
}

void TorchModel::printAvailDevices(){
    std::cout << "(TORCH) --- Printing available devices ---" << std::endl;
    // // Print available GPUs
    // int num_gpus = torch::cuda::device_count();
    // std::cout << "Available GPUs:" << std::endl;
    // for (int i = 0; i < num_gpus; ++i) {
    //     torch::cuda::device(i);
    //     auto gpu = torch::cuda::current_device();
    //     std::cout << "GPU " << i << ": " << gpu.name() << std::endl;
    //     std::cout << "    Device ID: " << gpu.index() << std::endl;
    //     std::cout << "    Compute Capability: " << gpu.major() << "." << gpu.minor() << std::endl;
    //     std::cout << "    Total Memory: " << gpu.total_memory() << " bytes" << std::endl;
    // }
// 
    // // Print CPU specifications
    // std::cout << "CPU Specifications:" << std::endl;
    // std::cout << "    Number of threads: " << torch::get_num_threads() << std::endl;
}

// Getters
torch::Device TorchModel::getDevice(){
    return device;
}

// Setters
void TorchModel::setDevice(const bool autodetect, const torch::Device dev){
    std::string string_device = "CPU";
    if(autodetect) {
        std::cout << "(TORCH) Device auto-detection enabled!" << std::endl;
        if (torch::cuda::is_available()) {
            std::cout << "(TORCH) GPU detected on system";
            // int dev_id = torch::cuda::current_device();
            // std::cout << "Found device: " << dev_id << ", name: " << torch::cuda::get_device_name(dev_id);
            device = torch::kCUDA;
            at::Device d(at::kCUDA);
            auto *g = c10::impl::getDeviceGuardImpl(d.type());
            std::cout << "(TORCH) Device: " << g->getDevice();
            string_device = "GPU";
        } else {
            std::cout << "(TORCH) No GPU detected" << std::endl;
            device = dev;
        }
    } else {
        std::cout << "(TORCH) Device auto-detection disabled!" << std::endl;
        if(dev == torch::kCUDA){
            if(torch::cuda::is_available()){
                std::cout << "(TORCH) GPU requested as device and found" << std::endl;
                device = torch::kCUDA;
                at::Device d(at::kCUDA);
                auto *g = c10::impl::getDeviceGuardImpl(d.type());
                std::cout << "(TORCH) Device: " << g->getDevice() << std::endl;
                string_device = "GPU";
            } else {
                std::cout << "(TORCH) GPU requested as device but not found" << std::endl;
                string_device = "CPU";
                device = torch::kCPU;
            }
        } else {
            std::cout << "(TORCH) CPU requested as device" << std::endl;
            device = torch::kCPU;
        }
    }
    std::cout << "(TORCH) Device set to " << string_device << std::endl;
}

}
}
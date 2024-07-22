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

#include "Framework/Logger.h"

namespace o2
{

namespace ml
{

// Inferencing
void TorchModel::load(const std::string filepath){
    LOG(info) << "(TORCH) >>> Model loader <<<";
    modelpath = filepath;
    model = torch::jit::load(filepath, device);
    LOG(info) << "(TORCH) Model " << filepath << " loaded";
    LOG(info) << "(TORCH) ------------";
}

std::vector<float> TorchModel::inference(std::vector<std::vector<float>> in){
    auto opts = torch::TensorOptions().dtype(dtype);
    torch::Tensor inputs = torch::from_blob(in.data(), {static_cast<long long>(in.size()), static_cast<long long>(in[0].size())}, opts).to(device, dtype);
    at::Tensor output = model.forward(std::vector<torch::jit::IValue>{inputs}).toTensor().to(torch::kCPU, torch::kFloat32);
    auto r_ptr = output.data_ptr<float>();
    std::vector<float> result{r_ptr, r_ptr + output.size(0)};
    torch::cuda::synchronize();
    return result;
}

at::Tensor TorchModel::inference(torch::Tensor in){
    at::Tensor output = model.forward({(torch::jit::IValue)in}).toTensor().to(torch::kCPU, torch::kFloat32);
    torch::cuda::synchronize();
    return output;
}

// Loggers & Printers
void TorchModel::printModel(){
    LOG(info) << "(TORCH) >>> Model <<<";
    model.dump(false, false, false);
}

void TorchModel::printAvailDevices(){
    LOG(info) << "(TORCH) >>> Available devices <<<";
    // Print available GPUs
    if(torch::cuda::is_available()){
        LOG(info) << "(TORCH) --- CUDA / AMD";
      int num_gpus = torch::cuda::device_count();
      LOG(info) << "(TORCH) Available GPUs:";
      for (int i = 0; i < num_gpus; ++i) {
          at::Device d(at::kCUDA, i);
          auto *g = c10::impl::getDeviceGuardImpl(d.type());
          g->getDevice();
          g->setDevice(d);
          if(d.is_cuda()){
            LOG(info) << "(TORCH) Device, GPU " << i;
            // LOG(info) << "GPU " << i << ": " << d.name();
            // LOG(info) << "    Device ID: " << d.index();
            // LOG(info) << "    Compute Capability: " << d.major() << "." << d.minor();
            // LOG(info) << "    Total Memory: " << d.total_memory() << " bytes";
          }
      }
    } 
    // if(torch::mps::is_available()){
    //     LOG(info) << "(TORCH) --- MPS";
    //     LOG(info) << "(TORCH) Metal backend detected!";
    // }
    
    // Print CPU specifications
    LOG(info) << "(TORCH) --- CPU";
    LOG(info) << "(TORCH) Number of threads: " << torch::get_num_threads();
    LOG(info) << "(TORCH) ------------";
}

// Getters
torch::Device TorchModel::getDevice(){
    return device;
}

c10::ScalarType TorchModel::getDType(){
    return dtype;
}

// Setters
void TorchModel::setDevice(const bool autodetect = true, const torch::Device dev = torch::kCPU){
    LOG(info) << "(TORCH) >>> Device-setter <<<";
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
        }
        // else if(torch::mps::is_available()){
        //     LOG(info) << "(TORCH) MPS detected on system";
        //     device = torch::kMPS;
        //     string_device = "MPS";
        // }
        else {
            LOG(info) << "(TORCH) No GPU detected";
            device = dev;
        }
    } else {
        LOG(info) << "(TORCH) Device auto-detection disabled!";
        if(dev == torch::kCUDA){
            if(torch::cuda::is_available()){
                LOG(debug) << "(TORCH) GPU requested as device and found";
                device = torch::kCUDA;
                at::Device d(at::kCUDA);
                auto *g = c10::impl::getDeviceGuardImpl(d.type());
                LOG(debug) << "(TORCH) Device: " << g->getDevice();
                string_device = "GPU";
            } else {
                LOG(debug) << "(TORCH) GPU requested as device but not found";
                string_device = "CPU";
                device = torch::kCPU;
            }
        }
        // else if(dev == torch::kMPS){
        //     if(torch::mps::is_available()){
        //         LOG(debug) << "(TORCH) MPS requested as device and found";
        //         device = torch::kMPS;
        //         string_device = "MPS";
        //     } else {
        //         LOG(debug) << "(TORCH) MPS requested as device but not found";
        //         string_device = "CPU";
        //         device = torch::kCPU;
        //     }
        // }
        else {
            LOG(debug) << "(TORCH) CPU requested as device";
            device = torch::kCPU;
        }
    }
    LOG(info) << "(TORCH) Device set to " << string_device;
    LOG(info) << "(TORCH) ------------";
}

void TorchModel::setDevice(const bool autodetect = true, const std::string dev = "cpu"){
    
    std::string tmp_dev = dev;
    std::transform(tmp_dev.begin(), tmp_dev.end(), tmp_dev.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if(autodetect) {
        setDevice(true, torch::kCPU);
    } else {
        if(tmp_dev == "cuda"){
            setDevice(0, torch::kCUDA);
        } else if(tmp_dev == "mps"){
            setDevice(0, torch::kMPS);
        } else if(tmp_dev == "cpu"){
            setDevice(0, torch::kCPU);
        } else {
            LOG(fatal) << "(TORCH) Device '" << tmp_dev << "' unknown! Please use 'cpu', 'cuda' (Nvidia or AMD backend) or 'mps' (Apple Metal GPU backend)";
        }
    }
}

void TorchModel::setDType(const c10::ScalarType set_dtype = torch::kFloat32){
    dtype = set_dtype;
}

}
}

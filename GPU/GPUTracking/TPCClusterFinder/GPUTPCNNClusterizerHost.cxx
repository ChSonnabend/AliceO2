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

/// \file GPUTPCNNClusterizerHost.cxx
/// \author Christian Sonnabend

#include "Rtypes.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "GPUTPCCFClusterizer.h"

#include "GPUTPCNNClusterizerHost.h"
#include "GPUTPCNNClusterizer.h"
#include "GPUSettings.h"
#include "ML/3rdparty/GPUORTFloat16.h"

using namespace o2::gpu;

GPUTPCNNClusterizerHost::GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer& settings, GPUTPCNNClusterizer& clusterer)
{
  OrtOptions = {
    {"model-path", settings.nnClassificationPath},
    {"device", settings.nnInferenceDevice},
    {"device-id", std::to_string(settings.nnInferenceDeviceId)},
    {"allocate-device-memory", std::to_string(settings.nnInferenceAllocateDevMem)},
    {"dtype", settings.nnInferenceDtype},
    {"intra-op-num-threads", std::to_string(settings.nnInferenceIntraOpNumThreads)},
    {"inter-op-num-threads", std::to_string(settings.nnInferenceInterOpNumThreads)},
    {"enable-optimizations", std::to_string(settings.nnInferenceEnableOrtOptimization)},
    {"enable-profiling", std::to_string(settings.nnInferenceOrtProfiling)},
    {"profiling-output-path", settings.nnInferenceOrtProfilingPath},
    {"logging-level", std::to_string(settings.nnInferenceVerbosity)}};

  model_class.init(OrtOptions);
  clusterer.nnClusterizerModelClassNumOutputNodes = model_class.getNumOutputNodes()[0][1];

  reg_model_paths = splitString(settings.nnRegressionPath, ":");

  if (!settings.nnClusterizerUseCfRegression) {
    if (model_class.getNumOutputNodes()[0][1] == 1 || reg_model_paths.size() == 1) {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
    } else {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
      OrtOptions["model-path"] = reg_model_paths[1];
      model_reg_2.init(OrtOptions);
      clusterer.nnClusterizerModelReg2NumOutputNodes = model_reg_2.getNumOutputNodes()[0][1];
    }
  }
}

void GPUTPCNNClusterizerHost::networkInference(o2::ml::OrtModel model, GPUTPCNNClusterizer& clustererNN, size_t size, float* output, int32_t dtype)
{
  if (dtype == 0) {
    model.inference<OrtDataType::Float16_t, float>(clustererNN.inputData16, size, output);
  } else {
    model.inference<float, float>(clustererNN.inputData32, size, output);
  }
}

// ---------------------------------
void GPUTPCNNClusterizerHost::digitWriter(GPUTPCClusterFinder& clusterer, std::string folder)
{

  ROOT::EnableThreadSafety();

  LOG(info) << "Streaming digits for NN clusterizer training, sector "
            << clusterer.mISector << ", fragment "
            << clusterer.mPmemory->fragment.index;

  if (gSystem->AccessPathName(folder.c_str())) {
    gSystem->mkdir(folder.c_str());
  }

  std::string outputFile = folder + "/tpcdigits_reco_" + std::to_string(clusterer.mISector) + ".root";
  TFile* file = TFile::Open(outputFile.c_str(), "UPDATE");

  if (!file || file->IsZombie()) {
    LOG(error) << "Error opening file: " << outputFile;
    return;
  }

  TTree* tree = (TTree*)file->Get("tr_data");
  std::vector<float> atomic_unit;
  std::vector<std::string> branchNames = {"sector", "row", "pad", "time", "charge", "has3x3Peak", "isSplit"};

  if (tree) {
    TObjArray* branch_list = tree->GetListOfBranches();
    int numBranches = branch_list->GetEntries();

    if (numBranches != branchNames.size()) {
      LOG(error) << "Mismatch in branch count: expected " << branchNames.size() << ", found " << numBranches;
      file->Close();
      delete file;
      return;
    }

    atomic_unit.resize(numBranches);
    for (int i = 0; i < numBranches; i++) {
      TBranch* branch = (TBranch*)branch_list->At(i);
      if (branch) {
        tree->SetBranchAddress(branch->GetName(), &atomic_unit[i]);
      }
    }
  } else {
    tree = new TTree("tr_data", "Neural Network Input Data");
    atomic_unit.resize(branchNames.size());

    for (size_t i = 0; i < branchNames.size(); i++) {
      tree->Branch(branchNames[i].c_str(), &atomic_unit[i], (branchNames[i] + "/F").c_str());
    }
  }

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  for (size_t entry = 0; entry < clusterer.mPmemory->counters.nPositions; entry++) {
    ChargePos pos = clusterer.mPpositions[entry];
    PackedCharge charge = chargeMap[pos];

    if (charge.unpack() > 0) {
      atomic_unit = {
        static_cast<float>(clusterer.mISector),
        static_cast<float>(pos.row()),
        static_cast<float>(pos.pad()),
        static_cast<float>(pos.time() + clusterer.mPmemory->fragment.start),
        static_cast<float>(charge.unpack()),
        static_cast<float>(charge.has3x3Peak()),
        static_cast<float>(charge.isSplit())
      };

      tree->Fill();
    }
  }

  tree->Write("", TObject::kOverwrite);
  file->Write();
  file->Close();
  delete file;

}

// ---------------------------------
void GPUTPCNNClusterizerHost::combineDigitFiles(int sector)
{
  LOG(info) << "Combining digit files for sector " << sector;

  std::string outputDir = "digits_stream";
  std::string outputFile = outputDir + "/tpcdigits_reco_" + std::to_string(sector) + ".root";

  // Check if the output directory exists
  if (gSystem->AccessPathName(outputDir.c_str())) {
    LOG(error) << "Output directory does not exist: " << outputDir;
    return;
  }

  // Find all fragment files for the given sector
  void* dir = gSystem->OpenDirectory(outputDir.c_str());
  if (!dir) {
    LOG(error) << "Error opening directory: " << outputDir;
    return;
  }

  // Combine the fragment files using hadd
  std::string haddCommand = "hadd -f " + outputFile + "tpcdigits_reco_" + std::to_string(sector) + "_*.root" ;

  int haddResult = gSystem->Exec(haddCommand.c_str());
  if (haddResult != 0) {
    LOG(error) << "Error combining files with hadd, command: " << haddCommand;
    return;
  }

  // Remove the individual fragment files
  gSystem->Exec(("rm -rf " + outputDir + "/tpcdigits_reco_" + std::to_string(sector) + "_*.root").c_str());

  LOG(info) << "Successfully combined digit files for sector " << sector << " into " << outputFile;
}

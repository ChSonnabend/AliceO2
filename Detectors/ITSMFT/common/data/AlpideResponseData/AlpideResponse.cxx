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

#include <boost/program_options.hpp>
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <TFile.h>
#include <TSystem.h>
#include <stdexcept>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

void alpideResponse(const std::string& inpath, const std::string& outpath, const std::string& chip_name)
{
  // Check input path validity
  if (gSystem->AccessPathName(inpath.c_str())) {
    throw std::invalid_argument("Input path does not exist or is inaccessible: " + inpath);
  }

  // Check output path validity
  if (gSystem->AccessPathName(outpath.c_str(), kWritePermission)) {
    throw std::invalid_argument("Output path is not writable: " + outpath);
  }

  o2::itsmft::AlpideSimResponse resp0, resp1;

  if (chip_name == "Alpide") {
    resp0.initData(0, inpath.c_str());
    resp1.initData(1, inpath.c_str());
  } else if (chip_name == "APTS") {
    resp1.setColMax(1.5e-4);
    resp1.setRowMax(1.5e-4);
    resp1.initData(1, inpath.c_str());
  } else {
    throw std::invalid_argument("Unknown chip name: " + chip_name);
  }

  std::string output_file = outpath + "/" + chip_name + "ResponseData.root";
  auto file = TFile::Open(output_file.c_str(), "recreate");

  if (!file || file->IsZombie()) {
    throw std::runtime_error("Failed to create output file: " + output_file);
  } else if (chip_name == "Alpide") {
    file->WriteObjectAny(&resp0, "o2::itsmft::AlpideSimResponse", "response0");
  }
  file->WriteObjectAny(&resp1, "o2::itsmft::AlpideSimResponse", "response1");
  file->Close();
  delete file;
}

int main(int argc, const char* argv[])
{
  namespace bpo = boost::program_options;
  bpo::variables_map vm;
  bpo::options_description options("Alpide response generator options");
  options.add_options()("inputdir,i", bpo::value<std::string>()->default_value("./"), "Path where Vbb-0.0V and Vbb-3.0V are located.")("outputdir,o", bpo::value<std::string>()->default_value("./"), "Path where to store the output.")("chip,c", bpo::value<std::string>()->default_value("Alpide"), "Chip name (Alpide or APTS).");

  try {
    bpo::store(parse_command_line(argc, argv, options), vm);

    if (vm.count("help")) {
      std::cout << options << std::endl;
      return 0;
    }

    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments. Available options:\n";
    std::cerr << options << std::endl;
    return 2;
  }

  try {
    std::cout << "Generating response for chip: " << vm["chip"].as<std::string>() << std::endl;
    std::cout << "Input directory: " << vm["inputdir"].as<std::string>() << std::endl;
    std::cout << "Output directory: " << vm["outputdir"].as<std::string>() << std::endl;

    alpideResponse(vm["inputdir"].as<std::string>(),
                   vm["outputdir"].as<std::string>(),
                   vm["chip"].as<std::string>());
    std::cout << "Response file generated successfully." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/RootArrowFilesystem.h"
#include "Framework/PluginManager.h"
#include <TDirectory.h>
#include <TDirectoryFile.h>
#include <getopt.h>
#include <TFile.h>
#include <iostream>
#include <TMap.h>
#include <TTree.h>
#include <fmt/format.h>

int main(int argc, char** argv)
{

  char* input_file = nullptr;
  char* output_file = nullptr;

  // Define long options
  static struct option long_options[] = {
    {"input", required_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
    {nullptr, 0, nullptr, 0} // End of options
  };

  int option_index = 0;
  int c;

  // Parse options
  while ((c = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'i':
        input_file = optarg;
        break;
      case 'o':
        output_file = optarg;
        break;
      case '?':
        // Unknown option
        printf("Unknown option. Use --input <file> and --output <file>\n");
        return 1;
      default:
        break;
    }
  }

  // Check if input and output files are provided
  if (input_file && output_file) {
    printf("Input file: %s\n", input_file);
    printf("Output file: %s\n", output_file);
  } else {
    fprintf(stderr, "Usage: %s --input <file> --output <file>\n", argv[0]);
    return 1;
  }

  // Plugins which understand
  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:RNTupleObjectReadingCapability",
    "O2Framework:TTreeObjectReadingCapability",
  };

  o2::framework::RootObjectReadingFactory factory;

  std::vector<LoadablePlugin> plugins;
  for (auto spec : capabilitiesSpecs) {
    auto morePlugins = o2::framework::PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
  }

  auto in = TFile::Open(input_file, "READ");
  auto out = TFile::Open(output_file, "RECREATE");

  auto fs = std::make_shared<o2::framework::TFileFileSystem>(in, 50 * 1024 * 1024, factory);
  auto outFs = std::make_shared<o2::framework::TFileFileSystem>(out, 0, factory);

  o2::framework::PluginManager::loadFromPlugin<o2::framework::RootObjectReadingCapability, o2::framework::RootObjectReadingCapabilityPlugin>(plugins, factory.capabilities);

  // Plugins are hardcoded for now...
  auto rNtupleFormat = factory.capabilities[0].factory().format();
  auto format = factory.capabilities[1].factory().format();

  for (TObject* dk : *in->GetListOfKeys()) {
    if (dk->GetName() == std::string("metaData")) {
      TMap* m = dynamic_cast<TMap*>(in->Get(dk->GetName()));
      m->Print();
      auto* copy = m->Clone("metaData");
      out->WriteTObject(copy);
      continue;
    }
    if (dk->GetName() == std::string("parentFiles")) {
      TMap* m = dynamic_cast<TMap*>(in->Get(dk->GetName()));
      m->Print();
      auto* copy = m->Clone("parentFiles");
      out->WriteTObject(copy);
      continue;
    }
    auto* d = (TDirectory*)in->Get(dk->GetName());
    std::cout << "Processing: " << dk->GetName() << std::endl;
    // For the moment RNTuple does not support TDirectory, so
    // we write everything at toplevel.
    auto destination = outFs->OpenOutputStream("/", {});
    if (!destination.ok()) {
      std::cerr << "Could not open destination folder " << output_file << std::endl;
      exit(1);
    }

    for (TObject* tk : *d->GetListOfKeys()) {
      auto sourceUrl = fmt::format("{}/{}", dk->GetName(), tk->GetName());
      // FIXME: there is no support for TDirectory yet. Let's write everything
      // at the same level.
      auto destUrl = fmt::format("/{}-{}", dk->GetName(), tk->GetName());
      arrow::dataset::FileSource source(sourceUrl, fs);
      if (!format->IsSupported(source).ok()) {
        std::cout << "Source " << source.path() << " is not supported" << std::endl;
        continue;
      }
      std::cout << "  Processing tree: " << tk->GetName() << std::endl;
      auto schemaOpt = format->Inspect(source);
      if (!schemaOpt.ok()) {
        std::cout << "Could not inspect source " << source.path() << std::endl;
      }
      auto schema = *schemaOpt;
      auto fragment = format->MakeFragment(source, {}, schema);
      if (!fragment.ok()) {
        std::cout << "Could not make fragment from " << source.path() << "with schema:" << schema->ToString() << std::endl;
        continue;
      }
      auto options = std::make_shared<arrow::dataset::ScanOptions>();
      options->dataset_schema = schema;
      auto scanner = format->ScanBatchesAsync(options, *fragment);
      if (!scanner.ok()) {
        std::cout << "Scanner not ok" << std::endl;
        continue;
      }
      auto batches = (*scanner)();
      auto result = batches.result();
      if (!result.ok()) {
        std::cout << "Could not get batches." << std::endl;
        continue;
      }
      std::cout << "   Found a table with " << (*result)->columns().size() << " columns " << (*result)->num_rows() << " rows." << std::endl;

      if ((*result)->num_rows() == 0) {
        std::cout << "Empty table, skipping for now" << std::endl;
        continue;
      }
      arrow::fs::FileLocator locator{outFs, destUrl};
      std::cout << schema->ToString() << std::endl;
      auto writer = rNtupleFormat->MakeWriter(*destination, schema, {}, locator);
      auto success = writer->get()->Write(*result);
      if (!success.ok()) {
        std::cout << "Error while writing" << std::endl;
        continue;
      }
    }
    out->ls();
    auto rootDestination = std::dynamic_pointer_cast<o2::framework::TDirectoryFileOutputStream>(*destination);
  }
  in->Close();
  out->Close();
}

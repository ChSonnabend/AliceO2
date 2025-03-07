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
#include "Framework/ConfigParamSpec.h"
#include "Framework/RawDeviceService.h"

#include <thread>
#include <chrono>
#include <vector>
#include <fairmq/Device.h>

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(
    ConfigParamSpec{"in-dataspec", VariantType::String, "", {"DataSpec for the outputs"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"out-dataspec", VariantType::String, "", {"DataSpec for the outputs"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"eos-dataspec", VariantType::String, "", {"DataSpec for the outputs during EoS"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"processing-delay", VariantType::Int, 0, {"How long the processing takes"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"eos-delay", VariantType::Int, 0, {"How long the takes to do eos"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"name", VariantType::String, "test-processor", {"Name of the processor"}});
}
#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& ctx)
{
  // Get the dataspec option and creates OutputSpecs from it
  auto inDataspec = ctx.options().get<std::string>("in-dataspec");
  auto outDataspec = ctx.options().get<std::string>("out-dataspec");
  // For data created at the End-Of-Stream
  auto eosDataspec = ctx.options().get<std::string>("eos-dataspec");

  auto processingDelay = ctx.options().get<int>("processing-delay");
  auto eosDelay = ctx.options().get<int>("eos-delay");

  std::vector<InputSpec> inputs = select(inDataspec.c_str());

  for (auto& input : inputs) {
    LOGP(info, "{} : lifetime {}", DataSpecUtils::describe(input), (int)input.lifetime);
  }

  std::vector<InputSpec> matchers = select(outDataspec.c_str());
  std::vector<std::string> outputRefs;
  std::vector<OutputSpec> outputs;

  for (auto const& matcher : matchers) {
    outputRefs.emplace_back(matcher.binding);
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  std::vector<InputSpec> eosMatchers = select(eosDataspec.c_str());
  std::vector<std::string> eosRefs;
  std::vector<OutputSpec> eosOutputs;

  for (auto const& matcher : eosMatchers) {
    eosRefs.emplace_back(matcher.binding);
    auto eosOut = DataSpecUtils::asOutputSpec(matcher);
    eosOut.lifetime = Lifetime::Sporadic;
    outputs.emplace_back(eosOut);
  }

  AlgorithmSpec algo = adaptStateful([outputRefs, eosRefs, processingDelay, eosDelay](CallbackService& service) {
    service.set<o2::framework::CallbackService::Id::EndOfStream>([eosRefs, eosDelay](EndOfStreamContext&) {
      LOG(info) << "Creating objects on end of stream reception.";
      std::this_thread::sleep_for(std::chrono::seconds(eosDelay));
    });

    return adaptStateless(
      [outputRefs, processingDelay](InputRecord& inputs, DataAllocator& outputs) {
        LOG(info) << "Received " << inputs.size() << " messages. Converting.";
        auto i = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(processingDelay));
        for (auto& ref : outputRefs) {
          LOGP(info, "Creating {}.", ref);
          outputs.make<int>(ref, ++i);
        }
      });
  });

  return WorkflowSpec{
    {.name = ctx.options().get<std::string>("name"),
     .inputs = inputs,
     .outputs = outputs,
     .algorithm = algo}};
}

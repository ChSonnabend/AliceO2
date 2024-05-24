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

/// \file   MUON/Workflow/src/ChamberEfficiencySpec.cxx
/// \brief  Device that computes the MID chamber efficiency
/// \author Livia Terlizzi <Livia.Terlizzi at cern.ch>
/// \date   20 September 2022

#include "ChamberEfficiencySpec.h"

#include <string>
#include <unordered_map>
#include <gsl/span>

#include "TFile.h"
#include "TH1.h"

#include "Framework/CallbackService.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "DataFormatsMID/Track.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDEfficiency/Efficiency.h"

namespace o2
{
namespace mid
{

using namespace o2::framework;

class EfficiencyTask
{
 public:
  EfficiencyTask(bool selectMatched) : mSelectMatched(selectMatched){};

  std::vector<TH1F> buildHistos()
  {
    Mapping mapping;
    std::vector<TH1F> histos;
    std::array<EffCountType, 4> types{EffCountType::BendPlane, EffCountType::NonBendPlane, EffCountType::BothPlanes, EffCountType::AllTracks};
    std::array<std::string, 4> names{"nFiredBP", "nFiredNBP", "nFiredBoth", "nTot"};
    std::array<Efficiency::ElementType, 3> elTypes{Efficiency::ElementType::Board, Efficiency::ElementType::RPC, Efficiency::ElementType::Plane};
    std::array<std::string, 3> elName{"Board", "RPC", "Plane"};
    std::array<int, 3> nBins{4 * detparams::NLocalBoards, detparams::NDetectionElements, detparams::NChambers};
    std::array<double, 3> firstVal{1., 0., 0.};
    for (size_t iel = 0; iel < elTypes.size(); ++iel) {
      for (size_t itype = 0; itype < types.size(); ++itype) {
        std::string name = names[itype] + "per";
        name += elName[iel];
        histos.emplace_back(name.c_str(), name.c_str(), nBins[iel], firstVal[iel], nBins[iel] + static_cast<int>(firstVal[iel]));
        histos.back().GetXaxis()->SetTitle(elName[iel].c_str());
        for (auto& count : mEfficiency.getChamberEfficiency(elTypes[iel]).getCountersAsVector()) {
          int ibin = count.deId + 1;
          if (elTypes[iel] == Efficiency::ElementType::Board) {
            int ich = detparams::getChamber(count.deId);
            ibin = ich * detparams::NLocalBoards + mapping.getBoardId(count.lineId, count.columnId, count.deId);
          }
          histos.back().SetBinContent(ibin, count.getCounts(types[itype]));
        }
      }
    }
    return histos;
  }

  /// prepare the efficiency
  void init(InitContext& ic)
  {

    // auto config = ic.options().get<std::string>("mid-eff");
    // if (!config.empty()) {
    //   conf::ConfigurableParam::updateFromFile(config, "MIDEff", true);
    // }

    auto stop = [this]() {
      TFile fout("mid-efficiency.root", "RECREATE");
      auto histos = buildHistos();
      for (auto& histo : histos) {
        histo.Write();
      }
      // fout.WriteObject(&mEfficiency.getChamberEfficiency().getCountersAsVector(), "counters");
      fout.Close();
    };
    ic.services().get<o2::framework::CallbackService>().set<o2::framework::CallbackService::Id::Stop>(stop);
  }

  /// run the efficiency
  void run(ProcessingContext& pc)
  {
    auto midTracks = pc.inputs().get<gsl::span<mid::Track>>("midtracks");
    if (mSelectMatched) {
      auto matchTracks = pc.inputs().get<gsl::span<dataformats::TrackMCHMID>>("matchtracks");
      mSelectedTracks.clear();
      mSelectedTracks.reserve(midTracks.size());
      for (auto& matchTrack : matchTracks) {
        auto idx = matchTrack.getMIDRef().getIndex();
        mSelectedTracks.emplace_back(midTracks[idx]);
      }
      midTracks = gsl::span<mid::Track>(mSelectedTracks);
    }

    mEfficiency.process(midTracks);
  }

 private:
  Efficiency mEfficiency{};                  /// Efficiency calculator
  bool mSelectMatched;                       /// Select matched tracks
  std::vector<mid::Track> mSelectedTracks{}; // Vector with matched tracks
};

DataProcessorSpec getChamberEfficiencySpec(bool selectMatched)
{

  Inputs inputSpecs{InputSpec{"midtracks", "MID", "TRACKS", 0, Lifetime::Timeframe}};
  if (selectMatched) {
    inputSpecs.emplace_back(InputSpec{
      "matchtracks", "GLO", "MTC_MCHMID", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "MIDEfficiency",
    inputSpecs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<EfficiencyTask>(selectMatched)},
    Options{
      {"mid-eff", VariantType::String, "mid-efficiency.root", {"Root MID RPCs Efficiency"}}}};
}

} // namespace mid
} // namespace o2
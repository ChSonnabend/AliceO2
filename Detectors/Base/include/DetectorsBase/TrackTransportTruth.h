// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#ifndef O2_TRACK_TRANSPORT_TRUTH_H
#define O2_TRACK_TRANSPORT_TRUTH_H

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace o2::data::detail
{
struct TransportTreeTruth {
  int mother = -1;
  int simulationRoot = -1;
  int hitlessRoot = -1;
  int depth = 0;
  uint32_t hitMask = 0;
  uint64_t descendants = 0;
};

// All tracks must be present, before any truth-based filtering. Input entries
// are independent transport roots, even if they have generator-history mothers.
// A leaf queue avoids both index-order assumptions and recursive stack overflow.
inline std::vector<TransportTreeTruth> labelTransportTrees(
  const std::vector<int>& mothers, const std::vector<uint32_t>& masks, int nInputs)
{
  const auto n = mothers.size();
  if (n > static_cast<size_t>(std::numeric_limits<int>::max()) || masks.size() != n ||
      nInputs < 0 || static_cast<size_t>(nInputs) > n) {
    throw std::runtime_error("Invalid transport-tree sizes");
  }
  std::vector<TransportTreeTruth> truth(n);
  std::vector<int> pending(n, 0), order;
  order.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto& row = truth[i];
    row.mother = i < static_cast<size_t>(nInputs) ? -1 : mothers[i];
    row.hitMask = masks[i];
    if (row.mother < -1 || row.mother >= static_cast<int>(n)) {
      throw std::runtime_error("Invalid transport mother index");
    }
    if (row.mother >= 0) {
      ++pending[row.mother];
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (pending[i] == 0) {
      order.push_back(i);
    }
  }
  for (size_t cursor = 0; cursor < order.size(); ++cursor) {
    const auto& child = truth[order[cursor]];
    if (child.mother >= 0) {
      auto& parent = truth[child.mother];
      parent.hitMask |= child.hitMask;
      parent.descendants += child.descendants + 1;
      if (--pending[child.mother] == 0) {
        order.push_back(child.mother);
      }
    }
  }
  if (order.size() != n) {
    throw std::runtime_error("Cycle in transport ancestry");
  }
  // Reverse the leaf order: parents now precede every descendant.
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int id = *it;
    auto& row = truth[id];
    row.simulationRoot = id < nInputs ? id : (row.mother >= 0 ? truth[row.mother].simulationRoot : -1);
    row.depth = row.mother >= 0 ? truth[row.mother].depth + 1 : 0;
    if (row.hitMask == 0) {
      row.hitlessRoot = row.mother >= 0 && truth[row.mother].hitMask == 0 ? truth[row.mother].hitlessRoot : id;
    }
  }
  return truth;
}

inline std::string quoteTransportCsv(const std::string& value)
{
  std::string result = "\"";
  for (const char c : value) {
    if (c == '"') {
      result += '"';
    }
    result += c;
  }
  return result + '"';
}

inline void writeTransportCsvHeader(std::ostream& out)
{
  out << "csv_schema_version,model,run_prefix,event_id,event_entry,track_id,mother_id,transport_mother_id,"
         "simulation_root_id,depth,is_simulation_root,is_transport_secondary,hitless_tree_root_id,"
         "is_hitless_tree_root,is_hitless_tree_descendant,n_descendants,model_evaluated,valid_score,"
         "score,can_avoid_subtree,own_contribution_mask,subtree_contribution_mask,own_has_contributions,subtree_has_contributions,"
         "parent_can_avoid_subtree,deduplicated_candidate,configured_threshold,would_reject,shadow_mode,"
         "pdg,process,px,py,pz,vx,vy,vz,t_ns,weight,detector_contribution_version,legacy_own_hit_mask,protection_mask,model_would_reject,effective_would_reject\n";
}

// Called once per fully merged event by the single hit-merger process, not by
// simulation workers. MCTrack fields (including scores) already use final IDs.
template <typename Tracks>
void writeTransportCsvEvent(std::ostream& out, const Tracks& tracks, int nInputs,
                            const std::string& model, const std::string& prefix,
                            int eventID, uint64_t eventEntry, double threshold, bool rootsOnly = false)
{
  std::vector<int> mothers;
  std::vector<uint32_t> masks;
  mothers.reserve(tracks.size());
  masks.reserve(tracks.size());
  for (const auto& track : tracks) {
    if (track.getInhibited() || (!track.getTransportModelShadow() && track.getTransportModelScore() != -1.f)) {
      throw std::runtime_error("Transport CSV requires an unpruned shadow event");
    }
    mothers.push_back(track.getMotherTrackId());
    // MCTrack stores a signed 22-bit field; discard sign-extension bits.
    if (track.getDetectorContributionVersion() != 1) {
      throw std::runtime_error("Contribution truth missing; regenerate with instrumented transport");
    }
    masks.push_back(track.getDetectorContributionMask());
  }
  const auto truth = labelTransportTrees(mothers, masks, nInputs);
  const auto modelCsv = quoteTransportCsv(model), prefixCsv = quoteTransportCsv(prefix);
  const bool hasThreshold = std::isfinite(threshold) && threshold >= 0. &&
                            threshold <= std::nextafter(1., std::numeric_limits<double>::infinity());
  // Filter only after full-event truth propagation, preserving descendant hits.
  const size_t nRows = rootsOnly ? static_cast<size_t>(nInputs) : tracks.size();
  for (size_t id = 0; id < nRows; ++id) {
    const auto& track = tracks[id];
    const auto& row = truth[id];
    const bool scored = track.getTransportModelShadow();
    const float score = track.getTransportModelScore();
    const bool valid = scored && std::isfinite(score) && score >= 0.f && score <= 1.f;
    const bool hitless = row.hitMask == 0;
    const bool firstHitless = row.hitlessRoot == static_cast<int>(id);
    const int parentHitless = row.mother >= 0 ? static_cast<int>(truth[row.mother].hitMask == 0) : -1;
    const int reject = scored && hasThreshold ? static_cast<int>(valid && static_cast<double>(score) >= threshold) : -1;
    out << "2," << modelCsv << ',' << prefixCsv << ',' << eventID << ',' << eventEntry << ',' << id << ','
        << mothers[id] << ',' << row.mother << ',' << row.simulationRoot << ',' << row.depth << ','
        << (id < static_cast<size_t>(nInputs)) << ',' << (id >= static_cast<size_t>(nInputs)) << ','
        << row.hitlessRoot << ',' << firstHitless << ',' << (hitless && !firstHitless) << ','
        << row.descendants << ',' << scored << ',' << valid << ','
        << std::setprecision(17) << static_cast<double>(score) << ',' << hitless << ','
        << masks[id] << ',' << row.hitMask << ',' << (masks[id] != 0) << ',' << !hitless << ','
        << parentHitless << ',' << (scored && (!hitless || firstHitless)) << ','
        << (hasThreshold ? threshold : -1.) << ',' << reject << ",1,"
        << track.GetPdgCode() << ',' << track.getProcess() << ',' << std::setprecision(9)
        << track.Px() << ',' << track.Py() << ',' << track.Pz() << ','
        << track.Vx() << ',' << track.Vy() << ',' << track.Vz() << ',' << track.T() << ',' << track.getWeight() << ','
        << static_cast<int>(track.getDetectorContributionVersion()) << ','
        << (static_cast<uint32_t>(track.getHitMask()) & ((uint32_t{1} << 22) - 1)) << ','
        << static_cast<int>(track.getTransportModelProtection()) << ',' << reject << ','
        << (reject < 0 ? -1 : static_cast<int>(reject == 1 && track.getTransportModelProtection() == 0)) << '\n';
  }
  if (!out) {
    throw std::runtime_error("Failed to write transport-score CSV event");
  }
}
} // namespace o2::data::detail
#endif

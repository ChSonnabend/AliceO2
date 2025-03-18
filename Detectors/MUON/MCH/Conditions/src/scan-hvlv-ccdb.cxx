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

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <fmt/format.h>

#include "TFile.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TLine.h"
#include "TMultiGraph.h"
#include "TStyle.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "MCHConditions/DCSAliases.h"
#include "MCHStatus/HVStatusCreator.h"
#include "MCHStatus/StatusMapCreatorParam.h"

namespace po = boost::program_options;

using namespace o2;
using DPID = dcs::DataPointIdentifier;
using DPVAL = dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;
using DPMAP2 = std::map<std::string, std::map<uint64_t, double>>;
using RBMAP = std::map<int, std::pair<uint64_t, uint64_t>>;
using DPBMAP = std::map<uint64_t, uint64_t>;
using ISSUE = std::tuple<uint64_t, uint64_t, double, double, std::string>;
using ISSUELIST = std::vector<ISSUE>;
using ISSUEMAP = std::map<std::string, ISSUELIST>;
using ULL = unsigned long long;

//----------------------------------------------------------------------------
bool containsAKey(std::string data, const std::set<std::string>& Keys)
{
  /// check if the data contains one of the keys

  auto itKey = std::find_if(Keys.begin(), Keys.end(), [&data](const auto& key) {
    return data.find(key) != data.npos;
  });

  return itKey != Keys.end();
}

//----------------------------------------------------------------------------
bool isValid(std::string alias)
{
  /// check if the alias is a valid (part of a) DCS alias

  static const std::vector<std::string> aliases =
    mch::dcs::aliases({mch::dcs::MeasurementType::HV_V,
                       mch::dcs::MeasurementType::LV_V_FEE_ANALOG,
                       mch::dcs::MeasurementType::LV_V_FEE_DIGITAL,
                       mch::dcs::MeasurementType::LV_V_SOLAR});

  auto itAlias = std::find_if(aliases.begin(), aliases.end(), [&alias](const auto& a) {
    return a.find(alias) != a.npos;
  });

  return itAlias != aliases.end();
}

//----------------------------------------------------------------------------
void scanWhat(std::string what, std::string& path, bool& scanHV, bool& scanAll, std::set<std::string>& aliases)
{
  /// get what to scan and where

  static const std::set<std::string> hvKeys{"HV", "Quad", "Slat"};
  static const std::set<std::string> lvKeys{"LV", "Group", "an", "di", "Sol"};

  // HV or LV ?
  path = "";
  scanHV = false;
  if (containsAKey(what, hvKeys)) {
    path = "MCH/Calib/HV";
    scanHV = true;
  }
  if (containsAKey(what, lvKeys)) {
    if (scanHV) {
      printf("error: cannot scan HV and LV channels at the same time\n");
      exit(1);
    }
    path = "MCH/Calib/LV";
  }
  if (path.empty()) {
    printf("error: no valid HV or LV channel to scan\n");
    exit(1);
  }

  // everything or specific aliases ?
  if (what.find(scanHV ? "HV" : "LV") != what.npos) {
    scanAll = true;
    aliases.clear();
  } else {
    scanAll = false;
    std::istringstream input(what);
    for (std::string alias; std::getline(input, alias, ',');) {
      if (isValid(alias)) {
        aliases.insert(alias);
      } else {
        printf("error: \"%s\" invalid (part of) HV or LV alias\n", alias.c_str());
        exit(1);
      }
    }
  }
}

//----------------------------------------------------------------------------
uint64_t ms2s(uint64_t ts)
{
  /// convert the time stamp from ms to s

  return (ts + 500) / 1000;
}

//----------------------------------------------------------------------------
std::string getTime(uint64_t ts)
{
  /// convert the time stamp (ms) to local time

  time_t t = ms2s(ts);

  std::string time = std::ctime(&t);
  time.pop_back(); // remove trailing \n

  return time;
}

//----------------------------------------------------------------------------
std::string getDuration(uint64_t tStart, uint64_t tStop)
{
  /// get the duration (dd hh:mm:ss) between the two time stamps (ms)

  auto dt = ms2s(tStop - tStart);
  auto s = dt % 60;
  auto m = (dt / 60) % 60;
  auto h = (dt / 3600) % 24;
  auto d = dt / 86400;

  return fmt::format("{:02}d {:02}:{:02}:{:02}", d, h, m, s);
}

//----------------------------------------------------------------------------
std::set<int> getRuns(std::string runList)
{
  /// read the runList from an ASCII file, or a comma separated run list, or a single run

  std::set<int> runs{};

  auto isNumber = [](std::string val) { return !val.empty() && val.find_first_not_of("0123456789") == val.npos; };

  if (isNumber(runList)) {

    runs.insert(std::stoi(runList));

  } else if (runList.find(",") != runList.npos) {

    std::istringstream input(runList);
    for (std::string run; std::getline(input, run, ',');) {
      if (isNumber(run)) {
        runs.insert(std::stoi(run));
      }
    }

  } else {

    std::ifstream input(runList);
    if (input.is_open()) {
      for (std::string run; std::getline(input, run);) {
        if (isNumber(run)) {
          runs.insert(std::stoi(run));
        }
      }
    }
  }

  return runs;
}

//----------------------------------------------------------------------------
RBMAP getRunBoundaries(ccdb::CcdbApi const& api, std::string runList)
{
  /// return the SOR / EOR time stamps for every runs in the list

  RBMAP runBoundaries{};

  auto runs = getRuns(runList);

  for (auto run : runs) {
    auto boundaries = ccdb::CCDBManagerInstance::getRunDuration(api, run);
    runBoundaries.emplace(run, boundaries);
  }

  return runBoundaries;
}

//----------------------------------------------------------------------------
void checkRunBoundaries(const RBMAP& runBoundaries)
{
  /// check the consistency of the run time boundaries

  if (runBoundaries.empty()) {
    printf("error: no run found from the list\n");
    exit(1);
  }

  bool error = false;
  int previousRun = 0;
  uint64_t endOfPreviousRun = 0;

  for (const auto& [run, boundaries] : runBoundaries) {
    if (boundaries.second <= boundaries.first) {
      printf("error: run %d EOR <= SOR: %llu - %llu (%s - %s)\n",
             run, (ULL)boundaries.first, (ULL)boundaries.second,
             getTime(boundaries.first).c_str(), getTime(boundaries.second).c_str());
      error = true;
    }
    if (boundaries.first <= endOfPreviousRun) {
      printf("error: SOR run %d <= EOR run %d: %llu (%s) <= %llu (%s)\n",
             run, previousRun, (ULL)boundaries.first, getTime(boundaries.first).c_str(),
             (ULL)endOfPreviousRun, getTime(endOfPreviousRun).c_str());
      error = true;
    }
    previousRun = run;
    endOfPreviousRun = boundaries.second;
  }

  if (error) {
    exit(1);
  }
}

//----------------------------------------------------------------------------
void printRunBoundaries(const RBMAP& runBoundaries)
{
  /// print the list of runs with their time boundaries

  printf("\nlist of runs with their boundaries:\n");
  printf("------------------------------------\n");

  for (const auto& [run, boundaries] : runBoundaries) {
    printf("%d: %llu - %llu (%s - %s)\n", run, (ULL)boundaries.first, (ULL)boundaries.second,
           getTime(boundaries.first).c_str(), getTime(boundaries.second).c_str());
  }

  printf("------------------------------------\n");
}

//----------------------------------------------------------------------------
void drawRunBoudaries(const RBMAP& runBoundaries, TCanvas* c)
{
  /// draw the run time boundaries

  c->cd();

  for (const auto& [run, boundaries] : runBoundaries) {

    TLine* startRunLine = new TLine(ms2s(boundaries.first), c->GetUymin(), ms2s(boundaries.first), c->GetUymax());
    startRunLine->SetUniqueID(run);
    startRunLine->SetLineColor(4);
    startRunLine->SetLineWidth(1);
    startRunLine->Draw();

    TLine* endRunLine = new TLine(ms2s(boundaries.second), c->GetUymin(), ms2s(boundaries.second), c->GetUymax());
    endRunLine->SetUniqueID(run);
    endRunLine->SetLineColor(2);
    endRunLine->SetLineWidth(1);
    endRunLine->Draw();
  }
}

//----------------------------------------------------------------------------
DPBMAP getDPBoundaries(ccdb::CcdbApi const& api, std::string what,
                       uint64_t tStart, uint64_t tStop, uint64_t timeInterval)
{
  /// get the time boundaries of every HV/LV files found in the time range

  // add an extra margin (ms) of ± 1 min to the creation time,
  // which corresponds to the end of the time interval covered by the file
  static const uint64_t timeMarging = 60000;

  std::istringstream fileInfo(api.list(what.c_str(), false, "text/plain",
                                       tStop + timeInterval + timeMarging, tStart - timeMarging));

  DPBMAP dpBoundaries{};
  std::string dummy{};
  uint64_t begin = 0;
  uint64_t end = 0;

  for (std::string line; std::getline(fileInfo, line);) {
    if (line.find("Validity:") == 0) {
      std::istringstream in(line);
      in >> dummy >> begin >> dummy >> end;
      dpBoundaries.emplace(begin, end);
    }
  }

  if (dpBoundaries.empty()) {
    printf("\e[0;31merror: no file found in %s in time range %llu - %llu (%s - %s) --> use the default one\e[0m\n",
           what.c_str(), (ULL)tStart, (ULL)tStop, getTime(tStart).c_str(), getTime(tStop).c_str());
    dpBoundaries.emplace(1, 9999999999999);
  }

  return dpBoundaries;
}

//----------------------------------------------------------------------------
void checkDPBoundaries(const DPBMAP& dpBoundaries, bool scanHV, uint64_t tStart, uint64_t tStop)
{
  /// check the consistency of HV/LV file time boundaries

  bool error = false;

  if (dpBoundaries.begin()->first > tStart) {
    printf("error: the beginning of the time range is not covered: %llu > %llu (%s > %s)\n",
           (ULL)dpBoundaries.begin()->first, (ULL)tStart,
           getTime(dpBoundaries.begin()->first).c_str(), getTime(tStart).c_str());
    error = true;
  }
  if (dpBoundaries.rbegin()->second < tStop) {
    printf("error: the end of the time range is not covered: %llu < %llu (%s < %s)\n",
           (ULL)dpBoundaries.rbegin()->second, (ULL)tStop,
           getTime(dpBoundaries.rbegin()->second).c_str(), getTime(tStop).c_str());
    error = true;
  }

  uint64_t previousTStop = dpBoundaries.begin()->first;
  for (auto [tStart, tStop] : dpBoundaries) {
    if (tStop <= tStart) {
      printf("error: EOF <= SOF: %llu - %llu (%s - %s)\n",
             (ULL)tStart, (ULL)tStop, getTime(tStart).c_str(), getTime(tStop).c_str());
      error = true;
    }
    if (tStart != previousTStop) {
      printf("error: end of %s file != start of next %s file: %llu (%s) != %llu (%s))\n",
             scanHV ? "HV" : "LV", scanHV ? "HV" : "LV",
             (ULL)previousTStop, getTime(previousTStop).c_str(), (ULL)tStart, getTime(tStart).c_str());
      error = true;
    }
    previousTStop = tStop;
  }

  if (error) {
    exit(1);
  }
}

//----------------------------------------------------------------------------
void printDPBoundaries(const DPBMAP& dpBoundaries, bool scanHV, uint64_t timeInterval)
{
  /// print the time boundaries of every HV/LV files found in the full time range

  printf("\nlist of %s file time boundaries:\n", scanHV ? "HV" : "LV");
  printf("------------------------------------\n");

  for (auto [tStart, tStop] : dpBoundaries) {
    printf("%llu - %llu (%s - %s)", (ULL)tStart, (ULL)tStop, getTime(tStart).c_str(), getTime(tStop).c_str());
    if (tStop - tStart < 60000 * (timeInterval - 1) || tStop - tStart > 60000 * (timeInterval + 1)) {
      printf("\e[0;31m ! warning: validity range %s != %llu±1 min\e[0m\n",
             getDuration(tStart, tStop).c_str(), (ULL)timeInterval);
    } else {
      printf("\n");
    }
  }

  printf("------------------------------------\n");
}

//----------------------------------------------------------------------------
double getLVLimit(std::string alias)
{
  /// return the LV limit for that channel

  static const double lvLimits[3] = {1.5, 1.5, 6.}; // FeeAnalog, FeeDigital, Solar

  if (alias.find("an") != alias.npos) {
    return lvLimits[0];
  } else if (alias.find("di") != alias.npos) {
    return lvLimits[1];
  }
  return lvLimits[2];
}

//----------------------------------------------------------------------------
void drawLimit(double limit, TCanvas* c)
{
  /// draw the HV/LV limit for the displayed chamber

  c->cd();

  TLine* l = new TLine(c->GetUxmin(), limit, c->GetUxmax(), limit);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(2);
  l->Draw();
}

//----------------------------------------------------------------------------
double getValue(DPVAL dp)
{
  /// return the value of this data point

  union Converter {
    uint64_t raw_data;
    double value;
  } converter;

  converter.raw_data = dp.payload_pt1;

  return converter.value;
}

//----------------------------------------------------------------------------
std::string getDE(std::string alias)
{
  /// for DCS HV alias: return the corresponding DE (and sector)
  /// for DCS LV alias: return an empty string

  auto de = mch::dcs::aliasToDetElemId(alias);

  if (de) {
    return (mch::dcs::isQuadrant(mch::dcs::aliasToChamber(alias)))
             ? fmt::format("DE{}-{}", *de, mch::dcs::aliasToNumber(alias) % 10)
             : fmt::format("DE{}", *de);
  }

  return "";
}

//----------------------------------------------------------------------------
void fillDataPoints(const std::vector<DPVAL>& dps, std::map<uint64_t, double>& dps2,
                    uint64_t tMin, uint64_t tMax, int warningLevel)
{
  /// fill the map of data points

  static const uint64_t tolerance = 5000;

  if (dps.empty()) {
    printf("error: the file does not contain any data point\n");
    exit(1);
  }

  auto itDP = dps.begin();
  auto ts = itDP->get_epoch_time();
  std::string header = "warning:";
  std::string color = (ts + tolerance < tMin || ts > tMin + tolerance) ? "\e[0;31m" : "\e[0;34m";
  bool printWarning = warningLevel > 1 || (warningLevel == 1 && color == "\e[0;31m");

  // check if the first data point is a copy of the last one from previous file
  if (!dps2.empty()) {
    auto previousTS = dps2.rbegin()->first;
    if (ts != previousTS || getValue(*itDP) != dps2.rbegin()->second) {
      if (ts <= previousTS) {
        printf("error: wrong data point order (%llu <= %llu)\n", (ULL)ts, (ULL)previousTS);
        exit(1);
      }
      if (printWarning) {
        printf("%s%s missing the previous data point (dt = %s%llu ms)", color.c_str(), header.c_str(),
               (previousTS < tMin) ? "-" : "+",
               (ULL)((previousTS < tMin) ? tMin - previousTS : previousTS - tMin));
        if (ts <= tMin) {
          printf(" but get one at dt = -%llu ms\e[0m\n", (ULL)(tMin - ts));
        } else {
          printf("\e[0m\n");
        }
        header = "        ";
      }
    }
  }

  // add the first data point (should be before the start of validity of the file)
  if (ts >= tMax) {
    printf("error: first data point exceeding file validity range (dt = +%llu ms)\n", (ULL)(ts - tMax));
    exit(1);
  } else if (ts > tMin && printWarning) {
    printf("%s%s missing data point prior file start of validity (dt = +%llu ms)\e[0m\n",
           color.c_str(), header.c_str(), (ULL)(ts - tMin));
    header = "        ";
  }
  dps2.emplace(ts, getValue(*itDP));

  // add other data points (should be within the validity range of the file)
  auto previousTS = ts;
  for (++itDP; itDP < dps.end(); ++itDP) {
    ts = itDP->get_epoch_time();
    if (ts <= previousTS) {
      printf("error: wrong data point order (%llu <= %llu)\n", (ULL)ts, (ULL)previousTS);
      exit(1);
    }
    if (ts < tMin && (warningLevel > 1 || (warningLevel == 1 && ts + tolerance < tMin))) {
      printf("%s%s data point outside of file validity range (dt = -%llu ms)\e[0m\n",
             (ts + tolerance < tMin) ? "\e[0;31m" : "\e[0;34m", header.c_str(), (ULL)(tMin - ts));
    } else if (ts >= tMax && warningLevel >= 1) {
      printf("\e[0;31m%s data point outside of file validity range (dt = +%llu ms)\e[0m\n",
             header.c_str(), (ULL)(ts - tMax));
    }
    dps2.emplace(ts, getValue(*itDP));
    previousTS = ts;
  }
}

//----------------------------------------------------------------------------
void selectDataPoints(DPMAP2 dpsMapsPerCh[10], uint64_t tStart, uint64_t tStop)
{
  /// remove the data points outside of the given time range and, if needed,
  /// add a data point at the boundaries with value equal to the preceding one

  for (int ch = 0; ch < 10; ++ch) {
    for (auto& [alias, dps] : dpsMapsPerCh[ch]) {

      // get the first data point in the time range, remove the previous ones
      // and add a data point with value equal to the preceding one if it exits
      // or to this one otherwise
      auto itFirst = dps.lower_bound(tStart);
      if (itFirst != dps.begin()) {
        double previousVal = std::prev(itFirst)->second;
        for (auto it = dps.begin(); it != itFirst;) {
          it = dps.erase(it);
        }
        dps.emplace(tStart, previousVal);
      } else if (itFirst->first != tStart) {
        if (itFirst->first > tStop) {
          printf("error (%s): all data points are posterior to the end of the time range\n", alias.c_str());
        } else {
          printf("error (%s): first data point is posterior to the beginning of the time range\n", alias.c_str());
        }
        dps.emplace(tStart, itFirst->second);
      }

      // get the first data point exceeding the time range, remove it and the next ones
      // and add a data point with value equal to the preceding one if needed
      auto itLast = dps.upper_bound(tStop);
      double previousVal = std::prev(itLast)->second;
      for (auto it = itLast; it != dps.end();) {
        it = dps.erase(it);
      }
      dps.emplace(tStop, previousVal);
    }
  }
}

//----------------------------------------------------------------------------
void printDataPoints(const DPMAP2 dpsMapsPerCh[10], std::string hvlvFormat, bool all)
{
  /// print all the registered data points

  const auto format1 = fmt::format("  %llu (%s): {} V\n", hvlvFormat.c_str());
  const auto format2 = fmt::format(": %llu (%s): {} V -- %llu (%s): {} V\n",
                                   hvlvFormat.c_str(), hvlvFormat.c_str());

  for (int ch = 0; ch < 10; ++ch) {

    printf("\n------------ chamber %d ------------\n", ch + 1);

    for (const auto& [alias, dps] : dpsMapsPerCh[ch]) {

      printf("- %s: %zu values", alias.c_str(), dps.size());

      if (all) {

        printf("\n");
        for (const auto& [ts, val] : dps) {
          printf(format1.c_str(), (ULL)ts, getTime(ts).c_str(), val);
        }

      } else if (!dps.empty()) {

        const auto firstdt = dps.begin();
        const auto lastdt = dps.rbegin();
        printf(format2.c_str(),
               (ULL)firstdt->first, getTime(firstdt->first).c_str(), firstdt->second,
               (ULL)lastdt->first, getTime(lastdt->first).c_str(), lastdt->second);

      } else {
        printf("\n");
      }
    }
  }
}

//----------------------------------------------------------------------------
TGraph* mapToGraph(std::string alias, const std::map<uint64_t, double>& dps)
{
  /// create a graph for the DCS channel and add the data points

  TGraph* g = new TGraph(dps.size());

  auto pos = alias.find(".");
  auto shortAlias = alias.substr(0, pos);
  auto de = getDE(alias);
  auto title = de.empty() ? fmt::format("{}", shortAlias.c_str())
                          : fmt::format("{} ({})", de.c_str(), shortAlias.c_str());
  g->SetNameTitle(alias.c_str(), title.c_str());

  int i(0);
  for (auto [ts, val] : dps) {
    g->SetPoint(i, ms2s(ts), val);
    ++i;
  }

  g->SetMarkerSize(1.5);
  g->SetMarkerStyle(2);
  g->SetLineStyle(2);

  return g;
}

//----------------------------------------------------------------------------
TCanvas* drawDataPoints(TMultiGraph* mg, double min, double max)
{
  /// display the data points of the given chamber

  TCanvas* c = new TCanvas(mg->GetName(), mg->GetHistogram()->GetTitle(), 1500, 900);

  mg->Draw("A plc pmc");
  mg->SetMinimum(min);
  mg->SetMaximum(max);
  mg->GetXaxis()->SetTimeDisplay(1);
  mg->GetXaxis()->SetTimeFormat("%d/%m %H:%M");
  mg->GetXaxis()->SetTimeOffset(0, "local");
  mg->GetXaxis()->SetNdivisions(21010);

  c->BuildLegend();
  c->Update();

  return c;
}

//----------------------------------------------------------------------------
void findIssues(const std::map<uint64_t, double>& dps, double limit, ISSUELIST& issues)
{
  /// return the list of HV/LV issues (time range, min value, mean value) for each DCS channel

  uint64_t tStart(0);
  double min(0.);
  double mean(0.);
  uint64_t prevTS(0);
  double prevVal(-1.);

  for (auto [ts, val] : dps) {

    if (val < limit) {

      if (tStart == 0) {

        // start a new issue...
        tStart = ts;
        min = val;
        mean = 0.;
        prevTS = ts;
        prevVal = val;

      } else {

        // ... or complement the current one
        min = std::min(min, val);
        mean += prevVal * (ts - prevTS);
        prevTS = ts;
        prevVal = val;
      }

    } else if (tStart > 0) {

      // complete the current issue, if any, and register it
      mean += prevVal * (ts - prevTS);
      mean /= (ts - tStart);
      issues.emplace_back(tStart, ts, min, mean, "");
      tStart = 0;
    }
  }

  // complete the last issue, if any and its duration is != 0, and register it
  if (tStart > 0 && prevTS != tStart) {
    mean /= (prevTS - tStart);
    issues.emplace_back(tStart, prevTS, min, mean, "");
  }
}

//----------------------------------------------------------------------------
void fillO2Issues(const std::vector<mch::HVStatusCreator::TimeRange>& o2issues, ISSUELIST& issues,
                  uint64_t tMin, uint64_t tMax)
{
  /// fill the list of issues from O2 (extend the previous one and/or create new ones)

  // the list must not be empty
  if (o2issues.empty()) {
    printf("error: O2 returns an empty list of issues\n");
    exit(1);
  }

  for (auto itIssue = o2issues.begin(); itIssue != o2issues.end(); ++itIssue) {

    // exclude issues fully outside of the DP file boudaries
    if (itIssue->end <= tMin || itIssue->begin >= tMax) {
      printf("\e[0;35mwarning: skipping O2 issue outside of file boundaries (%llu - %llu)\e[0m\n",
             (ULL)itIssue->begin, (ULL)itIssue->end);
      continue;
    }

    // only the first issue could in principle extend before the start of the DP file, to O
    if (itIssue->begin < tMin - mch::StatusMapCreatorParam::Instance().timeMargin &&
        (itIssue != o2issues.begin() || itIssue->begin != 0)) {
      printf("\e[0;35mwarning: O2 returns an issue with uncommon start time (%llu < %llu)\e[0m\n",
             (ULL)itIssue->begin, (ULL)(tMin - mch::StatusMapCreatorParam::Instance().timeMargin));
    }

    // only the last issue could in principle extend beyond the end of the DP file, to infinity
    if (itIssue->end >= tMax + mch::StatusMapCreatorParam::Instance().timeMargin &&
        (itIssue != std::prev(o2issues.end()) || itIssue->end != std::numeric_limits<uint64_t>::max())) {
      printf("\e[0;35mwarning: O2 returns an issue with uncommon end time (%llu >= %llu)\e[0m\n",
             (ULL)itIssue->end, (ULL)(tMax + mch::StatusMapCreatorParam::Instance().timeMargin));
    }

    // extend the last issue in case of continuity accross the DP files or add a new one,
    // restricting their time range within the DP file boundaries
    if (itIssue->begin <= tMin && !issues.empty() && std::get<1>(issues.back()) == tMin) {
      std::get<1>(issues.back()) = std::min(itIssue->end, tMax);
    } else {
      issues.emplace_back(std::max(itIssue->begin, tMin), std::min(itIssue->end, tMax), 0., 0., "");
    }
  }
}

//----------------------------------------------------------------------------
std::string findAffectedRuns(const RBMAP& runBoundaries, uint64_t tStart, uint64_t tStop)
{
  /// return the list of affected runs in this time range

  std::string runs;

  for (const auto& [run, boundaries] : runBoundaries) {

    if (boundaries.second <= tStart) {
      continue;
    } else if (boundaries.first >= tStop) {
      break;
    }

    runs += fmt::format("{},", run);
  }

  if (!runs.empty()) {
    runs.pop_back();
  }

  return runs;
}

//----------------------------------------------------------------------------
void selectIssues(ISSUEMAP issuesPerCh[10], const RBMAP& runBoundaries, uint64_t minDuration)
{
  /// select HV/LV issues of a minimum duration (ms) occurring during runs

  for (int ch = 0; ch < 10; ++ch) {
    for (auto& issues : issuesPerCh[ch]) {
      for (auto itIssue = issues.second.begin(); itIssue != issues.second.end();) {

        auto tStart = std::get<0>(*itIssue);
        auto tStop = std::get<1>(*itIssue);

        if (tStop - tStart < minDuration) {

          itIssue = issues.second.erase(itIssue);

        } else {

          auto runs = findAffectedRuns(runBoundaries, tStart, tStop);

          if (runs.empty()) {

            itIssue = issues.second.erase(itIssue);

          } else {

            std::get<4>(*itIssue) = runs;
            ++itIssue;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
void selectO2Issues(ISSUEMAP issuesPerCh[10], const RBMAP& runBoundaries)
{
  /// select HV issues from O2 algorithm occurring during runs
  /// and restrict the range of issues to the run range

  for (int ch = 0; ch < 10; ++ch) {
    for (auto& issues : issuesPerCh[ch]) {
      for (auto itIssue = issues.second.begin(); itIssue != issues.second.end();) {

        auto& tStart = std::get<0>(*itIssue);
        auto& tStop = std::get<1>(*itIssue);

        auto runs = findAffectedRuns(runBoundaries, tStart, tStop);

        if (runs.empty()) {

          itIssue = issues.second.erase(itIssue);

        } else {

          tStart = std::max(tStart, runBoundaries.begin()->second.first);
          tStop = std::min(tStop, runBoundaries.rbegin()->second.second);
          std::get<4>(*itIssue) = runs;
          ++itIssue;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
bool eraseIssue(const ISSUE& issue, ISSUELIST& issues)
{
  /// find an issue with the same time range and associated run list and erase it
  /// return true in case of success

  auto itIssue = std::find_if(issues.begin(), issues.end(), [&issue](const auto& i) {
    return (std::get<0>(i) == std::get<0>(issue) &&
            std::get<1>(i) == std::get<1>(issue) &&
            std::get<4>(i) == std::get<4>(issue));
  });

  if (itIssue != issues.end()) {
    issues.erase(itIssue);
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
void printIssues(const ISSUEMAP issuesPerCh[10], const ISSUEMAP o2IssuesPerCh[10],
                 bool scanHV, std::string hvlvFormat)
{
  /// print all HV/LV issues

  // copy the issues so that we can modify them (i.e. add empty lists or delete issues after printing)
  ISSUEMAP issuesPerChCopy[10];
  ISSUEMAP o2IssuesPerChCopy[10];
  for (int ch = 0; ch < 10; ++ch) {
    issuesPerChCopy[ch] = issuesPerCh[ch];
    o2IssuesPerChCopy[ch] = o2IssuesPerCh[ch];
  }

  // make sure that all alias keys in the map o2IssuesPerChCopy are also in issuesPerChCopy in order to
  // simplify the loop over all issues from both algorithms and fix the order in which they are printed
  for (int ch = 0; ch < 10; ++ch) {
    for (const auto& [alias, o2Issues] : o2IssuesPerChCopy[ch]) {
      if (!o2Issues.empty()) {
        issuesPerChCopy[ch].try_emplace(alias, ISSUELIST{});
      }
    }
  }

  auto printHeader = [](std::string alias) {
    auto de = getDE(alias);
    if (de.empty()) {
      printf("Problem found for %s:\n", alias.c_str());
    } else {
      printf("Problem found for %s (%s):\n", alias.c_str(), de.c_str());
    }
  };

  const auto format = fmt::format("%llu - %llu: %s (duration = %s, min = {} V, mean = {} V) --> run(s) %s\n",
                                  hvlvFormat.c_str(), hvlvFormat.c_str());

  auto printIssue = [&format](ISSUE issue, std::string color) {
    const auto& [tStart, tStop, min, mean, runs] = issue;
    printf("%s", color.c_str());
    printf(format.c_str(), (ULL)tStart, (ULL)tStop,
           getTime(tStart).c_str(), getDuration(tStart, tStop).c_str(), min, mean, runs.c_str());
    printf("\e[0m");
  };

  if (scanHV) {
    printf("\n------ list of issues from \e[0;31mthis macro only\e[0m, \e[0;35mO2 only\e[0m, or \e[0;32mboth\e[0m ------\n");
  } else {
    printf("\n------ list of issues ------\n");
  }

  bool foundIssues = false;

  for (int ch = 0; ch < 10; ++ch) {
    for (const auto& [alias, issues] : issuesPerChCopy[ch]) {

      auto& o2Issues = o2IssuesPerChCopy[ch][alias];

      if (!issues.empty() || !o2Issues.empty()) {

        foundIssues = true;
        printHeader(alias);

        // print all issues found by this macro
        for (const auto& issue : issues) {
          // change color if the issue is not found by the O2 algorithm (only for HV)
          std::string color = (scanHV && !eraseIssue(issue, o2Issues)) ? "\e[0;31m" : "\e[0;32m";
          printIssue(issue, color);
        }

        // print other issues found by the O2 algorithm
        for (const auto& issue : o2Issues) {
          printIssue(issue, "\e[0;35m");
        }

        printf("----------------------------\n");
      }
    }
  }

  if (!foundIssues) {
    printf("----------------------------\n");
  }
}

//----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  /// scan HV or LV CCDB objects looking for issues

  std::string runList = "";
  std::string what = "";
  std::string config = "";
  uint64_t minDuration = 0;
  uint64_t timeInterval = 30;
  int warningLevel = 1;
  int printLevel = 1;
  std::string outFileName = "";

  po::options_description usage("Usage");
  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("runs,r",po::value<std::string>(&runList)->default_value(""),"run(s) to scan (comma separated list of runs or ASCII file with one run per line)")
      ("channels,c",po::value<std::string>(&what)->default_value(""),R"(channel(s) to scan ("HV" or "LV" or comma separated list of (part of) DCS aliases))")
      ("configKeyValues",po::value<std::string>(&config)->default_value(""),"Semicolon separated key=value strings to change HV thresholds")
      ("duration,d",po::value<uint64_t>(&minDuration)->default_value(0),"minimum duration (ms) of HV/LV issues to consider")
      ("interval,i",po::value<uint64_t>(&timeInterval)->default_value(30),"creation time interval (minutes) between CCDB files")
      ("warning,w",po::value<int>(&warningLevel)->default_value(1),"warning level (0, 1 or 2)")
      ("print,p",po::value<int>(&printLevel)->default_value(1),"print level (0, 1, 2 or 3)")
      ("output,o",po::value<std::string>(&outFileName)->default_value("scan.root"),"output root file name")
        ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program scans HV or LV channels looking for issues\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (const po::error& e) {
    std::cout << "error: " << e.what() << "\n";
    exit(1);
  }

  if (runList.empty()) {
    printf("error: you must provide run(s) to scan\n");
    exit(1);
  }

  if (what.empty()) {
    printf("error: you must provide channel(s) to scan\n");
    exit(1);
  }

  // setup printout and display
  const double hvRange[2] = {-10., 1700.};
  const double lvRange[3] = {-1., 4., 8.}; // min, max FeeAnalog/FeeDigital, max Solar
  const std::string hvFormat = "%7.2f";
  const std::string lvFormat = "%4.2f";
  gStyle->SetPalette(kVisibleSpectrum);

  // setup algorithms searching for HV issues
  conf::ConfigurableParam::updateFromString(config);
  conf::ConfigurableParam::setValue("MCHStatusMap.hvMinDuration", std::to_string(minDuration));
  conf::ConfigurableParam::setValue("MCHStatusMap.timeMargin", "0"); // must be 0 to compare O2 with this scan

  // determine what is scanned
  std::string path{};
  bool scanHV = false;
  bool scanAll = false;
  std::set<std::string> aliases{};
  scanWhat(what, path, scanHV, scanAll, aliases);

  ccdb::CcdbApi api;
  api.init("http://alice-ccdb.cern.ch");

  // get the SOR/EOR of every runs from the list, ordered in run number
  auto runBoundaries = getRunBoundaries(api, runList);
  if (printLevel > 0) {
    printRunBoundaries(runBoundaries);
  }
  checkRunBoundaries(runBoundaries);

  // extract the time boundaries for each HV/LV file in the full time range
  auto dpBoundaries = getDPBoundaries(api, path.c_str(), runBoundaries.begin()->second.first,
                                      runBoundaries.rbegin()->second.second, timeInterval * 60000);
  if (printLevel > 0) {
    printDPBoundaries(dpBoundaries, scanHV, timeInterval);
  }
  checkDPBoundaries(dpBoundaries, scanHV, runBoundaries.begin()->second.first,
                    runBoundaries.rbegin()->second.second);

  // loop over the HV/LV files, fill the lists of data points per chamber and find issues using O2 algorithm
  DPMAP2 dpsMapsPerCh[10];
  mch::HVStatusCreator hvStatusCreator{};
  ISSUEMAP o2issuesPerCh[10];
  std::map<std::string, std::string> metadata;
  for (auto boundaries : dpBoundaries) {

    auto* dpMap = api.retrieveFromTFileAny<DPMAP>(path.c_str(), metadata, boundaries.first);

    // fill the lists of data points per chamber for requested aliases
    for (const auto& [dpid, dps] : *dpMap) {
      std::string alias(dpid.get_alias());
      if (!mch::dcs::isValid(alias)) {
        printf("error: invalid DCS alias: %s\n", alias.c_str());
        exit(1);
      }
      if ((scanAll || containsAKey(alias, aliases)) && (!scanHV || alias.find(".iMon") == alias.npos)) {
        int chamber = mch::dcs::toInt(mch::dcs::aliasToChamber(alias));
        fillDataPoints(dps, dpsMapsPerCh[chamber][alias], boundaries.first, boundaries.second, warningLevel);
      }
    }

    // find issues for requested aliases using O2 algorithm (only for HV)
    if (scanHV) {
      hvStatusCreator.findBadHVs(*dpMap);
      for (const auto& [alias, issues] : hvStatusCreator.getBadHVs()) {
        if (scanAll || containsAKey(alias, aliases)) {
          int chamber = mch::dcs::toInt(mch::dcs::aliasToChamber(alias));
          fillO2Issues(issues, o2issuesPerCh[chamber][alias], boundaries.first, boundaries.second);
        }
      }
    }
  }
  if (printLevel > 1) {
    printf("\nall data points:");
    printDataPoints(dpsMapsPerCh, scanHV ? hvFormat : lvFormat, printLevel > 2);
  }

  // select the data points in the time range
  selectDataPoints(dpsMapsPerCh, runBoundaries.begin()->second.first, runBoundaries.rbegin()->second.second);
  if (printLevel > 1) {
    printf("\ndata points in the time range covered by runs:");
    printDataPoints(dpsMapsPerCh, scanHV ? hvFormat : lvFormat, printLevel > 2);
  }

  // create and fill the graphs, and find HV/LV issues
  ISSUEMAP issuesPerCh[10];
  TMultiGraph* mg[10];
  std::set<double> limits;
  for (int ch = 0; ch < 10; ++ch) {
    mg[ch] = new TMultiGraph;
    mg[ch]->SetNameTitle(fmt::format("ch{}", ch + 1).c_str(),
                         fmt::format("chamber {};time;{} (V)", ch + 1, scanHV ? "HV" : "LV").c_str());
    for (const auto& [alias, dps] : dpsMapsPerCh[ch]) {
      mg[ch]->Add(mapToGraph(alias, dps), "lp");
      auto limit = scanHV ? mch::StatusMapCreatorParam::Instance().hvLimits[ch] : getLVLimit(alias);
      limits.emplace(limit);
      findIssues(dps, limit, issuesPerCh[ch][alias]);
    }
  }

  // select HV/LV issues of a minimum duration (ms) occurring during runs
  selectIssues(issuesPerCh, runBoundaries, minDuration);
  selectO2Issues(o2issuesPerCh, runBoundaries);
  printIssues(issuesPerCh, o2issuesPerCh, scanHV, scanHV ? hvFormat : lvFormat);

  // display
  TCanvas* c[10];
  for (int ch = 0; ch < 10; ++ch) {
    if (scanHV) {
      c[ch] = drawDataPoints(mg[ch], hvRange[0], hvRange[1]);
      drawLimit(mch::StatusMapCreatorParam::Instance().hvLimits[ch], c[ch]);
    } else {
      auto lvMax = (what.find("LV") != what.npos || what.find("Sol") != what.npos) ? lvRange[2] : lvRange[1];
      c[ch] = drawDataPoints(mg[ch], lvRange[0], lvMax);
      for (auto limit : limits) {
        drawLimit(limit, c[ch]);
      }
    }
    drawRunBoudaries(runBoundaries, c[ch]);
  }

  // save display
  TFile dataFile(outFileName.c_str(), "recreate");
  for (int ch = 0; ch < 10; ++ch) {
    c[ch]->Write();
  }
  dataFile.Close();

  return 0;
}

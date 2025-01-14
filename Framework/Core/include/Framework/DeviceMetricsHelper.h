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
//
#ifndef O2_FRAMEWORK_DEVICEMETRICSHELPERS_H_
#define O2_FRAMEWORK_DEVICEMETRICSHELPERS_H_

#include "Framework/DeviceMetricsInfo.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string_view>
#include <vector>

namespace o2::framework
{
struct DriverInfo;

// General definition of what can of values can be put in a metric.
// Notice that int8_t is used for enums.
template <typename T>
concept DeviceMetricValue = std::same_as<int, T> || std::same_as<float, T> || std::same_as<uint64_t, T> || std::same_as<int8_t, T>;

// Numeric like metrics values.
template <typename T>
concept DeviceMetricNumericValue = std::same_as<int, T> || std::same_as<float, T> || std::same_as<uint64_t, T>;

// Enum like values
template <typename T>
concept DeviceMetricEnumValue = std::same_as<int8_t, T>;

struct DeviceMetricsHelper {
  /// Type of the callback which can be provided to be invoked every time a new
  /// metric is found by the system.
  using NewMetricCallback = std::function<void(std::string const&, MetricInfo const&, int value, size_t metricIndex)>;

  /// Helper function to parse a metric string.
  static bool parseMetric(std::string_view const s, ParsedMetricMatch& results);

  /// Processes a parsed metric and stores in the backend store.
  ///
  /// @matches is the regexp_matches from the metric identifying regex
  /// @info is the DeviceInfo associated to the device posting the metric
  /// @newMetricsCallback is a callback that will be invoked every time a new metric is added to the list.
  static bool processMetric(ParsedMetricMatch& results,
                            DeviceMetricsInfo& info,
                            NewMetricCallback newMetricCallback = nullptr);
  /// @return the index in metrics for the information of given metric
  static size_t metricIdxByName(std::string_view const name,
                                const DeviceMetricsInfo& info);

  template <std::same_as<int> T>
  static auto& getMetricsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.intMetrics;
  }

  template <std::same_as<float> T>
  static auto& getMetricsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.floatMetrics;
  }

  template <std::same_as<uint64_t> T>
  static auto& getMetricsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.uint64Metrics;
  }

  template <std::same_as<int8_t> T>
  static auto& getMetricsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.enumMetrics;
  }

  template <std::same_as<int> T>
  static auto& getTimestampsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.intTimestamps;
  }

  template <std::same_as<float> T>
  static auto& getTimestampsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.floatTimestamps;
  }

  template <std::same_as<uint64_t> T>
  static auto& getTimestampsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.uint64Timestamps;
  }

  template <std::same_as<int8_t> T>
  static auto& getTimestampsStore(DeviceMetricsInfo& metrics)
  {
    return metrics.enumTimestamps;
  }

  template <std::same_as<int> T>
  static auto getMetricType() -> MetricType
  {
    return MetricType::Int;
  }

  template <std::same_as<float> T>
  static auto getMetricType() -> MetricType
  {
    return MetricType::Float;
  }

  template <std::same_as<uint64_t> T>
  static auto getMetricType() -> MetricType
  {
    return MetricType::Uint64;
  }

  template <std::same_as<int8_t> T>
  static auto getMetricType() -> MetricType
  {
    return MetricType::Enum;
  }

  static auto updateNumericInfo(DeviceMetricsInfo& metrics, size_t metricIndex, float value, size_t timestamp)
  {
    metrics.minDomain[metricIndex] = std::min(metrics.minDomain[metricIndex], timestamp);
    metrics.maxDomain[metricIndex] = std::max(metrics.maxDomain[metricIndex], timestamp);
    metrics.max[metricIndex] = std::max(metrics.max[metricIndex], (float)value);
    metrics.min[metricIndex] = std::min(metrics.min[metricIndex], (float)value);
    metrics.changed.at(metricIndex) = true;
  }

  template <DeviceMetricNumericValue T>
  static auto getNumericMetricCursor(size_t metricIndex)
  {
    return [metricIndex](DeviceMetricsInfo& metrics, T value, size_t timestamp) {
      MetricInfo& metric = metrics.metrics[metricIndex];
      updateNumericInfo(metrics, metricIndex, (float)value, timestamp);

      auto& store = getMetricsStore<T>(metrics);
      auto& timestamps = getTimestampsStore<T>(metrics);
      size_t pos = metric.pos++ % store[metric.storeIdx].size();
      timestamps[metric.storeIdx][pos] = timestamp;
      store[metric.storeIdx][pos] = value;
      metric.filledMetrics++;
    };
  }

  static size_t bookMetricInfo(DeviceMetricsInfo& metrics, char const* name, MetricType type);

  /// @return helper to insert a given value in the metrics
  template <DeviceMetricNumericValue T>
  static size_t
    bookNumericMetric(DeviceMetricsInfo& metrics,
                      char const* name,
                      NewMetricCallback newMetricsCallback = nullptr)
  {
    size_t metricIndex = bookMetricInfo(metrics, name, getMetricType<T>());
    auto& metricInfo = metrics.metrics[metricIndex];
    if (newMetricsCallback != nullptr) {
      newMetricsCallback(name, metricInfo, 0, metricIndex);
    }
    return metricIndex;
  }

  /// @return helper to insert a given value in the metrics
  template <DeviceMetricNumericValue T>
  static std::function<void(DeviceMetricsInfo&, T value, size_t timestamp)>
    createNumericMetric(DeviceMetricsInfo& metrics,
                        char const* name,
                        NewMetricCallback newMetricsCallback = nullptr)
  {
    size_t metricIndex = bookNumericMetric<T>(metrics, name, newMetricsCallback);
    return getNumericMetricCursor<T>(metricIndex);
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_

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
#define BOOST_TEST_MODULE Test Framework DataSamplingHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "DataSampling/DataSamplingHeader.h"
#include "Headers/Stack.h"
#include "Headers/DataHeader.h"

using namespace o2::utilities;
using namespace o2::header;

BOOST_AUTO_TEST_CASE(DataSamplingHeaderInit)
{
  o2::header::DataHeader original("A", "TST", 1);
  DataSamplingHeader header{123, 456, 789, "abc", original};

  BOOST_CHECK_EQUAL(header.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(header.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(header.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(header.deviceID.str, "abc"), 0);
  BOOST_CHECK_EQUAL(strcmp(header.dataOrigin.str, "TST"), 0);
  BOOST_CHECK_EQUAL(strcmp(header.dataDescription.str, "A"), 0);
  BOOST_CHECK_EQUAL(header.subSpecification, 1);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderCopy)
{
  o2::header::DataHeader original("A", "TST", 1);
  DataSamplingHeader header{123, 456, 789, "abc", original};
  DataSamplingHeader copy(header);

  BOOST_CHECK_EQUAL(copy.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(copy.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(copy.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(copy.deviceID.str, "abc"), 0);
  BOOST_CHECK_EQUAL(strcmp(copy.dataOrigin.str, "TST"), 0);
  BOOST_CHECK_EQUAL(strcmp(copy.dataDescription.str, "A"), 0);
  BOOST_CHECK_EQUAL(copy.subSpecification, 1);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderAssignement)
{
  o2::header::DataHeader original("A", "TST", 1);
  DataSamplingHeader first{123, 456, 789, "abc", original};
  DataSamplingHeader second = first;

  BOOST_CHECK_EQUAL(first.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(first.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(first.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(first.deviceID.str, "abc"), 0);
  BOOST_CHECK_EQUAL(strcmp(first.dataOrigin.str, "TST"), 0);
  BOOST_CHECK_EQUAL(strcmp(first.dataDescription.str, "A"), 0);
  BOOST_CHECK_EQUAL(first.subSpecification, 1);

  BOOST_CHECK_EQUAL(second.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(second.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(second.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(second.deviceID.str, "abc"), 0);
  BOOST_CHECK_EQUAL(strcmp(second.dataOrigin.str, "TST"), 0);
  BOOST_CHECK_EQUAL(strcmp(second.dataDescription.str, "A"), 0);
  BOOST_CHECK_EQUAL(second.subSpecification, 1);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderOnStack)
{
  o2::header::DataHeader original("A", "TST", 1);
  DataSamplingHeader header{123, 456, 789, "abc", original};
  Stack headerStack{header};

  const auto* dsHeaderFromStack = get<DataSamplingHeader*>(headerStack.data());
  BOOST_REQUIRE_NE(dsHeaderFromStack, nullptr);

  BOOST_CHECK_EQUAL(dsHeaderFromStack->sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(dsHeaderFromStack->totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(dsHeaderFromStack->totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(dsHeaderFromStack->deviceID.str, "abc"), 0);
  BOOST_CHECK_EQUAL(strcmp(dsHeaderFromStack->dataOrigin.str, "TST"), 0);
  BOOST_CHECK_EQUAL(strcmp(dsHeaderFromStack->dataDescription.str, "A"), 0);
  BOOST_CHECK_EQUAL(dsHeaderFromStack->subSpecification, 1);
}

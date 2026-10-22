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

#define BOOST_TEST_MODULE Test MCTrack class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCTrack.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "TFile.h"
#include "TParticle.h"
#include "TMCProcess.h"

using namespace o2;

BOOST_AUTO_TEST_CASE(MCTrack_test)
{
  MCTrack track;

  // auxiliary lookup table needed to fetch and set hit properties
  std::vector<int> hitLUT(o2::detectors::DetID::nDetectors, -1);
  // in this test we have a single fictional detector 1, which we map to
  // the first bit
  hitLUT[1] = 0;

  // check properties on default constructed object
  BOOST_CHECK(track.getStore() == false);
  for (auto i = o2::detectors::DetID::First; i < o2::detectors::DetID::nDetectors; ++i) {
    BOOST_CHECK(track.leftTrace(i, hitLUT) == false);
  }
  BOOST_CHECK(track.getNumDet() == 0);
  BOOST_CHECK(track.hasHits() == false);

  // check storing
  track.setStore(true);
  BOOST_CHECK(track.getStore() == true);
  track.setStore(false);
  BOOST_CHECK(track.getStore() == false);
  track.setStore(true);
  BOOST_CHECK(track.getStore() == true);

  // set hit for first detector
  BOOST_CHECK(track.leftTrace(1, hitLUT) == false);
  track.setHit(hitLUT[1]);
  BOOST_CHECK(track.hasHits() == true);
  BOOST_CHECK(track.leftTrace(1, hitLUT) == true);
  BOOST_CHECK(track.getNumDet() == 1);

  // check process encoding
  track.setProcess(TMCProcess::kPPrimary);
  BOOST_CHECK(track.getProcess() == TMCProcess::kPPrimary);
  track.setProcess(TMCProcess::kPTransportation);
  BOOST_CHECK(track.getProcess() == TMCProcess::kPTransportation);

  {
    // serialize it
    TFile f("MCTrackOut.root", "RECREATE");
    f.WriteObject(&track, "MCTrack");
    f.Close();
  }

  {
    MCTrack* intrack = nullptr;
    TFile f("MCTrackOut.root", "OPEN");
    f.GetObject("MCTrack", intrack);
    BOOST_CHECK(intrack->getStore() == true);
    BOOST_CHECK(intrack->hasHits() == true);
  }
}

BOOST_AUTO_TEST_CASE(MCTrack_detector_contributions)
{
  MCTrack contributor;
  // Default/legacy deserialization must not masquerade as instrumented truth.
  BOOST_CHECK_EQUAL(contributor.getDetectorContributionVersion(), 0);
  contributor.initializeDetectorContributions();
  contributor.setDetectorContribution(5);
  contributor.setTransportModelScore(.9f, true);
  contributor.setTransportModelProtection(3);
  BOOST_CHECK_EQUAL(contributor.getDetectorContributionVersion(), 1);
  BOOST_CHECK_EQUAL(contributor.getDetectorContributionMask(), uint32_t{1} << 5);
  BOOST_CHECK(!contributor.hasHits()); // signal bookkeeping must not change retention.
  {
    TFile file("MCTrackContributionOut.root", "RECREATE");
    file.WriteObject(&contributor, "contributor");
  }
  {
    TFile file("MCTrackContributionOut.root", "READ");
    MCTrack* restored = nullptr;
    file.GetObject("contributor", restored);
    BOOST_REQUIRE(restored);
    BOOST_CHECK_EQUAL(restored->getDetectorContributionVersion(), 1);
    BOOST_CHECK_EQUAL(restored->getDetectorContributionMask(), uint32_t{1} << 5);
    BOOST_CHECK_EQUAL(restored->getTransportModelProtection(), 3);
    BOOST_CHECK(restored->getTransportModelShadow());
    BOOST_CHECK(!restored->hasHits());
  }
}

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
/// \file Vertexer.cxx
/// \author Matteo Concas mconcas@cern.ch
///

#include "ITStracking/Vertexer.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

namespace o2::its
{

Vertexer::Vertexer(VertexerTraits* traits) : mTraits(traits)
{
  if (!mTraits) {
    LOG(fatal) << "nullptr passed to ITS vertexer construction.";
  }
  mVertParams.resize(1);
}

float Vertexer::clustersToVertices(LogFunc logger)
{
  LogFunc evalLog = [](const std::string&) {};
  TrackingParameters trkPars;
  TimeFrameGPUParameters tfGPUpar;
  mTraits->updateVertexingParameters(mVertParams, tfGPUpar);

  auto handleException = [&](const auto& err) {
    LOGP(error, "Encountered critical error in step {}, stopping further processing of this TF: {}", StateNames[mCurState], err.what());
    if (!mVertParams[0].DropTFUponFailure) {
      throw err;
    } else {
      LOGP(error, "Dropping this TF!");
      mTimeFrame->resetTracklets();
    }
  };

  float timeTracklet{0.f}, timeSelection{0.f}, timeVertexing{0.f}, timeInit{0.f};
  try {
    for (int iteration = 0; iteration < std::min(mVertParams[0].nIterations, (int)mVertParams.size()); ++iteration) {
      mMemoryPool->setMaxMemory(mVertParams[iteration].MaxMemory);
      unsigned int nTracklets01{0}, nTracklets12{0};
      logger(fmt::format("=== ITS {} Seeding vertexer iteration {} summary:", mTraits->getName(), iteration));
      trkPars.PhiBins = mTraits->getVertexingParameters()[0].PhiBins;
      trkPars.ZBins = mTraits->getVertexingParameters()[0].ZBins;
      auto timeInitIteration = evaluateTask(
        &Vertexer::initialiseVertexer, StateNames[mCurState = Init], iteration, evalLog, trkPars, iteration);
      auto timeTrackletIteration = evaluateTask(
        &Vertexer::findTracklets, StateNames[mCurState = Trackleting], iteration, evalLog, iteration);
      nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
      nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
      auto timeSelectionIteration = evaluateTask(
        &Vertexer::validateTracklets, StateNames[mCurState = Validating], iteration, evalLog, iteration);
      auto timeVertexingIteration = evaluateTask(&Vertexer::findVertices, StateNames[mCurState = Finding], iteration, evalLog, iteration);
      printEpilog(logger, nTracklets01, nTracklets12, mTimeFrame->getNLinesTotal(), mTimeFrame->getTotVertIteration()[iteration], timeInitIteration, timeTrackletIteration, timeSelectionIteration, timeVertexingIteration);
      timeInit += timeInitIteration;
      timeTracklet += timeTrackletIteration;
      timeSelection += timeSelectionIteration;
      timeVertexing += timeVertexingIteration;
    }
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    handleException(err);
  } catch (const std::bad_alloc& err) {
    handleException(err);
  } catch (...) {
    LOGP(fatal, "Uncaught exception!");
  }

  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

void Vertexer::getGlobalConfiguration()
{
  auto& vc = o2::its::VertexerParamConfig::Instance();
  auto& grc = o2::its::ITSGpuTrackingParamConfig::Instance();

  // This is odd: we override only the parameters for the first iteration.
  // Variations for the next iterations are set in the trackingInterfrace.
  mVertParams[0].nIterations = vc.nIterations;
  mVertParams[0].deltaRof = vc.deltaRof;
  mVertParams[0].allowSingleContribClusters = vc.allowSingleContribClusters;
  mVertParams[0].zCut = vc.zCut;
  mVertParams[0].phiCut = vc.phiCut;
  mVertParams[0].pairCut = vc.pairCut;
  mVertParams[0].clusterCut = vc.clusterCut;
  mVertParams[0].histPairCut = vc.histPairCut;
  mVertParams[0].tanLambdaCut = vc.tanLambdaCut;
  mVertParams[0].lowMultBeamDistCut = vc.lowMultBeamDistCut;
  mVertParams[0].vertNsigmaCut = vc.vertNsigmaCut;
  mVertParams[0].vertRadiusSigma = vc.vertRadiusSigma;
  mVertParams[0].trackletSigma = vc.trackletSigma;
  mVertParams[0].maxZPositionAllowed = vc.maxZPositionAllowed;
  mVertParams[0].clusterContributorsCut = vc.clusterContributorsCut;
  mVertParams[0].maxTrackletsPerCluster = vc.maxTrackletsPerCluster;
  mVertParams[0].phiSpan = vc.phiSpan;
  mVertParams[0].nThreads = vc.nThreads;
  mVertParams[0].ZBins = vc.ZBins;
  mVertParams[0].PhiBins = vc.PhiBins;
  mVertParams[0].SaveTimeBenchmarks = vc.saveTimeBenchmarks;
}

void Vertexer::adoptTimeFrame(TimeFrame7& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Vertexer::printEpilog(LogFunc& logger,
                           const unsigned int trackletN01, const unsigned int trackletN12,
                           const unsigned selectedN, const unsigned int vertexN, const float initT,
                           const float trackletT, const float selecT, const float vertexT)
{
  logger(fmt::format(" - {} Vertexer: found {} | {} tracklets in: {} ms", mTraits->getName(), trackletN01, trackletN12, trackletT));
  logger(fmt::format(" - {} Vertexer: selected {} tracklets in: {} ms", mTraits->getName(), selectedN, selecT));
  logger(fmt::format(" - {} Vertexer: found {} vertices in: {} ms", mTraits->getName(), vertexN, vertexT));
  if (mVertParams[0].PrintMemory) {
    mTimeFrame->printArtefactsMemory();
    mMemoryPool->print();
  }
}

} // namespace o2::its

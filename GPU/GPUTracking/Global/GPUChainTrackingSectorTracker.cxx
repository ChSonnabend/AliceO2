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

/// \file GPUChainTrackingSectorTracker.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTPCClusterData.h"
#include "GPUTrackingInputProvider.h"
#include "GPUTPCClusterOccupancyMap.h"
#include "utils/strtag.h"
#include <fstream>

using namespace o2::gpu;

int32_t GPUChainTracking::ExtrapolationTracking(uint32_t iSector, int32_t threadId, bool synchronizeOutput)
{
  runKernel<GPUTPCExtrapolationTracking>({GetGridBlk(256, iSector % mRec->NStreams()), {iSector}});
  TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, processors()->tpcTrackers[iSector].MemoryResCommon(), iSector % mRec->NStreams());
  if (synchronizeOutput) {
    SynchronizeStream(iSector % mRec->NStreams());
  }
  return (0);
}

int32_t GPUChainTracking::RunTPCTrackingSectors()
{
  if (mRec->GPUStuck()) {
    GPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
    return (1);
  }

  const auto& threadContext = GetThreadContext();

  int32_t retVal = RunTPCTrackingSectors_internal();
  if (retVal) {
    SynchronizeGPU();
  }
  return (retVal != 0);
}

int32_t GPUChainTracking::RunTPCTrackingSectors_internal()
{
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("Running TPC Sector Tracker");
  }
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCSectorTracking;
  if (!param().par.earlyTpcTransform) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      processors()->tpcTrackers[i].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]);
      if (doGPU) {
        processorsShadow()->tpcTrackers[i].Data().SetClusterData(nullptr, mIOPtrs.clustersNative->nClustersSector[i], mIOPtrs.clustersNative->clusterOffset[i][0]); // TODO: not needed I think, anyway copied in SetupGPUProcessor
      }
    }
    mRec->MemoryScalers()->nTPCHits = mIOPtrs.clustersNative->nClustersTotal;
  } else {
    int32_t offset = 0;
    for (uint32_t i = 0; i < NSECTORS; i++) {
      processors()->tpcTrackers[i].Data().SetClusterData(mIOPtrs.clusterData[i], mIOPtrs.nClusterData[i], offset);
      if (doGPU && GetRecoSteps().isSet(RecoStep::TPCConversion)) {
        processorsShadow()->tpcTrackers[i].Data().SetClusterData(processorsShadow()->tpcConverter.mClusters + processors()->tpcTrackers[i].Data().ClusterIdOffset(), processors()->tpcTrackers[i].NHitsTotal(), processors()->tpcTrackers[i].Data().ClusterIdOffset());
      }
      offset += mIOPtrs.nClusterData[i];
    }
    mRec->MemoryScalers()->nTPCHits = offset;
  }
  GPUInfo("Event has %u TPC Clusters, %d TRD Tracklets", (uint32_t)mRec->MemoryScalers()->nTPCHits, mIOPtrs.nTRDTracklets);

  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    processors()->tpcTrackers[iSector].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSector], false); // Prepare custom allocation for 1st stack level
    mRec->AllocateRegisteredMemory(processors()->tpcTrackers[iSector].MemoryResSectorScratch());
  }
  mRec->PushNonPersistentMemory(qStr2Tag("TPCSLTRK"));
  for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
    SetupGPUProcessor(&processors()->tpcTrackers[iSector], true);             // Now we allocate
    mRec->ResetRegisteredMemoryPointers(&processors()->tpcTrackers[iSector]); // TODO: The above call breaks the GPU ptrs to already allocated memory. This fixes them. Should actually be cleaned up at the source.
    processors()->tpcTrackers[iSector].SetupCommonMemory();
  }

  bool streamInit[GPUCA_MAX_STREAMS] = {false};
  int32_t streamInitAndOccMap = mRec->NStreams() - 1;

  if (doGPU) {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      processorsShadow()->tpcTrackers[iSector].GPUParametersConst()->gpumem = (char*)mRec->DeviceMemoryBase();
      // Initialize Startup Constants
      processors()->tpcTrackers[iSector].GPUParameters()->nextStartHit = (((getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSectors>().minBlocks * BlockCount()) + NSECTORS - 1 - iSector) / NSECTORS) * getKernelProperties<GPUTPCTrackletConstructor, GPUTPCTrackletConstructor::allSectors>().nThreads;
      processorsShadow()->tpcTrackers[iSector].SetGPUTextureBase(mRec->DeviceMemoryBase());
    }

    if (PrepareTextures()) {
      return (2);
    }

    // Copy Tracker Object to GPU Memory
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Copying Tracker objects to GPU");
    }
    if (PrepareProfile()) {
      return 2;
    }

    WriteToConstantMemory(RecoStep::TPCSectorTracking, (char*)processors()->tpcTrackers - (char*)processors(), processorsShadow()->tpcTrackers, sizeof(GPUTPCTracker) * NSECTORS, streamInitAndOccMap, &mEvents->init);

    std::fill(streamInit, streamInit + mRec->NStreams(), false);
    streamInit[streamInitAndOccMap] = true;
  }

  if (param().rec.tpc.occupancyMapTimeBins || param().rec.tpc.sysClusErrorC12Norm) {
    AllocateRegisteredMemory(mInputsHost->mResourceOccupancyMap, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcOccupancyMap)]);
  }
  if (param().rec.tpc.occupancyMapTimeBins) {
    if (doGPU) {
      ReleaseEvent(mEvents->init);
    }
    uint32_t* ptr = doGPU ? mInputsShadow->mTPCClusterOccupancyMap : mInputsHost->mTPCClusterOccupancyMap;
    auto* ptrTmp = (GPUTPCClusterOccupancyMapBin*)mRec->AllocateVolatileMemory(GPUTPCClusterOccupancyMapBin::getTotalSize(param()), doGPU);
    runKernel<GPUMemClean16>(GetGridAutoStep(streamInitAndOccMap, RecoStep::TPCSectorTracking), ptrTmp, GPUTPCClusterOccupancyMapBin::getTotalSize(param()));
    runKernel<GPUTPCCreateOccupancyMap, GPUTPCCreateOccupancyMap::fill>(GetGridBlk(GPUCA_NSECTORS * GPUCA_ROW_COUNT, streamInitAndOccMap), ptrTmp);
    runKernel<GPUTPCCreateOccupancyMap, GPUTPCCreateOccupancyMap::fold>(GetGridBlk(GPUTPCClusterOccupancyMapBin::getNBins(param()), streamInitAndOccMap), ptrTmp, ptr + 2);
    mRec->ReturnVolatileMemory();
    mInputsHost->mTPCClusterOccupancyMap[1] = param().rec.tpc.occupancyMapTimeBins * 0x10000 + param().rec.tpc.occupancyMapTimeBinsAverage;
    if (doGPU) {
      GPUMemCpy(RecoStep::TPCSectorTracking, mInputsHost->mTPCClusterOccupancyMap + 2, mInputsShadow->mTPCClusterOccupancyMap + 2, sizeof(*ptr) * GPUTPCClusterOccupancyMapBin::getNBins(mRec->GetParam()), streamInitAndOccMap, false, &mEvents->init);
    } else {
      TransferMemoryResourceLinkToGPU(RecoStep::TPCSectorTracking, mInputsHost->mResourceOccupancyMap, streamInitAndOccMap, &mEvents->init);
    }
  }
  if (param().rec.tpc.occupancyMapTimeBins || param().rec.tpc.sysClusErrorC12Norm) {
    uint32_t& occupancyTotal = *mInputsHost->mTPCClusterOccupancyMap;
    occupancyTotal = CAMath::Float2UIntRn(mRec->MemoryScalers()->nTPCHits / (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasNHBFPerTF ? mIOPtrs.settingsTF->nHBFPerTF : 128));
    mRec->UpdateParamOccupancyMap(param().rec.tpc.occupancyMapTimeBins ? mInputsHost->mTPCClusterOccupancyMap + 2 : nullptr, param().rec.tpc.occupancyMapTimeBins ? mInputsShadow->mTPCClusterOccupancyMap + 2 : nullptr, occupancyTotal, streamInitAndOccMap);
  }

  int32_t streamMap[NSECTORS];

  bool error = false;
  mRec->runParallelOuterLoop(doGPU, NSECTORS, [&](uint32_t iSector) {
    GPUTPCTracker& trk = processors()->tpcTrackers[iSector];
    GPUTPCTracker& trkShadow = doGPU ? processorsShadow()->tpcTrackers[iSector] : trk;
    int32_t useStream = (iSector % mRec->NStreams());

    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Creating Sector Data (Sector %d)", iSector);
    }
    if (doGPU) {
      TransferMemoryResourcesToGPU(RecoStep::TPCSectorTracking, &trk, useStream);
      runKernel<GPUTPCCreateTrackingData>({GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSector}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init}});
      streamInit[useStream] = true;
    } else {
      if (ReadEvent(iSector, 0)) {
        GPUError("Error reading event");
        error = 1;
        return;
      }
    }
    if (GetProcessingSettings().deterministicGPUReconstruction) {
      runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::hitData>({GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSector}});
    }
    if (!doGPU && trk.CheckEmptySector() && GetProcessingSettings().debugLevel == 0) {
      return;
    }

    if (GetProcessingSettings().debugLevel >= 6) {
      *mDebugFile << "\n\nReconstruction: Sector " << iSector << "/" << NSECTORS << std::endl;
      if (GetProcessingSettings().debugMask & 1) {
        if (doGPU) {
          TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &trk, -1, true);
        }
        trk.DumpTrackingData(*mDebugFile);
      }
    }

    runKernel<GPUMemClean16>(GetGridAutoStep(useStream, RecoStep::TPCSectorTracking), trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));
    runKernel<GPUTPCNeighboursFinder>({GetGridBlk(GPUCA_ROW_COUNT, useStream), {iSector}, {nullptr, streamInit[useStream] ? nullptr : &mEvents->init}});
    streamInit[useStream] = true;

    if (GetProcessingSettings().keepDisplayMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &trk, -1, true);
      memcpy(trk.LinkTmpMemory(), mRec->Res(trk.MemoryResLinks()).Ptr(), mRec->Res(trk.MemoryResLinks()).Size());
      if (GetProcessingSettings().debugMask & 2) {
        trk.DumpLinks(*mDebugFile, 0);
      }
    }

    runKernel<GPUTPCNeighboursCleaner>({GetGridBlk(GPUCA_ROW_COUNT - 2, useStream), {iSector}});
    DoDebugAndDump(RecoStep::TPCSectorTracking, 4, trk, &GPUTPCTracker::DumpLinks, *mDebugFile, 1);

    runKernel<GPUTPCStartHitsFinder>({GetGridBlk(GPUCA_ROW_COUNT - 6, useStream), {iSector}});
#ifdef GPUCA_SORT_STARTHITS_GPU
    if (doGPU) {
      runKernel<GPUTPCStartHitsSorter>({GetGridAuto(useStream), {iSector}});
    }
#endif
    if (GetProcessingSettings().deterministicGPUReconstruction) {
      runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::startHits>({GetGrid(1, 1, useStream), {iSector}});
    }
    DoDebugAndDump(RecoStep::TPCSectorTracking, 32, trk, &GPUTPCTracker::DumpStartHits, *mDebugFile);

    if (GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      trk.UpdateMaxData();
      AllocateRegisteredMemory(trk.MemoryResTracklets());
      AllocateRegisteredMemory(trk.MemoryResOutput());
    }

    if (!(doGPU || GetProcessingSettings().debugLevel >= 1) || GetProcessingSettings().trackletConstructorInPipeline) {
      runKernel<GPUTPCTrackletConstructor>({GetGridAuto(useStream), {iSector}});
      DoDebugAndDump(RecoStep::TPCSectorTracking, 128, trk, &GPUTPCTracker::DumpTrackletHits, *mDebugFile);
      if (GetProcessingSettings().debugMask & 256 && GetProcessingSettings().deterministicGPUReconstruction < 2) {
        trk.DumpHitWeights(*mDebugFile);
      }
    }

    if (!(doGPU || GetProcessingSettings().debugLevel >= 1) || GetProcessingSettings().trackletSelectorInPipeline) {
      runKernel<GPUTPCTrackletSelector>({GetGridAuto(useStream), {iSector}});
      runKernel<GPUTPCExtrapolationTrackingCopyNumbers>({{1, -ThreadCount(), useStream}, {iSector}}, 1);
      if (GetProcessingSettings().deterministicGPUReconstruction) {
        runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::sectorTracks>({GetGrid(1, 1, useStream), {iSector}});
      }
      TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, trk.MemoryResCommon(), useStream, &mEvents->sector[iSector]);
      streamMap[iSector] = useStream;
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Sector %u, Number of tracks: %d", iSector, *trk.NTracks());
      }
      DoDebugAndDump(RecoStep::TPCSectorTracking, 512, trk, &GPUTPCTracker::DumpTrackHits, *mDebugFile);
    }
  });
  mRec->SetNActiveThreadsOuterLoop(1);
  if (error) {
    return (3);
  }

  if (doGPU || GetProcessingSettings().debugLevel >= 1) {
    if (doGPU) {
      ReleaseEvent(mEvents->init);
    }

    if (!GetProcessingSettings().trackletSelectorInPipeline) {
      if (GetProcessingSettings().trackletConstructorInPipeline) {
        SynchronizeGPU();
      } else {
        for (int32_t i = 0; i < mRec->NStreams(); i++) {
          RecordMarker(&mEvents->stream[i], i);
        }
        runKernel<GPUTPCTrackletConstructor, 1>({GetGridAuto(0), krnlRunRangeNone, {&mEvents->single, mEvents->stream, mRec->NStreams()}});
        for (int32_t i = 0; i < mRec->NStreams(); i++) {
          ReleaseEvent(mEvents->stream[i]);
        }
        SynchronizeEventAndRelease(mEvents->single);
      }

      if (GetProcessingSettings().debugLevel >= 4) {
        for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
          DoDebugAndDump(RecoStep::TPCSectorTracking, 128, processors()->tpcTrackers[iSector], &GPUTPCTracker::DumpTrackletHits, *mDebugFile);
        }
      }

      int32_t runSectors = 0;
      int32_t useStream = 0;
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector += runSectors) {
        if (runSectors < GetProcessingSettings().trackletSelectorSectors) {
          runSectors++;
        }
        runSectors = CAMath::Min<int32_t>(runSectors, NSECTORS - iSector);
        if (getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount() < (uint32_t)runSectors) {
          runSectors = getKernelProperties<GPUTPCTrackletSelector>().minBlocks * BlockCount();
        }

        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Running TPC Tracklet selector (Stream %d, Sector %d to %d)", useStream, iSector, iSector + runSectors);
        }
        runKernel<GPUTPCTrackletSelector>({GetGridAuto(useStream), {iSector, runSectors}});
        runKernel<GPUTPCExtrapolationTrackingCopyNumbers>({{1, -ThreadCount(), useStream}, {iSector}}, runSectors);
        for (uint32_t k = iSector; k < iSector + runSectors; k++) {
          if (GetProcessingSettings().deterministicGPUReconstruction) {
            runKernel<GPUTPCSectorDebugSortKernels, GPUTPCSectorDebugSortKernels::sectorTracks>({GetGrid(1, 1, useStream), {k}});
          }
          TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, processors()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents->sector[k]);
          streamMap[k] = useStream;
        }
        useStream++;
        if (useStream >= mRec->NStreams()) {
          useStream = 0;
        }
      }
    }

    mSectorSelectorReady = 0;

    std::array<bool, NSECTORS> transferRunning;
    transferRunning.fill(true);
    if ((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (doGPU && !(GetRecoStepsGPU() & RecoStep::TPCMerging))) {
      if (param().rec.tpc.extrapolationTracking) {
        mWriteOutputDone.fill(0);
      }

      uint32_t tmpSector = 0;
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Transfering Tracks from GPU to Host");
        }

        if (tmpSector == iSector) {
          SynchronizeEvents(&mEvents->sector[iSector]);
        }
        while (tmpSector < NSECTORS && (tmpSector == iSector || IsEventDone(&mEvents->sector[tmpSector]))) {
          ReleaseEvent(mEvents->sector[tmpSector]);
          if (*processors()->tpcTrackers[tmpSector].NTracks() > 0) {
            TransferMemoryResourceLinkToHost(RecoStep::TPCSectorTracking, processors()->tpcTrackers[tmpSector].MemoryResOutput(), streamMap[tmpSector], &mEvents->sector[tmpSector]);
          } else {
            transferRunning[tmpSector] = false;
          }
          tmpSector++;
        }

        if (GetProcessingSettings().keepAllMemory) {
          TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &processors()->tpcTrackers[iSector], -1, true);
          if (!GetProcessingSettings().trackletConstructorInPipeline) {
            if (GetProcessingSettings().debugMask & 256 && GetProcessingSettings().deterministicGPUReconstruction < 2) {
              processors()->tpcTrackers[iSector].DumpHitWeights(*mDebugFile);
            }
          }
          if (!GetProcessingSettings().trackletSelectorInPipeline) {
            if (GetProcessingSettings().debugMask & 512) {
              processors()->tpcTrackers[iSector].DumpTrackHits(*mDebugFile);
            }
          }
        }

        if (transferRunning[iSector]) {
          SynchronizeEvents(&mEvents->sector[iSector]);
        }
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Tracks Transfered: %d / %d", *processors()->tpcTrackers[iSector].NTracks(), *processors()->tpcTrackers[iSector].NTrackHits());
        }

        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Data ready for sector %d", iSector);
        }
        mSectorSelectorReady = iSector;

        if (param().rec.tpc.extrapolationTracking) {
          for (uint32_t tmpSector2a = 0; tmpSector2a <= iSector; tmpSector2a++) {
            uint32_t tmpSector2 = GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorOrder(tmpSector2a);
            uint32_t sectorLeft, sectorRight;
            GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorLeftRight(tmpSector2, sectorLeft, sectorRight);

            if (tmpSector2 <= iSector && sectorLeft <= iSector && sectorRight <= iSector && mWriteOutputDone[tmpSector2] == 0) {
              ExtrapolationTracking(tmpSector2, 0);
              WriteOutput(tmpSector2, 0);
              mWriteOutputDone[tmpSector2] = 1;
            }
          }
        } else {
          WriteOutput(iSector, 0);
        }
      }
    }
    if (!(GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) && param().rec.tpc.extrapolationTracking) {
      std::vector<bool> blocking(NSECTORS * mRec->NStreams());
      for (int32_t i = 0; i < NSECTORS; i++) {
        for (int32_t j = 0; j < mRec->NStreams(); j++) {
          blocking[i * mRec->NStreams() + j] = i % mRec->NStreams() == j;
        }
      }
      for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
        uint32_t tmpSector = GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorOrder(iSector);
        if (!((GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) || (doGPU && !(GetRecoStepsGPU() & RecoStep::TPCMerging)))) {
          uint32_t sectorLeft, sectorRight;
          GPUTPCExtrapolationTracking::ExtrapolationTrackingSectorLeftRight(tmpSector, sectorLeft, sectorRight);
          if (doGPU && !blocking[tmpSector * mRec->NStreams() + sectorLeft % mRec->NStreams()]) {
            StreamWaitForEvents(tmpSector % mRec->NStreams(), &mEvents->sector[sectorLeft]);
            blocking[tmpSector * mRec->NStreams() + sectorLeft % mRec->NStreams()] = true;
          }
          if (doGPU && !blocking[tmpSector * mRec->NStreams() + sectorRight % mRec->NStreams()]) {
            StreamWaitForEvents(tmpSector % mRec->NStreams(), &mEvents->sector[sectorRight]);
            blocking[tmpSector * mRec->NStreams() + sectorRight % mRec->NStreams()] = true;
          }
        }
        ExtrapolationTracking(tmpSector, 0, false);
      }
    }
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      if (doGPU && transferRunning[iSector]) {
        ReleaseEvent(mEvents->sector[iSector]);
      }
    }
  } else {
    mSectorSelectorReady = NSECTORS;
    mRec->runParallelOuterLoop(doGPU, NSECTORS, [&](uint32_t iSector) {
      if (param().rec.tpc.extrapolationTracking) {
        ExtrapolationTracking(iSector, 0);
      }
      if (GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCSectorTracks) {
        WriteOutput(iSector, 0);
      }
    });
    mRec->SetNActiveThreadsOuterLoop(1);
  }

  if (param().rec.tpc.extrapolationTracking && GetProcessingSettings().debugLevel >= 3) {
    for (uint32_t iSector = 0; iSector < NSECTORS; iSector++) {
      GPUInfo("Sector %d - Tracks: Local %d Extrapolated %d - Hits: Local %d Extrapolated %d", iSector,
              processors()->tpcTrackers[iSector].CommonMemory()->nLocalTracks, processors()->tpcTrackers[iSector].CommonMemory()->nTracks, processors()->tpcTrackers[iSector].CommonMemory()->nLocalTrackHits, processors()->tpcTrackers[iSector].CommonMemory()->nTrackHits);
    }
  }

  if (GetProcessingSettings().debugMask & 1024 && !GetProcessingSettings().deterministicGPUReconstruction) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      processors()->tpcTrackers[i].DumpOutput(*mDebugFile);
    }
  }

  if (DoProfile()) {
    return (1);
  }
  for (uint32_t i = 0; i < NSECTORS; i++) {
    mIOPtrs.nSectorTracks[i] = *processors()->tpcTrackers[i].NTracks();
    mIOPtrs.sectorTracks[i] = processors()->tpcTrackers[i].Tracks();
    mIOPtrs.nSectorClusters[i] = *processors()->tpcTrackers[i].NTrackHits();
    mIOPtrs.sectorClusters[i] = processors()->tpcTrackers[i].TrackHits();
    if (GetProcessingSettings().keepDisplayMemory && !GetProcessingSettings().keepAllMemory) {
      TransferMemoryResourcesToHost(RecoStep::TPCSectorTracking, &processors()->tpcTrackers[i], -1, true);
    }
  }
  if (GetProcessingSettings().debugLevel >= 2) {
    GPUInfo("TPC Sector Tracker finished");
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCSectorTracking, qStr2Tag("TPCSLTRK"));
  return 0;
}

int32_t GPUChainTracking::ReadEvent(uint32_t iSector, int32_t threadId)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running ReadEvent for sector %d on thread %d\n", iSector, threadId);
  }
  runKernel<GPUTPCCreateTrackingData>({{GetGridAuto(0, GPUReconstruction::krnlDeviceType::CPU)}, {iSector}});
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished ReadEvent for sector %d on thread %d\n", iSector, threadId);
  }
  return (0);
}

void GPUChainTracking::WriteOutput(int32_t iSector, int32_t threadId)
{
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Running WriteOutput for sector %d on thread %d\n", iSector, threadId);
  }
  processors()->tpcTrackers[iSector].WriteOutputPrepare();
  processors()->tpcTrackers[iSector].WriteOutput();
  if (GetProcessingSettings().debugLevel >= 5) {
    GPUInfo("Finished WriteOutput for sector %d on thread %d\n", iSector, threadId);
  }
}

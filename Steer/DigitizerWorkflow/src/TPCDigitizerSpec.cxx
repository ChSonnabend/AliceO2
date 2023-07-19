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

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "Framework/RootSerializationSupport.h"
#include "TPCDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ParallelContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/DeviceSpec.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBase/CDBInterface.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Detector.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsBase/Detector.h"
#include "TPCCalibration/VDriftHelper.h"
#include "CommonDataFormat/RangeReference.h"
#include "SimConfig/DigiParams.h"
#include <filesystem>
#include "TH3.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DigiGroupRef = o2::dataformats::RangeReference<int, int>;
using SC = o2::tpc::SpaceCharge<double>;

namespace o2
{
namespace tpc
{

template <typename T>
void copyHelper(T const& origin, T& target)
{
  std::copy(origin.begin(), origin.end(), std::back_inserter(target));
}
template <>
void copyHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& origin, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& target)
{
  target.mergeAtBack(origin);
}

template <typename T>
void writeToBranchHelper(TTree& tree, const char* name, T* accum)
{
  auto targetbr = o2::base::getOrMakeBranch(tree, name, accum);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}
template <>
void writeToBranchHelper<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>(TTree& tree,
                                                                             const char* name, o2::dataformats::MCTruthContainer<o2::MCCompLabel>* accum)
{
  // we convert first of all to IOMCTruthContainer
  std::vector<char> buffer;
  accum->flatten_to(buffer);
  accum->clear_andfreememory();
  o2::dataformats::IOMCTruthContainerView view(buffer);
  auto targetbr = o2::base::getOrMakeBranch(tree, name, &view);
  targetbr->Fill();
  targetbr->ResetAddress();
  targetbr->DropBaskets("all");
}

std::string getBranchNameLeft(int sector)
{
  std::stringstream branchnamestreamleft;
  branchnamestreamleft << "TPCHitsShiftedSector" << int(o2::tpc::Sector::getLeft(o2::tpc::Sector(sector)));
  return branchnamestreamleft.str();
}

std::string getBranchNameRight(int sector)
{
  std::stringstream branchnamestreamright;
  branchnamestreamright << "TPCHitsShiftedSector" << sector;
  return branchnamestreamright.str();
}

using namespace o2::base;
class TPCDPLDigitizerTask : public BaseDPLDigitizer
{
 public:
  TPCDPLDigitizerTask(bool internalwriter) : mInternalWriter(internalwriter), BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM)
  {
  }

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing TPC digitization";

    mLaneId = ic.services().get<const o2::framework::DeviceSpec>().rank;

    mWithMCTruth = o2::conf::DigiParams::Instance().mctruth;
    auto useDistortions = ic.options().get<int>("distortionType");
    auto triggeredMode = ic.options().get<bool>("TPCtriggered");
    mUseCalibrationsFromCCDB = ic.options().get<bool>("TPCuseCCDB");
    LOG(info) << "TPC calibrations from CCDB: " << mUseCalibrationsFromCCDB;

    if (useDistortions > 0) {
      if (useDistortions == 1) {
        LOG(info) << "Using realistic space-charge distortions.";
      } else {
        LOG(info) << "Using constant space-charge distortions.";
      }
      auto readSpaceChargeString = ic.options().get<std::string>("readSpaceCharge");
      std::vector<std::string> readSpaceCharge;
      std::stringstream ssSpaceCharge(readSpaceChargeString);
      while (ssSpaceCharge.good()) {
        std::string substr;
        getline(ssSpaceCharge, substr, ',');
        readSpaceCharge.push_back(substr);
      }
      if (readSpaceCharge[0].size() != 0) { // use pre-calculated space-charge object
        if (std::filesystem::exists(readSpaceCharge[0])) {
          LOGP(info, "Reading space-charge object from file {}", readSpaceCharge[0].data());
          mDigitizer.setUseSCDistortions(readSpaceCharge[0]);
        } else {
          LOG(error) << "Space-charge object or file not found!";
        }
      } else { // create new space-charge object either with empty TPC or an initial space-charge density provided by histogram
        SCDistortionType distortionType = useDistortions == 2 ? SCDistortionType::SCDistortionsConstant : SCDistortionType::SCDistortionsRealistic;
        auto inputHistoString = ic.options().get<std::string>("initialSpaceChargeDensity");
        std::vector<std::string> inputHisto;
        std::stringstream ssHisto(inputHistoString);
        while (ssHisto.good()) {
          std::string substr;
          getline(ssHisto, substr, ',');
          inputHisto.push_back(substr);
        }
        std::unique_ptr<TH3> hisSCDensity;
        if (std::filesystem::exists(inputHisto[0])) {
          auto fileSCInput = std::unique_ptr<TFile>(TFile::Open(inputHisto[0].data()));
          if (fileSCInput->FindKey(inputHisto[1].data())) {
            hisSCDensity.reset((TH3*)fileSCInput->Get(inputHisto[1].data()));
            hisSCDensity->SetDirectory(nullptr);
          }
        }
        if (hisSCDensity.get() != nullptr) {
          LOG(info) << "TPC: Providing initial space-charge density histogram: " << hisSCDensity->GetName();
          mDigitizer.setUseSCDistortions(distortionType, hisSCDensity.get());
        } else {
          if (distortionType == SCDistortionType::SCDistortionsConstant) {
            LOG(error) << "Input space-charge density histogram or file not found!";
          }
        }
      }
    }
    mDigitizer.setContinuousReadout(!triggeredMode);

    // we send the GRP data once if the corresponding output channel is available
    // and set the flag to false after
    mWriteGRP = true;

    // clean up (possibly) existing digit files
    if (mInternalWriter) {
      cleanDigitFile();
    }
  }

  void cleanDigitFile()
  {
    // since we update digit files during ordinary processing
    // it is better to remove possibly existing files in the same dir
    std::stringstream tmp;
    tmp << "tpc_driftime_digits_lane" << mLaneId << ".root";
    if (std::filesystem::exists(tmp.str())) {
      std::filesystem::remove(tmp.str());
    }

    if (std::filesystem::exists("mclabels_digitizer.root")) {
      std::filesystem::remove("mclabels_digitizer.root");
    }
  }

  void writeToROOTFile()
  {
    if (!mInternalROOTFlushFile) {
      std::stringstream tmp;
      tmp << "tpc_driftime_digits_lane" << mLaneId << ".root";
      mInternalROOTFlushFile = new TFile(tmp.str().c_str(), "UPDATE");
      std::stringstream trname;
      trname << mSector;
      mInternalROOTFlushTTree = new TTree(trname.str().c_str(), "o2sim");
    }
    {
      std::stringstream brname;
      brname << "TPCDigit_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mDigits);
      br->Fill();
      br->ResetAddress();
    }
    if (mWithMCTruth) {
      // labels
      std::stringstream brname;
      brname << "TPCDigitMCTruth_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mLabels);
      br->Fill();
      br->ResetAddress();
    }
    {
      // common
      std::stringstream brname;
      brname << "TPCCommonMode_" << mSector;
      auto br = o2::base::getOrMakeBranch(*mInternalROOTFlushTTree, brname.str().c_str(), &mCommonMode);
      br->Fill();
      br->ResetAddress();
    }
  }

  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
      return;
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(info) << "Processing TPC digitization";

    /// For the time being use the defaults for the CDB
    auto& cdb = o2::tpc::CDBInterface::instance();
    cdb.setUseDefaults(!mUseCalibrationsFromCCDB);
    // whatever are global settings for CCDB usage, we have to extract the TPC vdrift from CCDB for anchored simulations
    mTPCVDriftHelper.extractCCDBInputs(pc);
    if (mTPCVDriftHelper.isUpdated()) {
      const auto& vd = mTPCVDriftHelper.getVDriftObject();
      LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
           vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper.getSourceName());
      mDigitizer.setVDrift(vd.getVDrift());
      mDigitizer.setTDriftOffset(vd.getTimeOffset());
      mTPCVDriftHelper.acknowledgeUpdate();
    }

    if (std::filesystem::exists("ThresholdMap.root")) {
      LOG(info) << "TPC: Using zero suppression map from 'ThresholdMap.root'";
      cdb.setThresholdMapFromFile("ThresholdMap.root");
    }

    if (std::filesystem::exists("GainMap.root")) {
      LOG(info) << "TPC: Using gain map from 'GainMap.root'";
      cdb.setGainMapFromFile("GainMap.root");
    }

    for (auto it = pc.inputs().begin(), end = pc.inputs().end(); it != end; ++it) {
      for (auto const& inputref : it) {
        if (inputref.spec->lifetime == o2::framework::Lifetime::Condition) { // process does not need conditions
          continue;
        }
        process(pc, inputref);
        if (mInternalWriter) {
          mInternalROOTFlushTTree->SetEntries(mFlushCounter);
          mInternalROOTFlushFile->Write("", TObject::kOverwrite);
          mInternalROOTFlushFile->Close();
          // delete mInternalROOTFlushTTree; --> automatically done by ->Close()
          delete mInternalROOTFlushFile;
          mInternalROOTFlushFile = nullptr;
        }
        // TODO: make generic reset method?
        mFlushCounter = 0;
        mDigitCounter = 0;
      }
    }
  }

  // process one sector
  void process(framework::ProcessingContext& pc, framework::DataRef const& inputref)
  {
    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>(inputref);
    context->initSimChains(o2::detectors::DetID::TPC, mSimChains);
    auto& irecords = context->getEventRecords();
    LOG(info) << "TPC: Processing " << irecords.size() << " collisions";
    if (irecords.size() == 0) {
      return;
    }
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(inputref);

    bool isContinuous = mDigitizer.isContinuousReadout();
    // we publish the GRP data once if the output channel is there
    if (mWriteGRP && pc.outputs().isAllowed({"TPC", "ROMode", 0})) {
      auto roMode = isContinuous ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(info) << "TPC: Sending ROMode= " << (mDigitizer.isContinuousReadout() ? "Continuous" : "Triggered")
                << " to GRPUpdater from channel " << dh->subSpecification;
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0, Lifetime::Timeframe}, roMode);
    }
    mWriteGRP = false;

    // extract which sector to treat
    auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputref);
    if (sectorHeader == nullptr) {
      LOG(error) << "TPC: Sector header missing, skipping processing";
      return;
    }
    auto sector = sectorHeader->sector();
    mSector = sector;
    mListOfSectors.push_back(sector);
    LOG(info) << "TPC: Processing sector " << sector;
    // the active sectors need to be propagated
    uint64_t activeSectors = 0;
    activeSectors = sectorHeader->activeSectors;

    // lambda that creates a DPL owned buffer to accumulate the digits (in shared memory)
    auto makeDigitBuffer = [this, sector, &pc, activeSectors, &dh]() {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (mInternalWriter) {
        using ContainerType = std::decay_t<decltype(pc.outputs().make<std::vector<o2::tpc::Digit>>(Output{"", "", 0}))>*;
        return ContainerType(nullptr);
      } else {
        // default case
        return &pc.outputs().make<std::vector<o2::tpc::Digit>>(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe, header});
      }
    };
    // lambda that snapshots the common mode vector to be sent out; prepares and attaches header with sector information
    auto snapshotCommonMode = [this, sector, &pc, activeSectors, &dh](std::vector<o2::tpc::CommonMode> const& commonMode) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (!mInternalWriter) {
        // note that snapshoting only works with non-const references (to be fixed?)
        pc.outputs().snapshot(Output{"TPC", "COMMONMODE", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe,
                                     header},
                              const_cast<std::vector<o2::tpc::CommonMode>&>(commonMode));
      }
    };
    // lambda that snapshots labels to be sent out; prepares and attaches header with sector information
    auto snapshotLabels = [this, &sector, &pc, activeSectors, &dh](o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (mWithMCTruth) {
        if (!mInternalWriter) {
          auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe, header});
          labels.flatten_to(sharedlabels);
        }
      }
    };
    // lambda that snapshots digits grouping (triggers) to be sent out; prepares and attaches header with sector information
    auto snapshotEvents = [this, sector, &pc, activeSectors, &dh](const std::vector<DigiGroupRef>& events) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = activeSectors;
      if (!mInternalWriter) {
        LOG(info) << "TPC: Send TRIGGERS for sector " << sector << " channel " << dh->subSpecification << " | size " << events.size();
        pc.outputs().snapshot(Output{"TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(dh->subSpecification), Lifetime::Timeframe,
                                     header},
                              const_cast<std::vector<DigiGroupRef>&>(events));
      }
    };

    auto digitsAccum = makeDigitBuffer();                          // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum; // timeframe accumulator for labels
    std::vector<CommonMode> commonModeAccum;
    std::vector<DigiGroupRef> eventAccum;

    // this should not happen any more, legacy condition when the sector variable was used
    // to transport control information
    if (sector < 0) {
      throw std::runtime_error("Legacy control information is not expected any more");
    }

    // the TPCSectorHeader now allows to transport information for more than one sector,
    // e.g. for transporting clusters in one single data block. The digitization is however
    // only on sector level
    if (sector >= TPCSectorHeader::NSectors) {
      throw std::runtime_error("Digitizer can only work on single sectors");
    }

    mDigitizer.setSector(sector);
    mDigitizer.init();

    auto& eventParts = context->getEventParts();

    auto flushDigitsAndLabels = [this, digitsAccum, &labelAccum, &commonModeAccum](bool finalFlush = false) {
      mFlushCounter++;
      // flush previous buffer
      mDigits.clear();
      mLabels.clear();
      mCommonMode.clear();
      mDigitizer.flush(mDigits, mLabels, mCommonMode, finalFlush);
      LOG(info) << "TPC: Flushed " << mDigits.size() << " digits, " << mLabels.getNElements() << " labels and " << mCommonMode.size() << " common mode entries";

      if (mInternalWriter) {
        // the natural place to write out this independent datachunk immediately ...
        writeToROOTFile();
      } else {
        // ... or to accumulate and later forward to next DPL proc
        std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(*digitsAccum));
        if (mWithMCTruth) {
          labelAccum.mergeAtBack(mLabels);
        }
        std::copy(mCommonMode.begin(), mCommonMode.end(), std::back_inserter(commonModeAccum));
      }
      mDigitCounter += mDigits.size();
    };

    if (isContinuous) {
      auto& hbfu = o2::raw::HBFUtils::Instance();
      double time = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).bc2ns() / 1000.;
      mDigitizer.setOutputDigitTimeOffset(time);
      mDigitizer.setStartTime(irecords[0].getTimeNS() / 1000.f);
    }

    TStopwatch timer;
    timer.Start();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      const double eventTime = irecords[collID].getTimeNS() / 1000.f;
      LOG(info) << "TPC: Event time " << eventTime << " us";
      mDigitizer.setEventTime(eventTime);
      if (!isContinuous) {
        mDigitizer.setStartTime(eventTime);
      }
      size_t startSize = mDigitCounter; // digitsAccum->size();

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)

      window_size = mDigitizer.getWindowSize();
      for (auto& part : eventParts[collID]) {
        const int eventID = part.entryID;
        const int sourceID = part.sourceID;

        // get the hits for this event and this source
        std::vector<o2::tpc::HitGroup> hitsLeft;
        std::vector<o2::tpc::HitGroup> hitsRight;
        context->retrieveHits(mSimChains, getBranchNameLeft(sector).c_str(), part.sourceID, part.entryID, &hitsLeft);
        context->retrieveHits(mSimChains, getBranchNameRight(sector).c_str(), part.sourceID, part.entryID, &hitsRight);
        LOG(debug) << "TPC: Found " << hitsLeft.size() << " hit groups left and " << hitsRight.size() << " hit groups right in collision " << collID << " eventID " << part.entryID;

        mDigitizer.process(hitsLeft, eventID, sourceID);

        tmp_sector_vec = mDigitizer.getSector();
        tmp_row_vec = mDigitizer.getRow();
        tmp_max_time = mDigitizer.getMaxTime();
        tmp_max_pad = mDigitizer.getMaxPad();
        tmp_max_q = mDigitizer.getMaxQ();
        tmp_cog_time = mDigitizer.getCogTime();
        tmp_cog_pad = mDigitizer.getCogPad();
        tmp_cog_q = mDigitizer.getCogQ();
        tmp_mc_label = mDigitizer.getMcLabels();
        tmp_mc_noise = mDigitizer.getIsNoise();
        tmp_mc_fake = mDigitizer.getIsFake();
        tmp_point_counter = mDigitizer.getPointCounter();

        sector_vec.insert(sector_vec.end(), tmp_sector_vec.begin(), tmp_sector_vec.end());
        row_vec.insert(row_vec.end(), tmp_row_vec.begin(), tmp_row_vec.end());
        max_time.insert(max_time.end(), tmp_max_time.begin(), tmp_max_time.end());
        max_pad.insert(max_pad.end(), tmp_max_pad.begin(), tmp_max_pad.end());
        max_q.insert(max_q.end(), tmp_max_q.begin(), tmp_max_q.end());
        cog_time.insert(cog_time.end(), tmp_cog_time.begin(), tmp_cog_time.end());
        cog_pad.insert(cog_pad.end(), tmp_cog_pad.begin(), tmp_cog_pad.end());
        cog_q.insert(cog_q.end(), tmp_cog_q.begin(), tmp_cog_q.end());
        mclabel.insert(mclabel.end(), tmp_mc_label.begin(), tmp_mc_label.end());
        mc_noise.insert(mc_noise.end(), tmp_mc_noise.begin(), tmp_mc_noise.end());
        mc_fake.insert(mc_fake.end(), tmp_mc_fake.begin(), tmp_mc_fake.end());
        point_counter.insert(point_counter.end(), tmp_point_counter.begin(), tmp_point_counter.end());
        elem_counter += mDigitizer.getElemCounter();
        mDigitizer.clearElements();


        mDigitizer.process(hitsRight, eventID, sourceID);

        tmp_sector_vec = mDigitizer.getSector();
        tmp_row_vec = mDigitizer.getRow();
        tmp_max_time = mDigitizer.getMaxTime();
        tmp_max_pad = mDigitizer.getMaxPad();
        tmp_max_q = mDigitizer.getMaxQ();
        tmp_cog_time = mDigitizer.getCogTime();
        tmp_cog_pad = mDigitizer.getCogPad();
        tmp_cog_q = mDigitizer.getCogQ();
        tmp_mc_label = mDigitizer.getMcLabels();
        tmp_mc_noise = mDigitizer.getIsNoise();
        tmp_mc_fake = mDigitizer.getIsFake();
        tmp_point_counter = mDigitizer.getPointCounter();

        sector_vec.insert(sector_vec.end(), tmp_sector_vec.begin(), tmp_sector_vec.end());
        row_vec.insert(row_vec.end(), tmp_row_vec.begin(), tmp_row_vec.end());
        max_time.insert(max_time.end(), tmp_max_time.begin(), tmp_max_time.end());
        max_pad.insert(max_pad.end(), tmp_max_pad.begin(), tmp_max_pad.end());
        max_q.insert(max_q.end(), tmp_max_q.begin(), tmp_max_q.end());
        cog_time.insert(cog_time.end(), tmp_cog_time.begin(), tmp_cog_time.end());
        cog_pad.insert(cog_pad.end(), tmp_cog_pad.begin(), tmp_cog_pad.end());
        cog_q.insert(cog_q.end(), tmp_cog_q.begin(), tmp_cog_q.end());
        mclabel.insert(mclabel.end(), tmp_mc_label.begin(), tmp_mc_label.end());
        mc_noise.insert(mc_noise.end(), tmp_mc_noise.begin(), tmp_mc_noise.end());
        mc_fake.insert(mc_fake.end(), tmp_mc_fake.begin(), tmp_mc_fake.end());
        point_counter.insert(point_counter.end(), tmp_point_counter.begin(), tmp_point_counter.end());
        elem_counter += mDigitizer.getElemCounter();
        mDigitizer.clearElements();

        LOG(info) << "Processed " << elem_counter << " clusters!";


        flushDigitsAndLabels();

        if (!isContinuous) {
          eventAccum.emplace_back(startSize, mDigits.size());
        }
      }
    }

    // final flushing step; getting everything not yet written out
    if (isContinuous) {
      LOG(info) << "TPC: Final flush";
      flushDigitsAndLabels(true);
      eventAccum.emplace_back(0, mDigitCounter); // all digits are grouped to 1 super-event pseudo-triggered mode
    }

    if (!mInternalWriter) {
      // send out to next stage
      snapshotEvents(eventAccum);
      // snapshotDigits(digitsAccum); --> done automatically
      snapshotCommonMode(commonModeAccum);
      snapshotLabels(labelAccum);
    }

    /// OWN IMPLEMENTATION
    LOG(info) << "Writing mcdigits to tree!";

    std::stringstream tmp;
    tmp << "sector_" << mSector;
    std::stringstream fileName;
    fileName << "mclabels_digitizer_" << mSector << ".root";
    TFile outputFile(fileName.str().c_str(), "RECREATE");
    TTree* mcTree = new TTree(tmp.str().c_str(), "MC tree");

    int sec=0, r=0, mp=0, mt=0, idx=0, p=0, lab=0, mcn=0, mcf=0;
    float sp=0, st=0, cp=0, ct=0, cq=-1, mq=0;

    mcTree->Branch("cluster_sector", &sec);
    mcTree->Branch("cluster_row", &r);
    mcTree->Branch("cluster_cog_pad", &cp);
    mcTree->Branch("cluster_cog_time", &ct);
    mcTree->Branch("cluster_cog_q", &cq);
    mcTree->Branch("cluster_sigma_pad", &sp);
    mcTree->Branch("cluster_sigma_time", &st);
    mcTree->Branch("cluster_max_pad", &mp);
    mcTree->Branch("cluster_max_time", &mt);
    mcTree->Branch("cluster_max_q", &mq);
    mcTree->Branch("cluster_mc_noise", &mcn);
    mcTree->Branch("cluster_mc_fake", &mcf);
    mcTree->Branch("cluster_points", &p);

    for(int i = 0; i<elem_counter; i++){
      sec = sector_vec[i];
      r = row_vec[i];
      cp = cog_pad[i];
      ct = cog_time[i];
      cq = cog_q[i];
      mcn = mc_noise[i];
      mcf = mc_fake[i];
      for(auto elem : max_q[i]){
        sp += std::pow(max_pad[i][idx] - cog_pad[i], 2);
        st += std::pow(max_time[i][idx] - cog_time[i], 2);
        if(elem > mq){
          mp = max_pad[i][idx];
          mt = max_time[i][idx];
          mq = elem;
        }
        else if(elem==mq){
          if((std::pow(mp-cp,2) + std::pow(mt-ct,2)) > (std::pow(mp-max_pad[i][idx],2) + std::pow(mt-max_time[i][idx],2))){
            mp = max_pad[i][idx];
            mt = max_time[i][idx];
            mq = elem;
          }
        }
        idx++;
      }
      if(mq > 3 && cq > 5){
        sp = std::sqrt(sp/max_q[i].size());
        st = std::sqrt(st/max_q[i].size());
        p = point_counter[i];
        mcTree->Fill();
      }
      sp = 0, st = 0, mp = 0; mt = 0; mq = 0; mcn=0; mcf=0; idx = 0;
    }

    mcTree->Write();
    delete mcTree;
    outputFile.Close();

    std::stringstream tmp2;
    tmp2 << "sector_" << mSector;
    std::stringstream fileName2;
    fileName2 << "mclabels_ideal_full_" << mSector << ".root";
    TFile outputFile2(fileName2.str().c_str(), "RECREATE");
    TTree* mcTree2 = new TTree(tmp2.str().c_str(), "MC tree");

    sec=0, r=0, mp=0, mt=0, idx=0; mcn=0; mcf=0; mq=0;

    mcTree2->Branch("cluster_sector", &sec);
    mcTree2->Branch("cluster_row", &r);
    mcTree2->Branch("cluster_pad", &mp);
    mcTree2->Branch("cluster_time", &mt);
    mcTree2->Branch("cluster_q", &mq);
    mcTree2->Branch("cluster_mc_noise", &mcn);
    mcTree2->Branch("cluster_mc_fake", &mcf);
    mcTree2->Branch("cluster_label", &lab);

    for(int i = 0; i<elem_counter; i++){
      sec = sector_vec[i];
      r = row_vec[i];
      lab = mclabel[i];
      mcn = mc_noise[i];
      mcf = mc_fake[i];
      for(auto elem : max_q[i]){
        mp = max_pad[i][idx];
        mt = max_time[i][idx];
        mq = elem;
        idx++;

        mcTree2->Fill();
      }
      mp = 0; mt = 0; mq = 0; idx = 0; mcn=0; mcf=0; lab = 0;
    }

    mcTree2->Write();
    delete mcTree2;
    outputFile2.Close();

    timer.Stop();
    LOG(info) << "TPC: Digitization took " << timer.CpuTime() << "s";
  }

 private:
  o2::tpc::Digitizer mDigitizer;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  std::vector<TChain*> mSimChains;
  std::vector<o2::tpc::Digit> mDigits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
  std::vector<o2::tpc::CommonMode> mCommonMode;
  std::vector<int> mListOfSectors; //  a list of sectors treated by this task
  TFile* mInternalROOTFlushFile = nullptr;
  TTree* mInternalROOTFlushTTree = nullptr;
  size_t mDigitCounter = 0;
  size_t mFlushCounter = 0;
  int mLaneId = 0; // the id of the current process within the parallel pipeline
  int mSector = 0;
  bool mWriteGRP = false;
  bool mWithMCTruth = true;
  bool mInternalWriter = false;
  bool mUseCalibrationsFromCCDB = false;

  /// OWN IMPLEMENTATION
  int64_t elem_counter = 0;
  std::vector<int> tmp_sector_vec, tmp_row_vec, tmp_point_counter, tmp_mc_label, tmp_mc_noise, tmp_mc_fake;
  std::vector<std::vector<int>>  tmp_max_time, tmp_max_pad;
  std::vector<float> tmp_cog_time, tmp_cog_pad, tmp_cog_q;
  std::vector<std::vector<float>> tmp_max_q;

  std::vector<int> sector_vec, row_vec, point_counter, mclabel, mc_noise, mc_fake;
  std::vector<float> cog_time, cog_pad, cog_q;
  std::vector<std::vector<int>>  max_time, max_pad;
  std::vector<std::vector<float>> max_q;

  std::vector<int> window_size;
};

o2::framework::DataProcessorSpec getTPCDigitizerSpec(int channel, bool writeGRP, bool mctruth, bool internalwriter)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::stringstream id;
  id << "TPCDigitizer" << channel;

  std::vector<OutputSpec> outputs; // define channel by triple of (origin, type id of data to be sent on this channel, subspecification)

  if (!internalwriter) {
    outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    outputs.emplace_back("TPC", "DIGTRIGGERS", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back("TPC", "DIGITSMCTR", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
    }
    outputs.emplace_back("TPC", "COMMONMODE", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  }
  if (writeGRP) {
    outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
    LOG(debug) << "TPC: Channel " << channel << " will supply ROMode";
  }

  std::vector<InputSpec> inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}};
  return DataProcessorSpec{
    id.str().c_str(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerTask>(internalwriter)},
    Options{
      {"distortionType", VariantType::Int, 0, {"Distortion type to be used. 0 = no distortions (default), 1 = realistic distortions (not implemented yet), 2 = constant distortions"}},
      {"initialSpaceChargeDensity", VariantType::String, "", {"Path to root file containing TH3 with initial space-charge density and name of the TH3 (comma separated)"}},
      {"readSpaceCharge", VariantType::String, "", {"Path to root file containing pre-calculated space-charge object and name of the object (comma separated)"}},
      {"TPCtriggered", VariantType::Bool, false, {"Impose triggered RO mode (default: continuous)"}},
      {"TPCuseCCDB", VariantType::Bool, false, {"true: load calibrations from CCDB; false: use random calibratoins"}},
    }};
}

o2::framework::WorkflowSpec getTPCDigitizerSpec(int nLanes, std::vector<int> const& sectors, bool mctruth, bool internalwriter)
{
  // channel parameter is deprecated in the TPCDigitizer processor, all descendants
  // are initialized not to publish GRP mode, but the channel will be added to the first
  // processor after the pipelines have been created. The processor will decide upon
  // the index in the ParallelContext whether to publish
  WorkflowSpec pipelineTemplate{getTPCDigitizerSpec(0, false, mctruth, internalwriter)};
  // override the predefined name, index will be added by parallelPipeline method
  pipelineTemplate[0].name = "TPCDigitizer";
  WorkflowSpec pipelines = parallelPipeline(
    pipelineTemplate, nLanes, [size = sectors.size()]() { return size; }, [&sectors](size_t index) { return sectors[index]; });
  // add the channel for the GRP information to the first processor
  for (auto& spec : pipelines) {
    o2::tpc::VDriftHelper::requestCCDBInputs(spec.inputs); // add the same CCDB request to each pipeline
  }
  pipelines[0].outputs.emplace_back("TPC", "ROMode", 0, Lifetime::Timeframe);
  return pipelines;
}

} // end namespace tpc
} // end namespace o2

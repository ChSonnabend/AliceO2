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

/// \file Digitizer.cxx
/// \brief Implementation of the ALICE TPC digitizer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TH3.h"

#include "TPCSimulation/Digitizer.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/CDBInterface.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCBase/Mapper.h"

#include <fairlogger/Logger.h>

#include "TFile.h"
#include "TTree.h"

ClassImp(o2::tpc::Digitizer);

using namespace o2::tpc;

Digitizer::~Digitizer() = default;

Digitizer::Digitizer() = default;

void Digitizer::init()
{
  // Calculate distortion lookup tables if initial space-charge density is provided
  if (mUseSCDistortions) {
    mSpaceCharge->init();
  }
  auto& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();
  auto& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters(mVDrift);
  auto& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters(mVDrift);
}

void Digitizer::process(const std::vector<o2::tpc::HitGroup>& hits,
                        const int eventID, const int sourceID)
{
  const Mapper& mapper = Mapper::instance();
  auto& detParam = ParameterDetector::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  auto& gemParam = ParameterGEM::Instance();

  auto& gemAmplification = GEMAmplification::instance();
  auto& electronTransport = ElectronTransport::instance();
  auto& sampaProcessing = SAMPAProcessing::instance();

  const int nShapedPoints = eleParam.NShapedPoints;
  const auto amplificationMode = gemParam.AmplMode;
  static std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

  /// Reserve space in the digit container for the current event
  mDigitContainer.reserve(sampaProcessing.getTimeBinFromTime(mEventTime - mOutputDigitTimeOffset));

  /// obtain max drift_time + hitTime which can be processed
  float maxEleTime = (int(mDigitContainer.size()) - nShapedPoints) * eleParam.ZbinWidth;

  sector.clear();
  row.clear();
  max_time.clear();
  max_pad.clear();
  max_q.clear();
  cog_time.clear();
  cog_pad.clear();
  cog_q.clear();
  point_counter.clear();
  mclabel.clear();
  elem_counter = 0;

  for (auto& hitGroup : hits) {
    const int MCTrackID = hitGroup.GetTrackID();
    for (size_t hitindex = 0; hitindex < hitGroup.getSize(); ++hitindex) {
      const auto& eh = hitGroup.getHit(hitindex);

      GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

      // Distort the electron position in case space-charge distortions are used
      if (mUseSCDistortions) {
        mSpaceCharge->distortElectron(posEle);
      }

      /// Remove electrons that end up more than three sigma of the hit's average diffusion away from the current sector
      /// boundary
      if (electronTransport.isCompletelyOutOfSectorCoarseElectronDrift(posEle, mSector)) {
        continue;
      }

      /// The energy loss stored corresponds to nElectrons
      const int nPrimaryElectrons = static_cast<int>(eh.GetEnergyLoss());
      const float hitTime = eh.GetTime() * 0.001; /// in us
      float driftTime = 0.f;

      /// TODO: add primary ions to space-charge density

      /// Loop over electrons
      for (int iEle = 0; iEle < nPrimaryElectrons; ++iEle) {

        /// Drift and Diffusion
        const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle, driftTime);
        const float eleTime = driftTime + hitTime; /// in us
        if (eleTime > maxEleTime) {
          LOG(warning) << "Skipping electron with driftTime " << driftTime << " from hit at time " << hitTime;
          continue;
        }
        const float absoluteTime = eleTime + mTDriftOffset + (mEventTime - mOutputDigitTimeOffset); /// in us

        /// Attachment
        if (electronTransport.isElectronAttachment(driftTime)) {
          continue;
        }

        /// Remove electrons that end up outside the active volume
        if (std::abs(posEleDiff.Z()) > detParam.TPClength) {
          continue;
        }

        /// When the electron is not in the sector we're processing, abandon
        if (mapper.isOutOfSector(posEleDiff, mSector)) {
          continue;
        }

        /// Compute digit position and check for validity
        const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff, mSector);
        if (!digiPadPos.isValid()) {
          continue;
        }

        /// Remove digits the end up outside the currently produced sector
        if (digiPadPos.getCRU().sector() != mSector) {
          continue;
        }

        /// Electron amplification
        const int nElectronsGEM = gemAmplification.getStackAmplification(digiPadPos.getCRU(), digiPadPos.getPadPos(), amplificationMode);
        if (nElectronsGEM == 0) {
          continue;
        }

        const GlobalPadNumber globalPad = mapper.globalPadNumber(digiPadPos.getGlobalPadPos());
        const float ADCsignal = sampaProcessing.getADCvalue(static_cast<float>(nElectronsGEM));
        const MCCompLabel label(MCTrackID, eventID, sourceID, false);
        sampaProcessing.getShapedSignal(ADCsignal, absoluteTime, signalArray);

        float currentSignal = 0;
        int currentPadPos = 0, currentTimeBin = 0, currentRow = 0, currentSector = 0, label_counter = 0, max_idx = 0;
        bool track_found = false, max_idx_found = false;

        for (float i = 0; i < nShapedPoints; ++i) {
          const float time = absoluteTime + i * eleParam.ZbinWidth;
          mDigitContainer.addDigit(label, digiPadPos.getCRU(), sampaProcessing.getTimeBinFromTime(time), globalPad,
                                   signalArray[i]);
          
          /// OWN IMPLEMENTATION
          if((float)signalArray[i]>0 && !std::isnan((float)signalArray[i])){

            currentSignal = (float)(signalArray[i]); currentPadPos = (int)(digiPadPos.getGlobalPadPos().getPad()); currentTimeBin = (int)(sampaProcessing.getTimeBinFromTime(time)); currentRow = (int)(digiPadPos.getGlobalPadPos().getRow()); currentSector = (int)(digiPadPos.getCRU().sector());
            label_counter = 0; track_found = false; max_idx = 0; max_idx_found = false;

            for(auto lab : mclabel){
              track_found = (label.compare(lab)==1) && (row[label_counter]==currentRow) && (std::abs(currentTimeBin - cog_time[label_counter])<=(window_size[0]/2))  && (std::abs(currentPadPos - cog_pad[label_counter])<=(window_size[1]/2));
              if(track_found){
                break;
              }
              else{
                label_counter++;
              }
            }
            if(track_found){

              // if(currentSignal>max_q[label_counter]){
              //   max_q[label_counter] = currentSignal;
              //   max_time[label_counter] =  currentTimeBin;
              //   max_pad[label_counter] = currentPadPos;
              // }

              for(auto elem : max_q[label_counter]){
                if((max_time[label_counter][max_idx] == currentTimeBin) && (max_pad[label_counter][max_idx] == currentPadPos)){
                  max_q[label_counter][max_idx] += currentSignal;
                  max_idx_found = true;
                  max_idx = 0;
                  break;
                }
                else{
                  max_idx++;
                }
              }

              if(!max_idx_found){
                max_time[label_counter].push_back(currentTimeBin);
                max_pad[label_counter].push_back(currentPadPos);
                max_q[label_counter].push_back(currentSignal);
              }

              /// On-the-fly center-of-gravity calculation
              cog_time[label_counter] = (cog_time[label_counter]*cog_q[label_counter] + currentTimeBin*currentSignal)/(cog_q[label_counter] + currentSignal);
              cog_pad[label_counter] = (cog_pad[label_counter]*cog_q[label_counter] + currentPadPos*currentSignal)/(cog_q[label_counter] + currentSignal);
              cog_q[label_counter] += currentSignal;

              /// Point counter
              point_counter[label_counter] += 1;

              // if(currentSector==2 && currentRow==6){
              //   LOG(info) << "Current point: (pad) " << currentPadPos << ", (time) " << currentTimeBin << ", (charge) " << currentSignal;
              //   LOG(info) << "New Max: (pad) " << max_pad[label_counter] << ", (time) " << max_time[label_counter] << ", (charge) " << max_q[label_counter];
              //   LOG(info) << "New CoG: (pad) " << cog_pad[label_counter] << ", (time) " << cog_time[label_counter] << ", (charge) " << cog_q[label_counter];
              // }
            }
            else{
              sector.push_back(currentSector);
              row.push_back(currentRow);
              max_time.push_back(std::vector<int>{currentTimeBin});
              max_pad.push_back(std::vector<int>{currentPadPos});
              max_q.push_back(std::vector<float>{currentSignal});
              cog_time.push_back(currentTimeBin);
              cog_pad.push_back(currentPadPos);
              cog_q.push_back(currentSignal);
              point_counter.push_back(1);
              mclabel.push_back(label);
              elem_counter++;
            }
          }
          
        }
        /// TODO: add ion backflow to space-charge density
      }
      /// end of loop over electrons
    }
  }
}

void Digitizer::flush(std::vector<o2::tpc::Digit>& digits,
                      o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels,
                      std::vector<o2::tpc::CommonMode>& commonModeOutput,
                      bool finalFlush)
{
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  mDigitContainer.fillOutputContainer(digits, labels, commonModeOutput, mSector, sampaProcessing.getTimeBinFromTime(mEventTime - mOutputDigitTimeOffset), mIsContinuous, finalFlush);
  // flushing debug output to file
  if (((finalFlush && mIsContinuous) || (!mIsContinuous)) && mSpaceCharge) {
    o2::utils::DebugStreamer::instance()->flush();
  }
}

void Digitizer::setUseSCDistortions(const SCDistortionType& distortionType, const TH3* hisInitialSCDensity)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceCharge = std::make_unique<SC>();
  }
  mSpaceCharge->setSCDistortionType(distortionType);
  if (hisInitialSCDensity) {
    mSpaceCharge->fillChargeDensityFromHisto(*hisInitialSCDensity);
    mSpaceCharge->setUseInitialSCDensity(true);
  }
}

void Digitizer::setUseSCDistortions(SC* spaceCharge)
{
  mUseSCDistortions = true;
  mSpaceCharge.reset(spaceCharge);
}

void Digitizer::setUseSCDistortions(std::string_view finp)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceCharge = std::make_unique<SC>();
  }

  // in case analytical distortions are loaded from file they are applied
  mSpaceCharge->setAnalyticalCorrectionsDistortionsFromFile(finp);
  if (!mSpaceCharge->getUseAnalyticalDistCorr()) {
    mSpaceCharge->setGlobalDistortionsFromFile(finp, Side::A);
    mSpaceCharge->setGlobalDistortionsFromFile(finp, Side::C);
  }
}

void Digitizer::setStartTime(double time)
{
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();
  mDigitContainer.setStartTime(sampaProcessing.getTimeBinFromTime(time - mOutputDigitTimeOffset));
}

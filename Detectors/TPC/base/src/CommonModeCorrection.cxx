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

/// \file   CommonModeCorrection.cxx
/// \brief  Calculate the common mode correction factor

// #include <random>
#include <algorithm>
#include <thread>
#include <mutex>
#include "CCDB/CcdbApi.h"
#include "TPCBase/CommonModeCorrection.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CRUCalibHelpers.h"
#include "TChain.h"
#include "TROOT.h"
#include "TFile.h"
#include "MathUtils/RandomRing.h"
#include "CommonUtils/TreeStreamRedirector.h"

using namespace o2::tpc;
using namespace o2::tpc::cru_calib_helpers;
CommonModeCorrection::CMInfo CommonModeCorrection::getCommonMode(gsl::span<const float> values, gsl::span<const float> cmKValues, gsl::span<const float> pedestals, CMDebug* cmDebug) const
{
  if (values.size() == 0) {
    return CMInfo{};
  }
  // sanity check
  if (values.size() != cmKValues.size() || values.size() != pedestals.size()) {
    LOGP(error, "vector sizes of input values, cmKValues and pedestals don't match: {}, {}, {}", values.size(), cmKValues.size(), pedestals.size());
    return CMInfo{};
  }
  static math_utils::RandomRing random(math_utils::RandomRing<>::RandomType::Flat);
  std::vector<float> adcCM; //< ADC values used for common mode calculation

  CMInfo cmInfo;
  if (cmDebug) {
    cmDebug->nPadsOk.resize(mNPadsCompRamdom + 1);
    cmDebug->adcDist.resize(10);
  }

  for (size_t iPad = 0; iPad < values.size(); ++iPad) {
    const float kCM = mLimitKFactor ? fixedSizeToFloat<6>(floatToFixedSize<8, 6>(cmKValues[iPad])) : cmKValues[iPad];
    const float pedestal = mLimitPedestal ? fixedSizeToFloat(floatToFixedSize(pedestals[iPad])) : pedestals[iPad];
    const float adcPadRaw = values[iPad];
    const float adcPad = adcPadRaw - pedestal;
    const float adcPadNorm = (kCM > 0) ? adcPad / kCM : 0;

    if (adcPadRaw > 1023.7) {
      ++cmInfo.nSaturation;
    }

    if (adcPad > mQEmpty) {
      continue;
    }

    float qCompAdd = 0;
    if ((mQCompScaleThreshold < 0) && (adcPadNorm < mQCompScaleThreshold)) {
      qCompAdd = (mQCompScaleThreshold - adcPadNorm) * mQCompScale;
      LOGP(info, "Setting qCompAdd to {} for {}", qCompAdd, adcPadNorm);
    }

    int nPadsOK = 0;

    for (int iRnd = 0; iRnd < mNPadsCompRamdom; ++iRnd) {
      int padRnd = 0;
      do {
        padRnd = int(random.getNextValue() * (values.size() - 1));
      } while (padRnd == iPad);
      const float kCMRnd = mLimitKFactor ? fixedSizeToFloat<6>(floatToFixedSize<8, 6>(cmKValues[padRnd])) : cmKValues[padRnd];
      const float pedestalRnd = mLimitPedestal ? fixedSizeToFloat(floatToFixedSize(pedestals[padRnd])) : pedestals[padRnd];
      const float adcPadRnd = values[padRnd] - pedestalRnd;
      const float adcPadRndNorm = (kCMRnd > 0) ? adcPadRnd / kCMRnd : 0;
      const float adcDist = std::abs(adcPadNorm - adcPadRndNorm);
      if (cmDebug) {
        const size_t distPos = std::min(cmDebug->adcDist.size() - 1, size_t(adcDist / 0.5));
        ++cmDebug->adcDist[distPos];
      }
      if (adcDist < mQComp) {
        ++nPadsOK;
      }
    }

    if (cmDebug) {
      ++cmDebug->nPadsOk[nPadsOK];
    }

    if (nPadsOK >= mNPadsCompMin) {
      adcCM.emplace_back(adcPadNorm);
    }
  }

  const int entriesCM = int(adcCM.size());
  float commonMode = 0; // std::accumulate(adcCM.begin(), adcCM.end(), 0.f);
  float commonModeStd = 0;

  if (entriesCM > 0) {
    std::for_each(adcCM.begin(), adcCM.end(), [&commonMode, &commonModeStd](const auto val) {
      commonMode += val;
      commonModeStd += val * val;
    });
    commonMode /= float(entriesCM);
    commonModeStd = std::sqrt(std::abs(commonModeStd / entriesCM - commonMode * commonMode));
  }
  cmInfo.cmValue = commonMode;
  cmInfo.cmValueStd = commonModeStd;
  cmInfo.nPadsUsed = entriesCM;

  for (size_t iPad = 0; iPad < values.size(); ++iPad) {
    const float kCM = mLimitKFactor ? fixedSizeToFloat<6>(floatToFixedSize<8, 6>(cmKValues[iPad])) : cmKValues[iPad];
    const float pedestal = mLimitPedestal ? fixedSizeToFloat(floatToFixedSize(pedestals[iPad])) : pedestals[iPad];
    const float adcPadRaw = values[iPad];
    const float adcPad = adcPadRaw - pedestal;
    const float adcPadNorm = (kCM > 0) ? adcPad / kCM : 0;
    const float adcPadCorr = adcPad - kCM * commonMode;

    if (adcPadCorr > mSumPosThreshold) {
      cmInfo.sumPos += adcPadCorr;
    } else {
      cmInfo.sumNeg += adcPadNorm;
      ++cmInfo.nNeg;
    }

    if (mOccupancyThreshold > 0) {
      if (adcPadCorr > mOccupancyThreshold) {
        ++cmInfo.nOccupancy;
      }
    }
  }

  return cmInfo;
}

void CommonModeCorrection::loadDefaultPadMaps(FEEConfig::Tags tag)
{
  o2::ccdb::CcdbApi cdbApi;
  cdbApi.init("http://alice-ccdb.cern.ch");
  const auto feeConfig = cdbApi.retrieveFromTFileAny<FEEConfig>("TPC/Config/FEE", {}, long(tag));
  if (!feeConfig) {
    LOGP(error, "Could not retrieve pad maps");
    return;
  }
  mPadMaps = feeConfig->padMaps;
  delete feeConfig;
}

CommonModeCorrection::CMdata CommonModeCorrection::collectCMdata(const std::vector<Digit>& digits, int cru, int timeBin)
{

  CMdata data;
  if (!padMapExists("CMkValues") || padMapExists("Pedestals")) {
    return data;
  }

  for (const auto& digit : digits) {
    if (digit.getTimeStamp() < timeBin) {
      continue;
    }

    if (digit.getTimeStamp() > timeBin) {
      break;
    }

    if (digit.getCRU() < cru) {
      continue;
    }

    if (digit.getCRU() > cru) {
      break;
    }

    const auto sector = CRU(digit.getCRU()).sector();
    data.adcValues.emplace_back(digit.getChargeFloat());
    data.cmKValues.emplace_back(mPadMaps["CMkValues"].getValue(sector, digit.getRow(), digit.getPad()));
    data.pedestals.emplace_back(mPadMaps["Pedestals"].getValue(sector, digit.getRow(), digit.getPad()));
  }
  return data;
}

int CommonModeCorrection::getCommonMode(std::vector<Digit>& digits, std::vector<std::vector<CMInfo>>& cmValues, bool negativeOnly, bool hasInjectedCMValue, std::vector<std::vector<CMDebug>>* cmDebug, int minTimeBin, int maxTimeBin) const
{
  // calculation common mode values
  int maxTimeBinProcessed = -1;
  int lastCRU = -1;
  int lastTimeBin = -1;
  CMdata data;
  const auto& cmkValues = mPadMaps.at("CMkValues");
  const auto& pedestals = mPadMaps.at("Pedestals");

  bool doArtificialCM = std::abs(mArtificialCM) > 0;

  // for decoding of the injected common mode signals
  float cmInjectedLower{};
  float cmInjectedUpper{};

  for (size_t iDigit = 0; iDigit < digits.size(); ++iDigit) {
    auto& digit = digits[iDigit];
    const auto timeBin = digit.getTimeStamp();
    if ((minTimeBin > -1) && (timeBin < minTimeBin)) {
      continue;
    }
    if ((maxTimeBin > -1) && (timeBin > maxTimeBin)) {
      continue;
    }
    if ((lastCRU > -1) && ((digit.getCRU() != lastCRU) || (digit.getTimeStamp() != lastTimeBin))) {
      auto& cmValuesCRU = cmValues[lastCRU];
      if (cmValuesCRU.size() <= lastTimeBin) {
        cmValuesCRU.resize(lastTimeBin + 500);
        if (cmDebug) {
          (*cmDebug)[lastCRU].resize(lastTimeBin + 500);
        }
      }
      if (mSubthreshold > 0) {
        const size_t nPadsCRU = Mapper::PADSPERREGION[lastCRU % 10];
        const auto dataSize = data.adcValues.size();
        if (dataSize < nPadsCRU) {
          data.resize(nPadsCRU);
          if (mSubthreshold == 2) {
            for (size_t i = dataSize; i < nPadsCRU; ++i) {
              data.adcValues[i] = gRandom->Gaus();
            }
          }
        }
      }
      cmValuesCRU[lastTimeBin] = getCommonMode(data.adcValues, data.cmKValues, data.pedestals, cmDebug ? &((*cmDebug)[lastCRU][lastTimeBin]) : nullptr);
      if (hasInjectedCMValue) {
        cmValuesCRU[lastTimeBin].cmValueCRU = decodeInjectedCMValue(cmInjectedLower, cmInjectedUpper);
      }
      // LOGP(info, "processing CRU {}, timeBin {}, CM = {}", lastCRU, lastTimeBin, cmValuesCRU[lastTimeBin].cmValue);

      data.clear();
    }
    const auto sector = CRU(digit.getCRU()).sector();
    const auto cmkValue = cmkValues.getValue(sector, digit.getRow(), digit.getPad());
    const auto pedestal = pedestals.getValue(sector, digit.getRow(), digit.getPad());
    float charge = digit.getChargeFloat();
    if (doArtificialCM) {
      charge = std::clamp(charge + mArtificialCM * cmkValue, 0.f, 1023.f);
    }
    lastCRU = digit.getCRU();
    lastTimeBin = timeBin;
    maxTimeBinProcessed = std::max(lastTimeBin, maxTimeBinProcessed);

    bool isInjectedCMPad = false;
    if (hasInjectedCMValue) {
      const auto posLow = mCMInjectIDLower[lastCRU % 10];
      const auto posUpper = mCMInjectIDUpper[lastCRU % 10];
      const auto row = digit.getRow();
      const auto pad = digit.getPad();
      if (row == posLow.row) {
        if (pad == posLow.pad) {
          cmInjectedLower = digit.getChargeFloat();
          isInjectedCMPad = true;
          // LOGP(info, "setting lower CM value cru {}, row {}, pad {}: {:012b}", digit.getCRU(), row, pad, floatToFixedSize(digit.getChargeFloat()));
        }
      }
      if (row == posUpper.row) {
        if (pad == posUpper.pad) {
          cmInjectedUpper = digit.getChargeFloat();
          isInjectedCMPad = true;
          // LOGP(info, "setting upper CM value cru {}, row {}, pad {}: {:012b}", digit.getCRU(), row, pad, floatToFixedSize(digit.getChargeFloat()));
          if (cmInjectedUpper == 0) {
            LOGP(info, "cm upper = 0 cru {}, row {}, pad {}", digit.getCRU(), row, pad);
          }
        }
      }
    }

    if (!isInjectedCMPad) {
      data.adcValues.emplace_back(charge);
      data.cmKValues.emplace_back(cmkValue);
      data.pedestals.emplace_back(pedestal);
    }
  }
  {
    auto& cmValuesCRU = cmValues[lastCRU];
    if (cmValuesCRU.size() <= lastTimeBin) {
      cmValuesCRU.resize(lastTimeBin + 500);
      if (cmDebug) {
        (*cmDebug)[lastCRU].resize(lastTimeBin + 500);
      }
    }
    cmValuesCRU[lastTimeBin] = getCommonMode(data.adcValues, data.cmKValues, data.pedestals, cmDebug ? &((*cmDebug)[lastCRU][lastTimeBin]) : nullptr);
    // LOGP(info, "processing CRU {}, timeBin {}, CM = {}", lastCRU, lastTimeBin, cmValuesCRU[lastTimeBin].cmValue);

    if (hasInjectedCMValue) {
      cmValuesCRU[lastTimeBin].cmValueCRU = decodeInjectedCMValue(cmInjectedLower, cmInjectedUpper);
    }

    data.clear();
  }
  return maxTimeBinProcessed;
}

int CommonModeCorrection::correctDigits(std::vector<Digit>& digits, std::vector<std::vector<CMInfo>>& cmValues, bool negativeOnly, bool hasInjectedCMValue, std::vector<std::vector<CMDebug>>* cmDebug, int minTimeBin, int maxTimeBin) const
{
  const auto maxTimeBinProcessed = getCommonMode(digits, cmValues, negativeOnly, hasInjectedCMValue, cmDebug, minTimeBin, maxTimeBin);
  const auto& cmkValues = mPadMaps.at("CMkValues");
  const auto& pedestals = mPadMaps.at("Pedestals");
  // ===| apply correction |====
  for (auto& digit : digits) {
    const auto timeBin = digit.getTimeStamp();
    if ((minTimeBin > -1) && (timeBin < minTimeBin)) {
      continue;
    }
    if ((maxTimeBin > -1) && (timeBin > maxTimeBin)) {
      continue;
    }
    const auto sector = CRU(digit.getCRU()).sector();
    const auto cmKValue = cmkValues.getValue(sector, digit.getRow(), digit.getPad());
    // LOGP(info, "correcting value for CRU {}, time bin {}", digit.getCRU(), digit.getTimeStamp());
    const auto cmValue = cmValues[digit.getCRU()][digit.getTimeStamp()].cmValue;
    const auto cmNPads = cmValues[digit.getCRU()][digit.getTimeStamp()].nPadsUsed;
    if ((!negativeOnly || cmValue < 0) && (cmNPads > mNPadsMinCM)) {
      digit.setCharge(digit.getCharge() - cmValue * cmKValue);
      if (mCorrectOutputForPedestal) {
        const auto sector = CRU(digit.getCRU()).sector();
        const auto pedestal = pedestals.getValue(sector, digit.getRow(), digit.getPad());
        digit.setCharge(digit.getChargeFloat() - pedestal);
      }
    }
  }

  return maxTimeBinProcessed;
}

void CommonModeCorrection::correctDigits(std::string_view digiFileIn, Long64_t maxEntries, std::string_view digitFileOut, std::string_view cmFileOut, bool negativeOnly, int nThreads, bool writeOnlyCM, bool writeDebug, bool hasInjectedCMValue, int minTimeBin, int maxTimeBin)
{
  ROOT::EnableThreadSafety();

  TChain* tree = o2::tpc::utils::buildChain(fmt::format("ls {}", digiFileIn), "o2sim", "o2sim");
  Long64_t nEntries = tree->GetEntries();
  if (maxEntries > 0) {
    nEntries = std::min(nEntries, maxEntries);
  }

  if (mPadMaps.find("Pedestals") == mPadMaps.end()) {
    LOGP(info, "Using empty pedestals");
    mPadMaps["Pedestals"] = CalPad("Pedestals");
  }

  std::unique_ptr<TFile> fOut;
  std::unique_ptr<TTree> tOut;
  if (!writeOnlyCM) {
    fOut.reset(TFile::Open(digitFileOut.data(), "RECREATE"));
    fOut->SetCompressionLevel(5);     // zstd default level
    fOut->SetCompressionAlgorithm(5); // zstd
    tOut = std::make_unique<TTree>("o2sim", "o2sim");
  }

  std::array<std::vector<o2::tpc::Digit>*, 36> digitizedSignal;
  std::array<TBranch*, 36> outBranches{};
  for (size_t iSec = 0; iSec < digitizedSignal.size(); ++iSec) {
    digitizedSignal[iSec] = nullptr;
    tree->SetBranchAddress(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
    if (tOut) {
      outBranches[iSec] = tOut->Branch(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
    }
  }

  o2::utils::TreeStreamRedirector pcstream(cmFileOut.data(), "recreate");
  pcstream.GetFile()->SetCompressionAlgorithm(5);
  pcstream.GetFile()->SetCompressionLevel(5);

  for (Long64_t iTF = 0; iTF < nEntries; ++iTF) {
    tree->GetEntry(iTF);
    LOGP(info, "Processing entry {}/{}", iTF + 1, nEntries);

    std::vector<std::vector<CMInfo>> cmValues; // CRU * timeBin
    std::vector<std::vector<CMDebug>> cmDebug; // CRU * timeBin

    cmValues.resize(CRU::MaxCRU);
    if (writeDebug) {
      cmDebug.resize(CRU::MaxCRU);
    }
    int maxTimeBinSeen = -1;

    auto worker = [&](int iTread) {
      // for (size_t iSector = 0; iSector < 36; ++iSector) {
      for (size_t iSector = iTread; iSector < 36; iSector += nThreads) {
        LOGP(info, "Processing entry {}/{}, starting sector {}", iTF + 1, nEntries, iSector);
        auto digits = digitizedSignal[iSector];
        int maxTimeBinSector = 0;
        if (digits && (digits->size() > 0)) {
          maxTimeBinSector = correctDigits(*digits, cmValues, negativeOnly, hasInjectedCMValue, writeDebug ? &cmDebug : nullptr, minTimeBin, maxTimeBin);
        }
        {
          static std::mutex maxMutex;
          std::lock_guard lock{maxMutex};
          maxTimeBinSeen = std::max(maxTimeBinSeen, maxTimeBinSector);
          if (outBranches[iSector]) {
            outBranches[iSector]->Fill();
            LOGP(info, "Filling branch for sector {}", iSector);
          }
        }
      }
    };

    std::vector<std::thread> threads(nThreads);

    for (int i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(worker, i);
    }

    // wait for the threads to finish
    for (auto& th : threads) {
      th.join();
    }

    size_t maxTimeCRU = 0;
    for (int iCRU = 0; iCRU < cmValues.size(); ++iCRU) {
      maxTimeCRU = std::max(maxTimeCRU, cmValues[iCRU].size());
    }
    const int maxTBCRU = std::min(maxTimeBinSeen, int(maxTimeCRU));

    for (int iTimeBin = 0; iTimeBin < maxTBCRU; ++iTimeBin) {

      std::vector<CMInfo> cm(CRU::MaxCRU);
      std::vector<CMDebug> cmD(CRU::MaxCRU);
      std::vector<float> sumPosStack(36 * 4);
      std::vector<float> nPosStack(36 * 4);
      std::vector<float> nSaturationStack(36 * 4);
      std::vector<float> sumPosStackCRU(CRU::MaxCRU);
      std::vector<float> sumPosStackCRUCorr(CRU::MaxCRU);
      std::vector<float> nSaturationStackCRU(CRU::MaxCRU);

      for (int iCRU = 0; iCRU < cmValues.size(); ++iCRU) {
        if (cmValues[iCRU].size() == 0) {
          continue;
        }
        cm[iCRU] = cmValues[iCRU][iTimeBin];
        if (writeDebug) {
          cmD[iCRU] = cmDebug[iCRU][iTimeBin];
        }
        const CRU cru(iCRU);
        const StackID stackID{cru.sector(), cru.gemStack()};
        const auto index = stackID.getIndex();
        sumPosStack[index] += cm[iCRU].sumPos;
        nPosStack[index] += (Mapper::PADSPERREGION[cru.region()] - cm[iCRU].nNeg);
        nSaturationStack[index] += cm[iCRU].nSaturation;
      }

      for (int iCRU = 0; iCRU < cmValues.size(); ++iCRU) {
        if (cmValues[iCRU].size() == 0) {
          continue;
        }
        const CRU cru(iCRU);
        const StackID stackID{cru.sector(), cru.gemStack()};
        const auto index = stackID.getIndex();
        sumPosStackCRU[iCRU] = sumPosStack[index];
        sumPosStackCRUCorr[iCRU] = sumPosStack[index] - nPosStack[index] * cm[iCRU].cmValue;
        nSaturationStackCRU[iCRU] = nSaturationStack[index];
      }

      pcstream << "cm"
               << "iTF=" << iTF
               << "iTimeBin=" << iTimeBin
               << "cmInfo=" << cm
               << "sumPosStack=" << sumPosStackCRU
               << "sumPosStackCorr=" << sumPosStackCRUCorr
               << "nSaturationStack=" << nSaturationStackCRU;

      if (writeDebug) {
        pcstream << "cm"
                 << "cmDebug=" << cmD;
      }

      pcstream << "cm"
               << "\n";
    }

    // if (tOut) {
    //   tOut->Fill();
    // }
  }

  pcstream.Close();
  if (fOut && tOut) {
    tOut->SetEntries(nEntries);
    fOut->cd();
    tOut->Write();
    tOut.reset();
    fOut->Close();
  }
}

float CommonModeCorrection::decodeInjectedCMValue(float lower, float upper)
{
  // CRU  row0 pad0 row1 pad1
  // 0     0    2    0    3
  // 1    20    1   20    3
  // 2    32    2   32    3
  // 3    51    1   51    3
  // 4    62    1   62    2
  // 5    84    1   84    4
  // 6    97    1   97    2
  // 7   116    2  115    5
  // 8   127    2  127    3
  // 9   142    0  142    4
  //
  // CM Value encoding:
  // Kanal 0 : Bit 11 ... 8 = 0x8. Bit 7..0 CM-Werte Bits 7...0
  // Kanal 1 : Bit 11.. 9 = "100". Bit 8 = CM Positive, Bits 6..0 = CM-Wert Bits 14..8
  const int ilower = floatToFixedSize(lower);
  const int iupper = floatToFixedSize(upper);
  if (!(ilower & 0x800) || !(iupper & 0x800)) {
    LOGP(error, "Not a CM word: lower: {:012b} upper: {:012b}", ilower, iupper);
    return 0;
  }
  const int fixedSizeCM = ((iupper & 0x7F) << 8) + (ilower & 0xFF);
  const float floatCM = fixedSizeToFloat<8>(fixedSizeCM);

  // bit 8 of upper word is the sign 1 = positive
  return (iupper & 0x100) ? floatCM : -floatCM;
}

float CommonModeCorrection::getCalPadValue(const std::string calibName, int icru, int pad) const
{
  if (mPadMaps.find(calibName) == mPadMaps.end()) {
    LOGP(error, "{} not set, cannot be used", calibName);
    return 0;
  }
  const auto& calPad = mPadMaps.at(calibName);
  const CRU cru(icru);
  const int roc = cru.roc();
  const int padOffset = (cru.isIROC()) ? Mapper::GLOBALPADOFFSET[cru.region()] : Mapper::GLOBALPADOFFSET[cru.region()] - Mapper::GLOBALPADOFFSET[4];

  const auto& calArray = calPad.getCalArray(roc);

  return calArray.getValue(padOffset + pad);
}

bool CommonModeCorrection::padMapExists(const std::string& calibName)
{
  if (mPadMaps.find(calibName) == mPadMaps.end()) {
    LOGP(error, "{} not in mPadMaps", calibName);
    return false;
  }
  return true;
}

void CommonModeCorrection::loadCalPad(std::string_view fileName, std::string_view nameInFile, std::string_view namePadMap)
{
  if (fileName.size() == 0) {
    return;
  }

  auto pads = o2::tpc::utils::readCalPads(fileName, nameInFile);
  if ((pads.size() == 0) || (pads.at(0) == nullptr)) {
    LOGP(error, "Could not load object {} from file {}", nameInFile, fileName);
    return;
  }

  if (namePadMap.size() == 0) {
    namePadMap = nameInFile;
  }

  mPadMaps[namePadMap.data()] = *pads[0];
}

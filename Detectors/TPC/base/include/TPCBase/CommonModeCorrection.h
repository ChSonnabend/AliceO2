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

/// \file CommonModeCorrection.h
/// \brief  Calculate the common mode correction factor
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_CommonModeCorrection_H_
#define AliceO2_TPC_CommonModeCorrection_H_

#include <gsl/span>
#include <string_view>
#include <vector>

#include "DataFormatsTPC/Digit.h"
#include "TPCBase/FEEConfig.h"

namespace o2::tpc
{

/// Class to calculate the common mode correction
///
/// Calculation of the common mode correction, based on the algorithm propsed by Marian Ivanov
/// The calculation is done for one single CRU and time bin
class CommonModeCorrection
{
 public:
  struct CMdata {
    std::vector<float> adcValues;
    std::vector<float> cmKValues;
    std::vector<float> pedestals;

    void resize(size_t newSize)
    {
      adcValues.resize(newSize);
      cmKValues.resize(newSize);
      pedestals.resize(newSize);
    }

    void clear()
    {
      adcValues.clear();
      cmKValues.clear();
      pedestals.clear();
    }
  };

  struct CMInfo {
    float cmValue{};        ///< common mode value from pseudo code
    float cmValueStd{};     ///< std dev of common mode values from pseudo code
    float cmValueCRU{};     ///< common mode value from firmware, if available
    float sumPos{};         ///< sum of positive signals > mSumPosThreshold
    float sumNeg{};         ///< sum of negative signals <= mSumPosThreshold, corrected for k-factor
    uint16_t nPadsUsed{};   ///< number of pads used for CM calculation
    uint16_t nNeg{};        ///< number of pads used for sumNeg
    uint16_t nOccupancy{};  ///< number of CM corrected pads larger than mOccupancyThreshold
    uint16_t nSaturation{}; ///< number of pads in saturation
  };

  struct CMDebug {
    std::vector<uint8_t> nPadsOk{};
    std::vector<uint16_t> adcDist{};
  };

  using CalPadMapType = std::unordered_map<std::string, CalPad>;

  /// Calculation of the common mode value
  ///
  /// \param value pad-by-pad charge values
  /// \param cmKValues corresponding pad-by-pad common mode k-factors
  /// \param pedestals corresponding pad-by-pad pedestals
  /// \param
  CMInfo getCommonMode(gsl::span<const float> values, gsl::span<const float> cmKValues, gsl::span<const float> pedestals, CMDebug* cmDebug = nullptr) const;
  CMInfo getCommonMode(const std::vector<float>& values, const std::vector<float>& cmKValues, const std::vector<float>& pedestals) const { return getCommonMode(gsl::span(values), gsl::span(cmKValues), gsl::span(pedestals)); }

  CMInfo getCommonMode(const CMdata& cmData) const { return getCommonMode(std::span(cmData.adcValues), std::span(cmData.cmKValues), std::span(cmData.pedestals)); }

  void setNPadsCompRandom(int n) { mNPadsCompRamdom = n; }
  int getNPadsCompRandom() const { return mNPadsCompRamdom; }

  void setNPadsCompMin(int n) { mNPadsCompMin = n; }
  int getNPadsCompMin() const { return mNPadsCompMin; }

  /// Minimum number of pads required in the CM calculation to be used for digit correction
  void setNPadsMinCM(int n) { mNPadsMinCM = n; }
  int getNPadsMinCM() const { return mNPadsMinCM; }

  void setQEmpty(float q) { mQEmpty = q; }
  float getQEmpty() const { return mQEmpty; }

  void setQComp(float q) { mQComp = q; }
  float getQComp() const { return mQComp; }

  /// The mQComp will be set to (cm - mQCompScaleThreshold) * mQCompScale, if cm > mQCompScaleThreshold
  void setQCompScaleThreshold(float q) { mQCompScaleThreshold = q; }
  float getQCompScaleThreshold() const { return mQCompScaleThreshold; }

  /// The mQComp will be set to (cm - mQCompScaleThreshold) * mQCompScale, if cm > mQCompScaleThreshold
  void setQCompScale(float q) { mQCompScale = q; }
  float getQCompScale() const { return mQCompScale; }

  /// Threshold above which a signal is considered for sumPos, if debug information is used
  void setSumPosThreshold(float threshold) { mSumPosThreshold = threshold; }
  float getSumPosThreshold() const { return mSumPosThreshold; }

  /// Threshold above which a signal is considered for the occupancy
  void setOccupancyThreshold(float threshold) { mOccupancyThreshold = threshold; }
  float getOccupancyThreshold() const { return mOccupancyThreshold; }

  /// Pad maps loaded from FEEConfig
  void setPadMaps(CalPadMapType& padMaps) { mPadMaps = padMaps; }

  /// load a CalPad from file and add it to the local mPadMaps
  /// \param fileName input file name
  /// \param nameInFile name of the CalPad object in the file
  /// \param namePadMap name under which to store the object in the mPadMaps, if empty use the same as nameInFile
  void loadCalPad(std::string_view fileName, std::string_view nameInFile, std::string_view namePadMap = "");

  /// load CMkValues from file, assuming it is stored under the name "CMkValues
  void loadCMkValues(std::string_view fileName) { loadCalPad(fileName, "CMkValues"); }

  /// load Pedestals from file, assuming it is stored under the name "Pedestals
  void loadPedestals(std::string_view fileName) { loadCalPad(fileName, "Pedestals"); }

  /// Custom setting of CalPad, overwriting what was set in mPadMaps
  void setCalPad(const CalPad& calPad, std::string_view name) { mPadMaps[name.data()] = calPad; }

  /// cmk value
  float getCMkValue(int sector, int row, int pad) { return mPadMaps["CMkValues"].getValue(sector, row, pad); }

  /// pedestal value
  float getPedestalValue(int sector, int row, int pad) { return mPadMaps["Pedestals"].getValue(sector, row, pad); }

  /// load the Pad maps from CCDB
  void
    loadDefaultPadMaps(FEEConfig::Tags feeTag = FEEConfig::Tags::Physics30sigma);

  CMdata collectCMdata(const std::vector<Digit>& digits, int cru, int timeBin);

  int getCommonMode(std::vector<Digit>& digits, std::vector<std::vector<CMInfo>>& cmValues, bool negativeOnly = false, bool hasInjectedCMValue = false, std::vector<std::vector<CMDebug>>* cmDebug = nullptr, int minTimeBin = -1, int maxTimeBin = -1) const;

  /// corret digits for common mode
  /// \param cmValues will contain CM information for each CRU and time bin
  /// \param negativeOnly only correct negative common mode signals
  /// \return maximum
  int correctDigits(std::vector<Digit>& digits, std::vector<std::vector<CMInfo>>& cmValues, bool negativeOnly = false, bool hasInjectedCMValue = false, std::vector<std::vector<CMDebug>>* cmDebug = nullptr, int minTimeBin = -1, int maxTimeBin = -1) const;

  void correctDigits(std::string_view digiFileIn, Long64_t maxEntries = -1, std::string_view digitFileOut = "tpcdigit_cmcorr.root", std::string_view cmFileOut = "CommonModeValues.root", bool negativeOnly = false, int nThreads = 1, bool writeOnlyCM = false, bool writeDebug = false, bool hasInjectedCMValue = false, int minTimeBin = -1, int maxTimeBin = -1);

  void limitKFactorPrecision(bool limit = true) { mLimitKFactor = limit; }
  void limitPedestalPrecision(bool limit = true) { mLimitPedestal = limit; }

  /// set the number of threads used for CM calculation
  /// \param nThreads number of threads
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// \return returns the number of threads used for decoding
  static int getNThreads() { return sNThreads; }

  /// add artificial common mode, only works when using the 'correctDigits' function
  void addCommonMode(float cm) { mArtificialCM = cm; }

  void setCorrectOutputForPedestal(bool corret = true) { mCorrectOutputForPedestal = corret; }
  bool getCorrectOutputForPedestal() const { return mCorrectOutputForPedestal; }

  /// Add zeros for pads without signal
  void setAddSubthreshold(bool addSubthreshold) { mSubthreshold = addSubthreshold; }
  bool getAddSubthreshold() const { return mSubthreshold; }

  static float decodeInjectedCMValue(float lower, float upper);

 private:
  inline static int sNThreads{1};        ///< Number of parallel threads for the CM calculation
  int mNPadsCompRamdom{10};              ///< Number of random pads to compare with to check if the present pad is empty
  int mNPadsCompMin{7};                  ///< Minimum number of neighbouring pads with q close to present pad to define this as empty
  int mNPadsMinCM{0};                    ///< Minimum number of pads required in the CM calculation to be used for digit correction
  float mQEmpty{2};                      ///< Threshold to enter check for empty pad
  float mQComp{1};                       ///< Threshold for comparison with random pads
  float mQCompScaleThreshold{0};         ///< Charge threshold from which on to increase mQComp
  float mQCompScale{0};                  ///< Slope with which to increase mQComp if below mQCompScaleThreshold
  float mSumPosThreshold{2};             ///< calculate sumPos > mSumPosThreshold, sumNeg M<= mSumPosThreshold
  float mOccupancyThreshold{3};          ///< calculate number of pads > mQCompScaleThreshold after CM correction
  bool mLimitKFactor{false};             ///< Limit the k-factor precision to 2I6F
  bool mLimitPedestal{false};            ///< Limit the preestal precision to 10I2F
  int mSubthreshold{0};                  ///< Add data for pads without signal. 1 = add zeros; 2 = add random noise
  float mArtificialCM{};                 ///< artificial common mode signals
  bool mCorrectOutputForPedestal{false}; ///< correct the writte out ADC for the pedestal value

  CalPadMapType mPadMaps; ///< Pad-by-pad CRU configuration values (Pedestal, Noise, ITF + CM parameters)

  struct pos {
    int row;
    int pad;
  };

  // positions of lower words per CRU in sector
  const std::array<pos, 10> mCMInjectIDLower{
    // row0 pad0 row1 pad1
    pos{0, 2},
    pos{20, 1},
    pos{32, 2},
    pos{51, 1},
    pos{63, 1},
    pos{84, 1},
    pos{97, 1},
    pos{116, 2},
    pos{127, 2},
    pos{142, 0},
  };

  // positions of upper words per CRU in sector
  const std::array<pos, 10> mCMInjectIDUpper{
    // row0 pad0 row1 pad1
    pos{0, 3},
    pos{20, 3},
    pos{32, 3},
    pos{51, 3},
    pos{63, 2},
    pos{84, 4},
    pos{97, 2},
    pos{115, 5},
    pos{127, 3},
    pos{142, 4},
  };

  /// Return the value stored in mPadMaps["calibName"]
  /// \param calibName name of calibraion in mPadMaps
  /// \param cru CRU number
  /// \param pad Pad number within the CRU
  float getCalPadValue(const std::string calibName, int icru, int pad) const;

  bool padMapExists(const std::string& calibName);

  ClassDefNV(CommonModeCorrection, 2);
};

} // namespace o2::tpc
#endif

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

#ifndef ALICEO2_TRK_GEOMETRYTGEO_H
#define ALICEO2_TRK_GEOMETRYTGEO_H

#include <memory>
#include <DetectorsCommonDataFormats/DetMatrixCache.h>

namespace o2
{
namespace trk
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  using Mat3D = o2::math_utils::Transform3D;
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2GRot;
  using DetMatrixCache::getMatrixT2L;
  // this method is not advised for ITS: for barrel detectors whose tracking frame is just a rotation
  // it is cheaper to use T2GRot
  using DetMatrixCache::getMatrixT2G;
  GeometryTGeo(bool build = false, int loadTrans = 0);
  ~GeometryTGeo();
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance()
  {
    if (!sInstance) {
      sInstance = std::make_unique<GeometryTGeo>(true, 0);
    }
    return sInstance.get();
  };
  static const char* getTRKVolPattern() { return sVolumeName.c_str(); }
  static const char* getTRKLayerPattern() { return sLayerName.c_str(); }
  static const char* getTRKPetalPattern() { return sPetalName.c_str(); }
  static const char* getTRKPetalDiskPattern() { return sPetalDiskName.c_str(); }
  static const char* getTRKPetalLayerPattern() { return sPetalLayerName.c_str(); }
  static const char* getTRKStavePattern() { return sStaveName.c_str(); }
  static const char* getTRKChipPattern() { return sChipName.c_str(); }
  static const char* getTRKSensorPattern() { return sSensorName.c_str(); }
  static const char* getTRKWrapVolPattern() { return sWrapperVolumeName.c_str(); }

  int getNumberOfChips() const { return mSize; }

  /// Determines the number of active parts in the Geometry
  int extractNumberOfLayersMLOT();
  int extractNumberOfLayersVD() const;
  int extractNumberOfPetalsVD() const;
  int extractNumberOfActivePartsVD() const;
  int extractNumberOfDisksVD() const;
  int extractNumberOfChipsPerPetalVD() const;
  int extractNumberOfStavesMLOT(int lay) const;

  /// Extract number following the prefix in the name string
  int extractVolumeCopy(const char* name, const char* prefix) const;

  int getNumberOfLayersMLOT() const { return mNumberOfLayersMLOT; }
  int getNumberOffActivePartsVD() const { return mNumberOfActivePartsVD; }

  bool isOwner() const { return mOwner; }
  void setOwner(bool v) { mOwner = v; }

  void Print(Option_t* opt = "") const;
  void PrintChipID(int index, int subDetID, int petalcase, int disk, int lay, int stave, int indexRetrieved) const;

  int getLayer(int index) const;
  int getStave(int index) const;
  int getSubDetID(int index) const;
  int getPetalCase(int index) const;
  int getDisk(int index) const;

  /// This routine computes the chip index number from the subDetID, petal, disk, layer, stave /// TODO: retrieve also from chip when chips will be available
  /// in substave
  /// \param int subDetID The subdetector ID, 0 for VD, 1 for MLOT
  /// \param int petalcase The petal case number for VD, from 0 to 3
  /// \param int disk The disk number for VD, from 0 to 5
  /// \param int lay The layer number. Starting from 0 both for VD and MLOT
  /// \param int stave The stave number for MLOT. Starting from 0
  int getChipIndex(int subDetID, int petalcase, int disk, int lay, int stave) const;

  /// This routine computes subDetID, petal, disk, layer, stave given the chip index number /// TODO: copute also from chip when chips will be available
  /// \param int index The chip index number, starting from 0
  /// \param int subDetID The subdetector ID, 0 for VD, 1 for MLOT
  /// \param int petalcase The petal case number for VD, from 0 to 3
  /// \param int disk The disk number for VD, from 0 to 5
  /// \param int lay The layer number. Starting from 0 both for VD and MLOT
  /// \param int stave The stave number for MLOT. Starting from 0
  bool getChipID(int index, int& subDetID, int& petalcase, int& disk, int& lay, int& stave) const;

  int getLastChipIndex(int lay) const { return mLastChipIndex[lay]; }
  int getFirstChipIndex(int lay, int petalcase, int subDetID) const
  {
    /// Get the first chip index of the active petal (VD) or layer (MLOT)
    if (subDetID == 0) { // VD
      return (petalcase == 0) ? 0 : mLastChipIndexVD[petalcase - 1] + 1;
    } else if (subDetID == 1) { // MLOT
      return mLastChipIndex[lay + mNumberOfPetalsVD - 1] + 1;
    }
    return -1; // not found
  }

  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(int index) const;

  TString getMatrixPath(int index) const;

  static const char* composeSymNameTRK(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::TRK).getName(), d);
  }
  static const char* composeSymNameLayer(int d, int layer);
  static const char* composeSymNameStave(int d, int layer);
  static const char* composeSymNameChip(int d, int lr);
  static const char* composeSymNameSensor(int d, int layer);

 protected:
  static constexpr int MAXLAYERS = 20; ///< max number of active layers

  static std::string sVolumeName;
  static std::string sLayerName;
  static std::string sPetalName;
  static std::string sPetalDiskName;
  static std::string sPetalLayerName;
  static std::string sStaveName;
  static std::string sChipName;
  static std::string sSensorName;
  static std::string sWrapperVolumeName; ///< Wrapper volume name

  Int_t mNumberOfLayersMLOT;                   ///< number of layers
  Int_t mNumberOfActivePartsVD;                ///< number of layers
  Int_t mNumberOfLayersVD;                     ///< number of layers
  Int_t mNumberOfPetalsVD;                     ///< number of Petals = chip in each VD layer
  Int_t mNumberOfDisksVD;                      ///< number of Disks = 6
  std::vector<int> mLastChipIndex;             ///< max ID of the detctor in the petal(VD) or layer(MLOT)
  std::vector<int> mLastChipIndexVD;           ///< max ID of the detctor in the layer for the VD
  std::vector<int> mLastChipIndexMLOT;         ///< max ID of the detctor in the layer for the MLOT
  std::vector<int> mNumberOfChipsPerLayerVD;   ///< number of chips per layer VD ( =  number of petals)
  std::vector<int> mNumberOfChipsPerLayerMLOT; ///< number of chips per layer MLOT ( = 1 for the moment)
  std::vector<int> mNumbersOfChipPerDiskVD;    ///< numbersOfChipPerDiskVD
  std::vector<int> mNumberOfChipsPerPetalVD;   ///< numbersOfChipPerPetalVD
  std::vector<int> mNumberOfStaves;            ///< Number Of Staves per layer in ML/OT
  std::array<char, MAXLAYERS> mLayerToWrapper; ///< Layer to wrapper correspondence

  bool mOwner = true; //! is it owned by the singleton?

 private:
  static std::unique_ptr<o2::trk::GeometryTGeo> sInstance;
};

} // namespace trk
} // namespace o2
#endif
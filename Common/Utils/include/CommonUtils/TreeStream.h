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

/// @brief Class for creating debug root trees with std::cout like intervace
/// @author Marian Ivanov, marian.ivanov@cern.ch (original code in AliRoot)
///         Ruben Shahoyan, ruben.shahoyan@cern.ch (porting to O2).

#ifndef ALICEO2_TREESTREAM_H
#define ALICEO2_TREESTREAM_H

#include <TString.h>
#include <TTree.h>
#include <vector>
#include <type_traits>
#include <concepts>
#include "GPUCommonDef.h"

class TBranch;
class TClass;
class TDataType;

namespace o2
{
namespace utils
{
/// The TreeStream class allows creating a root tree of any objects having root
/// dictionary, using operator<< interface, and w/o prior tree declaration.
/// The format is:
/// treeStream << "branchName0="<<objPtr
///            <<"branchName1="<<objRed
///            <<"branchName2="
///            <<elementaryTypeVar<<"\n"
///
/// See testTreeStream.cxx for functional example
///
namespace details
{
template <typename T>
struct IsTrivialRootType {
  static constexpr bool value =
    std::is_same_v<T, Float_t> ||                                                        // Float_t
    std::is_same_v<T, Double_t> ||                                                       // Double_t
    std::is_same_v<T, ULong64_t> || std::is_same_v<T, ULong_t> ||                        // ULong64_t or ULong_t
    std::is_same_v<T, Long64_t> || std::is_same_v<T, Long_t> ||                          // Long64_t or Long_t
    std::is_same_v<T, UInt_t> ||                                                         // UInt_t
    std::is_same_v<T, Int_t> ||                                                          // Int_t
    std::is_same_v<T, UShort_t> ||                                                       // UShort_t
    std::is_same_v<T, Short_t> ||                                                        // Short_t
    std::is_same_v<T, UChar_t> ||                                                        // UChar_t
    std::is_same_v<T, Char_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, Bool_t>; // Char_t, int8_t, or Bool_t
};

template <typename T>
struct IsTrivialRootType<T[]> {
  static constexpr bool value = IsTrivialRootType<T>::value;
};

template <typename T, std::size_t N>
struct IsTrivialRootType<T[N]> {
  static constexpr bool value = IsTrivialRootType<T>::value;
};

template <typename T>
concept TrivialRootType = IsTrivialRootType<T>::value;

template <typename T>
concept ComplexRootType = !IsTrivialRootType<T>::value;

template <TrivialRootType T>
static constexpr char getRootTypeCode()
{
  if constexpr (std::is_array_v<T>) {
    return getRootTypeCode<std::remove_all_extents_t<T>>();
  } else if constexpr (std::is_same_v<T, Float_t>) {
    return 'F';
  } else if constexpr (std::is_same_v<T, Double_t>) {
    return 'D';
  } else if constexpr (std::is_same_v<T, ULong64_t> ||
                       std::is_same_v<T, ULong_t>) {
    return 'l';
  } else if constexpr (std::is_same_v<T, Long64_t> ||
                       std::is_same_v<T, Long_t>) {
    return 'L';
  } else if constexpr (std::is_same_v<T, UInt_t>) {
    return 'i';
  } else if constexpr (std::is_same_v<T, Int_t>) {
    return 'I';
  } else if constexpr (std::is_same_v<T, UShort_t>) {
    return 's';
  } else if constexpr (std::is_same_v<T, Short_t>) {
    return 'S';
  } else if constexpr (std::is_same_v<T, UChar_t>) {
    return 'b';
  } else if constexpr (std::is_same_v<T, Char_t> ||
                       std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, Bool_t>) {
    return 'B';
  } else {
    static_assert(false, "unsupported type!");
  }
}
} // namespace details

class TreeStream
{
 public:
  struct TreeDataElement {
    int arsize = 1;              ///< size of array
    char type = 0;               ///< type of data element
    const TClass* cls = nullptr; ///< data type pointer
    const void* ptr = nullptr;   ///< pointer to element
    std::string name;            ///< name of the element
  };

  TreeStream(const char* treename);
  TreeStream() = default;
  virtual ~TreeStream() = default;
  void Close() { mTree.Write(); }
  Int_t CheckIn(Char_t type, const void* pointer);
  void BuildTree();
  void Fill();
  Double_t getSize() { return mTree.GetZipBytes(); }
  TreeStream& Endl();

  TTree& getTree() { return mTree; }
  const char* getName() const { return mTree.GetName(); }
  void setID(int id) { mID = id; }
  int getID() const { return mID; }

  template <details::TrivialRootType T>
  TreeStream& operator<<(const T& t)
  {
    CheckIn(details::getRootTypeCode<T>(), &t);
    return *this;
  }

  TreeStream& operator<<(const Char_t* name);

  template <class T>
  TreeStream& operator<<(const T* obj)
  {
    CheckIn(obj);
    return *this;
  }

  template <details::ComplexRootType T, typename std::enable_if<!std::is_pointer<GPUgeneric() T>::value, bool>::type* = nullptr>
  TreeStream& operator<<(const T& obj)
  {
    CheckIn(&obj);
    return *this;
  }

  template <class T>
  Int_t CheckIn(const T* obj);

 private:
  //
  std::vector<TreeDataElement> mElements;
  std::vector<TBranch*> mBranches; ///< pointers to branches
  TTree mTree;                     ///< data storage
  int mCurrentIndex = 0;           ///< index of current element
  int mID = -1;                    ///< identifier of layout
  int mNextNameCounter = 0;        ///< next name counter
  int mNextArraySize = 0;          ///< next array size
  int mStatus = 0;                 ///< status of the layout
  TString mNextName;               ///< name for next entry

  ClassDefNV(TreeStream, 0);
};

template <class T>
Int_t TreeStream::CheckIn(const T* obj)
{
  // check in arbitrary class having dictionary
  TClass* pClass = nullptr;
  if (obj) {
    pClass = TClass::GetClass(typeid(*obj));
  }

  if (mCurrentIndex >= static_cast<int>(mElements.size())) {
    auto& element = mElements.emplace_back();
    element.cls = pClass;
    TString name = mNextName;
    if (name.Length()) {
      if (mNextNameCounter > 0) {
        name += mNextNameCounter;
      }
    } else {
      name = TString::Format("B%d", static_cast<int>(mElements.size()));
    }
    element.name = name.Data();
    element.ptr = obj;
    element.arsize = mNextArraySize;
    mNextArraySize = 1; // reset
  } else {
    auto& element = mElements[mCurrentIndex];
    if (!element.cls) {
      element.cls = pClass;
    } else {
      if (element.cls != pClass && pClass) {
        mStatus++;
        return 1; // mismatched data element
      }
    }
    element.ptr = obj;
  }
  mCurrentIndex++;
  return 0;
}

} // namespace utils
} // namespace o2

#endif

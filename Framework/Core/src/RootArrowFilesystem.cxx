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
#include "Framework/RootArrowFilesystem.h"
#include "Framework/Endian.h"
#include "Framework/RuntimeError.h"
#include <Rtypes.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <memory>
#include <stdexcept>
#include <TFile.h>
#include <TLeaf.h>
#include <TBufferFile.h>
#include <TTree.h>
#include <TDirectoryFile.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>

namespace
{
struct BranchInfo {
  std::string name;
  TBranch* ptr;
  bool mVLA;
};
} // namespace

auto arrowTypeFromROOT(EDataType type, int size)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType> const& type, int size) -> std::shared_ptr<arrow::DataType> {
    switch (size) {
      case -1:
        return arrow::list(type);
      case 1:
        return std::move(type);
      default:
        return arrow::fixed_size_list(type, size);
    }
  };

  switch (type) {
    case EDataType::kBool_t:
      return typeGenerator(arrow::boolean(), size);
    case EDataType::kUChar_t:
      return typeGenerator(arrow::uint8(), size);
    case EDataType::kUShort_t:
      return typeGenerator(arrow::uint16(), size);
    case EDataType::kUInt_t:
      return typeGenerator(arrow::uint32(), size);
    case EDataType::kULong64_t:
      return typeGenerator(arrow::uint64(), size);
    case EDataType::kChar_t:
      return typeGenerator(arrow::int8(), size);
    case EDataType::kShort_t:
      return typeGenerator(arrow::int16(), size);
    case EDataType::kInt_t:
      return typeGenerator(arrow::int32(), size);
    case EDataType::kLong64_t:
      return typeGenerator(arrow::int64(), size);
    case EDataType::kFloat_t:
      return typeGenerator(arrow::float32(), size);
    case EDataType::kDouble_t:
      return typeGenerator(arrow::float64(), size);
    default:
      throw o2::framework::runtime_error_f("Unsupported branch type: %d", static_cast<int>(type));
  }
}
namespace o2::framework
{

TFileFileSystem::TFileFileSystem(TDirectoryFile* f, size_t readahead)
  : VirtualRootFileSystemBase(),
    mFile(f)
{
  ((TFile*)mFile)->SetReadaheadSize(50 * 1024 * 1024);
}

std::shared_ptr<VirtualRootFileSystemBase> TFileFileSystem::GetSubFilesystem(arrow::dataset::FileSource source)
{
  auto tree = (TTree*)mFile->GetObjectChecked(source.path().c_str(), TClass::GetClass<TTree>());
  if (tree) {
    return std::shared_ptr<VirtualRootFileSystemBase>(new SingleTreeFileSystem(tree));
  }


  auto directory = (TDirectoryFile*)mFile->GetObjectChecked(source.path().c_str(), TClass::GetClass<TDirectory>());
  if (directory) {
    return std::shared_ptr<VirtualRootFileSystemBase>(new TFileFileSystem(directory, 50 * 1024 * 1024));
  }
  throw runtime_error_f("Unsupported file layout");
}

arrow::Result<arrow::fs::FileInfo> TFileFileSystem::GetFileInfo(const std::string& path)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  result.set_path(path);
  arrow::dataset::FileSource source(path, shared_from_this());

  auto fs = GetSubFilesystem(source);

  // For now we only support single trees.
  if (std::dynamic_pointer_cast<SingleTreeFileSystem>(fs)) {
    result.set_type(arrow::fs::FileType::File);
    return result;
  }
  return result;
}

arrow::Result<arrow::fs::FileInfo> VirtualRootFileSystemBase::GetFileInfo(std::string const&)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  return result;
}

arrow::Result<arrow::fs::FileInfoVector> VirtualRootFileSystemBase::GetFileInfo(const arrow::fs::FileSelector& select)
{
  arrow::fs::FileInfoVector results;
  auto selected = this->GetFileInfo(select.base_dir);
  if (selected.ok()) {
    results.emplace_back(*selected);
  }
  return results;
}

arrow::Status VirtualRootFileSystemBase::CreateDir(const std::string& path, bool recursive)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteDir(const std::string& path)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::CopyFile(const std::string& src, const std::string& dest)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::Move(const std::string& src, const std::string& dest)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteDirContents(const std::string& path, bool missing_dir_ok)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteRootDirContents()
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteFile(const std::string& path)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> VirtualRootFileSystemBase::OpenInputStream(const std::string& path)
{
  return arrow::Status::NotImplemented("Non streamable filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> VirtualRootFileSystemBase::OpenInputFile(const std::string& path)
{
  return arrow::Status::NotImplemented("No random access file system");
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> VirtualRootFileSystemBase::OpenOutputStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  return arrow::Status::NotImplemented("Non streamable filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> VirtualRootFileSystemBase::OpenAppendStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  return arrow::Status::NotImplemented("No random access file system");
}

arrow::Result<std::shared_ptr<arrow::Schema>> TTreeFileFormat::Inspect(const arrow::dataset::FileSource& source) const
{
  arrow::Schema schema{{}};
  auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
  // Actually get the TTree from the ROOT file.
  auto treeFs = std::dynamic_pointer_cast<TTreeFileSystem>(fs->GetSubFilesystem(source));
  if (!treeFs.get()) {
    throw runtime_error_f("Unknown filesystem %s\n", source.filesystem()->type_name().c_str());
  }
  TTree* tree = treeFs->GetTree(source);

  auto branches = tree->GetListOfBranches();
  auto n = branches->GetEntries();

  std::vector<BranchInfo> branchInfos;
  for (auto i = 0; i < n; ++i) {
    auto branch = static_cast<TBranch*>(branches->At(i));
    auto name = std::string{branch->GetName()};
    auto pos = name.find("_size");
    if (pos != std::string::npos) {
      name.erase(pos);
      branchInfos.emplace_back(BranchInfo{name, (TBranch*)nullptr, true});
    } else {
      auto lookup = std::find_if(branchInfos.begin(), branchInfos.end(), [&](BranchInfo const& bi) {
        return bi.name == name;
      });
      if (lookup == branchInfos.end()) {
        branchInfos.emplace_back(BranchInfo{name, branch, false});
      } else {
        lookup->ptr = branch;
      }
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  tree->SetCacheSize(25000000);
  for (auto& bi : branchInfos) {
    static TClass* cls;
    EDataType type;
    bi.ptr->GetExpectedType(cls, type);
    auto listSize = -1;
    if (!bi.mVLA) {
      listSize = static_cast<TLeaf*>(bi.ptr->GetListOfLeaves()->At(0))->GetLenStatic();
    }
    auto field = std::make_shared<arrow::Field>(bi.ptr->GetName(), arrowTypeFromROOT(type, listSize));
    fields.push_back(field);

    tree->AddBranchToCache(bi.ptr);
    if (strncmp(bi.ptr->GetName(), "fIndexArray", strlen("fIndexArray")) == 0) {
      std::string sizeBranchName = bi.ptr->GetName();
      sizeBranchName += "_size";
      auto* sizeBranch = (TBranch*)tree->GetBranch(sizeBranchName.c_str());
      if (sizeBranch) {
        tree->AddBranchToCache(sizeBranch);
      }
    }
  }
  tree->StopCacheLearningPhase();

  return std::make_shared<arrow::Schema>(fields);
}

/// \brief Create a FileFragment for a FileSource.
arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> TTreeFileFormat::MakeFragment(
  arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
  std::shared_ptr<arrow::Schema> physical_schema)
{
  std::shared_ptr<arrow::dataset::FileFormat> format = std::make_shared<TTreeFileFormat>(mTotCompressedSize, mTotUncompressedSize);

  auto fragment = std::make_shared<TTreeFileFragment>(std::move(source), std::move(format),
                                                      std::move(partition_expression),
                                                      std::move(physical_schema));
  return std::dynamic_pointer_cast<arrow::dataset::FileFragment>(fragment);
}

arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> TTreeFileFormat::MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination, std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options, arrow::fs::FileLocator destination_locator) const
{
  throw std::runtime_error("Unsupported operation");
}

std::shared_ptr<arrow::dataset::FileWriteOptions> TTreeFileFormat::DefaultWriteOptions()
{
  std::shared_ptr<TTreeFileWriteOptions> options(
    new TTreeFileWriteOptions(shared_from_this()));
  return options;
}

arrow::Result<arrow::RecordBatchGenerator> TTreeFileFormat::ScanBatchesAsync(
  const std::shared_ptr<arrow::dataset::ScanOptions>& options,
  const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const
{
  // Get the fragment as a TTreeFragment. This might be PART of a TTree.
  auto treeFragment = std::dynamic_pointer_cast<TTreeFileFragment>(fragment);
  // This is the schema we want to read
  auto dataset_schema = options->dataset_schema;

  auto generator = [pool = options->pool, treeFragment, dataset_schema, &totalCompressedSize = mTotCompressedSize,
                    &totalUncompressedSize = mTotUncompressedSize]() -> arrow::Future<std::shared_ptr<arrow::RecordBatch>> {
    auto schema = treeFragment->format()->Inspect(treeFragment->source());

    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields = dataset_schema->fields();
    auto physical_schema = *treeFragment->ReadPhysicalSchema();

    static TBufferFile buffer{TBuffer::EMode::kWrite, 4 * 1024 * 1024};
    auto containerFS = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(treeFragment->source().filesystem());
    auto fs = std::dynamic_pointer_cast<TTreeFileSystem>(containerFS->GetSubFilesystem(treeFragment->source()));

    int64_t rows = -1;
    TTree* tree = fs->GetTree(treeFragment->source());
    for (auto& field : fields) {
      // The field actually on disk
      auto physicalField = physical_schema->GetFieldByName(field->name());
      TBranch* branch = tree->GetBranch(physicalField->name().c_str());
      assert(branch);
      buffer.Reset();
      auto totalEntries = branch->GetEntries();
      if (rows == -1) {
        rows = totalEntries;
      }
      if (rows != totalEntries) {
        throw runtime_error_f("Unmatching number of rows for branch %s", branch->GetName());
      }
      arrow::Status status;
      int readEntries = 0;
      std::shared_ptr<arrow::Array> array;
      auto listType = std::dynamic_pointer_cast<arrow::FixedSizeListType>(physicalField->type());
      if (physicalField->type() == arrow::boolean() ||
          (listType && physicalField->type()->field(0)->type() == arrow::boolean())) {
        if (listType) {
          std::unique_ptr<arrow::ArrayBuilder> builder = nullptr;
          auto status = arrow::MakeBuilder(pool, physicalField->type()->field(0)->type(), &builder);
          if (!status.ok()) {
            throw runtime_error("Cannot create value builder");
          }
          auto listBuilder = std::make_unique<arrow::FixedSizeListBuilder>(pool, std::move(builder), listType->list_size());
          auto valueBuilder = listBuilder.get()->value_builder();
          // boolean array special case: we need to use builder to create the bitmap
          status = valueBuilder->Reserve(totalEntries * listType->list_size());
          status &= listBuilder->Reserve(totalEntries);
          if (!status.ok()) {
            throw runtime_error("Failed to reserve memory for array builder");
          }
          while (readEntries < totalEntries) {
            auto readLast = branch->GetBulkRead().GetBulkEntries(readEntries, buffer);
            readEntries += readLast;
            status &= static_cast<arrow::BooleanBuilder*>(valueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer.GetCurrent()), readLast * listType->list_size());
          }
          status &= static_cast<arrow::FixedSizeListBuilder*>(listBuilder.get())->AppendValues(readEntries);
          if (!status.ok()) {
            throw runtime_error("Failed to append values to array");
          }
          status &= listBuilder->Finish(&array);
          if (!status.ok()) {
            throw runtime_error("Failed to create array");
          }
        } else if (listType == nullptr) {
          std::unique_ptr<arrow::ArrayBuilder> builder = nullptr;
          auto status = arrow::MakeBuilder(pool, physicalField->type(), &builder);
          if (!status.ok()) {
            throw runtime_error("Cannot create builder");
          }
          auto valueBuilder = static_cast<arrow::BooleanBuilder*>(builder.get());
          // boolean array special case: we need to use builder to create the bitmap
          status = valueBuilder->Reserve(totalEntries);
          if (!status.ok()) {
            throw runtime_error("Failed to reserve memory for array builder");
          }
          while (readEntries < totalEntries) {
            auto readLast = branch->GetBulkRead().GetBulkEntries(readEntries, buffer);
            readEntries += readLast;
            status &= valueBuilder->AppendValues(reinterpret_cast<uint8_t const*>(buffer.GetCurrent()), readLast);
          }
          if (!status.ok()) {
            throw runtime_error("Failed to append values to array");
          }
          status &= valueBuilder->Finish(&array);
          if (!status.ok()) {
            throw runtime_error("Failed to create array");
          }
        }
      } else {
        // other types: use serialized read to build arrays directly.
        auto typeSize = physicalField->type()->byte_width();
        // This is needed for branches which have not been persisted.
        auto bytes = branch->GetTotBytes();
        auto branchSize = bytes ? bytes : 1000000;
        auto&& result = arrow::AllocateResizableBuffer(branchSize, pool);
        if (!result.ok()) {
          throw runtime_error("Cannot allocate values buffer");
        }
        std::shared_ptr<arrow::Buffer> arrowValuesBuffer = std::move(result).ValueUnsafe();
        auto ptr = arrowValuesBuffer->mutable_data();
        if (ptr == nullptr) {
          throw runtime_error("Invalid buffer");
        }

        std::unique_ptr<TBufferFile> offsetBuffer = nullptr;

        uint32_t offset = 0;
        int count = 0;
        std::shared_ptr<arrow::Buffer> arrowOffsetBuffer;
        std::span<int> offsets;
        int size = 0;
        uint32_t totalSize = 0;
        TBranch* mSizeBranch = nullptr;
        int64_t listSize = 1;
        if (auto fixedSizeList = std::dynamic_pointer_cast<arrow::FixedSizeListType>(physicalField->type())) {
          listSize = fixedSizeList->list_size();
        } else if (auto vlaListType = std::dynamic_pointer_cast<arrow::ListType>(physicalField->type())) {
          listSize = -1;
        }
        if (listSize == -1) {
          mSizeBranch = branch->GetTree()->GetBranch((std::string{branch->GetName()} + "_size").c_str());
          offsetBuffer = std::make_unique<TBufferFile>(TBuffer::EMode::kWrite, 4 * 1024 * 1024);
          result = arrow::AllocateResizableBuffer((totalEntries + 1) * (int64_t)sizeof(int), pool);
          if (!result.ok()) {
            throw runtime_error("Cannot allocate offset buffer");
          }
          arrowOffsetBuffer = std::move(result).ValueUnsafe();
          unsigned char* ptrOffset = arrowOffsetBuffer->mutable_data();
          auto* tPtrOffset = reinterpret_cast<int*>(ptrOffset);
          offsets = std::span<int>{tPtrOffset, tPtrOffset + totalEntries + 1};

          // read sizes first
          while (readEntries < totalEntries) {
            auto readLast = mSizeBranch->GetBulkRead().GetEntriesSerialized(readEntries, *offsetBuffer);
            readEntries += readLast;
            for (auto i = 0; i < readLast; ++i) {
              offsets[count++] = (int)offset;
              offset += swap32_(reinterpret_cast<uint32_t*>(offsetBuffer->GetCurrent())[i]);
            }
          }
          offsets[count] = (int)offset;
          totalSize = offset;
          readEntries = 0;
        }

        while (readEntries < totalEntries) {
          auto readLast = branch->GetBulkRead().GetEntriesSerialized(readEntries, buffer);
          if (listSize == -1) {
            size = offsets[readEntries + readLast] - offsets[readEntries];
          } else {
            size = readLast * listSize;
          }
          readEntries += readLast;
          swapCopy(ptr, buffer.GetCurrent(), size, typeSize);
          ptr += (ptrdiff_t)(size * typeSize);
        }
        if (listSize >= 1) {
          totalSize = readEntries * listSize;
        }
        std::shared_ptr<arrow::PrimitiveArray> varray;
        switch (listSize) {
          case -1:
            varray = std::make_shared<arrow::PrimitiveArray>(physicalField->type()->field(0)->type(), totalSize, arrowValuesBuffer);
            array = std::make_shared<arrow::ListArray>(physicalField->type(), readEntries, arrowOffsetBuffer, varray);
            break;
          case 1:
            array = std::make_shared<arrow::PrimitiveArray>(physicalField->type(), readEntries, arrowValuesBuffer);
            break;
          default:
            varray = std::make_shared<arrow::PrimitiveArray>(physicalField->type()->field(0)->type(), totalSize, arrowValuesBuffer);
            array = std::make_shared<arrow::FixedSizeListArray>(physicalField->type(), readEntries, varray);
        }
      }

      branch->SetStatus(false);
      branch->DropBaskets("all");
      branch->Reset();
      branch->GetTransientBuffer(0)->Expand(0);

      columns.push_back(array);
    }
    auto batch = arrow::RecordBatch::Make(dataset_schema, rows, columns);
    totalCompressedSize += tree->GetZipBytes();
    totalUncompressedSize += tree->GetTotBytes();
    return batch;
  };
  return generator;
}

TBufferFileFS::TBufferFileFS(TBufferFile* f)
  : VirtualRootFileSystemBase(),
    mBuffer(f),
    mFilesystem(nullptr)
{
}

TTreeFileSystem::~TTreeFileSystem() = default;


arrow::Result<arrow::fs::FileInfo> TBufferFileFS::GetFileInfo(const std::string& path)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  result.set_path(path);
  arrow::dataset::FileSource source(path, shared_from_this());

  // Only once to avoid rereading the streamed tree.
  if (!mFilesystem.get()) {
    return result;
  }

  // For now we only support single trees.
  if (std::dynamic_pointer_cast<SingleTreeFileSystem>(mFilesystem)) {
    result.set_type(arrow::fs::FileType::File);
    return result;
  }
  return result;
}

std::shared_ptr<VirtualRootFileSystemBase> TBufferFileFS::GetSubFilesystem(arrow::dataset::FileSource source)
{
  if (!mFilesystem.get()) {
    auto tree = ((TTree*)mBuffer->ReadObject(TTree::Class()));
    mFilesystem = std::make_shared<SingleTreeFileSystem>(tree);
  }
  return mFilesystem;
}

} // namespace o2::framework

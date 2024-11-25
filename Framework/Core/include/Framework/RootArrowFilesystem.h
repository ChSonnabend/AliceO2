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
#ifndef O2_FRAMEWORK_ROOT_ARROW_FILESYSTEM_H_
#define O2_FRAMEWORK_ROOT_ARROW_FILESYSTEM_H_

#include <arrow/dataset/type_fwd.h>
#include <arrow/dataset/file_base.h>
#include <arrow/filesystem/type_fwd.h>
#include <arrow/type_fwd.h>
#include <memory>

class TTree;
class TBufferFile;
class TDirectoryFile;

namespace o2::framework
{

class TTreeFileWriteOptions : public arrow::dataset::FileWriteOptions
{
 public:
  TTreeFileWriteOptions(std::shared_ptr<arrow::dataset::FileFormat> format)
    : FileWriteOptions(format)
  {
  }
};

// This is to avoid having to implement a bunch of unimplemented methods
// for all the possible virtual filesystem we can invent on top of ROOT
// data structures.
class VirtualRootFileSystemBase : public arrow::fs::FileSystem
{
 public:
  // Dummy implementation to avoid
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& select) override;

  bool Equals(const FileSystem& other) const override
  {
    return this->type_name() == other.type_name();
  }

  virtual std::shared_ptr<VirtualRootFileSystemBase> GetSubFilesystem(arrow::dataset::FileSource source) = 0;

  arrow::Status CreateDir(const std::string& path, bool recursive) override;

  arrow::Status DeleteDir(const std::string& path) override;

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override;

  arrow::Status Move(const std::string& src, const std::string& dest) override;

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override;

  arrow::Status DeleteRootDirContents() override;

  arrow::Status DeleteFile(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;
};

// A filesystem which allows me to get a TTree
class TTreeFileSystem : public VirtualRootFileSystemBase
{
 public:
  ~TTreeFileSystem() override;

  std::shared_ptr<VirtualRootFileSystemBase> GetSubFilesystem(arrow::dataset::FileSource source) override
  {
    return std::dynamic_pointer_cast<VirtualRootFileSystemBase>(shared_from_this());
  };

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  virtual TTree* GetTree(arrow::dataset::FileSource source) = 0;
};

class SingleTreeFileSystem : public TTreeFileSystem
{
 public:
  SingleTreeFileSystem(TTree* tree)
    : TTreeFileSystem(),
      mTree(tree)
  {
  }

  std::string type_name() const override
  {
    return "ttree";
  }

  TTree* GetTree(arrow::dataset::FileSource) override
  {
    // Simply return the only TTree we have
    return mTree;
  }

 private:
  TTree* mTree;
};

class TFileFileSystem : public VirtualRootFileSystemBase
{
 public:
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;

  TFileFileSystem(TDirectoryFile* f, size_t readahead);

  std::string type_name() const override
  {
    return "TDirectoryFile";
  }

  std::shared_ptr<VirtualRootFileSystemBase> GetSubFilesystem(arrow::dataset::FileSource source) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  // We can go back to the TFile in case this is needed.
  TDirectoryFile* GetFile()
  {
    return mFile;
  }

 private:
  TDirectoryFile* mFile;
};

class TBufferFileFS : public VirtualRootFileSystemBase
{
 public:
  TBufferFileFS(TBufferFile* f);

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  std::string type_name() const override
  {
    return "tbufferfile";
  }

  std::shared_ptr<VirtualRootFileSystemBase> GetSubFilesystem(arrow::dataset::FileSource source) override;

 private:
  TBufferFile* mBuffer;
  std::shared_ptr<VirtualRootFileSystemBase> mFilesystem;
};

class TTreeFileFragment : public arrow::dataset::FileFragment
{
 public:
  TTreeFileFragment(arrow::dataset::FileSource source,
                    std::shared_ptr<arrow::dataset::FileFormat> format,
                    arrow::compute::Expression partition_expression,
                    std::shared_ptr<arrow::Schema> physical_schema)
    : FileFragment(std::move(source), std::move(format), std::move(partition_expression), std::move(physical_schema))
  {
  }
};

class TTreeFileFormat : public arrow::dataset::FileFormat
{
  size_t& mTotCompressedSize;
  size_t& mTotUncompressedSize;

 public:
  TTreeFileFormat(size_t& totalCompressedSize, size_t& totalUncompressedSize)
    : FileFormat({}),
      mTotCompressedSize(totalCompressedSize),
      mTotUncompressedSize(totalUncompressedSize)
  {
  }

  ~TTreeFileFormat() override = default;

  std::string type_name() const override
  {
    return "ttree";
  }

  bool Equals(const FileFormat& other) const override
  {
    return other.type_name() == this->type_name();
  }

  arrow::Result<bool> IsSupported(const arrow::dataset::FileSource& source) const override
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    auto subFs = fs->GetSubFilesystem(source);
    if (std::dynamic_pointer_cast<TTreeFileSystem>(subFs)) {
      return true;
    }
    return false;
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> Inspect(const arrow::dataset::FileSource& source) const override;
  /// \brief Create a FileFragment for a FileSource.
  arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> MakeFragment(
    arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
    std::shared_ptr<arrow::Schema> physical_schema) override;

  arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination, std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options, arrow::fs::FileLocator destination_locator) const override;

  std::shared_ptr<arrow::dataset::FileWriteOptions> DefaultWriteOptions() override;

  arrow::Result<arrow::RecordBatchGenerator> ScanBatchesAsync(
    const std::shared_ptr<arrow::dataset::ScanOptions>& options,
    const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const override;
};

// An arrow outputstream which allows to write to a ttree
class TTreeOutputStream : public arrow::io::OutputStream
{
 public:
  TTreeOutputStream(TTree* t);

  arrow::Status Close() override;

  arrow::Result<int64_t> Tell() const override;

  arrow::Status Write(const void* data, int64_t nbytes) override;

  bool closed() const override;

  TTree* GetTree()
  {
    return mTree;
  }

 private:
  TTree* mTree;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_ROOT_ARROW_FILESYSTEM_H_

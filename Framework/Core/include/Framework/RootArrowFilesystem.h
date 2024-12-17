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

#include <TBufferFile.h>
#include <arrow/dataset/type_fwd.h>
#include <arrow/dataset/file_base.h>
#include <arrow/filesystem/type_fwd.h>
#include <arrow/type_fwd.h>
#include <memory>

class TFile;
class TBufferFile;
class TDirectoryFile;

namespace o2::framework
{

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

struct RootArrowFactory final {
  std::function<std::shared_ptr<arrow::dataset::FileWriteOptions>()> options = nullptr;
  std::function<std::shared_ptr<arrow::dataset::FileFormat>()> format = nullptr;
  std::function<std::shared_ptr<VirtualRootFileSystemBase>(void*)> getSubFilesystem = nullptr;
};

struct RootArrowFactoryPlugin {
  virtual RootArrowFactory* create() = 0;
};

// A registry for all the possible ways of encoding a table in a TFile
struct RootObjectReadingCapability {
  // The unique name of this capability
  std::string name = "unknown";
  // Given a TFile, return the object which this capability support
  // Use a void * in order not to expose the kind of object to the
  // generic reading code. This is also where we load the plugin
  // which will be used for the actual creation.
  std::function<void*(TDirectoryFile* file, std::string const& path)> getHandle;
  // Same as the above, but uses a TBufferFile as storage
  std::function<void*(TBufferFile*, std::string const&)> getBufferHandle;
  // This must be implemented to load the actual RootArrowFactory plugin which
  // implements this capability. This way the detection of the file format
  // (via get handle) does not need to know about the actual code which performs
  // the serialization (and might depend on e.g. RNTuple).
  std::function<RootArrowFactory&()> factory;
};

struct RootObjectReadingCapabilityPlugin {
  virtual RootObjectReadingCapability* create() = 0;
};

// This acts as registry of all the capabilities (i.e. the ability to
// associate a given object in a root file to the serialization plugin) and
// the factory (i.e. the serialization plugin)
struct RootObjectReadingFactory {
  std::vector<RootObjectReadingCapability> capabilities;
};

class TFileFileSystem : public VirtualRootFileSystemBase
{
 public:
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;

  TFileFileSystem(TDirectoryFile* f, size_t readahead, RootObjectReadingFactory&);

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
  RootObjectReadingFactory& mObjectFactory;
};

class TBufferFileFS : public VirtualRootFileSystemBase
{
 public:
  TBufferFileFS(TBufferFile* f, RootObjectReadingFactory&);

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  std::string type_name() const override
  {
    return "tbufferfile";
  }

  std::shared_ptr<VirtualRootFileSystemBase> GetSubFilesystem(arrow::dataset::FileSource source) override;

 private:
  TBufferFile* mBuffer;
  std::shared_ptr<VirtualRootFileSystemBase> mFilesystem;
  RootObjectReadingFactory& mObjectFactory;
};

// An arrow outputstream which allows to write to a TDirectoryFile.
// This will point to the location of the file itself. You can
// specify the location of the actual object inside it by passing the
// associated path to the Write() API.
class TDirectoryFileOutputStream : public arrow::io::OutputStream
{
 public:
  TDirectoryFileOutputStream(TDirectoryFile*);

  arrow::Status Close() override;

  arrow::Result<int64_t> Tell() const override;

  arrow::Status Write(const void* data, int64_t nbytes) override;

  bool closed() const override;

  TDirectoryFile* GetDirectory()
  {
    return mDirectory;
  }

 private:
  TDirectoryFile* mDirectory;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_ROOT_ARROW_FILESYSTEM_H_

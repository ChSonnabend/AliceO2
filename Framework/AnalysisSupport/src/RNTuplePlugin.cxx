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

#include "Framework/RuntimeError.h"
#include "Framework/RootArrowFilesystem.h"
#include "Framework/Plugins.h"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RVec.hxx>
#include <memory>
#include <TBufferFile.h>

#include <TDirectory.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/dataset/file_base.h>

template class
  std::unique_ptr<ROOT::Experimental::RNTupleReader>;

namespace o2::framework
{

class RNTupleFileWriteOptions : public arrow::dataset::FileWriteOptions
{
 public:
  RNTupleFileWriteOptions(std::shared_ptr<arrow::dataset::FileFormat> format)
    : FileWriteOptions(format)
  {
  }
};

// A filesystem which allows me to get a RNTuple
class RNTupleFileSystem : public VirtualRootFileSystemBase
{
 public:
  ~RNTupleFileSystem() override;

  virtual ROOT::Experimental::RNTuple* GetRNTuple(arrow::dataset::FileSource source) = 0;
};

class SingleRNTupleFileSystem : public RNTupleFileSystem
{
 public:
  SingleRNTupleFileSystem(ROOT::Experimental::RNTuple* tuple)
    : RNTupleFileSystem(),
      mTuple(tuple)
  {
  }

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(std::string const& path) override;

  std::string type_name() const override
  {
    return "rntuple";
  }

  ROOT::Experimental::RNTuple* GetRNTuple(arrow::dataset::FileSource) override
  {
    // Simply return the only TTree we have
    return mTuple;
  }

 private:
  ROOT::Experimental::RNTuple* mTuple;
};

arrow::Result<arrow::fs::FileInfo> SingleRNTupleFileSystem::GetFileInfo(std::string const& path)
{
  arrow::dataset::FileSource source(path, shared_from_this());
  arrow::fs::FileInfo result;
  result.set_path(path);
  result.set_type(arrow::fs::FileType::File);
  return result;
}

class RNTupleFileFragment : public arrow::dataset::FileFragment
{
 public:
  RNTupleFileFragment(arrow::dataset::FileSource source,
                      std::shared_ptr<arrow::dataset::FileFormat> format,
                      arrow::compute::Expression partition_expression,
                      std::shared_ptr<arrow::Schema> physical_schema)
    : FileFragment(source, format, partition_expression, physical_schema)
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    if (!fs.get()) {
      throw runtime_error_f("Do not know how to extract %s from %s", source.path().c_str(), fs->type_name().c_str());
    }
    auto handler = fs->GetObjectHandler(source);
    if (!handler->format->Equals(*format)) {
      throw runtime_error_f("Format for %s does not match. Found %s, expected %s.", source.path().c_str(),
                            handler->format->type_name().c_str(),
                            format->type_name().c_str());
    }
    mNTuple = handler->GetObjectAsOwner<ROOT::Experimental::RNTuple>();
  }

  ROOT::Experimental::RNTuple* GetRNTuple()
  {
    return mNTuple.get();
  }

 private:
  std::unique_ptr<ROOT::Experimental::RNTuple> mNTuple;
};

class RNTupleFileFormat : public arrow::dataset::FileFormat
{
  size_t& mTotCompressedSize;
  size_t& mTotUncompressedSize;

 public:
  RNTupleFileFormat(size_t& totalCompressedSize, size_t& totalUncompressedSize)
    : FileFormat({}),
      mTotCompressedSize(totalCompressedSize),
      mTotUncompressedSize(totalUncompressedSize)
  {
  }

  ~RNTupleFileFormat() override = default;

  std::string type_name() const override
  {
    return "rntuple";
  }

  bool Equals(const FileFormat& other) const override
  {
    return other.type_name() == this->type_name();
  }

  arrow::Result<bool> IsSupported(const arrow::dataset::FileSource& source) const override
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    if (!fs) {
      return false;
    }
    return fs->CheckSupport(source);
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> Inspect(const arrow::dataset::FileSource& source) const override;

  arrow::Result<arrow::RecordBatchGenerator> ScanBatchesAsync(
    const std::shared_ptr<arrow::dataset::ScanOptions>& options,
    const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const override;

  std::shared_ptr<arrow::dataset::FileWriteOptions> DefaultWriteOptions() override;

  arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination,
                                                                        std::shared_ptr<arrow::Schema> schema,
                                                                        std::shared_ptr<arrow::dataset::FileWriteOptions> options,
                                                                        arrow::fs::FileLocator destination_locator) const override;
  arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> MakeFragment(
    arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
    std::shared_ptr<arrow::Schema> physical_schema) override;
};

struct RootNTupleVisitor : public ROOT::Experimental::Detail::RFieldVisitor {
  void VisitArrayField(const ROOT::Experimental::RArrayField& field) override
  {
    int size = field.GetLength();
    RootNTupleVisitor valueVisitor{};
    auto valueField = field.GetSubFields()[0];
    valueField->AcceptVisitor(valueVisitor);
    auto type = valueVisitor.datatype;
    this->datatype = arrow::fixed_size_list(type, size);
  }

  void VisitRVecField(const ROOT::Experimental::RRVecField& field) override
  {
    RootNTupleVisitor valueVisitor{};
    auto valueField = field.GetSubFields()[0];
    valueField->AcceptVisitor(valueVisitor);
    auto type = valueVisitor.datatype;
    this->datatype = arrow::list(type);
  }

  void VisitField(const ROOT::Experimental::RFieldBase& field) override
  {
    throw o2::framework::runtime_error_f("Unknown field %s with type %s", field.GetFieldName().c_str(), field.GetTypeName().c_str());
  }

  void VisitIntField(const ROOT::Experimental::RField<int>& field) override
  {
    this->datatype = arrow::int32();
  }

  void VisitInt8Field(const ROOT::Experimental::RField<std::int8_t>& field) override
  {
    this->datatype = arrow::int8();
  }

  void VisitInt16Field(const ROOT::Experimental::RField<std::int16_t>& field) override
  {
    this->datatype = arrow::int16();
  }

  void VisitUInt32Field(const ROOT::Experimental::RField<std::uint32_t>& field) override
  {
    this->datatype = arrow::uint32();
  }

  void VisitUInt8Field(const ROOT::Experimental::RField<std::uint8_t>& field) override
  {
    this->datatype = arrow::uint8();
  }

  void VisitUInt16Field(const ROOT::Experimental::RField<std::uint16_t>& field) override
  {
    this->datatype = arrow::int16();
  }

  void VisitBoolField(const ROOT::Experimental::RField<bool>& field) override
  {
    this->datatype = arrow::boolean();
  }

  void VisitFloatField(const ROOT::Experimental::RField<float>& field) override
  {
    this->datatype = arrow::float32();
  }

  void VisitDoubleField(const ROOT::Experimental::RField<double>& field) override
  {
    this->datatype = arrow::float64();
  }
  std::shared_ptr<arrow::DataType> datatype;
};
} // namespace o2::framework

auto arrowTypeFromRNTuple(ROOT::Experimental::RFieldBase const& field, int size)
{
  o2::framework::RootNTupleVisitor visitor;
  field.AcceptVisitor(visitor);
  return visitor.datatype;
}

namespace o2::framework
{
std::unique_ptr<ROOT::Experimental::RFieldBase> rootFieldFromArrow(std::shared_ptr<arrow::Field> field, std::string name)
{
  using namespace ROOT::Experimental;
  switch (field->type()->id()) {
    case arrow::Type::BOOL:
      return std::make_unique<RField<bool>>(name);
    case arrow::Type::UINT8:
      return std::make_unique<RField<uint8_t>>(name);
    case arrow::Type::UINT16:
      return std::make_unique<RField<uint16_t>>(name);
    case arrow::Type::UINT32:
      return std::make_unique<RField<uint32_t>>(name);
    case arrow::Type::UINT64:
      return std::make_unique<RField<uint64_t>>(name);
    case arrow::Type::INT8:
      return std::make_unique<RField<int8_t>>(name);
    case arrow::Type::INT16:
      return std::make_unique<RField<int16_t>>(name);
    case arrow::Type::INT32:
      return std::make_unique<RField<int32_t>>(name);
    case arrow::Type::INT64:
      return std::make_unique<RField<int64_t>>(name);
    case arrow::Type::FLOAT:
      return std::make_unique<RField<float>>(name);
    case arrow::Type::DOUBLE:
      return std::make_unique<RField<double>>(name);
    case arrow::Type::STRING:
      return std::make_unique<RField<std::string>>(name);
    default:
      throw runtime_error("Unsupported arrow column type");
  }
}

class RNTupleFileWriter : public arrow::dataset::FileWriter
{
  std::shared_ptr<ROOT::Experimental::RNTupleWriter> mWriter;
  bool firstBatch = true;
  std::vector<std::shared_ptr<arrow::Array>> valueArrays;
  std::vector<std::shared_ptr<arrow::DataType>> valueTypes;
  std::vector<size_t> valueCount;

 public:
  RNTupleFileWriter(std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options,
                    std::shared_ptr<arrow::io::OutputStream> destination,
                    arrow::fs::FileLocator destination_locator)
    : FileWriter(schema, options, destination, destination_locator)
  {
    using namespace ROOT::Experimental;

    auto model = RNTupleModel::CreateBare();
    // Let's create a model from the physical schema
    for (auto i = 0u; i < schema->fields().size(); ++i) {
      auto& field = schema->field(i);

      // Construct all the needed branches.
      switch (field->type()->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
          auto list = std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
          auto valueField = field->type()->field(0);
          model->AddField(std::make_unique<RArrayField>(field->name(), rootFieldFromArrow(valueField, "_0"), list->list_size()));
        } break;
        case arrow::Type::LIST: {
          auto valueField = field->type()->field(0);
          model->AddField(std::make_unique<RRVecField>(field->name(), rootFieldFromArrow(valueField, "_0")));
        } break;
        default: {
          model->AddField(rootFieldFromArrow(field, field->name()));
        } break;
      }
    }
    auto fileStream = std::dynamic_pointer_cast<TDirectoryFileOutputStream>(destination_);
    auto* file = dynamic_cast<TFile*>(fileStream->GetDirectory());
    mWriter = RNTupleWriter::Append(std::move(model), destination_locator_.path, *file, {});
  }

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch>& batch) override
  {
    if (firstBatch) {
      firstBatch = false;
    }

    // Support writing empty tables
    if (batch->columns().empty() || batch->num_rows() == 0) {
      return arrow::Status::OK();
    }

    for (auto i = 0u; i < batch->columns().size(); ++i) {
      auto column = batch->column(i);
      auto& field = batch->schema()->field(i);

      valueArrays.push_back(nullptr);
      valueTypes.push_back(nullptr);
      valueCount.push_back(1);

      switch (field->type()->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
          auto list = std::static_pointer_cast<arrow::FixedSizeListArray>(column);
          auto listType = std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
          if (field->type()->field(0)->type()->id() == arrow::Type::BOOL) {
            auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(list->values());
            int64_t length = boolArray->length();
            arrow::UInt8Builder builder;
            auto ok = builder.Reserve(length);

            for (int64_t i = 0; i < length; ++i) {
              if (boolArray->IsValid(i)) {
                // Expand each boolean value (true/false) to uint8 (1/0)
                uint8_t value = boolArray->Value(i) ? 1 : 0;
                auto ok = builder.Append(value);
              } else {
                // Append null for invalid entries
                auto ok = builder.AppendNull();
              }
            }
            valueArrays.back() = *builder.Finish();
            valueTypes.back() = valueArrays.back()->type();
          } else {
            valueArrays.back() = list->values();
            valueTypes.back() = field->type()->field(0)->type();
          }
          valueCount.back() = listType->list_size();
        } break;
        case arrow::Type::LIST: {
          auto list = std::static_pointer_cast<arrow::ListArray>(column);
          valueArrays.back() = list;
          valueTypes.back() = field->type()->field(0)->type();
          valueCount.back() = -1;
        } break;
        case arrow::Type::BOOL: {
          // We unpack the array
          auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(column);
          int64_t length = boolArray->length();
          arrow::UInt8Builder builder;
          auto ok = builder.Reserve(length);

          for (int64_t i = 0; i < length; ++i) {
            if (boolArray->IsValid(i)) {
              // Expand each boolean value (true/false) to uint8 (1/0)
              uint8_t value = boolArray->Value(i) ? 1 : 0;
              auto ok = builder.Append(value);
            } else {
              // Append null for invalid entries
              auto ok = builder.AppendNull();
            }
          }
          valueArrays.back() = *builder.Finish();
          valueTypes.back() = valueArrays.back()->type();
        } break;
        default:
          valueArrays.back() = column;
          valueTypes.back() = field->type();
          break;
      }
    }

    int64_t pos = 0;

    auto entry = mWriter->CreateEntry();
    std::vector<ROOT::Experimental::REntry::RFieldToken> tokens;
    tokens.reserve(batch->num_columns());
    std::vector<size_t> typeIds;
    typeIds.reserve(batch->num_columns());

    for (size_t ci = 0; ci < batch->num_columns(); ++ci) {
      auto& field = batch->schema()->field(ci);
      typeIds.push_back(batch->column(ci)->type()->id());
      tokens.push_back(entry->GetToken(field->name()));
    }

    while (pos < batch->num_rows()) {
      for (size_t ci = 0; ci < batch->num_columns(); ++ci) {
        auto typeId = typeIds[ci];
        auto token = tokens[ci];

        switch (typeId) {
          case arrow::Type::LIST: {
            auto list = std::static_pointer_cast<arrow::ListArray>(valueArrays[ci]);
            auto value_slice = list->value_slice(pos);

            valueCount[ci] = value_slice->length();
            auto bindValue = [&vc = valueCount, ci, token](auto array, std::unique_ptr<ROOT::Experimental::REntry>& entry) -> void {
              using value_type = std::decay_t<decltype(*array.get())>::value_type;
              auto v = std::make_shared<ROOT::VecOps::RVec<value_type>>((value_type*)array->raw_values(), vc[ci]);
              entry->BindValue(token, v);
            };
            switch (valueTypes[ci]->id()) {
              case arrow::Type::FLOAT: {
                bindValue(std::static_pointer_cast<arrow::FloatArray>(value_slice), entry);
              } break;
              case arrow::Type::DOUBLE: {
                bindValue(std::static_pointer_cast<arrow::DoubleArray>(value_slice), entry);
              } break;
              case arrow::Type::INT8: {
                bindValue(std::static_pointer_cast<arrow::Int8Array>(value_slice), entry);
              } break;
              case arrow::Type::INT16: {
                bindValue(std::static_pointer_cast<arrow::Int16Array>(value_slice), entry);
              } break;
              case arrow::Type::INT32: {
                bindValue(std::static_pointer_cast<arrow::Int32Array>(value_slice), entry);
              } break;
              case arrow::Type::INT64: {
                bindValue(std::static_pointer_cast<arrow::Int64Array>(value_slice), entry);
              } break;
              case arrow::Type::UINT8: {
                bindValue(std::static_pointer_cast<arrow::UInt8Array>(value_slice), entry);
              } break;
              case arrow::Type::UINT16: {
                bindValue(std::static_pointer_cast<arrow::UInt16Array>(value_slice), entry);
              } break;
              case arrow::Type::UINT32: {
                bindValue(std::static_pointer_cast<arrow::UInt32Array>(value_slice), entry);
              } break;
              case arrow::Type::UINT64: {
                bindValue(std::static_pointer_cast<arrow::UInt64Array>(value_slice), entry);
              } break;
              default: {
                throw runtime_error("Unsupported kind of VLA");
              } break;
            }
          } break;
          case arrow::Type::FIXED_SIZE_LIST: {
            entry->BindRawPtr<void>(token, (void*)(valueArrays[ci]->data()->buffers[1]->data() + pos * valueCount[ci] * valueTypes[ci]->byte_width()));
          } break;
          case arrow::Type::BOOL: {
            // Not sure we actually need this
            entry->BindRawPtr<bool>(token, (bool*)(valueArrays[ci]->data()->buffers[1]->data() + pos * 1));
          } break;
          default:
            // By default we consider things scalars.
            entry->BindRawPtr<void>(token, (void*)(valueArrays[ci]->data()->buffers[1]->data() + pos * valueTypes[ci]->byte_width()));
            break;
        }
      }
      mWriter->Fill(*entry);
      ++pos;
    }
    // mWriter->CommitCluster();

    return arrow::Status::OK();
  }

  arrow::Future<>
    FinishInternal() override
  {
    return {};
  };
};

arrow::Result<std::shared_ptr<arrow::Schema>> RNTupleFileFormat::Inspect(const arrow::dataset::FileSource& source) const
{

  auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
  // Actually get the TTree from the ROOT file.
  auto objectHandler = fs->GetObjectHandler(source);
  if (objectHandler->format->type_name() != this->type_name()) {
    throw runtime_error_f("Unexpected kind of filesystem %s to handle payload %s.\n", source.filesystem()->type_name().c_str(), source.path().c_str());
  }
  // We know this is a RNTuple, so we can continue with the inspection.
  auto rntuple = objectHandler->GetObjectAsOwner<ROOT::Experimental::RNTuple>().release();

  auto inspector = ROOT::Experimental::RNTupleInspector::Create(rntuple);

  auto reader = ROOT::Experimental::RNTupleReader::Open(rntuple);

  auto& tupleField0 = reader->GetModel().GetFieldZero();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& tupleField : tupleField0.GetSubFields()) {
    auto field = std::make_shared<arrow::Field>(tupleField->GetFieldName(), arrowTypeFromRNTuple(*tupleField, tupleField->GetValueSize()));
    fields.push_back(field);
  }

  return std::make_shared<arrow::Schema>(fields);
}

arrow::Result<arrow::RecordBatchGenerator> RNTupleFileFormat::ScanBatchesAsync(
  const std::shared_ptr<arrow::dataset::ScanOptions>& options,
  const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const
{
  auto dataset_schema = options->dataset_schema;
  auto ntupleFragment = std::dynamic_pointer_cast<RNTupleFileFragment>(fragment);

  auto generator = [pool = options->pool, ntupleFragment, dataset_schema, &totalCompressedSize = mTotCompressedSize,
                    &totalUncompressedSize = mTotUncompressedSize]() -> arrow::Future<std::shared_ptr<arrow::RecordBatch>> {
    using namespace ROOT::Experimental;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields = dataset_schema->fields();

    int64_t rows = -1;
    ROOT::Experimental::RNTuple* rntuple = ntupleFragment->GetRNTuple();
    auto reader = ROOT::Experimental::RNTupleReader::Open(rntuple);
    auto& model = reader->GetModel();
    for (auto& physicalField : fields) {
      auto bulk = model.CreateBulk(physicalField->name());

      auto listType = std::dynamic_pointer_cast<arrow::FixedSizeListType>(physicalField->type());

      auto& descriptor = reader->GetDescriptor();
      auto totalEntries = reader->GetNEntries();

      if (rows == -1) {
        rows = totalEntries;
      }
      if (rows != totalEntries) {
        throw runtime_error_f("Unmatching number of rows for branch %s", physicalField->name().c_str());
      }
      arrow::Status status;
      int readEntries = 0;
      std::shared_ptr<arrow::Array> array;
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
          auto clusterIt = descriptor.FindClusterId(0, 0);
          // No adoption for now...
          // bulk.AdoptBuffer(buffer, totalEntries)
          while (clusterIt != kInvalidDescriptorId) {
            auto& index = descriptor.GetClusterDescriptor(clusterIt);
            auto mask = std::make_unique<bool[]>(index.GetNEntries());
            std::fill(mask.get(), mask.get() + index.GetNEntries(), true);
            void* ptr = bulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries());
            int readLast = index.GetNEntries();
            readEntries += readLast;
            status &= static_cast<arrow::BooleanBuilder*>(valueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(ptr), readLast * listType->list_size());
            clusterIt = descriptor.FindNextClusterId(clusterIt);
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
          auto clusterIt = descriptor.FindClusterId(0, 0);
          while (clusterIt != kInvalidDescriptorId) {
            auto& index = descriptor.GetClusterDescriptor(clusterIt);
            auto mask = std::make_unique<bool[]>(index.GetNEntries());
            std::fill(mask.get(), mask.get() + index.GetNEntries(), true);
            void* ptr = bulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries());
            int readLast = index.GetNEntries();
            readEntries += readLast;
            status &= valueBuilder->AppendValues(reinterpret_cast<uint8_t const*>(ptr), readLast);
            clusterIt = descriptor.FindNextClusterId(clusterIt);
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
        // FIXME: for now...
        auto bytes = 0;
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

        std::shared_ptr<arrow::Buffer> arrowOffsetBuffer;
        std::span<int> offsets;
        int size = 0;
        uint32_t totalSize = 0;
        int64_t listSize = 1;
        if (auto fixedSizeList = std::dynamic_pointer_cast<arrow::FixedSizeListType>(physicalField->type())) {
          listSize = fixedSizeList->list_size();
          typeSize = fixedSizeList->field(0)->type()->byte_width();
          auto clusterIt = descriptor.FindClusterId(0, 0);
          while (clusterIt != kInvalidDescriptorId) {
            auto& index = descriptor.GetClusterDescriptor(clusterIt);
            auto mask = std::make_unique<bool[]>(index.GetNEntries());
            std::fill(mask.get(), mask.get() + index.GetNEntries(), true);
            void* inPtr = bulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries());

            int readLast = index.GetNEntries();
            if (listSize == -1) {
              size = offsets[readEntries + readLast] - offsets[readEntries];
            } else {
              size = readLast * listSize;
            }
            readEntries += readLast;
            memcpy(ptr, inPtr, size * typeSize);
            ptr += (ptrdiff_t)(size * typeSize);
            clusterIt = descriptor.FindNextClusterId(clusterIt);
          }
        } else if (auto vlaListType = std::dynamic_pointer_cast<arrow::ListType>(physicalField->type())) {
          listSize = -1;
          typeSize = vlaListType->field(0)->type()->byte_width();
          offsetBuffer = std::make_unique<TBufferFile>(TBuffer::EMode::kWrite, 4 * 1024 * 1024);
          result = arrow::AllocateResizableBuffer((totalEntries + 1) * (int64_t)sizeof(int), pool);
          if (!result.ok()) {
            throw runtime_error("Cannot allocate offset buffer");
          }
          arrowOffsetBuffer = result.MoveValueUnsafe();

          // Offset bulk
          auto offsetBulk = model.CreateBulk(physicalField->name());
          // Actual values are in a different place...
          bulk = model.CreateBulk(physicalField->name());
          auto clusterIt = descriptor.FindClusterId(0, 0);
          auto* ptrOffset = reinterpret_cast<int*>(arrowOffsetBuffer->mutable_data());
          auto* tPtrOffset = reinterpret_cast<int*>(ptrOffset);
          offsets = std::span<int>{tPtrOffset, tPtrOffset + totalEntries + 1};

          auto copyOffsets = [&arrowValuesBuffer, &pool, &ptrOffset, &ptr, &totalSize](auto inPtr, size_t total) {
            using value_type = typename std::decay_t<decltype(*inPtr)>::value_type;
            for (size_t i = 0; i < total; i++) {
              *ptrOffset++ = totalSize;
              totalSize += inPtr[i].size();
            }
            *ptrOffset = totalSize;
            auto&& result = arrow::AllocateResizableBuffer(totalSize * sizeof(value_type), pool);
            if (!result.ok()) {
              throw runtime_error("Cannot allocate values buffer");
            }
            arrowValuesBuffer = result.MoveValueUnsafe();
            ptr = (uint8_t*)(arrowValuesBuffer->mutable_data());
            // Calculate the size of the buffer here.
            for (size_t i = 0; i < total; i++) {
              int vlaSizeInBytes = inPtr[i].size() * sizeof(value_type);
              if (vlaSizeInBytes == 0) {
                continue;
              }
              memcpy(ptr, inPtr[i].data(), vlaSizeInBytes);
              ptr += vlaSizeInBytes;
            }
          };

          while (clusterIt != kInvalidDescriptorId) {
            auto& index = descriptor.GetClusterDescriptor(clusterIt);
            auto mask = std::make_unique<bool[]>(index.GetNEntries());
            std::fill(mask.get(), mask.get() + index.GetNEntries(), true);
            int readLast = index.GetNEntries();
            switch (vlaListType->field(0)->type()->id()) {
              case arrow::Type::FLOAT: {
                copyOffsets((ROOT::Internal::VecOps::RVec<float>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::DOUBLE: {
                copyOffsets((ROOT::Internal::VecOps::RVec<double>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::INT8: {
                copyOffsets((ROOT::Internal::VecOps::RVec<int8_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::INT16: {
                copyOffsets((ROOT::Internal::VecOps::RVec<int16_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::INT32: {
                copyOffsets((ROOT::Internal::VecOps::RVec<int32_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::INT64: {
                copyOffsets((ROOT::Internal::VecOps::RVec<int64_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::UINT8: {
                copyOffsets((ROOT::Internal::VecOps::RVec<uint8_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::UINT16: {
                copyOffsets((ROOT::Internal::VecOps::RVec<uint16_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::UINT32: {
                copyOffsets((ROOT::Internal::VecOps::RVec<uint32_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              case arrow::Type::UINT64: {
                copyOffsets((ROOT::Internal::VecOps::RVec<uint64_t>*)offsetBulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries()), readLast);
              } break;
              default: {
                throw runtime_error("Unsupported kind of VLA");
              } break;
            }

            readEntries += readLast;
            clusterIt = descriptor.FindNextClusterId(clusterIt);
          }
        } else {
          auto clusterIt = descriptor.FindClusterId(0, 0);
          while (clusterIt != kInvalidDescriptorId) {
            auto& index = descriptor.GetClusterDescriptor(clusterIt);
            auto mask = std::make_unique<bool[]>(index.GetNEntries());
            std::fill(mask.get(), mask.get() + index.GetNEntries(), true);
            void* inPtr = bulk.ReadBulk(RClusterIndex(clusterIt, index.GetFirstEntryIndex()), mask.get(), index.GetNEntries());

            int readLast = index.GetNEntries();
            if (listSize == -1) {
              size = offsets[readEntries + readLast] - offsets[readEntries];
            } else {
              size = readLast * listSize;
            }
            readEntries += readLast;
            memcpy(ptr, inPtr, size * typeSize);
            ptr += (ptrdiff_t)(size * typeSize);
            clusterIt = descriptor.FindNextClusterId(clusterIt);
          }
        }
        switch (listSize) {
          case -1: {
            auto varray = std::make_shared<arrow::PrimitiveArray>(physicalField->type()->field(0)->type(), totalSize, arrowValuesBuffer);
            array = std::make_shared<arrow::ListArray>(physicalField->type(), readEntries, arrowOffsetBuffer, varray);
          } break;
          case 1: {
            totalSize = readEntries * listSize;
            array = std::make_shared<arrow::PrimitiveArray>(physicalField->type(), readEntries, arrowValuesBuffer);

          } break;
          default: {
            totalSize = readEntries * listSize;
            auto varray = std::make_shared<arrow::PrimitiveArray>(physicalField->type()->field(0)->type(), totalSize, arrowValuesBuffer);
            array = std::make_shared<arrow::FixedSizeListArray>(physicalField->type(), readEntries, varray);
          }
        }
      }
      columns.push_back(array);
    }

    auto batch = arrow::RecordBatch::Make(dataset_schema, rows, columns);
    return batch;
  };

  return generator;
}

arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> RNTupleFileFormat::MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination,
                                                                                         std::shared_ptr<arrow::Schema> schema,
                                                                                         std::shared_ptr<arrow::dataset::FileWriteOptions> options,
                                                                                         arrow::fs::FileLocator destination_locator) const
{
  auto writer = std::make_shared<RNTupleFileWriter>(schema, options, destination, destination_locator);
  return std::dynamic_pointer_cast<arrow::dataset::FileWriter>(writer);
}

arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> RNTupleFileFormat::MakeFragment(
  arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
  std::shared_ptr<arrow::Schema> physical_schema)
{
  std::shared_ptr<arrow::dataset::FileFormat> format = std::make_shared<RNTupleFileFormat>(mTotCompressedSize, mTotUncompressedSize);

  auto fragment = std::make_shared<RNTupleFileFragment>(source, format,
                                                        partition_expression,
                                                        physical_schema);
  return std::dynamic_pointer_cast<arrow::dataset::FileFragment>(fragment);
}

RNTupleFileSystem::~RNTupleFileSystem() = default;

std::shared_ptr<arrow::dataset::FileWriteOptions>
  RNTupleFileFormat::DefaultWriteOptions()
{
  return std::make_shared<RNTupleFileWriteOptions>(shared_from_this());
}

struct RNTuplePluginContext {
  size_t totalCompressedSize = 0;
  size_t totalUncompressedSize = 0;
  std::shared_ptr<o2::framework::RNTupleFileFormat> format = nullptr;
};

struct RNTupleObjectReadingImplementation : public RootArrowFactoryPlugin {
  RootArrowFactory* create() override
  {
    auto context = new RNTuplePluginContext;
    context->format = std::make_shared<o2::framework::RNTupleFileFormat>(context->totalCompressedSize, context->totalUncompressedSize);
    return new RootArrowFactory{
      .options = [context]() { return context->format->DefaultWriteOptions(); },
      .format = [context]() { return context->format; },
    };
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(RNTupleObjectReadingImplementation, RootObjectReadingImplementation);
DEFINE_DPL_PLUGINS_END
} // namespace o2::framework

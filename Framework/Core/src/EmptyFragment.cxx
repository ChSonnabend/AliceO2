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
#include "Framework/EmptyFragment.h"
#include <arrow/type_fwd.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/array_nested.h>
#include <memory>

static constexpr int64_t kBufferMinimumSize = 256;

namespace o2::framework
{

// Scanner function which returns a batch where the space is not actually used.
arrow::Result<arrow::RecordBatchGenerator> EmptyFragment::ScanBatchesAsync(
  const std::shared_ptr<arrow::dataset::ScanOptions>& options)
{
  auto generator = [this]() -> arrow::Future<std::shared_ptr<arrow::RecordBatch>> {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(this->physical_schema_->fields().size());

    for (auto& field : this->physical_schema_->fields()) {
      if (auto listType = std::dynamic_pointer_cast<arrow::FixedSizeListType>(field->type())) {
        size_t size = mRows * listType->list_size();
        if (field->type()->field(0)->type()->byte_width() == 0) {
          size /= 8;
        } else {
          size *= field->type()->field(0)->type()->byte_width();
        }
        auto varray = std::make_shared<arrow::PrimitiveArray>(field->type()->field(0)->type(), mRows * listType->list_size(), GetPlaceholderForOp(size));
        columns.push_back(std::make_shared<arrow::FixedSizeListArray>(field->type(), (int32_t)mRows, varray));
      } else {
        size_t size = mRows;
        if (field->type()->byte_width() == 0) {
          size /= 8;
        } else {
          size *= field->type()->byte_width();
        }
        columns.push_back(std::make_shared<arrow::PrimitiveArray>(field->type(), mRows, GetPlaceholderForOp(size)));
      }
    }
    return arrow::RecordBatch::Make(physical_schema_, mRows, columns);
  };
  return generator;
}

PreallocatedOutputStream::PreallocatedOutputStream()
  : is_open_(false), capacity_(0), position_(0), mutable_data_(nullptr) {}

PreallocatedOutputStream::PreallocatedOutputStream(std::vector<size_t>& sizes,
                                                   const std::shared_ptr<arrow::ResizableBuffer>& buffer)
  : sizes_(sizes),
    buffer_(buffer),
    is_open_(true),
    capacity_(buffer->size()),
    position_(0),
    mutable_data_(buffer->mutable_data()) {}

arrow::Result<std::shared_ptr<PreallocatedOutputStream>> PreallocatedOutputStream::Create(
  std::vector<size_t>& ops,
  int64_t initial_capacity, arrow::MemoryPool* pool)
{
  // ctor is private, so cannot use make_shared
  auto ptr = std::shared_ptr<PreallocatedOutputStream>(new PreallocatedOutputStream);
  RETURN_NOT_OK(ptr->Reset(ops, initial_capacity, pool));
  return ptr;
}

arrow::Status PreallocatedOutputStream::Reset(std::vector<size_t> sizes,
                                              int64_t initial_capacity, arrow::MemoryPool* pool)
{
  ARROW_ASSIGN_OR_RAISE(buffer_, AllocateResizableBuffer(initial_capacity, pool));
  sizes_ = sizes;
  is_open_ = true;
  capacity_ = initial_capacity;
  position_ = 0;
  mutable_data_ = buffer_->mutable_data();
  return arrow::Status::OK();
}

arrow::Status PreallocatedOutputStream::Close()
{
  if (is_open_) {
    is_open_ = false;
    if (position_ < capacity_) {
      RETURN_NOT_OK(buffer_->Resize(position_, false));
    }
  }
  return arrow::Status::OK();
}

bool PreallocatedOutputStream::closed() const { return !is_open_; }

arrow::Result<std::shared_ptr<arrow::Buffer>> PreallocatedOutputStream::Finish()
{
  RETURN_NOT_OK(Close());
  buffer_->ZeroPadding();
  is_open_ = false;
  return std::move(buffer_);
}

arrow::Result<int64_t> PreallocatedOutputStream::Tell() const { return position_; }

arrow::Status PreallocatedOutputStream::Write(const void* data, int64_t nbytes)
{
  if (ARROW_PREDICT_FALSE(!is_open_)) {
    return arrow::Status::IOError("OutputStream is closed");
  }
  if (ARROW_PREDICT_TRUE(nbytes == 0)) {
    return arrow::Status::OK();
  }
  if (ARROW_PREDICT_FALSE(position_ + nbytes >= capacity_)) {
    RETURN_NOT_OK(Reserve(nbytes));
  }
  // This is a real address which needs to be copied. Do it!
  auto ref = (int64_t)data;
  if (ref >= sizes_.size()) {
    memcpy(mutable_data_ + position_, data, nbytes);
    position_ += nbytes;
    return arrow::Status::OK();
  }

  position_ += nbytes;
  return arrow::Status::OK();
}

arrow::Status PreallocatedOutputStream::Reserve(int64_t nbytes)
{
  // Always overallocate by doubling.  It seems that it is a better growth
  // strategy, at least for memory_benchmark.cc.
  // This may be because it helps match the allocator's allocation buckets
  // more exactly.  Or perhaps it hits a sweet spot in jemalloc.
  int64_t new_capacity = std::max(kBufferMinimumSize, capacity_);
  new_capacity = position_ + nbytes;
  if (new_capacity > capacity_) {
    RETURN_NOT_OK(buffer_->Resize(new_capacity));
    capacity_ = new_capacity;
    mutable_data_ = buffer_->mutable_data();
  }
  return arrow::Status::OK();
}

} // namespace o2::framework

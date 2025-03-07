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
#ifndef O2_FRAMEWORK_DEFERREDFRAGMENT_H
#define O2_FRAMEWORK_DEFERREDFRAGMENT_H

#include <arrow/dataset/api.h>

namespace o2::framework
{

// A Fragment which will create a preallocated batch in shared memory
// and fill it directly in place.
class EmptyFragment : public arrow::dataset::Fragment
{
 public:
  // @a numRows is the number of rows in the final result.
  // @a physical_schema the schema of the resulting batch
  // @a fillers helper functions to fill the given buffer.
  EmptyFragment(size_t rows,
                arrow::compute::Expression partition_expression,
                std::shared_ptr<arrow::Schema> physical_schema)
    : Fragment(std::move(partition_expression), physical_schema)
  {
  }

  // Scanner function which returns a batch where the space is not actually used.
  arrow::Result<arrow::RecordBatchGenerator> ScanBatchesAsync(
    const std::shared_ptr<arrow::dataset::ScanOptions>& options) override;

 private:
  /// The pointer to each allocation is an incremental number, indexing a collection to track
  /// the size of each allocation.
  std::shared_ptr<arrow::Buffer> GetPlaceholderForOp(size_t size)
  {
    mSizes.push_back(size);
    return std::make_shared<arrow::Buffer>((uint8_t*)(mSizes.size() - 1), size);
  }
  std::vector<size_t> mSizes;
  size_t mRows;
};

/// An OutputStream which does the reading of the input buffers directly
/// on writing, if needed. Each deferred operation is encoded in the source
/// buffer by an incremental number which can be used to lookup in the @a ops
/// vector the operation to perform.
class PreallocatedOutputStream : public arrow::io::OutputStream
{
 public:
  explicit PreallocatedOutputStream(std::vector<size_t>& sizes,
                                    const std::shared_ptr<arrow::ResizableBuffer>& buffer);

  /// \brief Create in-memory output stream with indicated capacity using a
  /// memory pool
  /// \param[in] initial_capacity the initial allocated internal capacity of
  /// the OutputStream
  /// \param[in,out] pool a MemoryPool to use for allocations
  /// \return the created stream
  static arrow::Result<std::shared_ptr<PreallocatedOutputStream>> Create(
    std::vector<size_t>& sizes,
    int64_t initial_capacity = 4096,
    arrow::MemoryPool* pool = arrow::default_memory_pool());

  // By the time we call the destructor, the contents
  // of the buffer are already moved to fairmq
  // for being sent.
  ~PreallocatedOutputStream() override = default;

  // Implement the OutputStream interface

  /// Close the stream, preserving the buffer (retrieve it with Finish()).
  arrow::Status Close() override;
  [[nodiscard]] bool closed() const override;
  [[nodiscard]] arrow::Result<int64_t> Tell() const override;
  arrow::Status Write(const void* data, int64_t nbytes) override;

  /// \cond FALSE
  using OutputStream::Write;
  /// \endcond

  /// Close the stream and return the buffer
  arrow::Result<std::shared_ptr<arrow::Buffer>> Finish();

  /// \brief Initialize state of OutputStream with newly allocated memory and
  /// set position to 0
  /// \param[in] initial_capacity the starting allocated capacity
  /// \param[in,out] pool the memory pool to use for allocations
  /// \return Status
  arrow::Status Reset(std::vector<size_t> sizes,
                      int64_t initial_capacity, arrow::MemoryPool* pool);

  [[nodiscard]] int64_t capacity() const { return capacity_; }

 private:
  std::vector<size_t> sizes_;
  PreallocatedOutputStream();

  // Ensures there is sufficient space available to write nbytes
  arrow::Status Reserve(int64_t nbytes);

  std::shared_ptr<arrow::ResizableBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};
} // namespace o2::framework

#endif

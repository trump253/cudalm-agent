// CUDALM — RAII device memory buffer.
//
// DeviceBuffer owns a single cudaMalloc allocation. It is move-only, has no
// implicit copies, and performs no implicit host<->device transfers (use the
// explicit copy helpers below). Provenance: CUDALM-native.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "cudalm/cuda_check.h"

namespace cudalm {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  // Allocate `bytes` of device memory. stream may be 0 (default stream).
  explicit DeviceBuffer(std::size_t bytes, cudaStream_t stream = 0);

  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  // (Re)allocate `bytes`, freeing any prior allocation.
  void allocate(std::size_t bytes, cudaStream_t stream = 0);
  // Release the allocation. Safe to call when empty.
  void reset();

  void* data() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }
  bool valid() const { return ptr_ != nullptr; }

  template <typename T>
  T* data() const {
    return static_cast<T*>(ptr_);
  }

  // Explicit host -> device copy of `bytes` from `host`.
  void copy_from_host(const void* host, std::size_t bytes, cudaStream_t stream = 0);
  // Explicit device -> host copy of `bytes` into `host`.
  void copy_to_host(void* host, std::size_t bytes, cudaStream_t stream = 0);
  // Fill with zero bytes.
  void clear(std::size_t bytes, cudaStream_t stream = 0);

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace cudalm

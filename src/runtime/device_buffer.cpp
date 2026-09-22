// CUDALM — DeviceBuffer implementation.

#include "cudalm/device_buffer.h"

#include <cstring>

namespace cudalm {

DeviceBuffer::DeviceBuffer(std::size_t bytes, cudaStream_t stream) {
  allocate(bytes, stream);
}

DeviceBuffer::~DeviceBuffer() { reset(); }

void DeviceBuffer::allocate(std::size_t bytes, cudaStream_t stream) {
  if (bytes == 0) {
    // Allow a zero-byte buffer to be "valid" with a nullptr; nothing to do.
    reset();
    return;
  }
  if (ptr_) {
    CUDA_CHECK(cudaFree(ptr_));
    ptr_ = nullptr;
    bytes_ = 0;
  }
  CUDA_CHECK(cudaMalloc(&ptr_, bytes));
  bytes_ = bytes;
  (void)stream;  // allocation is not stream-ordered in v0.1
}

void DeviceBuffer::reset() {
  if (ptr_) {
    CUDA_CHECK(cudaFree(ptr_));
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

void DeviceBuffer::copy_from_host(const void* host, std::size_t bytes,
                                  cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream));
}

void DeviceBuffer::copy_to_host(void* host, std::size_t bytes, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
}

void DeviceBuffer::clear(std::size_t bytes, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes, stream));
}

}  // namespace cudalm

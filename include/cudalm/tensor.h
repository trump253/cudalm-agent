// CUDALM — lightweight tensor metadata (non-owning).
//
// TensorView describes a region of memory that is NOT owned here. v0.1 is
// strictly contiguous row-major; stride is intentionally not modeled (the
// contiguous invariant is documented, not stored). Provenance: CUDALM-native.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudalm {

// Element types. `kInt4Packed` stores two signed INT4 values per byte; its
// "numel" is the number of *packed bytes* (K/2 for a [N, K] weight).
enum class Dtype : std::uint8_t {
  kFp16 = 1,        // 2 bytes per element
  kInt4Packed = 2,  // 1 byte per 2 elements (packed)
  kFp16Scale = 3,   // 2 bytes per element (group scale)
  kFp32 = 4,        // 4 bytes per element
  kBf16 = 5,        // 2 bytes per element (v0.2, .cudalm v2 only)
};

// Bytes per logical element. For kInt4Packed this returns 1 (the shape is
// already expressed in packed bytes), see TensorView::numel.
inline std::size_t dtype_element_bytes(Dtype d) {
  switch (d) {
    case Dtype::kFp16:
    case Dtype::kFp16Scale:
      return 2;
    case Dtype::kInt4Packed:
      return 1;
    case Dtype::kFp32:
      return 4;
    case Dtype::kBf16:
      return 2;
  }
  return 0;
}

inline const char* dtype_name(Dtype d) {
  switch (d) {
    case Dtype::kFp16: return "fp16";
    case Dtype::kInt4Packed: return "int4_packed";
    case Dtype::kFp16Scale: return "fp16_scale";
    case Dtype::kFp32: return "fp32";
    case Dtype::kBf16: return "bf16";
  }
  return "unknown";
}

class TensorView {
 public:
  TensorView() = default;

  // Non-owning view: data + dtype + shape (contiguous, row-major).
  TensorView(void* data, Dtype dtype, std::vector<std::int64_t> shape)
      : data_(data), dtype_(dtype), shape_(std::move(shape)) {}

  const void* data() const { return data_; }
  void* data() { return data_; }
  template <typename T>
  T* data() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  const T* data() const {
    return static_cast<const T*>(data_);
  }

  Dtype dtype() const { return dtype_; }
  const std::vector<std::int64_t>& shape() const { return shape_; }

  int ndim() const { return static_cast<int>(shape_.size()); }
  std::int64_t dim(int i) const { return shape_[i]; }

  // Product of shape. For kInt4Packed this is the number of packed bytes.
  std::int64_t numel() const {
    std::int64_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
  }

  // Total byte size = numel * element bytes.
  std::size_t bytes() const { return static_cast<std::size_t>(numel()) * dtype_element_bytes(dtype_); }

  // v0.1 invariant: always contiguous (stride not modeled).
  bool contiguous() const { return true; }

  std::string shape_string() const {
    std::string s = "[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
      if (i) s += ", ";
      s += std::to_string(shape_[i]);
    }
    s += "]";
    return s;
  }

 private:
  void* data_ = nullptr;
  Dtype dtype_ = Dtype::kFp16;
  std::vector<std::int64_t> shape_;
};

}  // namespace cudalm

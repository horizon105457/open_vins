/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_RING_BUFFER_H
#define OV_MSCKF_RING_BUFFER_H

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace ov_msckf {

/**
 * @brief Lock-free SPSC ring buffer with safe KEEP_LAST overwrite.
 *
 * ## Ownership model
 *
 * Only the producer writes `write_idx_`; only the consumer writes
 * `read_idx_`.  No cross-thread index mutation — no CAS, no regression
 * risk.
 *
 * ## Sequence numbers (per-slot)
 *
 * Each slot carries a monotonic `seq` set to `write_idx + 1` on commit
 * (release).  The consumer expects `seq == read_idx + 1` (acquire).
 * A mismatch signals one of:
 *  - seq == 0:       slot never written
 *  - seq >  r+1:     slot reclaimed by producer (overwrite); consumer
 *                    fast-forwards read_idx to seq-1  (O(1) skip)
 *
 * ## consume_one preemption safety
 *
 * The sequence is snapshotted before and after the user lambda.  If it
 * changed, the slot was overwritten while the consumer was preempted;
 * the read is discarded without advancing read_idx.
 *
 * ## Efficiency vs official deque
 *
 * | Operation          | Deque (official)        | RingBuffer (this)     |
 * |--------------------|-------------------------|-----------------------|
 * | Per-frame alloc    | cv::Mat::clone() 300KB  | 0 (pre-alloc copyTo)  |
 * | Ordering           | std::sort every push    | ring preserves order  |
 * | Synchronisation    | mutex lock/unlock       | lock-free atomics     |
 * | Thread spawn       | 1 per IMU drain cycle   | 0 (synchronous drain) |
 * | Memory bound       | unbounded               | N slots, fixed        |
 *
 * @tparam T  Default-constructible element type.
 * @tparam N  Physical slot count, power of two.
 */
template <typename T, size_t N>
class RingBuffer {
  static_assert(N >= 2, "capacity must be >= 2");
  static_assert((N & (N - 1)) == 0, "capacity must be a power of 2");

  static constexpr size_t kMask = N - 1;

  struct alignas(64) Impl {
    struct Slot {
      T data{};
      std::atomic<size_t> seq{0};
    };
    std::array<Slot, N> buf_{};
    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};
  };

public:
  using value_type = T;
  using size_type = size_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  RingBuffer() : impl_(std::make_unique<Impl>()) {}
  RingBuffer(RingBuffer &&) noexcept = default;
  RingBuffer &operator=(RingBuffer &&) noexcept = default;
  RingBuffer(const RingBuffer &) = delete;
  RingBuffer &operator=(const RingBuffer &) = delete;

  static constexpr size_type capacity() noexcept { return N; }

  size_type size() const noexcept {
    size_type w = impl_->write_idx_.load(std::memory_order_acquire);
    size_type r = impl_->read_idx_.load(std::memory_order_acquire);
    return static_cast<size_type>(w - r);
  }
  bool empty() const noexcept {
    return impl_->write_idx_.load(std::memory_order_acquire) ==
           impl_->read_idx_.load(std::memory_order_acquire);
  }
  bool full() const noexcept { return size() >= N; }

  reference front() {
    assert(!empty());
    return impl_->buf_[(impl_->read_idx_.load(std::memory_order_relaxed)) & kMask].data;
  }
  const_reference front() const {
    assert(!empty());
    return impl_->buf_[(impl_->read_idx_.load(std::memory_order_relaxed)) & kMask].data;
  }
  reference back() {
    assert(!empty());
    size_type w = impl_->write_idx_.load(std::memory_order_relaxed);
    return impl_->buf_[(w - 1) & kMask].data;
  }
  const_reference back() const {
    assert(!empty());
    size_type w = impl_->write_idx_.load(std::memory_order_relaxed);
    return impl_->buf_[(w - 1) & kMask].data;
  }

  void push_back(const T &item) { emplace_copy(item); }
  void push_back(T &&item) { emplace_move(std::move(item)); }
  bool try_push_back(const T &item) { return try_emplace_copy(item); }
  bool try_push_back(T &&item) { return try_emplace_move(std::move(item)); }

  void pop_front() {
    assert(!empty());
    impl_->read_idx_.fetch_add(1, std::memory_order_release);
  }
  bool try_pop_front() {
    size_type r = impl_->read_idx_.load(std::memory_order_acquire);
    size_type w = impl_->write_idx_.load(std::memory_order_acquire);
    if (r == w) return false;
    impl_->read_idx_.store(r + 1, std::memory_order_release);
    return true;
  }
  bool try_pop_front(T &out) {
    size_type r = impl_->read_idx_.load(std::memory_order_acquire);
    size_type w = impl_->write_idx_.load(std::memory_order_acquire);
    if (r == w) return false;
    out = std::move(impl_->buf_[r & kMask].data);
    impl_->read_idx_.store(r + 1, std::memory_order_release);
    return true;
  }

  // --- in-place slot access -----------------------------------------

  /// Always succeeds; overwrites oldest slot when full (KEEP_LAST).
  /// Producer NEVER touches read_idx — the consumer detects and skips.
  reference write_slot() {
    return impl_->buf_[impl_->write_idx_.load(std::memory_order_relaxed) & kMask].data;
  }

  pointer try_write_slot() {
    size_type w = impl_->write_idx_.load(std::memory_order_relaxed);
    size_type r = impl_->read_idx_.load(std::memory_order_acquire);
    if (w - r >= N) return nullptr;
    return &impl_->buf_[w & kMask].data;
  }

  void commit_write() {
    size_type w = impl_->write_idx_.load(std::memory_order_relaxed);
    impl_->buf_[w & kMask].seq.store(w + 1, std::memory_order_release);
    impl_->write_idx_.store(w + 1, std::memory_order_release);
  }

  /// Returns slot or nullptr.  If seq > r+1, fast-forwards read_idx.
  pointer try_read_slot() {
    size_type r = impl_->read_idx_.load(std::memory_order_relaxed);
    auto &slot = impl_->buf_[r & kMask];
    size_type seq = slot.seq.load(std::memory_order_acquire);
    if (seq == r + 1)
      return &slot.data;
    if (seq > r + 1)
      impl_->read_idx_.store(seq - 1, std::memory_order_release);
    return nullptr;
  }

  void commit_read() { impl_->read_idx_.fetch_add(1, std::memory_order_release); }

  // --- lambda API ---------------------------------------------------

  template <typename F, typename = std::enable_if_t<std::is_invocable_v<F, T &>>>
  void emplace_write(F &&fn) { fn(write_slot()); commit_write(); }

  template <typename F, typename = std::enable_if_t<std::is_invocable_v<F, T &>>>
  bool try_emplace_write(F &&fn) {
    pointer p = try_write_slot(); if (!p) return false;
    fn(*p); commit_write(); return true;
  }

  /// Safe consume with preemption-tolerant double-check.
  template <typename F, typename = std::enable_if_t<std::is_invocable_r_v<bool, F, const T &>>>
  bool consume_one(F &&fn) {
    size_type r = impl_->read_idx_.load(std::memory_order_relaxed);
    auto &slot = impl_->buf_[r & kMask];
    size_type s1 = slot.seq.load(std::memory_order_acquire);

    if (s1 != r + 1) {
      if (s1 > r + 1)
        impl_->read_idx_.store(s1 - 1, std::memory_order_release);
      return false;
    }
    if (!fn(slot.data))
      return false;
    if (slot.seq.load(std::memory_order_acquire) != s1)
      return false;                     // overwritten during fn — discard
    impl_->read_idx_.store(r + 1, std::memory_order_release);
    return true;
  }

  template <typename F, typename = std::enable_if_t<std::is_invocable_v<F, const T &>>>
  void consume_all(F &&fn) {
    pointer p;
    while ((p = try_read_slot())) { fn(*p); commit_read(); }
  }

private:
  void emplace_copy(const T &item) { write_slot() = item; commit_write(); }
  void emplace_move(T &&item) { write_slot() = std::move(item); commit_write(); }
  bool try_emplace_copy(const T &item) {
    pointer p = try_write_slot(); if (!p) return false;
    *p = item; commit_write(); return true;
  }
  bool try_emplace_move(T &&item) {
    pointer p = try_write_slot(); if (!p) return false;
    *p = std::move(item); commit_write(); return true;
  }

  std::unique_ptr<Impl> impl_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_RING_BUFFER_H

// Copyright © 2023-2024 Apple Inc.
#include <cstdlib>
#include <iostream>
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

// Opt-in (default OFF): when the environment variable
// MLX_ASYNC_GPU_ERRORS_NONFATAL=1 is set, a command-buffer error detected inside
// an ASYNC Metal completion handler is LOGGED instead of thrown.
//
// Rationale: a C++ throw from a completion handler runs on Metal's dispatch
// worker thread, where there is no surrounding catch frame, so it propagates
// straight to std::terminate -> abort. On iOS the system errors any in-flight
// GPU command buffer the instant the app leaves the foreground
// (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted), so any app
// whose GPU work straddles a foreground->background transition crashes
// UNCATCHABLY. With this opt-in enabled the async handlers fail soft; the
// SYNCHRONOUS check_error() in synchronize() still throws on the CALLER's thread
// (which IS catchable), so callers can still detect and recover from the
// failure. Default preserves upstream throw-everywhere behavior.
static bool async_gpu_errors_nonfatal() {
  static const bool enabled = [] {
    const char* v = std::getenv("MLX_ASYNC_GPU_ERRORS_NONFATAL");
    return v != nullptr && v[0] == '1';
  }();
  return enabled;
}

inline void check_error_async(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() != MTL::CommandBufferStatusError) {
    return;
  }
  if (async_gpu_errors_nonfatal()) {
    std::cerr << "[METAL] Command buffer execution failed (async, non-fatal): "
              << cbuf->error()->localizedDescription()->utf8String()
              << std::endl;
    return;
  }
  check_error(cbuf);
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error_async(cbuf);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error_async(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { check_error_async(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu

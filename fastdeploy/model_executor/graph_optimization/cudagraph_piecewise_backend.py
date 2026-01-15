"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import paddle
import paddle.jit.dy2static.utils as jit_utils
import paddle.nn.layer
from paddle.device.cuda import graphs

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import (
    capture_custom_allreduce,
    custom_ar_clear_ipc_handles,
)
from fastdeploy.utils import get_logger

logger = get_logger("cudagrpah_piecewise_backend", "cudagraph_piecewise_backend.log")


@dataclass
class ConcreteSizeEntry:
    """Record the concrete information corresponding to the current shape(num_tokens)"""

    # Concrete shape
    real_shape: int
    # The size is in cudagraph_capture_sizes
    use_cudagraph: bool = True
    # Has runtime-bs been captured before
    captured: bool = False

    # Need to be captured callable object（dynamic graph or static graph backend）
    runnable: Callable = None  # type: ignore
    # Number of completed warmups
    num_finished_warmup: int = 0
    # Captured cuda graph object corresponding to the current real shape
    cuda_graph: Optional[graphs.CUDAGraph] = None
    # Output buffers of cudagraph
    output_buffers: List[Optional[paddle.Tensor]] = field(default_factory=list)

    # In dynamic graph mode, directly execute the CUDA Graph for the Decode stage.
    # In static graph mode, execute CUDA Graphs for both the D (Decode) stage and PD/P (Prefill) stages.
    # Therefore, we need to distinguish whether it is a prefill CUDA Graph or a decode one.
    is_prefill: bool = False


class Dy2StCudaGraphManager:
    def __init__(self):

        self.state = jit_utils.CUDAGraphState.DISABLE
        self.captured_batch_size = set()
        self.batch_size = -1

    def run_impl(self, original_run_impl, inputs, parameters, attrs):

        run_state = self.state
        prog_attrs, cuda_graph_attrs = attrs
        if run_state == jit_utils.CUDAGraphState.REPLAY:
            if self.batch_size not in self.captured_batch_size:
                run_state = jit_utils.CUDAGraphState.DISABLE
        elif run_state == jit_utils.CUDAGraphState.CAPTURE:
            self.captured_batch_size.add(self.batch_size)

        cuda_graph_attrs |= {
            "cuda_graph_state": run_state,
            "cuda_graph_dispatch_key": self.batch_size if run_state != jit_utils.CUDAGraphState.DISABLE else 0,
        }
        return original_run_impl(inputs, parameters, (prog_attrs, cuda_graph_attrs))

    @contextmanager
    def run_impl_guard(self):
        with paddle.jit.dy2static.pir_partial_program.replace_run_impl_guard(
            self.run_impl,
        ):
            yield


class CudaGraphPiecewiseBackend:
    """Manage the capture and replay of CUDA graphs at the subgraph level."""

    PREFILL_FLAG_BIT = 1 << 30

    def __init__(
        self,
        fd_config: FDConfig,
        runnable: Callable,
    ):
        self.fd_config = fd_config
        self.runnable = runnable
        self.cudagraph_capture_sizes = fd_config.graph_opt_config.cudagraph_capture_sizes
        self.warm_up_size = fd_config.graph_opt_config.cudagraph_num_of_warmups
        self.real_shape_to_captured_size = fd_config.graph_opt_config.real_shape_to_captured_size
        self.real_shape_to_captured_size_for_prefill = (
            fd_config.graph_opt_config.real_shape_to_captured_size_for_prefill
        )
        self.unique_memory_pool_id = None
        if self.fd_config.graph_opt_config.use_unique_memory_pool:
            # TODO(gongshaotian): Optimize code
            if paddle.is_compiled_with_cuda():
                from paddle.base.core import CUDAGraph

                self.unique_memory_pool_id = CUDAGraph.gen_new_memory_pool_id()

        self._create_entry_dict()

        self.is_static_graph = self.fd_config.graph_opt_config.graph_opt_level > 0
        self.cuda_graph_manager = None
        if self.is_static_graph:
            self.cuda_graph_manager = Dy2StCudaGraphManager()

    def run_static_model(self, entry: ConcreteSizeEntry, **kwargs):
        dispatch_token_nums = entry.real_shape | self.PREFILL_FLAG_BIT if entry.is_prefill else entry.real_shape
        if not entry.captured:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                logger.debug(
                    f"[CUDA GRAPH][ID:{id(self)}] Warm up for batch size {entry.real_shape}, "
                    f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                )

            # Store input addresses for debug
            input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
            entry.input_addresses = input_addresses

            # Capture
            self.cuda_graph_manager.state = jit_utils.CUDAGraphState.CAPTURE
            self.cuda_graph_manager.batch_size = dispatch_token_nums
            entry.captured = True
            with capture_custom_allreduce():
                with self.cuda_graph_manager.run_impl_guard():
                    entry.runnable(**kwargs)

        # Replay
        self.cuda_graph_manager.state = jit_utils.CUDAGraphState.REPLAY
        self.cuda_graph_manager.batch_size = dispatch_token_nums
        with self.cuda_graph_manager.run_impl_guard():
            return entry.runnable(**kwargs)

    def __call__(self, **kwargs) -> List[paddle.Tensor] | paddle.Tensor:
        # Get real shape(all num tokens)
        ids_remove_padding: paddle.Tensor = kwargs["forward_meta"].ids_remove_padding
        print(f"__call__ {ids_remove_padding}")
        real_shape = ids_remove_padding.shape[0]

        seq_lens_encoder = kwargs.get("forward_meta").seq_lens_encoder
        is_prefill = bool((seq_lens_encoder > 0).sum().item())

        shape_to_captured_size = self.real_shape_to_captured_size
        if is_prefill and self.is_static_graph:
            shape_to_captured_size = self.real_shape_to_captured_size_for_prefill

        padding_real_shape = shape_to_captured_size[real_shape]
        print(
            f"[CUDAGRAPH_BACKEND] Bucket mapping: real_shape={real_shape} -> padding_real_shape={padding_real_shape}"
        )
        logger.debug(
            f"[CUDA GRAPH][ID:{id(self)}] The actual real shape obtained by CUDAGraph is :{real_shape}, "
            f"The padded shape is :{padding_real_shape}, If Padding :{real_shape != padding_real_shape}"
        )

        entry = self.concrete_size_entries.get((padding_real_shape, is_prefill))
        assert entry is not None, f"real shape:{padding_real_shape} is not in cuda graph capture list."
        if entry.runnable is None:
            entry.runnable = self.runnable
            logger.debug(f"[CUDA GRAPH][ID:{id(self)}] New entry lazy initialize with real shape {padding_real_shape}")

        print(f"[CUDAGRAPH_BACKEND] Entry info: use_cudagraph={entry.use_cudagraph}, captured={entry.captured}")
        if not entry.use_cudagraph:
            return entry.runnable(**kwargs)

        if self.is_static_graph:
            entry.is_prefill = is_prefill
            kwargs["forward_meta"].step_use_cudagraph = True  # 这部分能删除吗？
            return self.run_static_model(entry, **kwargs)

        # Capture a new cuda graph
        if entry.cuda_graph is None:
            # Warmup the model
            for n in range(entry.num_finished_warmup, self.warm_up_size):
                entry.num_finished_warmup += 1
                entry.runnable(**kwargs)
                logger.info(
                    f"[CUDA GRAPH][ID:{id(self)}] Warm up for real shape {padding_real_shape}, "
                    f"finished ({n + 1}/{entry.num_finished_warmup}) times"
                )

            # Store input addresses for debug
            input_addresses = [x.data_ptr() for (_, x) in kwargs.items() if isinstance(x, paddle.Tensor)]
            entry.input_addresses = input_addresses

            new_grpah = graphs.CUDAGraph(pool_id=self.unique_memory_pool_id)
            paddle.device.synchronize()

            # Capture
            with capture_custom_allreduce():
                new_grpah.capture_begin()
                outputs = entry.runnable(**kwargs)
                if isinstance(outputs, paddle.Tensor):
                    assert outputs is not None
                    outputs = [outputs]
                new_grpah.capture_end()

            # Store output buffer
            entry.cuda_graph = new_grpah
            for output in outputs:
                if output is not None:
                    output_buffer = paddle.zeros_like(output)
                    output._share_buffer_to(output_buffer)
                    output._clear
                    entry.output_buffers.append(output_buffer)
                else:
                    entry.output_buffers.append(None)

            paddle.device.synchronize()

            # For CUDAGraph debug
            # self._save_cudagrpah_dot_files(entry)
            logger.info(f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph captured for real shape {padding_real_shape}")

        # Replay
        entry.cuda_graph.replay()
        logger.debug(f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph replayed for real shape {padding_real_shape}")
        if len(entry.output_buffers) == 1:
            return entry.output_buffers[0]
        return entry.output_buffers

    def _create_entry_dict(self):
        """ """
        # Runtime real shape -> ConcreteSizeEntry
        self.concrete_size_entries: Dict[(int, bool), ConcreteSizeEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_size_entries[(shape, False)] = ConcreteSizeEntry(real_shape=shape)

        for shape in self.real_shape_to_captured_size_for_prefill:
            self.concrete_size_entries[(shape, True)] = ConcreteSizeEntry(real_shape=shape)

        logger.info(
            f"[CUDA GRAPH][ID:{id(self)}] CUDAGraph capture list {self.cudagraph_capture_sizes}, "
            "Created all real shape entry."
        )

    def clear_graph(self):
        """ """
        # Clear graphs
        custom_ar_clear_ipc_handles()
        for (_id, _), entry in self.concrete_size_entries.items():
            if entry.cuda_graph:
                del entry.cuda_graph
                logger.debug(f"[CUDA GRAPH][ID:{id(self)}] The CUDAGraph with shape {_id} has been cleared.")

        del self.concrete_size_entries
        paddle.device.cuda.empty_cache()

        # Create new entrys
        self._create_entry_dict()

    def _save_cudagrpah_dot_files(self, entry):
        """Print CUDAGrpah to dot files"""
        log_dir = envs.FD_LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        if entry.cuda_graph:
            entry.cuda_graph.print_to_dot_files(
                f"{log_dir}/GraphDotFiles/backend{id(self)}_shape{entry.real_shape}",
                1 << 0,
            )

    def check_capture_successful(self):
        """Check whether the shapes are captured or not"""
        for (shape, _), entry in self.concrete_size_entries.items():
            if not entry.captured:
                raise ValueError(f"[CUDA GRAPH][ID:{id(self)}] Shape {shape} capture failed.")

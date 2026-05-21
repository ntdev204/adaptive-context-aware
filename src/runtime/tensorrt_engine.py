from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class EngineRunner(Protocol):
    def run(self, *inputs: np.ndarray) -> np.ndarray:
        """Run one TensorRT-backed inference and return the first output."""


class TensorRTEngineRunner:
    """Load TensorRT `.engine` artifacts produced for the target Jetson runtime."""

    def __init__(self, engine_path: Path, input_names: tuple[str, ...] = ()) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        errors: list[str] = []
        try:
            self.runner: EngineRunner = _TorchScriptTensorRTRunner(engine_path)
            return
        except Exception as exc:
            errors.append(f"torch-tensorrt artifact load failed: {exc}")

        try:
            self.runner = _RawTensorRTRunner(engine_path, input_names)
            return
        except Exception as exc:
            errors.append(f"raw TensorRT engine load failed: {exc}")

        raise RuntimeError(f"unable to load TensorRT engine {engine_path}: {'; '.join(errors)}")

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        return self.runner.run(*inputs)

    def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if close is not None:
            close()


class _TorchScriptTensorRTRunner:
    def __init__(self, engine_path: Path) -> None:
        import torch

        try:
            import torch_tensorrt  # noqa: F401
        except ImportError:
            pass

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = torch.jit.load(str(engine_path), map_location=self.device).eval()

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        tensors = [
            self.torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=self.device)
            for values in inputs
        ]
        with self.torch.no_grad():
            output = self.module(*tensors)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().cpu().numpy()


@dataclass(slots=True)
class _Allocation:
    host: np.ndarray
    ptr: int
    nbytes: int


class _RawTensorRTRunner:
    def __init__(self, engine_path: Path, input_names: tuple[str, ...]) -> None:
        import tensorrt as trt

        self.trt = trt
        self.cudart = _load_cudart_module()
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize engine bytes")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create execution context")

        self.tensor_names = self._tensor_names()
        discovered_inputs = self._input_names()
        self.input_names = input_names if input_names else tuple(discovered_inputs)
        missing_inputs = set(self.input_names) - set(discovered_inputs)
        if missing_inputs:
            raise RuntimeError(f"engine is missing expected inputs: {sorted(missing_inputs)}")
        self.output_names = tuple(name for name in self.tensor_names if name not in discovered_inputs)
        if not self.output_names:
            raise RuntimeError("engine has no output tensors")
        self._allocations: dict[str, _Allocation] = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        allocations = getattr(self, "_allocations", {})
        for allocation in allocations.values():
            self._cuda_check(self.cudart.cudaFree(allocation.ptr), "cudaFree")
        allocations.clear()

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != len(self.input_names):
            raise ValueError(f"expected {len(self.input_names)} inputs, got {len(inputs)}")

        host_inputs = {
            name: np.ascontiguousarray(values, dtype=np.float32)
            for name, values in zip(self.input_names, inputs, strict=True)
        }
        for name, values in host_inputs.items():
            self._set_input_shape(name, values.shape)

        for name, values in host_inputs.items():
            allocation = self._get_or_alloc(name, values)
            np.copyto(allocation.host, values)
            self._copy_host_to_device(allocation)

        for name in self.output_names:
            shape = self._shape(name)
            dtype = self._dtype(name)
            output_host = np.empty(shape, dtype=dtype)
            self._get_or_alloc(name, output_host)

        self._execute(self._allocations)
        first_output = self._allocations[self.output_names[0]]
        self._copy_device_to_host(first_output)
        self._device_synchronize()
        return first_output.host

    def _tensor_names(self) -> tuple[str, ...]:
        if hasattr(self.engine, "num_io_tensors"):
            return tuple(self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors))
        return tuple(self.engine.get_binding_name(index) for index in range(self.engine.num_bindings))

    def _input_names(self) -> tuple[str, ...]:
        if hasattr(self.engine, "get_tensor_mode"):
            input_mode = self.trt.TensorIOMode.INPUT
            return tuple(name for name in self.tensor_names if self.engine.get_tensor_mode(name) == input_mode)
        return tuple(
            self.engine.get_binding_name(index)
            for index in range(self.engine.num_bindings)
            if self.engine.binding_is_input(index)
        )

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(name, shape)
            return
        index = self.engine.get_binding_index(name)
        if any(dim < 0 for dim in self.engine.get_binding_shape(index)):
            self.context.set_binding_shape(index, shape)

    def _shape(self, name: str) -> tuple[int, ...]:
        if hasattr(self.context, "get_tensor_shape"):
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        else:
            shape = tuple(int(dim) for dim in self.context.get_binding_shape(self.engine.get_binding_index(name)))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"TensorRT output shape for {name} is unresolved: {shape}")
        return shape

    def _dtype(self, name: str) -> np.dtype:
        if hasattr(self.engine, "get_tensor_dtype"):
            trt_dtype = self.engine.get_tensor_dtype(name)
        else:
            trt_dtype = self.engine.get_binding_dtype(self.engine.get_binding_index(name))
        return np.dtype(self.trt.nptype(trt_dtype))

    def _get_or_alloc(self, name: str, host: np.ndarray) -> _Allocation:
        existing = self._allocations.get(name)
        if existing is not None and existing.host.shape == host.shape and existing.host.dtype == host.dtype:
            return existing
        if existing is not None:
            self._cuda_check(self.cudart.cudaFree(existing.ptr), "cudaFree")
        allocation = self._allocate(np.empty_like(host))
        self._allocations[name] = allocation
        return allocation

    def _allocate(self, host: np.ndarray) -> _Allocation:
        result = self.cudart.cudaMalloc(host.nbytes)
        self._cuda_check(result, "cudaMalloc")
        return _Allocation(host=host, ptr=int(result[1]), nbytes=host.nbytes)

    def _copy_host_to_device(self, allocation: _Allocation) -> None:
        self._cuda_check(
            self.cudart.cudaMemcpy(
                allocation.ptr,
                int(allocation.host.ctypes.data),
                allocation.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpyHostToDevice",
        )

    def _copy_device_to_host(self, allocation: _Allocation) -> None:
        self._cuda_check(
            self.cudart.cudaMemcpy(
                int(allocation.host.ctypes.data),
                allocation.ptr,
                allocation.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpyDeviceToHost",
        )

    def _execute(self, allocations: dict[str, _Allocation]) -> None:
        if hasattr(self.context, "set_tensor_address"):
            for name in self.tensor_names:
                self.context.set_tensor_address(name, allocations[name].ptr)
            try:
                ok = self.context.execute_async_v3(stream_handle=0)
            except TypeError:
                ok = self.context.execute_async_v3(0)
        else:
            bindings = [
                allocations[self.engine.get_binding_name(index)].ptr for index in range(self.engine.num_bindings)
            ]
            ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    def _device_synchronize(self) -> None:
        self._cuda_check(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    @staticmethod
    def _cuda_check(result: object, operation: str) -> None:
        error = result[0] if isinstance(result, tuple) else result
        if int(getattr(error, "value", error)) != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {error}")


def _load_cudart_module() -> object:
    errors: list[str] = []
    candidates = (
        ("cuda.bindings.runtime", None),
        ("cuda.cudart", None),
        ("cuda", "cudart"),
        ("cuda.cuda", None),
    )
    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(module_name)
            candidate = getattr(module, attribute_name) if attribute_name is not None else module
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        if _looks_like_cudart(candidate):
            return candidate
        errors.append(f"{module_name}: missing cuda runtime symbols")
    raise RuntimeError(
        "could not import CUDA runtime bindings; tried "
        + ", ".join(module for module, _ in candidates)
        + f" ({'; '.join(errors)})"
    )


def _looks_like_cudart(candidate: object) -> bool:
    required = ("cudaMalloc", "cudaMemcpy", "cudaDeviceSynchronize", "cudaFree", "cudaMemcpyKind")
    return all(hasattr(candidate, name) for name in required)

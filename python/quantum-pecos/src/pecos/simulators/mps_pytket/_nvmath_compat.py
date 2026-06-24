# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Compatibility patches for pytket-cutensornet's nvmath dependency."""

from __future__ import annotations

from importlib import import_module

_PATCHED_ATTR = "_pecos_cupy_stream_from_external_patch"


class _CudaStreamHolder:
    """Adapter object implementing the CUDA stream protocol for a raw stream handle."""

    def __init__(self, handle: int) -> None:
        self.handle = int(handle)

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


def patch_nvmath_cupy_external_stream() -> bool:
    """Use CuPy's supported external-stream API in nvmath when it is available.

    nvmath-python 0.9.0 still wraps raw CUDA stream pointers with
    ``cupy.cuda.ExternalStream``, which is deprecated in CuPy 14. CuPy's
    replacement API accepts an object implementing the CUDA stream protocol.
    """
    try:
        cp = import_module("cupy")
        package_ifc_cupy = import_module("nvmath.internal.package_ifc_cupy")
    except ImportError:
        return False

    try:
        cupy_package = package_ifc_cupy.CupyPackage
        from_external = getattr(cp.cuda.Stream, "from_external", None)
    except AttributeError:
        return False

    if getattr(cupy_package, _PATCHED_ATTR, False):
        return True

    if from_external is None:
        return False

    def create_external_stream(device_id: int, stream_ptr: int) -> object:
        del device_id
        return from_external(_CudaStreamHolder(stream_ptr))

    cupy_package.create_external_stream = staticmethod(create_external_stream)
    setattr(cupy_package, _PATCHED_ATTR, True)
    return True

// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! GPU level enumeration for exact bounded-enumeration code distance.
//!
//! The information-set enumeration follows
//! [arXiv:2408.10743](https://arxiv.org/abs/2408.10743). The GPU returns only a level's minimum
//! weight. When that improves the upper bound, `pecos-qec` deterministically re-scans the level on
//! the CPU to reconstruct the first witness in native enumeration order.

use crate::gpu_probe::{GpuAdapterInfo, GpuStartupError, gpu_context};
use bytemuck::{Pod, Zeroable};
use pecos_qec::{
    BoundedEnumerationBackendError, BoundedEnumerationDistance, DistanceProblemError,
    LevelEnumerationBackend, LevelEnumerationInput, LevelEnumerationMinimum, ParityCheckMatrix,
    StabilizerCodeSpec, bounded_enumeration_code_distance_with_backend,
    bounded_enumeration_stabilizer_distance_with_backend,
    bounded_enumeration_x_distance_with_backend, bounded_enumeration_z_distance_with_backend,
};
use wgpu::util::DeviceExt;

const MAX_ROW_STRIDE_WORDS: usize = 8;
const MAX_DIMENSION: usize = 256;
const WORKGROUP_SIZE: u32 = 64;
const RANKS_PER_THREAD: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EnumerationParams {
    dimension: u32,
    level: u32,
    row_stride: u32,
    logical_count: u32,
    table_width: u32,
    rank_offset_lo: u32,
    rank_offset_hi: u32,
    rank_count: u32,
    ranks_per_thread: u32,
    invocation_count: u32,
    _padding_0: u32,
    _padding_1: u32,
}

/// Failure from GPU-accelerated bounded enumeration.
#[derive(Debug)]
pub enum GpuBoundedEnumerationError {
    /// No usable hardware adapter, or device creation failed.
    Startup(GpuStartupError),
    /// The QEC distance problem is invalid.
    DistanceProblem(DistanceProblemError),
    /// Packed codewords exceed the kernel's initial eight-word limit.
    CodewordTooWide { columns: usize, maximum: usize },
    /// The generator dimension exceeds the kernel's combination-array limit.
    DimensionTooLarge { dimension: usize, maximum: usize },
    /// The number of combinations cannot be represented by the kernel's 64-bit rank encoding.
    CombinationCountOverflow { dimension: usize, level: usize },
    /// A backend input could not be represented by the WGSL `u32` interface.
    IntegerConversion { field: &'static str, value: usize },
    /// A GPU result buffer could not be mapped.
    BufferMap(String),
    /// Waiting for GPU completion failed.
    DevicePoll(String),
}

impl std::fmt::Display for GpuBoundedEnumerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Startup(error) => error.fmt(formatter),
            Self::DistanceProblem(error) => error.fmt(formatter),
            Self::CodewordTooWide { columns, maximum } => write!(
                formatter,
                "GPU bounded enumeration supports at most {maximum} columns ({MAX_ROW_STRIDE_WORDS} packed u32 words), got {columns}"
            ),
            Self::DimensionTooLarge { dimension, maximum } => write!(
                formatter,
                "GPU bounded enumeration supports generator dimension at most {maximum}, got {dimension}"
            ),
            Self::CombinationCountOverflow { dimension, level } => write!(
                formatter,
                "C({dimension}, {level}) exceeds the GPU backend's 64-bit combination-rank range"
            ),
            Self::IntegerConversion { field, value } => write!(
                formatter,
                "GPU bounded-enumeration {field} value {value} does not fit in u32"
            ),
            Self::BufferMap(error) => {
                write!(formatter, "GPU result buffer mapping failed: {error}")
            }
            Self::DevicePoll(error) => {
                write!(formatter, "waiting for GPU completion failed: {error}")
            }
        }
    }
}

impl std::error::Error for GpuBoundedEnumerationError {}

impl From<GpuStartupError> for GpuBoundedEnumerationError {
    fn from(error: GpuStartupError) -> Self {
        Self::Startup(error)
    }
}

/// Explicit wgpu backend for bounded-enumeration levels.
///
/// Construction fails when the repository's shared GPU probe cannot select a hardware adapter;
/// callers receive that error and no CPU fallback is attempted.
pub struct GpuBoundedEnumerationBackend {
    adapter_info: GpuAdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl GpuBoundedEnumerationBackend {
    /// Detects a hardware adapter and creates the reusable level-enumeration pipeline.
    ///
    /// # Errors
    ///
    /// Returns the shared GPU probe's explicit startup error when no supported adapter exists.
    pub fn try_new() -> Result<Self, GpuBoundedEnumerationError> {
        let context = gpu_context()?;
        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("PECOS bounded-enumeration shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("bounded_enumeration_shader.wgsl").into(),
                ),
            });
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("PECOS bounded-enumeration bind-group layout"),
                    entries: &[
                        Self::buffer_layout_entry(0, wgpu::BufferBindingType::Uniform),
                        Self::buffer_layout_entry(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        Self::buffer_layout_entry(
                            2,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        Self::buffer_layout_entry(
                            3,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        Self::buffer_layout_entry(
                            4,
                            wgpu::BufferBindingType::Storage { read_only: false },
                        ),
                    ],
                });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("PECOS bounded-enumeration pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    ..Default::default()
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PECOS bounded-enumeration pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("enumerate_level"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok(Self {
            adapter_info: context.info,
            device: context.device,
            queue: context.queue,
            bind_group_layout,
            pipeline,
        })
    }

    /// Returns details of the hardware adapter selected by the shared GPU probe.
    #[must_use]
    pub fn adapter_info(&self) -> &GpuAdapterInfo {
        &self.adapter_info
    }

    fn buffer_layout_entry(
        binding: u32,
        ty: wgpu::BufferBindingType,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn checked_u32(field: &'static str, value: usize) -> Result<u32, GpuBoundedEnumerationError> {
        u32::try_from(value)
            .map_err(|_| GpuBoundedEnumerationError::IntegerConversion { field, value })
    }

    fn binomial(dimension: usize, level: usize) -> Option<u64> {
        let choose = level.min(dimension - level);
        let mut result = 1u128;
        for index in 1..=choose {
            result = result * (dimension - choose + index) as u128 / index as u128;
            if result > u128::from(u64::MAX) {
                return None;
            }
        }
        u64::try_from(result).ok()
    }

    fn binomial_clamped(dimension: usize, level: usize) -> u64 {
        if level > dimension {
            return 0;
        }
        let choose = level.min(dimension - level);
        let mut result = 1u128;
        for index in 1..=choose {
            result = result * (dimension - choose + index) as u128 / index as u128;
            if result > u128::from(u64::MAX) {
                return u64::MAX;
            }
        }
        u64::try_from(result).unwrap_or(u64::MAX)
    }

    fn split_u64(value: u64) -> (u32, u32) {
        let bytes = value.to_le_bytes();
        (
            u32::from_le_bytes(bytes[..4].try_into().expect("four-byte low rank half")),
            u32::from_le_bytes(bytes[4..].try_into().expect("four-byte high rank half")),
        )
    }

    fn binomial_table(dimension: usize, level: usize) -> Vec<u32> {
        let mut table = Vec::with_capacity((dimension + 1) * (level + 1) * 2);
        for n in 0..=dimension {
            for k in 0..=level {
                let value = Self::binomial_clamped(n, k);
                let (low, high) = Self::split_u64(value);
                table.push(low);
                table.push(high);
            }
        }
        table
    }

    fn create_storage_buffer(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn enumerate(
        &self,
        input: LevelEnumerationInput<'_>,
    ) -> Result<LevelEnumerationMinimum, GpuBoundedEnumerationError> {
        if input.row_stride_words > MAX_ROW_STRIDE_WORDS {
            return Err(GpuBoundedEnumerationError::CodewordTooWide {
                columns: input.codeword_bits,
                maximum: MAX_ROW_STRIDE_WORDS * 32,
            });
        }
        if input.dimension > MAX_DIMENSION {
            return Err(GpuBoundedEnumerationError::DimensionTooLarge {
                dimension: input.dimension,
                maximum: MAX_DIMENSION,
            });
        }
        let combination_count = Self::binomial(input.dimension, input.level).ok_or(
            GpuBoundedEnumerationError::CombinationCountOverflow {
                dimension: input.dimension,
                level: input.level,
            },
        )?;
        let dimension = Self::checked_u32("dimension", input.dimension)?;
        let level = Self::checked_u32("level", input.level)?;
        let row_stride = Self::checked_u32("row stride", input.row_stride_words)?;
        let logical_count = Self::checked_u32("logical count", input.logical_count)?;
        let table_width = Self::checked_u32("binomial table width", input.level + 1)?;

        let logical_buffer = self
            .create_storage_buffer("PECOS bounded-enumeration logical rows", input.logical_rows);
        let binomial_values = Self::binomial_table(input.dimension, input.level);
        let binomial_buffer = self
            .create_storage_buffer("PECOS bounded-enumeration binomial table", &binomial_values);
        let minimum_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PECOS bounded-enumeration minimum"),
                contents: bytemuck::bytes_of(&u32::MAX),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PECOS bounded-enumeration minimum readback"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let max_invocations = u64::from(self.device.limits().max_compute_workgroups_per_dimension)
            * u64::from(WORKGROUP_SIZE);
        let max_ranks_per_dispatch =
            (max_invocations * u64::from(RANKS_PER_THREAD)).min(u64::from(u32::MAX));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PECOS bounded-enumeration command encoder"),
            });

        for &systematic_index in input.active_systematic_indices {
            let generator_buffer = self.create_storage_buffer(
                "PECOS bounded-enumeration generator rows",
                &input.systematic_generators[systematic_index].rows,
            );
            let mut rank_offset = 0u64;
            while rank_offset < combination_count {
                let rank_count = (combination_count - rank_offset).min(max_ranks_per_dispatch);
                let invocation_count = rank_count.div_ceil(u64::from(RANKS_PER_THREAD));
                let (rank_offset_lo, rank_offset_hi) = Self::split_u64(rank_offset);
                let rank_count =
                    u32::try_from(rank_count).expect("dispatch rank count was capped at u32::MAX");
                let invocation_count = u32::try_from(invocation_count)
                    .expect("dispatch invocation count fits the device workgroup limit");
                let params = EnumerationParams {
                    dimension,
                    level,
                    row_stride,
                    logical_count,
                    table_width,
                    rank_offset_lo,
                    rank_offset_hi,
                    rank_count,
                    ranks_per_thread: RANKS_PER_THREAD,
                    invocation_count,
                    _padding_0: 0,
                    _padding_1: 0,
                };
                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("PECOS bounded-enumeration parameters"),
                            contents: bytemuck::bytes_of(&params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PECOS bounded-enumeration bind group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: generator_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: logical_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: binomial_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: minimum_buffer.as_entire_binding(),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("PECOS bounded-enumeration level pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(invocation_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                }
                rank_offset += u64::from(rank_count);
            }
        }
        encoder.copy_buffer_to_buffer(&minimum_buffer, 0, &readback, 0, size_of::<u32>() as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuBoundedEnumerationError::DevicePoll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| GpuBoundedEnumerationError::BufferMap(error.to_string()))?
            .map_err(|error| GpuBoundedEnumerationError::BufferMap(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let weight = bytemuck::from_bytes::<u32>(&mapped).to_owned();
        drop(mapped);
        readback.unmap();

        Ok(LevelEnumerationMinimum {
            weight: (weight != u32::MAX)
                .then_some(usize::try_from(weight).expect("u32 codeword weight always fits usize")),
            witness: None,
        })
    }
}

impl LevelEnumerationBackend for GpuBoundedEnumerationBackend {
    type Error = GpuBoundedEnumerationError;

    fn enumerate_level(
        &mut self,
        input: LevelEnumerationInput<'_>,
    ) -> Result<LevelEnumerationMinimum, Self::Error> {
        self.enumerate(input)
    }
}

fn flatten_backend_error(
    error: BoundedEnumerationBackendError<GpuBoundedEnumerationError>,
) -> GpuBoundedEnumerationError {
    match error {
        BoundedEnumerationBackendError::DistanceProblem(error) => {
            GpuBoundedEnumerationError::DistanceProblem(error)
        }
        BoundedEnumerationBackendError::Backend(error) => error,
    }
}

/// Computes binary `(H, L)` distance with explicit GPU level enumeration.
///
/// # Errors
///
/// Returns adapter, kernel-limit, dispatch, or readback failures. It never silently falls back to
/// CPU level enumeration.
pub fn gpu_bounded_enumeration_code_distance(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, GpuBoundedEnumerationError> {
    let mut backend = GpuBoundedEnumerationBackend::try_new()?;
    bounded_enumeration_code_distance_with_backend(h, l, max_level, &mut backend)
}

/// Computes pure-X CSS distance with explicit GPU level enumeration.
///
/// # Errors
///
/// Returns invalid-code, adapter, kernel-limit, dispatch, or readback failures.
pub fn gpu_bounded_enumeration_x_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, GpuBoundedEnumerationError> {
    let mut backend = GpuBoundedEnumerationBackend::try_new()?;
    bounded_enumeration_x_distance_with_backend(code, max_level, &mut backend)
        .map_err(flatten_backend_error)
}

/// Computes pure-Z CSS distance with explicit GPU level enumeration.
///
/// # Errors
///
/// Returns invalid-code, adapter, kernel-limit, dispatch, or readback failures.
pub fn gpu_bounded_enumeration_z_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, GpuBoundedEnumerationError> {
    let mut backend = GpuBoundedEnumerationBackend::try_new()?;
    bounded_enumeration_z_distance_with_backend(code, max_level, &mut backend)
        .map_err(flatten_backend_error)
}

/// Computes general stabilizer-code distance with explicit GPU level enumeration.
///
/// # Errors
///
/// Returns invalid-code, adapter, kernel-limit, dispatch, or readback failures.
pub fn gpu_bounded_enumeration_stabilizer_distance(
    code: &StabilizerCodeSpec,
    max_level: usize,
) -> Result<Option<BoundedEnumerationDistance>, GpuBoundedEnumerationError> {
    let mut backend = GpuBoundedEnumerationBackend::try_new()?;
    bounded_enumeration_stabilizer_distance_with_backend(code, max_level, &mut backend)
        .map_err(flatten_backend_error)
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_parses_and_validates_without_an_adapter() {
        let module =
            wgpu::naga::front::wgsl::parse_str(include_str!("bounded_enumeration_shader.wgsl"))
                .expect("bounded-enumeration WGSL must parse");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("bounded-enumeration WGSL must validate");
    }
}

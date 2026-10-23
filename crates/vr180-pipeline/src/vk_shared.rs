//! **Vulkan Video decode** on Windows: a `VkDevice` shared by wgpu and
//! FFmpeg's H.264/HEVC Vulkan hwaccel, so decoded frames reach wgpu without
//! D3D11 or any cross-API import. Two such devices exist: the export's
//! ([`shared_export_device`]) and the app's own presenting device
//! ([`create_ui_device`], which the live preview decodes on). Every zero-copy
//! decode path looks its device up with [`decode_device_for`] and falls back
//! to D3D11 decode when there is none or the stream can't be decoded here.
//!
//! Why: the D3D11 decode path hands every frame from D3D11 to Vulkan through
//! an NT-handle import ordered only by CPU waits; a reused (pooled) import
//! hung the GPU here. On these devices FFmpeg decodes into Vulkan images and
//! [`VkDecodeDevice::with_frame_planes`] hands each frame to wgpu under
//! FFmpeg's own frame lock + timeline semaphores with explicit barriers —
//! synchronized by construction, without a CPU wait on the GPU (safe on the
//! presenting device), and byte-identical to the D3D11 path's output.
//!
//! The ProRes ENCODER deliberately stays on FFmpeg's own device: with it on
//! the export device too, the 8K export ran ~3× slower (wgpu's work waited
//! behind the encoder's inside one GPU context, regardless of queue count or
//! priority) and its output was not repeatable run to run.
//!
//! Validating with the Khronos layer: disable its handle wrapping
//! (`VK_KHRONOS_VALIDATION_DISABLES=VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT`;
//! every check still runs). With wrapping on, the layer hands the driver a
//! copy of each decode command's `VkVideoDecodeH264PictureInfoKHR` and frees
//! it once the command is recorded — and the NVIDIA driver (32.0.16.1742,
//! RTX 5090) reads that struct again at `vkQueueSubmit2`, so H.264 decode
//! crashes in `nvoglv64.dll`. Proven without the layer: overwriting that
//! struct after `vkCmdDecodeVideoKHR` returns crashes the same way; its
//! `pStdPictureInfo` / slice offsets, the reference slots and HEVC's
//! picture info don't. FFmpeg keeps the struct alive until it submits, so
//! normal runs are unaffected. Left: `VUID-VkVideoBeginCodingInfoKHR-slotIndex-07245`
//! (FFmpeg creates reference views with decode-output usage only).
//!
//! How: wgpu-hal and FFmpeg describe device features differently — wgpu
//! chains the individual (promoted) feature structs, FFmpeg reads the
//! Vulkan 1.1/1.2/1.3 aggregates — and a create chain may not mix them. So
//! the device is created HERE: FFmpeg's own recipe (read off a throwaway
//! FFmpeg device, so its decoder sees the features it would have picked
//! itself) ∪ what wgpu-hal needs for the requested wgpu features, expressed
//! once as aggregates. Then it is wrapped twice: wgpu via `device_from_raw`
//! (queue family 0, queue 1) and FFmpeg via a user-filled
//! `AVVulkanDeviceContext` (decode, compute, transfer families + family 0
//! queue 0). The two libraries never submit to the same `VkQueue`.

use crate::{Error, Result};
use ash::vk;
use ffmpeg_next as ffmpeg;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, OnceLock};

/// FFmpeg 8.1 (libavutil 60) `AVVulkanDeviceQueueFamily` — ffmpeg-sys-next
/// generates no Vulkan bindings, so the layout is mirrored by hand (checked
/// against a live FFmpeg device by `examples/vk_ffmpeg_probe.rs`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AvQueueFamily {
    idx: c_int,
    num: c_int,
    flags: u32,
    video_caps: u32,
}

/// FFmpeg 8.1 (libavutil 60) `AVVulkanDeviceContext`, with both
/// `FF_API_VULKAN_FIXED_QUEUES` and `FF_API_VULKAN_SYNC_QUEUES` on.
#[repr(C)]
struct AvVulkanDeviceContext {
    alloc: *const c_void,
    get_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    inst: vk::Instance,
    phys_dev: vk::PhysicalDevice,
    act_dev: vk::Device,
    device_features: vk::PhysicalDeviceFeatures2<'static>,
    enabled_inst_extensions: *const *const c_char,
    nb_enabled_inst_extensions: c_int,
    enabled_dev_extensions: *const *const c_char,
    nb_enabled_dev_extensions: c_int,
    queue_family_index: c_int,
    nb_graphics_queues: c_int,
    queue_family_tx_index: c_int,
    nb_tx_queues: c_int,
    queue_family_comp_index: c_int,
    nb_comp_queues: c_int,
    queue_family_encode_index: c_int,
    nb_encode_queues: c_int,
    queue_family_decode_index: c_int,
    nb_decode_queues: c_int,
    lock_queue: *const c_void,
    unlock_queue: *const c_void,
    qf: [AvQueueFamily; 64],
    nb_qf: c_int,
}

/// The aggregate feature description used both to CREATE the device and to
/// DESCRIBE it to FFmpeg. Leaked once per process (the device is too).
struct Features {
    core: vk::PhysicalDeviceFeatures2<'static>,
    v11: vk::PhysicalDeviceVulkan11Features<'static>,
    v12: vk::PhysicalDeviceVulkan12Features<'static>,
    v13: vk::PhysicalDeviceVulkan13Features<'static>,
    coop_matrix: vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static>,
    atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
    expect_assume: vk::PhysicalDeviceShaderExpectAssumeFeaturesKHR<'static>,
    shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT<'static>,
    wg_explicit_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
    robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    video_maintenance1: vk::PhysicalDeviceVideoMaintenance1FeaturesKHR<'static>,
}

/// Extension feature structs we can carry, by the extension that gates them.
/// FFmpeg extensions whose features ash can't name are left off — FFmpeg
/// treats every device extension as optional.
const EXT_FEATURES: &[&CStr] = &[
    vk::KHR_COOPERATIVE_MATRIX_NAME,
    vk::EXT_SHADER_ATOMIC_FLOAT_NAME,
    vk::KHR_SHADER_SUBGROUP_ROTATE_NAME,
    vk::KHR_SHADER_EXPECT_ASSUME_NAME,
    vk::EXT_SHADER_OBJECT_NAME,
    vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME,
    vk::EXT_ROBUSTNESS2_NAME,
    vk::KHR_VIDEO_MAINTENANCE1_NAME,
];

/// FFmpeg device extensions that need no feature struct and are safe to
/// carry over (external memory/semaphores, push descriptors).
///
/// NOT `VK_EXT_external_memory_host`: FFmpeg only sets its
/// `avoid_host_import` (default on NVIDIA) when it creates the device
/// itself, so on a user-supplied device every upload imports the CPU frame
/// in place (`host_map_frame`) — which lost the device intermittently here.
/// Without the extension it uses its staging copy, as on its own device.
const EXT_PLAIN: &[&CStr] = &[
    vk::KHR_PUSH_DESCRIPTOR_NAME,
    vk::KHR_EXTERNAL_MEMORY_WIN32_NAME,
    vk::KHR_EXTERNAL_SEMAPHORE_WIN32_NAME,
    // Vulkan Video decode (FFmpeg's hevc/h264 hwaccel on this device).
    vk::KHR_VIDEO_QUEUE_NAME,
    vk::KHR_VIDEO_DECODE_QUEUE_NAME,
    vk::KHR_VIDEO_DECODE_H265_NAME,
    vk::KHR_VIDEO_DECODE_H264_NAME,
];

/// OR every VkBool32 of `src` into `dst` — Vulkan feature structs are
/// `{ sType, pNext, VkBool32... }`, so the bools start at byte 16 on 64-bit.
unsafe fn or_bools<T>(dst: &mut T, src: *const T) {
    let n = (std::mem::size_of::<T>() - 16) / 4;
    let d = (dst as *mut T as *mut u8).add(16) as *mut u32;
    let s = (src as *const u8).add(16) as *const u32;
    for i in 0..n {
        *d.add(i) |= *s.add(i);
    }
}

fn or_core(dst: &mut vk::PhysicalDeviceFeatures, src: &vk::PhysicalDeviceFeatures) {
    const N: usize = std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4;
    let d = unsafe { &mut *(dst as *mut _ as *mut [u32; N]) };
    let s = unsafe { &*(src as *const _ as *const [u32; N]) };
    for i in 0..N {
        d[i] |= s[i];
    }
}

impl Features {
    fn new() -> Box<Self> {
        Box::new(Self {
            core: vk::PhysicalDeviceFeatures2::default(),
            v11: vk::PhysicalDeviceVulkan11Features::default(),
            v12: vk::PhysicalDeviceVulkan12Features::default(),
            v13: vk::PhysicalDeviceVulkan13Features::default(),
            coop_matrix: vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default(),
            atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default(),
            subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR::default(),
            expect_assume: vk::PhysicalDeviceShaderExpectAssumeFeaturesKHR::default(),
            shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT::default(),
            wg_explicit_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default(),
            robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT::default(),
            video_maintenance1: vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default(),
        })
    }

    /// Link core → v11 → v12 → v13 → the extension structs whose extension
    /// is in `exts`. The struct must not move afterwards (it's boxed).
    fn link(&mut self, exts: &[&CStr]) {
        let has = |e: &CStr| exts.contains(&e);
        let mut chain: Vec<*mut vk::BaseOutStructure<'static>> = vec![
            &mut self.v11 as *mut _ as _,
            &mut self.v12 as *mut _ as _,
            &mut self.v13 as *mut _ as _,
        ];
        if has(vk::KHR_COOPERATIVE_MATRIX_NAME) { chain.push(&mut self.coop_matrix as *mut _ as _); }
        if has(vk::EXT_SHADER_ATOMIC_FLOAT_NAME) { chain.push(&mut self.atomic_float as *mut _ as _); }
        if has(vk::KHR_SHADER_SUBGROUP_ROTATE_NAME) { chain.push(&mut self.subgroup_rotate as *mut _ as _); }
        if has(vk::KHR_SHADER_EXPECT_ASSUME_NAME) { chain.push(&mut self.expect_assume as *mut _ as _); }
        if has(vk::EXT_SHADER_OBJECT_NAME) { chain.push(&mut self.shader_object as *mut _ as _); }
        if has(vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME) { chain.push(&mut self.wg_explicit_layout as *mut _ as _); }
        if has(vk::EXT_ROBUSTNESS2_NAME) { chain.push(&mut self.robustness2 as *mut _ as _); }
        if has(vk::KHR_VIDEO_MAINTENANCE1_NAME) { chain.push(&mut self.video_maintenance1 as *mut _ as _); }
        self.core.p_next = chain[0] as *mut c_void;
        for w in chain.windows(2) {
            unsafe { (*w[0]).p_next = w[1] };
        }
        unsafe { (**chain.last().unwrap()).p_next = std::ptr::null_mut() };
    }

    /// OR in a Vulkan feature struct by its sType: the aggregates and the
    /// extension structs above directly, and the individual promoted structs
    /// wgpu-hal chains by translating them onto the aggregates.
    unsafe fn absorb(&mut self, p: *const vk::BaseOutStructure<'static>) -> Result<()> {
        use vk::StructureType as S;
        let s = (*p).s_type;
        match s {
            S::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => or_bools(&mut self.v11, p as _),
            S::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => or_bools(&mut self.v12, p as _),
            S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => or_bools(&mut self.v13, p as _),
            S::PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR => or_bools(&mut self.coop_matrix, p as _),
            S::PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT => or_bools(&mut self.atomic_float, p as _),
            S::PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR => or_bools(&mut self.subgroup_rotate, p as _),
            S::PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES_KHR => or_bools(&mut self.expect_assume, p as _),
            S::PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT => or_bools(&mut self.shader_object, p as _),
            S::PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR => or_bools(&mut self.wg_explicit_layout, p as _),
            S::PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT => or_bools(&mut self.robustness2, p as _),
            S::PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR => or_bools(&mut self.video_maintenance1, p as _),
            // wgpu-hal's individual (promoted) structs → aggregates.
            S::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceTimelineSemaphoreFeatures);
                self.v12.timeline_semaphore |= f.timeline_semaphore;
            }
            S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceImageRobustnessFeatures);
                self.v13.robust_image_access |= f.robust_image_access;
            }
            S::PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceVulkanMemoryModelFeatures);
                self.v12.vulkan_memory_model |= f.vulkan_memory_model;
                self.v12.vulkan_memory_model_device_scope |= f.vulkan_memory_model_device_scope;
                self.v12.vulkan_memory_model_availability_visibility_chains |=
                    f.vulkan_memory_model_availability_visibility_chains;
            }
            S::PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES => {
                let f = &*(p as *const vk::PhysicalDevice8BitStorageFeatures);
                self.v12.storage_buffer8_bit_access |= f.storage_buffer8_bit_access;
                self.v12.uniform_and_storage_buffer8_bit_access |= f.uniform_and_storage_buffer8_bit_access;
                self.v12.storage_push_constant8 |= f.storage_push_constant8;
            }
            S::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceScalarBlockLayoutFeatures);
                self.v12.scalar_block_layout |= f.scalar_block_layout;
            }
            S::PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceUniformBufferStandardLayoutFeatures);
                self.v12.uniform_buffer_standard_layout |= f.uniform_buffer_standard_layout;
            }
            S::PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures);
                self.v12.shader_subgroup_extended_types |= f.shader_subgroup_extended_types;
            }
            S::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceHostQueryResetFeatures);
                self.v12.host_query_reset |= f.host_query_reset;
            }
            S::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceImagelessFramebufferFeatures);
                self.v12.imageless_framebuffer |= f.imageless_framebuffer;
            }
            S::PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceSeparateDepthStencilLayoutsFeatures);
                self.v12.separate_depth_stencil_layouts |= f.separate_depth_stencil_layouts;
            }
            S::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceSynchronization2Features);
                self.v13.synchronization2 |= f.synchronization2;
            }
            S::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceDynamicRenderingFeatures);
                self.v13.dynamic_rendering |= f.dynamic_rendering;
            }
            S::PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderIntegerDotProductFeatures);
                self.v13.shader_integer_dot_product |= f.shader_integer_dot_product;
            }
            S::PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures);
                self.v13.shader_demote_to_helper_invocation |= f.shader_demote_to_helper_invocation;
            }
            S::PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderTerminateInvocationFeatures);
                self.v13.shader_terminate_invocation |= f.shader_terminate_invocation;
            }
            S::PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES => {
                let f = &*(p as *const vk::PhysicalDevicePipelineCreationCacheControlFeatures);
                self.v13.pipeline_creation_cache_control |= f.pipeline_creation_cache_control;
            }
            S::PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES => {
                let f = &*(p as *const vk::PhysicalDevicePrivateDataFeatures);
                self.v13.private_data |= f.private_data;
            }
            S::PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceInlineUniformBlockFeatures);
                self.v13.inline_uniform_block |= f.inline_uniform_block;
                self.v13.descriptor_binding_inline_uniform_block_update_after_bind |=
                    f.descriptor_binding_inline_uniform_block_update_after_bind;
            }
            S::PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceVariablePointersFeatures);
                self.v11.variable_pointers_storage_buffer |= f.variable_pointers_storage_buffer;
                self.v11.variable_pointers |= f.variable_pointers;
            }
            S::PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceProtectedMemoryFeatures);
                self.v11.protected_memory |= f.protected_memory;
            }
            S::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderDrawParametersFeatures);
                self.v11.shader_draw_parameters |= f.shader_draw_parameters;
            }
            S::PHYSICAL_DEVICE_MULTIVIEW_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceMultiviewFeatures);
                self.v11.multiview |= f.multiview;
                self.v11.multiview_geometry_shader |= f.multiview_geometry_shader;
                self.v11.multiview_tessellation_shader |= f.multiview_tessellation_shader;
            }
            S::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceSamplerYcbcrConversionFeatures);
                self.v11.sampler_ycbcr_conversion |= f.sampler_ycbcr_conversion;
            }
            S::PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceTextureCompressionASTCHDRFeatures);
                self.v13.texture_compression_astc_hdr |= f.texture_compression_astc_hdr;
            }
            S::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderFloat16Int8Features);
                self.v12.shader_float16 |= f.shader_float16;
                self.v12.shader_int8 |= f.shader_int8;
            }
            S::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES => {
                let f = &*(p as *const vk::PhysicalDevice16BitStorageFeatures);
                self.v11.storage_buffer16_bit_access |= f.storage_buffer16_bit_access;
                self.v11.uniform_and_storage_buffer16_bit_access |= f.uniform_and_storage_buffer16_bit_access;
                self.v11.storage_push_constant16 |= f.storage_push_constant16;
                self.v11.storage_input_output16 |= f.storage_input_output16;
            }
            S::PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures);
                self.v13.shader_zero_initialize_workgroup_memory |= f.shader_zero_initialize_workgroup_memory;
            }
            S::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceBufferDeviceAddressFeatures);
                self.v12.buffer_device_address |= f.buffer_device_address;
                self.v12.buffer_device_address_capture_replay |= f.buffer_device_address_capture_replay;
                self.v12.buffer_device_address_multi_device |= f.buffer_device_address_multi_device;
            }
            S::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceShaderAtomicInt64Features);
                self.v12.shader_buffer_int64_atomics |= f.shader_buffer_int64_atomics;
                self.v12.shader_shared_int64_atomics |= f.shader_shared_int64_atomics;
            }
            S::PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceSubgroupSizeControlFeatures);
                self.v13.subgroup_size_control |= f.subgroup_size_control;
                self.v13.compute_full_subgroups |= f.compute_full_subgroups;
            }
            S::PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceMaintenance4Features);
                self.v13.maintenance4 |= f.maintenance4;
            }
            S::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES => {
                let f = &*(p as *const vk::PhysicalDeviceDescriptorIndexingFeatures);
                let v = &mut self.v12;
                v.descriptor_indexing = vk::TRUE;
                v.shader_input_attachment_array_dynamic_indexing |= f.shader_input_attachment_array_dynamic_indexing;
                v.shader_uniform_texel_buffer_array_dynamic_indexing |= f.shader_uniform_texel_buffer_array_dynamic_indexing;
                v.shader_storage_texel_buffer_array_dynamic_indexing |= f.shader_storage_texel_buffer_array_dynamic_indexing;
                v.shader_uniform_buffer_array_non_uniform_indexing |= f.shader_uniform_buffer_array_non_uniform_indexing;
                v.shader_sampled_image_array_non_uniform_indexing |= f.shader_sampled_image_array_non_uniform_indexing;
                v.shader_storage_buffer_array_non_uniform_indexing |= f.shader_storage_buffer_array_non_uniform_indexing;
                v.shader_storage_image_array_non_uniform_indexing |= f.shader_storage_image_array_non_uniform_indexing;
                v.shader_input_attachment_array_non_uniform_indexing |= f.shader_input_attachment_array_non_uniform_indexing;
                v.shader_uniform_texel_buffer_array_non_uniform_indexing |= f.shader_uniform_texel_buffer_array_non_uniform_indexing;
                v.shader_storage_texel_buffer_array_non_uniform_indexing |= f.shader_storage_texel_buffer_array_non_uniform_indexing;
                v.descriptor_binding_uniform_buffer_update_after_bind |= f.descriptor_binding_uniform_buffer_update_after_bind;
                v.descriptor_binding_sampled_image_update_after_bind |= f.descriptor_binding_sampled_image_update_after_bind;
                v.descriptor_binding_storage_image_update_after_bind |= f.descriptor_binding_storage_image_update_after_bind;
                v.descriptor_binding_storage_buffer_update_after_bind |= f.descriptor_binding_storage_buffer_update_after_bind;
                v.descriptor_binding_uniform_texel_buffer_update_after_bind |= f.descriptor_binding_uniform_texel_buffer_update_after_bind;
                v.descriptor_binding_storage_texel_buffer_update_after_bind |= f.descriptor_binding_storage_texel_buffer_update_after_bind;
                v.descriptor_binding_update_unused_while_pending |= f.descriptor_binding_update_unused_while_pending;
                v.descriptor_binding_partially_bound |= f.descriptor_binding_partially_bound;
                v.descriptor_binding_variable_descriptor_count |= f.descriptor_binding_variable_descriptor_count;
                v.runtime_descriptor_array |= f.runtime_descriptor_array;
            }
            other => {
                return Err(Error::Wgpu(format!(
                    "shared Vulkan device: no mapping for feature struct {other:?}")));
            }
        }
        Ok(())
    }
}

/// FFmpeg 8.1 `AVVkFrame` (hand mirror, like [`AvVulkanDeviceContext`]).
#[repr(C)]
struct AvVkFrame {
    img: [vk::Image; 8],
    tiling: vk::ImageTiling,
    mem: [vk::DeviceMemory; 8],
    size: [usize; 8],
    flags: vk::MemoryPropertyFlags,
    access: [vk::AccessFlags; 8],
    layout: [vk::ImageLayout; 8],
    sem: [vk::Semaphore; 8],
    sem_value: [u64; 8],
    internal: *mut c_void,
    offset: [isize; 8],
    queue_family: [u32; 8],
}

/// FFmpeg 8.1 `AVVulkanFramesContext` (hand mirror).
#[repr(C)]
struct AvVulkanFramesContext {
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    create_pnext: *mut c_void,
    alloc_pnext: [*mut c_void; 8],
    flags: u32,
    img_flags: vk::ImageCreateFlags,
    format: [vk::Format; 8],
    nb_layers: c_int,
    lock_frame: Option<unsafe extern "C" fn(*mut ffmpeg::ffi::AVHWFramesContext, *mut AvVkFrame)>,
    unlock_frame: Option<unsafe extern "C" fn(*mut ffmpeg::ffi::AVHWFramesContext, *mut AvVkFrame)>,
}

/// FFmpeg `get_format` callback: take `AV_PIX_FMT_VULKAN` when offered, with
/// a decode frame pool whose images are valid for the stream's video profile
/// ([`attach_valid_frames_ctx`]).
unsafe extern "C" fn vulkan_get_format(
    ctx: *mut ffmpeg::ffi::AVCodecContext,
    fmts: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    use ffmpeg::ffi::AVPixelFormat::*;
    let mut p = fmts;
    while *p != AV_PIX_FMT_NONE {
        if *p == AV_PIX_FMT_VULKAN {
            if let Err(e) = attach_valid_frames_ctx(ctx) {
                // FFmpeg's own pool still decodes (drivers tolerate it).
                tracing::warn!("vulkan decode: keeping FFmpeg's frame pool ({e})");
            }
            return AV_PIX_FMT_VULKAN;
        }
        p = p.add(1);
    }
    *fmts
}

/// Give the decoder a frame pool of images the driver accepts for this video
/// profile. FFmpeg's own pool asks for STORAGE usage (decode suggests it, and
/// `vulkan_frames_init` ORs in every usage the format supports) and, with no
/// create flags set, ALIAS | MUTABLE_FORMAT | EXTENDED_USAGE — but Vulkan
/// Video only allows what `vkGetPhysicalDeviceVideoFormatPropertiesKHR`
/// reports for the profile (NVIDIA 595-series: no STORAGE, flags only
/// MUTABLE_FORMAT | EXTENDED_USAGE). The images are invalid then
/// (`VUID-VkImageCreateInfo-pNext-06811`, and for H.264 also -07240 / -07142
/// on use); NVIDIA decodes them anyway, but under the Khronos validation
/// layer its checks of those images crash the driver (`nvoglv64.dll`), and
/// another driver may simply refuse them. So: start from FFmpeg's suggested
/// parameters (profile list, format, DPB usage), keep only what decoding and
/// our plane copies need within what the driver reports, and set the create
/// flags explicitly so FFmpeg doesn't add its defaults.
unsafe fn attach_valid_frames_ctx(ctx: *mut ffmpeg::ffi::AVCodecContext) -> Result<()> {
    use ffmpeg::ffi::*;
    let mut frames: *mut AVBufferRef = std::ptr::null_mut();
    let r = avcodec_get_hw_frames_parameters(ctx, (*ctx).hw_device_ctx, AVPixelFormat::AV_PIX_FMT_VULKAN, &mut frames);
    if r < 0 || frames.is_null() {
        return Err(Error::Ffmpeg(format!("avcodec_get_hw_frames_parameters: {r}")));
    }
    let fail = |frames: &mut *mut AVBufferRef, msg: String| {
        av_buffer_unref(frames);
        Err(Error::Ffmpeg(msg))
    };
    let fc = (*frames).data as *mut AVHWFramesContext;
    let vkfc = &mut *((*fc).hwctx as *mut AvVulkanFramesContext);
    let video = vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
    // What decoding needs (FFmpeg's DST, plus DPB unless the driver uses a
    // separate DPB) and what we need: TRANSFER_SRC for the plane copies.
    let wanted = (vkfc.usage & video) | vk::ImageUsageFlags::TRANSFER_SRC;
    let Some((allowed_usage, allowed_flags)) = video_format_support(ctx, vkfc, vkfc.usage & video) else {
        return fail(&mut frames, "the driver reports no decode format for this profile".into());
    };
    if !allowed_usage.contains(wanted) {
        return fail(&mut frames, format!("decode images can't have {wanted:?} (driver allows {allowed_usage:?})"));
    }
    let flags = allowed_flags & (vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE);
    // Without STORAGE in `usage`, `vulkan_frames_init` adds only
    // TRANSFER_DST / TRANSFER_SRC / SAMPLED from the format's features; a
    // non-zero `img_flags` keeps its default flags (incl. ALIAS) out.
    vkfc.usage = wanted;
    vkfc.img_flags = flags;
    let r = av_hwframe_ctx_init(frames);
    if r < 0 {
        return fail(&mut frames, format!("av_hwframe_ctx_init: {r}"));
    }
    // Pool images are created from these fields at allocation time: drop
    // whatever init added beyond what the driver allows for the profile.
    vkfc.usage &= allowed_usage;
    vkfc.img_flags = flags;
    av_buffer_unref(&mut (*ctx).hw_frames_ctx);
    (*ctx).hw_frames_ctx = frames;
    Ok(())
}

/// The image usage and create flags `vkGetPhysicalDeviceVideoFormatPropertiesKHR`
/// allows for the decode output format of `vkfc`'s video profile (its
/// `create_pnext` profile list), queried with `video_usage`.
unsafe fn video_format_support(
    ctx: *mut ffmpeg::ffi::AVCodecContext,
    vkfc: &AvVulkanFramesContext,
    video_usage: vk::ImageUsageFlags,
) -> Option<(vk::ImageUsageFlags, vk::ImageCreateFlags)> {
    let dev = (*(*ctx).hw_device_ctx).data as *const ffmpeg::ffi::AVHWDeviceContext;
    let hw = &*((*dev).hwctx as *const AvVulkanDeviceContext);
    let gipa = hw.get_proc_addr?;
    let get: vk::PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR = std::mem::transmute(
        gipa(hw.inst, c"vkGetPhysicalDeviceVideoFormatPropertiesKHR".as_ptr())?);
    // The profile list FFmpeg chains into image creation; copy it so the
    // query sees nothing else from that chain.
    let mut p = vkfc.create_pnext as *const vk::BaseInStructure<'static>;
    while !p.is_null() && (*p).s_type != vk::StructureType::VIDEO_PROFILE_LIST_INFO_KHR {
        p = (*p).p_next;
    }
    if p.is_null() {
        return None;
    }
    let src = &*(p as *const vk::VideoProfileListInfoKHR<'static>);
    let mut list = vk::VideoProfileListInfoKHR::default();
    list.profile_count = src.profile_count;
    list.p_profiles = src.p_profiles;
    let mut info = vk::PhysicalDeviceVideoFormatInfoKHR::default().image_usage(video_usage);
    info.p_next = &mut list as *mut _ as *const c_void;
    let mut n = 0u32;
    if get(hw.phys_dev, &info, &mut n, std::ptr::null_mut()) != vk::Result::SUCCESS || n == 0 {
        return None;
    }
    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); n as usize];
    if get(hw.phys_dev, &info, &mut n, props.as_mut_ptr()) != vk::Result::SUCCESS {
        return None;
    }
    props.iter().take(n as usize)
        .find(|f| f.format == vkfc.format[0] && f.image_tiling == vkfc.tiling)
        .map(|f| (f.image_usage_flags, f.image_create_flags))
}

/// The staging buffers and wgpu plane textures decoded frames of one size and
/// depth are copied through (plane 0 = Y, plane 1 = interleaved UV). wgpu-hal
/// can't hand back a wrapped buffer's raw handle, so the staging buffers are
/// created here and wrapped. One set per (width, height, depth), kept for the
/// life of the device; every frame of that shape reuses it (each frame's copy
/// and the work reading it are ONE submission, so reuse is queue-ordered).
struct PlaneSet {
    raw: [vk::Buffer; 2],
    staging: [wgpu::Buffer; 2],
    row_bytes: [u32; 2],
    /// Bytes per texel of each plane (P010: 2 / 4, NV12: 1 / 2).
    texel: [u32; 2],
    y: wgpu::Texture,
    uv: wgpu::Texture,
}

/// One decoded frame's planes as wgpu textures, valid inside
/// [`VkDecodeDevice::with_frame_planes`]: `R16Unorm` Y + `Rg16Unorm` UV with
/// the 10-bit samples in the high bits (P010 layout), or `R8Unorm` +
/// `Rg8Unorm` for an 8-bit (NV12) stream — the input of
/// `gpu::Device::record_p010_resolve`.
#[derive(Debug)]
pub struct FramePlanes<'a> {
    pub y: &'a wgpu::Texture,
    pub uv: &'a wgpu::Texture,
    pub w: u32,
    pub h: u32,
    /// 10 (P010) or 8 (NV12) — for `gpu::yuv_range_constants(bits, full)`.
    pub bits: u8,
}

/// A pooled `Rgba16Unorm` texture a decoded frame is resolved into.
#[derive(Debug)]
pub struct RgbaSlot {
    w: u32,
    h: u32,
    tex: wgpu::Texture,
    busy: std::sync::atomic::AtomicBool,
}

/// Hands its [`RgbaSlot`] back to the pool when dropped. Drop it once the GPU
/// work READING the texture has been submitted: a later resolve into the same
/// texture is queued behind that work on the same queue.
pub struct RgbaLease(Arc<RgbaSlot>);

impl RgbaLease {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.0.tex
    }
}

impl Drop for RgbaLease {
    fn drop(&mut self) {
        self.0.busy.store(false, std::sync::atomic::Ordering::Release);
    }
}

impl std::fmt::Debug for RgbaLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RgbaLease").field(&(self.0.w, self.0.h)).finish()
    }
}

/// Cap on pooled RGBA16 outputs per device; past it a frame gets a one-off
/// texture (allocated and freed — slower, never wrong).
const MAX_RGBA_SLOTS: usize = 32;

/// A Vulkan device shared by wgpu and FFmpeg's Vulkan Video decoder, plus
/// FFmpeg's `AVHWDeviceContext` on it. Two exist at most: the export device
/// ([`shared_export_device`]) and the app's own presenting device
/// ([`create_ui_device`]); [`decode_device_for`] finds the one behind a wgpu
/// device.
pub struct VkDecodeDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ffmpeg_device: *mut ffmpeg::ffi::AVBufferRef,
    raw: ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    sets: std::sync::Mutex<std::collections::HashMap<(u32, u32, u8), PlaneSet>>,
    rgba: std::sync::Mutex<Vec<Arc<RgbaSlot>>>,
}

// The AVBufferRef is only handed to FFmpeg (whose device context is
// thread-safe for decoder use; it locks its own queues); the plane sets and
// the RGBA pool are behind mutexes.
unsafe impl Send for VkDecodeDevice {}
unsafe impl Sync for VkDecodeDevice {}

impl std::fmt::Debug for VkDecodeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkDecodeDevice").finish_non_exhaustive()
    }
}

impl VkDecodeDevice {
    /// FFmpeg's device reference (borrowed — `av_buffer_ref` it to keep).
    pub fn ffmpeg_device(&self) -> *mut ffmpeg::ffi::AVBufferRef {
        self.ffmpeg_device
    }

    /// Make `ctx` decode through Vulkan Video on this device. Frames come out
    /// as `AV_PIX_FMT_VULKAN` when the GPU can decode the stream; otherwise
    /// FFmpeg falls back to software frames, which [`Self::with_frame_planes`]
    /// rejects — so a caller probes the first frame with
    /// [`is_vulkan_frame`] before committing to this path.
    pub fn attach_decoder(&self, ctx: &mut ffmpeg::codec::context::Context) -> Result<()> {
        unsafe {
            let raw = ctx.as_mut_ptr();
            let dev = ffmpeg::ffi::av_buffer_ref(self.ffmpeg_device);
            if dev.is_null() {
                return Err(Error::Ffmpeg("vulkan decode: av_buffer_ref(device) failed".into()));
            }
            (*raw).hw_device_ctx = dev;
            (*raw).get_format = Some(vulkan_get_format);
        }
        Ok(())
    }

    /// Drop the pooled RGBA16 textures no lease holds (an export's native-size
    /// outputs: ~120 MB each at 3840²) — see [`ExportDevice`].
    pub fn release_idle_rgba16(&self) {
        use std::sync::atomic::Ordering;
        self.rgba.lock().unwrap().retain(|slot| slot.busy.load(Ordering::Acquire));
    }

    /// A free pooled `Rgba16Unorm` texture of `w`×`h` (marked busy), created
    /// while under [`MAX_RGBA_SLOTS`]; past the cap a one-off texture.
    pub fn take_rgba16(&self, w: u32, h: u32) -> RgbaLease {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pool = self.rgba.lock().unwrap();
        for slot in pool.iter() {
            if slot.w == w && slot.h == h
                && slot.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
            {
                return RgbaLease(slot.clone());
            }
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vk_decode_rgba16"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            // COPY_SRC: consumers split / read back from it.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let slot = Arc::new(RgbaSlot { w, h, tex, busy: AtomicBool::new(true) });
        if pool.len() < MAX_RGBA_SLOTS {
            pool.push(slot.clone());
        }
        RgbaLease(slot)
    }

    /// Copy a decoded Vulkan frame's planes into wgpu textures and let
    /// `record` add wgpu work reading them (a resolve) to the SAME submission.
    /// Never waits on the GPU, so it is safe on the app's presenting device
    /// (a blocking `poll` there would hold wgpu's fence lock and stall the
    /// UI thread's submits).
    ///
    /// FFmpeg's frame protocol, done by hand because wgpu must never track an
    /// image FFmpeg also transitions: lock the frame and wait (CPU) for the
    /// decode's timeline value; in wgpu's queue, barrier the image from
    /// FFmpeg's recorded layout to TRANSFER_SRC, copy both planes into the
    /// staging buffers and from there into wgpu's textures, then run
    /// `record`'s work. Whoever uses a frame signals its semaphore at
    /// `sem_value + 1` when done: that signal rides the NEXT submission on
    /// wgpu's queue (an empty one if no other thread submits first) — a
    /// semaphore signal covers every earlier submission on the queue, so it
    /// fires only after our copy. FFmpeg waits for it before it writes the
    /// image again (as a new output or a reference). The image's new
    /// layout/access are recorded and the frame unlocked; the caller may drop
    /// the frame immediately. FFmpeg's images are CONCURRENT across its queue
    /// families, which include wgpu's family 0, so no ownership transfer.
    pub fn with_frame_planes<R>(
        &self,
        frame: &ffmpeg::frame::Video,
        record: impl FnOnce(&mut wgpu::CommandEncoder, &FramePlanes) -> Result<R>,
    ) -> Result<R> {
        use wgpu::hal::api::Vulkan;
        let (w, h) = (frame.width(), frame.height());
        unsafe {
            let av = frame.as_ptr();
            if !is_vulkan_frame(frame) {
                return Err(Error::Ffmpeg("vulkan decode: frame is not a Vulkan hw frame".into()));
            }
            let hwfc = (*(*av).hw_frames_ctx).data as *mut ffmpeg::ffi::AVHWFramesContext;
            let bits: u8 = match (*hwfc).sw_format {
                ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_P010LE => 10,
                ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NV12 => 8,
                other => return Err(Error::Ffmpeg(format!(
                    "vulkan decode: decoded layout {other:?} is not NV12/P010"))),
            };
            let vkfc = &*((*hwfc).hwctx as *const AvVulkanFramesContext);
            let vkf = (*av).data[0] as *mut AvVkFrame;
            if (*vkf).img[1] != vk::Image::null() {
                return Err(Error::Ffmpeg(
                    "vulkan decode: multi-image frames are not supported (expected one multiplane image)".into()));
            }
            let qf = (*vkf).queue_family[0];
            if qf != vk::QUEUE_FAMILY_IGNORED && qf != 0 {
                return Err(Error::Ffmpeg(format!(
                    "vulkan decode: frame owned by queue family {qf}; expected a CONCURRENT image")));
            }
            let lock = vkfc.lock_frame.ok_or_else(|| Error::Ffmpeg("vulkan decode: no lock_frame".into()))?;
            let unlock = vkfc.unlock_frame.ok_or_else(|| Error::Ffmpeg("vulkan decode: no unlock_frame".into()))?;

            // The plane set is shared by every frame of this shape: hold it for
            // the whole record + submit so each frame's copy is one unit.
            let mut sets = self.sets.lock().unwrap();
            let set = self.plane_set(&mut sets, w, h, bits)?;

            lock(hwfc, vkf);
            let result = (|| -> Result<R> {
                // The decode that wrote this frame signals sem at sem_value.
                let sem = (*vkf).sem[0];
                let value = (*vkf).sem_value[0];
                self.raw.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default().semaphores(&[sem]).values(&[value]),
                    10_000_000_000, // 10 s — a decode never takes that long
                ).map_err(|e| Error::Wgpu(format!("vulkan decode: wait for decode: {e}")))?;

                let img = (*vkf).img[0];
                let old_layout = (*vkf).layout[0];
                // wgpu forbids mixing raw and wgpu commands in one encoder:
                // the raw barrier + plane copies get their own command buffer,
                // ahead of the wgpu copies in the same submission.
                let mut raw_enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vk_decode_copy_raw"),
                });
                let recorded = raw_enc.as_hal_mut::<Vulkan, _, _>(|hal_enc| {
                    let Some(hal_enc) = hal_enc else { return false };
                    let cb = hal_enc.raw_handle();
                    let color = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1).layer_count(1);
                    // src ALL_COMMANDS also orders this frame's staging writes
                    // after every earlier submission's reads of the same set.
                    let to_src = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(img)
                        .subresource_range(color);
                    self.raw.cmd_pipeline_barrier(
                        cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &[], &[to_src]);
                    for (plane, (aspect, (pw, ph))) in [
                        (vk::ImageAspectFlags::PLANE_0, (w, h)),
                        (vk::ImageAspectFlags::PLANE_1, (w / 2, h / 2)),
                    ].into_iter().enumerate() {
                        let region = vk::BufferImageCopy::default()
                            .buffer_row_length(set.row_bytes[plane] / set.texel[plane])
                            .image_subresource(vk::ImageSubresourceLayers::default()
                                .aspect_mask(aspect).layer_count(1))
                            .image_extent(vk::Extent3D { width: pw, height: ph, depth: 1 });
                        self.raw.cmd_copy_image_to_buffer(
                            cb, img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, set.raw[plane], &[region]);
                    }
                    let bufs: Vec<vk::BufferMemoryBarrier> = set.raw.iter().map(|&b| {
                        vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(b).size(vk::WHOLE_SIZE)
                    }).collect();
                    self.raw.cmd_pipeline_barrier(
                        cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &bufs, &[]);
                    true
                });
                if !recorded {
                    return Err(Error::Wgpu("vulkan decode: wgpu encoder is not Vulkan".into()));
                }
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vk_decode_copy"),
                });
                for (plane, (tex, (pw, ph))) in [(&set.y, (w, h)), (&set.uv, (w / 2, h / 2))].into_iter().enumerate() {
                    enc.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: &set.staging[plane],
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(set.row_bytes[plane]),
                                rows_per_image: Some(ph),
                            },
                        },
                        tex.as_image_copy(),
                        wgpu::Extent3d { width: pw, height: ph, depth_or_array_layers: 1 },
                    );
                }
                let r = record(&mut enc, &FramePlanes { y: &set.y, uv: &set.uv, w, h, bits })?;
                // Look the hal queue up before submitting, so a failure leaves
                // the frame untouched.
                let hal_queue = self.queue.as_hal::<Vulkan>()
                    .ok_or_else(|| Error::Wgpu("vulkan decode: wgpu queue is not Vulkan".into()))?;
                self.queue.submit([raw_enc.finish(), enc.finish()]);
                let next = value + 1;
                hal_queue.add_signal_semaphore(sem, Some(next));
                drop(hal_queue);
                self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
                // Our barrier left the image in TRANSFER_SRC; FFmpeg's next
                // use waits for `next` and starts from what we record here.
                (*vkf).sem_value[0] = next;
                (*vkf).layout[0] = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                (*vkf).access[0] = vk::AccessFlags::TRANSFER_READ;
                Ok(r)
            })();
            unlock(hwfc, vkf);
            result
        }
    }

    /// The plane set for `w`×`h` at `bits`: created on first use, kept for
    /// the life of the device.
    unsafe fn plane_set<'a>(
        &self,
        sets: &'a mut std::collections::HashMap<(u32, u32, u8), PlaneSet>,
        w: u32, h: u32, bits: u8,
    ) -> Result<&'a PlaneSet> {
        use wgpu::hal::api::Vulkan;
        if let std::collections::hash_map::Entry::Vacant(e) = sets.entry((w, h, bits)) {
            let (texel, formats) = if bits == 8 {
                ([1u32, 2], [wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm])
            } else {
                ([2u32, 4], [wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm])
            };
            // 256-byte rows: wgpu's buffer→texture copy alignment.
            let row = [(w * texel[0] + 255) & !255, (w / 2 * texel[1] + 255) & !255];
            let size = [row[0] as u64 * h as u64, row[1] as u64 * (h / 2) as u64];
            let mut raw = [vk::Buffer::null(); 2];
            let mut staging = Vec::with_capacity(2);
            for i in 0..2 {
                let b = self.raw.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size[i])
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                ).map_err(|e| Error::Wgpu(format!("vulkan decode: create staging buffer: {e}")))?;
                let req = self.raw.get_buffer_memory_requirements(b);
                let ty = (0..self.mem_props.memory_type_count)
                    .find(|&t| req.memory_type_bits & (1 << t) != 0
                        && self.mem_props.memory_types[t as usize].property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
                    .ok_or_else(|| Error::Wgpu("vulkan decode: no device-local memory type".into()))?;
                let mem = self.raw.allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None,
                ).map_err(|e| Error::Wgpu(format!("vulkan decode: allocate staging memory: {e}")))?;
                self.raw.bind_buffer_memory(b, mem, 0)
                    .map_err(|e| Error::Wgpu(format!("vulkan decode: bind staging memory: {e}")))?;
                raw[i] = b;
                staging.push(self.device.create_buffer_from_hal::<Vulkan>(
                    wgpu::hal::vulkan::Buffer::from_raw(b),
                    &wgpu::BufferDescriptor {
                        label: Some("vk_decode_staging"),
                        size: size[i],
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                ));
            }
            let mk = |label: &str, tw: u32, th: u32, format: wgpu::TextureFormat| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            };
            let uv_buf = staging.pop().unwrap();
            let y_buf = staging.pop().unwrap();
            e.insert(PlaneSet {
                raw, staging: [y_buf, uv_buf], row_bytes: row, texel,
                y: mk("vk_decode_y", w, h, formats[0]),
                uv: mk("vk_decode_uv", w / 2, h / 2, formats[1]),
            });
        }
        Ok(&sets[&(w, h, bits)])
    }
}

/// True if `frame` is a Vulkan hardware frame (FFmpeg decoded it through
/// Vulkan Video) — the probe a decode path runs on its first frame, since a
/// stream the GPU can't decode silently comes back as software frames.
pub fn is_vulkan_frame(frame: &ffmpeg::frame::Video) -> bool {
    unsafe {
        let av = frame.as_ptr();
        (*av).format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_VULKAN as i32
            && !(*av).hw_frames_ctx.is_null()
    }
}

static DECODE_DEVICES: std::sync::Mutex<Vec<Arc<VkDecodeDevice>>> = std::sync::Mutex::new(Vec::new());

/// The Vulkan-decode side of `device`, if it was created here (the export
/// device or the app's presenting device), else `None` — a caller then
/// decodes through D3D11.
pub fn decode_device_for(device: &wgpu::Device) -> Option<Arc<VkDecodeDevice>> {
    if std::env::var_os("VR180_D3D11_DECODE").is_some() {
        return None; // A/B and fallback testing: force the D3D11 decode path
    }
    DECODE_DEVICES.lock().unwrap().iter().find(|d| d.device == *device).cloned()
}

/// Any Vulkan-decode device (the app's, else the export's) — for decoders
/// whose frames go to the CPU (`decode::try_enable_d3d11va_decode_n`): they
/// only need FFmpeg's Vulkan context, not a particular wgpu device. `None`
/// under `VR180_D3D11_DECODE`, like [`decode_device_for`].
pub fn any_decode_device() -> Option<Arc<VkDecodeDevice>> {
    if std::env::var_os("VR180_D3D11_DECODE").is_some() {
        return None;
    }
    DECODE_DEVICES.lock().unwrap().first().cloned()
}

/// One export's Vulkan-decode device: wgpu pipeline + FFmpeg decoder on one
/// `VkDevice`. The `VkDevice` and FFmpeg's context are created once and kept
/// for the process; the pipeline (`device`) is new for each export and the
/// decoder's idle output textures are dropped with this, so an export leaves
/// no VRAM behind — the D3D11 path gets the same from a dedicated device per
/// export. Keep it until the export has finished.
pub struct ExportDevice {
    /// The export's pipeline device (wgpu on queue family 0, queue 1).
    pub device: Arc<crate::gpu::Device>,
    pub decode: Arc<VkDecodeDevice>,
}

impl Drop for ExportDevice {
    fn drop(&mut self) {
        self.decode.release_idle_rgba16();
    }
}

impl std::fmt::Debug for ExportDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportDevice").finish_non_exhaustive()
    }
}

/// The export `VkDevice`, as wgpu + FFmpeg wrap it.
struct ExportShared {
    instance: wgpu::Instance,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    decode: Arc<VkDecodeDevice>,
}

static EXPORT: OnceLock<std::result::Result<ExportShared, String>> = OnceLock::new();

/// An [`ExportDevice`] on `adapter`: its `VkDevice` is created on first use
/// and kept (FFmpeg's Vulkan device creation is slow under load — see
/// `encode.rs`), its pipeline is new. Exports that run on it decode through
/// Vulkan Video.
pub fn shared_export_device(adapter: &Arc<wgpu::Adapter>) -> Result<Arc<ExportDevice>> {
    let shared = EXPORT
        .get_or_init(|| {
            let r = unsafe { create(adapter, "vr180 export (shared with ffmpeg)") }.map(|(device, queue, decode)| {
                let decode = Arc::new(decode);
                DECODE_DEVICES.lock().unwrap().push(decode.clone());
                ExportShared {
                    instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
                    device: Arc::new(device),
                    queue: Arc::new(queue),
                    decode,
                }
            });
            match &r {
                Ok(_) => tracing::info!("Vulkan-decode export device ready (wgpu + FFmpeg Vulkan Video on one VkDevice)"),
                Err(e) => tracing::warn!("Vulkan-decode export device unavailable: {e}"),
            }
            r.map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| Error::Wgpu(e.clone()))?;
    let device = Arc::new(crate::gpu::Device::from_existing(
        shared.instance.clone(), adapter.clone(), shared.device.clone(), shared.queue.clone(),
    )?);
    Ok(Arc::new(ExportDevice { device, decode: shared.decode.clone() }))
}

/// The app's presenting device with Vulkan Video decode, for eframe
/// (`egui_wgpu::WgpuSetup::Existing`): the live preview then decodes through
/// Vulkan Video on the device that draws it. The decode side is registered
/// for [`decode_device_for`]. wgpu runs on queue family 0, queue 1; FFmpeg's
/// queues are separate, so presenting never shares a `VkQueue` with it.
pub fn create_ui_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let (device, queue, decode) = unsafe { create(adapter, "vr180-gui device (shared with ffmpeg)") }?;
    DECODE_DEVICES.lock().unwrap().push(Arc::new(decode));
    tracing::info!("Vulkan-decode UI device ready (the preview decodes through Vulkan Video)");
    Ok((device, queue))
}

/// FFmpeg's own device recipe: (enabled device extensions, feature chain
/// absorbed into `feats`). Read off a throwaway FFmpeg-created device.
unsafe fn ffmpeg_recipe(feats: &mut Features) -> Result<Vec<CString>> {
    use ffmpeg::ffi::*;
    let mut dev: *mut AVBufferRef = std::ptr::null_mut();
    let ret = av_hwdevice_ctx_create(&mut dev, AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        std::ptr::null(), std::ptr::null_mut(), 0);
    if ret < 0 || dev.is_null() {
        return Err(Error::Ffmpeg(format!("recipe: av_hwdevice_ctx_create VULKAN: {ret}")));
    }
    let hw = &*((*((*dev).data as *mut AVHWDeviceContext)).hwctx as *const AvVulkanDeviceContext);
    let exts = (0..hw.nb_enabled_dev_extensions as usize)
        .map(|i| CStr::from_ptr(*hw.enabled_dev_extensions.add(i)).to_owned())
        .collect();
    or_core(&mut feats.core.features, &hw.device_features.features);
    let mut p = hw.device_features.p_next as *const vk::BaseOutStructure<'static>;
    while !p.is_null() {
        // FFmpeg also chains structs ash can't name; those features stay off
        // (their extensions are dropped below).
        let _ = feats.absorb(p);
        p = (*p).p_next;
    }
    av_buffer_unref(&mut dev);
    Ok(exts)
}

unsafe fn create(adapter: &wgpu::Adapter, label: &str) -> Result<(wgpu::Device, wgpu::Queue, VkDecodeDevice)> {
    use wgpu::hal::api::Vulkan;
    let wanted = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        | wgpu::Features::TEXTURE_FORMAT_P010
        | wgpu::Features::TEXTURE_FORMAT_NV12;
    let features = adapter.features() & wanted;
    let limits = wgpu::Limits {
        max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d.clamp(8192, 16384),
        ..wgpu::Limits::default()
    };

    let mut feats = Features::new();
    let ffmpeg_exts = ffmpeg_recipe(&mut feats)?;

    let hal = adapter.as_hal::<Vulkan>()
        .ok_or_else(|| Error::Wgpu("shared Vulkan device: adapter is not Vulkan".into()))?;
    let instance = hal.shared_instance().raw_instance().clone();
    let entry = hal.shared_instance().entry().clone();
    let phys = hal.raw_physical_device();

    // Extensions: wgpu's ∪ the FFmpeg ones we can carry.
    let mut exts: Vec<&'static CStr> = hal.required_device_extensions(features);
    for e in &ffmpeg_exts {
        let known = EXT_FEATURES.iter().chain(EXT_PLAIN).find(|k| **k == e.as_c_str());
        if let Some(k) = known {
            if !exts.contains(k) { exts.push(k); }
        }
    }

    // wgpu-hal's feature needs, translated onto the aggregates.
    let mut wgpu_phd = hal.physical_device_features(&exts, features);
    {
        let info = wgpu_phd.add_to_device_create(vk::DeviceCreateInfo::default());
        if !info.p_enabled_features.is_null() {
            or_core(&mut feats.core.features, &*info.p_enabled_features);
        }
        let mut p = info.p_next as *const vk::BaseOutStructure<'static>;
        while !p.is_null() {
            feats.absorb(p)?;
            p = (*p).p_next;
        }
    }
    // Vulkan 1.2 aggregate implies the extension-level flag for these.
    feats.link(&exts);

    // Never enable what the GPU lacks: AND with what it supports.
    let mut supported = Features::new();
    supported.link(&exts);
    instance.get_physical_device_features2(phys, &mut supported.core);
    and_with(&mut feats, &supported);
    feats.link(&exts);

    // Queues: wgpu = family 0 queue 0; FFmpeg = the compute + transfer
    // families (all their queues). Distinct VkQueues → no submit races.
    let qfp = instance.get_physical_device_queue_family_properties(phys);
    let fam = |want: vk::QueueFlags, not: vk::QueueFlags| {
        qfp.iter().position(|q| q.queue_flags.contains(want) && !q.queue_flags.intersects(not))
    };
    let compute = fam(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS)
        .ok_or_else(|| Error::Wgpu("shared Vulkan device: no compute-only queue family".into()))?;
    let transfer = fam(vk::QueueFlags::TRANSFER, vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
    if !qfp[0].queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
        return Err(Error::Wgpu("shared Vulkan device: family 0 is not graphics+compute".into()));
    }
    let n_comp = qfp[compute].queue_count;
    let n_tx = transfer.map(|t| qfp[t].queue_count).unwrap_or(0);
    // Video decode family + the codec operations it supports.
    let decode = qfp.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR));
    let decode_caps = decode.map(|d| {
        let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); qfp.len()];
        let mut props: Vec<vk::QueueFamilyProperties2> = video.iter_mut()
            .map(|v| vk::QueueFamilyProperties2::default().push_next(v)).collect();
        instance.get_physical_device_queue_family_properties2(phys, &mut props);
        drop(props);
        video[d].video_codec_operations
    }).unwrap_or_default();
    let n_dec = decode.map(|d| qfp[d].queue_count).unwrap_or(0);
    if qfp[0].queue_count < 2 {
        return Err(Error::Wgpu("shared Vulkan device: family 0 has fewer than 2 queues".into()));
    }
    let prio: &'static [f32] = Box::leak(vec![1.0f32; 16].into_boxed_slice());
    // Family 0: queue 0 for FFmpeg (listed so its images are shareable with
    // family 0), queue 1 for wgpu — never the same VkQueue.
    let mut qinfos = vec![
        vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&prio[..2]),
        vk::DeviceQueueCreateInfo::default().queue_family_index(compute as u32)
            .queue_priorities(&prio[..n_comp.min(16) as usize]),
    ];
    if let Some(t) = transfer {
        qinfos.push(vk::DeviceQueueCreateInfo::default().queue_family_index(t as u32)
            .queue_priorities(&prio[..n_tx.min(16) as usize]));
    }
    if let Some(d) = decode {
        qinfos.push(vk::DeviceQueueCreateInfo::default().queue_family_index(d as u32)
            .queue_priorities(&prio[..n_dec.min(16) as usize]));
    }

    let ext_ptrs: Vec<*const c_char> = exts.iter().map(|e| e.as_ptr()).collect();
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qinfos)
        .enabled_extension_names(&ext_ptrs);
    info.p_next = &feats.core as *const _ as *const c_void; // features via Features2
    let raw_device = instance.create_device(phys, &info, None)
        .map_err(|e| Error::Wgpu(format!("shared Vulkan device: vkCreateDevice: {e}")))?;
    let device_handle = raw_device.handle();
    let raw_keep = raw_device.clone();
    let mem_props = instance.get_physical_device_memory_properties(phys);

    // wgpu: wrap (wgpu-hal owns + destroys the device; FFmpeg never does
    // for a user-supplied one, and the whole thing lives for the process).
    let open = hal.device_from_raw(
        raw_device, None, &exts, features, &limits, &wgpu::MemoryHints::default(), 0, 1,
    ).map_err(|e| Error::Wgpu(format!("shared Vulkan device: device_from_raw: {e}")))?;
    drop(hal);
    let (device, queue) = adapter.create_device_from_hal::<Vulkan>(open, &wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }).map_err(|e| Error::Wgpu(format!("shared Vulkan device: create_device_from_hal: {e}")))?;

    // FFmpeg: a user-supplied device context on the same VkDevice.
    let feats: &'static mut Features = Box::leak(feats);
    let ext_ptrs: &'static [*const c_char] = Box::leak(ext_ptrs.into_boxed_slice());
    let inst_ext_ptrs: &'static [*const c_char] = Box::leak(
        adapter.as_hal::<Vulkan>().unwrap().shared_instance().extensions()
            .iter().map(|e| e.as_ptr()).collect::<Vec<_>>().into_boxed_slice());
    use ffmpeg::ffi::*;
    let mut dev = av_hwdevice_ctx_alloc(AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN);
    if dev.is_null() {
        return Err(Error::Ffmpeg("shared Vulkan device: av_hwdevice_ctx_alloc failed".into()));
    }
    let hw = &mut *((*((*dev).data as *mut AVHWDeviceContext)).hwctx as *mut AvVulkanDeviceContext);
    hw.get_proc_addr = Some(entry.static_fn().get_instance_proc_addr);
    hw.inst = instance.handle();
    hw.phys_dev = phys;
    hw.act_dev = device_handle;
    hw.device_features = feats.core;
    hw.enabled_inst_extensions = inst_ext_ptrs.as_ptr();
    hw.nb_enabled_inst_extensions = inst_ext_ptrs.len() as c_int;
    hw.enabled_dev_extensions = ext_ptrs.as_ptr();
    hw.nb_enabled_dev_extensions = ext_ptrs.len() as c_int;
    hw.queue_family_index = 0;
    hw.nb_graphics_queues = 1;
    hw.queue_family_comp_index = compute as c_int;
    hw.nb_comp_queues = n_comp.min(16) as c_int;
    hw.queue_family_tx_index = transfer.map(|t| t as c_int).unwrap_or(compute as c_int);
    hw.nb_tx_queues = if transfer.is_some() { n_tx.min(16) as c_int } else { hw.nb_comp_queues };
    hw.queue_family_encode_index = -1;
    hw.nb_encode_queues = 0;
    hw.queue_family_decode_index = decode.map(|d| d as c_int).unwrap_or(-1);
    hw.nb_decode_queues = n_dec.min(16) as c_int;
    hw.qf[0] = AvQueueFamily {
        idx: compute as c_int, num: hw.nb_comp_queues,
        flags: vk::QueueFlags::COMPUTE.as_raw(), video_caps: 0,
    };
    hw.nb_qf = 1;
    let mut push_qf = |q: AvQueueFamily| {
        hw.qf[hw.nb_qf as usize] = q;
        hw.nb_qf += 1;
    };
    if let Some(t) = transfer {
        push_qf(AvQueueFamily {
            idx: t as c_int, num: n_tx.min(16) as c_int,
            flags: vk::QueueFlags::TRANSFER.as_raw(), video_caps: 0,
        });
    }
    if let Some(d) = decode {
        push_qf(AvQueueFamily {
            idx: d as c_int, num: n_dec.min(16) as c_int,
            flags: vk::QueueFlags::VIDEO_DECODE_KHR.as_raw(), video_caps: decode_caps.as_raw(),
        });
    }
    // Family 0, queue 0 only (wgpu owns queue 1): listing it makes FFmpeg's
    // frames shareable with wgpu's family without ownership transfers.
    push_qf(AvQueueFamily { idx: 0, num: 1, flags: vk::QueueFlags::GRAPHICS.as_raw(), video_caps: 0 });
    let ret = av_hwdevice_ctx_init(dev);
    if ret < 0 {
        av_buffer_unref(&mut dev);
        return Err(Error::Ffmpeg(format!("shared Vulkan device: av_hwdevice_ctx_init: {ret}")));
    }
    tracing::info!(
        "shared Vulkan device: {} device extensions; wgpu on family 0 queue 1; FFmpeg: compute {compute} ×{}, transfer {:?}, decode {:?} ({:?}), graphics 0 ×1; lock_queue set: {}",
        exts.len(), hw.nb_comp_queues, transfer, decode, decode_caps, !hw.lock_queue.is_null(),
    );
    let decode = VkDecodeDevice {
        device: device.clone(), queue: queue.clone(), ffmpeg_device: dev, raw: raw_keep, mem_props,
        sets: std::sync::Mutex::new(std::collections::HashMap::new()),
        rgba: std::sync::Mutex::new(Vec::new()),
    };
    Ok((device, queue, decode))
}

/// `dst &= supported`, bool by bool, over core + every carried struct.
fn and_with(dst: &mut Features, sup: &Features) {
    unsafe fn and_bools<T>(d: &mut T, s: &T) {
        let n = (std::mem::size_of::<T>() - 16) / 4;
        let dp = (d as *mut T as *mut u8).add(16) as *mut u32;
        let sp = (s as *const T as *const u8).add(16) as *const u32;
        for i in 0..n {
            *dp.add(i) &= *sp.add(i);
        }
    }
    const N: usize = std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4;
    let d = unsafe { &mut *(&mut dst.core.features as *mut _ as *mut [u32; N]) };
    let s = unsafe { &*(&sup.core.features as *const _ as *const [u32; N]) };
    for i in 0..N {
        d[i] &= s[i];
    }
    unsafe {
        and_bools(&mut dst.v11, &sup.v11);
        and_bools(&mut dst.v12, &sup.v12);
        and_bools(&mut dst.v13, &sup.v13);
        and_bools(&mut dst.coop_matrix, &sup.coop_matrix);
        and_bools(&mut dst.atomic_float, &sup.atomic_float);
        and_bools(&mut dst.subgroup_rotate, &sup.subgroup_rotate);
        and_bools(&mut dst.expect_assume, &sup.expect_assume);
        and_bools(&mut dst.shader_object, &sup.shader_object);
        and_bools(&mut dst.wg_explicit_layout, &sup.wg_explicit_layout);
        and_bools(&mut dst.robustness2, &sup.robustness2);
        and_bools(&mut dst.video_maintenance1, &sup.video_maintenance1);
    }
}

//! Tauri command handlers — the Rust side of the IPC bridge.
//!
//! Each `#[tauri::command]` function is callable from the frontend
//! via `tauri.invoke("cmd_name", { args })`. Args + return values are
//! serialized via serde JSON.
//!
//! Convention: every command returns `Result<T, String>` rather than
//! `anyhow::Result<T>` because Tauri serializes `String` directly to
//! the JS error channel. Internally we use `.map_err(|e| e.to_string())`
//! at the boundary.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─── Versioning ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct VersionInfo {
    pub app: String,
    pub pipeline: String,
}

#[tauri::command]
pub fn version_info() -> VersionInfo {
    VersionInfo {
        app: env!("CARGO_PKG_VERSION").to_string(),
        pipeline: "0.9.x (Neo)".to_string(),
    }
}

// ─── Clip probe ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ClipInfo {
    /// Video stream width (one HEVC track of a .360, e.g. 5952).
    pub width: u32,
    /// Video stream height (one HEVC track, e.g. 1920).
    pub height: u32,
    /// Reported fps from the container (typically 29.97 or 30.0).
    pub fps: f32,
    /// Total duration of THIS file in seconds (single segment).
    pub duration_sec: f64,
    /// Approximate frame count = round(duration × fps).
    pub frame_count: u32,
    /// Sibling chapter paths from the same recording (sorted).
    /// One entry == standalone clip; >1 == multi-segment chain.
    pub segments: Vec<String>,
    /// Total duration across the whole chain, summing each segment.
    pub chain_duration_sec: f64,
    /// EAC tile width derived from `(width - 1920) / 4` — sanity
    /// check; non-zero means the file is a valid EAC layout.
    pub eac_tile_w: u32,
}

#[tauri::command]
pub fn probe_clip(path: String) -> Result<ClipInfo, String> {
    let p = PathBuf::from(&path);
    let probe = vr180_pipeline::decode::probe_video(&p)
        .map_err(|e| format!("probe failed: {e}"))?;
    let dims = vr180_core::eac::Dims::new(probe.width, probe.height);
    let segments = vr180_core::segments::detect_segments(&p);
    let mut chain_duration_sec = 0.0_f64;
    for seg in &segments {
        // Best-effort: if a sibling fails to probe, log and continue.
        match vr180_pipeline::decode::probe_video(seg) {
            Ok(p) => chain_duration_sec += p.duration_sec,
            Err(e) => tracing::warn!("probe sibling {} failed: {}", seg.display(), e),
        }
    }
    Ok(ClipInfo {
        width: probe.width,
        height: probe.height,
        fps: probe.fps,
        duration_sec: probe.duration_sec,
        frame_count: (probe.duration_sec * probe.fps as f64).round() as u32,
        segments: segments.into_iter().map(|p| p.display().to_string()).collect(),
        chain_duration_sec,
        eac_tile_w: dims.tile_w(),
    })
}

// ─── Segment detection (standalone — useful for a "show chain" UI) ─

#[tauri::command]
pub fn detect_segments(path: String) -> Vec<String> {
    let p = PathBuf::from(&path);
    vr180_core::segments::detect_segments(&p)
        .into_iter()
        .map(|p| p.display().to_string())
        .collect()
}

// ─── Preview frame extraction ───────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PreviewRequest {
    pub path: String,
    /// Output half-width per eye. SBS frame is `2 × eye_w × eye_w`.
    /// Use small values (e.g. 512) for fast preview.
    pub eye_w: u32,
    /// Optional time offset into the clip in seconds. Defaults to 0.0
    /// (first frame). When `Some(t)`, we seek the decoder to `t`.
    /// (TODO: not yet plumbed — first cut decodes frame 0 only.)
    pub time_s: Option<f64>,
    /// Optional CDL / color knobs — first cut ignores; preview is
    /// identity. Will be wired with the color stack in the next pass.
    #[serde(default)]
    pub identity_only: bool,
}

#[derive(Debug, Serialize)]
pub struct PreviewResponse {
    /// PNG-encoded SBS half-equirect, base64'd for IPC transport
    /// (Tauri's JS layer doesn't do binary cleanly through invoke).
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
    pub elapsed_ms: f64,
}

/// Decode the first frame of the clip, build EAC crosses, project to
/// half-equirect SBS, encode as PNG, return base64.
///
/// First-cut implementation; uses the CPU-assemble path for simplicity
/// (no zero-copy / no IOSurface dependency on macOS). Performance is
/// adequate for an editor preview (~250 ms cold, faster after first
/// call once the wgpu Device is cached by the Tauri command thread).
#[tauri::command]
pub async fn extract_preview_frame(req: PreviewRequest) -> Result<PreviewResponse, String> {
    use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::extract_first_stream_pair;
    use vr180_pipeline::gpu::{Device, EquirectRotation, EquirectRsParams};
    use base64::Engine;
    use image::ImageEncoder;

    let path = PathBuf::from(&req.path);
    let eye_w = req.eye_w.max(64).min(4096);
    let eye_h = eye_w;
    let _ = req.time_s;          // TODO: wire frame-time seek.
    let _ = req.identity_only;   // TODO: wire color knobs.

    let t = std::time::Instant::now();

    // 1. Decode one pair of HEVC frames + assemble both lens crosses.
    let pair = extract_first_stream_pair(&path)
        .map_err(|e| format!("decode failed: {e}"))?;
    let dims = pair.dims;
    if !dims.is_valid() {
        return Err(format!("invalid EAC layout (stream width {} not a multiple)", dims.stream_w));
    }
    let cw = dims.cross_w() as usize;
    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];
    assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
    assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

    // 2. GPU project both crosses to half-equirect, no rotation, no RS.
    let device = Device::new().map_err(|e| format!("wgpu init failed: {e}"))?;
    let id_rot = EquirectRotation::IDENTITY;
    let id_rs  = EquirectRsParams::DISABLED;
    let left_eye = device.project_cross_to_equirect(
        &cross_b, dims.cross_w(), eye_w, eye_h, id_rot, id_rs,
    ).map_err(|e| format!("project left failed: {e}"))?;
    let right_eye = device.project_cross_to_equirect(
        &cross_a, dims.cross_w(), eye_w, eye_h, id_rot, id_rs,
    ).map_err(|e| format!("project right failed: {e}"))?;

    // 3. Stitch L|R side-by-side.
    let sbs_w = eye_w * 2;
    let row_l = (eye_w * 3) as usize;
    let row_sbs = (sbs_w * 3) as usize;
    let mut sbs = vec![0u8; (sbs_w * eye_h * 3) as usize];
    for y in 0..eye_h as usize {
        sbs[y * row_sbs..y * row_sbs + row_l]
            .copy_from_slice(&left_eye[y * row_l..y * row_l + row_l]);
        sbs[y * row_sbs + row_l..y * row_sbs + row_l * 2]
            .copy_from_slice(&right_eye[y * row_l..y * row_l + row_l]);
    }

    // 4. PNG encode + base64 wrap for IPC.
    let mut png_buf: Vec<u8> = Vec::with_capacity(sbs.len() / 4);
    image::codecs::png::PngEncoder::new(&mut png_buf)
        .write_image(&sbs, sbs_w, eye_h, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("png encode failed: {e}"))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png_buf);

    Ok(PreviewResponse {
        png_base64: b64,
        width: sbs_w,
        height: eye_h,
        elapsed_ms: t.elapsed().as_secs_f64() * 1000.0,
    })
}

// ─── GEOC + SROT probe ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct GeocInfo {
    /// Sensor calibration dimension (4216 for GoPro Max).
    pub cal_dim: u32,
    /// FRNT lens (s0 / right eye after yaw mod) Kannala-Brandt c0..c4.
    pub front_klns: Option<[f64; 5]>,
    /// BACK lens (s4 / left eye after yaw mod) Kannala-Brandt c0..c4.
    pub back_klns: Option<[f64; 5]>,
}

#[tauri::command]
pub fn probe_geoc(path: String) -> Result<Option<GeocInfo>, String> {
    let p = PathBuf::from(&path);
    let g = vr180_core::geoc::parse_geoc(&p)
        .map_err(|e| format!("io error: {e}"))?;
    Ok(g.map(|g| GeocInfo {
        cal_dim: g.cal_dim,
        front_klns: g.front.map(|f| f.klns),
        back_klns:  g.back .map(|b| b.klns),
    }))
}

#[tauri::command]
pub fn lookup_srot_ms(path: String) -> Result<Option<f32>, String> {
    let p = PathBuf::from(&path);
    let s = vr180_core::geoc::lookup_srot_s(&p, None)
        .map_err(|e| format!("io error: {e}"))?;
    Ok(s.map(|s| s * 1000.0))
}

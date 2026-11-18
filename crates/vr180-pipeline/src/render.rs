//! What every arm does to one frame, in one place.
//!
//! The exports, the previews and the stills pose the two eyes, project
//! them, grade and compose them the same way; they differ in where a frame
//! comes from and where it goes. This module is the common part, growing
//! one step at a time out of the arms that each carried a copy of it.

use crate::fisheye_export::FisheyeExportProjection;
use crate::gpu::{Device, EquirectRotation, EyePicture, EyeProjection, FisheyeCalib};
use crate::panomap::ViewAdjust;
use vr180_fisheye::DjiOsvImu;

/// The per-row rolling-shutter correction of a clip: what turns the IMU's
/// high-rate samples into one small rotation per source row.
#[derive(Debug, Clone, Copy)]
pub struct RowReadout {
    /// Sensor readout time of the frame, seconds.
    pub readout_s: f32,
    /// Rows of the picture being projected (the working height, not the
    /// native one, where the two differ).
    pub src_h: u32,
    pub fps: f32,
}

/// How the two eyes of one frame are posed: the rotation each eye's
/// projection takes, and each eye's per-row matrices when it has them.
#[derive(Debug, Clone)]
pub struct FramePose {
    pub rot_left: EquirectRotation,
    pub rot_right: EquirectRotation,
    pub rows_left: Option<Vec<f32>>,
    pub rows_right: Option<Vec<f32>>,
}

impl FramePose {
    /// `rot` is the frame's stabilization rotation (identity when off),
    /// `stab_idx` its row in the IMU's per-frame tables, `swapped` the eye
    /// swap. Each eye gets its lens's own rotation where the source's two
    /// lenses expose at different times ([`crate::dji_imu::per_eye_rotations`]),
    /// then the view adjustment (pano-map + stereo offset) on top; a view
    /// adjustment of zero leaves the rotation exactly as it was. `rows`
    /// asks for the per-row matrices; without an IMU there are none.
    pub fn new(
        rot: EquirectRotation,
        imu: Option<&DjiOsvImu>,
        stab_idx: usize,
        swapped: bool,
        view: &ViewAdjust,
        rows: Option<RowReadout>,
    ) -> Self {
        let (left, right) = crate::dji_imu::per_eye_rotations(rot, imu, stab_idx, swapped);
        let (rot_left, rot_right) = if view.is_identity() {
            (left, right)
        } else {
            let (v_l, v_r) = view.per_eye_matrices();
            (
                EquirectRotation(crate::panomap::mat3_mul_row_major(&left.0, &v_l)),
                EquirectRotation(crate::panomap::mat3_mul_row_major(&right.0, &v_r)),
            )
        };
        let (rows_left, rows_right) = match (rows, imu) {
            (Some(r), Some(imu)) => crate::dji_imu::per_eye_rs_rows(
                imu, stab_idx, r.readout_s, r.src_h, r.fps, swapped),
            _ => (None, None),
        };
        Self { rot_left, rot_right, rows_left, rows_right }
    }
}

/// What projecting one frame's eyes takes besides the pictures themselves:
/// the same request from an export, the live preview and a still. The
/// `project_*` functions are the ways a frame holds its eyes; each makes
/// the two eyes' requests ([`EyeProjection`]) and runs them.
#[derive(Debug, Clone, Copy)]
pub struct EyeJob<'a> {
    /// One eye's size in the source: the size its picture is projected from.
    pub src_w: u32,
    pub src_h: u32,
    /// One eye's size in the output.
    pub eye_w: u32,
    pub eye_h: u32,
    pub pose: &'a FramePose,
    pub calib_left: FisheyeCalib,
    pub calib_right: FisheyeCalib,
    /// The eyes' calibrations before the view was put on them: what a raw
    /// frame's reference is drawn through.
    pub source_calib: (FisheyeCalib, FisheyeCalib),
    pub projection: FisheyeExportProjection,
    /// The frame holds one eye, or is shown as one: the left is projected
    /// and stands for both.
    pub mono: bool,
}

/// The build an eye that is a texture goes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeDepth {
    /// 16 bits a channel, whatever the texture holds: an export's 16-bit
    /// route, and eyes that were resolved to 16 bits.
    Wide,
    /// The texture's own: a 16-bit eye (a raw take's, demosaiced) through
    /// the 16-bit build, so that its linear values reach the colour stack
    /// whole; an 8-bit one (a zero-copy Blackmagic RAW slot) through the
    /// 8-bit build.
    AsHeld,
}

impl<'a> EyeJob<'a> {
    pub fn fisheye_out(&self) -> bool {
        self.projection == FisheyeExportProjection::Fisheye
    }

    /// The per-row matrices of the two eyes: rolling-shutter correction is
    /// applied to both eyes or to neither.
    pub fn rows(&self) -> (Option<&'a [f32]>, Option<&'a [f32]>) {
        let pose: &'a FramePose = self.pose;
        crate::gpu::rows_for_both(pose.rows_left.as_deref(), pose.rows_right.as_deref())
    }

    /// The two eyes' requests for their pictures: 8 bits a channel into
    /// textures of their own, which each `project_*` then widens or keeps.
    fn requests<'p>(&self, left: EyePicture<'p>, right: EyePicture<'p>) -> (EyeProjection<'p>, EyeProjection<'p>)
    where 'a: 'p {
        let (rows_l, rows_r) = self.rows();
        let eye = |picture, rotation, calib, rows| {
            EyeProjection::new(picture, self.eye_w, self.eye_h, rotation, calib)
                .fisheye_out(self.fisheye_out()).rs_rows(rows)
        };
        (
            eye(left, self.pose.rot_left, self.calib_left, rows_l),
            eye(right, self.pose.rot_right, self.calib_right, rows_r),
        )
    }

    /// Both eyes projected, or the left alone standing for both.
    fn run(&self, pipeline: &Device, left: &EyeProjection<'_>, right: &EyeProjection<'_>) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
        pipeline.project_eyes(left, (!self.mono).then_some(right))
    }

    /// Eyes held as RGBA8 pixels in memory, through the 8-bit builds.
    /// `upload`: the two eyes' upload slots (`EyePicture::Bytes8`): an
    /// export, the live preview and a still each have a pair of their own,
    /// since they run side by side.
    pub fn project_bytes_8(&self, pipeline: &Device, left: &[u8], right: &[u8], upload: (u32, u32)) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
        let (l, r) = self.requests(
            EyePicture::Bytes8 { rgba: left, w: self.src_w, h: self.src_h, slot: upload.0 },
            EyePicture::Bytes8 { rgba: right, w: self.src_w, h: self.src_h, slot: upload.1 },
        );
        self.run(pipeline, &l, &r)
    }

    /// Eyes held as pixels in memory (`bit_depth` 8 or 16), through the
    /// 16-bit builds. A frame with per-row matrices writes the outputs kept
    /// for the slots `out`; one without gets textures of its own.
    pub fn project_bytes_16(&self, pipeline: &Device, left: &[u8], right: &[u8], bit_depth: u8, out: (u32, u32)) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
        let (l, r) = self.requests(
            EyePicture::Bytes16 { bytes: left, bit_depth, w: self.src_w, h: self.src_h },
            EyePicture::Bytes16 { bytes: right, bit_depth, w: self.src_w, h: self.src_h },
        );
        fn kept<'p>(p: EyeProjection<'p>, slot: u32) -> EyeProjection<'p> {
            if p.rs_rows.is_some() { p.wide().slot(slot) } else { p.wide() }
        }
        self.run(pipeline, &kept(l, out.0), &kept(r, out.1))
    }

    /// Eyes that are textures on the GPU. A 16-bit build writes the output
    /// kept for its slot of `out`.
    pub fn project_textures(&self, pipeline: &Device, left: &wgpu::Texture, right: &wgpu::Texture, depth: EyeDepth, out: (u32, u32)) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
        // A frame of one eye has no right picture worth looking at.
        let right = if self.mono { left } else { right };
        let (view_l, view_r) = (left.create_view(&Default::default()), right.create_view(&Default::default()));
        let (l, r) = self.requests(EyePicture::Rgba(&view_l), EyePicture::Rgba(&view_r));
        fn build<'p>(p: EyeProjection<'p>, depth: EyeDepth, eye: &wgpu::Texture, slot: u32) -> EyeProjection<'p> {
            if depth == EyeDepth::Wide || eye.format() == wgpu::TextureFormat::Rgba16Unorm { p.wide().slot(slot) } else { p }
        }
        self.run(pipeline, &build(l, depth, left, out.0), &build(r, depth, right, out.1))
    }

    /// Eyes that are decoded surfaces, each its luma plane and its chroma
    /// plane, sampled in place (the Y'CbCr → RGB conversion is in the
    /// kernel): no resolve, no upload. 16 bits a channel.
    pub fn project_planes(&self, pipeline: &Device, left: (&wgpu::Texture, &wgpu::Texture), right: (&wgpu::Texture, &wgpu::Texture)) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
        let right = if self.mono { left } else { right };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (ly, luv, ry, ruv) = (view(left.0), view(left.1), view(right.0), view(right.1));
        let (l, r) = self.requests(EyePicture::Planes { y: &ly, uv: &luv }, EyePicture::Planes { y: &ry, uv: &ruv });
        self.run(pipeline, &l.wide(), &r.wide())
    }
}

impl EyeJob<'_> {
    /// The frame a raw clip's view is developed from
    /// ([`crate::raw_gpu::RawReference`]), where this job's view needs one:
    /// `eyes` are the frame's camera-linear source eyes, and the view is not
    /// the half-equirect (whose own eyes are that frame). `None` for the
    /// half-equirect, and for a frame whose eyes are not held that way. For
    /// a clip that is developed as raw: the caller knows.
    pub fn raw_reference(
        &self,
        pipeline: &Device,
        eyes: Option<(&wgpu::Texture, &wgpu::Texture)>,
        slot: u32,
    ) -> crate::Result<Option<crate::raw_gpu::RawReference>> {
        let Some((left, right)) = eyes else { return Ok(None) };
        if self.projection == FisheyeExportProjection::HalfEquirect || left.format() != wgpu::TextureFormat::Rgba16Unorm {
            return Ok(None);
        }
        raw_view_reference(
            pipeline, (left, right), self.mono, &self.calib_left, (self.eye_w, self.eye_h), self.fisheye_out(),
            self.source_calib, (self.pose.rot_left, self.pose.rot_right),
            (self.pose.rows_left.as_deref(), self.pose.rows_right.as_deref()), slot,
        ).map(Some)
    }
}

/// A frame's eyes after the 16-bit colour stack, each one itself where the
/// plan leaves it as it is. `eye_match`: the state of the automatic
/// eye-exposure match, for a source that has one. `mono`: the frame is one
/// eye, graded once and returned for both.
#[allow(clippy::too_many_arguments)]
pub fn grade_eyes(
    pipeline: &Device,
    left: &wgpu::Texture,
    right: &wgpu::Texture,
    eye_w: u32,
    eye_h: u32,
    plan: &crate::gpu::ColorStackPlan,
    eye_match: Option<(&crate::gpu::EyeMatchState, crate::gpu::EyeMatchFrame)>,
    mono: bool,
) -> crate::Result<(wgpu::Texture, wgpu::Texture)> {
    if mono {
        let eye = pipeline.apply_color_stack_per_eye_16(left, eye_w, eye_h, plan)?
            .unwrap_or_else(|| left.clone());
        return Ok((eye.clone(), eye));
    }
    let (l, r) = pipeline.apply_color_stack_pair_16(left, right, eye_w, eye_h, plan, eye_match)?;
    Ok((l.unwrap_or_else(|| left.clone()), r.unwrap_or_else(|| right.clone())))
}

/// The frame a raw clip's view is developed from when that view is not the
/// half-equirect ([`crate::raw_gpu::RawReference`]): the same moment as a
/// half-equirect, turned as the view is, so that a reframed view or a lens
/// frame comes out in the colours of the half-equirect. The export, the
/// live preview and the stills all draw it here.
///
/// `eyes`: the camera-linear source eyes (the left alone for `one_eye`);
/// `view_calib`: one eye's calibration with the view, drawing a frame of
/// `out`, a fisheye output for `fisheye_out`; `source_calib`: both eyes'
/// before the view; `rot`, `rows`: the frame's rotations and per-row
/// matrices, as its projection takes them ([`FramePose`]).
#[allow(clippy::too_many_arguments)]
fn raw_view_reference(
    pipeline: &crate::gpu::Device,
    eyes: (&wgpu::Texture, &wgpu::Texture),
    one_eye: bool,
    view_calib: &crate::gpu::FisheyeCalib,
    out: (u32, u32),
    fisheye_out: bool,
    source_calib: (crate::gpu::FisheyeCalib, crate::gpu::FisheyeCalib),
    rot: (EquirectRotation, EquirectRotation),
    rows: (Option<&[f32]>, Option<&[f32]>),
    slot: u32,
) -> crate::Result<crate::raw_gpu::RawReference> {
    use crate::raw_gpu::RawReferenceEye;
    // Per-row correction needs both eyes' rows, as in the projection itself.
    let (rows_l, rows_r) = crate::gpu::rows_for_both(rows.0, rows.1);
    let density = view_calib.center_px_per_rad(out.0, out.1, fisheye_out);
    let drawn = [
        RawReferenceEye { src: eyes.0, rotation: rot.0, calib: source_calib.0, rows: rows_l },
        RawReferenceEye { src: eyes.1, rotation: rot.1, calib: source_calib.1, rows: rows_r },
    ];
    pipeline.raw_reference(if one_eye { &drawn[..1] } else { &drawn[..] }, density, slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_view_adjustment_and_no_imu_is_the_stab_rotation_on_both_eyes() {
        let rot = EquirectRotation([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let pose = FramePose::new(rot, None, 3, false, &ViewAdjust::IDENTITY,
            Some(RowReadout { readout_s: 0.015, src_h: 16, fps: 30.0 }));
        assert_eq!(pose.rot_left.0, rot.0);
        assert_eq!(pose.rot_right.0, rot.0);
        assert!(pose.rows_left.is_none() && pose.rows_right.is_none());
    }

    fn job(pose: &FramePose, projection: FisheyeExportProjection) -> EyeJob<'_> {
        let calib = FisheyeCalib::new(4.0, 4.0, 4.0, 3.0, [0.0; 4], 8.0, 6.0, 90.0);
        EyeJob {
            src_w: 8, src_h: 6, eye_w: 32, eye_h: 16, pose,
            calib_left: calib, calib_right: calib, source_calib: (calib, calib),
            projection, mono: false,
        }
    }

    #[test]
    fn the_two_eyes_are_asked_for_at_the_jobs_size_with_their_own_rotation() {
        let (a, b) = (EquirectRotation([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]), EquirectRotation::IDENTITY);
        let pose = FramePose { rot_left: a, rot_right: b, rows_left: None, rows_right: None };
        let (left, right) = (vec![0u8; 8 * 6 * 4], vec![0u8; 8 * 6 * 4]);
        for (projection, fisheye) in [
            (FisheyeExportProjection::HalfEquirect, false),
            (FisheyeExportProjection::Fisheye, true),
            (FisheyeExportProjection::Reframe { hfov_deg: 90.0, defish: 0.0 }, false),
        ] {
            let (l, r) = job(&pose, projection).requests(
                EyePicture::Bytes8 { rgba: &left, w: 8, h: 6, slot: 0 },
                EyePicture::Bytes8 { rgba: &right, w: 8, h: 6, slot: 1 });
            for (eye, rot) in [(&l, a), (&r, b)] {
                assert_eq!((eye.out_w, eye.out_h, eye.fisheye_out, eye.wide), (32, 16, fisheye, false));
                assert_eq!(eye.rotation.0, rot.0);
                assert_eq!(eye.output, crate::gpu::EyeOutput::New);
                assert!(eye.rs_rows.is_none());
            }
        }
    }

    #[test]
    fn rows_go_to_both_eyes_or_to_neither() {
        let rows = vec![0.0f32; 12 * 6];
        let with = |l: bool, r: bool| FramePose {
            rot_left: EquirectRotation::IDENTITY, rot_right: EquirectRotation::IDENTITY,
            rows_left: l.then(|| rows.clone()), rows_right: r.then(|| rows.clone()),
        };
        let bytes = vec![0u8; 8 * 6 * 4];
        let picture = EyePicture::Bytes8 { rgba: &bytes, w: 8, h: 6, slot: 0 };
        for (l, r, both) in [(true, true, true), (true, false, false), (false, true, false), (false, false, false)] {
            let pose = with(l, r);
            let job = job(&pose, FisheyeExportProjection::HalfEquirect);
            let (a, b) = job.requests(picture, picture);
            assert_eq!((a.rs_rows.is_some(), b.rs_rows.is_some()), (both, both), "left {l}, right {r}");
            assert_eq!(job.rows().0.is_some(), both);
        }
    }

    #[test]
    fn the_view_adjustment_goes_on_top_of_each_eye() {
        let rot = EquirectRotation([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let view = ViewAdjust { pano_yaw_deg: 10.0, stereo_yaw_deg: 2.0, ..ViewAdjust::IDENTITY };
        let pose = FramePose::new(rot, None, 0, false, &view, None);
        let (v_l, v_r) = view.per_eye_matrices();
        assert_eq!(pose.rot_left.0, crate::panomap::mat3_mul_row_major(&rot.0, &v_l));
        assert_eq!(pose.rot_right.0, crate::panomap::mat3_mul_row_major(&rot.0, &v_r));
        assert_ne!(pose.rot_left.0, pose.rot_right.0);
    }
}

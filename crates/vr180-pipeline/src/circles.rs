//! Image-circle detection for a fisheye eye: where the lens's image circle
//! sits in the frame and how big it is — the seed for a manual lens
//! calibration when the file carries none (Blackmagic RAW).
//!
//! The rim outside the circle is a nearly uniform dark level (log black on
//! a BRAW decode is ~25/255, not 0), the image inside is brighter almost
//! everywhere. Per eye: the rim level from the frame corners, a threshold
//! a little above it, then the first / last lit pixel of every row and
//! column that does not touch the eye's border, and a least-squares circle
//! through those edge points with four rounds of outlier rejection. On the
//! PYXIS 12K sample (12288×6480, two 6360-px circles) the fit's median
//! residual is 7 px; a 2.4:1 window that cuts the circles top and bottom
//! still fits from the side arcs (13 px).

use crate::braw_imu::EyeCircle;

/// Least-squares circle through `pts` (x, y): `(cx, cy, r)`.
fn fit_circle(pts: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    // Solve [2x 2y 1]·[cx cy c] = x² + y² by normal equations (3×3).
    let n = pts.len();
    if n < 6 { return None; }
    let mut a = [[0f64; 3]; 3];
    let mut b = [0f64; 3];
    for &(x, y) in pts {
        let (x, y) = (x as f64, y as f64);
        let row = [2.0 * x, 2.0 * y, 1.0];
        let rhs = x * x + y * y;
        for i in 0..3 { for j in 0..3 { a[i][j] += row[i] * row[j]; } b[i] += row[i] * rhs; }
    }
    // Gaussian elimination with partial pivoting.
    let mut m = [[a[0][0], a[0][1], a[0][2], b[0]], [a[1][0], a[1][1], a[1][2], b[1]], [a[2][0], a[2][1], a[2][2], b[2]]];
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap())?;
        m.swap(c, p);
        if m[c][c].abs() < 1e-9 { return None; }
        for r in 0..3 {
            if r == c { continue; }
            let f = m[r][c] / m[c][c];
            for k in c..4 { m[r][k] -= f * m[c][k]; }
        }
    }
    let cx = m[0][3] / m[0][0];
    let cy = m[1][3] / m[1][1];
    let c = m[2][3] / m[2][2];
    let r2 = c + cx * cx + cy * cy;
    (r2 > 0.0).then(|| (cx as f32, cy as f32, r2.sqrt() as f32))
}

/// Find the image circle in ONE eye of a packed RGBA frame (`bpp` = 4 for
/// RGBA8, 8 for RGBA64LE), the eye being the sub-rectangle at `x0` with
/// width `eye_w` (full height). Works on a reduced-resolution decode: the
/// result is in the frame's own pixels — scale it to the native size.
pub fn detect_eye_circle(rgba: &[u8], frame_w: u32, frame_h: u32, bpp: usize, x0: u32, eye_w: u32) -> Option<EyeCircle> {
    let (fw, fh, ew) = (frame_w as usize, frame_h as usize, eye_w as usize);
    let x0 = x0 as usize;
    if ew < 16 || fh < 16 || rgba.len() < fw * fh * bpp { return None; }
    // Brightness = max(R, G, B), high byte for 16-bit.
    let lum = |x: usize, y: usize| -> u8 {
        let o = (y * fw + x) * bpp;
        if bpp == 8 { rgba[o + 1].max(rgba[o + 3]).max(rgba[o + 5]) } else { rgba[o].max(rgba[o + 1]).max(rgba[o + 2]) }
    };
    // Rim level: the eye's four corner patches (and the eye's outer edge mid-height).
    let patch = (ew / 40).clamp(4, 64);
    let mut corner: Vec<u8> = Vec::with_capacity(patch * patch * 4);
    for (cx, cy) in [(x0, 0), (x0 + ew - patch, 0), (x0, fh - patch), (x0 + ew - patch, fh - patch)] {
        for y in cy..cy + patch { for x in cx..cx + patch { corner.push(lum(x, y)); } }
    }
    corner.sort_unstable();
    let rim = corner[corner.len() / 2] as f32;
    // A brightness clearly above the rim: rim + 15 % of the way to the eye's
    // 90th percentile (sampled on a grid), at least +6.
    let mut grid: Vec<u8> = Vec::new();
    let step = (ew / 128).max(1);
    let mut y = 0; while y < fh { let mut x = x0; while x < x0 + ew { grid.push(lum(x, y)); x += step; } y += step; }
    grid.sort_unstable();
    let p90 = grid[grid.len() * 9 / 10] as f32;
    let thr = rim + (0.15 * (p90 - rim)).max(6.0);
    let lit = |x: usize, y: usize| lum(x, y) as f32 > thr;
    // Edge points: first/last lit pixel per row and per column, only where
    // the edge is inside the eye (not on its border).
    let mut pts: Vec<(f32, f32)> = Vec::new();
    let rs = (fh / 400).max(1);
    let mut y = 2;
    while y + 2 < fh {
        let mut first = None; let mut last = None;
        for x in x0..x0 + ew { if lit(x, y) { if first.is_none() { first = Some(x); } last = Some(x); } }
        if let (Some(f), Some(l)) = (first, last) {
            if l - f > ew / 20 {
                if f > x0 + 1 { pts.push((f as f32, y as f32)); }
                if l + 2 < x0 + ew { pts.push((l as f32, y as f32)); }
            }
        }
        y += rs;
    }
    let cs = (ew / 400).max(1);
    let mut x = x0 + 2;
    while x + 2 < x0 + ew {
        let mut first = None; let mut last = None;
        for yy in 0..fh { if lit(x, yy) { if first.is_none() { first = Some(yy); } last = Some(yy); } }
        if let (Some(f), Some(l)) = (first, last) {
            if l - f > fh / 20 {
                if f > 1 { pts.push((x as f32, f as f32)); }
                if l + 2 < fh { pts.push((x as f32, l as f32)); }
            }
        }
        x += cs;
    }
    if pts.len() < 50 { return None; }
    let (mut cx, mut cy, mut r) = fit_circle(&pts)?;
    let mut keep: Vec<(f32, f32)> = pts.clone();
    for _ in 0..4 {
        let mut d: Vec<f32> = keep.iter().map(|&(x, y)| (((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r).abs()).collect();
        let mut sorted = d.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut = sorted[sorted.len() * 7 / 10].max(2.0);
        let next: Vec<(f32, f32)> = keep.iter().zip(d.drain(..)).filter(|(_, dd)| *dd < cut).map(|(p, _)| *p).collect();
        if next.len() < 20 { break; }
        keep = next;
        match fit_circle(&keep) { Some(f) => { cx = f.0; cy = f.1; r = f.2; } None => break }
    }
    // Sanity: the circle must be a sensible size and roughly inside the eye.
    let min_r = ew.min(fh) as f32 * 0.25;
    let max_r = ew.max(fh) as f32 * 0.9;
    if !(r.is_finite() && r > min_r && r < max_r) { return None; }
    if cx < x0 as f32 - r * 0.5 || cx > (x0 + ew) as f32 + r * 0.5 || cy < -r * 0.5 || cy > fh as f32 + r * 0.5 { return None; }
    Some(EyeCircle { cx: cx - x0 as f32, cy, r })
}

/// Both eyes of a side-by-side frame: `[left, right]`, each in its own
/// eye's pixels, scaled by `scale` (native / decoded) to the native eye.
pub fn detect_sbs_circles(rgba: &[u8], frame_w: u32, frame_h: u32, bpp: usize, scale: f32) -> Option<[EyeCircle; 2]> {
    let ew = frame_w / 2;
    let l = detect_eye_circle(rgba, frame_w, frame_h, bpp, 0, ew)?;
    let r = detect_eye_circle(rgba, frame_w, frame_h, bpp, ew, ew)?;
    let s = |c: EyeCircle| EyeCircle { cx: c.cx * scale, cy: c.cy * scale, r: c.r * scale };
    Some([s(l), s(r)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_synthetic_circles() {
        let (w, h) = (640u32, 300u32);
        let mut img = vec![0u8; (w * h * 4) as usize];
        let circles = [(150.0f32, 160.0f32, 140.0f32), (480.0f32, 140.0f32, 138.0f32)];
        for y in 0..h { for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            let inside = circles.iter().any(|&(cx, cy, r)| ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < r);
            let v = if inside { 90 + ((x * 7 + y * 3) % 120) as u8 } else { 24 + ((x + y) % 3) as u8 };
            img[o] = v; img[o + 1] = v; img[o + 2] = v / 2; img[o + 3] = 255;
        } }
        let found = detect_sbs_circles(&img, w, h, 4, 2.0).expect("circles");
        assert!((found[0].cx - 300.0).abs() < 4.0 && (found[0].cy - 320.0).abs() < 4.0 && (found[0].r - 280.0).abs() < 4.0, "{:?}", found[0]);
        assert!((found[1].cx - (480.0 - 320.0) * 2.0).abs() < 4.0 && (found[1].cy - 280.0).abs() < 4.0 && (found[1].r - 276.0).abs() < 4.0, "{:?}", found[1]);
        // A flat frame has no circle.
        let flat = vec![30u8; (w * h * 4) as usize];
        assert!(detect_sbs_circles(&flat, w, h, 4, 1.0).is_none());
    }
}

//! Kannala-Brandt fisheye projection — forward + inverse.
//!
//! Ports the Metal kernel from `vr180_gui.py:1350-1480` to pure Rust.
//! Used by:
//! - Tests (validate the WGSL shader output against this reference).
//! - The Gyroflow JSON loader (FOV derivation from rim radius).
//! - The GUI's live preview sliders (single-point sanity checks).
//!
//! ## Forward model
//!
//! For a 3D world ray with elevation angle θ from the optical axis,
//! the pixel radius from the principal point is
//!
//! ```text
//! r(θ) = fx · θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
//! ```
//!
//! This is the "Kannala-Brandt 4" model (also called "fisheye4" in
//! OpenCV / Gyroflow). It's a generalised fisheye polynomial that
//! covers equidistant / equisolid / stereographic / orthographic
//! lenses with the right `k_n` values.
//!
//! ## Inverse
//!
//! There's no closed-form inverse for an 8th-degree polynomial in θ.
//! We do Newton-Raphson:
//!
//! ```text
//! f(θ)   = fx · θ · (1 + k1·θ² + …) - r_px
//! f'(θ)  = fx · (1 + 3·k1·θ² + 5·k2·θ⁴ + 7·k3·θ⁶ + 9·k4·θ⁸)
//! θ_{n+1} = θ_n - f(θ_n) / f'(θ_n)
//! ```
//!
//! In the shader we run 8 iterations from `θ₀ = r_px / fx`; here we
//! default to 32 for the reference path. Both converge well inside the
//! calibrated region (θ < ~80°).
//!
//! ## Cubic Hermite extension past θ_trans
//!
//! KB is calibrated up to some maximum angle `θ_trans` (Python uses
//! 80° = 1.3963 rad). Past that, the polynomial becomes unreliable —
//! Sirius lenses also tend to have steep falloff that the 8-degree
//! polynomial can't model. The Python kernel switches to a cubic
//! Hermite extension that matches r and dr/dθ at the boundary and
//! pins r_max at the image-circle radius:
//!
//! ```text
//! u = (θ - θ_trans) / (θ_max - θ_trans)
//! r(θ) = h00·r_trans + h10·(θ_max - θ_trans)·r'_trans
//!      + h01·r_max
//! ```
//!
//! where `h00`, `h10`, `h01` are the standard cubic Hermite basis
//! functions. The inverse on this region is a second Newton-Raphson on
//! the cubic — fast because cubics are well-behaved.

use std::f64::consts::PI;

/// Default transition angle for KB → cubic Hermite extension.
/// 80° in radians. Matches `vr180_gui.py:2531`.
pub const THETA_TRANS_DEFAULT: f64 = 80.0 * PI / 180.0;

/// Default maximum angle for the cubic extension.
/// 110° in radians. Most super-fisheye lenses (e.g. DJI Osmo 360 at
/// 207.68° full FOV → 103.84° half-FOV) stay within this.
pub const THETA_MAX_DEFAULT: f64 = 110.0 * PI / 180.0;

/// Kannala-Brandt forward projection of one ray.
///
/// `theta` in radians (elevation from optical axis), `fx` in pixels,
/// `k1..k4` are the distortion polynomial coefficients (Gyroflow /
/// OpenCV convention, NOT the GoPro GEOC convention which folds fx in).
///
/// Returns the pixel radius from the principal point.
#[inline]
pub fn kb_forward(theta: f64, fx: f64, k: [f64; 4]) -> f64 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    fx * theta * (1.0 + k[0] * t2 + k[1] * t4 + k[2] * t6 + k[3] * t8)
}

/// Derivative of `kb_forward` w.r.t. θ. Used by Newton-Raphson inverse.
#[inline]
pub fn kb_forward_deriv(theta: f64, fx: f64, k: [f64; 4]) -> f64 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    fx * (1.0
        + 3.0 * k[0] * t2
        + 5.0 * k[1] * t4
        + 7.0 * k[2] * t6
        + 9.0 * k[3] * t8)
}

/// Newton-Raphson inverse of [`kb_forward`].
///
/// Solves `kb_forward(θ) = r_px` for θ. Initial guess `θ₀ = r_px / fx`
/// (true for k=0). Stops at `max_iters` or when the step size drops
/// below `tol_rad` radians.
///
/// `theta_clamp` is the upper bound — past it the polynomial gets
/// unreliable and we should be in the cubic Hermite region instead.
/// Pass [`THETA_TRANS_DEFAULT`] for default behaviour, or `PI` to
/// disable clamping.
pub fn kb_inverse(
    r_px: f64,
    fx: f64,
    k: [f64; 4],
    theta_clamp: f64,
    max_iters: u32,
    tol_rad: f64,
) -> f64 {
    let mut theta = (r_px / fx).clamp(0.0, theta_clamp);
    for _ in 0..max_iters {
        let f_val = kb_forward(theta, fx, k) - r_px;
        let f_der = kb_forward_deriv(theta, fx, k);
        if f_der.abs() < 1e-12 {
            break;
        }
        let step = f_val / f_der;
        theta = (theta - step).clamp(0.0, theta_clamp);
        if step.abs() < tol_rad {
            break;
        }
    }
    theta
}

/// Cubic Hermite extension parameters.
///
/// Computed once per calibration. The shader and CPU paths both use
/// these to evaluate r(θ) and θ(r) past `theta_trans`.
#[derive(Debug, Clone, Copy)]
pub struct CubicExtension {
    /// Lower boundary (where KB stops being valid).
    pub theta_trans: f64,
    /// Upper boundary (where image circle ends).
    pub theta_max: f64,
    /// r evaluated at θ_trans (KB-forward).
    pub r_trans: f64,
    /// dr/dθ evaluated at θ_trans (KB-forward derivative).
    pub r_trans_deriv: f64,
    /// r at θ_max (image-circle radius in pixels).
    pub r_max: f64,
}

impl CubicExtension {
    /// Build the extension from a KB calibration. `r_max_px` is the
    /// image-circle radius (e.g. half the fisheye width for an
    /// in-frame circle).
    pub fn new(fx: f64, k: [f64; 4], theta_trans: f64, theta_max: f64, r_max_px: f64) -> Self {
        Self {
            theta_trans,
            theta_max,
            r_trans: kb_forward(theta_trans, fx, k),
            r_trans_deriv: kb_forward_deriv(theta_trans, fx, k),
            r_max: r_max_px,
        }
    }

    /// Evaluate r(θ) on the cubic extension. Caller guarantees
    /// `theta_trans ≤ θ ≤ theta_max`.
    #[inline]
    pub fn forward(&self, theta: f64) -> f64 {
        let span = self.theta_max - self.theta_trans;
        let u = ((theta - self.theta_trans) / span).clamp(0.0, 1.0);
        // Standard cubic Hermite basis with zero outgoing tangent
        // (free end at θ_max, just pin the value).
        let h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0;
        let h10 = u * u * u - 2.0 * u * u + u;
        let h01 = -2.0 * u * u * u + 3.0 * u * u;
        h00 * self.r_trans + h10 * span * self.r_trans_deriv + h01 * self.r_max
    }

    /// Derivative dr/dθ on the cubic extension (for Newton-Raphson).
    #[inline]
    pub fn forward_deriv(&self, theta: f64) -> f64 {
        let span = self.theta_max - self.theta_trans;
        let u = ((theta - self.theta_trans) / span).clamp(0.0, 1.0);
        let dh00 = 6.0 * u * u - 6.0 * u;
        let dh10 = 3.0 * u * u - 4.0 * u + 1.0;
        let dh01 = -6.0 * u * u + 6.0 * u;
        (dh00 * self.r_trans + dh10 * span * self.r_trans_deriv + dh01 * self.r_max) / span
    }

    /// Newton-Raphson inverse on the cubic extension.
    /// Returns θ in `[theta_trans, theta_max]`.
    pub fn inverse(&self, r_px: f64, max_iters: u32, tol_rad: f64) -> f64 {
        // Linear initial guess on the cubic's range.
        let r_lo = self.r_trans;
        let r_hi = self.r_max.max(r_lo + 1e-9);
        let u0 = ((r_px - r_lo) / (r_hi - r_lo)).clamp(0.0, 1.0);
        let mut theta = self.theta_trans + u0 * (self.theta_max - self.theta_trans);
        for _ in 0..max_iters {
            let f_val = self.forward(theta) - r_px;
            let f_der = self.forward_deriv(theta);
            if f_der.abs() < 1e-12 {
                break;
            }
            let step = f_val / f_der;
            theta = (theta - step).clamp(self.theta_trans, self.theta_max);
            if step.abs() < tol_rad {
                break;
            }
        }
        theta
    }
}

/// End-to-end "pixel radius → θ" inverse, switching to the cubic
/// extension past `ext.theta_trans`. This is what the shader does
/// per-pixel. The CPU reference is kept for tests, FOV derivation,
/// and live-preview sliders.
pub fn r_to_theta(
    r_px: f64,
    fx: f64,
    k: [f64; 4],
    ext: &CubicExtension,
) -> f64 {
    if r_px <= ext.r_trans {
        kb_inverse(r_px, fx, k, ext.theta_trans, 32, 1e-9)
    } else {
        ext.inverse(r_px, 16, 1e-9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip identity: KB forward → inverse → original θ.
    #[test]
    fn roundtrip_identity_centre_region() {
        // DJI Osmo 360 calibration (vr180_gui.py:2864-2868).
        let fx = 1000.0;
        let k = [0.063054046599, 0.003034146878, -0.004623015478, -0.000516517650];
        for deg in [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0] {
            let theta = deg * PI / 180.0;
            let r = kb_forward(theta, fx, k);
            let theta_back = kb_inverse(r, fx, k, THETA_TRANS_DEFAULT, 32, 1e-12);
            assert!(
                (theta_back - theta).abs() < 1e-9,
                "θ={deg}° round-trip: in={theta:.9}, out={theta_back:.9}"
            );
        }
    }

    /// Cubic extension matches KB at θ_trans (continuity check).
    #[test]
    fn cubic_extension_continuity_at_theta_trans() {
        let fx = 1000.0;
        let k = [0.063054046599, 0.003034146878, -0.004623015478, -0.000516517650];
        let r_max = 1800.0; // arbitrary image-circle radius
        let ext = CubicExtension::new(fx, k, THETA_TRANS_DEFAULT, THETA_MAX_DEFAULT, r_max);

        let r_kb_at_trans = kb_forward(THETA_TRANS_DEFAULT, fx, k);
        let r_ext_at_trans = ext.forward(THETA_TRANS_DEFAULT);
        assert!(
            (r_kb_at_trans - r_ext_at_trans).abs() < 1e-9,
            "continuity: kb={r_kb_at_trans} ext={r_ext_at_trans}"
        );

        let r_ext_at_max = ext.forward(THETA_MAX_DEFAULT);
        assert!(
            (r_ext_at_max - r_max).abs() < 1e-9,
            "extension pins r_max: ext={r_ext_at_max} target={r_max}"
        );
    }

    /// Cubic-extension round-trip inside its range.
    #[test]
    fn cubic_extension_roundtrip() {
        let fx = 1000.0;
        let k = [0.063054046599, 0.003034146878, -0.004623015478, -0.000516517650];
        let r_max = 1800.0;
        let ext = CubicExtension::new(fx, k, THETA_TRANS_DEFAULT, THETA_MAX_DEFAULT, r_max);
        for deg in [80.0, 90.0, 100.0, 105.0, 110.0] {
            let theta = deg * PI / 180.0;
            let r = ext.forward(theta);
            let theta_back = ext.inverse(r, 64, 1e-12);
            assert!(
                (theta_back - theta).abs() < 1e-8,
                "θ={deg}° cubic round-trip: in={theta:.9}, out={theta_back:.9}"
            );
        }
    }

    /// Full r_to_theta switchover at the boundary stays smooth.
    #[test]
    fn r_to_theta_no_discontinuity() {
        let fx = 1000.0;
        let k = [0.063054046599, 0.003034146878, -0.004623015478, -0.000516517650];
        let r_max = 1800.0;
        let ext = CubicExtension::new(fx, k, THETA_TRANS_DEFAULT, THETA_MAX_DEFAULT, r_max);
        let r_at_trans = ext.r_trans;
        let theta_below = r_to_theta(r_at_trans - 0.001, fx, k, &ext);
        let theta_above = r_to_theta(r_at_trans + 0.001, fx, k, &ext);
        // Should differ by <1 millirad across a 0.002-pixel jump.
        assert!(
            (theta_above - theta_below).abs() < 1e-3,
            "discontinuity at θ_trans: below={theta_below:.6} above={theta_above:.6}"
        );
    }
}

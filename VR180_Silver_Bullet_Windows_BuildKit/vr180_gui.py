#!/usr/bin/env python3
"""
VR180 SBS Half-Equirectangular Video Processor - GUI Edition
"""

import sys

# Fix console encoding on Windows (GBK can't handle Unicode symbols)
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import subprocess
import json
import os
import struct
import shutil
import platform

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_MACOS = platform.system() == 'Darwin'
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

# Import OpenCV for alignment overlay drawing
try:
    import cv2
    HAS_CV2 = True
    print(f"✓ OpenCV imported successfully - version: {cv2.__version__}")
except ImportError as e:
    HAS_CV2 = False
    print(f"⚠ Warning: OpenCV (cv2) not available - alignment preview will be disabled")
    print(f"  Import error: {e}")
except Exception as e:
    HAS_CV2 = False
    print(f"⚠ Warning: OpenCV import failed with error: {e}")

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
    print(f"✓ SciPy imported successfully")
except ImportError:
    HAS_SCIPY = False
    print(f"⚠ Warning: SciPy not available - gyro stabilization will be disabled")

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    print(f"✓ Numba imported successfully - version: {numba.__version__}")
except ImportError:
    HAS_NUMBA = False
    print(f"⚠ Warning: Numba not available - render will use numpy fallback")

try:
    import av
    HAS_AV = True
    print(f"✓ PyAV imported successfully - version: {av.__version__}")
except ImportError:
    HAS_AV = False

# ── CUDA PATH setup (MUST happen before importing numba.cuda) ─────────────
# PyInstaller frozen apps lose system PATH; Numba caches CUDA availability on
# first import, so we must fix PATH before `from numba import cuda`.
# IMPORTANT: Numba only supports CUDA 11.x and 12.x. CUDA 13+ is NOT supported.
# If multiple toolkits are installed, prefer 12.x over 13.x.
import os as _os
if _os.name == 'nt' and HAS_NUMBA:
    _cuda_found_path = None

    # 1. Scan for CUDA 12.x first (Numba-compatible), then fall back to CUDA_PATH
    #    CUDA_PATH may point to 13.x which Numba can't use.
    #    Check CUDA_PATH_V12_* env vars first
    for _key, _val in sorted(_os.environ.items(), reverse=True):
        if _key.startswith('CUDA_PATH_V12') and _os.path.isdir(_os.path.join(_val, 'bin')):
            _cuda_bin = _os.path.join(_val, 'bin')
            _os.environ['PATH'] = _cuda_bin + ';' + _os.environ.get('PATH', '')
            _cuda_found_path = _cuda_bin
            print(f"  {_key} found (Numba-compatible): {_cuda_bin}")
            break

    # 2. Scan common install locations for CUDA 12.x
    if not _cuda_found_path:
        for _cuda_ver in ['v12.8', 'v12.7', 'v12.6', 'v12.5', 'v12.4', 'v12.3', 'v12.2',
                           'v12.1', 'v12.0']:
            _cuda_default = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{_cuda_ver}\\bin'
            if _os.path.isdir(_cuda_default):
                _os.environ['PATH'] = _cuda_default + ';' + _os.environ.get('PATH', '')
                _cuda_found_path = _cuda_default
                print(f"  Found CUDA 12.x toolkit: {_cuda_default}")
                break

    # 3. Fall back to CUDA_PATH (even if it's 13.x — might work in future Numba)
    if not _cuda_found_path:
        _cuda_path = _os.environ.get('CUDA_PATH', '')
        if _cuda_path:
            _cuda_bin = _os.path.join(_cuda_path, 'bin')
            if _os.path.isdir(_cuda_bin):
                _os.environ['PATH'] = _cuda_bin + ';' + _os.environ.get('PATH', '')
                _cuda_found_path = _cuda_bin
                # Warn if it's CUDA 13+
                if 'v13' in _cuda_path or '\\13.' in _cuda_path:
                    print(f"  ⚠ CUDA_PATH points to {_cuda_path}")
                    print(f"    Numba only supports CUDA 11.x/12.x. Install CUDA 12.6 for GPU acceleration:")
                    print(f"    https://developer.nvidia.com/cuda-12-6-0-download-archive")
                else:
                    print(f"  CUDA_PATH found: {_cuda_bin}")

    # 4. Also check CUDA 11.x as last resort
    if not _cuda_found_path:
        for _cuda_ver in ['v11.8', 'v11.7', 'v11.6']:
            _cuda_default = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{_cuda_ver}\\bin'
            if _os.path.isdir(_cuda_default):
                _os.environ['PATH'] = _cuda_default + ';' + _os.environ.get('PATH', '')
                _cuda_found_path = _cuda_default
                print(f"  Found CUDA 11.x toolkit: {_cuda_default}")
                break

    # 5. Ensure NVIDIA driver path is accessible (for nvcuda.dll)
    _nv_driver = _os.path.join(_os.environ.get('SystemRoot', 'C:\\Windows'), 'System32')
    if _nv_driver not in _os.environ.get('PATH', ''):
        _os.environ['PATH'] = _os.environ.get('PATH', '') + ';' + _nv_driver

    if not _cuda_found_path:
        import glob as _glob
        for _p in _os.environ.get('PATH', '').split(';'):
            if _glob.glob(_os.path.join(_p, 'cudart64_*.dll')):
                _cuda_found_path = _p
                print(f"  Found cudart in PATH: {_p}")
                break
        if not _cuda_found_path:
            print("  ⚠ No Numba-compatible CUDA toolkit found")
            print("    Install CUDA 12.6: https://developer.nvidia.com/cuda-12-6-0-download-archive")

# ── Numba CUDA detection ──────────────────────────────────────────────────
HAS_NUMBA_CUDA = False
_numba_cuda = None
if HAS_NUMBA:
    try:
        from numba import cuda as _numba_cuda
        if _numba_cuda.is_available():
            HAS_NUMBA_CUDA = True
            _dev = _numba_cuda.get_current_device()
            print(f"✓ Numba CUDA available — {_dev.name}, compute {_dev.compute_capability}")
        else:
            # Diagnose
            _diag_msg = "unknown reason"
            try:
                from numba.cuda.cudadrv.driver import driver as _drv
                _cnt = _drv.get_device_count()
                _diag_msg = f"driver sees {_cnt} GPU(s) but CUDA runtime (cudart) not loadable"
            except Exception as _diag:
                _diag_msg = str(_diag)
            print(f"⚠ Numba CUDA: {_diag_msg}")
            print("  Tip: Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            if _os.name == 'nt':
                print(f"  Current PATH dirs with 'cuda' (case-insensitive):")
                for _p in _os.environ.get('PATH', '').split(';'):
                    if 'cuda' in _p.lower():
                        print(f"    {_p}")
    except Exception as _cuda_err:
        print(f"⚠ Numba CUDA detection failed: {_cuda_err}")
        import traceback
        traceback.print_exc()

# ── FFmpeg NVDEC hardware decode detection (independent of Numba CUDA) ──
# NVDEC depends on NVIDIA driver, not CUDA toolkit. Detect via ffmpeg -hwaccels.
_has_nvdec = False
if _os.name == 'nt':
    try:
        import shutil as _shutil
        _ffmpeg_bin = _shutil.which('ffmpeg') or 'ffmpeg'
        _nvdec_test = subprocess.run(
            [_ffmpeg_bin, '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=5)
        if 'cuda' in _nvdec_test.stdout.lower():
            _has_nvdec = True
            print(f"✓ FFmpeg NVDEC hardware decode available")
    except Exception:
        pass

# ── Numba JIT kernels for EAC cross remap (render hot path) ──────────────
if HAS_NUMBA:
    import math as _math

    @njit(cache=True, parallel=True)
    def _nb_dirs_to_cross_maps(xn, yn, zn, mx_out, my_out, h, w):
        """Numba JIT: map 3D direction vectors to EAC cross pixel coords.
        Fused single-pass loop — no temporary arrays, no mask allocation.
        Side/top/bottom face UVs are clamped to valid half-face range
        so edge pixels get nearest content instead of black."""
        TWO_OVER_PI = numba.float32(2.0 / _math.pi)
        for i in prange(h * w):
            x = xn[i]
            y = yn[i]
            z = zn[i]
            ax = abs(x)
            ay = abs(y)
            mx_out[i] = numba.float32(-1.0)
            my_out[i] = numba.float32(-1.0)

            if z > 0 and ax <= z and ay <= z:
                # Front face: cols [1008,2928), rows [1008,2928)
                u = TWO_OVER_PI * _math.atan(x / z) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / z)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x > 0 and z <= x and ay <= x:
                # Right face: cols [2928,3936), rows [1008,2928)
                u = TWO_OVER_PI * _math.atan(-z / x) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / x)
                fc = u * numba.float32(1920.0)
                if fc < numba.float32(0.0):
                    fc = numba.float32(0.0)
                elif fc > numba.float32(1007.0):
                    fc = numba.float32(1007.0)
                mx_out[i] = numba.float32(2928.0) + fc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x < 0 and z <= ax and ay <= ax:
                # Left face: cols [0,1008), rows [1008,2928)
                u = TWO_OVER_PI * _math.atan(z / ax) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / ax)
                pc = u * numba.float32(1920.0) - numba.float32(912.0)
                if pc < numba.float32(0.0):
                    pc = numba.float32(0.0)
                elif pc > numba.float32(1007.0):
                    pc = numba.float32(1007.0)
                mx_out[i] = pc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif y > 0 and ax <= y and z <= y:
                # Top face: rows [0,1008), cols [1008,2928)
                u = TWO_OVER_PI * _math.atan(x / y) + numba.float32(0.5)
                v = TWO_OVER_PI * _math.atan(z / y) + numba.float32(0.5)
                pr = v * numba.float32(1920.0) - numba.float32(912.0)
                if pr < numba.float32(0.0):
                    pr = numba.float32(0.0)
                elif pr > numba.float32(1007.0):
                    pr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = pr
            elif y < 0 and ax <= ay and z <= ay:
                # Bottom face: rows [2928,3936), cols [1008,2928)
                u = TWO_OVER_PI * _math.atan(x / ay) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(z / ay)
                fr = v * numba.float32(1920.0)
                if fr < numba.float32(0.0):
                    fr = numba.float32(0.0)
                elif fr > numba.float32(1007.0):
                    fr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(2928.0) + fr

            # Clip to valid range
            if mx_out[i] > numba.float32(3935.0):
                mx_out[i] = numba.float32(3935.0)
            if my_out[i] > numba.float32(3935.0):
                my_out[i] = numba.float32(3935.0)

    @njit(cache=True, parallel=True)
    def _nb_cross_remap_rs(xyz_x, xyz_y, xyz_z,
                           R00, R01, R02, R10, R11, R12, R20, R21, R22,
                           t_offset,
                           yaw_coeff, pitch_coeff, roll_coeff,
                           has_Rc, Rc00, Rc01, Rc02, Rc10, Rc11, Rc12, Rc20, Rc21, Rc22,
                           mx_out, my_out, n,
                           klns_c0=numba.float32(0), klns_c1=numba.float32(0),
                           klns_c2=numba.float32(0), klns_c3=numba.float32(0),
                           klns_c4=numba.float32(0),
                           cx_s=numba.float32(0), cy_s=numba.float32(0),
                           cal_d=numba.float32(0), srot_s=numba.float32(0),
                           use_klns=False):
        """Numba JIT: fused RS pipeline — R×dir → KLNS RS → IORI → cross maps.
        When use_klns=True, uses KLNS dr/dθ pixel-shift (matches Metal kernel).
        When False, uses 3D rotation fallback."""
        TWO_OVER_PI = numba.float32(2.0 / _math.pi)
        for i in prange(n):
            # Step 1: R_sensor × direction → sensor space
            ox = xyz_x[i]; oy = xyz_y[i]; oz = xyz_z[i]
            xr = R00 * ox + R01 * oy + R02 * oz
            yr = R10 * ox + R11 * oy + R12 * oz
            zr = R20 * ox + R21 * oy + R22 * oz

            # Step 2: RS correction
            xn = xr; yn = yr; zn = zr
            if use_klns and cal_d > numba.float32(0):
                # KLNS dr/dθ pixel-shift (matches Metal kernel exactly)
                theta = _math.acos(max(numba.float32(-1), min(numba.float32(1), zr)))
                sin_th = _math.sqrt(max(numba.float32(0), numba.float32(1) - zr * zr))
                safe_sin = numba.float32(1) if sin_th < numba.float32(1e-7) else sin_th
                th2 = theta * theta
                r_fish = theta * (klns_c0 + th2 * (klns_c1 + th2 * (klns_c2 + th2 * (klns_c3 + th2 * klns_c4))))
                phi_x = numba.float32(0) if sin_th < numba.float32(1e-7) else xr / safe_sin
                phi_y = numba.float32(0) if sin_th < numba.float32(1e-7) else yr / safe_sin
                sx = cx_s + r_fish * phi_x
                sy = cy_s - r_fish * phi_y
                dr_dth = klns_c0 + th2 * (numba.float32(3) * klns_c1 + th2 * (numba.float32(5) * klns_c2 + th2 * (numba.float32(7) * klns_c3 + numba.float32(9) * klns_c4 * th2)))
                t1 = (sy - cy_s) / cal_d * srot_s
                dsx = yaw_coeff * t1 * dr_dth
                dsy = pitch_coeff * t1 * dr_dth
                dsx += roll_coeff * t1 * (sy - cy_s)
                dsy += -roll_coeff * t1 * (sx - cx_s)
                sx_n = sx + dsx
                sy_n = sy + dsy
                # KLNS inverse (Newton, 2 iterations)
                dx = sx_n - cx_s
                dy = -(sy_n - cy_s)
                r_new = _math.sqrt(dx * dx + dy * dy)
                th_n = theta
                for _it in range(2):
                    t2n = th_n * th_n
                    r_est = th_n * (klns_c0 + t2n * (klns_c1 + t2n * (klns_c2 + t2n * (klns_c3 + t2n * klns_c4))))
                    dr_dt = klns_c0 + t2n * (numba.float32(3) * klns_c1 + t2n * (numba.float32(5) * klns_c2 + t2n * (numba.float32(7) * klns_c3 + numba.float32(9) * klns_c4 * t2n)))
                    if abs(dr_dt) > numba.float32(1e-6):
                        th_n += (r_new - r_est) / dr_dt
                    th_n = max(numba.float32(0), min(numba.float32(3.11), th_n))
                safe_r = numba.float32(1) if r_new < numba.float32(1e-6) else r_new
                np_x = numba.float32(0) if r_new < numba.float32(1e-6) else dx / safe_r
                np_y = numba.float32(0) if r_new < numba.float32(1e-6) else dy / safe_r
                stn = _math.sin(th_n)
                xn = stn * np_x
                yn = stn * np_y
                zn = _math.cos(th_n)
            else:
                # 3D rotation fallback
                t = t_offset[i]
                ya = yaw_coeff * t
                pa = pitch_coeff * t
                ra = roll_coeff * t
                xn = xr + ya * zr - ra * yr
                yn = yr + ra * xr - pa * zr
                zn = zr - ya * xr + pa * yr

            # Step 3: Apply IORI (sensor → cross space)
            if has_Rc:
                xc = Rc00 * xn + Rc01 * yn + Rc02 * zn
                yc = Rc10 * xn + Rc11 * yn + Rc12 * zn
                zc = Rc20 * xn + Rc21 * yn + Rc22 * zn
                xn = xc; yn = yc; zn = zc

            # Step 4: Map to EAC cross
            x = xn; y = yn; z = zn
            ax = abs(x); ay = abs(y)
            mx_out[i] = numba.float32(-1.0)
            my_out[i] = numba.float32(-1.0)

            if z > 0 and ax <= z and ay <= z:
                u = TWO_OVER_PI * _math.atan(x / z) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / z)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x > 0 and z <= x and ay <= x:
                u = TWO_OVER_PI * _math.atan(-z / x) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / x)
                fc = u * numba.float32(1920.0)
                if fc < numba.float32(0.0):
                    fc = numba.float32(0.0)
                elif fc > numba.float32(1007.0):
                    fc = numba.float32(1007.0)
                mx_out[i] = numba.float32(2928.0) + fc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x < 0 and z <= ax and ay <= ax:
                u = TWO_OVER_PI * _math.atan(z / ax) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / ax)
                pc = u * numba.float32(1920.0) - numba.float32(912.0)
                if pc < numba.float32(0.0):
                    pc = numba.float32(0.0)
                elif pc > numba.float32(1007.0):
                    pc = numba.float32(1007.0)
                mx_out[i] = pc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif y > 0 and ax <= y and z <= y:
                u = TWO_OVER_PI * _math.atan(x / y) + numba.float32(0.5)
                v = TWO_OVER_PI * _math.atan(z / y) + numba.float32(0.5)
                pr = v * numba.float32(1920.0) - numba.float32(912.0)
                if pr < numba.float32(0.0):
                    pr = numba.float32(0.0)
                elif pr > numba.float32(1007.0):
                    pr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = pr
            elif y < 0 and ax <= ay and z <= ay:
                u = TWO_OVER_PI * _math.atan(x / ay) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(z / ay)
                fr = v * numba.float32(1920.0)
                if fr < numba.float32(0.0):
                    fr = numba.float32(0.0)
                elif fr > numba.float32(1007.0):
                    fr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(2928.0) + fr

            if mx_out[i] > numba.float32(3935.0):
                mx_out[i] = numba.float32(3935.0)
            if my_out[i] > numba.float32(3935.0):
                my_out[i] = numba.float32(3935.0)

    @njit(cache=True, parallel=True)
    def _nb_cross_remap_rot(xyz_x, xyz_y, xyz_z,
                            R00, R01, R02, R10, R11, R12, R20, R21, R22,
                            mx_out, my_out, n):
        """Numba JIT: rotation only → cross maps (no RS). For non-RS path."""
        TWO_OVER_PI = numba.float32(2.0 / _math.pi)
        for i in prange(n):
            ox = xyz_x[i]; oy = xyz_y[i]; oz = xyz_z[i]
            x = R00 * ox + R01 * oy + R02 * oz
            y = R10 * ox + R11 * oy + R12 * oz
            z = R20 * ox + R21 * oy + R22 * oz
            ax = abs(x); ay = abs(y)
            mx_out[i] = numba.float32(-1.0)
            my_out[i] = numba.float32(-1.0)

            if z > 0 and ax <= z and ay <= z:
                u = TWO_OVER_PI * _math.atan(x / z) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / z)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x > 0 and z <= x and ay <= x:
                u = TWO_OVER_PI * _math.atan(-z / x) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / x)
                fc = u * numba.float32(1920.0)
                if fc < numba.float32(0.0):
                    fc = numba.float32(0.0)
                elif fc > numba.float32(1007.0):
                    fc = numba.float32(1007.0)
                mx_out[i] = numba.float32(2928.0) + fc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif x < 0 and z <= ax and ay <= ax:
                u = TWO_OVER_PI * _math.atan(z / ax) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(y / ax)
                pc = u * numba.float32(1920.0) - numba.float32(912.0)
                if pc < numba.float32(0.0):
                    pc = numba.float32(0.0)
                elif pc > numba.float32(1007.0):
                    pc = numba.float32(1007.0)
                mx_out[i] = pc
                my_out[i] = numba.float32(1008.0) + v * numba.float32(1920.0)
            elif y > 0 and ax <= y and z <= y:
                u = TWO_OVER_PI * _math.atan(x / y) + numba.float32(0.5)
                v = TWO_OVER_PI * _math.atan(z / y) + numba.float32(0.5)
                pr = v * numba.float32(1920.0) - numba.float32(912.0)
                if pr < numba.float32(0.0):
                    pr = numba.float32(0.0)
                elif pr > numba.float32(1007.0):
                    pr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = pr
            elif y < 0 and ax <= ay and z <= ay:
                u = TWO_OVER_PI * _math.atan(x / ay) + numba.float32(0.5)
                v = numba.float32(0.5) - TWO_OVER_PI * _math.atan(z / ay)
                fr = v * numba.float32(1920.0)
                if fr < numba.float32(0.0):
                    fr = numba.float32(0.0)
                elif fr > numba.float32(1007.0):
                    fr = numba.float32(1007.0)
                mx_out[i] = numba.float32(1008.0) + u * numba.float32(1920.0)
                my_out[i] = numba.float32(2928.0) + fr

            if mx_out[i] > numba.float32(3935.0):
                mx_out[i] = numba.float32(3935.0)
            if my_out[i] > numba.float32(3935.0):
                my_out[i] = numba.float32(3935.0)

    @njit(cache=True, parallel=True)
    def _nb_equirect_remap_rs(xyz_x, xyz_y, xyz_z,
                              R00, R01, R02, R10, R11, R12, R20, R21, R22,
                              t_offset,
                              yaw_coeff, pitch_coeff, roll_coeff,
                              mx_out, my_out, n,
                              half_w_f, height_f, inv_pi,
                              klns_c0=numba.float32(0), klns_c1=numba.float32(0),
                              klns_c2=numba.float32(0), klns_c3=numba.float32(0),
                              klns_c4=numba.float32(0),
                              cx_s=numba.float32(0), cy_s=numba.float32(0),
                              cal_d=numba.float32(0), srot_s=numba.float32(0),
                              use_klns=False):
        """Numba JIT: fused RS pipeline — R×dir → KLNS RS → half-equirect maps.
        When use_klns=True, uses KLNS dr/dθ pixel-shift (matches Metal kernel).
        When False, uses 3D rotation fallback."""
        mx_max = half_w_f - numba.float32(1.0)
        my_max = height_f - numba.float32(1.0)
        for i in prange(n):
            ox = xyz_x[i]; oy = xyz_y[i]; oz = xyz_z[i]
            # Step 1: R × direction
            xr = R00 * ox + R01 * oy + R02 * oz
            yr = R10 * ox + R11 * oy + R12 * oz
            zr = R20 * ox + R21 * oy + R22 * oz
            # Step 2: RS correction
            xn = xr; yn = yr; zn = zr
            if use_klns and cal_d > numba.float32(0):
                # KLNS dr/dθ pixel-shift
                theta = _math.acos(max(numba.float32(-1), min(numba.float32(1), zr)))
                sin_th = _math.sqrt(max(numba.float32(0), numba.float32(1) - zr * zr))
                safe_sin = numba.float32(1) if sin_th < numba.float32(1e-7) else sin_th
                th2 = theta * theta
                r_fish = theta * (klns_c0 + th2 * (klns_c1 + th2 * (klns_c2 + th2 * (klns_c3 + th2 * klns_c4))))
                phi_x = numba.float32(0) if sin_th < numba.float32(1e-7) else xr / safe_sin
                phi_y = numba.float32(0) if sin_th < numba.float32(1e-7) else yr / safe_sin
                sx = cx_s + r_fish * phi_x
                sy = cy_s - r_fish * phi_y
                dr_dth = klns_c0 + th2 * (numba.float32(3) * klns_c1 + th2 * (numba.float32(5) * klns_c2 + th2 * (numba.float32(7) * klns_c3 + numba.float32(9) * klns_c4 * th2)))
                t1 = (sy - cy_s) / cal_d * srot_s
                dsx = yaw_coeff * t1 * dr_dth
                dsy = pitch_coeff * t1 * dr_dth
                dsx += roll_coeff * t1 * (sy - cy_s)
                dsy += -roll_coeff * t1 * (sx - cx_s)
                sx_n = sx + dsx; sy_n = sy + dsy
                dx = sx_n - cx_s; dy = -(sy_n - cy_s)
                r_new = _math.sqrt(dx * dx + dy * dy)
                th_n = theta
                for _it in range(2):
                    t2n = th_n * th_n
                    r_est = th_n * (klns_c0 + t2n * (klns_c1 + t2n * (klns_c2 + t2n * (klns_c3 + t2n * klns_c4))))
                    dr_dt = klns_c0 + t2n * (numba.float32(3) * klns_c1 + t2n * (numba.float32(5) * klns_c2 + t2n * (numba.float32(7) * klns_c3 + numba.float32(9) * klns_c4 * t2n)))
                    if abs(dr_dt) > numba.float32(1e-6):
                        th_n += (r_new - r_est) / dr_dt
                    th_n = max(numba.float32(0), min(numba.float32(3.11), th_n))
                safe_r = numba.float32(1) if r_new < numba.float32(1e-6) else r_new
                np_x = numba.float32(0) if r_new < numba.float32(1e-6) else dx / safe_r
                np_y = numba.float32(0) if r_new < numba.float32(1e-6) else dy / safe_r
                stn = _math.sin(th_n)
                xn = stn * np_x; yn = stn * np_y; zn = _math.cos(th_n)
            else:
                t = t_offset[i]
                ya = yaw_coeff * t; pa = pitch_coeff * t; ra = roll_coeff * t
                xn = xr + ya * zr - ra * yr
                yn = yr + ra * xr - pa * zr
                zn = zr - ya * xr + pa * yr
            # Step 3: direction → equirect pixel coords
            lat_n = _math.asin(max(numba.float32(-1.0), min(numba.float32(1.0), yn)))
            lon_n = _math.atan2(xn, zn)
            px = (lon_n * inv_pi + numba.float32(0.5)) * half_w_f
            py = (numba.float32(0.5) - lat_n * inv_pi) * height_f
            if px < numba.float32(0.0):
                px = numba.float32(0.0)
            elif px > mx_max:
                px = mx_max
            if py < numba.float32(0.0):
                py = numba.float32(0.0)
            elif py > my_max:
                py = my_max
            mx_out[i] = px
            my_out[i] = py

    @njit(cache=True, parallel=True)
    def _nb_equirect_remap_rot(xyz_x, xyz_y, xyz_z,
                               R00, R01, R02, R10, R11, R12, R20, R21, R22,
                               mx_out, my_out, n,
                               half_w_f, height_f, inv_pi):
        """Numba JIT: rotation only → half-equirect maps (no RS). For non-360 path."""
        mx_max = half_w_f - numba.float32(1.0)
        my_max = height_f - numba.float32(1.0)
        for i in prange(n):
            ox = xyz_x[i]; oy = xyz_y[i]; oz = xyz_z[i]
            x = R00 * ox + R01 * oy + R02 * oz
            y = R10 * ox + R11 * oy + R12 * oz
            z = R20 * ox + R21 * oy + R22 * oz
            lat_n = _math.asin(max(numba.float32(-1.0), min(numba.float32(1.0), y)))
            lon_n = _math.atan2(x, z)
            px = (lon_n * inv_pi + numba.float32(0.5)) * half_w_f
            py = (numba.float32(0.5) - lat_n * inv_pi) * height_f
            if px < numba.float32(0.0):
                px = numba.float32(0.0)
            elif px > mx_max:
                px = mx_max
            if py < numba.float32(0.0):
                py = numba.float32(0.0)
            elif py > my_max:
                py = my_max
            mx_out[i] = px
            my_out[i] = py

    @njit(parallel=True)
    def _nb_apply_lut_3d(frame, lut, out, lut_size, pix_max=255):
        """Numba JIT: per-pixel trilinear 3D LUT interpolation.
        frame: (H, W, 3) uint8 BGR. lut: (S, S, S, 3) float32 RGB.
        out: (H, W, 3) uint8 BGR. 33³ LUT fits in L2 cache."""
        h = frame.shape[0]
        w = frame.shape[1]
        scale = numba.float32(lut_size - 1) / numba.float32(pix_max)

        for i in prange(h):
            for j in range(w):
                b_val = numba.float32(frame[i, j, 0]) * scale
                g_val = numba.float32(frame[i, j, 1]) * scale
                r_val = numba.float32(frame[i, j, 2]) * scale

                b0 = int(b_val); g0 = int(g_val); r0 = int(r_val)
                b1 = min(b0 + 1, lut_size - 1)
                g1 = min(g0 + 1, lut_size - 1)
                r1 = min(r0 + 1, lut_size - 1)

                fb = b_val - numba.float32(b0)
                fg = g_val - numba.float32(g0)
                fr = r_val - numba.float32(r0)

                for c in range(3):
                    c000 = lut[b0, g0, r0, c]; c001 = lut[b0, g0, r1, c]
                    c010 = lut[b0, g1, r0, c]; c011 = lut[b0, g1, r1, c]
                    c100 = lut[b1, g0, r0, c]; c101 = lut[b1, g0, r1, c]
                    c110 = lut[b1, g1, r0, c]; c111 = lut[b1, g1, r1, c]

                    c00 = c000 + (c001 - c000) * fr
                    c01 = c010 + (c011 - c010) * fr
                    c10 = c100 + (c101 - c100) * fr
                    c11 = c110 + (c111 - c110) * fr
                    c0_val = c00 + (c01 - c00) * fg
                    c1_val = c10 + (c11 - c10) * fg
                    val = c0_val + (c1_val - c0_val) * fb

                    out_c = 2 - c  # RGB→BGR
                    val_out = int(val * numba.float32(pix_max) + numba.float32(0.5))
                    if val_out < 0: val_out = 0
                    if val_out > pix_max: val_out = pix_max
                    out[i, j, out_c] = val_out

    # Warm up JIT on first import (compile with small dummy data)
    # Must be synchronous — numba's default workqueue layer is not thread-safe
    def _warmup_numba_kernels():
        """Trigger JIT compilation with small arrays so first render frame isn't slow."""
        n = 16
        d = np.zeros(n, dtype=np.float32)
        o = np.zeros(n, dtype=np.float32)
        _nb_dirs_to_cross_maps(d, d, d, o, o, 4, 4)
        _nb_cross_remap_rs(d, d, d,
                           np.float32(1), np.float32(0), np.float32(0),
                           np.float32(0), np.float32(1), np.float32(0),
                           np.float32(0), np.float32(0), np.float32(1),
                           d,
                           np.float32(0), np.float32(0), np.float32(0),
                           False,
                           np.float32(1), np.float32(0), np.float32(0),
                           np.float32(0), np.float32(1), np.float32(0),
                           np.float32(0), np.float32(0), np.float32(1),
                           o, o, n,
                           np.float32(1), np.float32(0), np.float32(0),
                           np.float32(0), np.float32(0),
                           np.float32(0), np.float32(0),
                           np.float32(1), np.float32(0.015),
                           False)
        _nb_cross_remap_rot(d, d, d,
                            np.float32(1), np.float32(0), np.float32(0),
                            np.float32(0), np.float32(1), np.float32(0),
                            np.float32(0), np.float32(0), np.float32(1),
                            o, o, n)
        # Warmup LUT kernel
        f4 = np.zeros((4, 4, 3), dtype=np.uint8)
        l2 = np.zeros((2, 2, 2, 3), dtype=np.float32)
        o4 = np.zeros_like(f4)
        _nb_apply_lut_3d(f4, l2, o4, 2, 255)

    @njit(parallel=True)
    def _nb_apply_1d_lut(src_flat, lut, dst_flat):
        """Numba parallel 1D LUT: 33x faster than numpy fancy indexing for uint16."""
        for i in prange(len(src_flat)):
            dst_flat[i] = lut[src_flat[i]]

    _warmup_numba_kernels()
    print("  ✓ Numba kernels compiled")

# ── Numba CUDA GPU kernels (NVIDIA — Windows/Linux) ─────────────────────
if HAS_NUMBA_CUDA:
    from numba import cuda as _cuda
    import math as _cuda_math

    @_cuda.jit
    def _cuda_cross_remap_rs_bilinear(xyz_x, xyz_y, xyz_z,
                                       R, t_offset, rs_coeffs,
                                       Rc, has_iori,
                                       cross_img, cross_w, cross_h,
                                       out_bgr, n, pix_max):
        """CUDA kernel: fused R×dir → RS → IORI → EAC face → bilinear sample.
        out_bgr: flat array (n*3), interleaved BGR output. pix_max: 255 for uint8, 65535 for uint16.
        cross_img is flat array (H*W*3), row-major BGR."""
        TWO_OVER_PI = 0.6366197723675814  # 2/pi
        idx = _cuda.grid(1)
        if idx >= n:
            return

        # Step 1: R × direction
        ox = xyz_x[idx]; oy = xyz_y[idx]; oz = xyz_z[idx]
        xr = R[0]*ox + R[1]*oy + R[2]*oz
        yr = R[3]*ox + R[4]*oy + R[5]*oz
        zr = R[6]*ox + R[7]*oy + R[8]*oz

        # Step 2: RS small-angle rotation
        t = t_offset[idx]
        ya = rs_coeffs[0] * t
        pa = rs_coeffs[1] * t
        ra = rs_coeffs[2] * t
        xn = xr + ya*zr - ra*yr
        yn = yr + ra*xr - pa*zr
        zn = zr - ya*xr + pa*yr

        # Step 3: IORI (optional)
        if has_iori:
            xc = Rc[0]*xn + Rc[1]*yn + Rc[2]*zn
            yc = Rc[3]*xn + Rc[4]*yn + Rc[5]*zn
            zc = Rc[6]*xn + Rc[7]*yn + Rc[8]*zn
            xn = xc; yn = yc; zn = zc

        # Step 4: Map to EAC cross coords
        x = xn; y = yn; z = zn
        ax = abs(x); ay = abs(y)
        mx = -1.0; my = -1.0

        if z > 0 and ax <= z and ay <= z:
            u = TWO_OVER_PI * _cuda_math.atan(x / z) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / z)
            mx = 1008.0 + u * 1920.0
            my = 1008.0 + v * 1920.0
        elif x > 0 and z <= x and ay <= x:
            u = TWO_OVER_PI * _cuda_math.atan(-z / x) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / x)
            fc = u * 1920.0
            if fc < 0.0: fc = 0.0
            elif fc > 1007.0: fc = 1007.0
            mx = 2928.0 + fc
            my = 1008.0 + v * 1920.0
        elif x < 0 and z <= ax and ay <= ax:
            u = TWO_OVER_PI * _cuda_math.atan(z / ax) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / ax)
            pc = u * 1920.0 - 912.0
            if pc < 0.0: pc = 0.0
            elif pc > 1007.0: pc = 1007.0
            mx = pc
            my = 1008.0 + v * 1920.0
        elif y > 0 and ax <= y and z <= y:
            u = TWO_OVER_PI * _cuda_math.atan(x / y) + 0.5
            v = TWO_OVER_PI * _cuda_math.atan(z / y) + 0.5
            pr = v * 1920.0 - 912.0
            if pr < 0.0: pr = 0.0
            elif pr > 1007.0: pr = 1007.0
            mx = 1008.0 + u * 1920.0
            my = pr
        elif y < 0 and ax <= ay and z <= ay:
            u = TWO_OVER_PI * _cuda_math.atan(x / ay) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(z / ay)
            fr = v * 1920.0
            if fr < 0.0: fr = 0.0
            elif fr > 1007.0: fr = 1007.0
            mx = 1008.0 + u * 1920.0
            my = 2928.0 + fr

        if mx > 3935.0: mx = 3935.0
        if my > 3935.0: my = 3935.0

        # Step 5: Bilinear sampling from cross image (BGR uint8)
        obase = idx * 3
        if mx < 0 or my < 0:
            out_bgr[obase] = 0; out_bgr[obase + 1] = 0; out_bgr[obase + 2] = 0
            return

        x0 = int(mx); y0 = int(my)
        x1 = min(x0 + 1, cross_w - 1)
        y1 = min(y0 + 1, cross_h - 1)
        fx = mx - x0; fy = my - y0
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = fx * (1.0 - fy)
        w10 = (1.0 - fx) * fy
        w11 = fx * fy

        # BGR layout: idx = (row * cross_w + col) * 3 + channel
        base00 = (y0 * cross_w + x0) * 3
        base01 = (y0 * cross_w + x1) * 3
        base10 = (y1 * cross_w + x0) * 3
        base11 = (y1 * cross_w + x1) * 3

        for c in range(3):
            val = (w00 * cross_img[base00 + c] +
                   w01 * cross_img[base01 + c] +
                   w10 * cross_img[base10 + c] +
                   w11 * cross_img[base11 + c])
            v_int = int(val + 0.5)
            if v_int > pix_max: v_int = pix_max
            out_bgr[obase + c] = v_int

    @_cuda.jit
    def _cuda_cross_remap_rot_bilinear(xyz_x, xyz_y, xyz_z,
                                        R,
                                        cross_img, cross_w, cross_h,
                                        out_bgr, n, pix_max):
        """CUDA kernel: rotation-only path (no RS, no IORI) + bilinear sample."""
        TWO_OVER_PI = 0.6366197723675814
        idx = _cuda.grid(1)
        if idx >= n:
            return

        ox = xyz_x[idx]; oy = xyz_y[idx]; oz = xyz_z[idx]
        x = R[0]*ox + R[1]*oy + R[2]*oz
        y = R[3]*ox + R[4]*oy + R[5]*oz
        z = R[6]*ox + R[7]*oy + R[8]*oz
        ax = abs(x); ay = abs(y)
        mx = -1.0; my = -1.0

        if z > 0 and ax <= z and ay <= z:
            u = TWO_OVER_PI * _cuda_math.atan(x / z) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / z)
            mx = 1008.0 + u * 1920.0
            my = 1008.0 + v * 1920.0
        elif x > 0 and z <= x and ay <= x:
            u = TWO_OVER_PI * _cuda_math.atan(-z / x) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / x)
            fc = u * 1920.0
            if fc < 0.0: fc = 0.0
            elif fc > 1007.0: fc = 1007.0
            mx = 2928.0 + fc
            my = 1008.0 + v * 1920.0
        elif x < 0 and z <= ax and ay <= ax:
            u = TWO_OVER_PI * _cuda_math.atan(z / ax) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(y / ax)
            pc = u * 1920.0 - 912.0
            if pc < 0.0: pc = 0.0
            elif pc > 1007.0: pc = 1007.0
            mx = pc
            my = 1008.0 + v * 1920.0
        elif y > 0 and ax <= y and z <= y:
            u = TWO_OVER_PI * _cuda_math.atan(x / y) + 0.5
            v = TWO_OVER_PI * _cuda_math.atan(z / y) + 0.5
            pr = v * 1920.0 - 912.0
            if pr < 0.0: pr = 0.0
            elif pr > 1007.0: pr = 1007.0
            mx = 1008.0 + u * 1920.0
            my = pr
        elif y < 0 and ax <= ay and z <= ay:
            u = TWO_OVER_PI * _cuda_math.atan(x / ay) + 0.5
            v = 0.5 - TWO_OVER_PI * _cuda_math.atan(z / ay)
            fr = v * 1920.0
            if fr < 0.0: fr = 0.0
            elif fr > 1007.0: fr = 1007.0
            mx = 1008.0 + u * 1920.0
            my = 2928.0 + fr

        if mx > 3935.0: mx = 3935.0
        if my > 3935.0: my = 3935.0

        obase = idx * 3
        if mx < 0 or my < 0:
            out_bgr[obase] = 0; out_bgr[obase + 1] = 0; out_bgr[obase + 2] = 0
            return

        x0 = int(mx); y0 = int(my)
        x1 = min(x0 + 1, cross_w - 1)
        y1 = min(y0 + 1, cross_h - 1)
        fx = mx - x0; fy = my - y0
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = fx * (1.0 - fy)
        w10 = (1.0 - fx) * fy
        w11 = fx * fy

        base00 = (y0 * cross_w + x0) * 3
        base01 = (y0 * cross_w + x1) * 3
        base10 = (y1 * cross_w + x0) * 3
        base11 = (y1 * cross_w + x1) * 3

        for c in range(3):
            val = (w00 * cross_img[base00 + c] +
                   w01 * cross_img[base01 + c] +
                   w10 * cross_img[base10 + c] +
                   w11 * cross_img[base11 + c])
            v_int = int(val + 0.5)
            if v_int > pix_max: v_int = pix_max
            out_bgr[obase + c] = v_int

    # ── CUDA equirect kernels (for non-360 SBS input) ──────────────────
    @_cuda.jit
    def _cuda_equirect_remap_rs_bilinear(xyz_x, xyz_y, xyz_z,
                                          R, t_offset, rs_coeffs,
                                          src_img, src_w, src_h,
                                          out_bgr, n, pix_max):
        """CUDA kernel: fused R×dir → RS rotation → equirect lookup → bilinear sample.
        For non-360 half-equirect source images. Same RS math as cross kernel."""
        INV_PI = 0.3183098861837907
        idx = _cuda.grid(1)
        if idx >= n:
            return

        ox = xyz_x[idx]; oy = xyz_y[idx]; oz = xyz_z[idx]
        xr = R[0]*ox + R[1]*oy + R[2]*oz
        yr = R[3]*ox + R[4]*oy + R[5]*oz
        zr = R[6]*ox + R[7]*oy + R[8]*oz

        t = t_offset[idx]
        ya = rs_coeffs[0] * t
        pa = rs_coeffs[1] * t
        ra = rs_coeffs[2] * t
        xn = xr + ya*zr - ra*yr
        yn = yr + ra*xr - pa*zr
        zn = zr - ya*xr + pa*yr

        lat = _cuda_math.asin(max(-1.0, min(1.0, yn)))
        lon = _cuda_math.atan2(xn, zn)
        mx = (lon * INV_PI + 0.5) * src_w
        my = (0.5 - lat * INV_PI) * src_h

        mx_max = src_w - 1.0
        my_max = src_h - 1.0
        if mx < 0.0: mx = 0.0
        elif mx > mx_max: mx = mx_max
        if my < 0.0: my = 0.0
        elif my > my_max: my = my_max

        x0 = int(mx); y0 = int(my)
        x1 = min(x0 + 1, int(src_w) - 1)
        y1 = min(y0 + 1, int(src_h) - 1)
        fx = mx - x0; fy = my - y0
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = fx * (1.0 - fy)
        w10 = (1.0 - fx) * fy
        w11 = fx * fy

        sw = int(src_w)
        base00 = (y0 * sw + x0) * 3
        base01 = (y0 * sw + x1) * 3
        base10 = (y1 * sw + x0) * 3
        base11 = (y1 * sw + x1) * 3

        obase = idx * 3
        for c in range(3):
            val = (w00 * src_img[base00 + c] +
                   w01 * src_img[base01 + c] +
                   w10 * src_img[base10 + c] +
                   w11 * src_img[base11 + c])
            v_int = int(val + 0.5)
            if v_int > pix_max: v_int = pix_max
            out_bgr[obase + c] = v_int

    @_cuda.jit
    def _cuda_equirect_remap_rot_bilinear(xyz_x, xyz_y, xyz_z,
                                           R,
                                           src_img, src_w, src_h,
                                           out_bgr, n, pix_max):
        """CUDA kernel: rotation-only equirect remap + bilinear sample (no RS)."""
        INV_PI = 0.3183098861837907
        idx = _cuda.grid(1)
        if idx >= n:
            return

        ox = xyz_x[idx]; oy = xyz_y[idx]; oz = xyz_z[idx]
        x = R[0]*ox + R[1]*oy + R[2]*oz
        y = R[3]*ox + R[4]*oy + R[5]*oz
        z = R[6]*ox + R[7]*oy + R[8]*oz

        lat = _cuda_math.asin(max(-1.0, min(1.0, y)))
        lon = _cuda_math.atan2(x, z)
        mx = (lon * INV_PI + 0.5) * src_w
        my = (0.5 - lat * INV_PI) * src_h

        mx_max = src_w - 1.0
        my_max = src_h - 1.0
        if mx < 0.0: mx = 0.0
        elif mx > mx_max: mx = mx_max
        if my < 0.0: my = 0.0
        elif my > my_max: my = my_max

        x0 = int(mx); y0 = int(my)
        x1 = min(x0 + 1, int(src_w) - 1)
        y1 = min(y0 + 1, int(src_h) - 1)
        fx = mx - x0; fy = my - y0
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = fx * (1.0 - fy)
        w10 = (1.0 - fx) * fy
        w11 = fx * fy

        sw = int(src_w)
        base00 = (y0 * sw + x0) * 3
        base01 = (y0 * sw + x1) * 3
        base10 = (y1 * sw + x0) * 3
        base11 = (y1 * sw + x1) * 3

        obase = idx * 3
        for c in range(3):
            val = (w00 * src_img[base00 + c] +
                   w01 * src_img[base01 + c] +
                   w10 * src_img[base10 + c] +
                   w11 * src_img[base11 + c])
            v_int = int(val + 0.5)
            if v_int > pix_max: v_int = pix_max
            out_bgr[obase + c] = v_int

    @_cuda.jit
    def _cuda_apply_lut_3d_kernel(frame, lut, out, lut_size, n, pix_max):
        """CUDA kernel: trilinear 3D LUT interpolation. pix_max: 255 for uint8, 65535 for uint16."""
        idx = _cuda.grid(1)
        if idx >= n:
            return
        base = idx * 3
        pm_f = float(pix_max)
        scale = (lut_size - 1) / pm_f
        b_val = frame[base] * scale
        g_val = frame[base + 1] * scale
        r_val = frame[base + 2] * scale

        b0 = int(b_val); g0 = int(g_val); r0 = int(r_val)
        b1 = min(b0 + 1, lut_size - 1)
        g1 = min(g0 + 1, lut_size - 1)
        r1 = min(r0 + 1, lut_size - 1)
        fb = b_val - b0; fg = g_val - g0; fr = r_val - r0

        ls = lut_size
        for c in range(3):
            c000 = lut[((b0*ls + g0)*ls + r0)*3 + c]
            c001 = lut[((b0*ls + g0)*ls + r1)*3 + c]
            c010 = lut[((b0*ls + g1)*ls + r0)*3 + c]
            c011 = lut[((b0*ls + g1)*ls + r1)*3 + c]
            c100 = lut[((b1*ls + g0)*ls + r0)*3 + c]
            c101 = lut[((b1*ls + g0)*ls + r1)*3 + c]
            c110 = lut[((b1*ls + g1)*ls + r0)*3 + c]
            c111 = lut[((b1*ls + g1)*ls + r1)*3 + c]

            c00 = c000 + (c001 - c000) * fr
            c01 = c010 + (c011 - c010) * fr
            c10 = c100 + (c101 - c100) * fr
            c11 = c110 + (c111 - c110) * fr
            c0v = c00 + (c01 - c00) * fg
            c1v = c10 + (c11 - c10) * fg
            val = c0v + (c1v - c0v) * fb

            out_c = 2 - c  # RGB→BGR
            v_int = int(val * pm_f + 0.5)
            if v_int < 0: v_int = 0
            if v_int > pix_max: v_int = pix_max
            out[base + out_c] = v_int

    @_cuda.jit
    def _cuda_blur_h_kernel(src, dst, kernel, K, H, W, pix_max):
        """Horizontal separable blur. src/dst: flat uint8 (H*W*3)."""
        idx = _cuda.grid(1)
        if idx >= H * W * 3:
            return
        c = idx % 3
        pix = idx // 3
        x = pix % W
        y = pix // W
        half_k = K // 2
        val = 0.0
        for k in range(K):
            sx = x + k - half_k
            if sx < 0:
                sx = 0
            if sx >= W:
                sx = W - 1
            val += float(src[(y * W + sx) * 3 + c]) * kernel[k]
        v = int(val + 0.5)
        if v < 0: v = 0
        if v > pix_max: v = pix_max
        dst[idx] = v

    @_cuda.jit
    def _cuda_blur_v_usm_kernel(src, h_blurred, lat_weights, out, amount, kernel, K, H, W, pix_max):
        """Vertical blur on h_blurred + fused USM: out = clamp(src + lat*amount*(src - blurred))."""
        idx = _cuda.grid(1)
        if idx >= H * W * 3:
            return
        c = idx % 3
        pix = idx // 3
        x = pix % W
        y = pix // W
        half_k = K // 2
        val = 0.0
        for k in range(K):
            sy = y + k - half_k
            if sy < 0:
                sy = 0
            if sy >= H:
                sy = H - 1
            val += float(h_blurred[(sy * W + x) * 3 + c]) * kernel[k]
        f = float(src[idx])
        w = lat_weights[y] * amount
        result = f + w * (f - val)
        v = int(result + 0.5)
        if v < 0: v = 0
        if v > pix_max: v = pix_max
        out[idx] = v

    @_cuda.jit
    def _cuda_apply_1d_lut_inplace(data, lut, n):
        """Apply 256-entry 1D LUT in-place to flat uint8 array."""
        idx = _cuda.grid(1)
        if idx < n:
            data[idx] = lut[data[idx]]

    @_cuda.jit
    def _cuda_apply_lut_3d_blend_kernel(src, lut, dst, lut_size, intensity, n, pix_max):
        """3D LUT with intensity blend: dst = src + intensity*(lut(src) - src). pix_max: 255 or 65535."""
        idx = _cuda.grid(1)
        if idx >= n:
            return
        base = idx * 3
        pm_f = float(pix_max)
        scale = (lut_size - 1) / pm_f
        b_val = src[base] * scale
        g_val = src[base + 1] * scale
        r_val = src[base + 2] * scale
        b0 = int(b_val); g0 = int(g_val); r0 = int(r_val)
        b1 = min(b0 + 1, lut_size - 1)
        g1 = min(g0 + 1, lut_size - 1)
        r1 = min(r0 + 1, lut_size - 1)
        fb = b_val - b0; fg = g_val - g0; fr = r_val - r0
        ls = lut_size
        for c in range(3):
            c000 = lut[((b0*ls + g0)*ls + r0)*3 + c]
            c001 = lut[((b0*ls + g0)*ls + r1)*3 + c]
            c010 = lut[((b0*ls + g1)*ls + r0)*3 + c]
            c011 = lut[((b0*ls + g1)*ls + r1)*3 + c]
            c100 = lut[((b1*ls + g0)*ls + r0)*3 + c]
            c101 = lut[((b1*ls + g0)*ls + r1)*3 + c]
            c110 = lut[((b1*ls + g1)*ls + r0)*3 + c]
            c111 = lut[((b1*ls + g1)*ls + r1)*3 + c]
            c00 = c000 + (c001 - c000) * fr
            c01 = c010 + (c011 - c010) * fr
            c10 = c100 + (c101 - c100) * fr
            c11 = c110 + (c111 - c110) * fr
            c0v = c00 + (c01 - c00) * fg
            c1v = c10 + (c11 - c10) * fg
            lut_val = (c0v + (c1v - c0v) * fb) * pm_f
            out_c = 2 - c  # RGB→BGR
            orig = float(src[base + out_c])
            blended = orig + (lut_val - orig) * intensity
            v = int(blended + 0.5)
            if v < 0: v = 0
            if v > pix_max: v = pix_max
            dst[base + out_c] = v

    # ── CUDA wrapper functions ────────────────────────────────────────────
    _CUDA_BLOCK = 256

    # Persistent device buffers — allocated once, reused every frame.
    # Dual cross/out buffers for left+right eye (avoids sync between eyes).
    _cuda_persistent = {
        'd_cross_A': None,     # device buffer for right-eye cross image
        'd_cross_B': None,     # device buffer for left-eye cross image
        'd_out_A': None,       # device buffer for right-eye output BGR
        'd_out_B': None,       # device buffer for left-eye output BGR
        'd_R_A': None,         # rotation matrix for right eye (9 float32)
        'd_R_B': None,         # rotation matrix for left eye (9 float32)
        'd_rs': None,          # RS coefficients (3 float32)
        'd_Rc': None,          # IORI rotation (9 float32)
        'd_lut': None,         # 3D LUT (persistent after first upload)
        'd_lut_frame': None,   # LUT input frame buffer
        'd_lut_out': None,     # LUT output frame buffer
        # Fused pipeline: per-eye auxiliary buffers (avoid GPU↔CPU round-trips)
        'd_aux_A': None,       # scratch buffer for right eye (same size as d_out_A)
        'd_aux_B': None,       # scratch buffer for left eye
        'd_final_A': None,     # sharpen output for right eye
        'd_final_B': None,     # sharpen output for left eye
        'd_1d_lut': None,      # 256-byte 1D color LUT on GPU
        'd_sharpen_kernel': None,  # Gaussian 1D kernel for fused sharpen
        'd_sharpen_lat': None,     # latitude weights for fused sharpen
        'cross_size': 0,
        'out_size': 0,
        'lut_size': 0,
        'lut_frame_size': 0,
        'stream_A': None,      # CUDA stream for right eye
        'stream_B': None,      # CUDA stream for left eye
    }

    def _cuda_ensure_buffers(cross_np, n_pixels, pix_dtype=np.uint8):
        """Ensure persistent device buffers are allocated and correctly sized.
        pix_dtype: np.uint8 for 8-bit, np.uint16 for 10-bit pipeline."""
        p = _cuda_persistent
        cross_elems = cross_np.size  # H*W*3 (element count, not bytes)
        out_elems = n_pixels * 3
        # Reallocate if size OR dtype changed
        _need_cross = (p['cross_size'] != cross_elems or
                       (p['d_cross_A'] is not None and p['d_cross_A'].dtype != pix_dtype))
        _need_out = (p['out_size'] != out_elems or
                     (p['d_out_A'] is not None and p['d_out_A'].dtype != pix_dtype))
        if _need_cross:
            p['d_cross_A'] = _cuda.device_array(cross_elems, dtype=pix_dtype)
            p['d_cross_B'] = _cuda.device_array(cross_elems, dtype=pix_dtype)
            p['cross_size'] = cross_elems
        if _need_out:
            p['d_out_A'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['d_out_B'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['d_aux_A'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['d_aux_B'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['d_final_A'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['d_final_B'] = _cuda.device_array(out_elems, dtype=pix_dtype)
            p['out_size'] = out_elems
        if p['d_R_A'] is None:
            p['d_R_A'] = _cuda.device_array(9, dtype=np.float32)
            p['d_R_B'] = _cuda.device_array(9, dtype=np.float32)
            p['d_rs'] = _cuda.device_array(3, dtype=np.float32)
            p['d_Rc'] = _cuda.device_array(9, dtype=np.float32)
            p['stream_A'] = _cuda.stream()
            p['stream_B'] = _cuda.stream()

    # Pre-allocated flat buffers for ravel() avoidance
    _cuda_eye3x3_flat = np.eye(3, dtype=np.float32).ravel()

    def _cuda_process_both_eyes(crossA_np, crossB_np, d_xyz_x, d_xyz_y, d_xyz_z,
                                R_right_np, R_left_np,
                                d_t_offset_right, d_t_offset_left,
                                rs_coeffs_np, R_cross_np, has_iori,
                                n_pixels, out_h, out_w, result_buf, pix_dtype=np.uint8):
        """Process BOTH eyes via CUDA with overlapping streams, writing directly into result_buf.
        pix_dtype: np.uint8 for 8-bit, np.uint16 for 10-bit pipeline."""
        p = _cuda_persistent
        _pix_max = np.int32(255 if pix_dtype == np.uint8 else 65535)
        _cuda_ensure_buffers(crossA_np, n_pixels, pix_dtype)

        cross_h_px, cross_w_px = crossA_np.shape[:2]
        half_w = out_w

        # Upload both crosses to GPU (pre-allocated buffers should be C-contiguous)
        p['d_cross_A'].copy_to_device(crossA_np.ravel() if crossA_np.flags['C_CONTIGUOUS'] else np.ascontiguousarray(crossA_np).ravel())
        p['d_cross_B'].copy_to_device(crossB_np.ravel() if crossB_np.flags['C_CONTIGUOUS'] else np.ascontiguousarray(crossB_np).ravel())

        # Upload rotation matrices
        p['d_R_A'].copy_to_device(R_right_np.ravel().astype(np.float32))
        p['d_R_B'].copy_to_device(R_left_np.ravel().astype(np.float32))

        grid = (n_pixels + _CUDA_BLOCK - 1) // _CUDA_BLOCK

        # RIGHT EYE kernel launch
        if rs_coeffs_np is not None:
            p['d_rs'].copy_to_device(rs_coeffs_np.astype(np.float32))
            if has_iori:
                p['d_Rc'].copy_to_device(R_cross_np.ravel().astype(np.float32))
            else:
                p['d_Rc'].copy_to_device(_cuda_eye3x3_flat)
            _cuda_cross_remap_rs_bilinear[grid, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'], d_t_offset_right, p['d_rs'],
                p['d_Rc'], has_iori,
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)
        else:
            _cuda_cross_remap_rot_bilinear[grid, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'],
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)

        # LEFT EYE kernel launch (no RS, rotation only)
        _cuda_cross_remap_rot_bilinear[grid, _CUDA_BLOCK](
            d_xyz_x, d_xyz_y, d_xyz_z,
            p['d_R_B'],
            p['d_cross_B'], cross_w_px, cross_h_px,
            p['d_out_B'], n_pixels, _pix_max)

        # Single sync for both kernels
        _cuda.synchronize()

        # Readback into contiguous temp buffers, then copy into result_buf slices
        # (result_buf[:, half_w:] is non-contiguous due to stride, can't copy_to_host directly)
        if not hasattr(_cuda_process_both_eyes, '_host_A') or _cuda_process_both_eyes._host_A.dtype != pix_dtype:
            _cuda_process_both_eyes._host_A = np.empty(n_pixels * 3, dtype=pix_dtype)
            _cuda_process_both_eyes._host_B = np.empty(n_pixels * 3, dtype=pix_dtype)
        elif _cuda_process_both_eyes._host_A.size != n_pixels * 3:
            _cuda_process_both_eyes._host_A = np.empty(n_pixels * 3, dtype=pix_dtype)
            _cuda_process_both_eyes._host_B = np.empty(n_pixels * 3, dtype=pix_dtype)
        p['d_out_A'].copy_to_host(_cuda_process_both_eyes._host_A)
        p['d_out_B'].copy_to_host(_cuda_process_both_eyes._host_B)
        np.copyto(result_buf[:, half_w:], _cuda_process_both_eyes._host_A.reshape(out_h, half_w, 3))
        np.copyto(result_buf[:, :half_w], _cuda_process_both_eyes._host_B.reshape(out_h, half_w, 3))

    def _cuda_process_eye(cross_np, d_xyz_x, d_xyz_y, d_xyz_z,
                          R_np, d_t_offset, rs_coeffs_np,
                          R_cross_np, has_iori, n_pixels, out_h, out_w,
                          pix_max=255):
        """Process one eye via CUDA (for preview path).
        pix_max: 255 for uint8 pipeline, 65535 for uint16 (Level B preview /
        export). The output dtype follows pix_max; buffer pool is
        auto-reallocated by _cuda_ensure_buffers when dtype changes."""
        p = _cuda_persistent
        _pix_max = np.int32(pix_max)
        _pix_dtype = np.uint16 if pix_max > 256 else np.uint8
        _cuda_ensure_buffers(cross_np, n_pixels, _pix_dtype)

        cross_h_px, cross_w_px = cross_np.shape[:2]
        p['d_cross_A'].copy_to_device(np.ascontiguousarray(cross_np).ravel())
        p['d_R_A'].copy_to_device(R_np.ravel().astype(np.float32))

        grid = (n_pixels + _CUDA_BLOCK - 1) // _CUDA_BLOCK

        if rs_coeffs_np is not None:
            p['d_rs'].copy_to_device(rs_coeffs_np.astype(np.float32))
            if has_iori:
                p['d_Rc'].copy_to_device(R_cross_np.ravel().astype(np.float32))
            else:
                p['d_Rc'].copy_to_device(_cuda_eye3x3_flat)
            _cuda_cross_remap_rs_bilinear[grid, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'], d_t_offset, p['d_rs'],
                p['d_Rc'], has_iori,
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)
        else:
            _cuda_cross_remap_rot_bilinear[grid, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'],
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)

        _cuda.synchronize()
        result = np.empty(n_pixels * 3, dtype=_pix_dtype)
        p['d_out_A'].copy_to_host(result)
        return result.reshape(out_h, out_w, 3)

    def _cuda_apply_lut_3d(frame_np, lut_np, lut_size):
        """Apply 3D LUT via CUDA. frame_np: (H,W,3) uint8/uint16 BGR, lut_np: (S,S,S,3) float32 RGB.
        Uses persistent device buffers."""
        p = _cuda_persistent
        h, w = frame_np.shape[:2]
        n = h * w
        frame_elems = n * 3
        lut_elems = lut_np.size
        pix_dtype = frame_np.dtype

        # Ensure LUT buffer (uploaded once, reused across frames)
        if p['lut_size'] != lut_elems:
            p['d_lut'] = _cuda.to_device(lut_np.ravel().astype(np.float32))
            p['lut_size'] = lut_elems

        # Ensure frame I/O buffers (reallocate if size or dtype changed)
        _need_realloc = (p['lut_frame_size'] != frame_elems or
                         (p.get('d_lut_frame') is not None and p['d_lut_frame'].dtype != pix_dtype))
        if _need_realloc:
            p['d_lut_frame'] = _cuda.device_array(frame_elems, dtype=pix_dtype)
            p['d_lut_out'] = _cuda.device_array(frame_elems, dtype=pix_dtype)
            p['lut_frame_size'] = frame_elems

        p['d_lut_frame'].copy_to_device(frame_np.ravel())
        grid = (n + _CUDA_BLOCK - 1) // _CUDA_BLOCK
        _pix_max = np.int32(65535 if pix_dtype == np.uint16 else 255)
        _cuda_apply_lut_3d_kernel[grid, _CUDA_BLOCK](p['d_lut_frame'], p['d_lut'], p['d_lut_out'], lut_size, n, _pix_max)
        _cuda.synchronize()
        return p['d_lut_out'].copy_to_host().reshape(h, w, 3)

    def _cuda_fused_process(crossA_np, crossB_np, d_xyz_x, d_xyz_y, d_xyz_z,
                            R_right_np, R_left_np,
                            d_t_offset_right, d_t_offset_left,
                            rs_coeffs_np, R_cross_np, has_iori,
                            n_pixels, out_h, out_w, result_buf,
                            color_1d_lut, lut_3d, lut_intensity,
                            sharpen_kernel_1d, sharpen_lat, sharpen_amount, sharpen_ksize,
                            pix_dtype=np.uint8):
        """Fused GPU pipeline: remap → 1D LUT → 3D LUT → sharpen → single download.
        pix_dtype: np.uint8 for 8-bit, np.uint16 for 10-bit pipeline."""
        p = _cuda_persistent
        _pix_max = np.int32(255 if pix_dtype == np.uint8 else 65535)
        _cuda_ensure_buffers(crossA_np, n_pixels, pix_dtype)

        cross_h_px, cross_w_px = crossA_np.shape[:2]
        half_w = out_w
        n_elems = n_pixels * 3

        # Upload crosses (pre-allocated buffers are C-contiguous, ravel is zero-copy view)
        if crossA_np.flags['C_CONTIGUOUS']:
            p['d_cross_A'].copy_to_device(crossA_np.ravel())
        else:
            p['d_cross_A'].copy_to_device(np.ascontiguousarray(crossA_np).ravel())
        if crossB_np.flags['C_CONTIGUOUS']:
            p['d_cross_B'].copy_to_device(crossB_np.ravel())
        else:
            p['d_cross_B'].copy_to_device(np.ascontiguousarray(crossB_np).ravel())

        # Upload rotation matrices
        p['d_R_A'].copy_to_device(R_right_np.ravel().astype(np.float32))
        p['d_R_B'].copy_to_device(R_left_np.ravel().astype(np.float32))

        grid_pix = (n_pixels + _CUDA_BLOCK - 1) // _CUDA_BLOCK
        grid_elem = (n_elems + _CUDA_BLOCK - 1) // _CUDA_BLOCK

        # ── Step 1: Remap both eyes ──
        # Right eye
        if rs_coeffs_np is not None:
            p['d_rs'].copy_to_device(rs_coeffs_np.astype(np.float32))
            if has_iori:
                p['d_Rc'].copy_to_device(R_cross_np.ravel().astype(np.float32))
            else:
                p['d_Rc'].copy_to_device(_cuda_eye3x3_flat)
            _cuda_cross_remap_rs_bilinear[grid_pix, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'], d_t_offset_right, p['d_rs'],
                p['d_Rc'], has_iori,
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)
        else:
            _cuda_cross_remap_rot_bilinear[grid_pix, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'],
                p['d_cross_A'], cross_w_px, cross_h_px,
                p['d_out_A'], n_pixels, _pix_max)
        # Left eye (no RS)
        _cuda_cross_remap_rot_bilinear[grid_pix, _CUDA_BLOCK](
            d_xyz_x, d_xyz_y, d_xyz_z,
            p['d_R_B'],
            p['d_cross_B'], cross_w_px, cross_h_px,
            p['d_out_B'], n_pixels, _pix_max)

        # After remap: d_out_A = right eye, d_out_B = left eye (on GPU)
        # cur_A/cur_B track which device buffer holds the "current" data
        cur_A, aux_A = p['d_out_A'], p['d_aux_A']
        cur_B, aux_B = p['d_out_B'], p['d_aux_B']

        # ── Step 2: 1D color LUT (in-place, ~0.1ms) ──
        if color_1d_lut is not None:
            _lut_len = len(color_1d_lut)
            if p['d_1d_lut'] is None or p['d_1d_lut'].size != _lut_len:
                p['d_1d_lut'] = _cuda.device_array(_lut_len, dtype=color_1d_lut.dtype)
            p['d_1d_lut'].copy_to_device(color_1d_lut)
            _cuda_apply_1d_lut_inplace[grid_elem, _CUDA_BLOCK](cur_A, p['d_1d_lut'], n_elems)
            _cuda_apply_1d_lut_inplace[grid_elem, _CUDA_BLOCK](cur_B, p['d_1d_lut'], n_elems)

        # ── Step 3: 3D LUT at full resolution (no CPU resize!) ──
        if lut_3d is not None and lut_intensity > 0.01:
            lut_size = lut_3d.shape[0]
            lut_elems = lut_3d.size
            if p['lut_size'] != lut_elems:
                p['d_lut'] = _cuda.to_device(lut_3d.ravel().astype(np.float32))
                p['lut_size'] = lut_elems
            if lut_intensity >= 0.99:
                _cuda_apply_lut_3d_kernel[grid_pix, _CUDA_BLOCK](cur_A, p['d_lut'], aux_A, lut_size, n_pixels, _pix_max)
                _cuda_apply_lut_3d_kernel[grid_pix, _CUDA_BLOCK](cur_B, p['d_lut'], aux_B, lut_size, n_pixels, _pix_max)
            else:
                _cuda_apply_lut_3d_blend_kernel[grid_pix, _CUDA_BLOCK](
                    cur_A, p['d_lut'], aux_A, lut_size, np.float32(lut_intensity), n_pixels, _pix_max)
                _cuda_apply_lut_3d_blend_kernel[grid_pix, _CUDA_BLOCK](
                    cur_B, p['d_lut'], aux_B, lut_size, np.float32(lut_intensity), n_pixels, _pix_max)
            cur_A, aux_A = aux_A, cur_A
            cur_B, aux_B = aux_B, cur_B

        # ── Step 4: Equirectangular-aware sharpen (separable blur + USM) ──
        if sharpen_kernel_1d is not None and sharpen_amount > 0.01:
            if p['d_sharpen_kernel'] is None or p['d_sharpen_kernel'].size != len(sharpen_kernel_1d):
                p['d_sharpen_kernel'] = _cuda.to_device(sharpen_kernel_1d)
            if p['d_sharpen_lat'] is None or p['d_sharpen_lat'].size != len(sharpen_lat):
                p['d_sharpen_lat'] = _cuda.to_device(sharpen_lat)
            K = np.int32(sharpen_ksize)
            H = np.int32(out_h)
            W = np.int32(half_w)
            amt = np.float32(sharpen_amount)
            _cuda_blur_h_kernel[grid_elem, _CUDA_BLOCK](
                cur_A, aux_A, p['d_sharpen_kernel'], K, H, W, _pix_max)
            _cuda_blur_v_usm_kernel[grid_elem, _CUDA_BLOCK](
                cur_A, aux_A, p['d_sharpen_lat'],
                p['d_final_A'], amt, p['d_sharpen_kernel'], K, H, W, _pix_max)
            cur_A = p['d_final_A']
            _cuda_blur_h_kernel[grid_elem, _CUDA_BLOCK](
                cur_B, aux_B, p['d_sharpen_kernel'], K, H, W, _pix_max)
            _cuda_blur_v_usm_kernel[grid_elem, _CUDA_BLOCK](
                cur_B, aux_B, p['d_sharpen_lat'],
                p['d_final_B'], amt, p['d_sharpen_kernel'], K, H, W, _pix_max)
            cur_B = p['d_final_B']

        # ── Step 5: Single sync + download ──
        _cuda.synchronize()

        if not hasattr(_cuda_fused_process, '_host_A') or _cuda_fused_process._host_A.dtype != pix_dtype:
            _cuda_fused_process._host_A = np.empty(n_elems, dtype=pix_dtype)
            _cuda_fused_process._host_B = np.empty(n_elems, dtype=pix_dtype)
        elif _cuda_fused_process._host_A.size != n_elems:
            _cuda_fused_process._host_A = np.empty(n_elems, dtype=pix_dtype)
            _cuda_fused_process._host_B = np.empty(n_elems, dtype=pix_dtype)

        cur_A.copy_to_host(_cuda_fused_process._host_A)
        cur_B.copy_to_host(_cuda_fused_process._host_B)
        np.copyto(result_buf[:, half_w:], _cuda_fused_process._host_A[:n_elems].reshape(out_h, half_w, 3))
        np.copyto(result_buf[:, :half_w], _cuda_fused_process._host_B[:n_elems].reshape(out_h, half_w, 3))

    def _cuda_fused_process_eq(srcA_np, srcB_np, d_xyz_x, d_xyz_y, d_xyz_z,
                               R_right_np, R_left_np,
                               d_t_offset_right,
                               rs_coeffs_np,
                               n_pixels, out_h, out_w, result_buf,
                               color_1d_lut, lut_3d, lut_intensity,
                               sharpen_kernel_1d, sharpen_lat, sharpen_amount, sharpen_ksize,
                               pix_dtype=np.uint8):
        """Fused GPU pipeline for non-360 equirect: remap → 1D LUT → 3D LUT → sharpen.
        pix_dtype: np.uint8 for 8-bit, np.uint16 for 10-bit pipeline."""
        p = _cuda_persistent
        _pix_max = np.int32(255 if pix_dtype == np.uint8 else 65535)
        _cuda_ensure_buffers(srcA_np, n_pixels, pix_dtype)

        src_h_px, src_w_px = srcA_np.shape[:2]
        half_w = out_w
        n_elems = n_pixels * 3

        # Upload source half-equirect images (caller provides contiguous buffers)
        if srcA_np.flags['C_CONTIGUOUS']:
            p['d_cross_A'].copy_to_device(srcA_np.ravel())
            p['d_cross_B'].copy_to_device(srcB_np.ravel())
        else:
            p['d_cross_A'].copy_to_device(np.ascontiguousarray(srcA_np).ravel())
            p['d_cross_B'].copy_to_device(np.ascontiguousarray(srcB_np).ravel())

        # Upload rotation matrices
        p['d_R_A'].copy_to_device(R_right_np.ravel().astype(np.float32))
        p['d_R_B'].copy_to_device(R_left_np.ravel().astype(np.float32))

        grid_pix = (n_pixels + _CUDA_BLOCK - 1) // _CUDA_BLOCK
        grid_elem = (n_elems + _CUDA_BLOCK - 1) // _CUDA_BLOCK

        # ── Step 1: Remap both eyes ──
        src_w_f = float(src_w_px)
        src_h_f = float(src_h_px)
        if rs_coeffs_np is not None:
            p['d_rs'].copy_to_device(rs_coeffs_np.astype(np.float32))
            _cuda_equirect_remap_rs_bilinear[grid_pix, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'], d_t_offset_right, p['d_rs'],
                p['d_cross_A'], src_w_f, src_h_f,
                p['d_out_A'], n_pixels, _pix_max)
        else:
            _cuda_equirect_remap_rot_bilinear[grid_pix, _CUDA_BLOCK](
                d_xyz_x, d_xyz_y, d_xyz_z,
                p['d_R_A'],
                p['d_cross_A'], src_w_f, src_h_f,
                p['d_out_A'], n_pixels, _pix_max)
        _cuda_equirect_remap_rot_bilinear[grid_pix, _CUDA_BLOCK](
            d_xyz_x, d_xyz_y, d_xyz_z,
            p['d_R_B'],
            p['d_cross_B'], src_w_f, src_h_f,
            p['d_out_B'], n_pixels, _pix_max)

        cur_A, aux_A = p['d_out_A'], p['d_aux_A']
        cur_B, aux_B = p['d_out_B'], p['d_aux_B']

        # ── Step 2: 1D color LUT ──
        if color_1d_lut is not None:
            _lut_len = len(color_1d_lut)
            if p['d_1d_lut'] is None or p['d_1d_lut'].size != _lut_len:
                p['d_1d_lut'] = _cuda.device_array(_lut_len, dtype=color_1d_lut.dtype)
            p['d_1d_lut'].copy_to_device(color_1d_lut)
            _cuda_apply_1d_lut_inplace[grid_elem, _CUDA_BLOCK](cur_A, p['d_1d_lut'], n_elems)
            _cuda_apply_1d_lut_inplace[grid_elem, _CUDA_BLOCK](cur_B, p['d_1d_lut'], n_elems)

        # ── Step 3: 3D LUT ──
        if lut_3d is not None and lut_intensity > 0.01:
            lut_size = lut_3d.shape[0]
            lut_elems = lut_3d.size
            if p['lut_size'] != lut_elems:
                p['d_lut'] = _cuda.to_device(lut_3d.ravel().astype(np.float32))
                p['lut_size'] = lut_elems
            if lut_intensity >= 0.99:
                _cuda_apply_lut_3d_kernel[grid_pix, _CUDA_BLOCK](cur_A, p['d_lut'], aux_A, lut_size, n_pixels, _pix_max)
                _cuda_apply_lut_3d_kernel[grid_pix, _CUDA_BLOCK](cur_B, p['d_lut'], aux_B, lut_size, n_pixels, _pix_max)
            else:
                _cuda_apply_lut_3d_blend_kernel[grid_pix, _CUDA_BLOCK](
                    cur_A, p['d_lut'], aux_A, lut_size, np.float32(lut_intensity), n_pixels, _pix_max)
                _cuda_apply_lut_3d_blend_kernel[grid_pix, _CUDA_BLOCK](
                    cur_B, p['d_lut'], aux_B, lut_size, np.float32(lut_intensity), n_pixels, _pix_max)
            cur_A, aux_A = aux_A, cur_A
            cur_B, aux_B = aux_B, cur_B

        # ── Step 4: Equirectangular-aware sharpen ──
        if sharpen_kernel_1d is not None and sharpen_amount > 0.01:
            if p['d_sharpen_kernel'] is None or p['d_sharpen_kernel'].size != len(sharpen_kernel_1d):
                p['d_sharpen_kernel'] = _cuda.to_device(sharpen_kernel_1d)
            if p['d_sharpen_lat'] is None or p['d_sharpen_lat'].size != len(sharpen_lat):
                p['d_sharpen_lat'] = _cuda.to_device(sharpen_lat)
            K = np.int32(sharpen_ksize)
            H = np.int32(out_h)
            W = np.int32(half_w)
            amt = np.float32(sharpen_amount)
            _cuda_blur_h_kernel[grid_elem, _CUDA_BLOCK](cur_A, aux_A, p['d_sharpen_kernel'], K, H, W, _pix_max)
            _cuda_blur_v_usm_kernel[grid_elem, _CUDA_BLOCK](
                cur_A, aux_A, p['d_sharpen_lat'], p['d_final_A'], amt, p['d_sharpen_kernel'], K, H, W, _pix_max)
            cur_A = p['d_final_A']
            _cuda_blur_h_kernel[grid_elem, _CUDA_BLOCK](cur_B, aux_B, p['d_sharpen_kernel'], K, H, W, _pix_max)
            _cuda_blur_v_usm_kernel[grid_elem, _CUDA_BLOCK](
                cur_B, aux_B, p['d_sharpen_lat'], p['d_final_B'], amt, p['d_sharpen_kernel'], K, H, W, _pix_max)
            cur_B = p['d_final_B']

        # ── Step 5: Single sync + download ──
        _cuda.synchronize()

        if not hasattr(_cuda_fused_process_eq, '_host_A') or _cuda_fused_process_eq._host_A.dtype != pix_dtype:
            _cuda_fused_process_eq._host_A = np.empty(n_elems, dtype=pix_dtype)
            _cuda_fused_process_eq._host_B = np.empty(n_elems, dtype=pix_dtype)
        elif _cuda_fused_process_eq._host_A.size != n_elems:
            _cuda_fused_process_eq._host_A = np.empty(n_elems, dtype=pix_dtype)
            _cuda_fused_process_eq._host_B = np.empty(n_elems, dtype=pix_dtype)

        cur_A.copy_to_host(_cuda_fused_process_eq._host_A)
        cur_B.copy_to_host(_cuda_fused_process_eq._host_B)
        np.copyto(result_buf[:, half_w:], _cuda_fused_process_eq._host_A[:n_elems].reshape(out_h, half_w, 3))
        np.copyto(result_buf[:, :half_w], _cuda_fused_process_eq._host_B[:n_elems].reshape(out_h, half_w, 3))

    print("  ✓ Numba CUDA kernels defined")

# ── MLX Metal GPU acceleration (Apple Silicon) ──────────────────────────
try:
    import mlx.core as mx
    from mlx.core.fast import metal_kernel as _mlx_metal_kernel
    HAS_MLX = True
    print(f"✓ MLX imported successfully - version: {mx.__version__}, device: {mx.default_device()}")
except ImportError:
    HAS_MLX = False
    mx = None
    if sys.platform == 'darwin':
        print(f"⚠ Warning: MLX not available - GPU acceleration disabled")

if HAS_MLX:
  try:
    # Test basic MLX Metal operation before compiling custom kernels.
    # This catches Metal version mismatches early (e.g. MLX requesting Metal 4.0
    # on M2 hardware that only supports Metal 3.0) without triggering native
    # error dialogs during kernel compilation.
    _mlx_test = mx.array([1.0, 2.0, 3.0])
    mx.eval(_mlx_test + _mlx_test)
    del _mlx_test

    # Metal shader: fused rotation → RS → IORI → EAC cross lookup → bilinear sample
    # One GPU thread per output pixel. Eliminates cv2.remap entirely.
    _MLX_CROSS_REMAP_SOURCE = '''
        uint idx = thread_position_in_grid.x;
        if (idx >= n[0]) return;

        // Read direction vector
        float ox = xyz_x[idx], oy = xyz_y[idx], oz = xyz_z[idx];

        // Step 1: R_sensor × direction → sensor space
        float xr = R[0]*ox + R[1]*oy + R[2]*oz;
        float yr = R[3]*ox + R[4]*oy + R[5]*oz;
        float zr = R[6]*ox + R[7]*oy + R[8]*oz;

        // Step 2: 3D rotation RS correction
        // Small-angle rotation in sensor space. Verified against firmware output
        // (GoPro Player equirect + raw EAC) — matches firmware RS at all angles
        // from center (θ=15°) to edge (θ=88°) within 5% (ratio 0.95-1.05).
        // rs_coeffs[0]=yaw, [1]=pitch, [2]=roll (rate × factor, in rad/s)
        // t_offset[idx] = precomputed time offset for this pixel's sensor row
        float xn = xr, yn = yr, zn = zr;
        float rs_yaw   = rs_coeffs[0];
        float rs_pitch = rs_coeffs[1];
        float rs_roll  = rs_coeffs[2];
        bool has_rs = (rs_pitch != 0.0f || rs_roll != 0.0f || rs_yaw != 0.0f);

        if (has_rs) {
            float t = t_offset[idx];
            float ya = rs_yaw   * t;
            float pa = rs_pitch * t;
            float ra = rs_roll  * t;
            xn = xr + ya * zr - ra * yr;
            yn = yr + ra * xr - pa * zr;
            zn = zr - ya * xr + pa * yr;
        }

        // Step 3: IORI rotation (conditional)
        if (params[0] > 0.5f) {
            float xc = Rc[0]*xn + Rc[1]*yn + Rc[2]*zn;
            float yc = Rc[3]*xn + Rc[4]*yn + Rc[5]*zn;
            float zc = Rc[6]*xn + Rc[7]*yn + Rc[8]*zn;
            xn = xc; yn = yc; zn = zc;
        }

        // Step 4: EAC cross face determination → (mx_f, my_f)
        float ax = metal::abs(xn), ay = metal::abs(yn);
        float mx_f = -1.0f, my_f = -1.0f;
        float TWO_OVER_PI = 0.6366197723675814f;

        if (zn > 0.0f && ax <= zn && ay <= zn) {
            float u = TWO_OVER_PI * metal::atan(xn / zn) + 0.5f;
            float v = 0.5f - TWO_OVER_PI * metal::atan(yn / zn);
            mx_f = 1008.0f + u * 1920.0f;
            my_f = 1008.0f + v * 1920.0f;
        } else if (xn > 0.0f && zn <= xn && ay <= xn) {
            float u = TWO_OVER_PI * metal::atan(-zn / xn) + 0.5f;
            float v = 0.5f - TWO_OVER_PI * metal::atan(yn / xn);
            float fc = metal::clamp(u * 1920.0f, 0.0f, 1007.0f);
            mx_f = 2928.0f + fc;
            my_f = 1008.0f + v * 1920.0f;
        } else if (xn < 0.0f && zn <= ax && ay <= ax) {
            float u = TWO_OVER_PI * metal::atan(zn / ax) + 0.5f;
            float v = 0.5f - TWO_OVER_PI * metal::atan(yn / ax);
            float pc = metal::clamp(u * 1920.0f - 912.0f, 0.0f, 1007.0f);
            mx_f = pc;
            my_f = 1008.0f + v * 1920.0f;
        } else if (yn > 0.0f && ax <= yn && zn <= yn) {
            float u = TWO_OVER_PI * metal::atan(xn / yn) + 0.5f;
            float v = TWO_OVER_PI * metal::atan(zn / yn) + 0.5f;
            float pr = metal::clamp(v * 1920.0f - 912.0f, 0.0f, 1007.0f);
            mx_f = 1008.0f + u * 1920.0f;
            my_f = pr;
        } else if (yn < 0.0f && ax <= ay && zn <= ay) {
            float u = TWO_OVER_PI * metal::atan(xn / ay) + 0.5f;
            float v = 0.5f - TWO_OVER_PI * metal::atan(zn / ay);
            float fr = metal::clamp(v * 1920.0f, 0.0f, 1007.0f);
            mx_f = 1008.0f + u * 1920.0f;
            my_f = 2928.0f + fr;
        }

        mx_f = metal::clamp(mx_f, -1.0f, 3935.0f);
        my_f = metal::clamp(my_f, -1.0f, 3935.0f);

        // Step 5: Bilinear interpolation from cross image (3936×3936×3 BGR)
        float pix_max = params[1];  // 255.0 for uint8, 65535.0 for uint16
        uint out_base = idx * 3;
        if (mx_f < 0.0f || my_f < 0.0f) {
            out[out_base] = 0; out[out_base + 1] = 0; out[out_base + 2] = 0;
            return;
        }

        int cross_w = 3936;
        int ix = (int)metal::floor(mx_f);
        int iy = (int)metal::floor(my_f);
        float fx = mx_f - (float)ix;
        float fy = my_f - (float)iy;
        int ix1 = metal::min(ix + 1, cross_w - 1);
        int iy1 = metal::min(iy + 1, cross_w - 1);

        for (int c = 0; c < 3; c++) {
            float v00 = (float)cross_img[(iy  * cross_w + ix ) * 3 + c];
            float v10 = (float)cross_img[(iy  * cross_w + ix1) * 3 + c];
            float v01 = (float)cross_img[(iy1 * cross_w + ix ) * 3 + c];
            float v11 = (float)cross_img[(iy1 * cross_w + ix1) * 3 + c];
            float val = v00*(1.0f-fx)*(1.0f-fy) + v10*fx*(1.0f-fy)
                      + v01*(1.0f-fx)*fy + v11*fx*fy;
            out[out_base + c] = (uint16_t)metal::clamp(val + 0.5f, 0.0f, pix_max);
        }
    '''

    _mlx_cross_kernel = _mlx_metal_kernel(
        name="cross_remap_bilinear",
        input_names=["xyz_x", "xyz_y", "xyz_z", "R", "t_offset", "rs_coeffs",
                     "Rc", "params", "cross_img", "n", "rs_mats"],
        output_names=["out"],
        source=_MLX_CROSS_REMAP_SOURCE,
    )

    # ── MLX Metal equirect remap shader (for non-360 SBS half-equirect input) ──
    _MLX_EQUIRECT_REMAP_SOURCE = '''
        uint idx = thread_position_in_grid.x;
        if (idx >= n[0]) return;

        float ox = xyz_x[idx], oy = xyz_y[idx], oz = xyz_z[idx];

        // R × direction → rotated direction
        float xn = R[0]*ox + R[1]*oy + R[2]*oz;
        float yn = R[3]*ox + R[4]*oy + R[5]*oz;
        float zn = R[6]*ox + R[7]*oy + R[8]*oz;

        // Equirect UV: lon = atan2(x, z), lat = asin(y)
        float lon = metal::atan2(xn, zn);
        float lat = metal::precise::asin(metal::clamp(yn, -1.0f, 1.0f));
        float INV_PI = 0.31830988618f;
        float img_w = img_dims[0];
        float img_h = img_dims[1];
        float mx_f = metal::clamp((lon * INV_PI + 0.5f) * img_w, 0.0f, img_w - 1.0f);
        float my_f = metal::clamp((0.5f - lat * INV_PI) * img_h, 0.0f, img_h - 1.0f);

        // Bilinear interpolation from source image (H × W × 3 BGR)
        float pix_max = img_dims[2];  // 255.0 for uint8, 65535.0 for uint16
        int iw = (int)img_w;
        int ix = (int)metal::floor(mx_f);
        int iy = (int)metal::floor(my_f);
        float fx = mx_f - (float)ix;
        float fy = my_f - (float)iy;
        int ix1 = metal::min(ix + 1, iw - 1);
        int iy1 = metal::min(iy + 1, (int)img_h - 1);

        uint out_base = idx * 3;
        for (int c = 0; c < 3; c++) {
            float v00 = (float)src_img[(iy  * iw + ix ) * 3 + c];
            float v10 = (float)src_img[(iy  * iw + ix1) * 3 + c];
            float v01 = (float)src_img[(iy1 * iw + ix ) * 3 + c];
            float v11 = (float)src_img[(iy1 * iw + ix1) * 3 + c];
            float val = v00*(1.0f-fx)*(1.0f-fy) + v10*fx*(1.0f-fy)
                      + v01*(1.0f-fx)*fy + v11*fx*fy;
            out[out_base + c] = (uint16_t)metal::clamp(val + 0.5f, 0.0f, pix_max);
        }
    '''

    _mlx_equirect_kernel = _mlx_metal_kernel(
        name="equirect_remap_bilinear",
        input_names=["xyz_x", "xyz_y", "xyz_z", "R", "img_dims", "src_img", "n"],
        output_names=["out"],
        source=_MLX_EQUIRECT_REMAP_SOURCE,
    )

    def _mlx_process_eye_equirect(src_np, mlx_xyz_x, mlx_xyz_y, mlx_xyz_z,
                                   R_np, n_pixels, pix_max=255.0):
        """Run equirect remap kernel for one eye (non-360 path). pix_max: 255 for preview, 65535 for export."""
        mlx_R = mx.array(R_np.ravel().astype(np.float32))
        src_h, src_w = src_np.shape[:2]
        mlx_dims = mx.array(np.array([float(src_w), float(src_h), float(pix_max)], dtype=np.float32))
        mlx_src = mx.array(src_np.ravel())
        mlx_n = mx.array(np.array([n_pixels], dtype=np.int32))
        _out_dtype = mx.uint16 if pix_max > 256 else mx.uint8

        outputs = _mlx_equirect_kernel(
            inputs=[mlx_xyz_x, mlx_xyz_y, mlx_xyz_z, mlx_R, mlx_dims, mlx_src, mlx_n],
            output_shapes=[(n_pixels * 3,)],
            output_dtypes=[_out_dtype],
            grid=(n_pixels, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(outputs[0])
        return np.array(outputs[0])

    # Preallocate constant MLX arrays (reused every frame)
    _mlx_identity_R = mx.array(np.eye(3, dtype=np.float32).ravel())
    _mlx_zero_rs = mx.array(np.zeros(3, dtype=np.float32))
    # KLNS params for sensor-space RS (set when .360 file is loaded, default zeros)
    _mlx_klns_params = np.zeros(9, dtype=np.float32)  # [c0,c1,c2,c3,c4, cx,cy, cal_dim, srot]

    # Dummy rs_mats buffer (identity, used when per-scanline mode not active)
    _mlx_rs_mats_identity = mx.array(np.eye(3, dtype=np.float32).ravel())  # 9 floats

    def _mlx_process_eye(cross_np, mlx_xyz_x, mlx_xyz_y, mlx_xyz_z,
                         R_np, mlx_t_offset, rs_coeffs_np,
                         R_cross_np, has_iori, n_pixels, pix_max=255.0,
                         two_step=False, rs_perscanline_mats=None):
        """Run fused Metal kernel for one eye.
        RS correction uses 3D rotation with precomputed t_offset per pixel.
        rs_coeffs_np: [yaw, pitch, roll] angular velocity × factor in rad/s."""
        mlx_R = mx.array(R_np.ravel().astype(np.float32))
        mlx_cross = mx.array(cross_np.ravel())

        if rs_coeffs_np is not None and np.any(rs_coeffs_np != 0):
            mlx_rs = mx.array(rs_coeffs_np.astype(np.float32))
        else:
            mlx_rs = mx.array(np.zeros(3, dtype=np.float32))
        mlx_rs_mats = _mlx_rs_mats_identity
        _rs_mode = 0.0

        iori_flag = 1.0 if (has_iori and R_cross_np is not None) else 0.0
        if has_iori and R_cross_np is not None:
            mlx_Rc = mx.array(R_cross_np.ravel().astype(np.float32))
        else:
            mlx_Rc = mx.array(np.eye(3, dtype=np.float32).ravel())

        # params: [has_iori, pix_max, c0,c1,c2,c3,c4, cx,cy, cal_dim, srot, rs_mode]
        _kp = _mlx_klns_params
        mlx_params = mx.array(np.array([iori_flag, float(pix_max),
                                         _kp[0], _kp[1], _kp[2], _kp[3], _kp[4],
                                         _kp[5], _kp[6], _kp[7], _kp[8], _rs_mode], dtype=np.float32))
        mlx_n = mx.array(np.array([n_pixels], dtype=np.int32))
        _out_dtype = mx.uint16 if pix_max > 256 else mx.uint8

        outputs = _mlx_cross_kernel(
            inputs=[mlx_xyz_x, mlx_xyz_y, mlx_xyz_z, mlx_R, mlx_t_offset,
                    mlx_rs, mlx_Rc, mlx_params, mlx_cross, mlx_n, mlx_rs_mats],
            output_shapes=[(n_pixels * 3,)],
            output_dtypes=[_out_dtype],
            grid=(n_pixels, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(outputs[0])
        return np.array(outputs[0])

    # ── MLX Metal 3D LUT kernel ──
    _MLX_LUT3D_SOURCE = '''
        uint idx = thread_position_in_grid.x;
        if (idx >= n[0]) return;
        uint base = idx * 3;
        float scale = params[0];
        int lut_size = (int)params[1];
        float pix_max = params[2];  // 255.0 for uint8, 65535.0 for uint16
        int ls = lut_size;
        int ls2 = ls * ls;

        float b_val = (float)frame[base]     * scale;
        float g_val = (float)frame[base + 1] * scale;
        float r_val = (float)frame[base + 2] * scale;

        int b0 = (int)b_val; int g0 = (int)g_val; int r0 = (int)r_val;
        int b1 = metal::min(b0 + 1, ls - 1);
        int g1 = metal::min(g0 + 1, ls - 1);
        int r1 = metal::min(r0 + 1, ls - 1);
        float fb = b_val - (float)b0;
        float fg = g_val - (float)g0;
        float fr = r_val - (float)r0;

        for (int c = 0; c < 3; c++) {
            float c000 = lut[(b0*ls2 + g0*ls + r0)*3 + c];
            float c001 = lut[(b0*ls2 + g0*ls + r1)*3 + c];
            float c010 = lut[(b0*ls2 + g1*ls + r0)*3 + c];
            float c011 = lut[(b0*ls2 + g1*ls + r1)*3 + c];
            float c100 = lut[(b1*ls2 + g0*ls + r0)*3 + c];
            float c101 = lut[(b1*ls2 + g0*ls + r1)*3 + c];
            float c110 = lut[(b1*ls2 + g1*ls + r0)*3 + c];
            float c111 = lut[(b1*ls2 + g1*ls + r1)*3 + c];
            float c00 = c000 + (c001 - c000) * fr;
            float c01 = c010 + (c011 - c010) * fr;
            float c10 = c100 + (c101 - c100) * fr;
            float c11 = c110 + (c111 - c110) * fr;
            float c0v = c00 + (c01 - c00) * fg;
            float c1v = c10 + (c11 - c10) * fg;
            float val = c0v + (c1v - c0v) * fb;
            int out_c = 2 - c;
            out[base + out_c] = (uint16_t)metal::clamp(val * pix_max + 0.5f, 0.0f, pix_max);
        }
    '''

    _mlx_lut3d_kernel = _mlx_metal_kernel(
        name="lut3d_trilinear",
        input_names=["frame", "lut", "params", "n"],
        output_names=["out"],
        source=_MLX_LUT3D_SOURCE,
    )

    def _mlx_apply_lut_3d(frame_np, lut_flat_mlx, lut_size, n_pixels, pix_max=255.0):
        """Apply 3D LUT via Metal kernel. pix_max: 255 for preview, 65535 for export."""
        mlx_frame = mx.array(frame_np.ravel())
        mlx_params = mx.array(np.array([(lut_size - 1) / float(pix_max), float(lut_size), float(pix_max)], dtype=np.float32))
        mlx_n = mx.array(np.array([n_pixels], dtype=np.int32))
        _out_dtype = mx.uint16 if pix_max > 256 else mx.uint8
        outputs = _mlx_lut3d_kernel(
            inputs=[mlx_frame, lut_flat_mlx, mlx_params, mlx_n],
            output_shapes=[(n_pixels * 3,)],
            output_dtypes=[_out_dtype],
            grid=(n_pixels, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(outputs[0])
        return np.array(outputs[0]).reshape(frame_np.shape)

    # ── MLX Metal sharpen kernels (separable Gaussian blur + latitude-weighted USM) ──
    _MLX_BLUR_H_SOURCE = '''
        uint idx = thread_position_in_grid.x;
        int total = (int)blur_dims[0];  // H * W * 3
        if ((int)idx >= total) return;
        int W = (int)blur_dims[1];
        int K = (int)blur_dims[2];
        int half_k = K / 2;
        int c = idx % 3;
        int pix = idx / 3;
        int x = pix % W;
        int y = pix / W;
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            int sx = x + k - half_k;
            if (sx < 0) sx = 0;
            if (sx >= W) sx = W - 1;
            val += (float)src[(y * W + sx) * 3 + c] * gkernel[k];
        }
        float pix_max = blur_dims[3];
        dst[idx] = (uint16_t)metal::clamp(val + 0.5f, 0.0f, pix_max);
    '''

    _MLX_BLUR_V_USM_SOURCE = '''
        uint idx = thread_position_in_grid.x;
        int total = (int)blur_dims[0];  // H * W * 3
        if ((int)idx >= total) return;
        int W = (int)blur_dims[1];
        int H = (int)blur_dims[2];
        int K = (int)blur_dims[3];
        int half_k = K / 2;
        float amount = usm_params[0];
        float pix_max = usm_params[1];
        int c = idx % 3;
        int pix = idx / 3;
        int x = pix % W;
        int y = pix / W;
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            int sy = y + k - half_k;
            if (sy < 0) sy = 0;
            if (sy >= H) sy = H - 1;
            val += (float)h_blurred[(sy * W + x) * 3 + c] * gkernel[k];
        }
        float f = (float)src[idx];
        float w = lat_weights[y] * amount;
        float result = f + w * (f - val);
        dst[idx] = (uint16_t)metal::clamp(result + 0.5f, 0.0f, pix_max);
    '''

    _mlx_blur_h_kernel = _mlx_metal_kernel(
        name="blur_horizontal",
        input_names=["src", "gkernel", "blur_dims"],
        output_names=["dst"],
        source=_MLX_BLUR_H_SOURCE,
    )

    _mlx_blur_v_usm_kernel = _mlx_metal_kernel(
        name="blur_v_usm",
        input_names=["src", "h_blurred", "gkernel", "lat_weights", "usm_params", "blur_dims"],
        output_names=["dst"],
        source=_MLX_BLUR_V_USM_SOURCE,
    )

    def _mlx_sharpen(frame_np, g1d_np, lat_weights_np, amount, ksize, pix_max=65535.0):
        """GPU-accelerated equirect-aware USM via MLX Metal: separable blur + latitude-weighted USM."""
        h, w = frame_np.shape[:2]
        n_elem = h * w * 3

        mlx_src = mx.array(frame_np.ravel())
        mlx_kernel = mx.array(g1d_np)
        _out_dtype = mx.uint16 if pix_max > 256 else mx.uint8

        # Pass 1: horizontal blur
        mlx_dims_h = mx.array(np.array([n_elem, w, ksize, pix_max], dtype=np.float32))
        h_out = _mlx_blur_h_kernel(
            inputs=[mlx_src, mlx_kernel, mlx_dims_h],
            output_shapes=[(n_elem,)],
            output_dtypes=[_out_dtype],
            grid=(n_elem, 1, 1),
            threadgroup=(256, 1, 1),
        )

        # Pass 2: vertical blur on h_blurred + fused USM
        mlx_lat = mx.array(lat_weights_np)
        mlx_params = mx.array(np.array([amount, pix_max], dtype=np.float32))
        mlx_dims_v = mx.array(np.array([n_elem, w, h, ksize], dtype=np.float32))
        v_out = _mlx_blur_v_usm_kernel(
            inputs=[mlx_src, h_out[0], mlx_kernel, mlx_lat, mlx_params, mlx_dims_v],
            output_shapes=[(n_elem,)],
            output_dtypes=[_out_dtype],
            grid=(n_elem, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(v_out[0])
        return np.array(v_out[0]).reshape(frame_np.shape)


    print("  ✓ MLX Metal kernels compiled")
  except Exception as _mlx_err:
    HAS_MLX = False
    print(f"⚠ MLX Metal kernel compilation failed: {_mlx_err}")
    print("  Falling back to CPU processing (Numba/NumPy)")

# ── wgpu disabled (slower than CPU in benchmarks) ─────────────────────────
HAS_WGPU = False

# wgpu globals (lazy-initialized on first use)
_wgpu_device = None
_wgpu_cross_pipeline = None
_wgpu_equirect_pipeline = None
_wgpu_lut_pipeline = None

# ── WGSL compute shader: fused cross remap + RS + IORI + bilinear ────────
_WGSL_CROSS_REMAP_SOURCE = """
@group(0) @binding(0) var<storage, read> xyz_x: array<f32>;
@group(0) @binding(1) var<storage, read> xyz_y: array<f32>;
@group(0) @binding(2) var<storage, read> xyz_z: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> t_offset: array<f32>;
@group(0) @binding(5) var<storage, read> rs_coeffs: array<f32>;
@group(0) @binding(6) var<storage, read> Rc: array<f32>;
@group(0) @binding(7) var<storage, read> params: array<f32>;
@group(0) @binding(8) var<storage, read> cross_img: array<u32>;
@group(0) @binding(9) var<storage, read> n_buf: array<u32>;
@group(0) @binding(10) var<storage, read_write> out: array<f32>;

fn read_byte(byte_idx: u32) -> f32 {
    let word_idx = byte_idx / 4u;
    let shift = (byte_idx % 4u) * 8u;
    return f32((cross_img[word_idx] >> shift) & 0xFFu);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if (idx >= n_buf[0]) { return; }

    let ox = xyz_x[idx];
    let oy = xyz_y[idx];
    let oz = xyz_z[idx];

    // Step 1: R_sensor * direction -> sensor space
    var xr = R[0]*ox + R[1]*oy + R[2]*oz;
    var yr = R[3]*ox + R[4]*oy + R[5]*oz;
    var zr = R[6]*ox + R[7]*oy + R[8]*oz;

    // Step 2: RS perturbation
    let t = t_offset[idx];
    var xn = xr + rs_coeffs[0]*t*zr - rs_coeffs[2]*t*yr;
    var yn = yr + rs_coeffs[2]*t*xr - rs_coeffs[1]*t*zr;
    var zn = zr - rs_coeffs[0]*t*xr + rs_coeffs[1]*t*yr;

    // Step 3: IORI rotation (conditional)
    if (params[0] > 0.5) {
        let xc = Rc[0]*xn + Rc[1]*yn + Rc[2]*zn;
        let yc = Rc[3]*xn + Rc[4]*yn + Rc[5]*zn;
        let zc = Rc[6]*xn + Rc[7]*yn + Rc[8]*zn;
        xn = xc; yn = yc; zn = zc;
    }

    // Step 4: EAC cross face -> (mx_f, my_f)
    let ax = abs(xn);
    let ay = abs(yn);
    var mx_f: f32 = -1.0;
    var my_f: f32 = -1.0;
    let TWO_OVER_PI: f32 = 0.6366197723675814;

    if (zn > 0.0 && ax <= zn && ay <= zn) {
        let u = TWO_OVER_PI * atan(xn / zn) + 0.5;
        let v = 0.5 - TWO_OVER_PI * atan(yn / zn);
        mx_f = 1008.0 + u * 1920.0;
        my_f = 1008.0 + v * 1920.0;
    } else if (xn > 0.0 && zn <= xn && ay <= xn) {
        let u = TWO_OVER_PI * atan(-zn / xn) + 0.5;
        let v = 0.5 - TWO_OVER_PI * atan(yn / xn);
        let fc = clamp(u * 1920.0, 0.0, 1007.0);
        mx_f = 2928.0 + fc;
        my_f = 1008.0 + v * 1920.0;
    } else if (xn < 0.0 && zn <= ax && ay <= ax) {
        let u = TWO_OVER_PI * atan(zn / ax) + 0.5;
        let v = 0.5 - TWO_OVER_PI * atan(yn / ax);
        let pc = clamp(u * 1920.0 - 912.0, 0.0, 1007.0);
        mx_f = pc;
        my_f = 1008.0 + v * 1920.0;
    } else if (yn > 0.0 && ax <= yn && zn <= yn) {
        let u = TWO_OVER_PI * atan(xn / yn) + 0.5;
        let v = TWO_OVER_PI * atan(zn / yn) + 0.5;
        let pr = clamp(v * 1920.0 - 912.0, 0.0, 1007.0);
        mx_f = 1008.0 + u * 1920.0;
        my_f = pr;
    } else if (yn < 0.0 && ax <= ay && zn <= ay) {
        let u = TWO_OVER_PI * atan(xn / ay) + 0.5;
        let v = 0.5 - TWO_OVER_PI * atan(zn / ay);
        let fr = clamp(v * 1920.0, 0.0, 1007.0);
        mx_f = 1008.0 + u * 1920.0;
        my_f = 2928.0 + fr;
    }

    mx_f = clamp(mx_f, -1.0, 3935.0);
    my_f = clamp(my_f, -1.0, 3935.0);

    // Step 5: Bilinear interpolation from cross image (3936x3936x3 BGR uint8)
    let out_base = idx * 3u;
    if (mx_f < 0.0 || my_f < 0.0) {
        out[out_base] = 0.0;
        out[out_base + 1u] = 0.0;
        out[out_base + 2u] = 0.0;
        return;
    }

    let cross_w: i32 = 3936;
    let ix = i32(floor(mx_f));
    let iy = i32(floor(my_f));
    let fx = mx_f - f32(ix);
    let fy = my_f - f32(iy);
    let ix1 = min(ix + 1, cross_w - 1);
    let iy1 = min(iy + 1, cross_w - 1);

    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let v00 = read_byte(u32(iy  * cross_w + ix ) * 3u + c);
        let v10 = read_byte(u32(iy  * cross_w + ix1) * 3u + c);
        let v01 = read_byte(u32(iy1 * cross_w + ix ) * 3u + c);
        let v11 = read_byte(u32(iy1 * cross_w + ix1) * 3u + c);
        let val = v00*(1.0-fx)*(1.0-fy) + v10*fx*(1.0-fy)
                + v01*(1.0-fx)*fy + v11*fx*fy;
        out[out_base + c] = clamp(val + 0.5, 0.0, 65535.0);
    }
}
"""

# ── WGSL compute shader: equirect remap + bilinear (non-360 path) ─────────
_WGSL_EQUIRECT_REMAP_SOURCE = """
@group(0) @binding(0) var<storage, read> xyz_x: array<f32>;
@group(0) @binding(1) var<storage, read> xyz_y: array<f32>;
@group(0) @binding(2) var<storage, read> xyz_z: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> img_dims: array<f32>;
@group(0) @binding(5) var<storage, read> src_img: array<u32>;
@group(0) @binding(6) var<storage, read> n_buf: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

fn read_byte(byte_idx: u32) -> f32 {
    let word_idx = byte_idx / 4u;
    let shift = (byte_idx % 4u) * 8u;
    return f32((src_img[word_idx] >> shift) & 0xFFu);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if (idx >= n_buf[0]) { return; }

    let ox = xyz_x[idx];
    let oy = xyz_y[idx];
    let oz = xyz_z[idx];

    // R × direction → rotated direction
    let xn = R[0]*ox + R[1]*oy + R[2]*oz;
    let yn = R[3]*ox + R[4]*oy + R[5]*oz;
    let zn = R[6]*ox + R[7]*oy + R[8]*oz;

    // Equirect UV: lon = atan2(x, z), lat = asin(y)
    let lon = atan2(xn, zn);
    let lat = asin(clamp(yn, -1.0, 1.0));
    let INV_PI: f32 = 0.31830988618;
    let img_w = img_dims[0];
    let img_h = img_dims[1];
    let mx_f = clamp((lon * INV_PI + 0.5) * img_w, 0.0, img_w - 1.0);
    let my_f = clamp((0.5 - lat * INV_PI) * img_h, 0.0, img_h - 1.0);

    // Bilinear interpolation from source image (H × W × 3 BGR uint8)
    let iw = i32(img_w);
    let ix = i32(floor(mx_f));
    let iy = i32(floor(my_f));
    let fx = mx_f - f32(ix);
    let fy = my_f - f32(iy);
    let ix1 = min(ix + 1, iw - 1);
    let iy1 = min(iy + 1, i32(img_h) - 1);

    let out_base = idx * 3u;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let v00 = read_byte(u32(iy  * iw + ix ) * 3u + c);
        let v10 = read_byte(u32(iy  * iw + ix1) * 3u + c);
        let v01 = read_byte(u32(iy1 * iw + ix ) * 3u + c);
        let v11 = read_byte(u32(iy1 * iw + ix1) * 3u + c);
        let val = v00*(1.0-fx)*(1.0-fy) + v10*fx*(1.0-fy)
                + v01*(1.0-fx)*fy + v11*fx*fy;
        out[out_base + c] = clamp(val + 0.5, 0.0, 65535.0);
    }
}
"""

# ── WGSL compute shader: 3D LUT trilinear interpolation ──────────────────
_WGSL_LUT3D_SOURCE = """
@group(0) @binding(0) var<storage, read> frame_packed: array<u32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read> n_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn read_frame_byte(byte_idx: u32) -> f32 {
    let word_idx = byte_idx / 4u;
    let shift = (byte_idx % 4u) * 8u;
    return f32((frame_packed[word_idx] >> shift) & 0xFFu);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    if (idx >= n_buf[0]) { return; }

    let base = idx * 3u;
    let scale = params[0];
    let lut_size = i32(params[1]);
    let ls = lut_size;
    let ls2 = ls * ls;

    let b_val = read_frame_byte(base) * scale;
    let g_val = read_frame_byte(base + 1u) * scale;
    let r_val = read_frame_byte(base + 2u) * scale;

    let b0 = i32(b_val); let g0 = i32(g_val); let r0 = i32(r_val);
    let b1 = min(b0 + 1, ls - 1);
    let g1 = min(g0 + 1, ls - 1);
    let r1 = min(r0 + 1, ls - 1);
    let fb = b_val - f32(b0);
    let fg = g_val - f32(g0);
    let fr = r_val - f32(r0);

    for (var c: i32 = 0; c < 3; c = c + 1) {
        let c000 = lut[(b0*ls2 + g0*ls + r0)*3 + c];
        let c001 = lut[(b0*ls2 + g0*ls + r1)*3 + c];
        let c010 = lut[(b0*ls2 + g1*ls + r0)*3 + c];
        let c011 = lut[(b0*ls2 + g1*ls + r1)*3 + c];
        let c100 = lut[(b1*ls2 + g0*ls + r0)*3 + c];
        let c101 = lut[(b1*ls2 + g0*ls + r1)*3 + c];
        let c110 = lut[(b1*ls2 + g1*ls + r0)*3 + c];
        let c111 = lut[(b1*ls2 + g1*ls + r1)*3 + c];
        let v00 = c000 + (c001 - c000) * fr;
        let v01 = c010 + (c011 - c010) * fr;
        let v10 = c100 + (c101 - c100) * fr;
        let v11 = c110 + (c111 - c110) * fr;
        let v0 = v00 + (v01 - v00) * fg;
        let v1 = v10 + (v11 - v10) * fg;
        let val = v0 + (v1 - v0) * fb;
        // BGR->RGB swap: out channel = 2 - c
        let out_c = u32(2 - c);
        let pix_max = params[2];  // 255.0 for uint8, 65535.0 for uint16
        out[base + out_c] = clamp(val * pix_max + 0.5, 0.0, pix_max);
    }
}
"""

def _wgpu_init():
    """Lazy-initialize wgpu device and compile compute pipelines."""
    global _wgpu_device, _wgpu_cross_pipeline, _wgpu_equirect_pipeline, _wgpu_lut_pipeline, HAS_WGPU
    if _wgpu_device is not None:
        return True
    if not HAS_WGPU:
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No GPU adapter found")
        _wgpu_device = adapter.request_device_sync()
        print(f"  wgpu device: {adapter.info.get('device', 'unknown')} "
              f"(backend: {adapter.info.get('backend_type', 'unknown')})")
        # Compile compute pipelines
        cross_module = _wgpu_device.create_shader_module(code=_WGSL_CROSS_REMAP_SOURCE)
        _wgpu_cross_pipeline = _wgpu_device.create_compute_pipeline(
            layout="auto",
            compute={"module": cross_module, "entry_point": "main"})
        equirect_module = _wgpu_device.create_shader_module(code=_WGSL_EQUIRECT_REMAP_SOURCE)
        _wgpu_equirect_pipeline = _wgpu_device.create_compute_pipeline(
            layout="auto",
            compute={"module": equirect_module, "entry_point": "main"})
        lut_module = _wgpu_device.create_shader_module(code=_WGSL_LUT3D_SOURCE)
        _wgpu_lut_pipeline = _wgpu_device.create_compute_pipeline(
            layout="auto",
            compute={"module": lut_module, "entry_point": "main"})
        print("  ✓ wgpu compute pipelines compiled")
        return True
    except Exception as e:
        print(f"  ✗ wgpu init failed: {e}")
        HAS_WGPU = False
        return False

def _wgpu_create_buffer(data_np, read_only=True):
    """Create a wgpu storage buffer from numpy array."""
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    if not read_only:
        usage |= wgpu.BufferUsage.COPY_SRC
    buf = _wgpu_device.create_buffer(size=data_np.nbytes, usage=usage)
    _wgpu_device.queue.write_buffer(buf, 0, data_np.tobytes())
    return buf

def _wgpu_create_empty_buffer(size_bytes, read_only=False):
    """Create an empty wgpu storage buffer."""
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    if not read_only:
        usage |= wgpu.BufferUsage.COPY_DST
    return _wgpu_device.create_buffer(size=size_bytes, usage=usage)

# ── wgpu persistent buffer pool for cross remap (avoids per-frame allocation) ──
_wgpu_cross_pool = None  # dict of pre-allocated buffers + cached bind group

def _wgpu_ensure_cross_pool(n_pixels, xyz_x_buf, xyz_y_buf, xyz_z_buf, t_offset_buf, out_buf):
    """Ensure persistent buffer pool exists for cross remap. Create once, reuse every frame."""
    global _wgpu_cross_pool
    if _wgpu_cross_pool is not None and _wgpu_cross_pool['n_pixels'] == n_pixels:
        return _wgpu_cross_pool

    # Cross image: 3936*3936*3 bytes packed as u32
    cross_size = (3936 * 3936 * 3 + 3) // 4 * 4  # round up to 4-byte boundary
    cross_n_u32 = cross_size // 4

    pool = {
        'n_pixels': n_pixels,
        'R_buf': _wgpu_device.create_buffer(size=9*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'rs_buf': _wgpu_device.create_buffer(size=3*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'Rc_buf': _wgpu_device.create_buffer(size=9*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'params_buf': _wgpu_device.create_buffer(size=1*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'cross_buf': _wgpu_device.create_buffer(size=cross_n_u32*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'n_buf': _wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
    }

    # Write n_pixels (constant)
    _wgpu_device.queue.write_buffer(pool['n_buf'], 0, np.array([n_pixels], dtype=np.uint32).tobytes())
    # Write default zero values
    _wgpu_device.queue.write_buffer(pool['rs_buf'], 0, np.zeros(3, dtype=np.float32).tobytes())
    _wgpu_device.queue.write_buffer(pool['Rc_buf'], 0, np.eye(3, dtype=np.float32).ravel().tobytes())
    _wgpu_device.queue.write_buffer(pool['params_buf'], 0, np.array([0.0], dtype=np.float32).tobytes())

    # Create bind group (references buffer objects — contents can change via write_buffer)
    pool['bind_group'] = _wgpu_device.create_bind_group(
        layout=_wgpu_cross_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": xyz_x_buf}},
            {"binding": 1, "resource": {"buffer": xyz_y_buf}},
            {"binding": 2, "resource": {"buffer": xyz_z_buf}},
            {"binding": 3, "resource": {"buffer": pool['R_buf']}},
            {"binding": 4, "resource": {"buffer": t_offset_buf}},
            {"binding": 5, "resource": {"buffer": pool['rs_buf']}},
            {"binding": 6, "resource": {"buffer": pool['Rc_buf']}},
            {"binding": 7, "resource": {"buffer": pool['params_buf']}},
            {"binding": 8, "resource": {"buffer": pool['cross_buf']}},
            {"binding": 9, "resource": {"buffer": pool['n_buf']}},
            {"binding": 10, "resource": {"buffer": out_buf}},
        ],
    )
    _wgpu_cross_pool = pool
    return pool

def _wgpu_process_eye(cross_np, xyz_x_buf, xyz_y_buf, xyz_z_buf,
                      R_np, t_offset_buf, rs_coeffs_np,
                      R_cross_np, has_iori, n_pixels, out_buf):
    """Run fused wgpu compute kernel for one eye. Returns (n_pixels*3,) uint8 numpy array.
    Reuses persistent buffers — only writes changed data per frame."""
    pool = _wgpu_ensure_cross_pool(n_pixels, xyz_x_buf, xyz_y_buf, xyz_z_buf, t_offset_buf, out_buf)
    q = _wgpu_device.queue

    # Update per-frame small buffers (tiny writes, ~36-48 bytes)
    q.write_buffer(pool['R_buf'], 0, R_np.ravel().astype(np.float32).tobytes())

    if rs_coeffs_np is not None and np.any(rs_coeffs_np != 0):
        q.write_buffer(pool['rs_buf'], 0, rs_coeffs_np.astype(np.float32).tobytes())
    else:
        q.write_buffer(pool['rs_buf'], 0, b'\x00' * 12)

    if has_iori and R_cross_np is not None:
        q.write_buffer(pool['Rc_buf'], 0, R_cross_np.ravel().astype(np.float32).tobytes())
        q.write_buffer(pool['params_buf'], 0, np.array([1.0], dtype=np.float32).tobytes())
    else:
        q.write_buffer(pool['params_buf'], 0, b'\x00' * 4)

    # Upload cross image (~46MB — the main cost, but no buffer allocation)
    cross_bytes = np.ascontiguousarray(cross_np).tobytes()
    pad = (4 - len(cross_bytes) % 4) % 4
    if pad:
        cross_bytes += b'\x00' * pad
    q.write_buffer(pool['cross_buf'], 0, cross_bytes)

    # Dispatch (reuse cached bind group)
    encoder = _wgpu_device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass()
    pass_enc.set_pipeline(_wgpu_cross_pipeline)
    pass_enc.set_bind_group(0, pool['bind_group'])
    _wg_x = (n_pixels + 255) // 256
    if _wg_x > 65535:
        pass_enc.dispatch_workgroups(65535, (_wg_x + 65534) // 65535, 1)
    else:
        pass_enc.dispatch_workgroups(_wg_x, 1, 1)
    pass_enc.end()
    _wgpu_device.queue.submit([encoder.finish()])

    # Readback — detect dtype from cross input
    _out_dtype = cross_np.dtype
    _out_pm = 65535 if _out_dtype == np.uint16 else 255
    raw = _wgpu_device.queue.read_buffer(out_buf)
    result = np.frombuffer(raw, dtype=np.float32)[:n_pixels * 3]
    return np.clip(result, 0, _out_pm).astype(_out_dtype)

# ── wgpu persistent buffer pool for equirect remap (non-360) ──
_wgpu_equirect_pool = None

def _wgpu_ensure_equirect_pool(n_pixels, src_w, src_h, xyz_x_buf, xyz_y_buf, xyz_z_buf, out_buf):
    """Ensure persistent buffer pool for equirect remap. Create once, reuse every frame."""
    global _wgpu_equirect_pool
    if (_wgpu_equirect_pool is not None and
            _wgpu_equirect_pool['n_pixels'] == n_pixels and
            _wgpu_equirect_pool['src_w'] == src_w and
            _wgpu_equirect_pool['src_h'] == src_h):
        return _wgpu_equirect_pool

    src_size = (src_h * src_w * 3 + 3) // 4 * 4
    pool = {
        'n_pixels': n_pixels,
        'src_w': src_w,
        'src_h': src_h,
        'R_buf': _wgpu_device.create_buffer(size=9*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'dims_buf': _wgpu_device.create_buffer(size=2*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'src_buf': _wgpu_device.create_buffer(size=src_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        'n_buf': _wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
    }
    _wgpu_device.queue.write_buffer(pool['n_buf'], 0, np.array([n_pixels], dtype=np.uint32).tobytes())
    _wgpu_device.queue.write_buffer(pool['dims_buf'], 0,
                                     np.array([float(src_w), float(src_h)], dtype=np.float32).tobytes())

    pool['bind_group'] = _wgpu_device.create_bind_group(
        layout=_wgpu_equirect_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": xyz_x_buf}},
            {"binding": 1, "resource": {"buffer": xyz_y_buf}},
            {"binding": 2, "resource": {"buffer": xyz_z_buf}},
            {"binding": 3, "resource": {"buffer": pool['R_buf']}},
            {"binding": 4, "resource": {"buffer": pool['dims_buf']}},
            {"binding": 5, "resource": {"buffer": pool['src_buf']}},
            {"binding": 6, "resource": {"buffer": pool['n_buf']}},
            {"binding": 7, "resource": {"buffer": out_buf}},
        ],
    )
    _wgpu_equirect_pool = pool
    return pool

def _wgpu_process_eye_equirect(src_np, xyz_x_buf, xyz_y_buf, xyz_z_buf,
                                R_np, n_pixels, out_buf):
    """Run wgpu equirect remap kernel for one eye. Returns (n_pixels*3,) uint8 numpy array."""
    src_h, src_w = src_np.shape[:2]
    pool = _wgpu_ensure_equirect_pool(n_pixels, src_w, src_h, xyz_x_buf, xyz_y_buf, xyz_z_buf, out_buf)
    q = _wgpu_device.queue

    q.write_buffer(pool['R_buf'], 0, R_np.ravel().astype(np.float32).tobytes())

    src_bytes = np.ascontiguousarray(src_np).tobytes()
    pad = (4 - len(src_bytes) % 4) % 4
    if pad:
        src_bytes += b'\x00' * pad
    q.write_buffer(pool['src_buf'], 0, src_bytes)

    encoder = _wgpu_device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass()
    pass_enc.set_pipeline(_wgpu_equirect_pipeline)
    pass_enc.set_bind_group(0, pool['bind_group'])
    _wg_x = (n_pixels + 255) // 256
    if _wg_x > 65535:
        pass_enc.dispatch_workgroups(65535, (_wg_x + 65534) // 65535, 1)
    else:
        pass_enc.dispatch_workgroups(_wg_x, 1, 1)
    pass_enc.end()
    _wgpu_device.queue.submit([encoder.finish()])

    raw = _wgpu_device.queue.read_buffer(out_buf)
    result = np.frombuffer(raw, dtype=np.float32)[:n_pixels * 3]
    _out_dtype = src_np.dtype
    _out_pm = 65535 if _out_dtype == np.uint16 else 255
    return np.clip(result, 0, _out_pm).astype(_out_dtype)

# ── wgpu persistent buffer pool for LUT ──
_wgpu_lut_pool = None

def _wgpu_apply_lut_3d(frame_np, lut_flat_np, lut_size, n_pixels):
    """Apply 3D LUT via wgpu compute kernel. Reuses persistent buffers."""
    global _wgpu_lut_pool
    q = _wgpu_device.queue

    # Ensure pool exists and is sized correctly
    frame_bytes_needed = (n_pixels * 3 + 3) // 4 * 4
    lut_bytes_needed = len(lut_flat_np) * 4
    need_rebuild = (_wgpu_lut_pool is None or
                    _wgpu_lut_pool['n_pixels'] != n_pixels or
                    _wgpu_lut_pool['lut_bytes'] != lut_bytes_needed)

    if need_rebuild:
        _wgpu_lut_pool = {
            'n_pixels': n_pixels,
            'lut_bytes': lut_bytes_needed,
            'frame_buf': _wgpu_device.create_buffer(size=frame_bytes_needed,
                         usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            'lut_buf': _wgpu_device.create_buffer(size=lut_bytes_needed,
                       usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            'params_buf': _wgpu_device.create_buffer(size=3*4,
                          usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            'n_buf': _wgpu_device.create_buffer(size=4,
                     usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            'out_buf': _wgpu_device.create_buffer(size=n_pixels*3*4,
                       usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST),
        }
        q.write_buffer(_wgpu_lut_pool['n_buf'], 0, np.array([n_pixels], dtype=np.uint32).tobytes())
        q.write_buffer(_wgpu_lut_pool['lut_buf'], 0, lut_flat_np.astype(np.float32).tobytes())

        _wgpu_lut_pool['bind_group'] = _wgpu_device.create_bind_group(
            layout=_wgpu_lut_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": _wgpu_lut_pool['frame_buf']}},
                {"binding": 1, "resource": {"buffer": _wgpu_lut_pool['lut_buf']}},
                {"binding": 2, "resource": {"buffer": _wgpu_lut_pool['params_buf']}},
                {"binding": 3, "resource": {"buffer": _wgpu_lut_pool['n_buf']}},
                {"binding": 4, "resource": {"buffer": _wgpu_lut_pool['out_buf']}},
            ],
        )

    pool = _wgpu_lut_pool

    # Upload frame (per-frame) and params
    frame_bytes = np.ascontiguousarray(frame_np).tobytes()
    pad = (4 - len(frame_bytes) % 4) % 4
    if pad:
        frame_bytes += b'\x00' * pad
    q.write_buffer(pool['frame_buf'], 0, frame_bytes)
    _wgpu_pm = 65535.0 if frame_np.dtype == np.uint16 else 255.0
    q.write_buffer(pool['params_buf'], 0,
                   np.array([(lut_size - 1) / _wgpu_pm, float(lut_size), _wgpu_pm], dtype=np.float32).tobytes())

    # Dispatch
    encoder = _wgpu_device.create_command_encoder()
    pass_enc = encoder.begin_compute_pass()
    pass_enc.set_pipeline(_wgpu_lut_pipeline)
    pass_enc.set_bind_group(0, pool['bind_group'])
    _wg_x = (n_pixels + 255) // 256
    if _wg_x > 65535:
        pass_enc.dispatch_workgroups(65535, (_wg_x + 65534) // 65535, 1)
    else:
        pass_enc.dispatch_workgroups(_wg_x, 1, 1)
    pass_enc.end()
    _wgpu_device.queue.submit([encoder.finish()])

    raw = _wgpu_device.queue.read_buffer(pool['out_buf'])
    result = np.frombuffer(raw, dtype=np.float32)[:n_pixels * 3]
    _pm = 65535 if frame_np.dtype == np.uint16 else 255
    return np.clip(result, 0, _pm).astype(frame_np.dtype).reshape(frame_np.shape)

# Windows-specific flag to hide console windows
def get_subprocess_flags():
    """Get subprocess creation flags to hide console windows on Windows"""
    if sys.platform == 'win32':
        import subprocess
        return subprocess.CREATE_NO_WINDOW
    return 0

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QGroupBox, QProgressBar,
    QMessageBox, QSplitter, QLineEdit, QStatusBar, QFrame,
    QScrollArea, QToolButton, QCheckBox, QRadioButton, QButtonGroup, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter


def get_ffmpeg_path():
    """Get the path to bundled ffmpeg or system ffmpeg"""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check multiple possible locations
        base_path = Path(sys._MEIPASS)

        # Try _internal folder (Windows/Linux style)
        # On Windows, add .exe extension
        ffmpeg_name = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
        ffmpeg_path = base_path / ffmpeg_name
        if ffmpeg_path.exists():
            return str(ffmpeg_path)

        # Try macOS app bundle Resources folder
        if sys.platform == 'darwin':
            # Go up from _MEIPASS to find Resources
            resources_path = base_path.parent / 'Resources' / 'ffmpeg'
            if resources_path.exists():
                return str(resources_path)

            # Try Frameworks folder
            frameworks_path = base_path.parent / 'Frameworks' / 'ffmpeg'
            if frameworks_path.exists():
                return str(frameworks_path)

    # Check for ffmpeg in system PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    # Last resort: return just 'ffmpeg' and hope it's in PATH
    return 'ffmpeg'


def get_ffprobe_path():
    """Get the path to bundled ffprobe or system ffprobe"""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check multiple possible locations
        base_path = Path(sys._MEIPASS)

        # Try _internal folder (Windows/Linux style)
        # On Windows, add .exe extension
        ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
        ffprobe_path = base_path / ffprobe_name
        if ffprobe_path.exists():
            return str(ffprobe_path)

        # Try macOS app bundle Resources folder
        if sys.platform == 'darwin':
            # Go up from _MEIPASS to find Resources
            resources_path = base_path.parent / 'Resources' / 'ffprobe'
            if resources_path.exists():
                return str(resources_path)

            # Try Frameworks folder
            frameworks_path = base_path.parent / 'Frameworks' / 'ffprobe'
            if frameworks_path.exists():
                return str(frameworks_path)

    # Check for ffprobe in system PATH
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        return ffprobe

    # Last resort: return just 'ffprobe' and hope it's in PATH
    return 'ffprobe'


def get_mvhevc_encode_path():
    """Get the path to the bundled mvhevc_encode helper binary.

    mvhevc_encode is a Swift helper that takes raw BGR48LE SBS frames on
    stdin and produces a MV-HEVC .mov with spatial metadata directly via
    VideoToolbox (VTCompressionSessionEncodeMultiImageFrame). It replaces
    the prior external `spatial` CLI dependency — faster (no re-encode)
    and self-contained (no Homebrew install required).

    macOS Apple Silicon only. Returns None on other platforms.
    """
    if sys.platform != 'darwin':
        return None
    # PyInstaller-frozen bundle
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
        for candidate in [
            base_path / 'mvhevc_encode',
            base_path.parent / 'Resources' / 'mvhevc_encode',
            base_path.parent / 'Frameworks' / 'mvhevc_encode',
        ]:
            if candidate.exists():
                return str(candidate)
    # Development: sibling to this script
    dev_path = Path(__file__).resolve().parent / 'mvhevc_encode'
    if dev_path.exists():
        return str(dev_path)
    return None


BUNDLED_LUT_FILENAME = "Recommended Lut GPLOG.cube"


def get_bundled_lut_path():
    """Return the absolute path to the bundled recommended LUT file,
    or None if it can't be located.

    Resolution order:
      1. PyInstaller frozen bundle — ``sys._MEIPASS`` (Windows / Linux / onefile)
         or macOS ``.app`` ``Resources`` / ``Frameworks`` folders.
      2. Development / script mode — sibling file next to ``vr180_gui.py``.
    """
    # 1. Frozen (PyInstaller) bundle
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
        candidate = base_path / BUNDLED_LUT_FILENAME
        if candidate.exists():
            return str(candidate)

        if sys.platform == 'darwin':
            # macOS .app bundle layouts: Resources/ and Frameworks/
            resources_candidate = base_path.parent / 'Resources' / BUNDLED_LUT_FILENAME
            if resources_candidate.exists():
                return str(resources_candidate)
            frameworks_candidate = base_path.parent / 'Frameworks' / BUNDLED_LUT_FILENAME
            if frameworks_candidate.exists():
                return str(frameworks_candidate)

    # 2. Development / script mode — next to vr180_gui.py
    try:
        script_dir = Path(__file__).resolve().parent
    except Exception:
        script_dir = Path.cwd()
    candidate = script_dir / BUNDLED_LUT_FILENAME
    if candidate.exists():
        return str(candidate)

    return None


# Cache for detected hardware encoders
_hw_encoder_cache = None

def detect_hardware_encoders():
    """Detect available hardware encoders on the system.

    Returns a dict with:
        'h264': encoder name or None
        'h265': encoder name or None
        'type': 'nvidia', 'amd', 'intel', 'videotoolbox', or None
    """
    global _hw_encoder_cache
    if _hw_encoder_cache is not None:
        return _hw_encoder_cache

    result = {'h264': None, 'h265': None, 'type': None}

    if sys.platform == 'darwin':
        # macOS uses VideoToolbox
        result['h264'] = 'h264_videotoolbox'
        result['h265'] = 'hevc_videotoolbox'
        result['type'] = 'videotoolbox'
        _hw_encoder_cache = result
        return result

    if sys.platform != 'win32':
        _hw_encoder_cache = result
        return result

    # Windows: Check for available encoders by querying ffmpeg
    try:
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, '-hide_banner', '-encoders']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                            creationflags=get_subprocess_flags())
        encoders_output = proc.stdout

        # Check NVIDIA NVENC (most common)
        if 'hevc_nvenc' in encoders_output:
            result['h265'] = 'hevc_nvenc'
            result['type'] = 'nvidia'
        if 'h264_nvenc' in encoders_output:
            result['h264'] = 'h264_nvenc'
            if result['type'] is None:
                result['type'] = 'nvidia'

        # Check AMD AMF (if no NVIDIA found)
        if result['h265'] is None and 'hevc_amf' in encoders_output:
            result['h265'] = 'hevc_amf'
            result['type'] = 'amd'
        if result['h264'] is None and 'h264_amf' in encoders_output:
            result['h264'] = 'h264_amf'
            if result['type'] is None:
                result['type'] = 'amd'

        # Check Intel QuickSync (if no NVIDIA/AMD found)
        if result['h265'] is None and 'hevc_qsv' in encoders_output:
            result['h265'] = 'hevc_qsv'
            result['type'] = 'intel'
        if result['h264'] is None and 'h264_qsv' in encoders_output:
            result['h264'] = 'h264_qsv'
            if result['type'] is None:
                result['type'] = 'intel'

        print(f"✓ Detected hardware encoders: {result['type']} - H.264: {result['h264']}, H.265: {result['h265']}")

    except Exception as e:
        print(f"⚠ Warning: Could not detect hardware encoders: {e}")

    _hw_encoder_cache = result
    return result


def get_hw_encoder_args(codec: str, encoder_type: str, quality: int, bitrate: int, use_bitrate: bool, bit_depth: int = 8):
    """Get FFmpeg encoder arguments based on encoder type selection.

    Args:
        codec: 'h264' or 'h265'
        encoder_type: 'auto', 'nvenc', 'qsv', 'amf', 'software', or 'videotoolbox'
        quality: CRF/CQ value
        bitrate: Target bitrate in Mbps
        use_bitrate: Whether to use bitrate mode instead of quality
        bit_depth: 8 or 10 bit

    Returns:
        List of FFmpeg encoder arguments
    """
    # Determine which encoder to use
    if encoder_type == 'auto':
        hw = detect_hardware_encoders()
        hw_type = hw.get('type')
        if hw_type:
            encoder_type = {'videotoolbox': 'videotoolbox', 'nvidia': 'nvenc', 'amd': 'amf', 'intel': 'qsv'}.get(hw_type, 'software')
        else:
            encoder_type = 'software'

    # macOS VideoToolbox
    if encoder_type == 'videotoolbox':
        hw_encoder = 'hevc_videotoolbox' if codec == 'h265' else 'h264_videotoolbox'
        if use_bitrate:
            enc = ["-c:v", hw_encoder, "-b:v", f"{bitrate}M"]
        else:
            enc = ["-c:v", hw_encoder, "-q:v", str(min(100, quality * 2))]
        if codec == 'h265':
            enc.extend(["-tag:v", "hvc1"])
            if bit_depth == 10:
                enc.extend(["-pix_fmt", "p010le"])
        return enc

    # NVIDIA NVENC
    if encoder_type == 'nvenc':
        hw_encoder = 'hevc_nvenc' if codec == 'h265' else 'h264_nvenc'
        if use_bitrate:
            enc = ["-c:v", hw_encoder, "-preset", "p4", "-rc", "vbr",
                   "-b:v", f"{bitrate}M", "-multipass", "qres"]
        else:
            enc = ["-c:v", hw_encoder, "-preset", "p4", "-rc", "vbr",
                   "-cq", str(quality), "-b:v", "0", "-multipass", "qres"]
        if codec == 'h265':
            enc.extend(["-tag:v", "hvc1"])
            if bit_depth == 10:
                enc.extend(["-pix_fmt", "p010le"])
        return enc

    # AMD AMF
    if encoder_type == 'amf':
        hw_encoder = 'hevc_amf' if codec == 'h265' else 'h264_amf'
        if use_bitrate:
            enc = ["-c:v", hw_encoder, "-quality", "balanced", "-rc", "vbr_peak", "-b:v", f"{bitrate}M"]
        else:
            enc = ["-c:v", hw_encoder, "-quality", "balanced", "-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality), "-qp_b", str(quality)]
        if codec == 'h265':
            enc.extend(["-tag:v", "hvc1"])
            if bit_depth == 10:
                enc.extend(["-pix_fmt", "p010le"])
        return enc

    # Intel QuickSync
    if encoder_type == 'qsv':
        hw_encoder = 'hevc_qsv' if codec == 'h265' else 'h264_qsv'
        if use_bitrate:
            enc = ["-c:v", hw_encoder, "-preset", "medium", "-b:v", f"{bitrate}M"]
        else:
            enc = ["-c:v", hw_encoder, "-preset", "medium", "-global_quality", str(quality)]
        if codec == 'h265':
            enc.extend(["-tag:v", "hvc1"])
            if bit_depth == 10:
                enc.extend(["-pix_fmt", "p010le"])
        return enc

    # Software encoding (fallback)
    if codec == 'h265':
        pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
        if use_bitrate:
            enc = ["-c:v", "libx265", "-b:v", f"{bitrate}M", "-preset", "fast", "-tag:v", "hvc1"]
        else:
            enc = ["-c:v", "libx265", "-crf", str(quality), "-preset", "fast", "-tag:v", "hvc1"]
        enc.extend(["-pix_fmt", pix_fmt])
        if bit_depth == 10:
            enc.extend(["-profile:v", "main10"])
    else:  # h264
        if use_bitrate:
            enc = ["-c:v", "libx264", "-b:v", f"{bitrate}M", "-preset", "fast"]
        else:
            enc = ["-c:v", "libx264", "-crf", str(quality), "-preset", "fast"]

    return enc


def load_cube_lut(lut_path: str) -> np.ndarray:
    """Load a .cube LUT file and return as 3D numpy array."""
    lut_data = []
    lut_size = None

    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('TITLE'):
                continue
            if line.startswith('LUT_3D_SIZE'):
                lut_size = int(line.split()[1])
                continue
            if line.startswith('DOMAIN_MIN') or line.startswith('DOMAIN_MAX'):
                continue

            # Parse RGB values
            parts = line.split()
            if len(parts) >= 3:
                try:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                    lut_data.append([r, g, b])
                except ValueError:
                    continue

    if lut_size is None:
        # Try to determine size from data length
        data_len = len(lut_data)
        lut_size = int(round(data_len ** (1/3)))

    # Reshape to 3D array [R, G, B, 3]
    lut_array = np.array(lut_data, dtype=np.float32)
    lut_3d = lut_array.reshape((lut_size, lut_size, lut_size, 3))

    return lut_3d



# ─── Quaternion Math Utilities ───────────────────────────────────────────────

def quat_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.

    Uses ZYX (Tait-Bryan) convention commonly used in aerospace/IMU:
    - Roll: rotation around Z-axis (forward)
    - Pitch: rotation around X-axis (right)
    - Yaw: rotation around Y-axis (up)

    Returns: (roll_deg, pitch_deg, yaw_deg)
    """
    import math
    # Roll (z-axis rotation)
    sinr_cosp = 2.0 * (w * z + x * y)
    cosr_cosp = 1.0 - 2.0 * (y * y + z * z)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (x-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (y-axis rotation)
    siny_cosp = 2.0 * (w * x + y * z)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (roll, pitch, yaw) in degrees to quaternion (w, x, y, z).

    Uses ZYX (Tait-Bryan) convention matching quat_to_euler.
    """
    import math
    roll = math.radians(roll_deg) / 2.0
    pitch = math.radians(pitch_deg) / 2.0
    yaw = math.radians(yaw_deg) / 2.0

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    w = cr * cp * cy + sr * sp * sy
    x = cr * cp * sy - sr * sp * cy
    y = cr * sp * cy + sr * cp * sy
    z = sr * cp * cy - cr * sp * sy

    return (w, x, y, z)


def quat_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2. Each is (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )


def quat_inverse(q):
    """Return the inverse (conjugate) of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_to_rotation_matrix(w, x, y, z):
    """Convert unit quaternion to 3x3 rotation matrix (numpy array)."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def _quat_slerp_array(q1, q2, t):
    """SLERP for numpy arrays (w, x, y, z). Returns numpy array."""
    import math
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result /= np.linalg.norm(result)
        return result
    theta_0 = math.acos(min(dot, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return s1 * q1 + s2 * q2


def quat_from_axis_angle(axis, angle_rad):
    """Create quaternion (w, x, y, z) from axis and angle in radians."""
    import math
    ha = angle_rad * 0.5
    s = math.sin(ha)
    return (math.cos(ha), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_decompose_roll(q):
    """Decompose quaternion q into yaw-pitch and roll components.

    Returns (q_yp, roll_angle_rad) where:
      q_yp = yaw+pitch quaternion (rotation with zero roll around Z)
      roll_angle_rad = roll angle around Z axis
      q ≈ q_yp × q_roll  (q_roll = rotation around Z by roll_angle_rad)

    Uses the ZYX Euler decomposition. Roll = rotation around Z (optical axis).
    """
    import math
    w, x, y, z = q
    # Extract roll (Z-axis rotation) from ZYX Euler
    # roll = atan2(2(wx + yz), 1 - 2(x² + y²))  — but this is the X-axis Euler for XYZ
    # For GoPro: roll around Z (optical axis) in their body frame
    # Standard aerospace ZYX: first rotate yaw(Z), then pitch(Y), then roll(X)
    # But our "roll" is rotation around optical axis = Z in camera frame
    # Extract the Z-component rotation:
    # For a quaternion, the Z-axis Euler angle (yaw in ZYX) is:
    #   yaw_z = atan2(2(wz + xy), 1 - 2(y² + z²))
    # But we want to decompose as q = q_yp × q_roll where q_roll is pure Z rotation
    # q_roll = q_yp⁻¹ × q

    # Method: project quaternion onto Z-axis rotation subspace
    # A pure Z rotation quaternion has form (cos(θ/2), 0, 0, sin(θ/2))
    # The Z-rotation component of q is found by:
    roll_angle = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Construct pure roll quaternion
    q_roll = quat_from_axis_angle((0, 0, 1), roll_angle)
    # q_yp = q × q_roll⁻¹
    q_yp = quat_multiply(q, quat_inverse(q_roll))

    return q_yp, roll_angle


def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    import math
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute dot product
    dot = w1*w2 + x1*x2 + y1*y2 + z1*z2

    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        w = w1 + t * (w2 - w1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        # Normalize
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        return (w/norm, x/norm, y/norm, z/norm)

    theta_0 = math.acos(min(dot, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    w = s1 * w1 + s2 * w2
    x = s1 * x1 + s2 * x2
    y = s1 * y1 + s2 * y2
    z = s1 * z1 + s2 * z2

    return (w, x, y, z)


# ─── GEOC (Geometry Calibrations) Parser ─────────────────────────────────────

def parse_geoc(file_path: str) -> dict:
    """Parse GEOC (Geometry Calibrations) from GoPro .360 file tail.

    Extracts factory lens calibration data including:
      - Per-lens KLNS: [c0, c1, c2, c3, c4] Kannala-Brandt polynomial coefficients
        r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹
      - Per-lens CTRX, CTRY: principal point offset from sensor center (pixels)
      - Global CALW, CALH: calibration sensor dimensions (4216 for GoPro MAX)
      - Global ANGX, ANGY, ANGZ: inter-lens angular offsets

    Returns dict with keys: 'global', 'BACK', 'FRNT'.
    The FRNT lens = right eye after yaw mod.
    """
    file_size = os.path.getsize(file_path)
    tail_size = 1024 * 1024
    with open(file_path, 'rb') as f:
        f.seek(max(0, file_size - tail_size))
        tail = f.read()
    idx = tail.find(b'GEOC')
    if idx < 0:
        return None  # GEOC not found — not a calibrated GoPro file

    result = {'global': {}, 'BACK': {}, 'FRNT': {}}
    current = 'global'
    pos = idx - 8
    end = min(idx + 2000, len(tail))
    while pos + 8 <= end:
        fourcc = tail[pos:pos+4].decode('ascii', errors='replace')
        tc = chr(tail[pos+4])
        ss = tail[pos+5]
        cnt = struct.unpack('>H', tail[pos+6:pos+8])[0]
        payload_size = ss * cnt
        padded = (payload_size + 3) & ~3
        payload = tail[pos+8:pos+8+payload_size]
        if fourcc == 'DEVC':
            pos += 8
            continue
        val = None
        if tc == 'd' and ss == 8:
            vals = [struct.unpack('>d', payload[i*8:(i+1)*8])[0] for i in range(cnt)]
            val = vals[0] if cnt == 1 else vals
        elif tc == 'L' and ss == 4:
            vals = [struct.unpack('>I', payload[i*4:(i+1)*4])[0] for i in range(cnt)]
            val = vals[0] if cnt == 1 else vals
        elif tc == 'F' and ss == 4:
            val = payload[:4].decode('ascii', errors='replace')
        elif tc == 'c':
            val = payload.decode('ascii', errors='replace').rstrip('\x00')
        elif fourcc == 'KLNS' and ss == 40 and cnt == 1:
            val = [struct.unpack('>d', payload[i*8:(i+1)*8])[0] for i in range(5)]
        if fourcc == 'DVID' and isinstance(val, str):
            if val in ('BACK', 'FRNT'):
                current = val
            elif val in ('USRM', 'HLMT'):
                current = None
        if current is not None and val is not None:
            result[current][fourcc] = val
        pos += 8 + padded
    return result


# ─── GoPro Multi-Segment Detection ──────────────────────────────────────────

def detect_gopro_segments(file_path: str) -> list:
    """Detect GoPro multi-segment recording files from a single segment.

    GoPro naming patterns (cc=chapter, IIII=recording id):
      .360:  GS01XXXX.360, GH01XXXX.360
      .MP4:  GX01XXXX.MP4, GH01XXXX.MP4, GP01XXXX.MP4
      .LRV:  GL01XXXX.LRV
      .MOV:  same prefixes with .MOV extension

    Given any segment, returns sorted list of all segment paths in order.
    Returns [file_path] if no other segments found (single-segment recording).
    """
    import re
    p = Path(file_path)
    name = p.name
    # Match GoPro naming: G[SHXLP]ccIIII.ext where cc=chapter, IIII=recording id
    m = re.match(r'^(G[SHXLP])(\d{2})(\d{4})(\.\w+)$', name, re.IGNORECASE)
    if not m:
        return [str(p)]

    prefix, _chapter, rec_id, ext = m.groups()
    # Find all segments with same prefix+recording_id, any chapter number
    segments = []
    parent = p.parent
    for chapter in range(1, 100):
        seg_name = f"{prefix}{chapter:02d}{rec_id}{ext}"
        seg_path = parent / seg_name
        if seg_path.exists():
            segments.append(str(seg_path))
        elif chapter > int(_chapter) + 5:
            # Stop searching after 5 consecutive missing chapters beyond current
            break
    if not segments:
        segments = [str(p)]
    return sorted(segments)


def concatenate_gyro_data(segment_paths: list) -> dict:
    """Parse and concatenate gyro data from multiple GoPro segments.

    Timestamps are adjusted so each segment continues from where the previous ended.
    Returns a single unified gyro_data dict compatible with GyroStabilizer.

    For no-firmware-RS clips (CORI generated by gyro integration), all
    segments' raw gyro samples are aggregated into a single continuous
    stream and integrated as one — eliminating any orientation discontinuity
    at segment boundaries that per-segment integration would introduce.
    For firmware-on clips (real CORI baked in by the camera), the existing
    per-segment parse + concatenate path is used unchanged.
    """
    if len(segment_paths) == 1:
        return parse_gopro_gyro_data(segment_paths[0])

    # ── Pre-pass: probe each segment for its actual frame count and detect
    # whether this recording is no-firmware-RS (VQF mode). The per-segment
    # frame count comes from ffprobe (the same source the existing slicing
    # logic in parse_gopro_gyro_data uses), so it correctly handles
    # multi-chapter files where every chapter's GPMF stores the entire
    # recording's CORI. ──
    n_chapter_frames_list = []
    fps = None
    srot_ms = None
    for seg_path in segment_paths:
        try:
            probe = subprocess.run([
                get_ffprobe_path(), '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames,duration,r_frame_rate',
                '-of', 'json', seg_path
            ], capture_output=True, text=True, creationflags=get_subprocess_flags())
            info = json.loads(probe.stdout)
            stream = info['streams'][0]
            if fps is None:
                fps_str = stream.get('r_frame_rate', '30/1')
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            nb_str = stream.get('nb_frames', '')
            if nb_str and nb_str != 'N/A':
                n = int(nb_str)
            elif stream.get('duration'):
                n = int(float(stream['duration']) * fps + 0.5)
            else:
                n = 0
            n_chapter_frames_list.append(n)
        except Exception as e:
            print(f"  Warning: ffprobe failed on {seg_path}: {e}")
            n_chapter_frames_list.append(0)

    # Parse the first segment in full to detect VQF mode and capture srot_ms.
    # parse_gopro_gyro_data already does the bias-drift detection on the
    # un-sliced full CORI (which is the only place that signal is reliable
    # in multi-chapter files).
    first_gyro = parse_gopro_gyro_data(segment_paths[0])
    is_vqf_mode = bool(first_gyro.get('cori_is_vqf', False))
    if fps is None:
        fps = first_gyro.get('fps')
    srot_ms = first_gyro.get('srot_ms')

    # ── VQF / no-firmware-RS path: aggregate gyro across segments, then
    # integrate ONCE as a continuous stream. ──
    if is_vqf_mode:
        from parse_gyro_raw import vqf_to_cori_quats_multi_segment

        # Concatenate per-segment GRAV samples for the smoother (so
        # GRAV-alignment can use the very first frame's gravity vector).
        combined_grav = []
        # Use first segment's grav as a fast path (already parsed above)
        if first_gyro.get('grav_samples'):
            combined_grav.extend(first_gyro['grav_samples'])
        for seg_idx, seg_path in enumerate(segment_paths[1:], start=1):
            try:
                seg_gyro = parse_gopro_gyro_data(seg_path)
                grav = seg_gyro.get('grav_samples') or []
                combined_grav.extend(grav)
            except Exception as e:
                print(f"  Warning: parse_gopro_gyro_data failed on {seg_path}: {e}")

        try:
            multi_result = vqf_to_cori_quats_multi_segment(segment_paths, n_chapter_frames_list)
        except Exception as e:
            print(f"  vqf_to_cori_quats_multi_segment failed: {e}, falling back to per-segment chaining")
            return _concatenate_gyro_data_chaining_fallback(segment_paths)

        # Build segment_boundaries on the unified frame index timeline so
        # downstream code (e.g. concatenate_800hz_gyro) knows where each
        # segment lives in the merged result.
        segment_boundaries = []
        cum_frames = 0
        cum_time = 0.0
        for seg_idx, seg_path in enumerate(segment_paths):
            n_seg = n_chapter_frames_list[seg_idx]
            segment_boundaries.append((cum_frames, cum_frames + n_seg, seg_path, cum_time))
            cum_frames += n_seg
            cum_time += n_seg / fps

        result = {
            'fps': fps,
            'frames': multi_result['frames'],
            'srot_ms': srot_ms if srot_ms is not None else multi_result.get('srot_ms'),
            'grav_samples': combined_grav,
            'source': multi_result.get('source', 'gyro_integration'),
            'segment_boundaries': segment_boundaries,
            'segment_paths': segment_paths,
            'cori_is_vqf': True,
            'gyro_bias_deg_s': multi_result.get('gyro_bias_deg_s', (0, 0, 0)),
        }
        return result

    # ── Firmware-on path: per-segment parse + concatenate (unchanged) ──
    all_frames = []
    combined_grav = []
    time_offset = 0.0
    segment_boundaries = []

    # Re-use the first segment we already parsed
    for seg_idx, seg_path in enumerate(segment_paths):
        gyro = first_gyro if seg_idx == 0 else parse_gopro_gyro_data(seg_path)
        if fps is None:
            fps = gyro['fps']
        if srot_ms is None:
            srot_ms = gyro.get('srot_ms')

        frames = gyro['frames']
        if not frames:
            continue

        seg_start_idx = len(all_frames)
        for f in frames:
            f_copy = dict(f)
            f_copy['time'] = f['time'] + time_offset
            all_frames.append(f_copy)

        seg_duration = frames[-1]['time'] + (1.0 / fps) if frames else 0
        segment_boundaries.append((seg_start_idx, len(all_frames), seg_path, time_offset))
        time_offset += seg_duration

        grav = gyro.get('grav_samples', [])
        for g in grav:
            g_copy = dict(g) if isinstance(g, dict) else g
            combined_grav.append(g_copy)

    result = {
        'fps': fps,
        'frames': all_frames,
        'srot_ms': srot_ms,
        'grav_samples': combined_grav,
        'source': 'cori',
        'segment_boundaries': segment_boundaries,
        'segment_paths': segment_paths,
        'cori_is_vqf': False,
    }
    return result


def _concatenate_gyro_data_chaining_fallback(segment_paths: list) -> dict:
    """Fallback path: per-segment VQF parse + quaternion chaining at boundaries.

    Used only if vqf_to_cori_quats_multi_segment fails for some reason.
    The combined-stream integration in vqf_to_cori_quats_multi_segment is the
    preferred path because it eliminates any boundary discontinuity.
    """
    all_frames = []
    combined_grav = []
    time_offset = 0.0
    fps = None
    srot_ms = None
    segment_boundaries = []
    running_q = None

    for seg_idx, seg_path in enumerate(segment_paths):
        gyro = parse_gopro_gyro_data(seg_path)
        if fps is None:
            fps = gyro['fps']
        if srot_ms is None:
            srot_ms = gyro.get('srot_ms')

        frames = gyro['frames']
        if not frames:
            continue

        seg_start_idx = len(all_frames)
        if running_q is not None:
            for f in frames:
                f_copy = dict(f)
                f_copy['time'] = f['time'] + time_offset
                qf = f['cori_quat']
                qabs = quat_multiply(running_q, qf)
                f_copy['cori_quat'] = qabs
                f_copy['cori_euler'] = quat_to_euler(*qabs)
                all_frames.append(f_copy)
        else:
            for f in frames:
                f_copy = dict(f)
                f_copy['time'] = f['time'] + time_offset
                all_frames.append(f_copy)

        if all_frames:
            running_q = all_frames[-1]['cori_quat']

        seg_duration = frames[-1]['time'] + (1.0 / fps) if frames else 0
        segment_boundaries.append((seg_start_idx, len(all_frames), seg_path, time_offset))
        time_offset += seg_duration

        grav = gyro.get('grav_samples', [])
        for g in grav:
            g_copy = dict(g) if isinstance(g, dict) else g
            combined_grav.append(g_copy)

    return {
        'fps': fps,
        'frames': all_frames,
        'srot_ms': srot_ms,
        'grav_samples': combined_grav,
        'source': 'gyro_integration',
        'segment_boundaries': segment_boundaries,
        'segment_paths': segment_paths,
        'cori_is_vqf': True,
    }


def concatenate_800hz_gyro(segment_paths: list, fps: float, combined_gyro_data: dict) -> tuple:
    """Parse and concatenate 800Hz GYRO angular velocity from multiple segments.

    Returns concatenated (gyro_times, gyro_angvel) with continuous timestamps.
    """
    import sys, os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from parse_gyro_raw import get_gyro_angular_velocity

    all_times = []
    all_angvel = []
    time_offset = 0.0
    boundaries = combined_gyro_data.get('segment_boundaries', [])

    for seg_idx, seg_path in enumerate(segment_paths):
        # Count frames in this segment
        if seg_idx < len(boundaries):
            start_idx, end_idx, _, t_off = boundaries[seg_idx]
            n_frames = end_idx - start_idx
            time_offset = t_off
        else:
            n_frames = 0
            continue

        gyro_times, gyro_angvel = get_gyro_angular_velocity(seg_path, fps, n_frames)
        if gyro_times is not None and len(gyro_times) > 0:
            all_times.append(gyro_times + time_offset)
            all_angvel.append(gyro_angvel)

    if not all_times:
        return None, None

    return np.concatenate(all_times), np.concatenate(all_angvel)


# ─── Full GPMF Gyro Data Parser ─────────────────────────────────────────────

def parse_gopro_gyro_data(file_path: str) -> dict:
    """Parse GoPro .360 file to extract full CORI and IORI quaternion data.

    CORI (Camera Orientation): Physical camera orientation (raw motion)
    IORI (Image Orientation): How the image was rotated by GoPro stabilization

    Returns dict with:
        'fps': float - video frame rate
        'frames': list of dicts with keys:
            'time': float - timestamp in seconds
            'cori_quat': (w, x, y, z) - camera orientation quaternion
            'cori_euler': (roll, pitch, yaw) - camera orientation in degrees
            'iori_quat': (w, x, y, z) - image orientation quaternion
            'iori_euler': (roll, pitch, yaw) - image orientation in degrees
    """
    import struct
    import math

    # Extract GPMD stream using FFmpeg
    result = subprocess.run([
        get_ffmpeg_path(), '-i', file_path, '-map', '0:3', '-c', 'copy', '-f', 'rawvideo', '-'
    ], capture_output=True, creationflags=get_subprocess_flags())

    if result.returncode != 0:
        raise Exception(f"Failed to extract GPMD data: {result.stderr.decode()}")

    data = result.stdout

    # Get video frame rate
    probe_result = subprocess.run([
        get_ffprobe_path(), '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,duration', '-of', 'json', file_path
    ], capture_output=True, text=True, creationflags=get_subprocess_flags())

    video_info = json.loads(probe_result.stdout)
    stream = video_info['streams'][0]

    fps_str = stream.get('r_frame_rate', '30/1')
    fps_parts = fps_str.split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    def parse_quaternion_stream(data, fourcc_bytes):
        """Parse quaternion samples for a given FourCC (CORI or IORI).

        Validates GPMF type='s' (int16) and struct_size=8 (4 × int16 = quaternion)
        to avoid false-positive matches inside DISP or other unrelated records.
        """
        samples = []
        idx = 0

        while True:
            idx = data.find(fourcc_bytes, idx)
            if idx < 0:
                break

            if idx + 8 > len(data):
                break

            type_char = chr(data[idx + 4])
            struct_size = data[idx + 5]  # Should be 8 (4 shorts = quaternion)

            # GPMF quaternion records must be type 's' (signed int16) with
            # struct_size 8 (w,x,y,z as 4 × int16). Skip false positives
            # where the FourCC appears inside other records (e.g. DISP payloads).
            if type_char != 's' or struct_size != 8:
                idx += 1
                continue

            count = struct.unpack('>H', data[idx + 6:idx + 8])[0]

            for i in range(count):
                offset = idx + 8 + (i * struct_size)
                if offset + struct_size > len(data):
                    break

                # GoPro quaternions are Q15 fixed-point (divide by 32768)
                # Order in file: W, X, Y, Z (standard order)
                w, x, y, z = struct.unpack('>hhhh', data[offset:offset + 8])
                wn = w / 32768.0
                xn = x / 32768.0
                yn = y / 32768.0
                zn = z / 32768.0
                samples.append((wn, xn, yn, zn))

            idx += 1

        return samples

    def parse_grav_stream(data):
        """Parse GRAV (gravity vector) samples from GPMF data.

        GRAV is a fused gravity vector (x, y, z) in sensor body frame.
        GoPro uses type 's' (signed int16) with SCAL divisor, or 'f' (float32).
        SCAL tag appears just before GRAV in the GPMF stream.
        """
        samples = []
        idx = 0

        while True:
            idx = data.find(b'GRAV', idx)
            if idx < 0:
                break
            if idx + 8 > len(data):
                break

            type_char = chr(data[idx + 4])
            struct_size = data[idx + 5]
            count = struct.unpack('>H', data[idx + 6:idx + 8])[0]

            if type_char == 'f' and struct_size == 12:
                for i in range(count):
                    offset = idx + 8 + (i * struct_size)
                    if offset + struct_size > len(data):
                        break
                    gx, gy, gz = struct.unpack('>fff', data[offset:offset + 12])
                    samples.append((gx, gy, gz))
            elif type_char in ('s', 'h') and struct_size == 6:
                # Signed int16 with SCAL divisor.
                # Find SCAL tag just before GRAV (within ~20 bytes).
                scal = 32767.0
                scal_search = data[max(0, idx - 20):idx]
                scal_pos = scal_search.find(b'SCAL')
                if scal_pos >= 0:
                    abs_pos = max(0, idx - 20) + scal_pos
                    sc_type = chr(data[abs_pos + 4])
                    sc_size = data[abs_pos + 5]
                    if sc_type in ('s', 'h') and sc_size == 2 and abs_pos + 10 <= len(data):
                        scal = float(struct.unpack('>h', data[abs_pos + 8:abs_pos + 10])[0])
                    elif sc_type == 'S' and sc_size == 2 and abs_pos + 10 <= len(data):
                        scal = float(struct.unpack('>H', data[abs_pos + 8:abs_pos + 10])[0])
                if scal == 0:
                    scal = 32767.0

                for i in range(count):
                    offset = idx + 8 + (i * struct_size)
                    if offset + struct_size > len(data):
                        break
                    gx, gy, gz = struct.unpack('>hhh', data[offset:offset + 6])
                    samples.append((gx / scal, gy / scal, gz / scal))

            idx += 1

        return samples

    def parse_srot(data):
        """Parse SROT (Sensor Readout Time) from GPMF data.

        SROT is a single value per stream — the sensor's total readout time.
        Returns readout time in milliseconds, or None if not found.
        """
        idx = 0
        while True:
            idx = data.find(b'SROT', idx)
            if idx < 0:
                return None
            if idx + 8 > len(data):
                return None

            type_char = chr(data[idx + 4])
            struct_size = data[idx + 5]
            count = struct.unpack('>H', data[idx + 6:idx + 8])[0]

            if count < 1 or idx + 8 + struct_size > len(data):
                idx += 1
                continue

            payload = data[idx + 8:idx + 8 + struct_size]

            if type_char == 'J' and struct_size == 8:
                # uint64, typically microseconds
                val = struct.unpack('>Q', payload)[0]
                return val / 1000.0  # μs → ms
            elif type_char == 'L' and struct_size == 4:
                # uint32, typically microseconds
                val = struct.unpack('>I', payload)[0]
                return val / 1000.0  # μs → ms
            elif type_char == 'f' and struct_size == 4:
                # float32
                val = struct.unpack('>f', payload)[0]
                # Heuristic: if > 1000, probably μs; else ms
                return val / 1000.0 if val > 1000 else val
            else:
                # Unknown type, try next occurrence
                idx += 1
                continue

    # Parse SROT (sensor readout time)
    # First try the GPMF telemetry stream
    srot_ms = parse_srot(data)
    if srot_ms is None:
        # SROT is often in a secondary GPMF block at the END of the raw MP4 file
        # (geometry calibration data, not in the telemetry stream)
        try:
            tail_size = 512 * 1024  # Read last 512KB
            file_size = os.path.getsize(file_path)
            read_offset = max(0, file_size - tail_size)
            with open(file_path, 'rb') as f:
                f.seek(read_offset)
                tail_data = f.read()
            srot_ms = parse_srot(tail_data)
            if srot_ms is not None:
                print(f"SROT found in MP4 file tail (geometry calibration block)")
        except Exception as e:
            print(f"Warning: could not read file tail for SROT: {e}")
    if srot_ms is not None:
        print(f"SROT (sensor readout time): {srot_ms:.3f} ms")

    # Parse CORI, IORI, and GRAV
    cori_samples = parse_quaternion_stream(data, b'CORI')
    iori_samples = parse_quaternion_stream(data, b'IORI')
    grav_samples = parse_grav_stream(data)

    if not cori_samples:
        raise Exception("No CORI data found in file")

    # ── Determine chapter number and video frame count early ──
    # Both are needed for bias-drift detection and multi-chapter slicing.
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    chapter_num = 1
    clip_id = name_no_ext
    try:
        # GoPro convention: characters [2:4] are the chapter number
        chapter_num = int(name_no_ext[2:4])
        clip_id = name_no_ext[4:]  # e.g. "0082" for GS010082
    except (ValueError, IndexError):
        pass

    video_nb_frames = None
    try:
        nb_str = stream.get('nb_frames', '')
        if nb_str and nb_str != 'N/A':
            video_nb_frames = int(nb_str)
        elif stream.get('duration'):
            video_nb_frames = int(float(stream['duration']) * fps + 0.5)
    except (ValueError, TypeError):
        pass

    # ── Detect uncorrected CORI (firmware ERS-off mode) ──
    # Two signatures:
    #   1. Zeroed CORI: w~0 (all components near zero)
    #   2. Bias-drifting CORI: first quaternion is exact identity (1,0,0,0) within
    #      quantization noise (~1/32768). Normal firmware CORI always starts with
    #      a real initial orientation (xyz_max > 0.001). Uncorrected gyro integration
    #      starts at exact zero rotation and drifts from sensor bias.
    #
    # IMPORTANT: for later chapters (GS02+), this file's CORI[0] may already
    # carry accumulated gyro drift and no longer look like identity. Some
    # recordings store the full CORI array in every chapter (len >> video
    # frames), others store only per-chapter CORI. When this file is a later
    # chapter with per-chapter CORI, we must check the FIRST segment's CORI[0]
    # to reliably detect bias-drift.
    _detect_samples = cori_samples
    if chapter_num > 1 and len(cori_samples) >= 2:
        # Check if CORI spans the full recording (all chapters). If
        # len(cori_samples) ~ this chapter's frame count, it's per-chapter
        # only and CORI[0] is NOT the recording start -> need segment 1.
        _is_per_chapter_cori = bool(video_nb_frames and
                                     len(cori_samples) < video_nb_frames * 1.5)
        if _is_per_chapter_cori:
            _prefix = name_no_ext[:2]
            _ext = os.path.splitext(basename)[1]
            _seg1_name = f"{_prefix}01{clip_id}{_ext}"
            _seg1_path = os.path.join(os.path.dirname(file_path) or '.', _seg1_name)
            if os.path.exists(_seg1_path):
                try:
                    _seg1_result = subprocess.run([
                        get_ffmpeg_path(), '-i', _seg1_path, '-map', '0:3',
                        '-c', 'copy', '-f', 'rawvideo', '-'
                    ], capture_output=True, creationflags=get_subprocess_flags())
                    _seg1_cori = parse_quaternion_stream(_seg1_result.stdout, b'CORI')
                    if _seg1_cori:
                        _detect_samples = _seg1_cori
                        print(f"Using segment 1 ({_seg1_name}) CORI[0] for bias-drift detection")
                except Exception as _e:
                    print(f"Warning: could not read segment 1 CORI: {_e}")

    cori_is_zero = all(abs(q[0]) < 0.01 and abs(q[1]) < 0.01 and abs(q[2]) < 0.01 and abs(q[3]) < 0.01
                       for q in _detect_samples[:min(10, len(_detect_samples))])
    cori_is_bias_drift = False
    if not cori_is_zero and len(_detect_samples) > 30:
        q0 = _detect_samples[0]
        xyz_max = max(abs(q0[1]), abs(q0[2]), abs(q0[3]))
        if xyz_max < 0.0005:
            cori_is_bias_drift = True
            print(f"⚠ CORI starts at exact identity (xyz_max={xyz_max:.6f}) — uncorrected gyro integration")

    # ── Multi-chapter detection ──
    # Some GoPro .360 files store CORI/IORI for the ENTIRE recording in every
    # chapter file; others store per-chapter only. When the full array is
    # present, len(cori_samples) >> video_nb_frames.
    is_multi_chapter = bool(video_nb_frames and len(cori_samples) > video_nb_frames * 1.5)
    cori_offset = 0
    if is_multi_chapter:
        print(f"Multi-chapter CORI detected: {len(cori_samples)} CORI samples vs {video_nb_frames} video frames")

        # Chapter number and clip_id already parsed above.
        if chapter_num > 1:
            # Find sibling chapter files and count their video frames
            parent_dir = os.path.dirname(file_path) or '.'
            prefix = name_no_ext[:2]  # "GS", "GH", "GX", etc.
            ext = os.path.splitext(basename)[1]  # ".360", ".MP4"

            for prev_ch in range(1, chapter_num):
                prev_name = f"{prefix}{prev_ch:02d}{clip_id}{ext}"
                prev_path = os.path.join(parent_dir, prev_name)
                if os.path.exists(prev_path):
                    try:
                        prev_probe = subprocess.run([
                            get_ffprobe_path(), '-v', 'quiet', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_frames,duration,r_frame_rate',
                            '-of', 'json', prev_path
                        ], capture_output=True, text=True, creationflags=get_subprocess_flags())
                        prev_info = json.loads(prev_probe.stdout)
                        prev_stream = prev_info['streams'][0]
                        prev_nb = prev_stream.get('nb_frames', '')
                        if prev_nb and prev_nb != 'N/A':
                            prev_frames = int(prev_nb)
                        elif prev_stream.get('duration'):
                            prev_fps_str = prev_stream.get('r_frame_rate', f'{fps}/1')
                            prev_fps_parts = prev_fps_str.split('/')
                            prev_fps = float(prev_fps_parts[0]) / float(prev_fps_parts[1]) if len(prev_fps_parts) == 2 else float(prev_fps_parts[0])
                            prev_frames = int(float(prev_stream['duration']) * prev_fps + 0.5)
                        else:
                            prev_frames = 0
                        cori_offset += prev_frames
                        print(f"  Chapter {prev_ch} ({prev_name}): {prev_frames} frames")
                    except Exception as e:
                        print(f"  Warning: could not probe chapter {prev_ch} ({prev_name}): {e}")
                else:
                    print(f"  Warning: previous chapter file not found: {prev_name}")

    # This chapter's actual frame count (single-chapter clips: just len(cori_samples))
    n_chapter_frames = video_nb_frames if is_multi_chapter else len(cori_samples)

    if cori_is_zero or cori_is_bias_drift:
        reason = "zeroed" if cori_is_zero else "bias-drifting (uncorrected gyro integration)"
        print(f"⚠ CORI is {reason} — using VQF IMU fusion for orientation")
        try:
            from parse_gyro_raw import vqf_to_cori_quats
            # Pass THIS chapter's frame count, not the full multi-chapter CORI count.
            # vqf_to_cori_quats integrates this file's per-chapter gyro starting from
            # identity; the chaining across chapters is handled by concatenate_gyro_data.
            vqf_result = vqf_to_cori_quats(file_path, n_video_frames=n_chapter_frames)
            if grav_samples:
                vqf_result['grav_samples'] = grav_samples
            vqf_result['srot_ms'] = srot_ms
            vqf_result['cori_is_vqf'] = True
            return vqf_result
        except Exception as e:
            print(f"⚠ VQF fusion failed: {e}, falling back to identity CORI")
            cori_samples = [(1.0, 0.0, 0.0, 0.0)] * len(cori_samples)

    if grav_samples:
        print(f"GRAV (gravity vector): {len(grav_samples)} samples, first=({grav_samples[0][0]:.3f}, {grav_samples[0][1]:.3f}, {grav_samples[0][2]:.3f})")
    else:
        print("GRAV not found in GPMF data")

    # ── Firmware-CORI multi-chapter slicing ──
    if is_multi_chapter:
        # Slice CORI and IORI to this chapter's range
        cori_end = cori_offset + video_nb_frames
        print(f"  Slicing CORI[{cori_offset}:{cori_end}] for chapter ({video_nb_frames} frames)")
        cori_samples = cori_samples[cori_offset:cori_end]
        if iori_samples:
            iori_samples = iori_samples[cori_offset:cori_end]

        if len(cori_samples) < video_nb_frames:
            print(f"  Warning: only got {len(cori_samples)} CORI samples after slicing (expected {video_nb_frames})")

    # Build per-frame data
    frame_duration = 1.0 / fps
    frames = []

    num_frames = max(len(cori_samples), len(iori_samples))

    for i in range(num_frames):
        timestamp = i * frame_duration

        # CORI data
        if i < len(cori_samples):
            cori_q = cori_samples[i]
            cori_e = quat_to_euler(*cori_q)
        else:
            cori_q = cori_samples[-1] if cori_samples else (1, 0, 0, 0)
            cori_e = quat_to_euler(*cori_q)

        # IORI data
        if i < len(iori_samples):
            iori_q = iori_samples[i]
            iori_e = quat_to_euler(*iori_q)
        else:
            iori_q = iori_samples[-1] if iori_samples else (1, 0, 0, 0)
            iori_e = quat_to_euler(*iori_q)

        frames.append({
            'time': timestamp,
            'cori_quat': cori_q,
            'cori_euler': cori_e,  # (roll, pitch, yaw) in degrees
            'iori_quat': iori_q,
            'iori_euler': iori_e,  # (roll, pitch, yaw) in degrees
        })

    print(f"Parsed gyro data: {len(frames)} frames, {len(cori_samples)} CORI, {len(iori_samples)} IORI samples")
    if frames:
        f0 = frames[0]
        print(f"  First CORI euler: roll={f0['cori_euler'][0]:.2f}° pitch={f0['cori_euler'][1]:.2f}° yaw={f0['cori_euler'][2]:.2f}°")
        print(f"  First IORI euler: roll={f0['iori_euler'][0]:.2f}° pitch={f0['iori_euler'][1]:.2f}° yaw={f0['iori_euler'][2]:.2f}°")

    return {
        'fps': fps,
        'frames': frames,
        'srot_ms': srot_ms,  # Sensor readout time in ms (None if not found)
        'grav_samples': grav_samples,  # Per-frame gravity vectors [(gx,gy,gz), ...]
    }


# ─── Gyro Stabilizer Engine ─────────────────────────────────────────────────

class GyroStabilizer:
    """Computes per-frame stabilization corrections from CORI/IORI orientation data.

    Modded GoPro 360 camera with one lens flipped to create stereo VR180.
    Only IORI is baked into the .mov pixels (not CORI). CORI is metadata.
    GoPro 360 applies opposite IORI to each lens:
      - Left eye pixels have +IORI baked in
      - Right eye pixels have -IORI baked in (opposite)

    To maintain orientation, the correction must account for both CORI and IORI:
      Left corr:  CORI + IORI                        = CORI * IORI
      Right corr: CORI + IORI - 2*IORI = CORI - IORI = CORI * IORI^-1
    The right eye needs -2*IORI extra because it has the wrong opposite IORI baked in.
    """

    def __init__(self, gyro_data: dict):
        self.fps = gyro_data['fps']
        self.frames = gyro_data['frames']
        self.num_frames = len(self.frames)
        self.timestamps = np.array([f['time'] for f in self.frames])
        self.source = gyro_data.get('source', 'cori')

        # Extract raw CORI and IORI quaternions and Euler angles
        self.cori_quats = [f['cori_quat'] for f in self.frames]
        self.iori_quats = [f['iori_quat'] for f in self.frames]
        self.cori_eulers = np.array([f['cori_euler'] for f in self.frames])  # (N, 3) roll,pitch,yaw
        self.iori_eulers = np.array([f['iori_euler'] for f in self.frames])

        # GRAV (gravity vector) for world-frame alignment
        self.grav_samples = gyro_data.get('grav_samples', [])

        # 800Hz GYRO angular velocity for RS correction (optional)
        # When available, provides much better temporal resolution than CORI delta
        self.gyro_angvel_times = gyro_data.get('gyro_angvel_times')  # (M,) seconds
        self.gyro_angvel = gyro_data.get('gyro_angvel')              # (M, 3) deg/s
        # Keep raw (unsmoothed) copy for per-scanline RS correction.
        # Per-scanline needs instantaneous angular velocity at each scanline's
        # readout time — smoothing over 20ms would destroy the angular acceleration
        # information within the 15ms readout window.
        self.gyro_angvel_raw = self.gyro_angvel.copy() if self.gyro_angvel is not None else None
        if self.gyro_angvel is not None and len(self.gyro_angvel) > 1:
            # Smooth for the single-value RS path (get_angular_velocity_at_time).
            # Window ≈ 15ms (matches SROT readout period).
            dt_avg = np.median(np.diff(self.gyro_angvel_times[:200]))
            win = max(3, int(round(0.015 / dt_avg)))
            if win % 2 == 0:
                win += 1  # odd window for symmetric smoothing
            n = len(self.gyro_angvel)
            pad = win // 2
            smoothed = np.empty_like(self.gyro_angvel)
            for ax in range(3):
                col = self.gyro_angvel[:, ax]
                # Reflect-pad edges to avoid boundary artifacts
                padded = np.concatenate([col[pad:0:-1], col, col[-2:-2-pad:-1]])
                cs = np.cumsum(padded)
                cs = np.insert(cs, 0, 0.0)
                smoothed[:, ax] = (cs[win:] - cs[:-win]) / win
            self.gyro_angvel = smoothed
            print(f"GyroStabilizer: using 800Hz GYRO for angular velocity "
                  f"({len(self.gyro_angvel_times)} samples, smoothed win={win} ≈ {win*dt_avg*1000:.1f}ms)")

        # Correction quaternions (for SLERP interpolation) and matrices (for remap)
        identity_q = (1.0, 0.0, 0.0, 0.0)
        self.left_quats_corr = [identity_q] * self.num_frames
        self.right_quats_corr = [identity_q] * self.num_frames
        self.left_matrices = [np.eye(3, dtype=np.float64)] * self.num_frames
        self.right_matrices = [np.eye(3, dtype=np.float64)] * self.num_frames

    @staticmethod
    def _smooth_quats_moving_avg(quats_array, win_frames):
        """Smooth quaternion array with centered moving average (cumsum O(N)).

        quats_array: (N, 4) float64 array, sign-continuous.
        win_frames: odd integer window size.
        Returns: (N, 4) smoothed + renormalized quaternion array.
        """
        half = win_frames // 2
        smooth_Q = np.empty_like(quats_array)
        for ax in range(4):
            col = quats_array[:, ax]
            padded = np.concatenate([col[half:0:-1], col, col[-2:-2-half:-1]])
            cs = np.cumsum(padded)
            cs = np.insert(cs, 0, 0.0)
            smooth_Q[:, ax] = (cs[win_frames:] - cs[:-win_frames]) / win_frames
        norms = np.linalg.norm(smooth_Q, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        smooth_Q /= norms
        return smooth_Q

    @staticmethod
    def _smooth_scalar_moving_avg(vals, win_frames):
        """Smooth 1D scalar array with centered moving average (cumsum O(N)).

        vals: (N,) float64 array.
        win_frames: odd integer window size.
        Returns: (N,) smoothed array.
        """
        half = win_frames // 2
        padded = np.concatenate([vals[half:0:-1], vals, vals[-2:-2-half:-1]])
        cs = np.cumsum(padded)
        cs = np.insert(cs, 0, 0.0)
        return (cs[win_frames:] - cs[:-win_frames]) / win_frames

    @staticmethod
    def _smooth_quats_velocity_dampened(quats_array, fps, smooth_ms, fast_ms=50.0,
                                        max_velocity=200.0, max_corr_deg=10.0,
                                        responsiveness=1.0):
        """Velocity-dampened bidirectional exponential quaternion smoothing.

        Adapts smoothing strength based on angular velocity:
          - Calm motion → heavy smoothing (smooth_ms time constant)
          - Fast motion → light smoothing (fast_ms time constant, follows camera)

        After smoothing, applies a soft elastic limit on correction angle to
        prevent black borders while keeping transitions smooth.

        Inspired by Gyroflow's velocity-dampened algorithm.

        Args:
            quats_array: (N, 4) float64, sign-continuous quaternions.
            fps: frame rate.
            smooth_ms: time constant for calm periods (ms). Higher = more stable.
            fast_ms: time constant at max velocity (ms). Lower = follows faster.
            max_velocity: deg/s at which smoothing is at minimum (fast_ms). Default 200.
            max_corr_deg: max angular correction before soft limit (degrees).
                          Set to 0 to disable limiting.
            responsiveness: power curve for velocity→smoothing mapping (0.2–3.0).
                          <1 = starts following early (anticipatory), 1 = linear,
                          >1 = holds longer then catches up (laggy).

        Returns: (N, 4) smoothed + renormalized quaternion array.
        """
        import math
        N = len(quats_array)
        if N < 2:
            return quats_array.copy()

        dt = 1.0 / fps

        # ── Step 1: Compute per-frame angular velocity (deg/s) ──
        velocities = np.zeros(N, dtype=np.float64)
        for i in range(1, N):
            q0 = quats_array[i - 1]
            q1 = quats_array[i]
            dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]
            if dot < 0:
                q1 = -q1
                dot = -dot
            dot = min(dot, 1.0)
            angle = 2.0 * math.acos(dot)
            velocities[i] = math.degrees(angle) / dt

        # Smooth velocities (forward+backward exponential, ~200ms window)
        # Larger window gives more look-ahead so smoothing relaxes before fast motion
        vel_alpha = min(1.0, dt / 0.2)  # 200ms time constant
        for i in range(1, N):
            velocities[i] = velocities[i - 1] * (1.0 - vel_alpha) + velocities[i] * vel_alpha
        for i in range(N - 2, -1, -1):
            velocities[i] = velocities[i + 1] * (1.0 - vel_alpha) + velocities[i] * vel_alpha

        # ── Step 2: Bidirectional adaptive exponential smoothing ──
        tau_smooth = smooth_ms / 1000.0
        tau_fast = fast_ms / 1000.0
        resp_power = max(0.1, responsiveness)

        # Forward pass
        fwd = quats_array.copy()
        for i in range(1, N):
            vel_linear = min(velocities[i] / max_velocity, 1.0) if max_velocity > 0 else 0.0
            # Power curve: <1 = early follow, >1 = late follow
            vel_ratio = vel_linear ** resp_power
            tau = tau_smooth * (1.0 - vel_ratio) + tau_fast * vel_ratio
            alpha = min(dt / (tau + dt), 1.0)
            fwd[i] = _quat_slerp_array(fwd[i - 1], quats_array[i], alpha)

        # Backward pass
        bwd = quats_array.copy()
        for i in range(N - 2, -1, -1):
            vel_linear = min(velocities[i] / max_velocity, 1.0) if max_velocity > 0 else 0.0
            vel_ratio = vel_linear ** resp_power
            tau = tau_smooth * (1.0 - vel_ratio) + tau_fast * vel_ratio
            alpha = min(dt / (tau + dt), 1.0)
            bwd[i] = _quat_slerp_array(bwd[i + 1], quats_array[i], alpha)

        # Average forward and backward (midpoint SLERP)
        smoothed = np.empty_like(quats_array)
        for i in range(N):
            smoothed[i] = _quat_slerp_array(fwd[i], bwd[i], 0.5)

        # ── Step 3: Soft elastic correction limit ──
        # Instead of hard clamping (which causes snapping), use a smooth curve:
        #   angle < limit     → no change
        #   angle > limit     → soft_angle = limit × (1 + ln(angle/limit))
        # This logarithmic curve gives smooth deceleration as we approach the limit
        # and allows gradual overshoot without sudden jumps.
        if max_corr_deg > 0:
            max_corr_rad = math.radians(max_corr_deg)
            for i in range(N):
                q_raw = quats_array[i]
                q_sm = smoothed[i]
                dot = q_raw[0]*q_sm[0] + q_raw[1]*q_sm[1] + q_raw[2]*q_sm[2] + q_raw[3]*q_sm[3]
                if dot < 0:
                    q_sm = -q_sm
                    dot = -dot
                dot = min(dot, 1.0)
                angle = 2.0 * math.acos(dot)

                if angle > max_corr_rad:
                    # Soft elastic: logarithmic compression beyond the limit
                    soft_angle = max_corr_rad * (1.0 + math.log(angle / max_corr_rad))
                    t = soft_angle / angle
                    t = min(t, 1.0)  # safety clamp
                    smoothed[i] = _quat_slerp_array(q_raw, q_sm, t)

        # Renormalize
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        smoothed /= norms

        return smoothed

    def smooth(self, window_ms=500, roll_window_ms=2000, horizon_lock=False, stabilize=True,
               max_corr_deg=10.0, responsiveness=1.0, upside_down=False):
        """Compute per-frame correction matrices with split-axis smoothing.

        Forward pipeline: cross → IORI cancel → RS → heading → global+stereo → output

        Produces per frame:
          - View matrices: IORI_cancel × heading (for non-RS remap path)
          - Heading matrices: heading only (for RS path: R_sensor = heading × global × stereo)
          - IORI matrices: per-frame IORI (for RS path: R_cross = IORI, applied after RS)

        Gimbal-lock-free split-axis approach:
          1. Smooth full CORI quaternions twice (heading window + roll window)
          2. YP: correction = raw × smoothed⁻¹, swing-twist decompose, take swing part
          3. Roll: gravity-based extraction from rotation matrices (stable at all yaw angles)
             — avoids the 180° roll flip that swing-twist around Z exhibits at yaw=180°

        window_ms: yaw+pitch smoothing time constant (0 = lock all axes)
        roll_window_ms: roll smoothing time constant (0 = lock roll)
        horizon_lock: True = fully cancel all roll motion
        max_corr_deg: max yaw+pitch correction angle before clamping (10° default)
        """
        import math as _math
        if self.num_frames == 0:
            return

        N = self.num_frames
        identity_q = (1.0, 0.0, 0.0, 0.0)

        # ── Build sign-continuous CORI quaternion array ──
        Q = np.array(self.cori_quats, dtype=np.float64)
        for i in range(1, N):
            if np.dot(Q[i], Q[i - 1]) < 0:
                Q[i] *= -1

        # ── Gravity-align CORI quaternions to true world frame ──
        # Without this, CORI axes are relative to the camera's initial orientation.
        # If the camera starts tilted, panning (yaw) couples into roll (horizon tilt).
        # GRAV gives us the true gravity direction; we use the first frame's GRAV to
        # compute a correction quaternion that aligns the CORI reference to world-up.
        #
        # Skip for VQF-generated CORI: VQF already aligns to gravity internally
        # using GRAV/ACCL. Applying GRAV alignment again would double-correct and
        # use the wrong convention (CORI expects Y-down, VQF uses Z-up).
        self._q_gravity_align = None
        if self.source and 'vqf' in self.source.lower():
            print(f"GRAV alignment: skipped (VQF already gravity-aligned)")
        elif self.grav_samples and len(self.grav_samples) >= 1:
            # Average first few GRAV samples for stability
            n_avg = min(10, len(self.grav_samples))
            gx = sum(g[0] for g in self.grav_samples[:n_avg]) / n_avg
            gy = sum(g[1] for g in self.grav_samples[:n_avg]) / n_avg
            gz = sum(g[2] for g in self.grav_samples[:n_avg]) / n_avg
            grav_len = _math.sqrt(gx*gx + gy*gy + gz*gz)

            if grav_len > 0.1:
                # Normalize gravity vector
                gx, gy, gz = gx/grav_len, gy/grav_len, gz/grav_len

                # When camera is mounted upside down, the GRAV sensor reports the
                # opposite "down" direction. Negate to restore correct world-up.
                if upside_down:
                    gx, gy, gz = -gx, -gy, -gz
                    print("GRAV alignment: upside-down mode — gravity vector negated")

                # GRAV is in the CORI reference frame (camera-local at identity CORI).
                # GoPro GRAV points toward gravity (downward), and in the default
                # orientation it's approximately (0, +1, 0) (Y-positive = down).
                # We need q_align such that q_align rotates measured gravity to (0, +1, 0).
                #
                # Rotation from vector A to vector B:
                #   axis = normalize(A × B), angle = acos(A · B)
                #   q = (cos(angle/2), sin(angle/2) * axis)
                target = (0.0, 1.0, 0.0)  # gravity-down direction in CORI convention
                src = (gx, gy, gz)

                dot = src[0]*target[0] + src[1]*target[1] + src[2]*target[2]
                dot = max(-1.0, min(1.0, dot))

                if dot > 0.9999:
                    # Already aligned
                    self._q_gravity_align = None
                    print(f"GRAV alignment: gravity already aligned to Y-down (dot={dot:.4f})")
                elif dot < -0.9999:
                    # Opposite direction — rotate 180° around X axis
                    self._q_gravity_align = (0.0, 1.0, 0.0, 0.0)
                    print(f"GRAV alignment: gravity opposite to Y-down, 180° correction")
                else:
                    # Cross product for rotation axis
                    cx = src[1]*target[2] - src[2]*target[1]
                    cy = src[2]*target[0] - src[0]*target[2]
                    cz = src[0]*target[1] - src[1]*target[0]
                    cn = _math.sqrt(cx*cx + cy*cy + cz*cz)
                    cx, cy, cz = cx/cn, cy/cn, cz/cn

                    angle = _math.acos(dot)
                    ha = angle / 2.0
                    sa = _math.sin(ha)
                    self._q_gravity_align = (_math.cos(ha), sa*cx, sa*cy, sa*cz)

                    tilt_deg = _math.degrees(angle)
                    print(f"GRAV alignment: initial tilt={tilt_deg:.1f}° "
                          f"(grav=[{gx:.3f},{gy:.3f},{gz:.3f}])")

                # Apply gravity alignment: Q_aligned = Q_raw × q_g⁻¹
                # Right-multiply changes the world/reference frame of CORI so that
                # the Y axis aligns with true gravity. This ensures _local_roll extracts
                # roll relative to the physical horizon, not the tilted initial frame.
                # (Left-multiply would change the camera frame, which is wrong.)
                if self._q_gravity_align is not None:
                    q_g_inv = quat_inverse(self._q_gravity_align)
                    for i in range(N):
                        qi = tuple(Q[i])
                        qa = quat_multiply(qi, q_g_inv)
                        Q[i] = np.array(qa, dtype=np.float64)
                    # Re-enforce sign continuity after alignment
                    for i in range(1, N):
                        if np.dot(Q[i], Q[i - 1]) < 0:
                            Q[i] *= -1
                    print(f"  Applied gravity alignment to {N} CORI frames")

        # ── Smooth full CORI (all axes: yaw, pitch, roll) ──
        _NO_STABILIZE = 'no_stab'
        _camera_lock = False
        smooth_q = None
        if not stabilize:
            smooth_q = _NO_STABILIZE
            print(f"IORI compensation only (no heading correction, source={self.source})")
        elif window_ms == 0:
            # Camera lock: fully counteract all camera rotation on all axes
            _camera_lock = True
            print(f"Gyro stabilization: CAMERA LOCK (all axes locked)")
        elif N > 1:
            smooth_q = self._smooth_quats_velocity_dampened(
                Q, self.fps, smooth_ms=window_ms, fast_ms=50.0,
                max_velocity=200.0, max_corr_deg=max_corr_deg,
                responsiveness=responsiveness)
            if horizon_lock:
                print(f"Gyro stabilization: {window_ms:.0f}ms (vel-dampened, soft-limit {max_corr_deg:.0f}°, resp {responsiveness:.1f}), horizon lock (GRAV)")
            else:
                print(f"Gyro stabilization: {window_ms:.0f}ms all-axis (vel-dampened, soft-limit {max_corr_deg:.0f}°, resp {responsiveness:.1f})")

        # ── Detect non-zero IORI ──
        self.has_iori = any(
            abs(q[0]) < 0.9999 or abs(q[1]) > 0.0001 or abs(q[2]) > 0.0001 or abs(q[3]) > 0.0001
            for q in self.iori_quats
        )
        if self.has_iori:
            print(f"  IORI detected (non-zero)")

        # ── Helper: camera-local roll from CORI quaternion ──
        # Computes the Z-axis rotation angle in camera-local coordinates that maps
        # the output "up" (0,1,0) to the world-up direction projected into the
        # camera's image plane.
        #
        # Key: this is atan2(-R[0,1], R[1,1]) where R is the CORI rotation matrix.
        # R[:,1] gives world-up expressed in camera coords (since R maps world→camera
        # for EAC cross lookup). We need Rz(θ)×(0,1,0) = (-sinθ, cosθ, 0) = R[:,1]_xy.
        #
        # Unlike the world-frame gravity roll (which is yaw-independent), this
        # correctly flips sign with yaw: at yaw=0 roll=5° → +5°, at yaw=180° → -5°.
        # This matches the camera-local Z-rotation needed to correct the roll.
        def _local_roll(q):
            """Return camera-local Z-rotation angle (radians) to correct gravity roll."""
            R = quat_to_rotation_matrix(*q)
            return _math.atan2(-R[0, 1], R[1, 1])

        # ── Pre-compute per-frame GRAV roll (drift-free horizon reference) ──
        # GRAV is measured by accelerometer at ~400Hz, independent of CORI's
        # gyro-based integration which suffers from yaw drift over time.
        # GRAV in body frame = R × (0,1,0) = R[:,1], so:
        #   roll_from_grav = atan2(-grav_x, grav_y)
        # This is mathematically identical to _local_roll(q) but drift-free.
        _grav_roll_per_frame = None
        if self.grav_samples and len(self.grav_samples) > 0:
            n_grav = len(self.grav_samples)
            grav_arr = np.array(self.grav_samples, dtype=np.float64)  # (n_grav, 3)
            if upside_down:
                grav_arr = -grav_arr
            # Downsample GRAV to per-frame by binning (n_grav / N samples per frame)
            _raw_grav_roll = np.empty(N, dtype=np.float64)
            bin_size = n_grav / N
            for i in range(N):
                g_start = int(i * bin_size)
                g_end = int((i + 1) * bin_size)
                g_end = max(g_end, g_start + 1)  # at least 1 sample
                g_end = min(g_end, n_grav)
                gx_avg = grav_arr[g_start:g_end, 0].mean()
                gy_avg = grav_arr[g_start:g_end, 1].mean()
                _raw_grav_roll[i] = _math.atan2(-gx_avg, gy_avg)

            # Complementary filter: fuse CORI roll (smooth, drifts) with GRAV roll
            # (noisy, drift-free). CORI provides frame-to-frame smoothness (high-freq),
            # GRAV anchors long-term drift correction (low-freq).
            #   fused[0] = grav[0]
            #   fused[i] = alpha * (fused[i-1] + cori_delta[i]) + (1-alpha) * grav[i]
            # where alpha = exp(-dt / tau), tau = time constant for drift correction.
            # tau ~2s means GRAV corrects drift over ~2 seconds while CORI keeps
            # frame-to-frame transitions smooth.
            _cori_roll = np.array([_local_roll(tuple(Q[i])) for i in range(N)], dtype=np.float64)
            dt = 1.0 / self.fps
            tau = 2.0  # seconds — drift correction time constant
            alpha = _math.exp(-dt / tau)
            _grav_roll_per_frame = np.empty(N, dtype=np.float64)
            _grav_roll_per_frame[0] = _raw_grav_roll[0]
            for i in range(1, N):
                cori_delta = _cori_roll[i] - _cori_roll[i - 1]
                # Wrap delta to [-pi, pi]
                if cori_delta > _math.pi:
                    cori_delta -= 2 * _math.pi
                elif cori_delta < -_math.pi:
                    cori_delta += 2 * _math.pi
                predicted = _grav_roll_per_frame[i - 1] + cori_delta
                _grav_roll_per_frame[i] = alpha * predicted + (1.0 - alpha) * _raw_grav_roll[i]
            print(f"GRAV roll: complementary filter (CORI+GRAV) from {n_grav} samples "
                  f"(bin={bin_size:.1f}, tau={tau:.1f}s, alpha={alpha:.4f})")

        # ── Build per-frame heading correction ──
        # Single smoother for all axes. Horizon lock overrides the roll component
        # with GRAV complementary filter for drift-free horizon.
        left_corr = []
        left_mats = []
        right_corr = []
        right_mats = []
        heading_corr = []
        heading_mats = []
        iori_left_mats = []
        iori_right_mats = []

        for i in range(N):
            q_iori = self.iori_quats[i]
            q_raw = tuple(Q[i])

            if smooth_q is _NO_STABILIZE:
                # No heading correction at all
                q_heading = identity_q
            elif _camera_lock:
                # Camera lock: counteract full CORI rotation
                q_heading = q_raw
            else:
                # Correction = raw × smooth⁻¹ (all axes)
                q_corr = quat_multiply(q_raw, quat_inverse(tuple(smooth_q[i])))

                if horizon_lock and _grav_roll_per_frame is not None:
                    # Horizon lock: keep yaw+pitch from smoother, replace roll with GRAV
                    # Swing-twist decompose to extract yaw+pitch (swing) from correction
                    w, x, y, z = q_corr
                    n_twist = _math.sqrt(w * w + z * z)
                    if n_twist > 1e-10:
                        tw, tz = w / n_twist, z / n_twist
                    else:
                        tw, tz = 1.0, 0.0
                    swing_yp = quat_multiply(q_corr, (tw, 0, 0, -tz))

                    # Replace roll with GRAV complementary filter (drift-free)
                    half_r = _grav_roll_per_frame[i] / 2.0
                    twist_roll = (_math.cos(half_r), 0.0, 0.0, _math.sin(half_r))
                    q_heading = quat_multiply(swing_yp, twist_roll)
                else:
                    # All-axis stabilization: use full correction as-is
                    q_heading = q_corr

            # Heading only (no IORI)
            heading_corr.append(q_heading)
            heading_mats.append(quat_to_rotation_matrix(*q_heading))

            # IORI per-frame (for RS R_cross)
            iori_left_mats.append(quat_to_rotation_matrix(*q_iori))
            iori_right_mats.append(quat_to_rotation_matrix(*quat_inverse(q_iori)))

            # IORI_cancel × heading (for non-RS remap)
            q_left = quat_multiply(q_iori, q_heading)
            q_right = quat_multiply(quat_inverse(q_iori), q_heading)
            left_corr.append(q_left)
            left_mats.append(quat_to_rotation_matrix(*q_left))
            right_corr.append(q_right)
            right_mats.append(quat_to_rotation_matrix(*q_right))

        # Store all outputs
        self.left_quats_corr = left_corr
        self.right_quats_corr = right_corr
        self.left_matrices = left_mats
        self.right_matrices = right_mats
        self.heading_quats = heading_corr
        self.heading_matrices = heading_mats
        self.iori_left_matrices = iori_left_mats
        self.iori_right_matrices = iori_right_mats

    def _interp_at_time(self, quats_corr, matrices, t):
        """Interpolate correction at timestamp t using quaternion SLERP.

        Uses SLERP on stored correction quaternions for mathematically correct
        interpolation, then converts to rotation matrix for the remap pipeline.
        """
        import bisect

        if self.num_frames == 0:
            return np.eye(3, dtype=np.float64)
        if t <= self.timestamps[0]:
            return matrices[0]
        if t >= self.timestamps[-1]:
            return matrices[-1]

        idx = bisect.bisect_right(self.timestamps, t) - 1
        if idx >= self.num_frames - 1:
            return matrices[-1]

        t0, t1 = self.timestamps[idx], self.timestamps[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0

        # For very small alpha, just return the nearest matrix (avoid SLERP overhead)
        if alpha < 1e-6:
            return matrices[idx]
        if alpha > 1.0 - 1e-6:
            return matrices[min(idx + 1, self.num_frames - 1)]

        # SLERP between correction quaternions
        q0 = quats_corr[idx]
        q1 = quats_corr[min(idx + 1, self.num_frames - 1)]
        q_interp = quat_slerp(q0, q1, alpha)
        return quat_to_rotation_matrix(*q_interp)

    def get_left_matrix_at_time(self, t):
        """Get left eye view matrix (IORI × heading) at timestamp t."""
        return self._interp_at_time(self.left_quats_corr, self.left_matrices, t)

    def get_right_matrix_at_time(self, t):
        """Get right eye view matrix (IORI × heading) at timestamp t."""
        return self._interp_at_time(self.right_quats_corr, self.right_matrices, t)

    def get_heading_matrix_at_time(self, t):
        """Get heading-only matrix at timestamp t (no IORI). Same for both eyes."""
        return self._interp_at_time(self.heading_quats, self.heading_matrices, t)

    def get_iori_left_matrix_at_time(self, t):
        """Get left eye IORI matrix at time t (maps sensor→cross space)."""
        idx = min(int(round(t * self.fps)), self.num_frames - 1)
        idx = max(0, idx)
        return self.iori_left_matrices[idx]

    def get_iori_right_matrix_at_time(self, t):
        """Get right eye IORI matrix at time t (maps sensor→cross space)."""
        idx = min(int(round(t * self.fps)), self.num_frames - 1)
        idx = max(0, idx)
        return self.iori_right_matrices[idx]

    def get_angular_velocity_at_time(self, t, use_cori=False):
        """Get angular velocity (deg/s) at timestamp t.

        use_cori=False (default): Uses 800Hz GYRO data when available.
            Best for normal footage — high temporal resolution.
        use_cori=True: Uses CORI quaternion delta at 30Hz.
            Best for vibrating platforms (drones, planes) where the raw
            gyro is dominated by high-frequency noise that doesn't match
            the firmware's internal IMU fusion.

        Returns [roll, pitch, yaw] ordered for apply_rs_correction():
          [0] = roll  = bodyY rotation (around optical axis)
          [1] = pitch = bodyX rotation (tilt → vertical shear)
          [2] = yaw   = bodyZ rotation (pan → horizontal shear)
        """
        import bisect, math

        # Prefer 800Hz GYRO when available (unless CORI mode requested)
        if not use_cori and self.gyro_angvel is not None and len(self.gyro_angvel) > 0:
            return self._gyro_angular_velocity_at_time(t)

        # CORI quaternion delta (30Hz)
        if self.num_frames < 2:
            return np.zeros(3)

        t = np.clip(t, self.timestamps[0], self.timestamps[-1])
        idx = bisect.bisect_right(self.timestamps, t) - 1
        idx = np.clip(idx, 0, self.num_frames - 2)

        dt = self.timestamps[idx + 1] - self.timestamps[idx]
        if dt < 1e-6:
            return np.zeros(3)

        # Delta quaternion: q_delta = q[i]^-1 * q[i+1]
        q0 = self.cori_quats[idx]
        q1 = self.cori_quats[idx + 1]
        q_delta = quat_multiply(quat_inverse(q0), q1)

        # Ensure w >= 0 (shorter path)
        w, x, y, z = q_delta
        if w < 0:
            w, x, y, z = -w, -x, -y, -z

        # Extract rotation angle and axis from q_delta
        sin_half = math.sqrt(x*x + y*y + z*z)
        if sin_half < 1e-10:
            return np.zeros(3)

        angle = 2.0 * math.atan2(sin_half, w)  # radians, more stable than acos
        deg_per_sec = math.degrees(angle) / dt

        # CORI quaternion components in Y↔Z-swapped convention:
        #   q.x = bodyX, q.y = bodyZ (swapped), q.z = bodyY (swapped)
        rate_x = (x / sin_half) * deg_per_sec  # bodyX
        rate_y = (y / sin_half) * deg_per_sec  # bodyZ (Y↔Z swap)
        rate_z = (z / sin_half) * deg_per_sec  # bodyY (Y↔Z swap)

        # Return [bodyY(roll), bodyX(pitch), bodyZ(yaw)]
        return np.array([rate_z, rate_x, rate_y])

    def _gyro_angular_velocity_at_time(self, t):
        """Interpolate angular velocity from 800Hz GYRO data at time t.

        Uses RAW (unsmoothed) gyro for RS correction — smoothing destroys
        the high-frequency vibration signal that the firmware RS corrects.
        Already in CORI convention order: [roll, pitch, yaw].

        Returns (roll_rate, pitch_rate, yaw_rate) in degrees per second.
        """
        import bisect

        times = self.gyro_angvel_times
        angvel = self.gyro_angvel

        t = np.clip(t, times[0], times[-1])
        idx = bisect.bisect_right(times, t) - 1
        idx = np.clip(idx, 0, len(times) - 2)

        t0, t1 = times[idx], times[idx + 1]
        dt = t1 - t0
        if dt < 1e-9:
            return angvel[idx]

        alpha = (t - t0) / dt
        return angvel[idx] * (1.0 - alpha) + angvel[idx + 1] * alpha

    def get_perscanline_rs_angvel(self, t, srot_s, n_groups=32,
                                   yaw_factor=0.0, pitch_factor=2.0, roll_factor=2.0):
        """Compute per-scanline-group angular velocities for RS correction.

        Samples the RAW (unsmoothed) 800Hz GYRO data at each row group's
        readout time to get the instantaneous angular velocity. This captures
        angular acceleration during the 15ms readout window — the linear
        approach uses a single smoothed value which blurs this out.

        Returns:
            rs_angvel: (n_groups, 3) float32 angular velocity coefficients
                       [yaw, pitch, roll] in rad/s with factors applied.
                       Returns None if 800Hz GYRO data is not available.
        """
        if self.gyro_angvel_raw is None or len(self.gyro_angvel_raw) < 10:
            return None

        import bisect

        times = self.gyro_angvel_times
        angvel = self.gyro_angvel_raw  # RAW unsmoothed, (M, 3) deg/s [roll, pitch, yaw]

        deg2rad = np.float32(np.pi / 180.0)
        rs_angvel = np.zeros((n_groups, 3), dtype=np.float32)

        # Row group time offsets: linearly spaced across readout
        group_offsets = np.linspace(-srot_s / 2, srot_s / 2, n_groups)

        for g in range(n_groups):
            gt = np.clip(t + group_offsets[g], times[0], times[-1])
            idx = bisect.bisect_right(times, gt) - 1
            idx = np.clip(idx, 0, len(times) - 2)
            t0, t1 = times[idx], times[idx + 1]
            dt = t1 - t0
            alpha = (gt - t0) / dt if dt > 0 else 0.0
            av = angvel[idx] * (1 - alpha) + angvel[idx + 1] * alpha
            # av is [roll, pitch, yaw] in deg/s
            # Apply factors and convert to rad/s
            # Output order: [yaw, pitch, roll] to match rs_coeffs convention in kernel
            rs_angvel[g, 0] = np.float32(av[2] * yaw_factor) * deg2rad    # yaw
            rs_angvel[g, 1] = np.float32(av[1] * pitch_factor) * deg2rad  # pitch
            rs_angvel[g, 2] = np.float32(av[0] * roll_factor) * deg2rad   # roll

        return rs_angvel

    # Number of row groups for per-scanline RS correction
    RS_N_GROUPS = 32

    def precompute_export_matrices(self, start_time, fps, total_frames,
                                    rs_enabled=False,
                                    R_view_left=None, R_view_right=None,
                                    srot_s=0.0,
                                    rs_yaw_factor=0.0, rs_pitch_factor=-2.0, rs_roll_factor=-2.0,
                                    rs_use_cori=False):
        """Pre-compute all per-frame stabilization matrices for export.

        Batches all interpolation + matrix multiplies upfront so the render
        loop does zero stabilization math — just array indexing.

        R_view_left/R_view_right are the static global+stereo view matrices.
        When provided, they're baked into the output matrices (R_cam @ R_view).

        Returns dict of numpy arrays indexed by frame number.
        """
        left_mats = np.empty((total_frames, 3, 3), dtype=np.float64)
        right_mats = np.empty((total_frames, 3, 3), dtype=np.float64)
        heading_mats = np.empty((total_frames, 3, 3), dtype=np.float64) if rs_enabled else None
        angvel_arr = np.empty((total_frames, 3), dtype=np.float64) if rs_enabled else None
        iori_right_mats = np.empty((total_frames, 3, 3), dtype=np.float64) if rs_enabled else None
        # Per-scanline RS rotation matrices (None if not available)
        rs_perscanline = None
        if rs_enabled and srot_s > 0 and self.gyro_angvel is not None:
            rs_perscanline = np.empty((total_frames, self.RS_N_GROUPS, 3), dtype=np.float32)

        for i in range(total_frames):
            t = start_time + (i / fps)

            # Left/right eye: IORI × heading (× view if provided)
            L = self._interp_at_time(self.left_quats_corr, self.left_matrices, t)
            R = self._interp_at_time(self.right_quats_corr, self.right_matrices, t)
            if R_view_left is not None:
                L = L @ R_view_left
            if R_view_right is not None:
                R = R @ R_view_right
            left_mats[i] = L
            right_mats[i] = R

            # RS-specific: heading (× view), angular velocity, IORI
            if rs_enabled:
                H = self._interp_at_time(self.heading_quats, self.heading_matrices, t)
                if R_view_right is not None:
                    H = H @ R_view_right
                heading_mats[i] = H
                angvel_arr[i] = self.get_angular_velocity_at_time(t, use_cori=rs_use_cori)
                iori_right_mats[i] = self.get_iori_right_matrix_at_time(t)

                # Per-scanline RS matrices
                if rs_perscanline is not None:
                    av = self.get_perscanline_rs_angvel(
                        t, srot_s, self.RS_N_GROUPS,
                        yaw_factor=rs_yaw_factor, pitch_factor=rs_pitch_factor,
                        roll_factor=rs_roll_factor)
                    if av is not None:
                        rs_perscanline[i] = av
                    else:
                        rs_perscanline[i] = 0.0  # zero angular velocity fallback

        return {
            'left_matrices': left_mats,
            'right_matrices': right_mats,
            'heading_matrices': heading_mats,
            'angular_vel': angvel_arr,
            'iori_right_matrices': iori_right_mats,
            'rs_perscanline': rs_perscanline,
        }




def klns_forward(theta, klns):
    """KLNS polynomial: θ → r (fisheye radius in pixels). Vectorized."""
    c0, c1, c2, c3, c4 = [np.float32(c) for c in klns]
    t2 = theta * theta
    return theta * (c0 + t2 * (c1 + t2 * (c2 + t2 * (c3 + t2 * c4))))


def build_fisheye_remap(klns, ctrx=0.0, ctry=0.0, cal_dim=4216):
    """Build EAC cross → circular fisheye remap tables for one lens.

    Reconstructs the original sensor view from EAC cross data using inverse
    KLNS projection. Output is cal_dim × cal_dim with a circular fisheye image.

    Returns (map_x, map_y, valid_mask) where maps are float32 arrays.
    """
    cx = cal_dim / 2.0 + ctrx
    cy = cal_dim / 2.0 + ctry
    yy, xx = np.mgrid[0:cal_dim, 0:cal_dim].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi = np.arctan2(yy - cy, xx - cx)

    # Inverse KLNS: r → θ (Newton's method)
    c0, c1, c2, c3, c4 = [np.float32(c) for c in klns]
    theta = r / c0
    for _ in range(5):
        t2 = theta * theta
        r_est = theta * (c0 + t2 * (c1 + t2 * (c2 + t2 * (c3 + t2 * c4))))
        dr = c0 + t2 * (3*c1 + t2 * (5*c2 + t2 * (7*c3 + t2 * 9*c4)))
        theta -= (r_est - r) / np.maximum(dr, 1e-6)
        theta = np.clip(theta, 0, np.pi)

    sin_t = np.sin(theta); cos_t = np.cos(theta)
    xn = sin_t * np.cos(phi)
    yn = sin_t * np.sin(phi)
    zn = cos_t

    # Map 3D direction → EAC cross (same face formulas as _dirs_to_cross_maps)
    tp = np.float32(2.0 / np.pi)
    abs_x = np.abs(xn); abs_y = np.abs(yn); fs = np.float32(1920)
    mx = np.full((cal_dim, cal_dim), np.float32(-1))
    my = np.full((cal_dim, cal_dim), np.float32(-1))

    m = (zn > 0) & (zn >= abs_x) & (zn >= abs_y)  # Front
    if np.any(m):
        u = tp * np.arctan(xn[m] / zn[m]) + 0.5
        v = 0.5 - tp * np.arctan(yn[m] / zn[m])
        mx[m] = 1008 + u * fs; my[m] = 1008 + v * fs

    m = (xn > 0) & (zn <= xn) & (abs_y <= xn)  # Right
    if np.any(m):
        u = tp * np.arctan(-zn[m] / xn[m]) + 0.5
        v = 0.5 - tp * np.arctan(yn[m] / xn[m])
        mx[m] = 2928 + np.clip(u * fs, 0, 1007); my[m] = 1008 + v * fs

    m = (xn < 0) & (zn <= abs_x) & (abs_y <= abs_x)  # Left
    if np.any(m):
        xl = abs_x[m]
        u = tp * np.arctan(zn[m] / xl) + 0.5
        v = 0.5 - tp * np.arctan(yn[m] / xl)
        mx[m] = np.clip(u * fs - 912, 0, 1007); my[m] = 1008 + v * fs

    m = (yn > 0) & (abs_x <= yn) & (zn <= yn)  # Top
    if np.any(m):
        u = tp * np.arctan(xn[m] / yn[m]) + 0.5
        v = tp * np.arctan(zn[m] / yn[m]) + 0.5
        mx[m] = 1008 + u * fs; my[m] = np.clip(v * fs - 912, 0, 1007)

    m = (yn < 0) & (abs_x <= abs_y) & (zn <= abs_y)  # Bottom
    if np.any(m):
        yb = abs_y[m]
        u = tp * np.arctan(xn[m] / yb) + 0.5
        v = 0.5 - tp * np.arctan(zn[m] / yb)
        mx[m] = 1008 + u * fs; my[m] = 2928 + np.clip(v * fs, 0, 1007)

    np.clip(mx, -1, 3935, out=mx); np.clip(my, -1, 3935, out=my)

    # Flip to match sensor orientation: 180° rotation + horizontal mirror
    mx = mx[::-1, :].copy()
    my = my[::-1, :].copy()

    max_r = klns_forward(np.float32(np.pi * 184.5 / 360), klns)
    valid = r < max_r
    return mx, my, valid



def build_rs_time_map(height, width, klns, ctrx=0.0, ctry=0.0, cal_dim=4216):
    """Build a 2D time-offset map for RS correction using GEOC KLNS fisheye geometry.

    The camera captures in fisheye then converts to EAC in-camera.
    Rolling shutter scans the fisheye sensor top→bottom, so each pixel's capture
    time depends on its original sensor row — NOT the equirect row.

    For each equirect pixel at (lat, lon):
      1. Compute polar angle θ from optical axis: θ = acos(cos(lat)·cos(lon))
      2. Compute fisheye radius r via KLNS polynomial:
         r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹
      3. Compute sensor y-position (actual pixel row):
         sensor_y = (cal_dim/2 + ctry) - r · sin(lat) / sin(θ)
      4. Normalize to [-0.5, 0.5] for time offset.

    Args:
        height, width: equirect output dimensions
        klns: [c0, c1, c2, c3, c4] raw GEOC KLNS polynomial coefficients
              (c0 = focal length in pixels, c1-c4 = distortion)
        ctrx, ctry: GEOC principal point offset from sensor center (pixels)
        cal_dim: sensor calibration dimension (4216 for GoPro MAX)

    Returns (H, W) float32 array of normalized time offsets in [-0.5, 0.5].
    Multiply by readout_time (seconds) to get actual time offset per pixel.
    """
    # Equirect coordinates: each eye covers 180° × 180°
    lat = np.linspace(np.pi / 2, -np.pi / 2, height, dtype=np.float32)   # top=+90°, bottom=-90°
    lon = np.linspace(-np.pi / 2, np.pi / 2, width, dtype=np.float32)    # left=-90°, right=+90°

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')  # (H, W)

    cos_lat = np.cos(lat_grid)
    sin_lat = np.sin(lat_grid)
    cos_lon = np.cos(lon_grid)

    # Polar angle from optical axis
    cos_theta = np.clip(cos_lat * cos_lon, -1.0, 1.0)
    theta = np.arccos(cos_theta)       # (H, W)
    sin_theta = np.sin(theta)

    # GEOC KLNS: r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹
    c0, c1, c2, c3, c4 = [np.float32(c) for c in klns]
    theta2 = theta * theta
    r = theta * (c0 + theta2 * (c1 + theta2 * (c2 + theta2 * (c3 + theta2 * c4))))

    # Sensor y-position (actual pixel row, 0 = top of sensor)
    # Optical center is at row (cal_dim/2 + ctry)
    # Points above center (sin(lat)>0) → smaller row → earlier capture
    safe_sin_theta = np.where(np.abs(sin_theta) < 1e-7, np.float32(1.0), sin_theta)
    center_y = np.float32(cal_dim / 2.0 + ctry)
    sensor_y = np.where(
        np.abs(sin_theta) < 1e-7,
        center_y - np.float32(c0) * sin_lat,   # limit at θ→0: r/sin(θ) → c0
        center_y - r * sin_lat / safe_sin_theta
    )

    # Normalize: row 0 → t=-0.5 (first captured), row cal_dim → t=+0.5 (last)
    t_norm = np.ascontiguousarray(
        sensor_y / np.float32(cal_dim) - np.float32(0.5), dtype=np.float32)

    return t_norm


def apply_rs_correction(eye_img, angular_vel, readout_ms, t_map=None,
                        rs_factor=0.0, roll_factor=-1.5, pitch_factor=-2.5,
                        klns=None, ctrx=0.0, ctry=0.0, cal_dim=4216):
    """Apply rolling shutter correction to an equirectangular eye image.

    Corrects RS on the right eye (modded lens) using all three axes.
    The yaw mod flips the sensor 180°, so the firmware's baked-in RS correction
    for tilt, pitch and roll is reversed — all need correction.

    Derived from the fish version's 3D small-angle rotation
        Ry(yaw) · Rx(pitch) · Rz(roll):
        x' = x + yaw·z − roll·y
        y' = y + roll·x − pitch·z
        z' = z − yaw·x + pitch·y
    projected into equirect (lon, lat) coordinates:

    Yaw  (idx2, body-Z pan):   uniform vertical shift.
    Roll (idx0, body-Y roll):  Δlat =  sin(lon)·δ,  Δlon = −tan(lat)·cos(lon)·δ
    Pitch(idx1, body-X tilt):  Δlat = −cos(lon)·δ,  Δlon = +tan(lat)·sin(lon)·δ

    Uses a fisheye-aware 2D time-offset map: sensor row depends on both latitude
    and longitude via GEOC KLNS fisheye model.

    Args:
        eye_img:      H×W×3 equirectangular half (right eye)
        angular_vel:  (roll, pitch, yaw) angular velocity in deg/s
        readout_ms:   sensor readout time in milliseconds
        t_map:        precomputed (H,W) float32 time-offset map from
                      build_rs_time_map(). If None, computed on the fly.
        rs_factor:    yaw RS correction factor (default 0)
        roll_factor:  roll RS correction factor (default 2)
        pitch_factor: pitch RS correction factor (default 2)
        klns:         [c0,c1,c2,c3,c4] raw GEOC KLNS (used if t_map is None)
        ctrx, ctry:   GEOC principal point offsets (used if t_map is None)
        cal_dim:      sensor dimension (used if t_map is None)
    Returns:
        Corrected image (same shape).
    """
    if np.isscalar(angular_vel):
        angular_vel = np.array([0.0, 0.0, angular_vel])

    roll_rate  = angular_vel[0]   # idx0 = roll  = body-Y (optical axis)
    pitch_rate = angular_vel[1]   # idx1 = pitch = body-X (tilt / nod)
    yaw_rate   = angular_vel[2]   # idx2 = yaw   = body-Z (pan)

    if abs(readout_ms) < 0.01 or (abs(yaw_rate) < 0.01 and abs(roll_rate) < 0.01
                                   and abs(pitch_rate) < 0.01):
        return eye_img

    h, w = eye_img.shape[:2]
    readout_s = readout_ms / 1000.0

    # Get or compute the 2D time-offset map (fisheye sensor row geometry)
    if t_map is None:
        t_map = build_rs_time_map(h, w, klns, ctrx=ctrx, ctry=ctry, cal_dim=cal_dim)

    # Per-pixel time offset in seconds
    t_offsets_2d = t_map * np.float32(readout_s)           # (H, W) seconds

    v_px_per_deg = np.float32(h / 180.0)
    h_px_per_deg = np.float32(w / 180.0)

    # Build remap tables
    col_coords = np.arange(w, dtype=np.float32)
    row_coords = np.arange(h, dtype=np.float32)
    map_x = np.tile(col_coords, (h, 1))                                # (H, W) identity
    map_y = np.tile(row_coords[:, None], (1, w))                        # (H, W) identity

    # Yaw correction: yaw axis (idx2) = body-Z pan
    # Uniform vertical shift — same at every longitude
    if abs(yaw_rate) > 0.01:
        v_shifts_yaw = yaw_rate * t_offsets_2d * v_px_per_deg * np.float32(rs_factor)
        map_y = map_y - v_shifts_yaw

    # Precompute trig grids shared by roll and pitch corrections
    need_trig = abs(roll_rate) > 0.01 or abs(pitch_rate) > 0.01
    if need_trig:
        lat = np.linspace(np.float32(np.pi / 2), np.float32(-np.pi / 2), h, dtype=np.float32)
        lon = np.linspace(np.float32(-np.pi / 2), np.float32(np.pi / 2), w, dtype=np.float32)
        sin_lon = np.tile(np.sin(lon), (h, 1))                         # (H, W)
        cos_lon = np.tile(np.cos(lon), (h, 1))                         # (H, W)
        tan_lat = np.clip(np.tan(lat), -50.0, 50.0)                    # clamp near poles
        tan_lat_grid = np.tile(tan_lat[:, None], (1, w))                # (H, W)

    # Roll correction: roll axis (idx0) = body-Y (optical axis)
    #   Δlat =  sin(lon) · δ           → vertical, strongest at left/right edges
    #   Δlon = −tan(lat) · cos(lon) · δ  → horizontal, grows toward poles
    if abs(roll_rate) > 0.01:
        roll_t = roll_rate * t_offsets_2d * np.float32(roll_factor)     # degrees per pixel

        v_shifts_roll = roll_t * v_px_per_deg * sin_lon
        map_y = map_y - v_shifts_roll

        h_shifts_roll = roll_t * h_px_per_deg * tan_lat_grid * cos_lon
        map_x = map_x - h_shifts_roll

    # Pitch correction: pitch axis (idx1) = body-X (tilt / nod)
    #   Δlat = −cos(lon) · δ           → vertical, strongest at center, zero at edges
    #   Δlon = +tan(lat) · sin(lon) · δ  → horizontal, grows toward poles & edges
    if abs(pitch_rate) > 0.01:
        pitch_t = pitch_rate * t_offsets_2d * np.float32(pitch_factor)  # degrees per pixel

        v_shifts_pitch = pitch_t * v_px_per_deg * (-cos_lon)
        map_y = map_y - v_shifts_pitch

        h_shifts_pitch = pitch_t * h_px_per_deg * tan_lat_grid * sin_lon
        map_x = map_x - h_shifts_pitch

    map_x = np.ascontiguousarray(map_x, dtype=np.float32)
    map_y = np.ascontiguousarray(map_y, dtype=np.float32)

    return cv2.remap(eye_img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


class PreviewMode(Enum):
    SIDE_BY_SIDE = "Side by Side"
    ANAGLYPH = "Anaglyph (Red/Cyan)"
    OVERLAY_50 = "Overlay 50%"
    SINGLE_EYE = "Single Eye Mode"
    DIFFERENCE = "Difference"
    CHECKERBOARD = "Checkerboard"


@dataclass
class PanomapAdjustment:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class ProcessingConfig:
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    global_shift: int = 0
    global_adjustment: PanomapAdjustment = field(default_factory=PanomapAdjustment)
    stereo_offset: PanomapAdjustment = field(default_factory=PanomapAdjustment)
    output_codec: str = "auto"
    output_format: str = "equirect"  # "equirect" or "fisheye" (.360 only)
    quality: int = 18
    bitrate: int = 200  # Mbps
    use_bitrate: bool = True  # If False, use CRF quality; if True, use bitrate
    prores_profile: str = "standard"
    encoder_type: str = "auto"  # auto, nvenc, qsv, amf, software
    encoder_speed: str = "fast"  # fast, medium, slow
    lut_path: Optional[Path] = None  # Optional LUT file for color grading
    lut_intensity: float = 1.0  # LUT intensity 0.0 to 1.0
    gamma: float = 1.0  # Gamma: midtone adjustment via power function
    gain: float = 1.0  # Gain: overall brightness multiplier, affects highlights
    saturation: float = 1.0  # Saturation: 0.0 = grayscale, 1.0 = neutral, 2.0 = max
    lift: float = 0.0  # Lift: raises/lowers black level (shadows only, preserves white point)
    h265_bit_depth: int = 8  # 8-bit or 10-bit for H.265
    inject_vr180_metadata: bool = False  # Inject VR180 metadata for YouTube
    vision_pro_mode: str = "standard"  # standard, hvc1, or mvhevc
    # Trim settings
    trim_start: float = 0.0  # Start time in seconds
    trim_end: float = 0.0  # End time in seconds (0 = use full duration)
    # Gyro stabilization (full 3-axis CORI-based)
    gyro_data: Optional[dict] = None  # Full CORI/IORI data from parse_gopro_gyro_data()
    gyro_smooth_ms: float = 1000.0  # Heading (yaw+pitch) smoothing window in ms
    gyro_horizon_lock: bool = False  # True = fully lock roll (cancel all roll motion)
    gyro_max_corr_deg: float = 15.0  # Max heading correction angle (degrees) before soft limit
    gyro_responsiveness: float = 1.0  # Velocity response curve power (0.2–3.0, <1=anticipatory, >1=laggy)
    gyro_stabilize: bool = False  # Whether gyro stabilization is enabled
    is_360_input: bool = False  # True when input is raw .360 file (EAC dual-track)
    # Rolling shutter correction
    rs_correction_ms: float = 0.0  # Readout time in ms (0 = disabled, typical 10-16ms for GoPro MAX)
    rs_correction_enabled: bool = False  # Whether RS correction is enabled (forced on when gyro is on)
    rs_yaw_factor: float = 0.0  # Yaw RS factor: body-Z pan → horizontal shear (default 0)
    rs_pitch_factor: float = 2.0  # Pitch RS factor: undo wrong firmware RS + apply correct
    rs_roll_factor: float = 2.0  # Roll RS factor: undo wrong firmware RS + apply correct
    rs_use_cori: bool = False  # True = use CORI angular velocity (better for vibrating platforms)
    no_firmware_rs: bool = False  # True = firmware ERS disabled; apply RS (1,1,1) to BOTH eyes
    # Audio output for .360 input
    audio_ambisonics: bool = False  # True = include ambisonic 4ch PCM; False = stereo AAC only
    # GEOC lens calibration (auto-parsed from .360 file)
    geoc_klns: list = None  # [c0,c1,c2,c3,c4] raw KLNS for right eye (FRNT after yaw mod)
    geoc_ctrx: float = 0.0  # FRNT principal point X offset (pixels)
    geoc_ctry: float = 0.0  # FRNT principal point Y offset (pixels)
    geoc_cal_dim: int = 4216  # Sensor calibration dimension
    # Edge mask (circular vignette per eye in half-equirect space)
    mask_size: float = 100.0  # % of half-width radius (100 = no mask, smaller = tighter circle)
    mask_feather: float = 0.0  # Feather width in % of half-width (0 = hard edge)
    edge_fill: bool = True  # Clamp out-of-bounds EAC pixels to face edges instead of black
    # Output resolution for .360 EAC→equirect
    eac_out_w: int = 8192   # Output equirect width (SBS: each eye = eac_out_w/2 × eac_out_h). Options: 8192/8640/7680
    eac_out_h: int = 4096   # Output equirect height. Options: 4096/4320/3840
    # Equirectangular-aware sharpening
    sharpen_amount: float = 0.0  # Sharpening amount (0=off, 0.5=subtle, 1.0=moderate, 2.0=strong)
    sharpen_radius: float = 1.5  # Sharpening radius / sigma (0.5=fine detail, 3.0=coarse structure)
    denoise_strength: float = 0.0  # VideoToolbox temporal denoise (0=off, 0.5=moderate, 1.0=max, macOS 26+)
    # Multi-segment GoPro recording
    segment_paths: Optional[list] = None  # List of segment file paths (GS01..., GS02..., etc.)
    upside_down: bool = False  # Camera mounted upside down: rotates output 180° and inverts gravity


class FrameExtractor(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    raw_frame_ready = pyqtSignal(np.ndarray)  # New signal for unprocessed frame
    error = pyqtSignal(str)

    # EAC filters for .360 files
    EAC_RAW_FILTER = "[0:0][0:4]vstack"  # Raw vstack of both tracks (5952×3840)
    EAC_RAW_W, EAC_RAW_H = 5952, 3840   # Raw vstacked dimensions
    EAC_OUT_W, EAC_OUT_H = 8192, 4096   # Default full equirect (can be overridden by config)

    # Cached remap tables for EAC cross → half-equirect conversion, keyed by (out_w, out_h)
    _eac_remap_tables_cache = {}

    @staticmethod
    def _get_eac_remap_tables(out_w=None, out_h=None):
        """Get precomputed full-equirect remap tables (cached by resolution)."""
        if out_w is None:
            out_w = FrameExtractor.EAC_OUT_W
        if out_h is None:
            out_h = FrameExtractor.EAC_OUT_H
        key = (out_w, out_h)
        if key not in FrameExtractor._eac_remap_tables_cache:
            map_x, map_y, mask_A = FrameExtractor._build_eac_remap_tables(out_w, out_h)
            # Bake lens B offset into map_y for stacked-cross single-remap:
            # combined image = crossA (rows 0:3936) + 2px gap + crossB (rows 3938:7874)
            # For lens B pixels, shift map_y by 3938 to index into crossB region
            map_y_stacked = np.where(mask_A, map_y, map_y + np.float32(3938))
            FrameExtractor._eac_remap_tables_cache[key] = (map_x, map_y_stacked)
        return FrameExtractor._eac_remap_tables_cache[key]

    @staticmethod
    def build_cross_remap(R, half_w, h):
        """Build remap tables for one eye: output equirect (half_w × h) → single EAC cross (3936×3936).

        Maps each output pixel's 3D direction (after rotation R) to coordinates in a single
        EAC cross. This gives each eye access to its own lens's full 184.5° FOV without
        contamination from the other lens (no parallax seam at the lens boundary).

        Pixels outside the cross's coverage (past ±92.25°) get map coords of -1 → black with BORDER_CONSTANT.
        """
        u = np.linspace(0, 1, half_w, dtype=np.float32)
        v = np.linspace(0, 1, h, dtype=np.float32)
        ug, vg = np.meshgrid(u, v)
        lon = (ug - 0.5) * np.float32(np.pi)   # ±90°
        lat = (0.5 - vg) * np.float32(np.pi)   # ±90°
        cos_lat = np.cos(lat)
        x = cos_lat * np.sin(lon)
        y = np.sin(lat)
        z = cos_lat * np.cos(lon)

        # Apply rotation
        R = R.astype(np.float32)
        xn = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
        yn = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
        zn = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

        abs_x = np.abs(xn)
        abs_y = np.abs(yn)

        # 5-face EAC assignment (same logic as _build_eac_remap_tables, single cross)
        is_front = (zn > 0) & (abs_x <= zn) & (abs_y <= zn)
        is_right = (xn > 0) & (zn <= xn) & (abs_y <= xn)
        is_left = (xn < 0) & (zn <= abs_x) & (abs_y <= abs_x)
        is_top = (yn > 0) & (abs_x <= yn) & (zn <= yn)
        is_bottom = (yn < 0) & (abs_x <= abs_y) & (zn <= abs_y)

        map_x = np.full((h, half_w), -1, dtype=np.float32)
        map_y = np.full((h, half_w), -1, dtype=np.float32)

        # Front face: cross cols [1008, 2928), rows [1008, 2928)
        if is_front.any():
            u_eac = (2 / np.pi) * np.arctan(xn[is_front] / zn[is_front]) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yn[is_front] / zn[is_front])
            map_x[is_front] = (1008 + u_eac * 1920).astype(np.float32)
            map_y[is_front] = (1008 + v_eac * 1920).astype(np.float32)

        # Right face: cross cols [2928, 3936), rows [1008, 2928)
        if is_right.any():
            xr, yr, zr = xn[is_right], yn[is_right], zn[is_right]
            u_eac = (2 / np.pi) * np.arctan(-zr / xr) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yr / xr)
            full_col = u_eac * 1920
            valid = (full_col >= 0) & (full_col < 1008)
            mask = is_right.copy(); mask[mask] = valid
            map_x[mask] = (2928 + full_col[valid]).astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)

        # Left face: cross cols [0, 1008), rows [1008, 2928)
        if is_left.any():
            xl, yl, zl = np.abs(xn[is_left]), yn[is_left], zn[is_left]
            u_eac = (2 / np.pi) * np.arctan(zl / xl) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yl / xl)
            partial_col = u_eac * 1920 - 912
            valid = (partial_col >= 0) & (partial_col < 1008)
            mask = is_left.copy(); mask[mask] = valid
            map_x[mask] = partial_col[valid].astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)

        # Top face: cross rows [0, 1008), cols [1008, 2928)
        if is_top.any():
            xt, yt, zt = xn[is_top], yn[is_top], zn[is_top]
            u_eac = (2 / np.pi) * np.arctan(xt / yt) + 0.5
            v_eac = (2 / np.pi) * np.arctan(zt / yt) + 0.5
            partial_row = v_eac * 1920 - 912
            valid = (partial_row >= 0) & (partial_row < 1008)
            mask = is_top.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = partial_row[valid].astype(np.float32)

        # Bottom face: cross rows [2928, 3936), cols [1008, 2928)
        if is_bottom.any():
            xb, yb, zb = xn[is_bottom], np.abs(yn[is_bottom]), zn[is_bottom]
            u_eac = (2 / np.pi) * np.arctan(xb / yb) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(zb / yb)
            full_row = v_eac * 1920
            valid = (full_row >= 0) & (full_row < 1008)
            mask = is_bottom.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = (2928 + full_row[valid]).astype(np.float32)

        map_x = np.clip(map_x, -1, 3935).astype(np.float32)
        map_y = np.clip(map_y, -1, 3935).astype(np.float32)
        return map_x, map_y

    @staticmethod
    def build_cross_remap_rs(R_sensor, half_w, h, angular_vel, readout_ms,
                             yaw_factor, pitch_factor, roll_factor,
                             klns, ctrx=0.0, ctry=0.0, cal_dim=4216,
                             R_cross=None):
        """Build remap tables with combined RS correction + rotation in a single pass.

        Forward pipeline: cross → IORI cancel → RS → heading → global+stereo → output
        Remap pipeline:   d_out → heading × global × stereo → RS → IORI → cross

        R_sensor: heading × global × stereo — maps d_out to sensor space (no IORI).
        R_cross:  IORI rotation — maps sensor space back to cross space (applied after RS).
                  None or identity when IORI=0.

        RS operates in sensor space (between heading and IORI), which is correct because
        the rolling shutter time offset depends on the physical sensor row position.
        """
        u = np.linspace(0, 1, half_w, dtype=np.float32)
        v = np.linspace(0, 1, h, dtype=np.float32)
        ug, vg = np.meshgrid(u, v)
        lon = (ug - 0.5) * np.float32(np.pi)   # ±90°
        lat = (0.5 - vg) * np.float32(np.pi)   # ±90°
        cos_lat = np.cos(lat)
        x = cos_lat * np.sin(lon)
        y = np.sin(lat)
        z = cos_lat * np.cos(lon)

        # Step 1: Apply heading/global/stereo rotation → sensor-space direction
        R = R_sensor.astype(np.float32)
        xr = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
        yr = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
        zr = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

        # Step 2: Compute RS time offset from sensor position
        # theta = angle from optical axis, sensor_y = fisheye sensor row
        zr_clip = np.clip(zr, np.float32(-1.0), np.float32(1.0))
        sin_theta_sq = 1.0 - zr_clip * zr_clip
        sin_theta = np.sqrt(np.maximum(sin_theta_sq, np.float32(0.0)))
        theta = np.arccos(zr_clip)
        r_fish = klns_forward(theta, klns)

        cy = np.float32(cal_dim / 2.0 + ctry)
        c0 = np.float32(klns[0])
        cal_f = np.float32(cal_dim)
        readout_s = np.float32(readout_ms / 1000.0)

        # sensor_y = cy - r * yr / sin(theta); limit at theta→0: r/sin(theta) → c0
        safe_sin = np.where(sin_theta < np.float32(1e-7), np.float32(1.0), sin_theta)
        sensor_y = np.where(
            sin_theta < np.float32(1e-7),
            cy - c0 * yr,
            cy - r_fish * yr / safe_sin
        )
        t_offset = (sensor_y / cal_f - np.float32(0.5)) * readout_s  # seconds

        # Step 3: Apply 3D small-angle RS rotation in sensor space
        if np.isscalar(angular_vel):
            angular_vel = np.array([0.0, 0.0, angular_vel])
        roll_rate  = angular_vel[0]
        pitch_rate = angular_vel[1]
        yaw_rate   = angular_vel[2]
        deg2rad = np.float32(np.pi / 180.0)

        yaw_a   = np.float32(-yaw_rate   * yaw_factor)   * deg2rad * t_offset
        pitch_a = np.float32( pitch_rate * pitch_factor) * deg2rad * t_offset
        roll_a  = np.float32( roll_rate  * roll_factor)  * deg2rad * t_offset

        xn = xr + yaw_a * zr - roll_a * yr
        yn = yr + roll_a * xr - pitch_a * zr
        zn = zr - yaw_a * xr + pitch_a * yr

        # Step 3.5: Apply IORI rotation (sensor space → cross space)
        if R_cross is not None:
            Rc = R_cross.astype(np.float32)
            xc = Rc[0, 0] * xn + Rc[0, 1] * yn + Rc[0, 2] * zn
            yc = Rc[1, 0] * xn + Rc[1, 1] * yn + Rc[1, 2] * zn
            zc = Rc[2, 0] * xn + Rc[2, 1] * yn + Rc[2, 2] * zn
            xn, yn, zn = xc, yc, zc

        # Step 4: Map to EAC cross (same 5-face logic as build_cross_remap)
        abs_x = np.abs(xn)
        abs_y = np.abs(yn)
        is_front = (zn > 0) & (abs_x <= zn) & (abs_y <= zn)
        is_right = (xn > 0) & (zn <= xn) & (abs_y <= xn)
        is_left = (xn < 0) & (zn <= abs_x) & (abs_y <= abs_x)
        is_top = (yn > 0) & (abs_x <= yn) & (zn <= yn)
        is_bottom = (yn < 0) & (abs_x <= abs_y) & (zn <= abs_y)

        map_x = np.full((h, half_w), -1, dtype=np.float32)
        map_y = np.full((h, half_w), -1, dtype=np.float32)

        if is_front.any():
            u_eac = (2 / np.pi) * np.arctan(xn[is_front] / zn[is_front]) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yn[is_front] / zn[is_front])
            map_x[is_front] = (1008 + u_eac * 1920).astype(np.float32)
            map_y[is_front] = (1008 + v_eac * 1920).astype(np.float32)
        if is_right.any():
            xr2, yr2, zr2 = xn[is_right], yn[is_right], zn[is_right]
            u_eac = (2 / np.pi) * np.arctan(-zr2 / xr2) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yr2 / xr2)
            full_col = u_eac * 1920
            valid = (full_col >= 0) & (full_col < 1008)
            mask = is_right.copy(); mask[mask] = valid
            map_x[mask] = (2928 + full_col[valid]).astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)
        if is_left.any():
            xl, yl, zl = np.abs(xn[is_left]), yn[is_left], zn[is_left]
            u_eac = (2 / np.pi) * np.arctan(zl / xl) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(yl / xl)
            partial_col = u_eac * 1920 - 912
            valid = (partial_col >= 0) & (partial_col < 1008)
            mask = is_left.copy(); mask[mask] = valid
            map_x[mask] = partial_col[valid].astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)
        if is_top.any():
            xt, yt, zt = xn[is_top], yn[is_top], zn[is_top]
            u_eac = (2 / np.pi) * np.arctan(xt / yt) + 0.5
            v_eac = (2 / np.pi) * np.arctan(zt / yt) + 0.5
            partial_row = v_eac * 1920 - 912
            valid = (partial_row >= 0) & (partial_row < 1008)
            mask = is_top.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = partial_row[valid].astype(np.float32)
        if is_bottom.any():
            xb, yb, zb = xn[is_bottom], np.abs(yn[is_bottom]), zn[is_bottom]
            u_eac = (2 / np.pi) * np.arctan(xb / yb) + 0.5
            v_eac = 0.5 - (2 / np.pi) * np.arctan(zb / yb)
            full_row = v_eac * 1920
            valid = (full_row >= 0) & (full_row < 1008)
            mask = is_bottom.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = (2928 + full_row[valid]).astype(np.float32)

        map_x = np.clip(map_x, -1, 3935).astype(np.float32)
        map_y = np.clip(map_y, -1, 3935).astype(np.float32)
        return map_x, map_y

    @staticmethod
    def _build_eac_remap_tables(out_w=7680, out_h=3840):
        """Build remap tables for full 360°×180° equirect → both EAC crosses.

        Returns (map_x, map_y, mask_A):
        - map_x, map_y: cross coordinates (crossA coords where mask_A, crossB coords elsewhere)
        - mask_A: boolean mask — True for pixels sourced from crossA (front hemisphere)

        Each pixel's 3D direction determines which lens:
        - z >= 0 → lens A (front hemisphere)
        - z <  0 → lens B (rear hemisphere), direction transformed to B's local frame (-x, y, -z)
        """
        u = np.linspace(0, 1, out_w, dtype=np.float64)
        v = np.linspace(0, 1, out_h, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u, v)

        lon = (u_grid - 0.5) * (2 * np.pi)  # -180° to +180°
        lat = (0.5 - v_grid) * np.pi          # +90° to -90°

        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = np.cos(lat) * np.cos(lon)

        # Lens assignment: front hemisphere → A, rear → B
        mask_A = z >= 0

        # Transform rear-hemisphere pixels to lens B's local frame: (-x, y, -z)
        # After transform, z_local >= 0 for all pixels
        x_local = np.where(mask_A, x, -x)
        y_local = y  # same for both lenses
        z_local = np.abs(z)  # equivalent to np.where(mask_A, z, -z) since mask_A = (z>=0)

        abs_x = np.abs(x_local)
        abs_y = np.abs(y_local)

        is_front  = (z_local > 0) & (abs_x <= z_local) & (abs_y <= z_local)
        is_right  = (x_local > 0) & (z_local <= x_local) & (abs_y <= x_local)
        is_left   = (x_local < 0) & (z_local <= abs_x) & (abs_y <= abs_x)
        is_top    = (y_local > 0) & (abs_x <= y_local) & (z_local <= y_local)
        is_bottom = (y_local < 0) & (abs_x <= abs_y) & (z_local <= abs_y)

        map_x = np.full((out_h, out_w), -1, dtype=np.float32)
        map_y = np.full((out_h, out_w), -1, dtype=np.float32)

        # Front face: cross cols [1008, 2928), rows [1008, 2928)
        if is_front.any():
            u_eac = (2/np.pi) * np.arctan(x_local[is_front] / z_local[is_front]) + 0.5
            v_eac = 0.5 - (2/np.pi) * np.arctan(y_local[is_front] / z_local[is_front])
            map_x[is_front] = (1008 + u_eac * 1920).astype(np.float32)
            map_y[is_front] = (1008 + v_eac * 1920).astype(np.float32)

        # Right face: cross cols [2928, 3936), rows [1008, 2928)
        if is_right.any():
            xr, yr, zr = x_local[is_right], y_local[is_right], z_local[is_right]
            u_eac = (2/np.pi) * np.arctan(-zr / xr) + 0.5
            v_eac = 0.5 - (2/np.pi) * np.arctan(yr / xr)
            full_col = u_eac * 1920
            valid = (full_col >= 0) & (full_col < 1008)
            mask = is_right.copy(); mask[mask] = valid
            map_x[mask] = (2928 + full_col[valid]).astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)

        # Left face: cross cols [0, 1008), rows [1008, 2928)
        if is_left.any():
            xl, yl, zl = np.abs(x_local[is_left]), y_local[is_left], z_local[is_left]
            u_eac = (2/np.pi) * np.arctan(zl / xl) + 0.5
            v_eac = 0.5 - (2/np.pi) * np.arctan(yl / xl)
            partial_col = u_eac * 1920 - 912
            valid = (partial_col >= 0) & (partial_col < 1008)
            mask = is_left.copy(); mask[mask] = valid
            map_x[mask] = partial_col[valid].astype(np.float32)
            map_y[mask] = (1008 + v_eac[valid] * 1920).astype(np.float32)

        # Top face: cross rows [0, 1008), cols [1008, 2928)
        if is_top.any():
            xt, yt, zt = x_local[is_top], y_local[is_top], z_local[is_top]
            u_eac = (2/np.pi) * np.arctan(xt / yt) + 0.5
            v_eac = (2/np.pi) * np.arctan(zt / yt) + 0.5
            partial_row = v_eac * 1920 - 912
            valid = (partial_row >= 0) & (partial_row < 1008)
            mask = is_top.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = partial_row[valid].astype(np.float32)

        # Bottom face: cross rows [2928, 3936), cols [1008, 2928)
        if is_bottom.any():
            xb, yb, zb = x_local[is_bottom], np.abs(y_local[is_bottom]), z_local[is_bottom]
            u_eac = (2/np.pi) * np.arctan(xb / yb) + 0.5
            v_eac = 0.5 - (2/np.pi) * np.arctan(zb / yb)
            full_row = v_eac * 1920
            valid = (full_row >= 0) & (full_row < 1008)
            mask = is_bottom.copy(); mask[mask] = valid
            map_x[mask] = (1008 + u_eac[valid] * 1920).astype(np.float32)
            map_y[mask] = (2928 + full_row[valid]).astype(np.float32)

        map_x = np.clip(map_x, 0, 3935).astype(np.float32)
        map_y = np.clip(map_y, 0, 3935).astype(np.float32)
        return map_x, map_y, mask_A

    @staticmethod
    def _fill_cross_corners(cross):
        """Fill the 4 black corners of an EAC cross with edge-replicated pixels.

        Prevents bilinear interpolation from bleeding black into the equirect output.
        Each corner is filled by replicating the nearest edge from the adjacent side face.
        """
        # Top-left corner (0:1008, 0:1008): replicate left face's top edge row
        cross[0:1008, 0:1008] = cross[1008, 0:1008][np.newaxis, :, :]
        # Top-right corner (0:1008, 2928:3936): replicate right face's top edge row
        cross[0:1008, 2928:3936] = cross[1008, 2928:3936][np.newaxis, :, :]
        # Bottom-left corner (2928:3936, 0:1008): replicate left face's bottom edge row
        cross[2928:3936, 0:1008] = cross[2927, 0:1008][np.newaxis, :, :]
        # Bottom-right corner (2928:3936, 2928:3936): replicate right face's bottom edge row
        cross[2928:3936, 2928:3936] = cross[2927, 2928:3936][np.newaxis, :, :]

        # Fix sub-pixel black lines at face seams.
        # Bilinear interpolation at face boundaries samples from pixels on both
        # sides. If there's any discontinuity between adjacent face tiles (e.g.,
        # from different streams s0 vs s4), it creates a thin dark line.
        # Fix by replicating the inner face's edge pixel over the boundary pixel.
        #
        # Cross layout:  [corner] [top]    [corner]   rows 0-1007
        #                [left]   [center] [right]    rows 1008-2927
        #                [corner] [bottom] [corner]   rows 2928-3935

        # ── Seam fixes: replicate inner-face edge pixels over boundary pixels ──
        # Bilinear interpolation samples floor(coord) and floor(coord)+1, so
        # we need 1 pixel of padding at every face↔face and face↔corner boundary.

        # Center face ↔ adjacent faces (horizontal seams at rows 1007/1008 and 2927/2928)
        cross[1007, 1008:2928] = cross[1008, 1008:2928]   # top↔center
        cross[2928, 1008:2928] = cross[2927, 1008:2928]   # center↔bottom

        # Center face ↔ adjacent faces (vertical seams at cols 1007/1008 and 2927/2928)
        cross[1008:2928, 1007] = cross[1008:2928, 1008]   # left↔center
        cross[1008:2928, 2928] = cross[1008:2928, 2927]   # center↔right

        # Top/Bottom face ↔ corner areas (vertical seams).
        # Without these, bilinear interpolation at the Top face's right edge
        # (col 2927→2928) reads TR corner data (broadcast from Right face top
        # row), creating a visible seam line — most noticeable upper-right in VR.
        cross[0:1008, 2928] = cross[0:1008, 2927]         # top face → TR corner
        cross[0:1008, 1007] = cross[0:1008, 1008]         # top face → TL corner
        cross[2928:3936, 2928] = cross[2928:3936, 2927]   # bottom face → BR corner
        cross[2928:3936, 1007] = cross[2928:3936, 1008]   # bottom face → BL corner

    @staticmethod
    def _assemble_lensA(s0, s4, cross=None):
        """Assemble Lens A EAC cross (3936×3936) from streams.
        If cross buffer is provided, fills it in-place (avoids allocation).
        Otherwise a new buffer is allocated with the same dtype as the
        inputs (uint8 for the legacy subprocess path, uint16 for the new
        Level B preview path that keeps 10-bit precision through decode).
        Corner regions are filled by _fill_cross_corners via edge replication,
        so we skip zeroing the entire buffer (saves ~46MB memset per frame)."""
        if cross is None:
            cross = np.zeros((3936, 3936, 3), dtype=s0.dtype)
        # 5 face tiles overwrite the cross arms; corners are filled by _fill_cross_corners
        cross[0:1008, 1008:2928] = cv2.rotate(s4[:, 4944:5952], cv2.ROTATE_90_CLOCKWISE)
        cross[1008:2928, 0:1008] = s0[:, 1008:2016]
        cross[1008:2928, 1008:2928] = s0[:, 2016:3936]
        cross[1008:2928, 2928:3936] = s0[:, 3936:4944]
        cross[2928:3936, 1008:2928] = cv2.rotate(s4[:, 0:1008], cv2.ROTATE_90_CLOCKWISE)
        FrameExtractor._fill_cross_corners(cross)
        return cross

    @staticmethod
    def _assemble_lensB(s0, s4, cross=None):
        """Assemble Lens B EAC cross (3936×3936) from streams.
        If cross buffer is provided, fills it in-place (avoids allocation).
        Otherwise a new buffer is allocated with the same dtype as the
        inputs (see _assemble_lensA for context).
        Corner regions are filled by _fill_cross_corners via edge replication,
        so we skip zeroing the entire buffer (saves ~46MB memset per frame)."""
        s4_rot = cv2.rotate(s4[:, 1008:4944], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if cross is None:
            cross = np.zeros((3936, 3936, 3), dtype=s0.dtype)
        # 5 face tiles overwrite the cross arms; corners are filled by _fill_cross_corners
        cross[0:1008, 1008:2928] = s4_rot[0:1008, :]
        cross[1008:2928, 0:1008] = s0[:, 4944:5952]
        cross[1008:2928, 1008:2928] = s4_rot[1008:2928, :]
        cross[1008:2928, 2928:3936] = s0[:, 0:1008]
        cross[2928:3936, 1008:2928] = s4_rot[2928:3936, :]
        FrameExtractor._fill_cross_corners(cross)
        return cross

    @staticmethod
    def convert_360_raw_to_equirect(raw_vstack):
        """Convert raw vstacked EAC frame (5952×3840) to 360° equirect (7680×3840).

        Full 360°×180° equirect. Lens A (front hemisphere, z>=0) centered at lon=0°,
        lens B (rear hemisphere, z<0) wraps around the sides.
        Uses a single cv2.remap on a stacked cross image for performance.
        """
        s0 = raw_vstack[:1920, :]
        s4 = raw_vstack[1920:, :]
        crossA = FrameExtractor._assemble_lensA(s0, s4)
        crossB = FrameExtractor._assemble_lensB(s0, s4)
        # Stack both crosses with a 2px gap to prevent bilinear bleed across boundary
        # Layout: crossA (rows 0:3936) | gap (3936:3938) | crossB (rows 3938:7874)
        combined = np.zeros((3936 + 2 + 3936, 3936, 3), dtype=raw_vstack.dtype)
        combined[:3936] = crossA
        combined[3938:] = crossB
        map_x, map_y = FrameExtractor._get_eac_remap_tables()
        return cv2.remap(combined, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def __init__(self, video_path: Path, timestamp: float = 0.0, filter_complex: str = None,
                 extract_raw: bool = False, is_360: bool = False):
        super().__init__()
        self.video_path = video_path
        self.timestamp = timestamp
        self.filter_complex = filter_complex
        self.extract_raw = extract_raw  # If True, extract without filters for caching
        self.is_360 = is_360  # True for raw .360 EAC input
        self._process = None  # Store FFmpeg process for termination
        self._cancelled = False

    def cancel(self):
        """Cancel the frame extraction by killing FFmpeg process"""
        self._cancelled = True
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=1)
            except:
                try:
                    self._process.kill()
                except:
                    pass

    def _extract_raw_via_pyav(self):
        """Decode a single frame (or s0/s4 pair for .360) via PyAV + videotoolbox
        hwaccel, using the same gbrp16le color path as the export's
        _pyav_decode_stream. Emits raw_frame_ready on success, error on failure.
        Returns True if it handled the request (even with error), False if
        PyAV is not available and the caller should fall through to the
        legacy FFmpeg subprocess path.

        Why this exists: the preview used to shell out to FFmpeg with
        `-pix_fmt rgb24` which goes through a DIFFERENT swscale codepath than
        the export's uint16 gbrp16le reformat, producing a small but
        systematic color shift between preview and export (measured: ~6 codes
        mean, up to 23 max on uint8). By routing preview decode through the
        same PyAV hwaccel + gbrp16le path as the export and then
        downconverting to uint8 only at emit time, preview and export share
        identical decode-side color math. The rest of the preview pipeline
        (LUTs, sharpen, etc.) stays uint8 — this is "Level A" of the
        WYSIWYG alignment.
        """
        if not HAS_AV:
            return False
        # Open container with videotoolbox hwaccel on macOS; fall back to
        # plain software decode if the hwaccel context can't be created.
        container = None
        if sys.platform == 'darwin':
            try:
                from av.codec.hwaccel import HWAccel
                _hw = HWAccel(device_type='videotoolbox', allow_software_fallback=True)
                container = av.open(str(self.video_path), hwaccel=_hw)
            except Exception:
                container = None
        if container is None:
            try:
                container = av.open(str(self.video_path))
            except Exception as e:
                self.error.emit(f"PyAV open error: {e}")
                return True

        try:
            target_ts = max(0.0, float(self.timestamp))
            try:
                container.seek(int(target_ts * 1_000_000))
            except Exception:
                # Seek may fail on some containers; decoding from the
                # start still yields the correct frame below.
                pass

            if self.is_360:
                # GoPro .360 has two video streams — stream 0 and stream 4 in
                # the file, but PyAV indexes them as streams.video[0] and [1].
                if len(container.streams.video) < 2:
                    self.error.emit(".360 file missing second video stream")
                    return True
                vs0 = container.streams.video[0]
                vs1 = container.streams.video[1]
                vs0.thread_type = 'AUTO'
                vs1.thread_type = 'AUTO'

                s0_rgb16 = None
                s4_rgb16 = None
                _vs0_index = vs0.index
                _vs1_index = vs1.index
                # PyAV 17 removed frame.stream — we must tag frames at packet
                # level and track which stream they came from.
                for packet in container.demux(vs0, vs1):
                    if self._cancelled:
                        return True
                    _pkt_stream_idx = packet.stream.index if packet.stream is not None else -1
                    if _pkt_stream_idx == _vs0_index:
                        _slot = 0
                    elif _pkt_stream_idx == _vs1_index:
                        _slot = 1
                    else:
                        continue
                    # Skip already-filled slots to avoid wasted reformat.
                    if (_slot == 0 and s0_rgb16 is not None) or \
                       (_slot == 1 and s4_rgb16 is not None):
                        continue
                    for frame in packet.decode():
                        if frame is None or frame.pts is None:
                            continue
                        frame_time = float(frame.pts * frame.time_base)
                        if frame_time < target_ts - 0.001:
                            continue
                        # PyAV's gbrp16le → to_ndarray() returns (H, W, 3)
                        # in RGB order (verified empirically 2026-04-08).
                        _rgb16 = frame.reformat(format='gbrp16le').to_ndarray()
                        if _slot == 0 and s0_rgb16 is None:
                            s0_rgb16 = _rgb16
                        elif _slot == 1 and s4_rgb16 is None:
                            s4_rgb16 = _rgb16
                        if s0_rgb16 is not None and s4_rgb16 is not None:
                            break
                    if s0_rgb16 is not None and s4_rgb16 is not None:
                        break
                if self._cancelled:
                    return True
                if s0_rgb16 is None or s4_rgb16 is None:
                    self.error.emit("Could not extract .360 frame at target timestamp")
                    return True
                # Level B: emit the full preview frame as uint16 RGB. The
                # entire downstream preview filter pipeline (color grade,
                # 3D LUT, sharpen, saturation, mask) is uint16 now, matching
                # the export's working precision. uint16 → uint8 downconvert
                # is deferred to _update_preview() right before the QImage
                # conversion (QImage.Format_RGB888 is the display boundary).
                self.crossA = self._assemble_lensA(s0_rgb16, s4_rgb16)
                self.crossB = self._assemble_lensB(s0_rgb16, s4_rgb16)
                combined = np.zeros((3936 + 2 + 3936, 3936, 3), dtype=np.uint16)
                combined[:3936] = self.crossA
                combined[3938:] = self.crossB
                map_x, map_y = self._get_eac_remap_tables()
                frame = cv2.remap(combined, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                if self._cancelled:
                    return True
                self.raw_frame_ready.emit(frame)
                return True

            # Non-.360 single-stream extract_raw path
            vs = container.streams.video[0]
            vs.thread_type = 'AUTO'
            for frame in container.decode(vs):
                if self._cancelled:
                    return True
                if frame is None or frame.pts is None:
                    continue
                frame_time = float(frame.pts * frame.time_base)
                if frame_time < target_ts - 0.001:
                    continue
                # Level B: emit uint16 RGB (same rationale as the .360 path)
                _rgb16 = frame.reformat(format='gbrp16le').to_ndarray()
                if self._cancelled:
                    return True
                self.raw_frame_ready.emit(_rgb16)
                return True

            self.error.emit("No frame found at target timestamp")
            return True
        finally:
            try:
                container.close()
            except Exception:
                pass

    def run(self):
        try:
            # Preview cached-raw-frame path: decode via PyAV + videotoolbox
            # hwaccel + gbrp16le so the preview shares the same color math as
            # the export. The legacy FFmpeg subprocess path below is only kept
            # as a fallback and for the filter-complex (extract_raw=False)
            # code path which needs ffmpeg filter graphs we don't port.
            if self.extract_raw:
                if self._extract_raw_via_pyav():
                    return
                # PyAV not available → fall through to legacy subprocess path.

            # ── Legacy FFmpeg subprocess path ──────────────────────────────
            # For .360 input: decode raw vstack, then assemble crosses + remap in Python
            if self.is_360:
                raw_w, raw_h = self.EAC_RAW_W, self.EAC_RAW_H
                eac_filter = self.EAC_RAW_FILTER + "[out]"
                cmd = [get_ffmpeg_path(),
                       "-ss", str(self.timestamp), "-i", str(self.video_path),
                       "-filter_complex", eac_filter, "-map", "[out]",
                       "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"]

                self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=get_subprocess_flags())
                stdout, stderr = self._process.communicate()

                if self._cancelled:
                    return
                if self._process.returncode != 0:
                    self.error.emit(f"FFmpeg error: {stderr.decode()[:200]}")
                    return

                raw_frame = np.frombuffer(stdout, dtype=np.uint8).reshape((raw_h, raw_w, 3))
                # Assemble individual crosses (cache for per-eye preview remap)
                s0 = raw_frame[:1920, :]
                s4 = raw_frame[1920:, :]
                self.crossA = self._assemble_lensA(s0, s4)
                self.crossB = self._assemble_lensB(s0, s4)
                # Convert raw EAC vstack → 360° equirect (7680×3840)
                combined = np.zeros((3936 + 2 + 3936, 3936, 3), dtype=raw_frame.dtype)
                combined[:3936] = self.crossA
                combined[3938:] = self.crossB
                map_x, map_y = self._get_eac_remap_tables()
                frame = cv2.remap(combined, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                if self.extract_raw:
                    self.raw_frame_ready.emit(frame)
                else:
                    self.frame_ready.emit(frame)
                return

            # Standard (non-.360) path
            probe_cmd = [get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", str(self.video_path)]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            info = json.loads(result.stdout)
            width, height = info["streams"][0]["width"], info["streams"][0]["height"]

            # For raw extraction (caching), use hardware decode and no filters
            if self.extract_raw:
                # Check codec for hardware decode
                probe_codec = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                            "-show_entries", "stream=codec_name", "-of", "json", str(self.video_path)],
                                            capture_output=True, text=True, creationflags=get_subprocess_flags())
                codec = json.loads(probe_codec.stdout)["streams"][0]["codec_name"]

                # Use hardware decode for HEVC on macOS
                hwaccel_args = []
                if codec in ["hevc", "h265"] and sys.platform == 'darwin':
                    hwaccel_args = ["-hwaccel", "videotoolbox"]

                cmd = [get_ffmpeg_path()] + hwaccel_args + [
                    "-ss", str(self.timestamp), "-i", str(self.video_path),
                    "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"
                ]

                self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=get_subprocess_flags())
                stdout, stderr = self._process.communicate()

                if self._cancelled:
                    return

                if self._process.returncode != 0:
                    self.error.emit(f"FFmpeg error")
                    return

                frame = np.frombuffer(stdout, dtype=np.uint8).reshape((height, width, 3))
                self.raw_frame_ready.emit(frame)
            else:
                # Normal extraction with filters
                # Check codec for hardware decode
                probe_codec = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                            "-show_entries", "stream=codec_name", "-of", "json", str(self.video_path)],
                                            capture_output=True, text=True, creationflags=get_subprocess_flags())
                codec = json.loads(probe_codec.stdout)["streams"][0]["codec_name"]

                # Use hardware decode for HEVC on macOS (works with filters too)
                hwaccel_args = []
                if codec in ["hevc", "h265"] and sys.platform == 'darwin':
                    hwaccel_args = ["-hwaccel", "videotoolbox"]

                cmd = [get_ffmpeg_path()] + hwaccel_args + ["-ss", str(self.timestamp), "-i", str(self.video_path)]
                if self.filter_complex:
                    cmd.extend(["-filter_complex", self.filter_complex, "-map", "[out]"])
                cmd.extend(["-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"])

                self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=get_subprocess_flags())
                stdout, stderr = self._process.communicate()

                if self._cancelled:
                    return

                if self._process.returncode != 0:
                    self.error.emit(f"FFmpeg error")
                    return

                frame = np.frombuffer(stdout, dtype=np.uint8).reshape((height, width, 3))
                self.frame_ready.emit(frame)
        except Exception as e:
            self.error.emit(str(e))


class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    output_line = pyqtSignal(str)  # New signal for FFmpeg output lines
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        self._cancelled = False
        self._process = None

    def cancel(self):
        self._cancelled = True
        # Also kill the FFmpeg process if running
        if self._process:
            try:
                self._process.terminate()
            except:
                pass
    
    def run(self):
        """Process video using per-frame OpenCV pipeline.

        Handles all input types (.360, equirect SBS) with GPU-accelerated remap,
        optional gyro stabilization, RS correction, LUT, color grading, and sharpening.
        """
        import bisect
        import math

        if not HAS_CV2:
            self.finished_signal.emit(False, "OpenCV (cv2) is required for gyro stabilization")
            return

        cfg = self.config
        self.status.emit("Processing with OpenCV pipeline...")

        try:
            # Get video info
            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,r_frame_rate,pix_fmt,codec_name,color_range,color_space,color_transfer,color_primaries",
                                   "-show_entries", "format=duration", "-of", "json",
                                   str(cfg.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            video_info = json.loads(probe.stdout)
            stream = video_info["streams"][0]
            # Extract color space tags from input to preserve in output
            _src_color_range = stream.get("color_range")       # e.g. "tv", "pc"
            _src_color_space = stream.get("color_space")       # e.g. "bt709", "bt2020nc"
            _src_color_trc = stream.get("color_transfer")      # e.g. "bt709", "arib-std-b67"
            _src_color_primaries = stream.get("color_primaries")  # e.g. "bt709", "bt2020"
            full_duration = float(video_info.get("format", {}).get("duration", 0))

            # Multi-segment: compute per-segment durations and total duration
            is_multi_segment = cfg.segment_paths is not None and len(cfg.segment_paths) > 1
            segment_durations = []  # (seg_path, seg_duration) for each segment
            if is_multi_segment:
                total_seg_duration = 0.0
                for seg_path in cfg.segment_paths:
                    try:
                        seg_probe = subprocess.run([get_ffprobe_path(), "-v", "quiet",
                                                    "-show_entries", "format=duration", "-of", "json",
                                                    str(seg_path)], capture_output=True, text=True, check=True,
                                                   creationflags=get_subprocess_flags())
                        seg_info = json.loads(seg_probe.stdout)
                        seg_dur = float(seg_info.get("format", {}).get("duration", 0))
                    except Exception:
                        seg_dur = 0.0
                    segment_durations.append((seg_path, seg_dur))
                    total_seg_duration += seg_dur
                full_duration = total_seg_duration

            # For .360 input, override dimensions to EAC→equirect output size
            if cfg.is_360_input:
                if cfg.output_format == "fisheye":
                    width = 8192                      # 4096 per eye SBS
                    height = 4096
                    out_height = height
                else:
                    width = cfg.eac_out_w             # 8192 (or 7680)
                    height = cfg.eac_out_h            # 4096 (or 3840)
                    out_height = height               # same — standard 180° vertical
            else:
                width = stream["width"]
                height = stream["height"]
                out_height = height
            codec = stream["codec_name"]

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

            # Calculate trim bounds
            start_time = cfg.trim_start if cfg.trim_start > 0 else 0
            end_time = cfg.trim_end if cfg.trim_end > 0 else full_duration
            duration = end_time - start_time
            total_frames = int(duration * fps)

            print(f"[EXPORT] trim_start={cfg.trim_start:.3f}s, trim_end={cfg.trim_end:.3f}s, "
                  f"start_time={start_time:.3f}s, end_time={end_time:.3f}s, "
                  f"duration={duration:.3f}s, total_frames={total_frames}, fps={fps:.2f}, "
                  f"gyro_stabilize={cfg.gyro_stabilize}, rs_enabled={cfg.rs_correction_enabled}")
            self.status.emit(f"Will process {total_frames} frames at {fps:.2f} fps using OpenCV")

            # Initialize gyro stabilizer (also used for RS correction angular velocity)
            if cfg.gyro_data:
                self.status.emit("Computing gyro stabilization corrections...")
                gyro_stabilizer = GyroStabilizer(cfg.gyro_data)
            else:
                gyro_stabilizer = None
            if gyro_stabilizer:
                gyro_stabilizer.smooth(
                    window_ms=cfg.gyro_smooth_ms,
                    horizon_lock=cfg.gyro_horizon_lock,
                    stabilize=cfg.gyro_stabilize,
                    max_corr_deg=cfg.gyro_max_corr_deg,
                    responsiveness=cfg.gyro_responsiveness,
                    upside_down=cfg.upside_down,
                )
                if cfg.gyro_stabilize:
                    hl_desc = ", horizon lock" if cfg.gyro_horizon_lock else ""
                    self.status.emit(f"Gyro stabilization ready ({cfg.gyro_smooth_ms:.0f}ms{hl_desc})")
                    # Force -1920 shift for .360 when gyro stabilization is enabled
                    # (non-360 SBS eye swap is handled by _effective_shift below)
                    if cfg.is_360_input:
                        cfg.global_shift = -1920
                else:
                    self.status.emit("RS/IORI correction mode (no gyro stabilization)")

            # Precompute global+stereo view adjustment matrices (separate from camera correction)
            # These are user view adjustments, applied AFTER RS correction.
            ga = cfg.global_adjustment
            so = cfg.stereo_offset
            ga_arr = np.array([ga.roll, ga.yaw, ga.pitch])
            so_arr = np.array([so.roll, so.yaw, so.pitch])
            has_view_adj = np.any(np.abs(ga_arr) > 0.01) or np.any(np.abs(so_arr) > 0.01)
            if has_view_adj:
                q_view_left = euler_to_quat(*(ga_arr - so_arr))
                q_view_right = euler_to_quat(*(ga_arr + so_arr))
                R_view_left = quat_to_rotation_matrix(*q_view_left)
                R_view_right = quat_to_rotation_matrix(*q_view_right)
            else:
                R_view_left = None
                R_view_right = None

            # Pre-compute all per-frame stabilization matrices upfront
            # Eliminates per-frame SLERP + bisect + quat_to_matrix + matrix multiply overhead
            _export_rs_enabled = cfg.rs_correction_enabled and cfg.rs_correction_ms > 0.01
            precomputed = None
            if gyro_stabilizer:
                self.status.emit("Pre-computing stabilization matrices...")
                _srot_s = cfg.rs_correction_ms / 1000.0 if _export_rs_enabled else 0.0
                precomputed = gyro_stabilizer.precompute_export_matrices(
                    start_time, fps, total_frames, _export_rs_enabled,
                    R_view_left, R_view_right, srot_s=_srot_s,
                    rs_yaw_factor=cfg.rs_yaw_factor, rs_pitch_factor=cfg.rs_pitch_factor,
                    rs_roll_factor=cfg.rs_roll_factor,
                    rs_use_cori=cfg.rs_use_cori)
                print(f"[EXPORT] Precomputed {total_frames} matrices: "
                      f"gyro timestamps[0]={gyro_stabilizer.timestamps[0]:.3f}s, "
                      f"[-1]={gyro_stabilizer.timestamps[-1]:.3f}s, "
                      f"first lookup t={start_time:.3f}s, last lookup t={start_time + (total_frames-1)/fps:.3f}s")
                # Debug: show first few frame matrices to verify correctness
                _dbg_L0 = precomputed['left_matrices'][0]
                _dbg_mid = total_frames // 2
                _dbg_Lm = precomputed['left_matrices'][_dbg_mid]
                print(f"[EXPORT] Frame 0 left mat diag: [{_dbg_L0[0,0]:.6f}, {_dbg_L0[1,1]:.6f}, {_dbg_L0[2,2]:.6f}]")
                print(f"[EXPORT] Frame {_dbg_mid} left mat diag: [{_dbg_Lm[0,0]:.6f}, {_dbg_Lm[1,1]:.6f}, {_dbg_Lm[2,2]:.6f}]")
                self.status.emit(f"Pre-computed {total_frames} frame matrices"
                                 f"{' (with RS)' if _export_rs_enabled else ''}")

            # Determine output codec
            if codec in ["hevc", "h265"]:
                codec = "h265"
            elif codec in ["prores", "prores_ks"]:
                codec = "prores"
            output_codec = codec if cfg.output_codec == "auto" else cfg.output_codec

            # Function to apply equirectangular roll using OpenCV
            def apply_equirect_roll(img, roll_deg):
                """Apply roll rotation to a half-equirectangular image using OpenCV remap.

                For equirectangular projection, roll (rotation around view axis) requires
                remapping each pixel through spherical coordinates.
                """
                if abs(roll_deg) < 0.01:
                    return img

                h, w = img.shape[:2]
                roll_rad = math.radians(roll_deg)
                cos_r = math.cos(roll_rad)
                sin_r = math.sin(roll_rad)

                # Create coordinate grids
                # For half-equirectangular: longitude spans -90 to +90 degrees (π/2 range)
                # latitude spans -90 to +90 degrees
                u = np.linspace(0, 1, w, dtype=np.float32)
                v = np.linspace(0, 1, h, dtype=np.float32)
                u_grid, v_grid = np.meshgrid(u, v)

                # Convert to spherical angles (half equirectangular)
                # longitude: -π/2 to +π/2, latitude: -π/2 to +π/2
                lon = (u_grid - 0.5) * math.pi  # -π/2 to +π/2
                lat = (0.5 - v_grid) * math.pi  # +π/2 to -π/2 (top to bottom)

                # Convert to 3D cartesian coordinates
                cos_lat = np.cos(lat)
                x = cos_lat * np.sin(lon)
                y = np.sin(lat)
                z = cos_lat * np.cos(lon)

                # Apply roll rotation around Z axis (forward direction)
                x_new = cos_r * x - sin_r * y
                y_new = sin_r * x + cos_r * y
                z_new = z

                # Convert back to spherical coordinates
                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)

                # Convert back to image coordinates
                u_new = (lon_new / math.pi) + 0.5  # Map -π/2..+π/2 to 0..1
                v_new = 0.5 - (lat_new / math.pi)  # Map +π/2..-π/2 to 0..1

                # Scale to pixel coordinates
                map_x = (u_new * w).astype(np.float32)
                map_y = (v_new * h).astype(np.float32)

                # Handle wrapping for out-of-bounds coordinates
                map_x = np.clip(map_x, 0, w - 1)
                map_y = np.clip(map_y, 0, h - 1)

                # Apply remapping with Lanczos interpolation for sharper output
                result = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                return result

            # (Roll remap is baked into per-frame rotation matrix R — no separate table needed)

            # Load LUT if specified
            lut_3d = None
            if cfg.lut_path and cfg.lut_path.exists():
                try:
                    lut_3d = load_cube_lut(str(cfg.lut_path))
                    self.status.emit(f"Loaded LUT: {cfg.lut_path.name}")
                except Exception as e:
                    self.status.emit(f"Warning: Failed to load LUT: {e}")

            def build_color_1d_lut(lift, gamma, gain, bit_depth=8):
                """Build 1D LUT for color adjustments. 256-entry uint8 for preview, 65536-entry uint16 for export."""
                n_entries = 256 if bit_depth == 8 else 65536
                pix_max = 255.0 if bit_depth == 8 else 65535.0
                x = np.arange(n_entries, dtype=np.float32) / pix_max
                if abs(lift) > 0.01:
                    x = x + lift * (1.0 - x)
                if abs(gain - 1.0) > 0.01:
                    x = x * gain
                x = np.clip(x, 0.0, 1.0)
                if abs(gamma - 1.0) > 0.01:
                    x = np.power(x, 1.0 / gamma)
                return np.clip(x * pix_max, 0, pix_max).astype(np.uint8 if bit_depth == 8 else np.uint16)

            def apply_color_1d(frame, lut_1d):
                """Apply 1D color LUT via cv2.LUT (uint8) or Numba parallel (uint16)."""
                if lut_1d is None:
                    return frame
                if frame.dtype == np.uint16:
                    if HAS_NUMBA:
                        out = np.empty_like(frame)
                        _nb_apply_1d_lut(frame.ravel(), lut_1d, out.ravel())
                        return out
                    return lut_1d[frame]  # numpy fallback
                return cv2.LUT(frame, lut_1d)

            # Precompute MLX LUT array (persistent on GPU)
            _mlx_lut_flat = None
            _lut_3d_size = 0
            def _prepare_mlx_lut(lut):
                nonlocal _mlx_lut_flat, _lut_3d_size
                if HAS_MLX and lut is not None and _mlx_lut_flat is None:
                    _mlx_lut_flat = mx.array(lut.ravel())
                    _lut_3d_size = lut.shape[0]

            def apply_lut_3d_fast(frame, lut, intensity):
                """Apply 3D LUT at full resolution. Uses Metal GPU > Numba > numpy."""
                if lut is None or intensity < 0.01:
                    return frame
                h, w = frame.shape[:2]
                lut_size = lut.shape[0]
                n_pixels = h * w

                if HAS_MLX:
                    _prepare_mlx_lut(lut)
                    _pm = 65535.0 if frame.dtype == np.uint16 else 255.0
                    if intensity < 1.0:
                        result = _mlx_apply_lut_3d(frame, _mlx_lut_flat, lut_size, n_pixels, pix_max=_pm)
                        orig_f = frame.astype(np.float32)
                        res_f = result.astype(np.float32)
                        blended = orig_f + (res_f - orig_f) * np.float32(intensity)
                        return np.clip(blended, 0, _pm).astype(frame.dtype)
                    else:
                        return _mlx_apply_lut_3d(frame, _mlx_lut_flat, lut_size, n_pixels, pix_max=_pm)

                if HAS_NUMBA_CUDA:
                    result = _cuda_apply_lut_3d(frame, lut, lut_size)
                    if intensity < 1.0:
                        orig_f = frame.astype(np.float32)
                        res_f = result.astype(np.float32)
                        blended = orig_f + (res_f - orig_f) * np.float32(intensity)
                        _pm = 65535 if frame.dtype == np.uint16 else 255
                        return np.clip(blended, 0, _pm).astype(frame.dtype)
                    return result

                if HAS_WGPU and _wgpu_device is not None:
                    result = _wgpu_apply_lut_3d(frame, lut.ravel(), lut_size, n_pixels)
                    if intensity < 1.0:
                        orig_f = frame.astype(np.float32)
                        res_f = result.astype(np.float32)
                        blended = orig_f + (res_f - orig_f) * np.float32(intensity)
                        _pm = 65535 if frame.dtype == np.uint16 else 255
                        return np.clip(blended, 0, _pm).astype(frame.dtype)
                    return result

                if HAS_NUMBA:
                    out = np.empty_like(frame)
                    _pix_max = 65535 if frame.dtype == np.uint16 else 255
                    _nb_apply_lut_3d(frame, lut, out, lut_size, _pix_max)
                    if intensity < 1.0:
                        orig_f = frame.astype(np.float32)
                        res_f = out.astype(np.float32)
                        blended = orig_f + (res_f - orig_f) * np.float32(intensity)
                        return np.clip(blended, 0, _pix_max).astype(frame.dtype)
                    return out

                # Numpy fallback
                _pm_np = 65535.0 if frame.dtype == np.uint16 else 255.0
                img = frame.astype(np.float32) / _pm_np

                b_idx = np.clip(img[:, :, 0] * (lut_size - 1), 0, lut_size - 1.001)
                g_idx = np.clip(img[:, :, 1] * (lut_size - 1), 0, lut_size - 1.001)
                r_idx = np.clip(img[:, :, 2] * (lut_size - 1), 0, lut_size - 1.001)

                b0 = b_idx.astype(np.int32); b1 = np.minimum(b0 + 1, lut_size - 1)
                g0 = g_idx.astype(np.int32); g1 = np.minimum(g0 + 1, lut_size - 1)
                r0 = r_idx.astype(np.int32); r1 = np.minimum(r0 + 1, lut_size - 1)

                fb = (b_idx - b0)[:, :, np.newaxis]
                fg = (g_idx - g0)[:, :, np.newaxis]
                fr = (r_idx - r0)[:, :, np.newaxis]

                c000 = lut[b0, g0, r0]; c001 = lut[b0, g0, r1]
                c010 = lut[b0, g1, r0]; c011 = lut[b0, g1, r1]
                c100 = lut[b1, g0, r0]; c101 = lut[b1, g0, r1]
                c110 = lut[b1, g1, r0]; c111 = lut[b1, g1, r1]

                c00 = c000 + (c001 - c000) * fr
                c01 = c010 + (c011 - c010) * fr
                c10 = c100 + (c101 - c100) * fr
                c11 = c110 + (c111 - c110) * fr
                c0 = c00 + (c01 - c00) * fg
                c1 = c10 + (c11 - c10) * fg
                result_rgb = c0 + (c1 - c0) * fb

                if intensity < 1.0:
                    img_rgb = img[:, :, ::-1]
                    result_rgb = img_rgb + (result_rgb - img_rgb) * intensity

                return np.clip(result_rgb[:, :, ::-1] * _pm_np, 0, _pm_np).astype(frame.dtype)

            # ── Equirectangular-aware sharpening ────────────────────────────
            # CUDA (Windows/NVIDIA): full GPU separable blur + fused USM kernel
            # CPU fallback (macOS/other): stackBlur + band-based addWeighted
            _sharpen_enabled = cfg.sharpen_amount > 0.01
            _sharpen_sigma = cfg.sharpen_radius if hasattr(cfg, 'sharpen_radius') else 1.5
            _sharpen_amount = cfg.sharpen_amount
            _sharpen_bands = None
            _sharpen_blur_buf = None
            _sharpen_stack_ksize = 5
            _sharpen_use_cuda = False
            _sharpen_use_mlx = False
            _cuda_sharpen_kernel_d = None
            _cuda_sharpen_lat_d = None
            _cuda_sharpen_temp_d = None
            _cuda_sharpen_out_d = None
            _cuda_sharpen_src_d = None
            _cuda_sharpen_ksize = 5
            # For CUDA fused pipeline: numpy arrays passed to _cuda_fused_process
            _fused_sharpen_kernel = None
            _fused_sharpen_lat = None
            _fused_sharpen_ksize = 5
            # For MLX Metal sharpen
            _mlx_sharpen_g1d = None
            _mlx_sharpen_lat = None
            _mlx_sharpen_ksize = 5
            if _sharpen_enabled:
                # Gaussian 1D kernel for separable blur
                ksize = max(3, int(np.ceil(_sharpen_sigma * 6)) | 1)
                x_k = np.arange(ksize, dtype=np.float32) - ksize // 2
                g1d = np.exp(-0.5 * (x_k / _sharpen_sigma) ** 2).astype(np.float32)
                g1d /= g1d.sum()

                # Latitude weight map. In equirect output, `cos(latitude)`
                # gives a per-row falloff that attenuates the sharpen toward
                # the poles — each pixel there represents a tiny angular
                # region, so full-strength USM would over-sharpen noise
                # into visible artifacts. In fisheye output each pixel
                # has roughly uniform angular density (determined by the
                # KLNS projection), so the falloff would only waste
                # sharpening budget on the edge of the circle. Use a flat
                # weight in that mode.
                _sharpen_is_fisheye = (cfg.output_format == "fisheye"
                                        and cfg.is_360_input)
                if _sharpen_is_fisheye:
                    lat_weights = np.ones(height, dtype=np.float32)
                else:
                    lat_weights = np.array([
                        float(np.cos((0.5 - y / height) * np.pi))
                        for y in range(height)
                    ], dtype=np.float32)
                    lat_weights = np.clip(lat_weights, 0.02, 1.0)

                if HAS_NUMBA_CUDA:
                    # CUDA: upload kernel + lat weights, preallocate device buffers
                    n_elements = height * width * 3
                    _cuda_sharpen_kernel_d = _numba_cuda.to_device(g1d)
                    _cuda_sharpen_lat_d = _numba_cuda.to_device(lat_weights)
                    _cuda_sharpen_temp_d = _numba_cuda.device_array(n_elements, dtype=np.uint16)
                    _cuda_sharpen_out_d = _numba_cuda.device_array(n_elements, dtype=np.uint16)
                    _cuda_sharpen_src_d = _numba_cuda.device_array(n_elements, dtype=np.uint16)
                    _cuda_sharpen_ksize = ksize
                    _sharpen_use_cuda = True
                    # Also store numpy arrays for fused pipeline
                    _fused_sharpen_kernel = g1d
                    _fused_sharpen_lat = lat_weights
                    _fused_sharpen_ksize = ksize
                    print(f"  CUDA sharpening: sigma={_sharpen_sigma:.1f}, kernel={ksize}px, "
                          f"{height}x{width} ({n_elements//3} pixels)")
                elif HAS_MLX:
                    # MLX Metal GPU: separable blur + latitude-weighted USM
                    _mlx_sharpen_g1d = g1d
                    _mlx_sharpen_lat = lat_weights
                    _mlx_sharpen_ksize = ksize
                    _sharpen_use_mlx = True
                    print(f"  MLX sharpening: sigma={_sharpen_sigma:.1f}, kernel={ksize}px, "
                          f"{height}x{width}")
                else:
                    # CPU fallback: smooth per-row latitude-weighted USM
                    _sharpen_bands = True  # flag: use smooth path
                    # Precompute per-row sharpening weights. In equirect
                    # output we use a cos(lat) falloff (see comment above);
                    # in fisheye output use a uniform weight since pixel
                    # density is already roughly constant.
                    if _sharpen_is_fisheye:
                        _sharpen_row_weights = np.full(
                            height,
                            np.clip(_sharpen_amount, 0.02, 4.0),
                            dtype=np.float32)
                    else:
                        _row_centers = (np.arange(height, dtype=np.float32) + 0.5) / height
                        _row_lat = (0.5 - _row_centers) * np.float32(np.pi)
                        _sharpen_row_weights = np.clip(np.cos(_row_lat) * _sharpen_amount, 0.02, 4.0).astype(np.float32)
                    # Reshape to (H, 1, 1) for broadcasting with (H, W, 3) frames
                    _sharpen_row_weights = _sharpen_row_weights[:, np.newaxis, np.newaxis]
                    _sharpen_blur_buf = np.empty((height, width, 3), dtype=np.uint16)
                    _sharpen_stack_ksize = max(3, int(np.sqrt(12 * _sharpen_sigma**2 + 1) + 0.5) | 1)

            _has_stack_blur = hasattr(cv2, 'stackBlur')

            def apply_equirect_sharpen(frame, bands):
                """Equirectangular-aware USM. CUDA/MLX GPU accelerated, CPU fallback."""
                if not _sharpen_enabled:
                    return frame

                if _sharpen_use_mlx:
                    _pm = 65535.0 if frame.dtype == np.uint16 else 255.0
                    return _mlx_sharpen(frame, _mlx_sharpen_g1d, _mlx_sharpen_lat,
                                        _sharpen_amount, _mlx_sharpen_ksize, pix_max=_pm)

                if _sharpen_use_cuda:
                    # Full GPU path: separable blur + fused USM (persistent buffers)
                    h_f, w_f = frame.shape[:2]
                    n_elem = h_f * w_f * 3
                    _cuda_sharpen_src_d.copy_to_device(np.ascontiguousarray(frame).ravel())
                    grid = (n_elem + _CUDA_BLOCK - 1) // _CUDA_BLOCK
                    # Pass 1: horizontal blur
                    _pm = np.int32(65535 if frame.dtype == np.uint16 else 255)
                    _cuda_blur_h_kernel[grid, _CUDA_BLOCK](
                        _cuda_sharpen_src_d, _cuda_sharpen_temp_d, _cuda_sharpen_kernel_d,
                        np.int32(_cuda_sharpen_ksize), np.int32(h_f), np.int32(w_f), _pm)
                    # Pass 2: vertical blur + fused latitude USM
                    _cuda_blur_v_usm_kernel[grid, _CUDA_BLOCK](
                        _cuda_sharpen_src_d, _cuda_sharpen_temp_d, _cuda_sharpen_lat_d,
                        _cuda_sharpen_out_d, np.float32(_sharpen_amount),
                        _cuda_sharpen_kernel_d, np.int32(_cuda_sharpen_ksize),
                        np.int32(h_f), np.int32(w_f), _pm)
                    _numba_cuda.synchronize()
                    return _cuda_sharpen_out_d.copy_to_host().reshape(h_f, w_f, 3)

                # CPU fallback: smooth per-row latitude-weighted USM
                if not bands:
                    return frame
                # cv2.stackBlur/GaussianBlur may not handle uint16 correctly on all
                # platforms — convert to float32 for blur to guarantee correctness.
                frame_f = frame.astype(np.float32)
                if _has_stack_blur:
                    blurred = cv2.stackBlur(frame_f, (_sharpen_stack_ksize, _sharpen_stack_ksize))
                else:
                    blurred = cv2.GaussianBlur(frame_f, (0, 0), sigmaX=_sharpen_sigma)
                # Smooth USM: frame + weight * (frame - blur)
                diff = frame_f - blurred
                result = frame_f + _sharpen_row_weights * diff
                _pm = 65535.0 if frame.dtype == np.uint16 else 255.0
                np.clip(result, 0, _pm, out=result)
                return result.astype(frame.dtype)

            # (Yaw/pitch remap is baked into per-frame rotation matrix R — no separate table needed)

            # --- Optimized single-remap pipeline ---
            # Precompute base 3D coordinate grid once in float32 (half memory, better cache)
            half_w = width // 2
            _fisheye_mode = (cfg.output_format == "fisheye" and cfg.is_360_input)

            if _fisheye_mode and cfg.geoc_klns:
                # Fisheye output: 4096×4096 per eye (cropped from 4216 sensor, centered)
                # Use center (0,0) so both eyes produce identical, centered circles
                _cal = cfg.geoc_cal_dim or 4216
                _fish_out = 4096  # output size per eye
                _offset = (_cal - _fish_out) / 2.0  # center crop offset
                _cx = _cal / 2.0 - _offset  # center in cropped coords
                _cy = _cal / 2.0 - _offset
                _yy, _xx = np.mgrid[0:_fish_out, 0:_fish_out].astype(np.float32)
                _r = np.sqrt((_xx - _cx)**2 + (_yy - _cy)**2)
                _phi = np.arctan2(_yy - _cy, _xx - _cx)
                _c0,_c1,_c2,_c3,_c4 = [np.float32(c) for c in cfg.geoc_klns]
                _theta = _r / _c0
                for _ in range(5):
                    _t2 = _theta * _theta
                    _r_est = _theta*(_c0+_t2*(_c1+_t2*(_c2+_t2*(_c3+_t2*_c4))))
                    _dr = _c0+_t2*(3*_c1+_t2*(5*_c2+_t2*(7*_c3+_t2*9*_c4)))
                    _theta -= (_r_est - _r) / np.maximum(_dr, 1e-6)
                    _theta = np.clip(_theta, 0, np.pi)
                _sin_t = np.sin(_theta); _cos_t = np.cos(_theta)
                # Direction vectors (will be flipped at output via remap)
                _xn = _sin_t * np.cos(_phi)
                _yn = _sin_t * np.sin(_phi)
                _zn = _cos_t
                # Apply 180° rotation + mirror: flip rows (same as remap fix)
                _xn = _xn[::-1, :]; _yn = _yn[::-1, :]; _zn = _zn[::-1, :]
                xyz_x = _xn.ravel().copy()
                xyz_y = _yn.ravel().copy()
                xyz_z = _zn.ravel().copy()
                # Fisheye circle mask: precompute uint16 3-channel for cv2.bitwise_and (~6ms/frame)
                _max_r = klns_forward(np.float32(np.pi * 184.5 / 360), cfg.geoc_klns)
                _fish_mask_1ch = ((_r < _max_r)[::-1, :]).astype(np.uint16)  # 1 inside, 0 outside
                _fish_mask_3ch = np.stack([_fish_mask_1ch]*3, axis=-1) * np.uint16(65535)
                _fish_invalid_2d = None  # not used, using bitwise_and instead
                del _xx, _yy, _r, _phi, _theta, _sin_t, _cos_t, _xn, _yn, _zn, _fish_mask_1ch
            else:
                # Equirectangular output: out_height × half_w (180° vertical)
                u_base = np.linspace(0, 1, half_w, dtype=np.float32)
                v_base = np.linspace(0, 1, out_height, dtype=np.float32)
                u_grid, v_grid = np.meshgrid(u_base, v_base)
                lon_base = (u_grid - 0.5) * np.float32(math.pi)
                lat_base = (0.5 - v_grid) * np.float32(math.pi)
                cos_lat = np.cos(lat_base)
                xyz_x = (cos_lat * np.sin(lon_base)).ravel()
                xyz_y = np.sin(lat_base).ravel()
                xyz_z = (cos_lat * np.cos(lon_base)).ravel()
                _fish_invalid_2d = None
                del u_base, v_base, u_grid, v_grid, lon_base, lat_base, cos_lat

            # CUDA: upload direction grids to GPU (persist across all frames)
            _cuda_xyz_x = _cuda_xyz_y = _cuda_xyz_z = None
            _cuda_t_offset = _cuda_zero_t_offset = None
            _cuda_n_pixels = 0
            if HAS_NUMBA_CUDA:
                _cuda_xyz_x = _cuda.to_device(xyz_x)
                _cuda_xyz_y = _cuda.to_device(xyz_y)
                _cuda_xyz_z = _cuda.to_device(xyz_z)
                _cuda_n_pixels = xyz_x.shape[0]
                _cuda_zero_t_offset = _cuda.to_device(np.zeros(xyz_x.shape[0], dtype=np.float32))
                _cuda_t_offset = _cuda_zero_t_offset

            # MLX: upload direction grids to GPU (persist across all frames)
            _mlx_xyz_x = _mlx_xyz_y = _mlx_xyz_z = None
            _mlx_n_pixels = 0
            _mlx_zero_t_offset = _mlx_t_offset = None
            if HAS_MLX:
                _mlx_xyz_x = mx.array(xyz_x)
                _mlx_xyz_y = mx.array(xyz_y)
                _mlx_xyz_z = mx.array(xyz_z)
                _mlx_n_pixels = xyz_x.shape[0]
                _mlx_zero_t_offset = mx.zeros((xyz_x.shape[0],), dtype=mx.float32)
                _mlx_t_offset = _mlx_zero_t_offset  # default; overwritten if RS enabled

            # wgpu persistent buffers
            _wgpu_xyz_x_buf = _wgpu_xyz_y_buf = _wgpu_xyz_z_buf = None
            _wgpu_t_offset_buf = None
            _wgpu_out_buf = None
            _wgpu_n_pixels_render = 0
            if HAS_WGPU and _wgpu_init():
                _wgpu_n_pixels_render = xyz_x.shape[0]
                _wgpu_xyz_x_buf = _wgpu_create_buffer(xyz_x)
                _wgpu_xyz_y_buf = _wgpu_create_buffer(xyz_y)
                _wgpu_xyz_z_buf = _wgpu_create_buffer(xyz_z)
                _wgpu_t_offset_buf = _wgpu_create_buffer(np.zeros(xyz_x.shape[0], dtype=np.float32))
                _wgpu_out_buf = _wgpu_create_empty_buffer(_wgpu_n_pixels_render * 3 * 4)
                print(f"  wgpu render buffers ready ({_wgpu_n_pixels_render} pixels)")

            # Precompute the pixel scale constants
            # Input/output both cover 180° vertical (standard equirect ±90°)
            inv_pi = np.float32(1.0 / math.pi)
            half_w_f = np.float32(half_w)
            height_f = np.float32(height)
            inv_v_fov = inv_pi  # ±90° vertical for all inputs

            def build_ypr_matrix(yaw_deg, pitch_deg, roll_deg):
                """Build a 3x3 rotation matrix from yaw/pitch/roll in degrees (float32)."""
                yaw = math.radians(yaw_deg)
                pitch = math.radians(pitch_deg)
                roll = math.radians(roll_deg)
                cy, sy = math.cos(yaw), math.sin(yaw)
                cp, sp = math.cos(pitch), math.sin(pitch)
                cr, sr = math.cos(roll), math.sin(roll)
                R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
                R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
                R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                return R_yaw @ R_pitch @ R_roll

            def compute_remap_tables(R):
                """Compute remap tables from rotation matrix using precomputed xyz grid.
                Output: (out_height × half_w) remap tables mapping to input pixels.
                """
                R = R.astype(np.float32)
                x_new = R[0, 0] * xyz_x + R[0, 1] * xyz_y + R[0, 2] * xyz_z
                y_new = R[1, 0] * xyz_x + R[1, 1] * xyz_y + R[1, 2] * xyz_z
                z_new = R[2, 0] * xyz_x + R[2, 1] * xyz_y + R[2, 2] * xyz_z
                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)
                # Horizontal: input covers ±90° (same as output)
                map_x = np.clip((lon_new * inv_pi + 0.5) * half_w_f, 0, half_w_f - 1).reshape(out_height, half_w)
                # Vertical: standard ±90° (same as horizontal)
                map_y = np.clip((0.5 - lat_new * inv_v_fov) * height_f, 0, height_f - 1).reshape(out_height, half_w)
                return map_x, map_y

            # For non-360 SBS input: always apply -width/2 base shift to swap eye halves
            # (code puts left_src → right output, so left half of input must contain right-eye data)
            _effective_shift = cfg.global_shift
            if not cfg.is_360_input:
                _effective_shift = -(width // 2) + cfg.global_shift

            # Precompute global shift slicing offsets (avoid np.roll which copies entire frame)
            # After shift: "left var" = right-eye data, "right var" = left-eye data
            shift_abs = abs(_effective_shift) if _effective_shift != 0 else 0
            if _effective_shift < 0:
                left_start = shift_abs
            elif _effective_shift > 0:
                left_start = width - shift_abs
            else:
                left_start = 0
            left_end = left_start + half_w
            right_start = left_end % width
            right_end = right_start + half_w
            # Check if right eye slice wraps around
            right_wraps = right_end > width

            # Global adjustment and stereo offset are baked into gyro corrections
            # No separate manual adjustment matrices needed

            # Pre-allocate triple output buffers (out_height × width)
            # Triple-buffering: process writes to one buffer, encode thread reads another
            # via memoryview (eliminating tobytes() copy ~13ms/frame at 8K), and a third
            # buffer is free for the next frame. This allows full pipeline overlap.
            _result_bufs = [
                np.empty((out_height, width, 3), dtype=np.uint16),
                np.empty((out_height, width, 3), dtype=np.uint16),
                np.empty((out_height, width, 3), dtype=np.uint16),
            ]
            _result_buf_idx = 0
            result_buf = _result_bufs[0]
            _identity3 = np.eye(3, dtype=np.float64)

            # Pre-allocate cross assembly buffers (avoid 92MB allocation per frame)
            # Ring-buffered cross arrays: need enough sets so decode thread never
            # overwrites a buffer still in the queue or being processed.
            # Queue maxsize=4 + 1 being processed + 1 being filled = 6 sets.
            _N_CROSS_BUFS = 6
            if cfg.is_360_input:
                _cross_bufs_A = [np.zeros((3936, 3936, 3), dtype=np.uint16) for _ in range(_N_CROSS_BUFS)]
                _cross_bufs_B = [np.zeros((3936, 3936, 3), dtype=np.uint16) for _ in range(_N_CROSS_BUFS)]
                _cross_buf_A = _cross_bufs_A[0]
                _cross_buf_B = _cross_bufs_B[0]
                _cross_buf_idx = [0]  # mutable for closure
            else:
                _cross_buf_A = _cross_buf_B = None
                _cross_bufs_A = _cross_bufs_B = None

            # Precompute circular edge mask (applied per eye in half-equirect space)
            _eye_mask = None
            if cfg.mask_size < 100.0:
                # Build angular distance map in half-equirect space so mask
                # appears circular in VR (not in flat projection).
                # Each eye covers ±90° H and ±90° V.
                yy, xx = np.mgrid[:out_height, :half_w]
                # Normalize to [-1, 1] from center
                u = (xx + 0.5) / half_w * 2.0 - 1.0       # -1..+1 horizontal
                v = (yy + 0.5) / out_height * 2.0 - 1.0   # -1..+1 vertical
                lon = u * (np.pi / 2.0)   # ±90° in radians
                lat = v * (np.pi / 2.0)   # ±90° in radians
                # Angular distance from center: acos(cos(lat)*cos(lon))
                cos_ang = np.clip(np.cos(lat) * np.cos(lon), -1.0, 1.0)
                ang_dist = np.arccos(cos_ang)  # 0 at center, π/2 at corners
                # Normalize so 1.0 = 90° (the edge of the half-equirect)
                r = ang_dist / (np.pi / 2.0)
                # radius where mask starts fading (as fraction of 90°)
                r_inner = cfg.mask_size / 100.0
                r_feather = cfg.mask_feather / 100.0
                r_outer = r_inner + r_feather
                if r_feather > 0.001:
                    mask = np.clip((r_outer - r) / r_feather, 0.0, 1.0)
                else:
                    mask = (r <= r_inner).astype(np.float32)
                # Precompute as uint16 (0-256) 3-channel for fast integer multiply:
                # (img.astype(uint16) * mask_u16) >> 8  avoids float conversion
                mask_u16 = (mask * 256.0).astype(np.uint16)
                _eye_mask = np.stack([mask_u16, mask_u16, mask_u16], axis=-1)  # (H, half_w, 3)

            # Each eye needs its own xyz grid for thread-safe parallel remap computation
            xyz_x_r, xyz_y_r, xyz_z_r = xyz_x.copy(), xyz_y.copy(), xyz_z.copy()

            # Export uses Lanczos4 (4×4 sinc) for sharper output; preview uses bilinear for speed
            _export_interp = cv2.INTER_LANCZOS4

            def _process_right_eye(left_src, R_right_eye):
                """Process right eye (runs in thread). Uses _r grid copies for thread safety."""
                R = R_right_eye.astype(np.float32)
                x_new = R[0, 0] * xyz_x_r + R[0, 1] * xyz_y_r + R[0, 2] * xyz_z_r
                y_new = R[1, 0] * xyz_x_r + R[1, 1] * xyz_y_r + R[1, 2] * xyz_z_r
                z_new = R[2, 0] * xyz_x_r + R[2, 1] * xyz_y_r + R[2, 2] * xyz_z_r
                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)
                mx = np.clip((lon_new * inv_pi + 0.5) * half_w_f, 0, half_w_f - 1).reshape(out_height, half_w)
                my = np.clip((0.5 - lat_new * inv_v_fov) * height_f, 0, height_f - 1).reshape(out_height, half_w)
                result_buf[:, half_w:] = cv2.remap(left_src, mx, my,
                                                   _export_interp, borderMode=cv2.BORDER_REFLECT)

            def _process_left_eye(right_src, R_left_eye):
                """Process left eye (runs in thread). Uses original grid."""
                R = R_left_eye.astype(np.float32)
                x_new = R[0, 0] * xyz_x + R[0, 1] * xyz_y + R[0, 2] * xyz_z
                y_new = R[1, 0] * xyz_x + R[1, 1] * xyz_y + R[1, 2] * xyz_z
                z_new = R[2, 0] * xyz_x + R[2, 1] * xyz_y + R[2, 2] * xyz_z
                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)
                mx = np.clip((lon_new * inv_pi + 0.5) * half_w_f, 0, half_w_f - 1).reshape(out_height, half_w)
                my = np.clip((0.5 - lat_new * inv_v_fov) * height_f, 0, height_f - 1).reshape(out_height, half_w)
                result_buf[:, :half_w] = cv2.remap(right_src, mx, my,
                                                   _export_interp, borderMode=cv2.BORDER_REFLECT)

            # RS correction config (needed early for precompute decision)
            rs_ms = cfg.rs_correction_ms
            rs_yaw_factor = cfg.rs_yaw_factor
            rs_pitch_factor = cfg.rs_pitch_factor
            rs_roll_factor = cfg.rs_roll_factor
            # Two-step RS mode: when factors are (0, 2, 2), use two-step undo+reapply
            # which computes a new time offset at the corrected sensor position.
            _rs_two_step = (abs(rs_yaw_factor) < 0.01 and
                           abs(rs_pitch_factor - 2.0) < 0.01 and
                           abs(rs_roll_factor - 2.0) < 0.01)
            geoc_klns = cfg.geoc_klns
            geoc_ctrx = cfg.geoc_ctrx
            geoc_ctry = cfg.geoc_ctry
            geoc_cal_dim = cfg.geoc_cal_dim
            rs_enabled = cfg.rs_correction_enabled and rs_ms > 0.01
            # RS requires GEOC lens calibration for correct time offset mapping
            if rs_enabled and geoc_klns is None:
                print("[RS] WARNING: RS correction disabled — GEOC lens calibration not found in .360 file")
                rs_enabled = False
            # Check if per-scanline RS will be available (needs 800Hz gyro)
            _rs_perscanline_available = (gyro_stabilizer is not None and
                                         hasattr(gyro_stabilizer, 'gyro_angvel') and
                                         gyro_stabilizer.gyro_angvel is not None)
            if _rs_perscanline_available and rs_enabled:
                print(f"[RS] Per-scanline quaternion mode: {GyroStabilizer.RS_N_GROUPS} row groups, "
                      f"exact rotation from 800Hz gyro integration")
            elif _rs_two_step:
                print("[RS] Two-step mode: undo firmware (-1,+1,+1) then apply correct (+1,+1,+1)")

            # For .360 input: per-eye cross remap (no lens mixing at boundary)
            # Each eye remaps directly from its own EAC cross, avoiding the parallax
            # seam that occurs when both lenses are merged into a single equirect.
            # All .360 paths use cross remap for full 184.5° FOV access.
            if cfg.is_360_input:
                # Preallocate scratch buffers for cross map computation (avoid per-frame alloc)
                _cmx_l = np.full(xyz_x.shape[0], np.float32(-1), dtype=np.float32)
                _cmy_l = np.full(xyz_x.shape[0], np.float32(-1), dtype=np.float32)
                _cmx_r = np.full(xyz_x.shape[0], np.float32(-1), dtype=np.float32)
                _cmy_r = np.full(xyz_x.shape[0], np.float32(-1), dtype=np.float32)
                _two_over_pi = np.float32(2.0 / np.pi)

                def _dirs_to_cross_maps(xn, yn, zn, mx_buf, my_buf):
                    """Map rotated direction vectors to EAC cross (3936×3936) coordinates.
                    Uses Numba JIT when available, numpy fallback otherwise."""
                    if HAS_NUMBA:
                        _nb_dirs_to_cross_maps(xn, yn, zn, mx_buf, my_buf, out_height, half_w)
                        return (mx_buf.reshape(out_height, half_w),
                                my_buf.reshape(out_height, half_w))
                    # Numpy fallback
                    mx_buf[:] = np.float32(-1)
                    my_buf[:] = np.float32(-1)
                    abs_x = np.abs(xn)
                    abs_y = np.abs(yn)
                    is_front = (zn > 0) & (abs_x <= zn) & (abs_y <= zn)
                    is_right = (xn > 0) & (zn <= xn) & (abs_y <= xn)
                    is_left = (xn < 0) & (zn <= abs_x) & (abs_y <= abs_x)
                    is_top = (yn > 0) & (abs_x <= yn) & (zn <= yn)
                    is_bottom = (yn < 0) & (abs_x <= abs_y) & (zn <= abs_y)
                    if is_front.any():
                        u_eac = _two_over_pi * np.arctan(xn[is_front] / zn[is_front]) + np.float32(0.5)
                        v_eac = np.float32(0.5) - _two_over_pi * np.arctan(yn[is_front] / zn[is_front])
                        mx_buf[is_front] = np.float32(1008) + u_eac * np.float32(1920)
                        my_buf[is_front] = np.float32(1008) + v_eac * np.float32(1920)
                    if is_right.any():
                        idx = np.flatnonzero(is_right)
                        xr, yr, zr = xn[idx], yn[idx], zn[idx]
                        u_eac = _two_over_pi * np.arctan(-zr/xr) + np.float32(0.5)
                        v_eac = np.float32(0.5) - _two_over_pi * np.arctan(yr/xr)
                        full_col = np.clip(u_eac * np.float32(1920), 0, 1007)
                        mx_buf[idx] = np.float32(2928) + full_col
                        my_buf[idx] = np.float32(1008) + v_eac * np.float32(1920)
                    if is_left.any():
                        idx = np.flatnonzero(is_left)
                        xl = np.abs(xn[idx])
                        yl, zl = yn[idx], zn[idx]
                        u_eac = _two_over_pi * np.arctan(zl/xl) + np.float32(0.5)
                        v_eac = np.float32(0.5) - _two_over_pi * np.arctan(yl/xl)
                        partial_col = np.clip(u_eac * np.float32(1920) - np.float32(912), 0, 1007)
                        mx_buf[idx] = partial_col
                        my_buf[idx] = np.float32(1008) + v_eac * np.float32(1920)
                    if is_top.any():
                        idx = np.flatnonzero(is_top)
                        xt, yt, zt = xn[idx], yn[idx], zn[idx]
                        u_eac = _two_over_pi * np.arctan(xt/yt) + np.float32(0.5)
                        v_eac = _two_over_pi * np.arctan(zt/yt) + np.float32(0.5)
                        partial_row = np.clip(v_eac * np.float32(1920) - np.float32(912), 0, 1007)
                        mx_buf[idx] = np.float32(1008) + u_eac * np.float32(1920)
                        my_buf[idx] = partial_row
                    if is_bottom.any():
                        idx = np.flatnonzero(is_bottom)
                        xb, yb = xn[idx], np.abs(yn[idx])
                        zb = zn[idx]
                        u_eac = _two_over_pi * np.arctan(xb/yb) + np.float32(0.5)
                        v_eac = np.float32(0.5) - _two_over_pi * np.arctan(zb/yb)
                        full_row = np.clip(v_eac * np.float32(1920), 0, 1007)
                        mx_buf[idx] = np.float32(1008) + u_eac * np.float32(1920)
                        my_buf[idx] = np.float32(2928) + full_row
                    np.clip(mx_buf, -1, 3935, out=mx_buf)
                    np.clip(my_buf, -1, 3935, out=my_buf)
                    return (mx_buf.reshape(out_height, half_w),
                            my_buf.reshape(out_height, half_w))

                def _process_right_eye_cross(crossA, R_right_eye):
                    """Process right eye from its own EAC cross (thread). Uses _r grid copies."""
                    R = R_right_eye.astype(np.float32)
                    if HAS_NUMBA:
                        _nb_cross_remap_rot(xyz_x_r, xyz_y_r, xyz_z_r,
                                            R[0,0], R[0,1], R[0,2],
                                            R[1,0], R[1,1], R[1,2],
                                            R[2,0], R[2,1], R[2,2],
                                            _cmx_r, _cmy_r, xyz_x_r.shape[0])
                        mx = _cmx_r.reshape(out_height, half_w)
                        my = _cmy_r.reshape(out_height, half_w)
                    else:
                        xn = R[0, 0] * xyz_x_r + R[0, 1] * xyz_y_r + R[0, 2] * xyz_z_r
                        yn = R[1, 0] * xyz_x_r + R[1, 1] * xyz_y_r + R[1, 2] * xyz_z_r
                        zn = R[2, 0] * xyz_x_r + R[2, 1] * xyz_y_r + R[2, 2] * xyz_z_r
                        mx, my = _dirs_to_cross_maps(xn, yn, zn, _cmx_r, _cmy_r)
                    result_buf[:, half_w:] = cv2.remap(crossA, mx, my,
                                                       _export_interp, borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=(0, 0, 0))

                def _process_left_eye_cross(crossB, R_left_eye):
                    """Process left eye from its own EAC cross. Uses original grid."""
                    R = R_left_eye.astype(np.float32)
                    if HAS_NUMBA:
                        _nb_cross_remap_rot(xyz_x, xyz_y, xyz_z,
                                            R[0,0], R[0,1], R[0,2],
                                            R[1,0], R[1,1], R[1,2],
                                            R[2,0], R[2,1], R[2,2],
                                            _cmx_l, _cmy_l, xyz_x.shape[0])
                        mx = _cmx_l.reshape(out_height, half_w)
                        my = _cmy_l.reshape(out_height, half_w)
                    else:
                        xn = R[0, 0] * xyz_x + R[0, 1] * xyz_y + R[0, 2] * xyz_z
                        yn = R[1, 0] * xyz_x + R[1, 1] * xyz_y + R[1, 2] * xyz_z
                        zn = R[2, 0] * xyz_x + R[2, 1] * xyz_y + R[2, 2] * xyz_z
                        mx, my = _dirs_to_cross_maps(xn, yn, zn, _cmx_l, _cmy_l)
                    result_buf[:, :half_w] = cv2.remap(crossB, mx, my,
                                                       _export_interp, borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=(0, 0, 0))

            # Precompute constant used by RS code paths (must be in scope for closures)
            _rs_deg2rad = np.float32(np.pi / 180.0)

            # For .360 with RS: precompute constants for combined single-pass RS+rotation
            if cfg.is_360_input and rs_enabled and geoc_klns:

                # Precompute RS time offset map from identity (unrotated) directions.
                # R is always a small rotation (few degrees), so the sensor-row time
                # offset is nearly the same as the identity case. This eliminates
                # per-frame arccos, sqrt, klns_forward, and np.where on 14.7M elements.
                _rs_cx = np.float32(geoc_cal_dim / 2.0 + geoc_ctrx)
                _rs_cy = np.float32(geoc_cal_dim / 2.0 + geoc_ctry)
                _rs_c0 = np.float32(geoc_klns[0])
                _rs_cal_f = np.float32(geoc_cal_dim)
                _rs_readout_s = np.float32(rs_ms / 1000.0)
                _id_z_clip = np.clip(xyz_z, np.float32(-1.0), np.float32(1.0))
                _id_sin_theta = np.sqrt(np.maximum(np.float32(1.0) - _id_z_clip * _id_z_clip, np.float32(0.0)))
                _id_theta = np.arccos(_id_z_clip)
                _id_r_fish = klns_forward(_id_theta, geoc_klns)
                _id_safe_sin = np.where(_id_sin_theta < np.float32(1e-7), np.float32(1.0), _id_sin_theta)
                _id_sensor_y = np.where(
                    _id_sin_theta < np.float32(1e-7),
                    _rs_cy - _rs_c0 * xyz_y,
                    _rs_cy - _id_r_fish * xyz_y / _id_safe_sin
                )
                _rs_t_offset_identity = ((_id_sensor_y / _rs_cal_f - np.float32(0.5)) * _rs_readout_s).copy()
                _rs_t_offset_identity_r = _rs_t_offset_identity.copy()  # thread-safe copy for right eye
                # Upload RS t_offset to GPU
                if HAS_MLX:
                    _mlx_t_offset = mx.array(_rs_t_offset_identity)
                if HAS_NUMBA_CUDA and _cuda_zero_t_offset is not None:
                    _cuda_t_offset = _cuda.to_device(_rs_t_offset_identity)
                if HAS_WGPU and _wgpu_t_offset_buf is not None:
                    _wgpu_device.queue.write_buffer(_wgpu_t_offset_buf, 0, _rs_t_offset_identity.tobytes())
                del _id_z_clip, _id_sin_theta, _id_theta, _id_r_fish, _id_safe_sin, _id_sensor_y

                # Set KLNS params for sensor-space RS correction in MLX kernel
                if HAS_MLX:
                    _mlx_klns_params[:] = [
                        geoc_klns[0], geoc_klns[1], geoc_klns[2], geoc_klns[3], geoc_klns[4],
                        _rs_cx, _rs_cy, _rs_cal_f, _rs_readout_s
                    ]

                def _process_right_eye_cross_rs(crossA, R_sensor, angular_vel, R_cross=None):
                    """Combined RS + rotation from EAC cross in a single remap pass (thread).

                    Forward: cross → IORI cancel → RS → heading → global+stereo → output
                    Remap:   d_out → heading × global × stereo → RS → IORI → cross

                    R_sensor = heading × global × stereo (maps d_out to sensor space).
                    R_cross  = IORI (maps sensor space → cross space, applied after RS).
                    """
                    R = R_sensor.astype(np.float32)
                    roll_rate  = np.float32(angular_vel[0])
                    pitch_rate = np.float32(angular_vel[1])
                    yaw_rate   = np.float32(angular_vel[2])
                    yaw_coeff   = np.float32(yaw_rate   * rs_yaw_factor)   * _rs_deg2rad
                    pitch_coeff = np.float32( pitch_rate * rs_pitch_factor) * _rs_deg2rad
                    roll_coeff  = np.float32( roll_rate  * rs_roll_factor)  * _rs_deg2rad

                    # Skip IORI rotation when R_cross ≈ identity (optimization #6)
                    has_Rc = R_cross is not None and np.max(np.abs(R_cross - np.eye(3, dtype=np.float32))) > 1e-6

                    if HAS_NUMBA:
                        if has_Rc:
                            Rc = R_cross.astype(np.float32)
                        else:
                            Rc = np.eye(3, dtype=np.float32)
                        _nb_cross_remap_rs(
                            xyz_x_r, xyz_y_r, xyz_z_r,
                            R[0,0], R[0,1], R[0,2],
                            R[1,0], R[1,1], R[1,2],
                            R[2,0], R[2,1], R[2,2],
                            _rs_t_offset_identity_r,
                            yaw_coeff, pitch_coeff, roll_coeff,
                            has_Rc,
                            Rc[0,0], Rc[0,1], Rc[0,2],
                            Rc[1,0], Rc[1,1], Rc[1,2],
                            Rc[2,0], Rc[2,1], Rc[2,2],
                            _cmx_r, _cmy_r, xyz_x_r.shape[0])
                        mx = _cmx_r.reshape(out_height, half_w)
                        my = _cmy_r.reshape(out_height, half_w)
                    else:
                        # Numpy fallback — 3D rotation (matches Metal kernel)
                        xr = R[0, 0] * xyz_x_r + R[0, 1] * xyz_y_r + R[0, 2] * xyz_z_r
                        yr = R[1, 0] * xyz_x_r + R[1, 1] * xyz_y_r + R[1, 2] * xyz_z_r
                        zr = R[2, 0] * xyz_x_r + R[2, 1] * xyz_y_r + R[2, 2] * xyz_z_r
                        t_offset = _rs_t_offset_identity_r
                        yaw_a   = yaw_coeff * t_offset
                        pitch_a = pitch_coeff * t_offset
                        roll_a  = roll_coeff * t_offset
                        xn = xr + yaw_a * zr - roll_a * yr
                        yn = yr + roll_a * xr - pitch_a * zr
                        zn = zr - yaw_a * xr + pitch_a * yr
                        if has_Rc:
                            Rc = R_cross.astype(np.float32)
                            xc = Rc[0, 0] * xn + Rc[0, 1] * yn + Rc[0, 2] * zn
                            yc = Rc[1, 0] * xn + Rc[1, 1] * yn + Rc[1, 2] * zn
                            zc = Rc[2, 0] * xn + Rc[2, 1] * yn + Rc[2, 2] * zn
                            xn, yn, zn = xc, yc, zc
                        mx, my = _dirs_to_cross_maps(xn, yn, zn, _cmx_r, _cmy_r)

                    result_buf[:, half_w:] = cv2.remap(crossA, mx, my,
                                                       _export_interp, borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=(0, 0, 0))

                def _process_left_eye_cross_rs(crossB, R_sensor, angular_vel, R_cross=None):
                    """RS + rotation for left eye (no-firmware-RS mode only).
                    Same as _process_right_eye_cross_rs but uses left-eye grid/buffers."""
                    R = R_sensor.astype(np.float32)
                    yaw_coeff   = np.float32(angular_vel[2] * rs_yaw_factor) * _rs_deg2rad
                    pitch_coeff = np.float32(angular_vel[1] * rs_pitch_factor) * _rs_deg2rad
                    roll_coeff  = np.float32(angular_vel[0] * rs_roll_factor) * _rs_deg2rad
                    has_Rc = R_cross is not None and np.max(np.abs(R_cross - np.eye(3, dtype=np.float32))) > 1e-6
                    if HAS_NUMBA:
                        Rc = R_cross.astype(np.float32) if has_Rc else np.eye(3, dtype=np.float32)
                        _nb_cross_remap_rs(
                            xyz_x, xyz_y, xyz_z,
                            R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2],
                            _rs_t_offset_identity,
                            yaw_coeff, pitch_coeff, roll_coeff,
                            has_Rc,
                            Rc[0,0], Rc[0,1], Rc[0,2], Rc[1,0], Rc[1,1], Rc[1,2], Rc[2,0], Rc[2,1], Rc[2,2],
                            _cmx_l, _cmy_l, xyz_x.shape[0])
                        mx = _cmx_l.reshape(out_height, half_w)
                        my = _cmy_l.reshape(out_height, half_w)
                    else:
                        xr = R[0,0]*xyz_x + R[0,1]*xyz_y + R[0,2]*xyz_z
                        yr = R[1,0]*xyz_x + R[1,1]*xyz_y + R[1,2]*xyz_z
                        zr = R[2,0]*xyz_x + R[2,1]*xyz_y + R[2,2]*xyz_z
                        t_offset = _rs_t_offset_identity
                        xn = xr + yaw_coeff*t_offset*zr - roll_coeff*t_offset*yr
                        yn = yr + roll_coeff*t_offset*xr - pitch_coeff*t_offset*zr
                        zn = zr - yaw_coeff*t_offset*xr + pitch_coeff*t_offset*yr
                        if has_Rc:
                            Rc = R_cross.astype(np.float32)
                            xc = Rc[0,0]*xn + Rc[0,1]*yn + Rc[0,2]*zn
                            yc = Rc[1,0]*xn + Rc[1,1]*yn + Rc[1,2]*zn
                            zc = Rc[2,0]*xn + Rc[2,1]*yn + Rc[2,2]*zn
                            xn, yn, zn = xc, yc, zc
                        mx, my = _dirs_to_cross_maps(xn, yn, zn, _cmx_l, _cmy_l)
                    result_buf[:, :half_w] = cv2.remap(crossB, mx, my,
                                                       _export_interp, borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=(0, 0, 0))

            # Precompute fisheye-aware RS time-offset map for .mov path only
            rs_time_map = build_rs_time_map(out_height, half_w, geoc_klns,
                                            ctrx=geoc_ctrx, ctry=geoc_ctry,
                                            cal_dim=geoc_cal_dim) if (rs_enabled and geoc_klns and not cfg.is_360_input) else None

            # Preallocate equirect remap scratch buffers for non-360 path (avoid per-frame alloc)
            _cuda_t_offset_eq = None  # CUDA device array for non-360 RS time offset
            _eq_src_buf_A = None  # Pre-allocated contiguous source buffer (right eye = left_src)
            _eq_src_buf_B = None  # Pre-allocated contiguous source buffer (left eye = right_src)
            _eq_src_flat_A = None  # Flat view for GPU upload (avoids per-frame ravel)
            _eq_src_flat_B = None
            if not cfg.is_360_input:
                _n_eq = xyz_x.shape[0]
                _emx_r = np.empty(_n_eq, dtype=np.float32)
                _emy_r = np.empty(_n_eq, dtype=np.float32)
                _emx_l = np.empty(_n_eq, dtype=np.float32)
                _emy_l = np.empty(_n_eq, dtype=np.float32)
                # Pre-allocate contiguous source buffers for GPU upload (avoids np.ascontiguousarray per frame)
                _eq_src_buf_A = np.empty((out_height, half_w, 3), dtype=np.uint16)
                _eq_src_buf_B = np.empty((out_height, half_w, 3), dtype=np.uint16)
                _eq_src_flat_A = _eq_src_buf_A.ravel()  # flat view (zero copy, stays valid)
                _eq_src_flat_B = _eq_src_buf_B.ravel()
                # Pre-scale RS time map once (constant across all frames)
                if rs_time_map is not None:
                    _rs_t_eq = (rs_time_map * np.float32(rs_ms / 1000.0)).ravel().astype(np.float32)
                    # Upload to GPU for CUDA equirect path
                    if HAS_NUMBA_CUDA:
                        _cuda_t_offset_eq = _cuda.to_device(_rs_t_eq)
                else:
                    _rs_t_eq = None
                _rs_deg2rad_eq = np.float32(np.pi / 180.0)

            _no_fw_rs = cfg.no_firmware_rs  # Apply RS to both eyes (no firmware RS)

            def process_frame_opencv(frame, gyro_left, gyro_right, angular_vel=None,
                                     rs_R_sensor_right=None, rs_R_cross_right=None,
                                     pre_split_s0=None, pre_split_s4=None,
                                     pre_split_eyes=None, pre_crosses=None):
                """Process a single SBS frame with parallel remap per eye.

                gyro_left/gyro_right: IORI_cancel × heading × global × stereo (for non-RS path).
                rs_R_sensor_right: heading × global × stereo (for RS path, no IORI).
                rs_R_cross_right: IORI (maps sensor→cross, applied after RS).
                pre_split_s0/s4: for parallel .360 decode, pre-split stream arrays (avoids vstack copy).

                When no_firmware_rs=True, RS is applied to BOTH eyes using the same
                angular velocity and factors (1,1,1). The left eye uses gyro_left as
                its RS sensor matrix (since IORI=0, heading = full correction).
                """
                R_right_eye = gyro_right
                R_left_eye = gyro_left
                _cuda_fused_done = False  # Set True when CUDA fused pipeline handles all post-processing

                if cfg.is_360_input:
                    if pre_crosses is not None:
                        # Crosses pre-assembled in decode thread (overlapped with GPU)
                        crossA, crossB = pre_crosses
                    else:
                        # Single-pass cross remap: full 184.5° FOV access for all .360 paths
                        if pre_split_s0 is not None:
                            s0 = pre_split_s0
                            s4 = pre_split_s4
                        else:
                            s0 = frame[:1920, :]
                            s4 = frame[1920:, :]
                        # Assemble both crosses in parallel using double-buffered arrays
                        _cbi = _cross_buf_idx[0]
                        _cb_A = _cross_bufs_A[_cbi]
                        _cb_B = _cross_bufs_B[_cbi]
                        _cross_buf_idx[0] = (_cbi + 1) % _N_CROSS_BUFS
                        _t_crossB = threading.Thread(target=FrameExtractor._assemble_lensB,
                                                     args=(s0, s4, _cb_B))
                        _t_crossB.start()
                        crossA = FrameExtractor._assemble_lensA(s0, s4, _cb_A)
                        _t_crossB.join()
                        crossB = _cb_B

                    if HAS_MLX:
                        # MLX Metal GPU: fused rotation+RS+IORI+cross+bilinear in one kernel
                        if rs_enabled and angular_vel is not None and rs_R_sensor_right is not None:
                            # Right eye: RS path
                            has_Rc = rs_R_cross_right is not None and np.max(np.abs(rs_R_cross_right - np.eye(3, dtype=np.float32))) > 1e-6
                            roll_rate  = np.float32(angular_vel[0])
                            pitch_rate = np.float32(angular_vel[1])
                            yaw_rate   = np.float32(angular_vel[2])

                            # 3D rotation RS: rate × factor, applied via t_offset
                            rs_c = np.array([
                                np.float32(yaw_rate * rs_yaw_factor) * _rs_deg2rad,
                                np.float32(pitch_rate * rs_pitch_factor) * _rs_deg2rad,
                                np.float32(roll_rate * rs_roll_factor) * _rs_deg2rad
                            ], dtype=np.float32)
                            right_out = _mlx_process_eye(
                                crossA, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                                rs_R_sensor_right, _mlx_t_offset, rs_c,
                                rs_R_cross_right if has_Rc else None, has_Rc, _mlx_n_pixels,
                                pix_max=65535.0)
                        else:
                            # Right eye: no RS
                            right_out = _mlx_process_eye(
                                crossA, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                                R_right_eye, _mlx_zero_t_offset, None,
                                None, False, _mlx_n_pixels, pix_max=65535.0)
                        result_buf[:, half_w:] = right_out.reshape(out_height, half_w, 3)
                        # Left eye: RS only when no_firmware_rs (both eyes need correction)
                        if _no_fw_rs and rs_enabled and angular_vel is not None:
                            left_out = _mlx_process_eye(
                                crossB, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                                R_left_eye, _mlx_t_offset, rs_c,
                                None, False, _mlx_n_pixels, pix_max=65535.0)
                        else:
                            left_out = _mlx_process_eye(
                                crossB, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                                R_left_eye, _mlx_zero_t_offset, None,
                                None, False, _mlx_n_pixels, pix_max=65535.0)
                        result_buf[:, :half_w] = left_out.reshape(out_height, half_w, 3)
                    elif HAS_NUMBA_CUDA:
                        # Numba CUDA: fused GPU pipeline — remap + LUT + sharpen in one pass
                        if rs_enabled and angular_vel is not None and rs_R_sensor_right is not None:
                            has_Rc = rs_R_cross_right is not None and np.max(np.abs(rs_R_cross_right - np.eye(3, dtype=np.float32))) > 1e-6
                            rs_c = np.array([
                                np.float32(angular_vel[2] * rs_yaw_factor) * _rs_deg2rad,
                                np.float32(angular_vel[1] * rs_pitch_factor) * _rs_deg2rad,
                                np.float32(angular_vel[0] * rs_roll_factor) * _rs_deg2rad
                            ], dtype=np.float32)
                            _d_t_left = _cuda_t_offset if _no_fw_rs else _cuda_zero_t_offset
                            _cuda_fused_process(
                                crossA, crossB, _cuda_xyz_x, _cuda_xyz_y, _cuda_xyz_z,
                                rs_R_sensor_right, R_left_eye,
                                _cuda_t_offset, _d_t_left,
                                rs_c, rs_R_cross_right if has_Rc else None, has_Rc,
                                _cuda_n_pixels, out_height, half_w, result_buf,
                                color_1d_lut, lut_3d, cfg.lut_intensity,
                                _fused_sharpen_kernel, _fused_sharpen_lat,
                                _sharpen_amount if _sharpen_enabled else 0.0, _fused_sharpen_ksize,
                                pix_dtype=np.uint16)
                        else:
                            _cuda_fused_process(
                                crossA, crossB, _cuda_xyz_x, _cuda_xyz_y, _cuda_xyz_z,
                                R_right_eye, R_left_eye,
                                _cuda_zero_t_offset, _cuda_zero_t_offset,
                                None, None, False,
                                _cuda_n_pixels, out_height, half_w, result_buf,
                                color_1d_lut, lut_3d, cfg.lut_intensity,
                                _fused_sharpen_kernel, _fused_sharpen_lat,
                                _sharpen_amount if _sharpen_enabled else 0.0, _fused_sharpen_ksize,
                                pix_dtype=np.uint16)
                        _cuda_fused_done = True
                    elif HAS_WGPU and _wgpu_out_buf is not None:
                        # wgpu cross-platform GPU: fused rotation+RS+IORI+cross+bilinear
                        if rs_enabled and angular_vel is not None and rs_R_sensor_right is not None:
                            has_Rc = rs_R_cross_right is not None and np.max(np.abs(rs_R_cross_right - np.eye(3, dtype=np.float32))) > 1e-6
                            rs_c = np.array([
                                np.float32(angular_vel[2] * rs_yaw_factor) * _rs_deg2rad,
                                np.float32(angular_vel[1] * rs_pitch_factor) * _rs_deg2rad,
                                np.float32(angular_vel[0] * rs_roll_factor) * _rs_deg2rad
                            ], dtype=np.float32)
                            right_out = _wgpu_process_eye(
                                crossA, _wgpu_xyz_x_buf, _wgpu_xyz_y_buf, _wgpu_xyz_z_buf,
                                rs_R_sensor_right, _wgpu_t_offset_buf, rs_c,
                                rs_R_cross_right if has_Rc else None, has_Rc,
                                _wgpu_n_pixels_render, _wgpu_out_buf)
                        else:
                            right_out = _wgpu_process_eye(
                                crossA, _wgpu_xyz_x_buf, _wgpu_xyz_y_buf, _wgpu_xyz_z_buf,
                                R_right_eye, _wgpu_t_offset_buf, None,
                                None, False, _wgpu_n_pixels_render, _wgpu_out_buf)
                        result_buf[:, half_w:] = right_out.reshape(out_height, half_w, 3)
                        if _no_fw_rs and rs_enabled and angular_vel is not None:
                            left_out = _wgpu_process_eye(
                                crossB, _wgpu_xyz_x_buf, _wgpu_xyz_y_buf, _wgpu_xyz_z_buf,
                                R_left_eye, _wgpu_t_offset_buf, rs_c,
                                None, False, _wgpu_n_pixels_render, _wgpu_out_buf)
                        else:
                            left_out = _wgpu_process_eye(
                                crossB, _wgpu_xyz_x_buf, _wgpu_xyz_y_buf, _wgpu_xyz_z_buf,
                                R_left_eye, _wgpu_t_offset_buf, None,
                                None, False, _wgpu_n_pixels_render, _wgpu_out_buf)
                        result_buf[:, :half_w] = left_out.reshape(out_height, half_w, 3)
                    elif HAS_NUMBA:
                        # Numba prange uses all cores internally — run eyes sequentially
                        # to avoid concurrent prange crash (workqueue layer not thread-safe)
                        if rs_enabled and angular_vel is not None and rs_R_sensor_right is not None and geoc_klns is not None:
                            _process_right_eye_cross_rs(crossA, rs_R_sensor_right, angular_vel, rs_R_cross_right)
                            if _no_fw_rs:
                                _process_left_eye_cross_rs(crossB, R_left_eye, angular_vel, None)
                            else:
                                _process_left_eye_cross(crossB, R_left_eye)
                        else:
                            _process_right_eye_cross(crossA, R_right_eye)
                            _process_left_eye_cross(crossB, R_left_eye)
                    elif rs_enabled and angular_vel is not None and rs_R_sensor_right is not None and geoc_klns is not None:
                        t_right = threading.Thread(
                            target=_process_right_eye_cross_rs,
                            args=(crossA, rs_R_sensor_right, angular_vel, rs_R_cross_right))
                        t_right.start()
                        if _no_fw_rs:
                            _process_left_eye_cross_rs(crossB, R_left_eye, angular_vel)
                        else:
                            _process_left_eye_cross(crossB, R_left_eye)
                        t_right.join()
                    else:
                        # No RS: single-pass cross remap with combined rotation
                        t_right = threading.Thread(target=_process_right_eye_cross,
                                                   args=(crossA, R_right_eye))
                        t_right.start()
                        _process_left_eye_cross(crossB, R_left_eye)
                        t_right.join()

                else:
                    # .mov path: split frame into per-eye halves with shift baked in
                    if pre_split_eyes is not None:
                        # Eyes already split in decode thread (contiguous, zero copy here)
                        left_src, right_src = pre_split_eyes
                    elif _eq_src_buf_A is not None and frame is not None and frame.shape[0] == out_height:
                        # Copy into pre-allocated contiguous buffers (avoids per-frame np.ascontiguousarray)
                        np.copyto(_eq_src_buf_A, frame[:, left_start:left_end])
                        left_src = _eq_src_buf_A
                        if right_wraps:
                            _rw = width - right_start
                            _eq_src_buf_B[:, :_rw] = frame[:, right_start:]
                            _eq_src_buf_B[:, _rw:] = frame[:, :right_end - width]
                        else:
                            np.copyto(_eq_src_buf_B, frame[:, right_start:right_end])
                        right_src = _eq_src_buf_B
                    else:
                        left_src = frame[:, left_start:left_end]
                        if right_wraps:
                            right_src = np.concatenate([frame[:, right_start:], frame[:, :right_end - width]], axis=1)
                        else:
                            right_src = frame[:, right_start:right_end]

                    # RS correction: fused into remap in 3D space (matches .360 kernel)
                    # right eye = left_src (after shift swap), left eye = right_src
                    _has_rs = (rs_enabled and angular_vel is not None and rs_time_map is not None)

                    # Fast path: skip expensive remap when rotation is identity and no RS
                    _is_identity_R = (np.max(np.abs(R_right_eye - _identity3)) < 1e-6 and
                                      np.max(np.abs(R_left_eye - _identity3)) < 1e-6)
                    _needs_gpu_post = (color_1d_lut is not None or lut_3d is not None or
                                       (_sharpen_enabled and _sharpen_amount > 0.01))
                    if HAS_MLX and not HAS_NUMBA_CUDA and (not _is_identity_R or _needs_gpu_post):
                        # ── MLX Metal GPU: equirect remap per eye ──
                        right_out = _mlx_process_eye_equirect(
                            left_src, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                            R_right_eye.astype(np.float32), _mlx_n_pixels, pix_max=65535.0)
                        result_buf[:, half_w:] = right_out.reshape(out_height, half_w, 3)
                        left_out = _mlx_process_eye_equirect(
                            right_src, _mlx_xyz_x, _mlx_xyz_y, _mlx_xyz_z,
                            R_left_eye.astype(np.float32), _mlx_n_pixels, pix_max=65535.0)
                        result_buf[:, :half_w] = left_out.reshape(out_height, half_w, 3)
                    elif HAS_NUMBA_CUDA and (_has_rs or not _is_identity_R or _needs_gpu_post):
                        # ── CUDA GPU fused pipeline: remap + LUT + sharpen in one pass ──
                        if _has_rs:
                            rs_c = np.array([
                                np.float32(angular_vel[2] * rs_yaw_factor) * _rs_deg2rad_eq,
                                np.float32(angular_vel[1] * rs_pitch_factor) * _rs_deg2rad_eq,
                                np.float32(angular_vel[0] * rs_roll_factor) * _rs_deg2rad_eq
                            ], dtype=np.float32)
                            _cuda_fused_process_eq(
                                left_src, right_src, _cuda_xyz_x, _cuda_xyz_y, _cuda_xyz_z,
                                R_right_eye, R_left_eye,
                                _cuda_t_offset_eq,
                                rs_c,
                                _cuda_n_pixels, out_height, half_w, result_buf,
                                color_1d_lut, lut_3d, cfg.lut_intensity,
                                _fused_sharpen_kernel, _fused_sharpen_lat,
                                _sharpen_amount if _sharpen_enabled else 0.0, _fused_sharpen_ksize,
                                pix_dtype=np.uint16)
                        else:
                            _cuda_fused_process_eq(
                                left_src, right_src, _cuda_xyz_x, _cuda_xyz_y, _cuda_xyz_z,
                                R_right_eye, R_left_eye,
                                None,
                                None,
                                _cuda_n_pixels, out_height, half_w, result_buf,
                                color_1d_lut, lut_3d, cfg.lut_intensity,
                                _fused_sharpen_kernel, _fused_sharpen_lat,
                                _sharpen_amount if _sharpen_enabled else 0.0, _fused_sharpen_ksize,
                                pix_dtype=np.uint16)
                        _cuda_fused_done = True
                    elif _is_identity_R and not _has_rs:
                        # CPU fast path: no remap needed when rotation is identity and no RS
                        if left_src.shape == (out_height, half_w, 3):
                            result_buf[:, half_w:] = left_src
                            result_buf[:, :half_w] = right_src
                        else:
                            result_buf[:, half_w:] = cv2.resize(left_src, (half_w, out_height), interpolation=cv2.INTER_AREA)
                            result_buf[:, :half_w] = cv2.resize(right_src, (half_w, out_height), interpolation=cv2.INTER_AREA)
                    else:
                        # ── CPU: Threaded parallel remap for both eyes ──
                        def _remap_right_eye_eq():
                            """Build remap + cv2.remap for right eye (with RS if active)."""
                            if _has_rs:
                                _yaw_c = np.float32(angular_vel[2] * rs_yaw_factor) * _rs_deg2rad_eq
                                _pitch_c = np.float32(angular_vel[1] * rs_pitch_factor) * _rs_deg2rad_eq
                                _roll_c = np.float32(angular_vel[0] * rs_roll_factor) * _rs_deg2rad_eq
                                R_r = R_right_eye.astype(np.float32)
                                _use_klns_eq = geoc_klns is not None
                                _kl_eq = geoc_klns if _use_klns_eq else [0]*5
                                _cx_eq = np.float32(geoc_cal_dim / 2.0 + geoc_ctrx) if _use_klns_eq else np.float32(0)
                                _cy_eq = np.float32(geoc_cal_dim / 2.0 + geoc_ctry) if _use_klns_eq else np.float32(0)
                                _cd_eq = np.float32(geoc_cal_dim) if _use_klns_eq else np.float32(0)
                                _srot_eq = np.float32(rs_ms / 1000.0)
                                if HAS_NUMBA:
                                    _nb_equirect_remap_rs(
                                        xyz_x, xyz_y, xyz_z,
                                        R_r[0,0], R_r[0,1], R_r[0,2],
                                        R_r[1,0], R_r[1,1], R_r[1,2],
                                        R_r[2,0], R_r[2,1], R_r[2,2],
                                        _rs_t_eq,
                                        _yaw_c, _pitch_c, _roll_c,
                                        _emx_r, _emy_r, _n_eq,
                                        half_w_f, height_f, inv_pi,
                                        np.float32(_kl_eq[0]), np.float32(_kl_eq[1]),
                                        np.float32(_kl_eq[2]), np.float32(_kl_eq[3]),
                                        np.float32(_kl_eq[4]),
                                        _cx_eq, _cy_eq, _cd_eq, _srot_eq,
                                        False)  # 3D rotation (proven best)
                                    r_mx = _emx_r.reshape(out_height, half_w)
                                    r_my = _emy_r.reshape(out_height, half_w)
                                else:
                                    # Numpy KLNS dr/dθ fallback
                                    xr = R_r[0,0]*xyz_x + R_r[0,1]*xyz_y + R_r[0,2]*xyz_z
                                    yr = R_r[1,0]*xyz_x + R_r[1,1]*xyz_y + R_r[1,2]*xyz_z
                                    zr = R_r[2,0]*xyz_x + R_r[2,1]*xyz_y + R_r[2,2]*xyz_z
                                    if _use_klns_eq:
                                        _c = geoc_klns
                                        _zc = np.clip(zr,-1,1).astype(np.float32)
                                        _sth = np.sqrt(np.maximum(np.float32(1)-_zc*_zc,np.float32(0)))
                                        _theta = np.arccos(_zc); _th2 = _theta*_theta
                                        _rf = _theta*(np.float32(_c[0])+_th2*(np.float32(_c[1])+_th2*(np.float32(_c[2])+_th2*(np.float32(_c[3])+_th2*np.float32(_c[4])))))
                                        _ssafe = np.where(_sth<1e-7,np.float32(1),_sth)
                                        _phx = np.where(_sth<1e-7,np.float32(0),xr/_ssafe)
                                        _phy = np.where(_sth<1e-7,np.float32(0),yr/_ssafe)
                                        _sx = _cx_eq+_rf*_phx; _sy = _cy_eq-_rf*_phy
                                        _drdth = np.float32(_c[0])+_th2*(3*np.float32(_c[1])+_th2*(5*np.float32(_c[2])+_th2*(7*np.float32(_c[3])+9*np.float32(_c[4])*_th2)))
                                        _t1 = (_sy-_cy_eq)/_cd_eq*_srot_eq
                                        _dsx = _yaw_c*_t1*_drdth; _dsy = _pitch_c*_t1*_drdth
                                        _dsx += _roll_c*_t1*(_sy-_cy_eq); _dsy += -_roll_c*_t1*(_sx-_cx_eq)
                                        _sxn=_sx+_dsx;_syn=_sy+_dsy
                                        _dx=_sxn-_cx_eq;_dy=-(_syn-_cy_eq)
                                        _rnew=np.sqrt(_dx*_dx+_dy*_dy).astype(np.float32);_thn=_theta.copy()
                                        for _ in range(2):
                                            _t2n=_thn*_thn;_rest=_thn*(np.float32(_c[0])+_t2n*(np.float32(_c[1])+_t2n*(np.float32(_c[2])+_t2n*(np.float32(_c[3])+_t2n*np.float32(_c[4])))))
                                            _drdt=np.float32(_c[0])+_t2n*(3*np.float32(_c[1])+_t2n*(5*np.float32(_c[2])+_t2n*(7*np.float32(_c[3])+9*np.float32(_c[4])*_t2n)))
                                            _ok=np.abs(_drdt)>1e-6;_thn=np.where(_ok,_thn+(_rnew-_rest)/np.where(_ok,_drdt,np.float32(1)),_thn)
                                            _thn=np.clip(_thn,0,3.11)
                                        _sr=np.where(_rnew<1e-6,np.float32(1),_rnew)
                                        xn=(np.sin(_thn)*np.where(_rnew<1e-6,0,_dx/_sr)).astype(np.float32)
                                        yn=(np.sin(_thn)*np.where(_rnew<1e-6,0,_dy/_sr)).astype(np.float32)
                                        zn=np.cos(_thn).astype(np.float32)
                                    else:
                                        xn = xr + _yaw_c * _rs_t_eq * zr - _roll_c * _rs_t_eq * yr
                                        yn = yr + _roll_c * _rs_t_eq * xr - _pitch_c * _rs_t_eq * zr
                                        zn = zr - _yaw_c * _rs_t_eq * xr + _pitch_c * _rs_t_eq * yr
                                    lat_n = np.arcsin(np.clip(yn, -1, 1))
                                    lon_n = np.arctan2(xn, zn)
                                    r_mx = np.clip((lon_n * inv_pi + 0.5) * half_w_f, 0, half_w_f - 1).reshape(out_height, half_w).astype(np.float32)
                                    r_my = np.clip((0.5 - lat_n * inv_pi) * height_f, 0, height_f - 1).reshape(out_height, half_w).astype(np.float32)
                            else:
                                R_r = R_right_eye.astype(np.float32)
                                if HAS_NUMBA:
                                    _nb_equirect_remap_rot(
                                        xyz_x, xyz_y, xyz_z,
                                        R_r[0,0], R_r[0,1], R_r[0,2],
                                        R_r[1,0], R_r[1,1], R_r[1,2],
                                        R_r[2,0], R_r[2,1], R_r[2,2],
                                        _emx_r, _emy_r, _n_eq,
                                        half_w_f, height_f, inv_pi)
                                    r_mx = _emx_r.reshape(out_height, half_w)
                                    r_my = _emy_r.reshape(out_height, half_w)
                                else:
                                    r_mx, r_my = compute_remap_tables(R_right_eye)
                                    r_mx = r_mx.astype(np.float32)
                                    r_my = r_my.astype(np.float32)
                            result_buf[:, half_w:] = cv2.remap(left_src, r_mx, r_my,
                                                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                                borderValue=(0, 0, 0))

                        def _remap_left_eye_eq():
                            """Build remap + cv2.remap for left eye (rotation only, no RS)."""
                            R_l = R_left_eye.astype(np.float32)
                            if HAS_NUMBA:
                                _nb_equirect_remap_rot(
                                    xyz_x, xyz_y, xyz_z,
                                    R_l[0,0], R_l[0,1], R_l[0,2],
                                    R_l[1,0], R_l[1,1], R_l[1,2],
                                    R_l[2,0], R_l[2,1], R_l[2,2],
                                    _emx_l, _emy_l, _n_eq,
                                    half_w_f, height_f, inv_pi)
                                l_mx = _emx_l.reshape(out_height, half_w)
                                l_my = _emy_l.reshape(out_height, half_w)
                            else:
                                l_mx, l_my = compute_remap_tables(R_left_eye)
                                l_mx = l_mx.astype(np.float32)
                                l_my = l_my.astype(np.float32)
                            result_buf[:, :half_w] = cv2.remap(right_src, l_mx, l_my,
                                                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                                borderValue=(0, 0, 0))

                        if HAS_NUMBA:
                            # Numba prange uses all cores — run sequentially to avoid
                            # concurrent workqueue access crash
                            _remap_right_eye_eq()
                            _remap_left_eye_eq()
                        else:
                            # Pure numpy/cv2: run in parallel (both release GIL)
                            t_right = threading.Thread(target=_remap_right_eye_eq)
                            t_right.start()
                            _remap_left_eye_eq()
                            t_right.join()

                # Pipeline order (unified across all backends):
                #   remap → 1D LUT → 3D LUT → sharpen → saturation → edge mask
                # Saturation is applied LAST (post-LUT, post-sharpen) so the
                # 3D LUT operates on the original color-graded pixels and the
                # saturation adjustment only stretches/compresses the final
                # graded chroma without feeding back into the LUT lookup.
                if _cuda_fused_done:
                    # CUDA fused pipeline already ran: remap → 1D LUT → 3D LUT → sharpen
                    result = result_buf
                else:
                    # Non-fused CPU / MLX / wgpu path
                    result = apply_color_1d(result_buf, color_1d_lut)
                    result = apply_lut_3d_fast(result, lut_3d, cfg.lut_intensity)
                    result = apply_equirect_sharpen(result, _sharpen_bands)

                # Apply saturation (post-LUT, post-sharpen — cross-channel, can't
                # be done via 1D LUT). Same formula in both branches.
                if _has_saturation:
                    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cv2.addWeighted(gray_3ch, 1.0 - _export_saturation, result, _export_saturation, 0, dst=result)

                # Apply circular edge mask per eye (after color/LUT)
                if _eye_mask is not None:
                    if result.dtype == np.uint16:
                        # uint16: use uint32 intermediate to avoid overflow
                        result[:, half_w:] = (result[:, half_w:].astype(np.uint32) * _eye_mask >> 8).astype(np.uint16)
                        result[:, :half_w] = (result[:, :half_w].astype(np.uint32) * _eye_mask >> 8).astype(np.uint16)
                    else:
                        # Fast integer multiply: uint8→uint16 * uint16(0-256) >> 8 → uint8
                        result[:, half_w:] = (result[:, half_w:].astype(np.uint16) * _eye_mask >> 8).astype(np.uint8)
                        result[:, :half_w] = (result[:, :half_w].astype(np.uint16) * _eye_mask >> 8).astype(np.uint8)
                    if not result.flags['C_CONTIGUOUS']:
                        result = np.ascontiguousarray(result)

                # Upside-down mount: rotate 180° (flips image + swaps L/R eyes)
                if cfg.upside_down:
                    result = cv2.rotate(result, cv2.ROTATE_180)

                return result

            # (Global+stereo view adjustment computed above as R_view_left/R_view_right)

            # Precompute 1D color LUT (256 bytes, applied via cv2.LUT = near instant per frame)
            has_color = abs(cfg.lift) > 0.01 or abs(cfg.gamma - 1.0) > 0.01 or abs(cfg.gain - 1.0) > 0.01
            if has_color:
                self.status.emit("Building color adjustment lookup table...")
                color_1d_lut = build_color_1d_lut(cfg.lift, cfg.gamma, cfg.gain, bit_depth=16)
            else:
                color_1d_lut = None
            _export_saturation = cfg.saturation
            _has_saturation = abs(_export_saturation - 1.0) > 0.01

            # Set up FFmpeg decode pipeline
            # _disable_hwaccel persists across retries via class attribute
            if not hasattr(self, '_disable_hwaccel'):
                self._disable_hwaccel = False
            _disable_hwaccel = [self._disable_hwaccel]
            # Build list of decode commands (one per segment for multi-segment, or single)
            def _build_decode_cmd(input_path, ss=0, t=None, stream_index=None, crop_rect=None):
                """Build FFmpeg decode command for a single input file.
                If stream_index is given, decode only that stream (for parallel .360 decode).
                If crop_rect is given as (x, y, w, h), apply crop filter (for parallel non-360 decode).
                """
                cmd = [get_ffmpeg_path()]
                # Hardware decode: NVDEC on Windows (NVIDIA), VideoToolbox on macOS
                # Can be disabled at runtime if hwaccel causes decode failures.
                if sys.platform == 'win32' and _has_nvdec and not _disable_hwaccel[0]:
                    cmd += ["-hwaccel", "cuda"]
                elif sys.platform == 'darwin':
                    cmd += ["-hwaccel", "videotoolbox"]
                if ss > 0:
                    cmd += ["-ss", str(ss)]
                if t is not None:
                    cmd += ["-t", str(t)]
                cmd += ["-threads", str(min(os.cpu_count() or 4, 8)),
                        "-i", str(input_path)]
                if stream_index is not None:
                    # Parallel .360: decode a single stream directly (no vstack filter)
                    cmd += ["-map", f"0:{stream_index}"]
                elif cfg.is_360_input:
                    eac_raw_filter = FrameExtractor.EAC_RAW_FILTER + "[out]"
                    cmd += ["-filter_complex", eac_raw_filter, "-map", "[out]"]
                elif crop_rect is not None:
                    # Parallel non-360: crop one eye from SBS frame
                    cx, cy, cw, ch = crop_rect
                    cmd += ["-vf", f"crop={cw}:{ch}:{cx}:{cy}"]
                cmd += ["-f", "rawvideo", "-pix_fmt", "bgr48le", "-v", "quiet", "-"]
                return cmd

            # PyAV decode generator (2x faster than FFmpeg subprocess by eliminating pipe overhead).
            # On macOS this opens the container with videotoolbox hwaccel when available —
            # crucial for cold-start trim where hundreds/thousands of frames may be decoded
            # and discarded before the first yielded frame. Benchmarks on Apple Silicon:
            #   skip cost:  sw ≈ 14.7 ms/frame, hw ≈ 6.4 ms/frame (~2.3× faster)
            #   yield cost: sw ≈ 40.3 ms/frame, hw ≈ 46.9 ms/frame (hw +17% due to p010le
            #               reformat being slower than yuv420p10le through swscale, but
            #               still fast enough that downstream GPU remap is the real
            #               bottleneck in the full pipeline).
            # Net effect: deep trims finish well under the dead-man timeout; shallow trims
            # and no-trim exports have unchanged or slightly slower decode which is
            # absorbed by the triple-buffered pipeline.
            _pyav_hwaccel_state = [None]  # [None / 'videotoolbox' / 'sw_fallback']
            def _pyav_decode_stream(input_path, stream_index=0, skip_frames=0, max_frames=None, crop_rect=None):
                """Generator yielding uint16 BGR numpy frames via PyAV.
                stream_index: absolute stream index (0 or 4 for .360).
                skip_frames: number of frames to skip at start (for trim sync).
                crop_rect: optional (x, y, w, h) for numpy slicing."""
                container = None
                # Try videotoolbox hwaccel on macOS. allow_software_fallback=True
                # lets libavcodec handle per-frame hw→sw fallback if an individual
                # frame can't be hardware-decoded; we still catch setup errors
                # here so the whole stream can fall through to plain sw decode.
                if sys.platform == 'darwin':
                    try:
                        from av.codec.hwaccel import HWAccel
                        _hw = HWAccel(device_type='videotoolbox', allow_software_fallback=True)
                        container = av.open(str(input_path), hwaccel=_hw)
                        if _pyav_hwaccel_state[0] != 'videotoolbox':
                            _pyav_hwaccel_state[0] = 'videotoolbox'
                            print("[DECODE] PyAV videotoolbox hwaccel enabled")
                    except Exception as _hw_err:
                        if _pyav_hwaccel_state[0] != 'sw_fallback':
                            _pyav_hwaccel_state[0] = 'sw_fallback'
                            print(f"⚠ PyAV videotoolbox hwaccel setup failed "
                                  f"({_hw_err}), using software decode")
                        container = None
                if container is None:
                    container = av.open(str(input_path))
                try:
                    # Map absolute stream index to PyAV video stream
                    target_stream = None
                    for vs in container.streams.video:
                        if vs.index == stream_index:
                            target_stream = vs
                            break
                    if target_stream is None:
                        # Fallback: use video stream position (0→first, 4→second for .360)
                        vid_idx = 0
                        for vs in container.streams.video:
                            if vs.index >= stream_index:
                                target_stream = vs
                                break
                            vid_idx += 1
                        if target_stream is None:
                            target_stream = container.streams.video[-1]
                    target_stream.thread_type = 'AUTO'
                    # For .360 with GPU remap (MLX/CUDA): use more decode threads since
                    # remap runs on GPU and doesn't compete for CPU.
                    # For non-360 CPU remap: limit decode threads to leave cores for remap.
                    _has_gpu_remap = HAS_MLX or HAS_NUMBA_CUDA or HAS_WGPU
                    if cfg.is_360_input or _has_gpu_remap:
                        target_stream.thread_count = min(os.cpu_count() or 4, 8)
                    else:
                        target_stream.thread_count = min(os.cpu_count() or 2, 3)
                    # Detect if source needs gbrp16le workaround. PyAV's direct
                    # bgr48le conversion has a wrong color matrix for:
                    #   • software yuv420p10le (original bug)
                    #   • hwaccel videotoolbox_vld → p010le (same swscale path)
                    # Measured deltas vs the gbrp16le path (on GS010172 frame 5):
                    #   sw yuv420p10le → bgr48le direct: mean |Δ|=13851, max=59368
                    #   hw p010le       → bgr48le direct: mean |Δ|=13540, max=65445
                    # Both are clearly wrong. We force gbrp16le whenever hwaccel
                    # is active, and keep the yuv*10* detection for the sw path.
                    # 8-bit ProRes etc. still use the fast direct bgr48le path.
                    _src_fmt = target_stream.codec_context.pix_fmt or ''
                    _is_hwaccel = bool(getattr(target_stream.codec_context, 'is_hwaccel', False))
                    _use_gbrp = _is_hwaccel or ('yuv' in _src_fmt and '10' in _src_fmt)
                    _skipped = 0
                    _yielded = 0
                    # Seek to nearest keyframe before the skip point instead of
                    # decoding and discarding thousands of frames sequentially.
                    # PyAV's seek goes to the keyframe AT or BEFORE the target
                    # timestamp, so we still need to decode+discard the few
                    # frames between the keyframe and the exact trim point.
                    if skip_frames > 30:
                        _seek_fps = float(target_stream.average_rate) if target_stream.average_rate else fps
                        _seek_ts = (skip_frames - 2) / _seek_fps  # 2-frame safety margin
                        container.seek(int(_seek_ts * 1_000_000), stream=target_stream)
                        # After seek, count decoded frames from the seek point
                        # to determine how many more to skip to reach exact trim
                        _pre_skip_count = 0
                        for _pre_frame in container.decode(target_stream):
                            if _pre_frame is None or _pre_frame.pts is None:
                                continue
                            _pre_time = float(_pre_frame.pts * _pre_frame.time_base)
                            _pre_frame_n = int(_pre_time * _seek_fps + 0.5)
                            if _pre_frame_n >= skip_frames:
                                # This is our first wanted frame — reformat and yield it
                                if _use_gbrp:
                                    _gbr = _pre_frame.reformat(format='gbrp16le').to_ndarray()
                                    arr = _gbr[:, :, ::-1].copy()
                                else:
                                    arr = _pre_frame.to_ndarray(format='bgr48le')
                                if crop_rect is not None:
                                    cx, cy, cw, ch = crop_rect
                                    arr = arr[cy:cy+ch, cx:cx+cw].copy()
                                yield arr
                                _yielded += 1
                                _skipped = skip_frames  # mark skip as done
                                break
                            _pre_skip_count += 1
                        if _skipped < skip_frames:
                            _skipped = skip_frames  # seek landed past skip point
                        print(f"[DECODE] Seeked to {_seek_ts:.2f}s, skipped {_pre_skip_count} post-seek frames")
                    for frame in container.decode(target_stream):
                        if self._cancelled:
                            break
                        if _skipped < skip_frames:
                            _skipped += 1
                            continue
                        if _use_gbrp:
                            # Workaround: gbrp16le returns (H,W,3) with channels [R,B,G]
                            # Reorder [2,1,0] → BGR to match FFmpeg's bgr48le output
                            _gbr = frame.reformat(format='gbrp16le').to_ndarray()
                            arr = _gbr[:, :, ::-1].copy()
                        else:
                            arr = frame.to_ndarray(format='bgr48le')
                        if crop_rect is not None:
                            cx, cy, cw, ch = crop_rect
                            arr = arr[cy:cy+ch, cx:cx+cw].copy()
                        yield arr
                        _yielded += 1
                        if max_frames is not None and _yielded >= max_frames:
                            break
                finally:
                    container.close()

            # For .360: use parallel decode (2 FFmpeg processes) to avoid slow vstack filter.
            # The vstack filter forces sequential decoding of both HEVC streams, ~4x slower
            # than decoding them in parallel (benchmarked: 126ms/frame vs 37ms/frame).
            use_parallel_360_decode = cfg.is_360_input
            # For non-360 SBS: use parallel decode with FFmpeg crop to split eyes at decode level.
            # Halves per-pipe bandwidth (50MB vs 100MB per frame at 8K) and gives contiguous eye frames.
            use_parallel_sbs_decode = (not cfg.is_360_input and not right_wraps
                                       and half_w > 0 and out_height > 0)

            # For parallel .360 decode with trim: we must NOT use FFmpeg -ss on
            # individual streams because s0 and s4 may seek to different frame
            # positions (GoPro .360 uses all P-frames, so timestamp-based seeking
            # can land on different frames for each HEVC track). Instead, decode
            # from the beginning and skip frames in the decode thread to ensure
            # perfect s0/s4 synchronization.
            #
            # Trim in multi-segment mode requires PER-SEGMENT skip + max counts:
            #   _seg_skip_frames[i] = frames to discard at the start of segment i
            #                         (only the segment containing trim_start has >0)
            #   _seg_max_frames[i]  = max frames to emit from segment i
            #                         (the segment containing trim_end may be clipped)
            # Both lists parallel decode_segment_list. Consumers use seg_idx to
            # look up their per-segment values. A previous version stored skip
            # as a single scalar — that scalar was overwritten inside the loop
            # by each subsequent segment, losing the first segment's skip count
            # and breaking trim for multi-segment recordings.
            decode_segment_list = []  # [(decode_cmd, seg_path, ss, t)]
            decode_segment_list_s4 = []  # parallel .360: second stream commands
            _seg_skip_frames = []     # per-segment "skip at decode start" counts
            _seg_max_frames = []      # per-segment max emit counts (None = no cap)

            if is_multi_segment:
                # Build per-segment decode commands
                # For trim: skip segments before start_time, truncate at end_time
                seg_time_offset = 0.0
                for seg_path, seg_dur in segment_durations:
                    seg_end = seg_time_offset + seg_dur
                    if seg_end <= start_time:
                        # Entire segment is before trim start — skip
                        seg_time_offset = seg_end
                        continue
                    if seg_time_offset >= end_time:
                        # Past trim end — skip
                        break
                    # Compute local seek and duration within this segment
                    local_ss = max(0.0, start_time - seg_time_offset)
                    local_end = min(seg_dur, end_time - seg_time_offset)
                    local_t = local_end - local_ss
                    _skip_n = int(round(local_ss * fps))
                    _max_n = int(round(local_t * fps))
                    if use_parallel_360_decode:
                        # No -ss for parallel .360: decode full segment, skip+cap in Python
                        cmd_s0 = _build_decode_cmd(seg_path, 0, None, stream_index=0)
                        cmd_s4 = _build_decode_cmd(seg_path, 0, None, stream_index=4)
                        decode_segment_list.append((cmd_s0, str(seg_path), 0, None))
                        decode_segment_list_s4.append((cmd_s4, str(seg_path), 0, None))
                        _seg_skip_frames.append(_skip_n)
                        _seg_max_frames.append(_max_n)
                    elif use_parallel_sbs_decode:
                        # Parallel non-360 SBS: two FFmpeg processes, each cropping one eye.
                        # ffmpeg -ss/-t already trims, no extra skip needed.
                        crop_A = (left_start, 0, half_w, height)  # left_src = right eye data
                        crop_B = (right_start, 0, half_w, height)  # right_src = left eye data
                        cmd_A = _build_decode_cmd(seg_path, local_ss, local_t, crop_rect=crop_A)
                        cmd_B = _build_decode_cmd(seg_path, local_ss, local_t, crop_rect=crop_B)
                        decode_segment_list.append((cmd_A, str(seg_path), local_ss, local_t))
                        decode_segment_list_s4.append((cmd_B, str(seg_path), local_ss, local_t))
                        _seg_skip_frames.append(0)
                        _seg_max_frames.append(_max_n)
                    else:
                        cmd = _build_decode_cmd(seg_path, local_ss, local_t)
                        decode_segment_list.append((cmd, str(seg_path), local_ss, local_t))
                        _seg_skip_frames.append(0)
                        _seg_max_frames.append(_max_n)
                    seg_time_offset = seg_end
            else:
                if use_parallel_360_decode:
                    # No -ss for parallel .360: decode from start, skip frames in Python
                    _single_skip = int(round(start_time * fps))
                    if _single_skip > 0:
                        print(f"[EXPORT] Parallel .360 decode: will skip {_single_skip} frames for trim sync")
                    cmd_s0 = _build_decode_cmd(cfg.input_path, 0, None, stream_index=0)
                    cmd_s4 = _build_decode_cmd(cfg.input_path, 0, None, stream_index=4)
                    decode_segment_list = [(cmd_s0, str(cfg.input_path), 0, None)]
                    decode_segment_list_s4 = [(cmd_s4, str(cfg.input_path), 0, None)]
                    _seg_skip_frames = [_single_skip]
                    _seg_max_frames = [total_frames + 2]
                elif use_parallel_sbs_decode:
                    # Parallel non-360 SBS: two FFmpeg processes, each cropping one eye
                    crop_A = (left_start, 0, half_w, height)
                    crop_B = (right_start, 0, half_w, height)
                    cmd_A = _build_decode_cmd(cfg.input_path, start_time, duration, crop_rect=crop_A)
                    cmd_B = _build_decode_cmd(cfg.input_path, start_time, duration, crop_rect=crop_B)
                    decode_segment_list = [(cmd_A, str(cfg.input_path), start_time, duration)]
                    decode_segment_list_s4 = [(cmd_B, str(cfg.input_path), start_time, duration)]
                    _seg_skip_frames = [0]
                    _seg_max_frames = [total_frames + 2]
                    print(f"[EXPORT] Parallel SBS decode: eye A crop=({left_start},0,{half_w},{height}), "
                          f"eye B crop=({right_start},0,{half_w},{height})")
                else:
                    decode_cmd = _build_decode_cmd(cfg.input_path, start_time, duration)
                    decode_segment_list = [(decode_cmd, str(cfg.input_path), start_time, duration)]
                    decode_segment_list_s4 = []
                    _seg_skip_frames = [0]
                    _seg_max_frames = [total_frames + 2]

            # Cap the last segment's max_frames so the TOTAL across all segments
            # never exceeds total_frames + 2 (the pipeline's safety margin).
            # For multi-segment with trim, the per-segment max is already local,
            # but sum may overshoot the global total by rounding — trim here.
            _cum = 0
            for _i, _m in enumerate(_seg_max_frames):
                _remaining = (total_frames + 2) - _cum
                if _remaining <= 0:
                    _seg_max_frames[_i] = 0
                elif _m is not None and _m > _remaining:
                    _seg_max_frames[_i] = _remaining
                _cum += _seg_max_frames[_i] if _seg_max_frames[_i] is not None else 0

            if is_multi_segment and sum(1 for s in _seg_skip_frames if s > 0) > 0:
                print(f"[EXPORT] Multi-segment trim: per-segment skip={_seg_skip_frames}, "
                      f"max={_seg_max_frames}")

            # Set up FFmpeg encode pipeline
            # Hardware encoders have max resolution limits (NVENC/AMF/QSV: 8192 per dimension).
            # Fall back to software encoding if the output exceeds these limits.
            _hw_max_dim = 8192
            _enc_type = cfg.encoder_type
            if _enc_type in ('auto', 'nvenc', 'amf', 'qsv') and (width > _hw_max_dim or out_height > _hw_max_dim):
                _enc_type = 'software'
                print(f"⚠ Output {width}×{out_height} exceeds hardware encoder limit ({_hw_max_dim}px) — using software encoder")

            # When VideoToolbox denoise is actually going to run, vt_denoise
            # truncates its intermediate buffer to 8-bit BGRA (because
            # VTTemporalNoiseFilter requires lossless-compressed 8-bit source
            # formats) and up-replicates back to 16-bit via ×257, yielding only
            # 256 distinct levels per channel. Encoding 10-bit output on top of
            # that just wastes bitrate, so override H.265 to 8-bit and warn
            # loudly for ProRes (which has no 8-bit profile).
            _denoise_will_run = False
            if sys.platform == 'darwin' and cfg.denoise_strength > 0.01:
                _vt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vt_denoise')
                if cfg.is_360_input:
                    # .360 uses pipe mode → requires vt_denoise binary
                    _denoise_will_run = os.path.isfile(_vt_path)
                else:
                    # non-360 uses file mode — same binary dependency
                    _denoise_will_run = os.path.isfile(_vt_path)

            _effective_h265_bit_depth = cfg.h265_bit_depth
            if _denoise_will_run and output_codec == "h265" and _effective_h265_bit_depth > 8:
                print(f"⚠ Denoise is active — overriding H.265 output to 8-bit. "
                      f"vt_denoise truncates the working buffer to 8 bits internally, "
                      f"so 10-bit encoding would only waste bitrate (256 distinct "
                      f"levels per channel instead of 1024).")
                self.status.emit("Denoise active → H.265 output forced to 8-bit")
                _effective_h265_bit_depth = 8
            elif _denoise_will_run and output_codec == "prores":
                print(f"⚠ Denoise is active with ProRes output. ProRes has no 8-bit "
                      f"profile, so the file will still be written in 10-bit yuv422p10le "
                      f"(or yuv444p10le for 4444/4444XQ), but the pixel data carries only "
                      f"~8 bits of real precision because vt_denoise truncates internally.")
                self.status.emit("Denoise active → ProRes still 10-bit, but only ~8-bit real precision")

            # Determine encoder settings using selected encoder type
            if output_codec == "h265":
                enc = get_hw_encoder_args('h265', _enc_type, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate, _effective_h265_bit_depth)
            elif output_codec == "prores":
                profile_map = {"proxy": "0", "lt": "1", "standard": "2", "hq": "3", "4444": "4", "4444xq": "5"}
                _prores_prof = profile_map.get(cfg.prores_profile, "3")
                # ProRes 4444/4444XQ use yuv444p10le, others use yuv422p10le
                _prores_pix = "yuv444p10le" if _prores_prof in ("4", "5") else "yuv422p10le"
                if cfg.encoder_type == 'videotoolbox' and sys.platform == 'darwin':
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", _prores_prof, "-pix_fmt", _prores_pix]
                else:
                    enc = ["-c:v", "prores_ks", "-profile:v", _prores_prof, "-vendor", "apl0", "-pix_fmt", _prores_pix]
            else:
                enc = get_hw_encoder_args('h264', _enc_type, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate)

            # Build audio source for encoder
            # For multi-segment: use ffmpeg concat demuxer to merge audio from all segments
            import tempfile
            audio_concat_file = None
            if is_multi_segment:
                # Create concat list file for audio from all segments
                audio_concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                                                delete=False, prefix='vr180_concat_')
                for seg_path, _ in segment_durations:
                    audio_concat_file.write(f"file '{seg_path}'\n")
                audio_concat_file.flush()
                audio_concat_file.close()
                audio_input_args = ["-f", "concat", "-safe", "0",
                                    "-ss", str(start_time), "-t", str(duration),
                                    "-i", audio_concat_file.name]
            else:
                audio_input_args = ["-ss", str(start_time), "-t", str(duration),
                                    "-i", str(cfg.input_path)]

            # Build audio mapping: stereo AAC (stream 1:1) always,
            # plus ambisonic PCM (stream 1:5) if requested for .360 input
            audio_args = ["-map", "1:1", "-c:a", "copy"]
            if cfg.is_360_input and cfg.audio_ambisonics:
                audio_args += ["-map", "1:5", "-c:a:1", "copy"]
            elif not cfg.is_360_input:
                # Non-.360: map all audio streams
                audio_args = ["-map", "1:a?", "-c:a", "copy"]

            # Only set yuv420p if encoder args don't already specify a pixel format
            # (e.g. 10-bit H.265 uses p010le, ProRes uses yuv422p10le)
            # When source is full-range (pc), use yuvj420p to preserve full range;
            # yuv420p would crush to limited range (16-235), losing contrast/saturation.
            if not any(a == "-pix_fmt" for a in enc):
                if _src_color_range == "pc":
                    pix_fmt_args = ["-pix_fmt", "yuvj420p"]
                else:
                    pix_fmt_args = ["-pix_fmt", "yuv420p"]
            else:
                pix_fmt_args = []
            # Preserve input color space tags in output.
            # Default to BT.709 when source tags are missing (GoPro .360 always
            # records BT.709 but some containers omit the tags).
            _out_trc = _src_color_trc or "bt709"
            _out_primaries = _src_color_primaries or "bt709"
            _out_space = _src_color_space or "bt709"
            _out_range = _src_color_range or "pc"
            # hevc_videotoolbox and prores_videotoolbox/prores_ks all drop the
            # -color_trc / -color_primaries / -colorspace flags. The workaround
            # that works for all three is the `setparams` video filter, which
            # stamps the tags onto AVFrames before the encoder — both VT (via
            # the VUI) and ProRes (via frame header) propagate them correctly.
            _color_vf = (f"setparams=range={_out_range}"
                         f":color_primaries={_out_primaries}"
                         f":color_trc={_out_trc}"
                         f":colorspace={_out_space}")
            _color_args = ["-color_range", _out_range,
                           "-colorspace", _out_space,
                           "-color_trc", _out_trc,
                           "-color_primaries", _out_primaries]
            # Tell FFmpeg the raw input color range matches the source
            _raw_input_color = ["-color_range", _out_range]

            # ── MV-HEVC direct-encode path ───────────────────────────────
            # When the user picks "Vision Pro MV-HEVC" the bytes that would
            # have gone to ffmpeg go to our mvhevc_encode Swift helper
            # instead. The helper produces a video-only .mov directly via
            # VTCompressionSessionEncodeMultiImageFrame with proper spatial
            # metadata baked in. We post-mux source audio with ffmpeg after
            # the main render loop finishes. This eliminates the prior
            # double-encode (SBS HEVC → spatial CLI re-decode → MV-HEVC).
            _is_mvhevc_direct = (cfg.vision_pro_mode == "mvhevc"
                                 and cfg.is_360_input
                                 and sys.platform == 'darwin')
            _mvhevc_video_only_path = None
            if _is_mvhevc_direct:
                _mvhevc_helper = get_mvhevc_encode_path()
                if not _mvhevc_helper:
                    raise Exception(
                        "mvhevc_encode helper not found. Expected next to vr180_gui.py "
                        "(development) or in the app bundle (frozen build). "
                        "Pick a different Vision Pro mode or rebuild the helper via "
                        "`swiftc -O -o mvhevc_encode mvhevc_encode.swift "
                        "-framework AVFoundation -framework VideoToolbox "
                        "-framework CoreMedia -framework CoreVideo -framework Accelerate`."
                    )
                # Half-width is the per-eye width (left/right halves of SBS)
                _per_eye_w = width // 2
                # Bitrate: use user's slider if set, otherwise a sensible default
                # for MV-HEVC main10 8K (~100 Mbps). MV-HEVC's inter-view
                # prediction makes each view cheap so this is lower than the
                # old spatial CLI default of 350 Mbps.
                if cfg.use_bitrate:
                    _mvhevc_bitrate_bps = max(5_000_000, cfg.bitrate * 1_000_000)
                else:
                    _mvhevc_bitrate_bps = 100_000_000
                # Use a temp .mov for the helper's video-only output; ffmpeg
                # post-muxes audio into cfg.output_path after the main loop.
                _mvhevc_video_only_path = Path(tempfile.mktemp(suffix='_mvhevc_video.mov'))
                _hero_eye = "left"
                encode_cmd = [
                    _mvhevc_helper,
                    "--width", str(_per_eye_w),
                    "--height", str(out_height),
                    "--fps", str(fps),
                    "--bitrate", str(_mvhevc_bitrate_bps),
                    "--baseline-mm", "65",
                    "--hfov-deg", "180",
                    "--hero", _hero_eye,
                    "--output", str(_mvhevc_video_only_path),
                ]
                print(f"[EXPORT] MV-HEVC direct encode: {_per_eye_w}×{out_height} per eye, "
                      f"{_mvhevc_bitrate_bps//1_000_000} Mbps → {_mvhevc_video_only_path}")
            else:
                encode_cmd = [
                    get_ffmpeg_path(), "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{out_height}",
                    "-pix_fmt", "bgr48le",
                    "-r", str(fps),
                ] + _raw_input_color + [
                    "-i", "-",
                ] + audio_input_args + [
                    "-map", "0:v",
                    "-vf", _color_vf,
                    "-f", "mov"
                ] + pix_fmt_args + _color_args + audio_args + enc + [str(cfg.output_path)]

            # Start encode process (decode started per-segment in decode_thread)
            self.status.emit("Starting video decode...")
            # Use DEVNULL for encoder stderr to prevent buffer blocking on Windows.
            # For mvhevc helper we want to capture stderr so we can report errors,
            # since it prints useful progress + error messages.
            _encode_stderr = subprocess.PIPE if _is_mvhevc_direct else subprocess.DEVNULL
            encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=_encode_stderr, creationflags=get_subprocess_flags())
            # Build backend summary for status (reflect actual code path used)
            # Mutable so decode thread can update on fallback
            _decode_tag_ref = ["PyAV" if HAS_AV else "FFmpeg"]
            _decode_tag = _decode_tag_ref[0]
            if cfg.is_360_input:
                _remap_tag = "MLX" if HAS_MLX else "CUDA" if HAS_NUMBA_CUDA else "wgpu" if HAS_WGPU else "Numba" if HAS_NUMBA else "CPU"
            else:
                # Non-360: MLX equirect (if no CUDA), CUDA fused, or CPU
                _remap_tag = "CUDA" if HAS_NUMBA_CUDA else "MLX" if HAS_MLX else "Numba" if HAS_NUMBA else "CPU"
            _sharpen_tag = ""
            if _sharpen_enabled:
                _sharpen_tag = " | Sharpen: CUDA" if _sharpen_use_cuda else " | Sharpen: CPU"
            _mlx_sharpen_tag = " | Sharpen: MLX" if _sharpen_use_mlx else ""
            if _mlx_sharpen_tag:
                _sharpen_tag = _mlx_sharpen_tag
            _rs_tag = ""
            if rs_enabled:
                _rs_tag = f" | RS: 3D-rot ({rs_yaw_factor:.1f},{rs_pitch_factor:.1f},{rs_roll_factor:.1f})"
            _denoise_tag = f" | Denoise: {cfg.denoise_strength*100:.0f}%" if cfg.denoise_strength > 0.01 else ""
            self.status.emit(f"Processing frames... [Decode: {_decode_tag_ref[0]} | Remap: {_remap_tag}{_denoise_tag}{_sharpen_tag}{_rs_tag}]")
            self.output_line.emit(
                f"Export: {total_frames} frames @ {fps:.2f} fps  |  "
                f"Decode: {_decode_tag_ref[0]}  |  Remap: {_remap_tag}{_sharpen_tag}{_rs_tag}  |  "
                f"Output: {cfg.output_path.name}")

            # For .360: decode raw EAC frames. With parallel decode, each stream is
            # 5952×1920; with vstack filter, it's 5952×3840.
            if cfg.is_360_input:
                if use_parallel_360_decode:
                    decode_w, decode_h = 5952, 1920  # single stream size
                else:
                    decode_w, decode_h = FrameExtractor.EAC_RAW_W, FrameExtractor.EAC_RAW_H
            elif use_parallel_sbs_decode:
                decode_w, decode_h = half_w, height  # each FFmpeg outputs one cropped eye
            else:
                decode_w, decode_h = width, height
            frame_size = decode_w * decode_h * 6  # BGR48LE (uint16, 3 channels × 2 bytes)
            frame_count = 0
            last_update_frame = 0
            import time
            import threading
            import queue
            start_process_time = time.time()

            # Threaded pipeline: decode -> process -> encode run concurrently
            decode_queue = queue.Queue(maxsize=4)   # decoded frames waiting to be processed
            # Encode queue maxsize=1: with triple-buffered numpy arrays, at most 1 buffer
            # can be in the queue + 1 being written by encode thread + 1 being filled by
            # main thread = 3 total. maxsize=2 would allow a race where main thread
            # starts writing to a buffer that encode thread is still writing to stdin.
            encode_queue = queue.Queue(maxsize=1)   # processed frames waiting to be encoded
            pipeline_error = [None]  # shared error state
            _dec_split_non360 = False  # set True below for non-360 pre-split path
            _dec_eye_pool = None

            if use_parallel_360_decode:
                def decode_thread():
                    """Parallel .360 decode: decode stream 0 + stream 4 concurrently,
                    assemble crosses, put (crossA, crossB) on queue.
                    Uses PyAV when available (2x faster), falls back to FFmpeg subprocess."""
                    def _decode_360_pyav():
                        for seg_idx in range(len(decode_segment_list)):
                            if self._cancelled:
                                break
                            seg_path = decode_segment_list[seg_idx][1]
                            _seg_skip = _seg_skip_frames[seg_idx]
                            _seg_max = _seg_max_frames[seg_idx]
                            if _seg_max is not None and _seg_max <= 0:
                                continue  # segment fully outside trim range
                            gen_s0 = _pyav_decode_stream(seg_path, stream_index=0,
                                                         skip_frames=_seg_skip, max_frames=_seg_max)
                            # s4 generator in helper thread
                            s4_buf = [None]
                            s4_err = [None]
                            _s4_ready = threading.Event()
                            _s4_consumed = threading.Event()
                            def _read_s4_pyav():
                                try:
                                    gen_s4 = _pyav_decode_stream(seg_path, stream_index=4,
                                                                 skip_frames=_seg_skip, max_frames=_seg_max)
                                    for arr in gen_s4:
                                        if self._cancelled:
                                            break
                                        s4_buf[0] = arr
                                        _s4_ready.set()
                                        _s4_consumed.wait()
                                        _s4_consumed.clear()
                                except Exception as e:
                                    s4_err[0] = e
                                finally:
                                    s4_buf[0] = None
                                    _s4_ready.set()
                            t_s4 = threading.Thread(target=_read_s4_pyav, daemon=True)
                            t_s4.start()
                            try:
                                _frames_queued = 0
                                for _d_s0 in gen_s0:
                                    if self._cancelled:
                                        break
                                    _s4_ready.wait()
                                    _s4_ready.clear()
                                    _d_s4 = s4_buf[0]
                                    if _d_s4 is None:
                                        break
                                    if s4_err[0]:
                                        raise s4_err[0]
                                    if _seg_max is not None and _frames_queued >= _seg_max:
                                        _s4_consumed.set()
                                        break
                                    # Assemble crosses in decode thread (overlaps with GPU processing)
                                    if _cross_bufs_A is not None:
                                        _dbi = _frames_queued % _N_CROSS_BUFS
                                        _d_cb_A = _cross_bufs_A[_dbi]
                                        _d_cb_B = _cross_bufs_B[_dbi]
                                        _d_tB = threading.Thread(target=FrameExtractor._assemble_lensB,
                                                                  args=(_d_s0, _d_s4, _d_cb_B))
                                        _d_tB.start()
                                        FrameExtractor._assemble_lensA(_d_s0, _d_s4, _d_cb_A)
                                        _d_tB.join()
                                        decode_queue.put((_d_cb_A, _d_cb_B))
                                    else:
                                        decode_queue.put((_d_s0, _d_s4))
                                    _frames_queued += 1
                                    _s4_consumed.set()
                            finally:
                                _s4_consumed.set()
                                t_s4.join(timeout=5)

                    def _decode_360_subprocess():
                        for seg_idx in range(len(decode_segment_list)):
                            if self._cancelled:
                                break
                            cmd_s0 = decode_segment_list[seg_idx][0]
                            cmd_s4 = decode_segment_list_s4[seg_idx][0]
                            _seg_skip = _seg_skip_frames[seg_idx]
                            _seg_max = _seg_max_frames[seg_idx]
                            if _seg_max is not None and _seg_max <= 0:
                                continue  # segment fully outside trim range
                            proc_s0 = subprocess.Popen(cmd_s0, stdout=subprocess.PIPE,
                                                        stderr=subprocess.DEVNULL,
                                                        creationflags=get_subprocess_flags())
                            proc_s4 = subprocess.Popen(cmd_s4, stdout=subprocess.PIPE,
                                                        stderr=subprocess.DEVNULL,
                                                        creationflags=get_subprocess_flags())
                            s4_buf = [None]
                            s4_err = [None]
                            def _read_s4():
                                try:
                                    while not self._cancelled:
                                        raw = proc_s4.stdout.read(frame_size)
                                        if len(raw) < frame_size:
                                            s4_buf[0] = None
                                            return
                                        s4_buf[0] = raw
                                        _s4_ready.set()
                                        _s4_consumed.wait()
                                        _s4_consumed.clear()
                                except Exception as e:
                                    s4_err[0] = e
                                finally:
                                    s4_buf[0] = None
                                    _s4_ready.set()
                            _s4_ready = threading.Event()
                            _s4_consumed = threading.Event()
                            t_s4 = threading.Thread(target=_read_s4, daemon=True)
                            t_s4.start()
                            try:
                                _dec_frame_n = 0
                                _frames_queued = 0
                                while not self._cancelled:
                                    raw_s0 = proc_s0.stdout.read(frame_size)
                                    if len(raw_s0) < frame_size:
                                        break
                                    _s4_ready.wait()
                                    _s4_ready.clear()
                                    raw_s4 = s4_buf[0]
                                    if raw_s4 is None:
                                        break
                                    if s4_err[0]:
                                        raise s4_err[0]
                                    _dec_frame_n += 1
                                    if _dec_frame_n <= _seg_skip:
                                        _s4_consumed.set()
                                        continue
                                    if _seg_max is not None and _frames_queued >= _seg_max:
                                        _s4_consumed.set()
                                        break
                                    if _cross_bufs_A is not None:
                                        _dbi = _frames_queued % _N_CROSS_BUFS
                                        _d_s0 = np.frombuffer(raw_s0, dtype=np.uint16).reshape((1920, 5952, 3))
                                        _d_s4 = np.frombuffer(raw_s4, dtype=np.uint16).reshape((1920, 5952, 3))
                                        _d_cb_A = _cross_bufs_A[_dbi]
                                        _d_cb_B = _cross_bufs_B[_dbi]
                                        _d_tB = threading.Thread(target=FrameExtractor._assemble_lensB,
                                                                  args=(_d_s0, _d_s4, _d_cb_B))
                                        _d_tB.start()
                                        FrameExtractor._assemble_lensA(_d_s0, _d_s4, _d_cb_A)
                                        _d_tB.join()
                                        decode_queue.put((_d_cb_A, _d_cb_B))
                                    else:
                                        decode_queue.put((raw_s0, raw_s4))
                                    _frames_queued += 1
                                    _s4_consumed.set()
                            finally:
                                _s4_consumed.set()
                                proc_s0.stdout.close()
                                proc_s0.wait()
                                proc_s4.stdout.close()
                                proc_s4.wait()
                                t_s4.join(timeout=5)

                    def _decode_360_denoise():
                        """Decode .360 via FFmpeg, pipe each stream through vt_denoise for temporal denoise."""
                        vt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vt_denoise')
                        for seg_idx in range(len(decode_segment_list)):
                            if self._cancelled:
                                break
                            seg_path = decode_segment_list[seg_idx][1]
                            cmd_s0 = decode_segment_list[seg_idx][0]
                            cmd_s4 = decode_segment_list_s4[seg_idx][0]
                            _seg_skip = _seg_skip_frames[seg_idx]
                            _seg_max = _seg_max_frames[seg_idx]
                            if _seg_max is not None and _seg_max <= 0:
                                continue  # segment fully outside trim range

                            # Chain: FFmpeg decode → vt_denoise --pipe → output
                            # Stream 0
                            ffmpeg_s0 = subprocess.Popen(cmd_s0, stdout=subprocess.PIPE,
                                                          stderr=subprocess.DEVNULL, creationflags=get_subprocess_flags())
                            denoise_s0 = subprocess.Popen(
                                [vt_path, '--pipe', '--width', '5952', '--height', '1920',
                                 '--strength', str(cfg.denoise_strength)],
                                stdin=ffmpeg_s0.stdout, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
                            ffmpeg_s0.stdout.close()  # allow SIGPIPE

                            # Stream 4 in helper thread
                            ffmpeg_s4 = subprocess.Popen(cmd_s4, stdout=subprocess.PIPE,
                                                          stderr=subprocess.DEVNULL, creationflags=get_subprocess_flags())
                            denoise_s4 = subprocess.Popen(
                                [vt_path, '--pipe', '--width', '5952', '--height', '1920',
                                 '--strength', str(cfg.denoise_strength)],
                                stdin=ffmpeg_s4.stdout, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
                            ffmpeg_s4.stdout.close()

                            s4_buf = [None]
                            _s4_ready = threading.Event()
                            _s4_consumed = threading.Event()
                            def _read_s4_dn():
                                try:
                                    while not self._cancelled:
                                        raw = denoise_s4.stdout.read(frame_size)
                                        if len(raw) < frame_size:
                                            s4_buf[0] = None; return
                                        s4_buf[0] = raw
                                        _s4_ready.set()
                                        _s4_consumed.wait()
                                        _s4_consumed.clear()
                                except Exception:
                                    pass
                                finally:
                                    s4_buf[0] = None; _s4_ready.set()
                            t_s4 = threading.Thread(target=_read_s4_dn, daemon=True)
                            t_s4.start()

                            try:
                                _dec_frame_n = 0; _frames_queued = 0
                                while not self._cancelled:
                                    raw_s0 = denoise_s0.stdout.read(frame_size)
                                    if len(raw_s0) < frame_size:
                                        break
                                    _s4_ready.wait(); _s4_ready.clear()
                                    raw_s4 = s4_buf[0]
                                    if raw_s4 is None:
                                        break
                                    _dec_frame_n += 1
                                    if _dec_frame_n <= _seg_skip:
                                        _s4_consumed.set(); continue
                                    if _seg_max is not None and _frames_queued >= _seg_max:
                                        _s4_consumed.set(); break
                                    if _cross_bufs_A is not None:
                                        _dbi = _frames_queued % _N_CROSS_BUFS
                                        _d_s0 = np.frombuffer(raw_s0, dtype=np.uint16).reshape((1920, 5952, 3))
                                        _d_s4 = np.frombuffer(raw_s4, dtype=np.uint16).reshape((1920, 5952, 3))
                                        _d_cb_A = _cross_bufs_A[_dbi]
                                        _d_cb_B = _cross_bufs_B[_dbi]
                                        _d_tB = threading.Thread(target=FrameExtractor._assemble_lensB,
                                                                  args=(_d_s0, _d_s4, _d_cb_B))
                                        _d_tB.start()
                                        FrameExtractor._assemble_lensA(_d_s0, _d_s4, _d_cb_A)
                                        _d_tB.join()
                                        decode_queue.put((_d_cb_A, _d_cb_B))
                                    else:
                                        decode_queue.put((raw_s0, raw_s4))
                                    _frames_queued += 1
                                    _s4_consumed.set()
                            finally:
                                _s4_consumed.set()
                                denoise_s0.stdout.close(); denoise_s4.stdout.close()
                                for p in [denoise_s0, denoise_s4, ffmpeg_s0, ffmpeg_s4]:
                                    try: p.wait(timeout=5)
                                    except: pass
                                t_s4.join(timeout=5)

                    _use_360_denoise = (sys.platform == 'darwin' and cfg.denoise_strength > 0.01
                                        and cfg.is_360_input
                                        and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vt_denoise')))
                    try:
                        if _use_360_denoise:
                            try:
                                print(f"[DECODE] Using vt_denoise pipe for .360 (strength={cfg.denoise_strength:.2f})")
                                _decode_tag_ref[0] = "VT Denoise"
                                _decode_360_denoise()
                            except Exception as _dn_err:
                                print(f"⚠ vt_denoise failed ({_dn_err}), falling back")
                                _decode_tag_ref[0] = "PyAV (fallback)"
                                if HAS_AV:
                                    _decode_360_pyav()
                                else:
                                    _decode_360_subprocess()
                        elif HAS_AV:
                            try:
                                print("[DECODE] Using PyAV for .360 parallel decode")
                                _decode_360_pyav()
                            except Exception as _av_err:
                                print(f"⚠ PyAV decode failed ({_av_err}), falling back to FFmpeg subprocess")
                                _decode_tag_ref[0] = "FFmpeg (fallback)"
                                _decode_360_subprocess()
                        else:
                            print("[DECODE] Using FFmpeg subprocess for .360 parallel decode")
                            _decode_360_subprocess()
                    except Exception as e:
                        pipeline_error[0] = e
                    finally:
                        decode_queue.put(None)  # sentinel
            elif use_parallel_sbs_decode:
                _sbs_crop_A = (left_start, 0, half_w, height)
                _sbs_crop_B = (right_start, 0, half_w, height)
                def decode_thread():
                    """Parallel non-360 SBS decode: decode full frame via PyAV and numpy-crop
                    each eye, or use FFmpeg subprocess with crop filters as fallback."""
                    def _decode_sbs_pyav():
                        # Decode once, crop both eyes from each frame (avoids decoding twice)
                        cx_a, cy_a, cw_a, ch_a = _sbs_crop_A
                        cx_b, cy_b, cw_b, ch_b = _sbs_crop_B
                        for seg_idx in range(len(decode_segment_list)):
                            if self._cancelled:
                                break
                            seg_path = decode_segment_list[seg_idx][1]
                            _max_f = int(duration * fps) + 2 if duration else None
                            gen = _pyav_decode_stream(seg_path, stream_index=0, max_frames=_max_f)
                            for full_frame in gen:
                                if self._cancelled:
                                    break
                                arr_A = full_frame[cy_a:cy_a+ch_a, cx_a:cx_a+cw_a].copy()
                                arr_B = full_frame[cy_b:cy_b+ch_b, cx_b:cx_b+cw_b].copy()
                                decode_queue.put((arr_A, arr_B))

                    def _decode_sbs_subprocess():
                        for seg_idx in range(len(decode_segment_list)):
                            if self._cancelled:
                                break
                            cmd_A = decode_segment_list[seg_idx][0]
                            cmd_B = decode_segment_list_s4[seg_idx][0]
                            proc_A = subprocess.Popen(cmd_A, stdout=subprocess.PIPE,
                                                       stderr=subprocess.DEVNULL,
                                                       creationflags=get_subprocess_flags())
                            proc_B = subprocess.Popen(cmd_B, stdout=subprocess.PIPE,
                                                       stderr=subprocess.DEVNULL,
                                                       creationflags=get_subprocess_flags())
                            _eyeB_buf = [None]
                            _eyeB_err = [None]
                            def _read_eye_B():
                                try:
                                    while not self._cancelled:
                                        raw = proc_B.stdout.read(frame_size)
                                        if len(raw) < frame_size:
                                            _eyeB_buf[0] = None
                                            return
                                        _eyeB_buf[0] = raw
                                        _eyeB_ready.set()
                                        _eyeB_consumed.wait()
                                        _eyeB_consumed.clear()
                                except Exception as e:
                                    _eyeB_err[0] = e
                                finally:
                                    _eyeB_buf[0] = None
                                    _eyeB_ready.set()
                            _eyeB_ready = threading.Event()
                            _eyeB_consumed = threading.Event()
                            t_B = threading.Thread(target=_read_eye_B, daemon=True)
                            t_B.start()
                            try:
                                while not self._cancelled:
                                    raw_A = proc_A.stdout.read(frame_size)
                                    if len(raw_A) < frame_size:
                                        break
                                    _eyeB_ready.wait()
                                    _eyeB_ready.clear()
                                    raw_B = _eyeB_buf[0]
                                    if raw_B is None:
                                        break
                                    if _eyeB_err[0]:
                                        raise _eyeB_err[0]
                                    decode_queue.put((raw_A, raw_B))
                                    _eyeB_consumed.set()
                            finally:
                                _eyeB_consumed.set()
                                proc_A.stdout.close()
                                proc_A.wait()
                                proc_B.stdout.close()
                                proc_B.wait()
                                t_B.join(timeout=5)

                    try:
                        if HAS_AV:
                            try:
                                print("[DECODE] Using PyAV for SBS parallel decode")
                                _decode_sbs_pyav()
                            except Exception as _av_err:
                                print(f"⚠ PyAV decode failed ({_av_err}), falling back to FFmpeg subprocess")
                                _decode_tag_ref[0] = "FFmpeg (fallback)"
                                _decode_sbs_subprocess()
                        else:
                            _decode_sbs_subprocess()
                    except Exception as e:
                        pipeline_error[0] = e
                    finally:
                        decode_queue.put(None)  # sentinel
            else:
                def decode_thread():
                    """Read raw frames from decoder(s). Uses PyAV when available,
                    falls back to FFmpeg subprocess."""
                    def _decode_single_pyav():
                        for seg_info in decode_segment_list:
                            if self._cancelled:
                                break
                            seg_path = seg_info[1]
                            _max_f = int(seg_info[3] * fps) + 2 if seg_info[3] else None
                            gen = _pyav_decode_stream(seg_path, stream_index=0, max_frames=_max_f)
                            for arr in gen:
                                if self._cancelled:
                                    break
                                decode_queue.put(arr)

                    def _decode_single_subprocess():
                        for seg_info in decode_segment_list:
                            if self._cancelled:
                                break
                            seg_cmd = seg_info[0]
                            proc = subprocess.Popen(seg_cmd, stdout=subprocess.PIPE,
                                                    stderr=subprocess.DEVNULL,
                                                    creationflags=get_subprocess_flags())
                            try:
                                while not self._cancelled:
                                    raw = proc.stdout.read(frame_size)
                                    if len(raw) < frame_size:
                                        break
                                    decode_queue.put(raw)
                            finally:
                                proc.stdout.close()
                                proc.wait()

                    def _decode_vt_denoise():
                        """Decode + denoise via vt_denoise Swift helper (macOS 26+)."""
                        vt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vt_denoise')
                        if not os.path.isfile(vt_path):
                            raise FileNotFoundError(f"vt_denoise not found at {vt_path}")
                        for seg_info in decode_segment_list:
                            if self._cancelled:
                                break
                            seg_path = seg_info[1]
                            _skip = int((seg_info[2] or 0) * fps) if seg_info[2] else 0
                            _max_f = int((seg_info[3] or 9999) * fps) + 2 if seg_info[3] else total_frames + 2
                            cmd = [vt_path, seg_path,
                                   '--strength', str(cfg.denoise_strength),
                                   '--skip', str(_skip),
                                   '--max', str(_max_f),
                                   '--format', 'bgr48le']
                            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                    stderr=subprocess.DEVNULL)
                            try:
                                while not self._cancelled:
                                    raw = proc.stdout.read(frame_size)
                                    if len(raw) < frame_size:
                                        break
                                    decode_queue.put(raw)
                            finally:
                                proc.stdout.close()
                                proc.wait()

                    _use_vt_denoise = (sys.platform == 'darwin' and cfg.denoise_strength > 0.01
                                       and not cfg.is_360_input)
                    try:
                        if _use_vt_denoise:
                            try:
                                print(f"[DECODE] Using vt_denoise (strength={cfg.denoise_strength:.2f})")
                                _decode_tag_ref[0] = "VT Denoise"
                                _decode_vt_denoise()
                            except Exception as _vt_err:
                                print(f"⚠ vt_denoise failed ({_vt_err}), falling back to PyAV")
                                _decode_tag_ref[0] = "PyAV (fallback)"
                                if HAS_AV:
                                    _decode_single_pyav()
                                else:
                                    _decode_single_subprocess()
                        elif HAS_AV:
                            try:
                                print("[DECODE] Using PyAV for single-stream decode")
                                _decode_single_pyav()
                            except Exception as _av_err:
                                print(f"⚠ PyAV decode failed ({_av_err}), falling back to FFmpeg subprocess")
                                _decode_tag_ref[0] = "FFmpeg (fallback)"
                                _decode_single_subprocess()
                        else:
                            _decode_single_subprocess()
                    except Exception as e:
                        pipeline_error[0] = e
                    finally:
                        decode_queue.put(None)  # sentinel

            def encode_thread():
                """Write processed frames to FFmpeg encoder.
                Accepts either bytes or numpy arrays (written via memoryview to avoid copy)."""
                written = 0
                try:
                    while True:
                        data = encode_queue.get()
                        if data is None:
                            break
                        if isinstance(data, np.ndarray):
                            # Write directly from numpy buffer via memoryview (avoids tobytes copy)
                            encode_proc.stdin.write(data.data)
                        else:
                            encode_proc.stdin.write(data)
                        written += 1
                        if sys.platform == 'win32' and written % 30 == 0:
                            encode_proc.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    pipeline_error[0] = Exception(f"Encoder terminated: {e}")
                except Exception as e:
                    pipeline_error[0] = e

            # Start pipeline threads
            dec_t = threading.Thread(target=decode_thread, daemon=True)
            enc_t = threading.Thread(target=encode_thread, daemon=True)
            dec_t.start()
            enc_t.start()

            try:
                while True:
                    if self._cancelled:
                        encode_proc.terminate()
                        self.finished_signal.emit(False, "Cancelled")
                        return

                    if pipeline_error[0]:
                        raise pipeline_error[0]

                    # Get decoded frame from decode thread.
                    #
                    # Poll in short intervals and check dec_t liveness so we
                    # don't wrongly time out while the decoder is still chewing
                    # through a deep trim-start skip (multi-segment .360 with
                    # trim can need tens of seconds of software HEVC decode
                    # before the first real frame is queued — PyAV doesn't
                    # hwaccel this path yet).
                    #
                    # NVDEC hang watchdog semantics still apply: we only bail
                    # when dec_t is alive AND not producing frames AND the
                    # dead-man wait limit has been exceeded. Cold-start gets
                    # a longer limit (skip phase); steady-state keeps the
                    # original 30s budget.
                    _first_frame = (frame_count == 0)
                    _dead_man_limit = 180.0 if _first_frame else 30.0
                    _poll_interval = 2.0
                    raw_frame = None
                    _t_wait_start = time.time()
                    while True:
                        if self._cancelled:
                            break
                        if pipeline_error[0]:
                            raise pipeline_error[0]
                        try:
                            raw_frame = decode_queue.get(timeout=_poll_interval)
                            break
                        except queue.Empty:
                            if not dec_t.is_alive():
                                # Decode thread exited without producing the
                                # next frame. Surface any captured error, else
                                # report that the thread died silently.
                                if pipeline_error[0]:
                                    raise pipeline_error[0]
                                raise RuntimeError(
                                    "Decode thread exited without producing a frame")
                            _waited = time.time() - _t_wait_start
                            if _waited < _dead_man_limit:
                                # Decode thread alive and still within the
                                # wait budget — keep waiting. For cold-start
                                # with deep trim, progress may be invisible
                                # until the skip phase finishes.
                                continue
                            # Exceeded dead-man limit: likely a real hang.
                            if sys.platform == 'win32' and _has_nvdec and not _disable_hwaccel[0]:
                                _disable_hwaccel[0] = True
                                self._disable_hwaccel = True  # persist for session
                                print("[EXPORT] ⚠ Hardware decode (NVDEC) timed out — retrying without hardware acceleration...")
                                raise RuntimeError("__RETRY_NO_HWACCEL__")
                            raise RuntimeError(
                                f"Decode queue timeout: no frame received in {int(_waited)}s "
                                f"(decode thread still alive). Suspected hang.")
                    if self._cancelled:
                        encode_proc.terminate()
                        self.finished_signal.emit(False, "Cancelled")
                        return
                    if raw_frame is None:
                        break

                    _dec_eye_pair = None  # track for buffer pool return
                    _pre_crosses = None
                    if use_parallel_360_decode:
                        # Parallel .360 decode: crosses pre-assembled in decode thread
                        _pre_crosses = raw_frame  # (crossA, crossB) tuple
                        _pre_s0 = None
                        _pre_s4 = None
                        frame = None  # not used for .360 parallel path
                    elif use_parallel_sbs_decode and isinstance(raw_frame, tuple):
                        # Parallel SBS decode: (eyeA, eyeB) — numpy arrays (PyAV) or bytes (subprocess)
                        raw_A, raw_B = raw_frame
                        if isinstance(raw_A, np.ndarray):
                            np.copyto(_eq_src_buf_A, raw_A)
                            np.copyto(_eq_src_buf_B, raw_B)
                        else:
                            np.copyto(_eq_src_buf_A, np.frombuffer(raw_A, dtype=np.uint16).reshape((decode_h, half_w, 3)))
                            np.copyto(_eq_src_buf_B, np.frombuffer(raw_B, dtype=np.uint16).reshape((decode_h, half_w, 3)))
                        _dec_eye_pair = (_eq_src_buf_A, _eq_src_buf_B)
                        frame = None
                        _pre_s0 = None
                        _pre_s4 = None
                    else:
                        # Single-stream: numpy array (PyAV) or bytes (subprocess)
                        if isinstance(raw_frame, np.ndarray):
                            frame = raw_frame
                        else:
                            frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape((decode_h, decode_w, 3))
                        _pre_s0 = None
                        _pre_s4 = None
                    # frombuffer is read-only but frame/s0/s4 are only used as remap source (read-only).
                    # RS correction copies its own slice internally when needed.

                    # Per-frame stabilization: pre-computed array lookup (zero math)
                    # Clamp index: FFmpeg may decode 1-2 extra frames beyond total_frames estimate
                    if precomputed is not None:
                        _fc = min(frame_count, total_frames - 1)
                        if frame_count < 3:
                            _dbg_t = start_time + _fc / fps
                            print(f"[EXPORT] frame_count={frame_count} _fc={_fc} lookup_t={_dbg_t:.4f}s "
                                  f"mat_diag=[{precomputed['left_matrices'][_fc][0,0]:.6f},"
                                  f"{precomputed['left_matrices'][_fc][1,1]:.6f},"
                                  f"{precomputed['left_matrices'][_fc][2,2]:.6f}]")
                        gyro_left = precomputed['left_matrices'][_fc]
                        gyro_right = precomputed['right_matrices'][_fc]
                        if _export_rs_enabled:
                            angular_vel = precomputed['angular_vel'][_fc]
                            rs_R_sensor_right = precomputed['heading_matrices'][_fc]
                            rs_R_cross_right = precomputed['iori_right_matrices'][_fc]
                            _rs_perscanline_mats = precomputed['rs_perscanline'][_fc] if precomputed.get('rs_perscanline') is not None else None
                        else:
                            angular_vel = None
                            rs_R_sensor_right = None
                            rs_R_cross_right = None
                            _rs_perscanline_mats = None
                    else:
                        # No gyro: apply manual view adjustments (panomap rotation + stereo) directly
                        gyro_left = R_view_left.astype(np.float64) if R_view_left is not None else np.eye(3, dtype=np.float64)
                        gyro_right = R_view_right.astype(np.float64) if R_view_right is not None else np.eye(3, dtype=np.float64)
                        angular_vel = None
                        _rs_perscanline_mats = None
                        rs_R_sensor_right = None
                        rs_R_cross_right = None

                    processed = process_frame_opencv(frame, gyro_left, gyro_right, angular_vel,
                                                     rs_R_sensor_right, rs_R_cross_right,
                                                     pre_split_s0=_pre_s0, pre_split_s4=_pre_s4,
                                                     pre_split_eyes=_dec_eye_pair,
                                                     pre_crosses=_pre_crosses)

                    # Fisheye: zero pixels outside circular FOV (cv2.bitwise_and, ~6ms)
                    if _fisheye_mode and _fish_mask_3ch is not None:
                        cv2.bitwise_and(processed[:, half_w:], _fish_mask_3ch, dst=processed[:, half_w:])
                        cv2.bitwise_and(processed[:, :half_w], _fish_mask_3ch, dst=processed[:, :half_w])

                    # Return pre-split eye buffers to pool (data already consumed by GPU/remap)
                    if _dec_eye_pair is not None and _dec_eye_pool is not None:
                        _dec_eye_pool.put(_dec_eye_pair)

                    # Triple-buffer: enqueue current buffer as numpy array (encode thread
                    # writes via memoryview, avoiding tobytes() copy ~13ms/frame at 8K),
                    # then cycle to the next buffer for processing.
                    if processed.flags['C_CONTIGUOUS']:
                        encode_queue.put(processed, timeout=30)
                    else:
                        # Non-contiguous (e.g. after eye mask) — must copy
                        encode_queue.put(processed.tobytes(), timeout=30)
                    # Cycle to next result buffer
                    _result_buf_idx = (_result_buf_idx + 1) % 3
                    result_buf = _result_bufs[_result_buf_idx]

                    frame_count += 1

                    # Update progress periodically
                    if frame_count - last_update_frame >= 10:
                        progress = int((frame_count / total_frames) * 90)
                        self.progress.emit(min(progress, 90))
                        elapsed = time.time() - start_process_time
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        percent = (frame_count / total_frames) * 100
                        eta_s = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                        eta_m, eta_sec = divmod(int(eta_s), 60)
                        self.status.emit(f"{percent:.1f}% - Frame {frame_count}/{total_frames} @ {fps_actual:.1f} fps [{_remap_tag}{_sharpen_tag}]")
                        self.output_line.emit(
                            f"Frame {frame_count}/{total_frames}  |  {fps_actual:.1f} fps  |  "
                            f"ETA {eta_m}:{eta_sec:02d}  |  Elapsed {int(elapsed)}s  |  "
                            f"Decode: {_decode_tag_ref[0]}  |  Remap: {_remap_tag}{_sharpen_tag}{_rs_tag}")
                        last_update_frame = frame_count

                # Signal encode thread to finish and wait
                encode_queue.put(None)
                enc_t.join(timeout=10)
                dec_t.join(timeout=10)
                try:
                    encode_proc.stdin.flush()
                except:
                    pass
                try:
                    encode_proc.stdin.close()
                except:
                    pass

                self.status.emit("Finalizing video encoding...")

                try:
                    encode_proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    encode_proc.kill()
                    encode_proc.wait()
                    raise Exception("Encoding timed out during finalization")

                # For mvhevc direct mode, the helper writes to a temp path
                # (video-only). Audio post-mux into cfg.output_path happens
                # below after the return-code check.
                _check_path = _mvhevc_video_only_path if _is_mvhevc_direct else cfg.output_path

                if encode_proc.returncode != 0:
                    # Some encoders (e.g. ProRes on macOS) return non-zero even on success.
                    # If the output file exists and has reasonable size, treat as success.
                    if _check_path.exists() and _check_path.stat().st_size > 1024:
                        print(f"Warning: encoder returned code {encode_proc.returncode} but output file looks valid")
                    else:
                        # For mvhevc helper, capture and forward stderr for diagnosis
                        _err_tail = ""
                        if _is_mvhevc_direct and encode_proc.stderr is not None:
                            try:
                                _err_tail = encode_proc.stderr.read().decode(errors='replace')[-2000:]
                            except Exception:
                                pass
                        raise Exception(
                            f"Encoding failed (return code {encode_proc.returncode})"
                            + (f"\n\nhelper stderr:\n{_err_tail}" if _err_tail else "")
                        )

                if pipeline_error[0]:
                    raise pipeline_error[0]

                # ── MV-HEVC audio post-mux ──────────────────────────────
                # The mvhevc_encode helper produced a video-only .mov at
                # _mvhevc_video_only_path. Run ffmpeg once to copy the
                # video stream as-is and add the source audio, writing
                # cfg.output_path. Same audio mapping logic as the normal
                # export path so multi-segment concat + ambisonic tracks
                # still work.
                if _is_mvhevc_direct:
                    self.status.emit("Muxing source audio into MV-HEVC file...")
                    # For multi-segment sources, concat audio via the same
                    # concat list used for the normal export path (already
                    # built above as audio_concat_file / audio_input_args).
                    _mux_cmd = [
                        get_ffmpeg_path(), "-y",
                        "-i", str(_mvhevc_video_only_path),
                    ] + audio_input_args + [
                        "-map", "0:v",
                        "-c:v", "copy",
                    ] + audio_args + [
                        "-movflags", "+faststart",
                        "-f", "mov",
                        str(cfg.output_path),
                    ]
                    print(f"[EXPORT] MV-HEVC audio post-mux: {' '.join(str(x) for x in _mux_cmd)}")
                    _mux_proc = subprocess.run(_mux_cmd, capture_output=True,
                                               creationflags=get_subprocess_flags())
                    if _mux_proc.returncode != 0:
                        _mux_err = _mux_proc.stderr.decode(errors='replace')[-2000:]
                        raise Exception(
                            f"MV-HEVC audio post-mux failed (code {_mux_proc.returncode}):\n{_mux_err}"
                        )
                    # Clean up the video-only intermediate
                    try:
                        _mvhevc_video_only_path.unlink()
                    except Exception:
                        pass

                self.progress.emit(100)

                # Post-processing: VR180 metadata + Vision Pro
                completion_tags = []
                seg_msg = f" from {len(decode_segment_list)} segments" if len(decode_segment_list) > 1 else ""
                completion_tags.append(f"{frame_count} frames{seg_msg}")

                if cfg.inject_vr180_metadata:
                    self.status.emit("Injecting VR180 metadata for YouTube...")
                    try:
                        self._inject_vr180_metadata(cfg.output_path)
                        completion_tags.append("YouTube VR180")
                    except Exception as meta_error:
                        completion_tags.append(f"VR180 metadata failed: {meta_error}")

                if cfg.vision_pro_mode == "hvc1":
                    # The hvc1 tag is HEVC-specific (Apple's fourcc for HEVC video
                    # sample descriptions). Skip injection for non-HEVC outputs
                    # such as ProRes (apch/apcn) or H.264 (avc1) — ffmpeg would
                    # reject -tag:v hvc1 with "Tag hvc1 incompatible with output
                    # codec id" and the byte-walking variant can't find the box.
                    if output_codec != "h265":
                        completion_tags.append(
                            f"hvc1 tag skipped (output codec is {output_codec}, not HEVC)")
                    else:
                        self.status.emit("Adding hvc1 tag for Apple compatibility...")
                        try:
                            self._inject_hvc1_tag(cfg.output_path)
                            completion_tags.append("Apple compatible")
                        except Exception as meta_error:
                            completion_tags.append(f"hvc1 tag failed: {meta_error}")
                elif cfg.vision_pro_mode == "mvhevc":
                    # Direct-encoded via mvhevc_encode + post-muxed above.
                    # No second transcode — the file is already MV-HEVC
                    # with spatial metadata baked into the format
                    # description by VideoToolbox.
                    completion_tags.append("Vision Pro MV-HEVC (direct)")

                # Emit export summary to info panel
                _total_time = time.time() - start_process_time
                _avg_fps = frame_count / _total_time if _total_time > 0 else 0
                try:
                    _out_size = cfg.output_path.stat().st_size
                    _out_str = f"{_out_size/1e9:.2f} GB" if _out_size > 1e9 else f"{_out_size/1e6:.1f} MB"
                except:
                    _out_str = "?"
                self.output_line.emit(
                    f"Export complete: {frame_count} frames in {_total_time:.1f}s ({_avg_fps:.1f} fps avg)  |  "
                    f"Output: {_out_str}  |  {cfg.output_path.name}")
                self.finished_signal.emit(True, f"Complete! ({', '.join(completion_tags)})")

            except Exception as e:
                try:
                    encode_proc.stdin.close()
                except:
                    pass
                try:
                    encode_proc.terminate()
                except:
                    pass
                raise e
            finally:
                # Clean up concat file if created
                if audio_concat_file is not None:
                    try:
                        os.unlink(audio_concat_file.name)
                    except:
                        pass

        except Exception as e:
            import traceback
            error_msg = str(e)
            # Don't show internal retry sentinel as user-facing error
            if error_msg == "__RETRY_NO_HWACCEL__":
                error_msg = ("FFmpeg hardware decode (NVDEC) failed on this file. "
                             "Try processing again — hardware acceleration has been disabled for this session.")
            print(f"GoPro Roll Fix Error: {error_msg}")
            print(traceback.format_exc())
            self.finished_signal.emit(False, error_msg)

    def _inject_hvc1_tag(self, video_path: Path):
        """Inject VR180 SBS APMP metadata for Vision Pro (fast, no re-encode).

        Writes Apple Projected Media Profile atoms directly into the MP4:
        - vexu/eyes/stri: stereo side-by-side indication
        - vexu/proj/prji: halfEquirectangular projection (VR180)
        - vexu/pack/pkin: sideBySide view packing
        - vexu/cams/blin: stereo baseline (interpupillary distance) = 65 mm
        - hfov: 180° horizontal field of view (sibling of vexu)
        Works with visionOS 26+ native player.
        """
        import shutil, tempfile

        # Build APMP atoms (matches avconvert output byte-for-byte)
        stri = struct.pack('>I', 13) + b'stri' + b'\x00\x00\x00\x00' + b'\x03'
        eyes = struct.pack('>I', 8 + len(stri)) + b'eyes' + stri
        prji = struct.pack('>I', 16) + b'prji' + b'\x00\x00\x00\x00' + b'hequ'
        proj = struct.pack('>I', 8 + len(prji)) + b'proj' + prji
        pkin = struct.pack('>I', 16) + b'pkin' + b'\x00\x00\x00\x00' + b'side'
        pack = struct.pack('>I', 8 + len(pkin)) + b'pack' + pkin

        # blin: stereo baseline (interpupillary distance) in MICROMETERS.
        # 65 mm = 65 000 µm — the standard human IPD, used as a comfortable
        # default so Vision Pro's depth perception matches natural eye spacing.
        # (ISO BMFF "full box": 4 size + 4 'blin' + 4 version/flags + 4 value)
        baseline_um = 65000
        blin = (struct.pack('>I', 16) + b'blin' + b'\x00\x00\x00\x00'
                + struct.pack('>I', baseline_um))
        # cams: camera configuration container (child of vexu)
        cams = struct.pack('>I', 8 + len(blin)) + b'cams' + blin

        vexu = (struct.pack('>I', 8 + len(eyes) + len(proj) + len(pack) + len(cams))
                + b'vexu' + eyes + proj + pack + cams)
        # hfov: horizontal field of view = 180.0° (stored as 180000 fixed-point ×1000)
        # Sibling of vexu inside hvc1 (same as Apple's reference files)
        hfov = struct.pack('>I', 12) + b'hfov' + struct.pack('>I', 180000)
        inject_data = vexu + hfov
        inject_size = len(inject_data)

        # Read entire file
        # Scan for moov position without loading entire file (handles multi-GB files)
        _moov_off = None; _moov_sz = None
        with open(str(video_path), 'rb') as f:
            f.seek(0, 2); _fsize = f.tell(); f.seek(0)
            while f.tell() < _fsize:
                _pos = f.tell()
                _hdr = f.read(8)
                if len(_hdr) < 8: break
                _sz, _tag = struct.unpack('>I4s', _hdr)
                if _sz == 1:
                    _sz = struct.unpack('>Q', f.read(8))[0]
                elif _sz == 0:
                    _sz = _fsize - _pos
                if _sz < 8: break
                if _tag == b'moov':
                    _moov_off = _pos; _moov_sz = _sz; break
                f.seek(_pos + _sz)
        if _moov_off is None:
            raise Exception("moov atom not found")
        with open(str(video_path), 'rb') as f:
            f.seek(_moov_off)
            data = bytearray(f.read(_moov_sz))

        # Walk atom tree to find the exact parent chain: moov → trak(video) → mdia → minf → stbl → stsd → hvc1
        # This avoids corrupting the audio trak by using find() on duplicate tags
        def find_atom(buf, start, end, tag):
            """Find atom with given 4-byte tag within [start, end). Returns (offset, size) or None."""
            pos = start
            while pos < end - 8:
                sz = struct.unpack('>I', buf[pos:pos+4])[0]
                t = buf[pos+4:pos+8]
                if sz == 1 and pos + 16 <= end:
                    sz = struct.unpack('>Q', buf[pos+8:pos+16])[0]
                elif sz == 0:
                    sz = end - pos
                if sz < 8 or pos + sz > end:
                    break
                if t == tag:
                    return (pos, sz)
                pos += sz
            return None

        def find_video_trak(buf, moov_off, moov_sz):
            """Find the video trak (contains vmhd or hdlr with 'vide')."""
            pos = moov_off + 8
            end = moov_off + moov_sz
            while pos < end - 8:
                sz = struct.unpack('>I', buf[pos:pos+4])[0]
                t = buf[pos+4:pos+8]
                if sz < 8 or pos + sz > end:
                    break
                if t == b'trak':
                    # Check if this trak contains a video handler
                    if buf[pos:pos+sz].find(b'vide') >= 0:
                        return (pos, sz)
                pos += sz
            return None

        # Find moov
        moov = find_atom(data, 0, len(data), b'moov')
        if not moov:
            raise Exception("moov atom not found")
        moov_off, moov_sz = moov

        # Find video trak
        trak = find_video_trak(data, moov_off, moov_sz)
        if not trak:
            raise Exception("Video trak not found")
        trak_off, trak_sz = trak

        # Walk: trak → mdia → minf → stbl → stsd → hvc1
        mdia = find_atom(data, trak_off + 8, trak_off + trak_sz, b'mdia')
        if not mdia: raise Exception("mdia not found")
        minf = find_atom(data, mdia[0] + 8, mdia[0] + mdia[1], b'minf')
        if not minf: raise Exception("minf not found")
        stbl = find_atom(data, minf[0] + 8, minf[0] + minf[1], b'stbl')
        if not stbl: raise Exception("stbl not found")
        stsd = find_atom(data, stbl[0] + 8, stbl[0] + stbl[1], b'stsd')
        if not stsd: raise Exception("stsd not found")
        # hvc1 is inside stsd after 16-byte header (8 + version(4) + entry_count(4))
        hvc1 = find_atom(data, stsd[0] + 16, stsd[0] + stsd[1], b'hvc1')
        if not hvc1:
            hvc1 = find_atom(data, stsd[0] + 16, stsd[0] + stsd[1], b'hev1')
        if not hvc1:
            # Detect what codec the file actually uses so we give a useful error
            # instead of "hvc1/hev1 not found". Common non-HEVC sample entries:
            #   apch/apcn/apco/apcs/ap4h/ap4x = ProRes
            #   avc1/avc3                     = H.264
            for _codec_tag in (b'apch', b'apcn', b'apco', b'apcs', b'ap4h', b'ap4x'):
                if find_atom(data, stsd[0] + 16, stsd[0] + stsd[1], _codec_tag):
                    raise Exception(
                        f"hvc1 tag injection requires HEVC output, but this file is "
                        f"ProRes ({_codec_tag.decode()}). Re-export with H.265/HEVC "
                        f"codec, or use the MV-HEVC Vision Pro mode instead.")
            for _codec_tag in (b'avc1', b'avc3'):
                if find_atom(data, stsd[0] + 16, stsd[0] + stsd[1], _codec_tag):
                    raise Exception(
                        f"hvc1 tag injection requires HEVC output, but this file is "
                        f"H.264 ({_codec_tag.decode()}). Re-export with H.265/HEVC.")
            raise Exception("hvc1/hev1 sample description not found in video track")
        hvc1_off, hvc1_sz = hvc1

        # Strip existing VR metadata atoms (st3d, sv3d, vexu, hfov) from hvc1
        # to avoid conflicting signals (Spherical V2 sv3d says 360° equirect,
        # which overrides APMP halfEquirect on Vision Pro).
        # Rebuild hvc1 sub-atom list, keeping everything except VR atoms.
        _strip_tags = {b'st3d', b'sv3d', b'vexu', b'hfov'}
        _hvc1_header = data[hvc1_off:hvc1_off + 86]  # 4(size)+4(tag)+78(video sample entry)
        _kept = bytearray()
        _pos = hvc1_off + 86
        _end = hvc1_off + hvc1_sz
        while _pos < _end - 8:
            _asz = struct.unpack('>I', data[_pos:_pos+4])[0]
            _atag = data[_pos+4:_pos+8]
            if _asz < 8 or _pos + _asz > _end:
                break
            if bytes(_atag) not in _strip_tags:
                _kept.extend(data[_pos:_pos+_asz])
            _pos += _asz

        # Build new hvc1: header + kept atoms + APMP inject
        _new_hvc1_body = bytes(_kept) + inject_data
        _new_hvc1_sz = 86 + len(_new_hvc1_body)
        _new_hvc1 = struct.pack('>I', _new_hvc1_sz) + _hvc1_header[4:86] + _new_hvc1_body
        _size_delta = _new_hvc1_sz - hvc1_sz

        # Replace hvc1 in data
        data[hvc1_off:hvc1_off + hvc1_sz] = _new_hvc1

        # Update sizes up the parent chain
        for off, _ in [stsd, stbl, minf, mdia, trak, moov]:
            old_sz = struct.unpack('>I', data[off:off+4])[0]
            struct.pack_into('>I', data, off, old_sz + _size_delta)
        inject_size = _size_delta  # for stco update below

        # If moov is before mdat, chunk offsets need updating
        _moov_before_mdat = (_moov_off + _moov_sz < _fsize)
        if inject_size != 0 and _moov_before_mdat:
            for tag in [b'stco', b'co64']:
                search_start = 0
                while True:
                    idx = data.find(tag, search_start)
                    if idx < 4 or idx >= len(data):
                        break
                    atom_sz = struct.unpack('>I', data[idx-4:idx])[0]
                    if tag == b'stco':
                        n_entries = struct.unpack('>I', data[idx+8:idx+12])[0]
                        for e in range(n_entries):
                            eoff = idx + 12 + e * 4
                            old_val = struct.unpack('>I', data[eoff:eoff+4])[0]
                            struct.pack_into('>I', data, eoff, old_val + inject_size)
                    else:  # co64
                        n_entries = struct.unpack('>I', data[idx+8:idx+12])[0]
                        for e in range(n_entries):
                            eoff = idx + 12 + e * 8
                            old_val = struct.unpack('>Q', data[eoff:eoff+8])[0]
                            struct.pack_into('>Q', data, eoff, old_val + inject_size)
                    search_start = idx + atom_sz

        # Write back moov only (memory-efficient for large files)
        _moov_at_end = (_moov_off + _moov_sz >= _fsize)
        if _moov_at_end:
            with open(str(video_path), 'r+b') as f:
                f.seek(_moov_off)
                f.write(data)
                f.truncate()
        else:
            with open(str(video_path), 'rb') as f:
                f.seek(_moov_off + _moov_sz)
                after_moov = f.read()
            with open(str(video_path), 'r+b') as f:
                f.seek(_moov_off)
                f.write(data)
                f.write(after_moov)
                f.truncate()

    # Note: _convert_to_mvhevc() was removed 2026-04-09. MV-HEVC export now
    # uses mvhevc_encode (Swift helper) piped directly from VideoProcessor.run()
    # above — see the _is_mvhevc_direct branch around line 7680. This
    # replaces the prior external `spatial` CLI dependency and eliminates
    # the second transcode (SBS HEVC → re-decode → MV-HEVC). The helper
    # reads raw BGR48LE SBS frames from stdin and produces a
    # VideoToolbox-direct MV-HEVC .mov with spatial metadata baked into
    # the format description. Audio is post-muxed with ffmpeg.

    def _inject_vr180_metadata(self, video_path: Path):
        """Inject VR180 metadata for YouTube using Google's Spatial Media method

        VR180 format requirements for YouTube:
        - Projection: equirectangular with 180° bounds
        - Stereo mode: left-right (side-by-side)
        - Proper sv3d and st3d boxes as per Spherical Video V2 spec
        """
        import tempfile
        import shutil
        from spatialmedia import metadata_utils

        self.status.emit("Injecting VR180 metadata for YouTube...")

        # Create metadata object exactly as spatial-media does it
        metadata = metadata_utils.Metadata()

        # Set stereo mode to left-right (side-by-side)
        metadata.stereo = "left-right"

        # Set spherical projection to equirectangular
        metadata.spherical = "equirectangular"

        # For 180° VR (not 360°), set clip_left_right to non-zero
        # This is critical for YouTube to recognize it as VR180
        # Value from spatial-media: 1073741823 for 180°, 0 for 360°
        metadata.clip_left_right = 1073741823

        # Set orientation (default: no rotation)
        metadata.orientation = {"yaw": 0, "pitch": 0, "roll": 0}

        # Create a temporary output file
        temp_file = Path(tempfile.mktemp(suffix='.mov'))

        try:
            # Use spatial-media's inject_metadata function
            # This properly injects both st3d and sv3d boxes into the MP4
            def console_output(msg):
                # Relay spatial-media messages to status
                if msg and not msg.startswith("\t"):
                    self.status.emit(msg)

            metadata_utils.inject_metadata(
                str(video_path),
                str(temp_file),
                metadata,
                console_output
            )

            # Replace original with metadata-injected version
            shutil.move(str(temp_file), str(video_path))
            self.status.emit("✓ VR180 metadata injected successfully")

        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise Exception(f"VR180 metadata injection failed: {str(e)}")


class TrimSlider(QSlider):
    """Custom slider that shows trim range visualization"""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.trim_start = 0.0  # 0.0 to 1.0 (normalized)
        self.trim_end = 1.0    # 0.0 to 1.0 (normalized)
        self._has_trim = False

    def setTrimRange(self, start, end, has_trim=True):
        """Set trim range (normalized 0.0 to 1.0)"""
        self.trim_start = max(0.0, min(1.0, start))
        self.trim_end = max(0.0, min(1.0, end))
        self._has_trim = has_trim
        self.update()

    def clearTrim(self):
        """Clear trim visualization"""
        self.trim_start = 0.0
        self.trim_end = 1.0
        self._has_trim = False
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
        from PyQt6.QtCore import QRect

        # Call the parent paint event first to draw the slider
        super().paintEvent(event)

        # Draw the trim range overlay on top
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get the groove area
        groove_margin = 8  # Margin from edges for slider handle
        bar_height = 16  # Height of the trim visualization bar
        bar_y = (self.height() - bar_height) // 2

        # Draw trim range highlight
        if self._has_trim and self.trim_end > self.trim_start:
            available_width = self.width() - 2 * groove_margin
            start_x = groove_margin + int(self.trim_start * available_width)
            end_x = groove_margin + int(self.trim_end * available_width)

            # Draw excluded areas (darker, more visible)
            excluded_color = QColor(0, 0, 0, 150)  # Dark semi-transparent

            # Left excluded area
            if start_x > groove_margin:
                painter.fillRect(QRect(groove_margin, bar_y, start_x - groove_margin, bar_height), excluded_color)

            # Right excluded area
            if end_x < self.width() - groove_margin:
                painter.fillRect(QRect(end_x, bar_y, self.width() - groove_margin - end_x, bar_height), excluded_color)

            # Draw trim range with colored border (selected area)
            trim_fill = QColor(76, 175, 80, 60)  # Light green fill
            trim_border = QColor(76, 175, 80, 255)  # Solid green border

            # Fill the selected area
            painter.fillRect(QRect(start_x, bar_y, end_x - start_x, bar_height), trim_fill)

            # Draw border around selected area
            pen = QPen(trim_border, 2)
            painter.setPen(pen)
            painter.drawRect(QRect(start_x, bar_y, end_x - start_x, bar_height))

            # Draw trim markers (thick vertical lines at in/out points)
            marker_pen = QPen(QColor(255, 193, 7), 3)  # Yellow/gold markers
            painter.setPen(marker_pen)
            painter.drawLine(start_x, bar_y - 4, start_x, bar_y + bar_height + 4)
            painter.drawLine(end_x, bar_y - 4, end_x, bar_y + bar_height + 4)

        painter.end()


class PreviewWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #e0e0e0; border: 2px solid #cccccc; border-radius: 4px; }")
        self.processed_pixmap = None
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.dragging = False
        self.last_mouse_pos = None

    def set_frame(self, pixmap: QPixmap):
        self.processed_pixmap = pixmap
        # Don't reset pan offset - preserve user's pan/zoom position
        self._update_display()

    def _update_display(self):
        if self.processed_pixmap:
            scaled = self.processed_pixmap.scaled(self.size() * self.zoom_level, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            if self.zoom_level > 1.0:
                # When zoomed in, create a canvas and draw the image with offset
                canvas = QPixmap(self.size())
                canvas.fill(Qt.GlobalColor.gray)

                painter = QPainter(canvas)

                # Calculate centered position with pan offset
                x = (self.width() - scaled.width()) // 2 + self.pan_offset_x
                y = (self.height() - scaled.height()) // 2 + self.pan_offset_y

                painter.drawPixmap(x, y, scaled)
                painter.end()

                self.setPixmap(canvas)
            else:
                self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: self.zoom_level = min(4.0, self.zoom_level * 1.1)
        else: self.zoom_level = max(0.25, self.zoom_level / 1.1)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_level > 1.0:
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset_x += delta.x()
            self.pan_offset_y += delta.y()
            self.last_mouse_pos = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None
            if self.zoom_level > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def enterEvent(self, event):
        if self.zoom_level > 1.0:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)


class SliderWithSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, min_val, max_val, default=0.0, decimals=1, step=0.1, suffix=""):
        super().__init__()
        self.multiplier = 10 ** decimals
        self.default_value = default
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setSingleStep(step)
        self.spinbox.setSuffix(suffix)
        self.spinbox.setFixedWidth(100)
        # Ensure up/down buttons are visible and functional
        self.spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)

        self.reset_btn = QToolButton()
        self.reset_btn.setText("⟲")
        self.reset_btn.setFixedSize(24, 24)

        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spinbox)
        layout.addWidget(self.reset_btn)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
        self.reset_btn.clicked.connect(lambda: self.setValue(self.default_value))
    
    def _slider_changed(self, value):
        if not self._updating:
            self._updating = True
            self.spinbox.setValue(value / self.multiplier)
            self.valueChanged.emit(value / self.multiplier)
            self._updating = False
    
    def _spinbox_changed(self, value):
        if not self._updating:
            self._updating = True
            self.slider.setValue(int(value * self.multiplier))
            self.valueChanged.emit(value)
            self._updating = False
    
    def value(self): return self.spinbox.value()
    
    def setValue(self, value):
        self._updating = True
        self.spinbox.setValue(value)
        self.slider.setValue(int(value * self.multiplier))
        self._updating = False
        self.valueChanged.emit(value)


class IntSliderWithSpinBox(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, min_val, max_val, default=0, suffix=""):
        super().__init__()
        self.default_value = default
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)

        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default)
        self.spinbox.setSuffix(suffix)
        self.spinbox.setFixedWidth(120)

        self.reset_btn = QToolButton()
        self.reset_btn.setText("⟲")
        self.reset_btn.setFixedSize(24, 24)

        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spinbox)
        layout.addWidget(self.reset_btn)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
        self.reset_btn.clicked.connect(lambda: self.setValue(self.default_value))
    
    def _slider_changed(self, value):
        if not self._updating:
            self._updating = True
            self.spinbox.setValue(value)
            self.valueChanged.emit(value)
            self._updating = False
    
    def _spinbox_changed(self, value):
        if not self._updating:
            self._updating = True
            self.slider.setValue(value)
            self.valueChanged.emit(value)
            self._updating = False
    
    def value(self): return self.spinbox.value()
    
    def setValue(self, value):
        self._updating = True
        self.spinbox.setValue(value)
        self.slider.setValue(value)
        self._updating = False
        self.valueChanged.emit(value)


class VR180ProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = ProcessingConfig()
        self.original_frame = None
        self.cached_raw_frame = None  # Cache unprocessed frame for instant adjustments
        self.cached_timestamp = None  # Timestamp of cached frame
        self._gyro_stabilizer = None  # GyroStabilizer instance for preview
        self.video_duration = 0.0
        self.preview_timestamp = 0.0
        self.current_eye = "left"  # Track which eye is shown in Single Eye Mode
        self.trim_start = 0.0  # Trim start time in seconds
        self.trim_end = 0.0  # Trim end time in seconds (0 = full duration)

        # Settings for persistence
        from PyQt6.QtCore import QSettings
        self.settings = QSettings("VR180Processor", "VR180Processor")

        # Initialize preview timer before UI (needed for _load_settings)
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._on_preview_timer)

        # Enable drag and drop
        self.setAcceptDrops(True)

        self._init_ui()
        self._apply_styles()
        self._connect_signals()
        self._load_settings()
    
    def _init_ui(self):
        self.setWindowTitle("VR180 Silver Bullet V1.1")
        self.setMinimumSize(1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout(file_group)
        file_layout.addWidget(QLabel("Input:"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        file_layout.addWidget(self.input_path_edit, stretch=1)
        self.browse_input_btn = QPushButton("Browse...")
        file_layout.addWidget(self.browse_input_btn)
        file_layout.addSpacing(20)
        file_layout.addWidget(QLabel("Output:"))
        self.output_path_edit = QLineEdit()
        file_layout.addWidget(self.output_path_edit, stretch=1)
        self.browse_output_btn = QPushButton("Browse...")
        file_layout.addWidget(self.browse_output_btn)
        main_layout.addWidget(file_group)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Preview section
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Preview Mode:"))
        self.preview_mode_combo = QComboBox()
        for mode in PreviewMode:
            self.preview_mode_combo.addItem(mode.value, mode)
        self.preview_mode_combo.setCurrentIndex(1)
        mode_layout.addWidget(self.preview_mode_combo)

        # Add eye toggle for Single Eye Mode
        self.eye_toggle_btn = QPushButton("Switch Eyes")
        self.eye_toggle_btn.setFixedWidth(120)
        self.eye_toggle_btn.setVisible(False)  # Hidden by default
        mode_layout.addWidget(self.eye_toggle_btn)

        self.eye_label = QLabel("L")
        self.eye_label.setFixedWidth(20)
        self.eye_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eye_label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        self.eye_label.setVisible(False)  # Hidden by default
        mode_layout.addWidget(self.eye_label)

        mode_layout.addStretch()
        mode_layout.addWidget(QLabel("Zoom:"))
        self.zoom_out_btn = QPushButton("Out")
        self.zoom_out_btn.setFixedWidth(60)
        mode_layout.addWidget(self.zoom_out_btn)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        mode_layout.addWidget(self.zoom_label)
        self.zoom_in_btn = QPushButton("In")
        self.zoom_in_btn.setFixedWidth(60)
        mode_layout.addWidget(self.zoom_in_btn)
        self.zoom_reset_btn = QPushButton("Reset")
        mode_layout.addWidget(self.zoom_reset_btn)
        preview_layout.addLayout(mode_layout)
        
        self.preview_widget = PreviewWidget()
        preview_layout.addWidget(self.preview_widget, stretch=1)
        
        timeline_layout = QHBoxLayout()
        self.timeline_slider = TrimSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.setTracking(False)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        timeline_layout.addWidget(self.timeline_slider, stretch=1)
        timeline_layout.addWidget(self.time_label)
        preview_layout.addLayout(timeline_layout)

        # Trim controls
        trim_layout = QHBoxLayout()
        trim_layout.setSpacing(4)
        trim_layout.setContentsMargins(0, 0, 0, 0)

        # In point - let button auto-size to fit text
        self.set_in_btn = QPushButton("In")
        self.set_in_btn.setToolTip("Set trim start point at current position (shortcut: I)")
        trim_layout.addWidget(self.set_in_btn)

        self.in_time_edit = QLineEdit("00:00:00.000")
        self.in_time_edit.setFixedWidth(90)
        self.in_time_edit.setToolTip("Trim start time (format: HH:MM:SS.mmm or seconds)")
        trim_layout.addWidget(self.in_time_edit)

        dash_label = QLabel("-")
        dash_label.setFixedWidth(10)
        dash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trim_layout.addWidget(dash_label)

        # Out point - let button auto-size to fit text
        self.out_time_edit = QLineEdit("00:00:00.000")
        self.out_time_edit.setFixedWidth(90)
        self.out_time_edit.setToolTip("Trim end time (format: HH:MM:SS.mmm or seconds)")
        trim_layout.addWidget(self.out_time_edit)

        self.set_out_btn = QPushButton("Out")
        self.set_out_btn.setToolTip("Set trim end point at current position (shortcut: O)")
        trim_layout.addWidget(self.set_out_btn)

        # Clear trim - let button auto-size to fit text
        self.clear_trim_btn = QPushButton("Clear")
        self.clear_trim_btn.setToolTip("Clear trim points (use full video)")
        trim_layout.addWidget(self.clear_trim_btn)

        # Duration label
        self.trim_duration_label = QLabel("--:--")
        self.trim_duration_label.setFixedWidth(50)
        trim_layout.addWidget(self.trim_duration_label)

        trim_layout.addStretch()
        preview_layout.addLayout(trim_layout)

        splitter.addWidget(preview_container)
        
        # Controls section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(16)
        
        # Shift
        shift_group = QGroupBox("Global Horizontal Shift")
        shift_layout = QVBoxLayout(shift_group)
        shift_layout.addWidget(QLabel("Fix split-eye frames by shifting horizontally."))
        self.global_shift_slider = IntSliderWithSpinBox(-3840, 3840, 0, " px")
        shift_layout.addWidget(self.global_shift_slider)
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick:"))
        for v in [-1920, 0, 1920]:
            btn = QPushButton(f"{v:+d}" if v else "0")
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda c, val=v: self.global_shift_slider.setValue(val))
            quick_layout.addWidget(btn)
        quick_layout.addStretch()
        shift_layout.addLayout(quick_layout)
        controls_layout.addWidget(shift_group)
        
        # Global adjustment
        global_group = QGroupBox("Global Panomap Adjustment")
        global_layout = QGridLayout(global_group)
        global_layout.addWidget(QLabel("Yaw:"), 0, 0)
        self.global_yaw = SliderWithSpinBox(-180, 180, 0, 3, 0.1, "°")
        global_layout.addWidget(self.global_yaw, 0, 1)
        global_layout.addWidget(QLabel("Pitch:"), 1, 0)
        self.global_pitch = SliderWithSpinBox(-90, 90, 0, 3, 0.1, "°")
        global_layout.addWidget(self.global_pitch, 1, 1)
        global_layout.addWidget(QLabel("Roll:"), 2, 0)
        self.global_roll = SliderWithSpinBox(-45, 45, 0, 3, 0.1, "°")
        global_layout.addWidget(self.global_roll, 2, 1)
        self.upside_down_checkbox = QCheckBox("Camera mounted upside down")
        self.upside_down_checkbox.setToolTip("Enable when the camera is mounted upside down.\nRotates output 180° and inverts gravity for horizon lock.")
        global_layout.addWidget(self.upside_down_checkbox, 3, 0, 1, 2)
        controls_layout.addWidget(global_group)

        # Stereo offset
        offset_group = QGroupBox("Stereo Offset (Applied Oppositely)")
        offset_layout = QGridLayout(offset_group)
        offset_layout.addWidget(QLabel("Yaw:"), 0, 0)
        self.stereo_yaw_offset = SliderWithSpinBox(-10, 10, 0, 3, 0.1, "°")
        offset_layout.addWidget(self.stereo_yaw_offset, 0, 1)
        offset_layout.addWidget(QLabel("Pitch:"), 1, 0)
        self.stereo_pitch_offset = SliderWithSpinBox(-10, 10, 0, 3, 0.1, "°")
        offset_layout.addWidget(self.stereo_pitch_offset, 1, 1)
        offset_layout.addWidget(QLabel("Roll:"), 2, 0)
        self.stereo_roll_offset = SliderWithSpinBox(-10, 10, 0, 3, 0.1, "°")
        offset_layout.addWidget(self.stereo_roll_offset, 2, 1)
        controls_layout.addWidget(offset_group)

        # GoPro Gyro Stabilization section
        gopro_group = QGroupBox("GoPro Settings")
        gopro_layout = QVBoxLayout(gopro_group)
        self.gopro_status = QLabel("Gyro data auto-loaded from .360 input")
        self.gopro_status.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        gopro_layout.addWidget(self.gopro_status)

        # Output format (equirect vs fisheye, .360 only)
        _fmt_row = QWidget()
        _fmt_layout = QHBoxLayout(_fmt_row)
        _fmt_layout.setContentsMargins(0, 0, 0, 0)
        _fmt_layout.addWidget(QLabel("Output:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["Equirectangular SBS", "Fisheye SBS"])
        self.output_format_combo.setToolTip(
            "Output projection format.\n"
            "Equirectangular SBS: standard VR180 format for headsets/YouTube.\n"
            "Fisheye SBS: reconstructed circular fisheye (raw sensor view, 4216×4216 per eye).\n"
            "Fisheye mode outputs raw sensor data without stabilization or RS correction."
        )
        self.output_format_combo.currentIndexChanged.connect(self._schedule_preview_update)
        _fmt_layout.addWidget(self.output_format_combo)
        _fmt_layout.addStretch()
        _fmt_row.setVisible(False)
        self._output_format_row = _fmt_row
        gopro_layout.addWidget(_fmt_row)

        # "Load .360 Gyro" button row — only visible for non-.360 inputs
        self.gyro_load_row = QWidget()
        _gyro_btn_layout = QHBoxLayout(self.gyro_load_row)
        _gyro_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.load_gyro_360_btn = QPushButton("Load .360 Gyro...")
        self.load_gyro_360_btn.setToolTip("Load gyro stabilization data from a matching .360 file")
        _gyro_btn_layout.addWidget(self.load_gyro_360_btn)
        self.clear_gyro_btn = QPushButton("Clear")
        self.clear_gyro_btn.setToolTip("Clear loaded gyro data")
        _gyro_btn_layout.addWidget(self.clear_gyro_btn)
        _gyro_btn_layout.addStretch()
        self.gyro_load_row.setVisible(False)  # shown when non-.360 input loaded
        gopro_layout.addWidget(self.gyro_load_row)

        self.gyro_stabilize_checkbox = QCheckBox("Enable Gyro Stabilization")
        if HAS_SCIPY:
            self.gyro_stabilize_checkbox.setToolTip("Use CORI (Camera Orientation) data to smooth camera motion.\nRequires .360 file to be loaded.")
        else:
            self.gyro_stabilize_checkbox.setToolTip("Requires scipy package. Install with: pip install scipy")
        self.gyro_stabilize_checkbox.setChecked(False)
        self.gyro_stabilize_checkbox.setEnabled(False)
        gopro_layout.addWidget(self.gyro_stabilize_checkbox)

        # Gyro controls container (shown/hidden based on checkbox)
        self.gyro_controls_widget = QWidget()
        gyro_grid = QGridLayout(self.gyro_controls_widget)
        gyro_grid.setContentsMargins(0, 0, 0, 0)

        # Smoothing window slider (all axes)
        gyro_grid.addWidget(QLabel("Smooth:"), 0, 0)
        self.gyro_smooth_slider = SliderWithSpinBox(0, 5000, 1000, 0, 50, " ms")
        self.gyro_smooth_slider.setToolTip("Stabilization smoothing window. 0 = camera lock.\nHigher = more stabilization (removes shake, follows slow pans).")
        gyro_grid.addWidget(self.gyro_smooth_slider, 0, 1)

        # Max correction angle slider
        gyro_grid.addWidget(QLabel("Max Corr:"), 1, 0)
        self.gyro_max_corr_slider = SliderWithSpinBox(1, 30, 15, 0, 1, "°")
        self.gyro_max_corr_slider.setToolTip("Maximum heading correction angle (degrees).\nSoft elastic limit — smoothly resists beyond this angle.\n15° recommended. Lower = fewer borders but less stabilization.")
        gyro_grid.addWidget(self.gyro_max_corr_slider, 1, 1)

        # Responsiveness slider (velocity curve power)
        gyro_grid.addWidget(QLabel("Response:"), 2, 0)
        self.gyro_responsiveness_slider = SliderWithSpinBox(0.2, 3.0, 1.0, 1, 0.1, "")
        self.gyro_responsiveness_slider.setToolTip(
            "How quickly the camera follows motion (velocity response curve).\n"
            "Lower = starts following early, eases in gradually (anticipatory).\n"
            "Higher = holds still longer, then catches up (cinematic lag).\n"
            "Default 1.0 = linear response.")
        gyro_grid.addWidget(self.gyro_responsiveness_slider, 2, 1)

        # Horizon lock checkbox
        self.gyro_horizon_lock_checkbox = QCheckBox("Horizon Lock")
        self.gyro_horizon_lock_checkbox.setToolTip("Lock horizon level using accelerometer data.\nPrevents roll drift during panning.")
        self.gyro_horizon_lock_checkbox.setChecked(False)
        gyro_grid.addWidget(self.gyro_horizon_lock_checkbox, 3, 0, 1, 2)

        self.gyro_controls_widget.setVisible(False)
        gopro_layout.addWidget(self.gyro_controls_widget)

        # RS correction checkbox (independent — can be used without gyro stabilization)
        # When gyro stabilization is on, RS is forced on automatically.
        self.rs_correction_checkbox = QCheckBox("Enable Rolling Shutter Correction")
        self.rs_correction_checkbox.setToolTip(
            "Apply rolling shutter correction to the right eye.\n"
            "Requires orientation data to be loaded.\n"
            "Automatically enabled when gyro stabilization is on."
        )
        self.rs_correction_checkbox.setChecked(False)
        self.rs_correction_checkbox.setEnabled(False)
        self.rs_correction_checkbox.setVisible(False)
        gopro_layout.addWidget(self.rs_correction_checkbox)

        # ── Advanced Settings (collapsible) ──
        self.advanced_toggle_btn = QToolButton()
        self.advanced_toggle_btn.setText("Advanced Settings")
        self.advanced_toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.setChecked(False)
        self.advanced_toggle_btn.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; color: #555; padding: 4px 0; }"
            "QToolButton:hover { color: #0066cc; }"
        )
        gopro_layout.addWidget(self.advanced_toggle_btn)

        # Advanced settings container (RS controls, hidden by default)
        self.advanced_settings_widget = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_settings_widget)
        advanced_layout.setContentsMargins(8, 0, 0, 0)

        # RS controls container
        self.rs_controls_widget = QWidget()
        rs_grid = QGridLayout(self.rs_controls_widget)
        rs_grid.setContentsMargins(0, 0, 0, 0)

        # Scan time (readout time) slider
        rs_grid.addWidget(QLabel("Scan Time:"), 0, 0)
        self.rs_correction_slider = SliderWithSpinBox(0, 33.0, 15.2, 1, 0.5, " ms")
        self.rs_correction_slider.setToolTip(
            "Sensor readout (scan) time in milliseconds.\n"
            "Typical values: 10-16ms for GoPro MAX. 0 = disabled."
        )
        rs_grid.addWidget(self.rs_correction_slider, 0, 1)

        # Yaw RS factor slider (pan around body-Z → horizontal shear)
        rs_grid.addWidget(QLabel("Yaw Factor:"), 1, 0)
        self.rs_factor_slider = SliderWithSpinBox(-4.0, 4.0, 0.0, 2, 0.01, "")
        self.rs_factor_slider.setToolTip(
            "Yaw RS correction factor (pan around body-Z → horizontal shear).\n"
            "Negated internally. Positive = correct. Default = 0."
        )
        rs_grid.addWidget(self.rs_factor_slider, 1, 1)

        # Pitch RS factor slider (tilt around body-X → vertical shear)
        rs_grid.addWidget(QLabel("Pitch Factor:"), 2, 0)
        self.rs_pitch_factor_slider = SliderWithSpinBox(-4.0, 4.0, 2.0, 2, 0.01, "")
        self.rs_pitch_factor_slider.setToolTip(
            "Pitch RS correction factor (tilt around body-X → vertical shear).\n"
            "Positive = cancel firmware's wrong RS. Default = 2.0."
        )
        rs_grid.addWidget(self.rs_pitch_factor_slider, 2, 1)

        # Roll RS factor slider (roll around body-Y → rotational shear)
        rs_grid.addWidget(QLabel("Roll Factor:"), 3, 0)
        self.rs_roll_factor_slider = SliderWithSpinBox(-4.0, 4.0, 2.0, 2, 0.01, "")
        self.rs_roll_factor_slider.setToolTip(
            "Roll RS correction factor (roll around body-Y → rotational shear).\n"
            "Positive = cancel firmware's wrong RS. Default = 2.0."
        )
        rs_grid.addWidget(self.rs_roll_factor_slider, 3, 1)

        self.rs_use_cori_checkbox = QCheckBox("Use CORI angular velocity (for vibrating platforms)")
        self.rs_use_cori_checkbox.setToolTip(
            "Use CORI quaternion delta for RS angular velocity instead of 800Hz gyro.\n"
            "Enable for drone/plane footage where engine vibration corrupts the gyro.\n"
            "CORI is the firmware's own orientation output — immune to vibration noise.\n"
            "Lower temporal resolution (30fps vs 800Hz) but correct magnitude."
        )
        self.rs_use_cori_checkbox.setChecked(False)
        rs_grid.addWidget(self.rs_use_cori_checkbox, 4, 0, 1, 2)

        self.rs_no_firmware_checkbox = QCheckBox("No Firmware RS (both eyes)")
        self.rs_no_firmware_checkbox.setToolTip(
            "Force no-firmware rolling shutter mode.\n"
            "When enabled, RS correction is applied to BOTH eyes with factors (1,1,1)\n"
            "instead of only the right eye with factors (0,2,2).\n\n"
            "This is auto-detected for .360 files with disabled firmware ERS,\n"
            "but may need manual override when loading later segments (GS02+)\n"
            "without the first segment file available."
        )
        self.rs_no_firmware_checkbox.setChecked(False)
        rs_grid.addWidget(self.rs_no_firmware_checkbox, 5, 0, 1, 2)

        advanced_layout.addWidget(self.rs_controls_widget)
        self.advanced_settings_widget.setVisible(False)
        gopro_layout.addWidget(self.advanced_settings_widget)

        controls_layout.addWidget(gopro_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel("Codec:"), 0, 0)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["Auto", "H.265", "ProRes"])
        output_layout.addWidget(self.codec_combo, 0, 1)

        # H.265 quality settings (CRF)
        self.quality_label = QLabel("Quality (CRF):")
        output_layout.addWidget(self.quality_label, 1, 0)
        self.quality_spinbox = QSpinBox()
        self.quality_spinbox.setRange(0, 51)
        self.quality_spinbox.setValue(18)
        self.quality_spinbox.setToolTip("Lower = better quality, 18 = visually lossless")
        output_layout.addWidget(self.quality_spinbox, 1, 1)

        # H.265 bitrate settings (mutually exclusive with quality)
        self.bitrate_label = QLabel("Bitrate (Mbps):")
        output_layout.addWidget(self.bitrate_label, 2, 0)
        self.bitrate_spinbox = QSpinBox()
        self.bitrate_spinbox.setRange(1, 700)
        self.bitrate_spinbox.setValue(200)
        self.bitrate_spinbox.setEnabled(True)
        self.bitrate_spinbox.setToolTip("Target bitrate in Mbps")
        output_layout.addWidget(self.bitrate_spinbox, 2, 1)

        # Radio buttons for quality vs bitrate
        self.use_crf_radio = QRadioButton("Use Quality (CRF)")
        self.use_crf_radio.setChecked(False)
        output_layout.addWidget(self.use_crf_radio, 3, 0, 1, 2)

        self.use_bitrate_radio = QRadioButton("Use Bitrate")
        self.use_bitrate_radio.setChecked(True)
        output_layout.addWidget(self.use_bitrate_radio, 4, 0, 1, 2)

        self.encoding_mode_group = QButtonGroup()
        self.encoding_mode_group.addButton(self.use_crf_radio)
        self.encoding_mode_group.addButton(self.use_bitrate_radio)

        # ProRes settings
        self.prores_label = QLabel("ProRes:")
        output_layout.addWidget(self.prores_label, 5, 0)
        self.prores_combo = QComboBox()
        self.prores_combo.addItems(["Proxy", "LT", "Standard", "HQ", "4444", "4444 XQ"])
        self.prores_combo.setCurrentIndex(2)  # Standard
        output_layout.addWidget(self.prores_combo, 5, 1)

        # Encoder selection dropdown
        output_layout.addWidget(QLabel("Encoder:"), 6, 0)
        self.encoder_combo = QComboBox()
        if sys.platform == 'darwin':
            self.encoder_combo.addItems(["VideoToolbox (Hardware)", "Software (x264/x265)"])
            self._encoder_map = ["videotoolbox", "software"]
        else:
            # Windows/Linux - show all options
            self.encoder_combo.addItems(["Auto-detect", "NVIDIA NVENC", "Intel QuickSync", "AMD AMF", "Software (x264/x265)"])
            self._encoder_map = ["auto", "nvenc", "qsv", "amf", "software"]
        self.encoder_combo.setToolTip("Select hardware encoder:\n• Auto-detect: Automatically select best available\n• NVIDIA NVENC: For NVIDIA GPUs\n• Intel QuickSync: For Intel CPUs with integrated GPU\n• AMD AMF: For AMD GPUs\n• Software: CPU-based encoding (slower but always works)")
        output_layout.addWidget(self.encoder_combo, 6, 1)

        # H.265 bit depth
        self.h265_bit_depth_label = QLabel("H.265 Bit Depth:")
        output_layout.addWidget(self.h265_bit_depth_label, 7, 0)
        self.h265_bit_depth_combo = QComboBox()
        self.h265_bit_depth_combo.addItems(["8-bit", "10-bit"])
        self.h265_bit_depth_combo.setToolTip("8-bit: Standard compatibility\n10-bit: Higher quality, better gradients")
        output_layout.addWidget(self.h265_bit_depth_combo, 7, 1)

        # Audio output option (visible for .360 input only)
        self.audio_format_label = QLabel("Audio:")
        self.audio_format_combo = QComboBox()
        self.audio_format_combo.addItem("Stereo AAC", "stereo")
        self.audio_format_combo.addItem("Ambisonic (4ch spatial)", "ambisonic")
        self.audio_format_combo.setCurrentIndex(0)  # stereo default
        self.audio_format_combo.setToolTip(
            "Stereo AAC: standard 2-channel audio.\n"
            "Ambisonic: first-order ambisonics (4ch PCM) from .360 file."
        )
        self.audio_format_label.setVisible(False)
        self.audio_format_combo.setVisible(False)
        output_layout.addWidget(self.audio_format_label, 8, 0)
        output_layout.addWidget(self.audio_format_combo, 8, 1)

        # Output resolution (for .360 EAC→equirect)
        self.resolution_label = QLabel("Resolution:")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["8192x4096", "8640x4320 for YT", "7680x3840"])
        self.resolution_combo.setToolTip(
            "Output equirectangular resolution (SBS: each eye = half width × height).\n"
            "8192×4096: default — uses all available .360 sensor pixels.\n"
            "8640×4320 for YT: YouTube 8K upload resolution (software encode on Windows).\n"
            "7680×3840: standard 8K VR180 (slightly faster processing)."
        )
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.resolution_label.setVisible(False)
        self.resolution_combo.setVisible(False)
        output_layout.addWidget(self.resolution_label, 9, 0)
        output_layout.addWidget(self.resolution_combo, 9, 1)

        # Temporal denoise (VideoToolbox, macOS 26+ only)
        output_layout.addWidget(QLabel("Denoise:"), 10, 0)
        self.denoise_slider = SliderWithSpinBox(0, 100, 0, decimals=0, step=1, suffix="%")
        self.denoise_slider.setToolTip(
            "VideoToolbox temporal noise filter (macOS 26+).\n"
            "Uses past & future frames for hardware-accelerated noise reduction.\n"
            "0% = off, 50% = moderate, 100% = maximum.\n"
            "Applied before all other processing for best quality."
        )
        if sys.platform != 'darwin':
            self.denoise_slider.setEnabled(False)
            self.denoise_slider.setToolTip("Temporal denoise requires macOS 26+")
        output_layout.addWidget(self.denoise_slider, 10, 1)

        # Equirectangular-aware sharpening
        output_layout.addWidget(QLabel("Sharpen:"), 11, 0)
        self.sharpen_slider = SliderWithSpinBox(0, 300, 0, decimals=0, step=1, suffix="%")
        self.sharpen_slider.setToolTip(
            "Equirectangular-aware unsharp mask.\n"
            "Adapts to latitude: full sharpening at equator, reduced at poles.\n"
            "0% = off, 50% = subtle, 100% = moderate, 200% = strong, 300% = extreme."
        )
        self.sharpen_slider.valueChanged.connect(self._schedule_preview_update)
        output_layout.addWidget(self.sharpen_slider, 11, 1)

        # Sharpen radius (sigma)
        output_layout.addWidget(QLabel("Sharpen Radius:"), 12, 0)
        self.sharpen_radius_slider = SliderWithSpinBox(5, 50, 13, decimals=0, step=1, suffix="")
        self.sharpen_radius_slider.setToolTip(
            "Blur radius for unsharp mask (×0.1 = sigma).\n"
            "5 = fine detail (0.5σ), 15 = default (1.5σ), 50 = coarse structure (5.0σ)."
        )
        self.sharpen_radius_slider.valueChanged.connect(self._schedule_preview_update)
        output_layout.addWidget(self.sharpen_radius_slider, 12, 1)

        # Pre-LUT color adjustments (ASC CDL: Lift, Gamma, Gain)
        output_layout.addWidget(QLabel("Lift:"), 13, 0)
        self.lift_slider = SliderWithSpinBox(-100, 100, 0, decimals=0, step=1, suffix="")
        self.lift_slider.setToolTip("Lift (Offset): Raises/lowers black level, affects shadows most\nNegative = crush blacks, 0 = neutral, Positive = lift shadows")
        output_layout.addWidget(self.lift_slider, 13, 1)

        output_layout.addWidget(QLabel("Gamma:"), 14, 0)
        self.gamma_slider = SliderWithSpinBox(10, 300, 100, decimals=0, step=1, suffix="")
        self.gamma_slider.setToolTip("Gamma (Power): Adjusts midtones while preserving black/white points\n<100 = darker midtones, 100 = neutral, >100 = brighter midtones")
        output_layout.addWidget(self.gamma_slider, 14, 1)

        output_layout.addWidget(QLabel("Gain:"), 15, 0)
        self.gain_slider = SliderWithSpinBox(50, 200, 100, decimals=0, step=1, suffix="")
        self.gain_slider.setToolTip("Gain (Slope): Overall brightness multiplier, affects highlights most\n<100 = darker, 100 = neutral, >100 = brighter")
        output_layout.addWidget(self.gain_slider, 15, 1)

        output_layout.addWidget(QLabel("Saturation:"), 16, 0)
        self.saturation_slider = SliderWithSpinBox(0, 200, 100, decimals=0, step=1, suffix="")
        self.saturation_slider.setToolTip("Saturation: Controls color intensity\n0 = grayscale, 100 = neutral, 200 = double saturation")
        output_layout.addWidget(self.saturation_slider, 16, 1)

        # LUT file selection (spanning both columns for more space)
        output_layout.addWidget(QLabel("LUT File:"), 17, 0)
        lut_file_layout = QHBoxLayout()
        self.lut_path_edit = QLineEdit()
        self.lut_path_edit.setPlaceholderText("Optional: .cube LUT file")
        self.lut_path_edit.setReadOnly(True)
        lut_file_layout.addWidget(self.lut_path_edit, 1)
        self.lut_browse_btn = QPushButton("Browse")
        self.lut_browse_btn.setMinimumWidth(80)
        lut_file_layout.addWidget(self.lut_browse_btn)
        self.lut_clear_btn = QPushButton("Clear")
        self.lut_clear_btn.setMinimumWidth(70)
        lut_file_layout.addWidget(self.lut_clear_btn)
        output_layout.addLayout(lut_file_layout, 17, 1)

        # LUT intensity slider
        output_layout.addWidget(QLabel("LUT Intensity:"), 18, 0)
        self.lut_intensity_slider = SliderWithSpinBox(0, 100, 100, 0, 1, "%")
        self.lut_intensity_slider.setToolTip("Blend strength: 0% = no LUT, 100% = full LUT")
        output_layout.addWidget(self.lut_intensity_slider, 18, 1)

        # Edge mask controls
        output_layout.addWidget(QLabel("Mask Size:"), 19, 0)
        self.mask_size_slider = SliderWithSpinBox(50, 100, 100, decimals=0, step=1, suffix="%")
        self.mask_size_slider.setToolTip("Circular mask radius as % of half-width\n100% = no mask (full frame), lower = tighter crop circle")
        output_layout.addWidget(self.mask_size_slider, 19, 1)

        output_layout.addWidget(QLabel("Mask Feather:"), 20, 0)
        self.mask_feather_slider = SliderWithSpinBox(0, 50, 0, decimals=0, step=1, suffix="%")
        self.mask_feather_slider.setToolTip("Feather width as % of half-width\n0% = hard edge, higher = softer fade to black")
        output_layout.addWidget(self.mask_feather_slider, 20, 1)


        # VR180 metadata checkbox for YouTube
        self.vr180_metadata_checkbox = QCheckBox("Inject VR180 Metadata for YouTube")
        self.vr180_metadata_checkbox.setToolTip("Add VR180 spherical metadata for YouTube upload\n"
                                                 "Uses Spherical Video V2 specification (fast, no re-encode)")
        output_layout.addWidget(self.vr180_metadata_checkbox, 21, 0)

        self.inject_only_btn = QPushButton("Inject Only (Output)")
        self.inject_only_btn.setToolTip("Skip all processing — just inject VR180 YouTube metadata\n"
                                         "into the output file as-is")
        self.inject_only_btn.clicked.connect(self._inject_only_output)
        output_layout.addWidget(self.inject_only_btn, 21, 1)

        # Vision Pro / Apple compatibility mode (macOS only)
        if IS_MACOS:
            output_layout.addWidget(QLabel("Vision Pro Mode:"), 22, 0)
            self.vision_pro_combo = QComboBox()
            self.vision_pro_combo.addItems([
                "Standard (no special metadata)",
                "Vision Pro VR180 (APMP - fast)",
                "Vision Pro MV-HEVC (spatial video)"
            ])
            self.vision_pro_combo.setToolTip(
                "Standard: Normal video output\n"
                "VR180 APMP: Inject VR180 stereo metadata for Vision Pro (instant, no re-encode, visionOS 26+)\n"
                "MV-HEVC: Direct spatial video encode via VideoToolbox — single pass, no external tools. Requires macOS 26+."
            )
            output_layout.addWidget(self.vision_pro_combo, 22, 1)
        else:
            # Windows: Create dummy combo for compatibility
            self.vision_pro_combo = QComboBox()
            self.vision_pro_combo.addItem("Standard (no special metadata)")
            self.vision_pro_combo.setVisible(False)

        controls_layout.addWidget(output_group)
        
        controls_layout.addStretch()
        scroll.setWidget(controls_container)
        splitter.addWidget(scroll)
        splitter.setSizes([900, 500])
        main_layout.addWidget(splitter, stretch=1)
        
        # Process section
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)

        # Info panel (dark terminal-style text box)
        self.ffmpeg_output = QTextEdit()
        self.ffmpeg_output.setReadOnly(True)
        self.ffmpeg_output.setMaximumHeight(100)
        self.ffmpeg_output.setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 11px; background-color: #1e1e1e; color: #d4d4d4; }")
        self.ffmpeg_output.setPlaceholderText("Load a video file to see info here...")
        process_layout.addWidget(self.ffmpeg_output)

        btn_layout = QHBoxLayout()
        self.reset_all_btn = QPushButton("Reset All")
        btn_layout.addWidget(self.reset_all_btn)
        btn_layout.addStretch()
        self.process_btn = QPushButton("▶ Process Video")
        self.process_btn.setFixedHeight(40)
        self.process_btn.setFixedWidth(200)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        process_layout.addLayout(btn_layout)
        main_layout.addWidget(process_group)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QWidget { color: #1a1a1a; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 6px; margin-top: 12px; padding: 12px; background: #f8f8f8; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #0066cc; }
            QPushButton { background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 6px 16px; }
            QPushButton:hover { background: #e0e0e0; border-color: #0066cc; }
            QPushButton:disabled { background: #f5f5f5; color: #999; }
            QPushButton#processBtn { background: #0066cc; color: white; font-weight: bold; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: white; border: 1px solid #ccc; border-radius: 4px; padding: 6px; }
            QSlider::groove:horizontal { height: 6px; background: #ddd; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0066cc; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; background: #f0f0f0; }
            QProgressBar::chunk { background: #0066cc; border-radius: 3px; }
            QToolButton { background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; }
        """)
        self.process_btn.setObjectName("processBtn")
    
    def _connect_signals(self):
        self.browse_input_btn.clicked.connect(self._browse_input)
        self.load_gyro_360_btn.clicked.connect(self._browse_gyro_360)
        self.clear_gyro_btn.clicked.connect(self._clear_gopro_gyro_data)
        self.browse_output_btn.clicked.connect(self._browse_output)
        self.preview_mode_combo.currentIndexChanged.connect(self._schedule_preview_update)
        self.zoom_in_btn.clicked.connect(lambda: self._zoom(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self._zoom(0.8))
        self.zoom_reset_btn.clicked.connect(self._zoom_reset)
        self.timeline_slider.sliderMoved.connect(self._timeline_dragging)
        self.timeline_slider.valueChanged.connect(self._timeline_changed)
        self.global_shift_slider.valueChanged.connect(self._schedule_preview_update)
        self.global_yaw.valueChanged.connect(self._on_adjustment_changed)
        self.global_pitch.valueChanged.connect(self._on_adjustment_changed)
        self.global_roll.valueChanged.connect(self._on_adjustment_changed)
        self.upside_down_checkbox.stateChanged.connect(self._on_adjustment_changed)
        self.stereo_yaw_offset.valueChanged.connect(self._on_adjustment_changed)
        self.stereo_pitch_offset.valueChanged.connect(self._on_adjustment_changed)
        self.stereo_roll_offset.valueChanged.connect(self._on_adjustment_changed)
        self.reset_all_btn.clicked.connect(self._reset_all)
        self.process_btn.clicked.connect(self._start_processing)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.codec_combo.currentTextChanged.connect(self._update_codec_settings)
        self.use_crf_radio.toggled.connect(self._update_encoding_mode)
        self.lut_browse_btn.clicked.connect(self._browse_lut)
        self.lut_clear_btn.clicked.connect(self._clear_lut)
        # Update preview when slider is released or spinbox value changes
        self.lift_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.lift_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.lift_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.gamma_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.gamma_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.gamma_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.gain_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.gain_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.gain_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.saturation_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.saturation_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.saturation_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.lut_intensity_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.mask_size_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.mask_size_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.mask_size_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.mask_feather_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.mask_feather_slider.spinbox.valueChanged.connect(lambda: self._schedule_preview_update())
        self.mask_feather_slider.reset_btn.clicked.connect(self._schedule_preview_update)
        self.vision_pro_combo.currentIndexChanged.connect(self._update_vision_pro_mode)
        self.eye_toggle_btn.clicked.connect(self._toggle_eye)
        self.preview_mode_combo.currentIndexChanged.connect(self._update_preview_mode_ui)

        # Trim controls
        self.set_in_btn.clicked.connect(self._set_in_point)
        self.set_out_btn.clicked.connect(self._set_out_point)
        self.clear_trim_btn.clicked.connect(self._clear_trim)
        self.in_time_edit.editingFinished.connect(self._in_time_edited)
        self.out_time_edit.editingFinished.connect(self._out_time_edited)

        # GoPro gyro stabilization controls
        # Gyro stabilization controls
        self.gyro_stabilize_checkbox.toggled.connect(self._on_gyro_stabilize_toggled)
        self.gyro_smooth_slider.valueChanged.connect(self._on_gyro_param_changed)
        self.gyro_max_corr_slider.valueChanged.connect(self._on_gyro_param_changed)
        self.gyro_responsiveness_slider.valueChanged.connect(self._on_gyro_param_changed)
        self.gyro_horizon_lock_checkbox.toggled.connect(self._on_horizon_lock_toggled)
        self.rs_correction_checkbox.toggled.connect(self._on_rs_checkbox_toggled)
        self.advanced_toggle_btn.toggled.connect(self._on_advanced_toggle)
        self.rs_correction_slider.valueChanged.connect(self._schedule_preview_update)
        self.rs_factor_slider.valueChanged.connect(self._schedule_preview_update)
        self.rs_pitch_factor_slider.valueChanged.connect(self._schedule_preview_update)
        self.rs_roll_factor_slider.valueChanged.connect(self._schedule_preview_update)
        self.rs_use_cori_checkbox.toggled.connect(self._schedule_preview_update)
        self.rs_no_firmware_checkbox.toggled.connect(self._on_no_firmware_rs_toggled)

        # Initial state
        self._update_codec_settings()
        self._update_vision_pro_mode()
        self._update_preview_mode_ui()
    
    def _browse_input(self):
        # Use last used folder or home directory
        last_folder = self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", last_folder, "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.mts *.m2ts *.osv *.360)")
        if path:
            # Remember the folder for next time
            self.settings.setValue("last_input_folder", str(Path(path).parent))
            self.config.input_path = Path(path)
            self.config.is_360_input = path.lower().endswith('.360')
            self.input_path_edit.setText(path)
            # Detect multi-segment GoPro recordings (works for .360, .MP4, .MOV, .LRV)
            segments = detect_gopro_segments(path)
            if len(segments) > 1:
                seg_names = [Path(s).name for s in segments]
                msg = QMessageBox(self)
                msg.setWindowTitle("Multi-Segment Recording Detected")
                msg.setText(
                    f"Found {len(segments)} segments for this recording:\n"
                    + "\n".join(f"  {name}" for name in seg_names)
                )
                msg.setInformativeText("Import all segments as one combined clip, or just this file?")
                combine_btn = msg.addButton("Combined Clip", QMessageBox.ButtonRole.AcceptRole)
                single_btn = msg.addButton("This File Only", QMessageBox.ButtonRole.RejectRole)
                msg.setDefaultButton(combine_btn)
                msg.exec()
                if msg.clickedButton() == combine_btn:
                    self.config.segment_paths = segments
                    stem = Path(segments[0]).stem
                    output = Path(path).parent / f"{stem}_combined.mov"
                    self.input_path_edit.setText(f"{path} (+{len(segments)-1} segments)")
                else:
                    self.config.segment_paths = None
                    output = Path(path).parent / f"{Path(path).stem}_adjusted.mov"
            else:
                self.config.segment_paths = None
                output = Path(path).parent / f"{Path(path).stem}_adjusted.mov"
            self.output_path_edit.setText(str(output))
            self.config.output_path = output
            self._load_video_info()
            self.process_btn.setEnabled(True)
            # Show/hide audio format option based on input type
            self.audio_format_label.setVisible(self.config.is_360_input)
            self.audio_format_combo.setVisible(self.config.is_360_input)
            self.resolution_label.setVisible(self.config.is_360_input)
            self.resolution_combo.setVisible(self.config.is_360_input)
            self._output_format_row.setVisible(self.config.is_360_input)
            if not self.config.is_360_input:
                self.audio_format_combo.setCurrentIndex(0)  # reset to stereo
            # Show/hide gyro load buttons based on input type
            if self.config.is_360_input:
                self.gyro_load_row.setVisible(False)  # auto-load handles it
                self._auto_load_360_gyro(path)
                # Auto-populate bundled LUT for .360 (GP Log) if no LUT is set
                if not self.lut_path_edit.text():
                    bundled = get_bundled_lut_path()
                    if bundled:
                        self.lut_path_edit.setText(bundled)
                self._schedule_preview_update()
            else:
                self.gyro_load_row.setVisible(True)  # show manual load button
                self._clear_gopro_gyro_data()
                # Clear bundled LUT for non-360 input (it's GP Log specific)
                bundled = get_bundled_lut_path()
                if bundled and self.lut_path_edit.text() == bundled:
                    self.lut_path_edit.clear()
                self._schedule_preview_update()
    
    def _browse_output(self):
        # Use current output path, or last input folder, or home
        default_path = self.output_path_edit.text() or self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", default_path, "Video (*.mov)")
        if path:
            self.config.output_path = Path(path)
            self.output_path_edit.setText(path)

    def _browse_lut(self):
        """Browse for LUT file"""
        last_folder = self.settings.value("last_lut_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Select LUT File", last_folder, "LUT Files (*.cube *.3dl);;All Files (*.*)")
        if path:
            self.settings.setValue("last_lut_folder", str(Path(path).parent))
            self.lut_path_edit.setText(path)
            self._schedule_preview_update()

    def _clear_lut(self):
        """Clear the LUT file selection"""
        self.lut_path_edit.clear()
        self._schedule_preview_update()

    def _browse_gyro_360(self):
        """Browse for a .360 file to load gyro stabilization data for non-.360 input."""
        last_folder = self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Select .360 File for Gyro Data", last_folder, "GoPro 360 (*.360)")
        if not path:
            return
        # Detect multi-segment .360 recordings
        segments = detect_gopro_segments(path)
        gyro_segments = None
        if len(segments) > 1:
            seg_names = [Path(s).name for s in segments]
            msg = QMessageBox(self)
            msg.setWindowTitle("Multi-Segment .360 Recording Detected")
            msg.setText(
                f"Found {len(segments)} .360 segments:\n"
                + "\n".join(f"  {name}" for name in seg_names)
            )
            msg.setInformativeText("Load gyro data from all segments combined, or just this file?")
            combine_btn = msg.addButton("Combined", QMessageBox.ButtonRole.AcceptRole)
            single_btn = msg.addButton("This File Only", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(combine_btn)
            msg.exec()
            if msg.clickedButton() == combine_btn:
                gyro_segments = segments
        # Load gyro data (reuses same code as _auto_load_360_gyro)
        self.config.gyro_data = None
        self._gyro_stabilizer = None
        try:
            is_multi = gyro_segments is not None and len(gyro_segments) > 1
            self.gopro_status.setText(f"Loading gyro from .360{'...' if not is_multi else f' ({len(gyro_segments)} segments)...'}")
            self.gopro_status.setStyleSheet("QLabel { color: #0066cc; font-style: italic; }")
            QApplication.processEvents()

            if is_multi:
                gyro_data = concatenate_gyro_data(gyro_segments)
            else:
                gyro_data = parse_gopro_gyro_data(path)
            self.config.gyro_data = gyro_data
            # Detect no-firmware-RS mode: VQF fusion was used instead of firmware CORI
            if gyro_data.get('cori_is_vqf', False):
                self.config.no_firmware_rs = True
                self.config.rs_yaw_factor = 1.0
                self.config.rs_pitch_factor = 1.0
                self.config.rs_roll_factor = 1.0
                print("No-firmware-RS mode: RS factors set to (1,1,1) for both eyes")
                self.rs_factor_slider.setValue(1.0)
                self.rs_pitch_factor_slider.setValue(1.0)
                self.rs_roll_factor_slider.setValue(1.0)
                self.rs_no_firmware_checkbox.blockSignals(True)
                self.rs_no_firmware_checkbox.setChecked(True)
                self.rs_no_firmware_checkbox.blockSignals(False)
            else:
                self.config.no_firmware_rs = False
                self.rs_no_firmware_checkbox.blockSignals(True)
                self.rs_no_firmware_checkbox.setChecked(False)
                self.rs_no_firmware_checkbox.blockSignals(False)
            self._rebuild_gyro_stabilizer()

            self.gyro_stabilize_checkbox.setEnabled(True)
            self.rs_correction_checkbox.setEnabled(True)
            self.rs_correction_checkbox.setVisible(True)

            frames = gyro_data['frames']
            duration = frames[-1]['time'] if frames else 0
            cori_rolls = [f['cori_euler'][0] for f in frames]
            cori_pitches = [f['cori_euler'][1] for f in frames]
            cori_yaws = [f['cori_euler'][2] for f in frames]
            seg_info = f" ({len(gyro_segments)} segments)" if is_multi else ""
            if gyro_data.get('cori_is_vqf'):
                bias = gyro_data.get('gyro_bias_deg_s', (0,0,0))
                self.gopro_status.setText(
                    f"Loaded from .360{seg_info}: {len(frames)} frames ({duration:.1f}s)\n"
                    f"No firmware RS detected — RS (1,1,1) both eyes\n"
                    f"Gyro bias: [{bias[0]:.3f}, {bias[1]:.3f}, {bias[2]:.3f}] deg/s"
                )
                self.gopro_status.setStyleSheet("QLabel { color: #cc6600; }")
            else:
                self.gopro_status.setText(
                    f"Loaded from .360{seg_info}: {len(frames)} frames ({duration:.1f}s)\n"
                    f"CORI range: R[{min(cori_rolls):.1f},{max(cori_rolls):.1f}] "
                    f"P[{min(cori_pitches):.1f},{max(cori_pitches):.1f}] "
                    f"Y[{min(cori_yaws):.1f},{max(cori_yaws):.1f}]"
                )
                self.gopro_status.setStyleSheet("QLabel { color: #009900; }")

            # Duration mismatch warning
            if self.video_duration > 0 and duration > 0:
                ratio = abs(duration - self.video_duration) / self.video_duration
                if ratio > 0.05:
                    self.gopro_status.setText(
                        self.gopro_status.text() +
                        f"\n⚠ Duration mismatch: video {self.video_duration:.1f}s vs gyro {duration:.1f}s"
                    )
                    self.gopro_status.setStyleSheet("QLabel { color: #cc6600; }")

            # Load 800Hz GYRO angular velocity for RS correction
            try:
                fps = gyro_data['fps']
                if is_multi:
                    gyro_times, gyro_angvel = concatenate_800hz_gyro(gyro_segments, fps, gyro_data)
                else:
                    import sys as _sys_tmp
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    if script_dir not in _sys_tmp.path:
                        _sys_tmp.path.insert(0, script_dir)
                    from parse_gyro_raw import get_gyro_angular_velocity
                    n_frames = len(gyro_data['frames'])
                    gyro_times, gyro_angvel = get_gyro_angular_velocity(path, fps, n_frames)
                if gyro_times is not None and len(gyro_times) > 0:
                    gyro_data['gyro_angvel_times'] = gyro_times
                    gyro_data['gyro_angvel'] = gyro_angvel
                    print(f"Loaded 800Hz GYRO angular velocity: {len(gyro_times)} samples")
            except Exception as e:
                print(f"Warning: Could not load GYRO angular velocity: {e}")

            # Auto-set RS from SROT
            srot_ms = gyro_data.get('srot_ms')
            if srot_ms is not None and srot_ms > 0:
                self.rs_correction_slider.setValue(srot_ms)
                self.rs_correction_checkbox.setChecked(True)
                print(f"Auto-set RS correction from SROT: {srot_ms:.3f} ms")

            # Parse GEOC lens calibration
            geoc_path = gyro_segments[0] if is_multi else path
            try:
                geoc = parse_geoc(geoc_path)
                if geoc and 'FRNT' in geoc and 'KLNS' in geoc['FRNT']:
                    frnt = geoc['FRNT']
                    self.config.geoc_klns = frnt['KLNS']
                    self.config.geoc_ctrx = frnt.get('CTRX', 0.0)
                    self.config.geoc_ctry = frnt.get('CTRY', 0.0)
                    self.config.geoc_cal_dim = frnt.get('CAL_DIM', 4216)
                    print(f"GEOC: KLNS={self.config.geoc_klns[:3]}..., CTR=({self.config.geoc_ctrx:.1f}, {self.config.geoc_ctry:.1f})")
            except Exception as e:
                print(f"Warning: Could not parse GEOC: {e}")

            self.status_bar.showMessage(f"Loaded gyro from .360: {Path(path).name}")
            self._schedule_preview_update()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gopro_status.setText(f"Gyro load error: {str(e)}")
            self.gopro_status.setStyleSheet("QLabel { color: #cc0000; }")
            self.status_bar.showMessage(f"Failed to load gyro from .360")

    def _auto_load_360_gyro(self, path):
        """Auto-load GPMF gyro data when a .360 file is loaded as video input.
        Handles multi-segment recordings by concatenating gyro data across segments."""
        # Clear old gyro state first to prevent stale data if loading fails
        self.config.gyro_data = None
        self._gyro_stabilizer = None
        try:
            segments = self.config.segment_paths
            is_multi = segments is not None and len(segments) > 1

            if is_multi:
                self.gopro_status.setText(f"Loading orientation data from {len(segments)} segments...")
            else:
                self.gopro_status.setText("Auto-loading orientation data from .360...")
            self.gopro_status.setStyleSheet("QLabel { color: #0066cc; font-style: italic; }")
            QApplication.processEvents()

            if is_multi:
                self.gopro_status.setText(f"Parsing gyro from {len(segments)} segments...")
                QApplication.processEvents()
                gyro_data = concatenate_gyro_data(segments)
            else:
                gyro_data = parse_gopro_gyro_data(path)
            self.config.gyro_data = gyro_data
            if gyro_data.get('cori_is_vqf', False):
                self.config.no_firmware_rs = True
                self.config.rs_yaw_factor = 1.0
                self.config.rs_pitch_factor = 1.0
                self.config.rs_roll_factor = 1.0
                self.rs_factor_slider.setValue(1.0)
                self.rs_pitch_factor_slider.setValue(1.0)
                self.rs_roll_factor_slider.setValue(1.0)
                self.rs_no_firmware_checkbox.blockSignals(True)
                self.rs_no_firmware_checkbox.setChecked(True)
                self.rs_no_firmware_checkbox.blockSignals(False)
                print("No-firmware-RS mode: RS factors set to (1,1,1) for both eyes")
            else:
                self.config.no_firmware_rs = False
                self.rs_no_firmware_checkbox.blockSignals(True)
                self.rs_no_firmware_checkbox.setChecked(False)
                self.rs_no_firmware_checkbox.blockSignals(False)
            # Build stabilizer immediately for IORI compensation (even without gyro/RS enabled)
            self.gopro_status.setText("Building gyro stabilizer...")
            QApplication.processEvents()
            self._rebuild_gyro_stabilizer()

            self.gyro_stabilize_checkbox.setEnabled(True)

            # Enable and show RS correction checkbox
            self.rs_correction_checkbox.setEnabled(True)
            self.rs_correction_checkbox.setVisible(True)

            frames = gyro_data['frames']
            duration = frames[-1]['time'] if frames else 0
            cori_rolls = [f['cori_euler'][0] for f in frames]
            cori_pitches = [f['cori_euler'][1] for f in frames]
            cori_yaws = [f['cori_euler'][2] for f in frames]

            seg_info = f" ({len(segments)} segments)" if is_multi else ""
            if gyro_data.get('cori_is_vqf'):
                bias = gyro_data.get('gyro_bias_deg_s', (0,0,0))
                self.gopro_status.setText(
                    f"Auto-loaded from .360{seg_info}: {len(frames)} frames ({duration:.1f}s)\n"
                    f"No firmware RS detected — RS (1,1,1) both eyes\n"
                    f"Gyro bias: [{bias[0]:.3f}, {bias[1]:.3f}, {bias[2]:.3f}] deg/s"
                )
                self.gopro_status.setStyleSheet("QLabel { color: #cc6600; }")
            else:
                self.gopro_status.setText(
                    f"Auto-loaded from .360{seg_info}: {len(frames)} frames ({duration:.1f}s)\n"
                    f"CORI range: R[{min(cori_rolls):.1f}°,{max(cori_rolls):.1f}°] "
                    f"P[{min(cori_pitches):.1f}°,{max(cori_pitches):.1f}°] "
                    f"Y[{min(cori_yaws):.1f}°,{max(cori_yaws):.1f}°]"
                )
                self.gopro_status.setStyleSheet("QLabel { color: #009900; }")

            # Load 800Hz GYRO angular velocity for RS correction
            self.gopro_status.setText("Loading 800Hz gyro for RS correction...")
            QApplication.processEvents()
            try:
                fps = gyro_data['fps']
                if is_multi:
                    gyro_times, gyro_angvel = concatenate_800hz_gyro(segments, fps, gyro_data)
                else:
                    import sys, os
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    if script_dir not in sys.path:
                        sys.path.insert(0, script_dir)
                    from parse_gyro_raw import get_gyro_angular_velocity
                    n_frames = len(gyro_data['frames'])
                    gyro_times, gyro_angvel = get_gyro_angular_velocity(path, fps, n_frames)
                if gyro_times is not None and len(gyro_times) > 0:
                    gyro_data['gyro_angvel_times'] = gyro_times
                    gyro_data['gyro_angvel'] = gyro_angvel
                    print(f"Loaded 800Hz GYRO angular velocity: {len(gyro_times)} samples")
            except Exception as e:
                print(f"Warning: Could not load GYRO angular velocity: {e}")

            # Auto-set RS correction slider from SROT if available
            srot_ms = gyro_data.get('srot_ms')
            if srot_ms is not None and srot_ms > 0:
                self.rs_correction_slider.setValue(srot_ms)
                self.rs_correction_checkbox.setChecked(True)
                print(f"Auto-set RS correction from SROT: {srot_ms:.3f} ms")

            # Auto-parse GEOC lens calibration for proper RS fisheye geometry
            self.gopro_status.setText("Parsing lens calibration (GEOC)...")
            QApplication.processEvents()
            # Use first segment (all segments from same camera have identical GEOC)
            geoc_path = segments[0] if is_multi else path
            try:
                geoc = parse_geoc(geoc_path)
                if geoc and 'FRNT' in geoc and 'KLNS' in geoc['FRNT']:
                    # FRNT lens = right eye after yaw mod
                    self.config.geoc_klns = geoc['FRNT']['KLNS']
                    self.config.geoc_ctrx = geoc['FRNT'].get('CTRX', 0.0)
                    self.config.geoc_ctry = geoc['FRNT'].get('CTRY', 0.0)
                    self.config.geoc_cal_dim = geoc['global'].get('CALW', 4216)
                    # Store BACK lens calibration for fisheye output mode
                    if 'BACK' in geoc and 'KLNS' in geoc['BACK']:
                        self.config._geoc_back = geoc['BACK']
                    klns = self.config.geoc_klns
                    print(f"GEOC KLNS loaded: f={klns[0]:.2f} "
                          f"CTR=({self.config.geoc_ctrx:.2f}, {self.config.geoc_ctry:.2f}) "
                          f"dim={self.config.geoc_cal_dim}")
                else:
                    print("GEOC not found in .360 file — RS correction will be disabled (requires lens calibration)")
                    self.config.geoc_klns = None
            except Exception as geoc_err:
                print(f"GEOC parse warning: {geoc_err}")

            self.status_bar.showMessage(f"Auto-loaded gyro data from .360{seg_info}: {len(frames)} frames")
            self._schedule_preview_update()

        except Exception as e:
            self.gopro_status.setText(f"Gyro auto-load error: {str(e)}")
            self.gopro_status.setStyleSheet("QLabel { color: #cc0000; }")
            print(f"Failed to auto-load gyro from .360: {e}")

    def _clear_gopro_gyro_data(self):
        """Clear the loaded GoPro gyro data and GEOC calibration"""
        self.config.gyro_data = None
        self._gyro_stabilizer = None
        self.config.geoc_klns = None
        self.config.geoc_ctrx = 0.0
        self.config.geoc_ctry = 0.0
        self.config.geoc_cal_dim = 4216
        self.gopro_status.setText("No orientation data loaded")
        self.gopro_status.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        self.gyro_stabilize_checkbox.setChecked(False)
        self.gyro_stabilize_checkbox.setEnabled(False)
        self.gyro_controls_widget.setVisible(False)
        self.rs_correction_checkbox.setChecked(False)
        self.rs_correction_checkbox.setEnabled(False)
        self.rs_correction_checkbox.setVisible(False)
        self.rs_controls_widget.setVisible(False)
        self.status_bar.showMessage("GoPro gyro data cleared")
        self._schedule_preview_update()

    def _on_gyro_stabilize_toggled(self, checked):
        """Toggle gyro stabilization controls. When gyro is on, RS is forced on."""
        self.gyro_controls_widget.setVisible(checked)
        if checked:
            # Force RS on when gyro is enabled
            self.rs_correction_checkbox.setChecked(True)
            self.rs_correction_checkbox.setEnabled(False)  # Lock it on
        else:
            # Restore RS checkbox to user control
            self.rs_correction_checkbox.setEnabled(True)
        if self.config.gyro_data:
            # Always rebuild: stabilize=checked controls heading correction,
            # but IORI compensation is always needed
            self._rebuild_gyro_stabilizer()
            self._schedule_preview_update()

    def _on_advanced_toggle(self, checked):
        """Toggle advanced settings visibility"""
        self.advanced_settings_widget.setVisible(checked)
        self.advanced_toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def _on_rs_checkbox_toggled(self, checked):
        """Toggle RS controls visibility and update preview"""
        self.rs_controls_widget.setVisible(checked)
        # Build stabilizer for angular velocity + IORI if none exists yet
        if checked and self.config.gyro_data and self._gyro_stabilizer is None:
            self._rebuild_gyro_stabilizer()
        self._schedule_preview_update()

    def _on_no_firmware_rs_toggled(self, checked):
        """Toggle no-firmware RS mode and update factors accordingly."""
        self.config.no_firmware_rs = checked
        if checked:
            # Set factors to (1,1,1) for fresh RS on both eyes
            self.rs_factor_slider.setValue(1.0)
            self.rs_pitch_factor_slider.setValue(1.0)
            self.rs_roll_factor_slider.setValue(1.0)
        else:
            # Restore firmware-RS defaults (0,2,2)
            self.rs_factor_slider.setValue(0.0)
            self.rs_pitch_factor_slider.setValue(2.0)
            self.rs_roll_factor_slider.setValue(2.0)
        self._schedule_preview_update()

    def _on_adjustment_changed(self, _value=None):
        """Called when global adjustment or stereo offset sliders change.
        Rebuilds stabilizer if it exists (adjustments are baked into correction), otherwise just updates preview."""
        if self._gyro_stabilizer is not None and self.config.gyro_data:
            self._rebuild_gyro_stabilizer()
        self._schedule_preview_update()

    def _on_horizon_lock_toggled(self, checked):
        """Toggle horizon lock."""
        self._on_gyro_param_changed()

    def _on_resolution_changed(self, _idx=None):
        """Called when output resolution dropdown changes"""
        res = self.resolution_combo.currentText()  # "8192x4096", "8640x4320 for YT", "7680x3840"
        # Strip any suffix after the resolution (e.g. " for YT")
        res_part = res.split()[0]  # "8640x4320"
        parts = res_part.split("x")
        self.config.eac_out_w = int(parts[0])
        self.config.eac_out_h = int(parts[1])
        # Update class-level defaults so preview uses the new resolution
        FrameExtractor.EAC_OUT_W = self.config.eac_out_w
        FrameExtractor.EAC_OUT_H = self.config.eac_out_h
        # Invalidate preview caches that depend on resolution
        self._pv_xyz_cache = None
        if hasattr(self, '_pv_dims'):
            self._pv_dims = None  # Forces preview xyz grid rebuild
        self._schedule_preview_update()

    def _on_gyro_param_changed(self, _value=None):
        """Called when any gyro parameter slider changes"""
        if self.gyro_stabilize_checkbox.isChecked() and self.config.gyro_data:
            self._rebuild_gyro_stabilizer()
            self._schedule_preview_update()

    def _rebuild_gyro_stabilizer(self):
        """Rebuild the GyroStabilizer with current slider values"""
        if not self.config.gyro_data:
            return
        self._gyro_stabilizer = GyroStabilizer(self.config.gyro_data)
        self._gyro_stabilizer.smooth(
            window_ms=self.gyro_smooth_slider.value(),
            horizon_lock=self.gyro_horizon_lock_checkbox.isChecked(),
            stabilize=self.gyro_stabilize_checkbox.isChecked(),
            max_corr_deg=self.gyro_max_corr_slider.value(),
            responsiveness=self.gyro_responsiveness_slider.value(),
            upside_down=self.upside_down_checkbox.isChecked(),
        )
        # Clear preview remap cache (stale rotation matrix keys from previous file/params)
        if hasattr(self, '_preview_remap_cache'):
            self._preview_remap_cache.clear()

    def _update_codec_settings(self):
        """Show/hide settings based on selected codec"""
        codec = self.codec_combo.currentText()
        is_h265 = codec == "H.265"
        is_prores = codec == "ProRes"

        # Show H.265 settings only when H.265 is selected
        self.quality_label.setVisible(is_h265)
        self.quality_spinbox.setVisible(is_h265)
        self.bitrate_label.setVisible(is_h265)
        self.bitrate_spinbox.setVisible(is_h265)
        self.use_crf_radio.setVisible(is_h265)
        self.use_bitrate_radio.setVisible(is_h265)

        # Show ProRes settings only when ProRes is selected
        self.prores_label.setVisible(is_prores)
        self.prores_combo.setVisible(is_prores)

        # Show H.265 bit depth only when H.265 is selected
        self.h265_bit_depth_label.setVisible(is_h265)
        self.h265_bit_depth_combo.setVisible(is_h265)

        # Update encoding mode based on current selection
        if is_h265:
            self._update_encoding_mode()

        # Always use .mov container (supports all codecs + ambisonic audio)
        if self.output_path_edit.text():
            output_path = Path(self.output_path_edit.text())
            if output_path.suffix.lower() != '.mov':
                new_path = output_path.with_suffix('.mov')
                self.output_path_edit.setText(str(new_path))
                self.config.output_path = new_path

    def _update_encoding_mode(self):
        """Enable/disable quality or bitrate spinbox based on radio selection"""
        use_crf = self.use_crf_radio.isChecked()
        self.quality_spinbox.setEnabled(use_crf)
        self.bitrate_spinbox.setEnabled(not use_crf)

    def _update_vision_pro_mode(self):
        """Gray out YouTube metadata checkbox when APMP or MV-HEVC is selected.
        APMP strips st3d/sv3d atoms that YouTube injection adds, so they conflict."""
        if IS_MACOS:
            vp_idx = self.vision_pro_combo.currentIndex()
            is_vp_mode = vp_idx >= 1  # APMP (1) or MV-HEVC (2)
            self.vr180_metadata_checkbox.setEnabled(not is_vp_mode)
            if is_vp_mode:
                self.vr180_metadata_checkbox.setChecked(False)

    def _update_preview_mode_ui(self):
        """Show/hide eye toggle button based on preview mode"""
        mode = self.preview_mode_combo.currentData()
        is_single_eye = mode == PreviewMode.SINGLE_EYE
        self.eye_toggle_btn.setVisible(is_single_eye)
        self.eye_label.setVisible(is_single_eye)
        if is_single_eye:
            self.eye_label.setText("L" if self.current_eye == "left" else "R")

    def _toggle_eye(self):
        """Toggle between left and right eye in Single Eye Mode"""
        self.current_eye = "right" if self.current_eye == "left" else "left"
        self.eye_label.setText("L" if self.current_eye == "left" else "R")
        self._update_preview()

    def _update_info_panel(self, lines):
        """Update the info panel with a list of text lines."""
        self.ffmpeg_output.setHtml(
            '<span style="color:#d4d4d4;">' +
            '<br>'.join(lines) +
            '</span>'
        )

    def _load_video_info(self):
        if not self.config.input_path: return
        try:
            result = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_range,color_space,color_transfer,bit_rate",
                                    "-show_entries", "format=duration,size",
                                    "-of", "json", str(self.config.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            info = json.loads(result.stdout)
            self.video_duration = float(info.get("format", {}).get("duration", 0))
            # For multi-segment recordings, sum durations and cache per-segment info
            self._segment_durations = None  # [(path, duration, cumulative_start), ...]
            if self.config.segment_paths and len(self.config.segment_paths) > 1:
                seg_durs = []
                cumulative = 0.0
                for seg_path in self.config.segment_paths:
                    try:
                        seg_probe = subprocess.run([get_ffprobe_path(), "-v", "quiet",
                                                    "-show_entries", "format=duration", "-of", "json",
                                                    str(seg_path)], capture_output=True, text=True, check=True,
                                                   creationflags=get_subprocess_flags())
                        seg_info = json.loads(seg_probe.stdout)
                        dur = float(seg_info.get("format", {}).get("duration", 0))
                    except Exception:
                        dur = 0.0
                    seg_durs.append((seg_path, dur, cumulative))
                    cumulative += dur
                if cumulative > 0:
                    self.video_duration = cumulative
                    self._segment_durations = seg_durs
            # For .360 input, use EAC→equirect output dimensions instead of raw EAC dims
            if self.config.is_360_input:
                width = self.config.eac_out_w
            else:
                width = info["streams"][0]["width"]
            self.time_label.setText(f"00:00 / {self._format_time(self.video_duration)}")
            half = width // 2
            self.global_shift_slider.slider.setMinimum(-half)
            self.global_shift_slider.slider.setMaximum(half)
            self.global_shift_slider.spinbox.setMinimum(-half)
            self.global_shift_slider.spinbox.setMaximum(half)

            # Reset trim points for new video
            self.trim_start = 0.0
            self.trim_end = 0.0
            self.in_time_edit.setText("00:00:00.000")
            self.out_time_edit.setText(self._format_time_full(self.video_duration))
            self._update_trim_duration()

            if self.config.is_360_input:
                # Build detailed backend info
                remap_backend = "MLX (Metal)" if HAS_MLX else "CUDA (NVIDIA)" if HAS_NUMBA_CUDA else "wgpu" if HAS_WGPU else "Numba (CPU)" if HAS_NUMBA else "NumPy (CPU)"
                accel_parts = [remap_backend]
                if HAS_NUMBA_CUDA:
                    try:
                        dev = _numba_cuda.get_current_device()
                        accel_parts[0] = f"CUDA ({dev.name})"
                    except Exception:
                        pass
                self.status_bar.showMessage(
                    f"Loaded .360: {self.config.input_path.name} "
                    f"(EAC→{width}x{self.config.eac_out_h}) — "
                    f"Remap: {accel_parts[0]}")
            else:
                seg_info = ""
                if self.config.segment_paths and len(self.config.segment_paths) > 1:
                    seg_info = f" ({len(self.config.segment_paths)} segments, {self.video_duration:.1f}s total)"
                self.status_bar.showMessage(f"Loaded: {self.config.input_path.name} ({width}x{info['streams'][0]['height']}){seg_info}")
            # Populate info panel with video details
            stream_info = info.get("streams", [{}])[0]
            _v_codec = stream_info.get("codec_name", "?")
            _v_pix = stream_info.get("pix_fmt", "?")
            _v_range = stream_info.get("color_range", "?")
            _v_space = stream_info.get("color_space", "?")
            _v_trc = stream_info.get("color_transfer", "?")
            _v_w = stream_info.get("width", "?")
            _v_h = stream_info.get("height", "?")
            _v_rfr = stream_info.get("r_frame_rate", "0/1")
            try:
                _n, _d = _v_rfr.split('/')
                _v_fps = f"{int(_n)/int(_d):.2f}"
            except:
                _v_fps = _v_rfr
            _f_size = int(info.get("format", {}).get("size", 0))
            _f_size_str = f"{_f_size/1e9:.2f} GB" if _f_size > 1e9 else f"{_f_size/1e6:.1f} MB"
            _dur_str = self._format_time(self.video_duration)
            _bit_depth = "10-bit" if "10" in _v_pix else "8-bit"
            if self.config.is_360_input:
                _remap = "MLX Metal" if HAS_MLX else "CUDA" if HAS_NUMBA_CUDA else "wgpu" if HAS_WGPU else "Numba" if HAS_NUMBA else "CPU"
            else:
                _remap = "CUDA" if HAS_NUMBA_CUDA else "MLX Metal" if HAS_MLX else "Numba" if HAS_NUMBA else "CPU"
            _decode = "PyAV" if HAS_AV else "FFmpeg"
            _gyro = "Yes" if self.config.gyro_data else "No"
            _seg = ""
            if self.config.segment_paths and len(self.config.segment_paths) > 1:
                _seg = f"  |  Segments: {len(self.config.segment_paths)}"
            if self.config.is_360_input:
                _type = f".360 EAC → {self.config.eac_out_w}x{self.config.eac_out_h} SBS"
            else:
                _type = f"SBS {_v_w}x{_v_h}"
            info_lines = [
                f"<b>{self.config.input_path.name}</b>  |  {_type}  |  {_dur_str}  |  {_f_size_str}{_seg}",
                f"{_v_codec.upper()}  {_bit_depth} ({_v_pix})  |  {_v_fps} fps  |  Range: {_v_range}  |  Color: {_v_space}/{_v_trc}",
                f"Decode: {_decode}  |  Remap: {_remap}  |  Gyro: {_gyro}",
            ]
            self._update_info_panel(info_lines)

            # Invalidate all caches so the new video is always decoded fresh
            self.cached_raw_frame = None
            self.cached_timestamp = -1
            if hasattr(self, '_preview_remap_cache'):
                self._preview_remap_cache.clear()
            if hasattr(self, '_pv_dims'):
                self._pv_dims = None
            self._extract_frame(0)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _resolve_segment_timestamp(self, timestamp):
        """For multi-segment recordings, map combined timestamp to (segment_path, local_ts).
        Returns (input_path, local_timestamp) for single files or resolved segment."""
        if hasattr(self, '_segment_durations') and self._segment_durations:
            for seg_path, dur, start in self._segment_durations:
                if timestamp < start + dur:
                    return Path(seg_path), timestamp - start
            # Past the end — use last segment
            seg_path, dur, start = self._segment_durations[-1]
            return Path(seg_path), dur - 0.01
        return self.config.input_path, timestamp

    def _extract_frame(self, timestamp, force_filter=False):
        if not self.config.input_path: return
        if hasattr(self, 'extractor') and self.extractor and self.extractor.isRunning():
            self.extractor.cancel()  # Kill FFmpeg process properly
            self.extractor.wait(500)  # Wait longer for cleanup
        self.preview_timestamp = timestamp

        # If we have cached frame at this timestamp, apply OpenCV filters directly
        if self.cached_timestamp == timestamp and self.cached_raw_frame is not None:
            self._apply_preview_filters_to_cached_frame()
        else:
            # Resolve correct segment file for multi-segment recordings
            input_path, local_ts = self._resolve_segment_timestamp(timestamp)
            # Need to decode new frame - extract raw frame first for caching
            # OpenCV will apply all adjustments after frame is loaded
            self.status_bar.showMessage(f"Loading frame at {timestamp:.2f}s...")
            self.extractor = FrameExtractor(input_path, local_ts, extract_raw=True,
                                               is_360=self.config.is_360_input)
            self.extractor.raw_frame_ready.connect(self._on_raw_frame_extracted)
            self.extractor.error.connect(lambda e: self.status_bar.showMessage(f"Error: {e}"))
            self.extractor.start()
    
    def _build_preview_filter(self):
        shift = self.global_shift_slider.value()
        yaw, pitch, roll = self.global_yaw.value(), self.global_pitch.value(), self.global_roll.value()
        syaw, spitch, sroll = self.stereo_yaw_offset.value(), self.stereo_pitch_offset.value(), self.stereo_roll_offset.value()
        lut_path = self.lut_path_edit.text()
        lut_intensity = self.lut_intensity_slider.value() / 100.0

        # Get color adjustment values (ASC CDL: Lift, Gamma, Gain, Saturation)
        lift = self.lift_slider.value() / 100.0  # -1.0 to 1.0
        gamma = self.gamma_slider.value() / 100.0  # 0.1 to 3.0
        gain = self.gain_slider.value() / 100.0  # 0.5 to 2.0
        saturation = self.saturation_slider.value() / 100.0  # 0.0 to 2.0

        # Check if any adjustments are needed
        has_adjustments = any([shift, yaw, pitch, roll, syaw, spitch, sroll])
        has_lut = lut_path and Path(lut_path).exists() and lut_intensity > 0.01
        has_color_adjustments = (abs(lift) > 0.01 or
                                abs(gamma - 1.0) > 0.01 or
                                abs(gain - 1.0) > 0.01 or
                                abs(saturation - 1.0) > 0.01)

        if not has_adjustments and not has_lut and not has_color_adjustments:
            return None

        filters = []

        # Check if input is 10-bit and convert to 8-bit first for much faster processing
        try:
            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                  "-show_entries", "stream=pix_fmt", "-of", "json",
                                  str(self.config.input_path)],
                                  capture_output=True, text=True, creationflags=get_subprocess_flags())
            pix_fmt = json.loads(probe.stdout)["streams"][0].get("pix_fmt", "")
            is_10bit = "10le" in pix_fmt or "p010" in pix_fmt

            if is_10bit:
                # Convert to 8-bit first - 10x faster for preview filters
                filters.append("[0:v]format=yuv420p[input_8bit]")
                input_label = "[input_8bit]"
            else:
                input_label = "[0:v]"
        except:
            input_label = "[0:v]"

        if shift != 0:
            if shift > 0:
                filters.extend([f"{input_label}split=2[sh_a][sh_b]", f"[sh_a]crop={shift}:ih:0:0[sh_right]",
                               f"[sh_b]crop=iw-{shift}:ih:{shift}:0[sh_left]", f"[sh_left][sh_right]hstack=inputs=2[shifted]"])
            else:
                s = abs(shift)
                filters.extend([f"{input_label}split=2[sh_a][sh_b]", f"[sh_a]crop={s}:ih:iw-{s}:0[sh_left]",
                               f"[sh_b]crop=iw-{s}:ih:0:0[sh_right]", f"[sh_left][sh_right]hstack=inputs=2[shifted]"])
            inp = "[shifted]"
        else:
            inp = input_label

        filters.extend([f"{inp}split=2[full1][full2]", "[full1]crop=iw/2:ih:0:0[left_in]", "[full2]crop=iw/2:ih:iw/2:0[right_in]"])

        ly, lp, lr = yaw + syaw, pitch + spitch, roll + sroll
        ry, rp, rr = yaw - syaw, pitch - spitch, roll - sroll

        if any([ly, lp, lr]):
            filters.append(f"[left_in]v360=input=hequirect:output=hequirect:yaw={ly}:pitch={lp}:roll={lr}:interp=lanczos[left_out]")
        else:
            filters.append("[left_in]null[left_out]")
        if any([ry, rp, rr]):
            filters.append(f"[right_in]v360=input=hequirect:output=hequirect:yaw={ry}:pitch={rp}:roll={rr}:interp=lanczos[right_out]")
        else:
            filters.append("[right_in]null[right_out]")

        # Combine left and right
        filters.append("[left_out][right_out]hstack=inputs=2[stacked]")

        # Apply pre-LUT color adjustments (ASC CDL: Lift, Gamma, Gain)
        current_label = "[stacked]"

        if has_color_adjustments:
            # Use classic Lift/Gamma/Gain formula: out = (gain * (x + lift * (1-x)))^(1/gamma)
            # Lift affects shadows (preserves white at 1.0)
            # Gain affects highlights (preserves black at 0.0)
            # Gamma affects midtones (power function)
            # lutrgb works with pixel values (0-255 for 8-bit), normalize to 0-1 first

            # Normalize to 0-1 range
            lut_expr = "val/maxval"

            # Apply Lift: x + lift * (1-x)
            # This lifts shadows while preserving white point
            # lift range: -1 to 1 (0 = neutral)
            if abs(lift) > 0.01:
                lut_expr = f"({lut_expr}+{lift}*(1-{lut_expr}))"

            # Apply Gain: multiply the result
            # This scales highlights while preserving black point (after lift adjustment)
            # gain range: 0.5 to 2.0 (1.0 = neutral)
            if abs(gain - 1.0) > 0.01:
                lut_expr = f"({lut_expr}*{gain})"

            # Clamp before gamma to avoid pow() on negative values
            lut_expr = f"clip({lut_expr},0,1)"

            # Apply Gamma: power function
            # gamma range: 0.1 to 3.0 (1.0 = neutral)
            if abs(gamma - 1.0) > 0.01:
                power = 1.0 / gamma
                lut_expr = f"pow({lut_expr},{power})"

            # Final clamp and denormalize back to pixel values
            lut_expr = f"clip({lut_expr},0,1)*maxval"

            filters.append(f"{current_label}lutrgb=r='{lut_expr}':g='{lut_expr}':b='{lut_expr}'[lgg_adjusted]")
            current_label = "[lgg_adjusted]"

        # Apply LUT BEFORE saturation so saturation only stretches the final
        # graded chroma (matches export pipeline: 1D LUT → 3D LUT → saturation).
        if has_lut:
            lut_path_str = lut_path.replace('\\', '/').replace(':', '\\:')
            if lut_intensity >= 0.99:
                # Full intensity - apply LUT directly without blending (much faster)
                filters.append(f"{current_label}format=gbrp,lut3d=file='{lut_path_str}':interp=tetrahedral[lut_applied]")
                current_label = "[lut_applied]"
            elif lut_intensity > 0.01:
                # Partial intensity - use tetrahedral interpolation for better performance
                filters.append(f"{current_label}format=gbrp,split[original][lut_input]")
                filters.append(f"[lut_input]lut3d=file='{lut_path_str}':interp=tetrahedral[lut_output]")
                filters.append(f"[original][lut_output]blend=all_expr='A*(1-{lut_intensity})+B*{lut_intensity}'[lut_applied]")
                current_label = "[lut_applied]"

        # Apply saturation AFTER the LUT (post-LUT, matches export pipeline)
        if abs(saturation - 1.0) > 0.01:
            filters.append(f"{current_label}eq=saturation={saturation}[sat_adjusted]")
            current_label = "[sat_adjusted]"

        # Final null label to produce [out]
        filters.append(f"{current_label}null[out]")

        return ";".join(filters)
    
    def _on_raw_frame_extracted(self, frame):
        """Callback when raw frame is extracted - cache it and apply filters"""
        self.cached_raw_frame = frame
        self.cached_timestamp = self.preview_timestamp
        # Cache individual crosses for .360 per-eye remap (avoids lens mixing at boundary)
        if self.config.is_360_input and hasattr(self, 'extractor') and hasattr(self.extractor, 'crossA'):
            self.cached_crossA = self.extractor.crossA
            self.cached_crossB = self.extractor.crossB
        self.status_bar.showMessage(f"Frame loaded - adjustments are now instant")
        self._apply_preview_filters_to_cached_frame()

    def _apply_preview_filters_to_cached_frame(self):
        """Apply adjustments to cached frame using OpenCV (instant preview)"""
        if self.cached_raw_frame is None:
            return

        if not HAS_CV2:
            # Fallback: use cached frame directly without adjustments
            self.original_frame = self.cached_raw_frame.copy()
            self._update_preview()
            self.status_bar.showMessage("Ready (OpenCV not available for instant preview)")
            return

        import math

        frame = self.cached_raw_frame.copy()
        h, full_w = frame.shape[:2]
        half_w = full_w // 2

        # Get adjustment values
        global_shift = self.global_shift_slider.value()
        global_yaw = self.global_yaw.value()
        global_pitch = self.global_pitch.value()
        global_roll = self.global_roll.value()
        stereo_yaw = self.stereo_yaw_offset.value()
        stereo_pitch = self.stereo_pitch_offset.value()
        stereo_roll = self.stereo_roll_offset.value()

        # Get per-eye gyro correction matrices if enabled
        gyro_left = None
        gyro_right = None
        # Stabilizer available = has gyro data (handles IORI + optional heading correction)
        has_stabilizer = (hasattr(self, '_gyro_stabilizer') and self._gyro_stabilizer is not None)
        gyro_enabled = (self.gyro_stabilize_checkbox.isChecked() and has_stabilizer)

        # Stereo offset: same convention as .360 path
        # right = global + stereo, left = global - stereo
        right_yaw = global_yaw + stereo_yaw
        right_pitch = global_pitch + stereo_pitch
        right_roll = global_roll + stereo_roll
        left_yaw = global_yaw - stereo_yaw
        left_pitch = global_pitch - stereo_pitch
        left_roll = global_roll - stereo_roll
        if has_stabilizer:
            preview_time = 0
            if self.video_duration > 0:
                preview_time = (self.timeline_slider.value() / 1000.0) * self.video_duration
            gyro_left = self._gyro_stabilizer.get_left_matrix_at_time(preview_time)
            gyro_right = self._gyro_stabilizer.get_right_matrix_at_time(preview_time)

        # Force -1920 shift for .360 with gyro (GoPro equirect eye alignment)
        if gyro_enabled and self.config.is_360_input:
            global_shift = -1920

        # For non-360 SBS: always apply -width/2 base shift to swap eye halves
        # (code puts left half → right output, so input halves must be swapped)
        if not self.config.is_360_input:
            global_shift = -(full_w // 2) + global_shift

        # Apply global shift
        if global_shift != 0:
            frame = np.roll(frame, global_shift, axis=1)

        # In the rolled frame:
        #   Cols 0:half_w = right eye data (lens A), center at col 1920 → lon = -π/2
        #   Cols half_w:width = left eye data (lens B), center at col 5760 → lon = +π/2
        # For .360 input: frame IS a full 360° equirect → use BORDER_WRAP (cv2 handles seam)
        # For .mov input: frame is SBS halves → pad + BORDER_REPLICATE (legacy behavior)
        is_360 = getattr(self.config, 'is_360_input', False)
        if is_360:
            remap_src = frame  # Full equirect, BORDER_WRAP handles wrap-around
            remap_border = cv2.BORDER_WRAP
        else:
            frame_padded = np.concatenate([frame, frame[:, :2]], axis=1)
            remap_src = frame_padded
            remap_border = cv2.BORDER_REPLICATE

        RIGHT_EYE_LON = np.float32(-math.pi / 2)
        LEFT_EYE_LON  = np.float32( math.pi / 2)

        if not hasattr(self, '_preview_remap_cache'):
            self._preview_remap_cache = {}

        def _build_full_remap(R, eye_lon_offset, cache_key):
            """Build remap: (half_w × h) output sourced from full equirect with eye offset.

            Instead of remapping from a split half-equirect (which clips at ±90°),
            this maps output pixels to the full 360° equirect, allowing rotations
            to access content beyond 180° FOV from the other lens.
            """
            if cache_key in self._preview_remap_cache:
                return self._preview_remap_cache[cache_key]
            u = np.linspace(0, 1, half_w, dtype=np.float32)
            v = np.linspace(0, 1, h, dtype=np.float32)
            ug, vg = np.meshgrid(u, v)
            lon = (ug - 0.5) * np.float32(math.pi)
            lat = (0.5 - vg) * np.float32(math.pi)
            cos_lat = np.cos(lat)
            x = cos_lat * np.sin(lon)
            y = np.sin(lat)
            z = cos_lat * np.cos(lon)
            R = R.astype(np.float32)
            xn = R[0,0]*x + R[0,1]*y + R[0,2]*z
            yn = R[1,0]*x + R[1,1]*y + R[1,2]*z
            zn = R[2,0]*x + R[2,1]*y + R[2,2]*z
            lat_n = np.arcsin(np.clip(yn, -1, 1))
            lon_n = np.arctan2(xn, zn)
            # Map to full equirect with eye center offset, wrapping horizontally
            src_lon = lon_n + eye_lon_offset
            mx = (src_lon / np.float32(2 * math.pi) + 0.5) * np.float32(full_w)
            mx = np.fmod(mx, np.float32(full_w))
            mx[mx < 0] += np.float32(full_w)
            my = np.clip((0.5 - lat_n / np.float32(math.pi)) * np.float32(h), 0, np.float32(h - 1))
            maps = (mx, my)
            if len(self._preview_remap_cache) > 20:
                self._preview_remap_cache.clear()
            self._preview_remap_cache[cache_key] = maps
            return maps

        def _build_ypr_matrix(yaw_deg, pitch_deg, roll_deg):
            """Build 3×3 rotation matrix from yaw/pitch/roll in degrees."""
            yr, pr, rr = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
            cy, sy = math.cos(yr), math.sin(yr)
            cp, sp = math.cos(pr), math.sin(pr)
            cr, sr = math.cos(rr), math.sin(rr)
            return (np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32) @
                    np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32) @
                    np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]], dtype=np.float32))

        def _build_halfequirect_remap(R, half_w, h):
            """Build remap: rotated half-equirect → source half-equirect.

            Both output and source are half_w × h, covering ±90° in lon and lat.
            Used to apply gyro rotation to an already-RS-corrected half-equirect.
            """
            cache_key = ('hR_rot', h, half_w) + tuple(np.round(R.flatten(), 4))
            if cache_key in self._preview_remap_cache:
                return self._preview_remap_cache[cache_key]
            u = np.linspace(0, 1, half_w, dtype=np.float32)
            v = np.linspace(0, 1, h, dtype=np.float32)
            ug, vg = np.meshgrid(u, v)
            lon = (ug - 0.5) * np.float32(math.pi)
            lat = (0.5 - vg) * np.float32(math.pi)
            cos_lat = np.cos(lat)
            x = cos_lat * np.sin(lon)
            y = np.sin(lat)
            z = cos_lat * np.cos(lon)
            R = R.astype(np.float32)
            xn = R[0,0]*x + R[0,1]*y + R[0,2]*z
            yn = R[1,0]*x + R[1,1]*y + R[1,2]*z
            zn = R[2,0]*x + R[2,1]*y + R[2,2]*z
            lat_n = np.arcsin(np.clip(yn, -1, 1))
            lon_n = np.arctan2(xn, zn)
            mx = np.clip((lon_n / np.float32(math.pi) + 0.5) * np.float32(half_w),
                         0, np.float32(half_w - 1)).astype(np.float32)
            my = np.clip((0.5 - lat_n / np.float32(math.pi)) * np.float32(h),
                         0, np.float32(h - 1)).astype(np.float32)
            maps = (mx, my)
            if len(self._preview_remap_cache) > 30:
                self._preview_remap_cache.clear()
            self._preview_remap_cache[cache_key] = maps
            return maps

        def _build_halfequirect_remap_rs(R, half_w, h, angular_vel, rs_time_map,
                                          rs_ms, rs_yaw_factor, rs_pitch_factor, rs_roll_factor):
            """Build remap: rotation + KLNS dr/dθ RS correction → source half-equirect.

            Uses the same KLNS sensor-space pixel-shift model as the .360 Metal kernel:
              1. R × direction → rotated direction
              2. KLNS forward → sensor pixel → dr/dθ shift → KLNS inverse → corrected direction
              3. Convert back to equirect source coordinates
            """
            u = np.linspace(0, 1, half_w, dtype=np.float32)
            v = np.linspace(0, 1, h, dtype=np.float32)
            ug, vg = np.meshgrid(u, v)
            lon = (ug - 0.5) * np.float32(math.pi)
            lat = (0.5 - vg) * np.float32(math.pi)
            cos_lat = np.cos(lat)
            x = cos_lat * np.sin(lon)
            y = np.sin(lat)
            z = cos_lat * np.cos(lon)

            R = R.astype(np.float32)
            xr = R[0,0]*x + R[0,1]*y + R[0,2]*z
            yr = R[1,0]*x + R[1,1]*y + R[1,2]*z
            zr = R[2,0]*x + R[2,1]*y + R[2,2]*z

            readout_s = np.float32(rs_ms / 1000.0)
            _deg2rad = np.float32(np.pi / 180.0)
            yaw_c = np.float32(angular_vel[2] * rs_yaw_factor) * _deg2rad
            pitch_c = np.float32(angular_vel[1] * rs_pitch_factor) * _deg2rad
            roll_c = np.float32(angular_vel[0] * rs_roll_factor) * _deg2rad

            _kl = self.config.geoc_klns
            if _kl is not None:
                _c0,_c1,_c2,_c3,_c4 = [np.float32(c) for c in _kl]
                _cx = np.float32(self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctrx)
                _cy = np.float32(self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctry)
                _cd = np.float32(self.config.geoc_cal_dim)
                # KLNS forward
                _zc = np.clip(zr, -1, 1).astype(np.float32)
                _sth = np.sqrt(np.maximum(np.float32(1) - _zc*_zc, np.float32(0)))
                _theta = np.arccos(_zc)
                _th2 = _theta * _theta
                _rf = _theta * (_c0 + _th2*(_c1 + _th2*(_c2 + _th2*(_c3 + _th2*_c4))))
                _ssafe = np.where(_sth < 1e-7, np.float32(1), _sth)
                _phx = np.where(_sth < 1e-7, np.float32(0), xr / _ssafe)
                _phy = np.where(_sth < 1e-7, np.float32(0), yr / _ssafe)
                _sx = _cx + _rf * _phx
                _sy = _cy - _rf * _phy
                _drdth = _c0 + _th2*(3*_c1 + _th2*(5*_c2 + _th2*(7*_c3 + 9*_c4*_th2)))
                _t1 = (_sy - _cy) / _cd * readout_s
                _dsx = yaw_c * _t1 * _drdth
                _dsy = pitch_c * _t1 * _drdth
                _dsx += roll_c * _t1 * (_sy - _cy)
                _dsy += -roll_c * _t1 * (_sx - _cx)
                _sxn = _sx + _dsx; _syn = _sy + _dsy
                # KLNS inverse
                _dx = _sxn - _cx; _dy = -(_syn - _cy)
                _rnew = np.sqrt(_dx*_dx + _dy*_dy).astype(np.float32)
                _thn = _theta.copy()
                for _ in range(2):
                    _t2n = _thn*_thn
                    _rest = _thn*(_c0+_t2n*(_c1+_t2n*(_c2+_t2n*(_c3+_t2n*_c4))))
                    _drdt = _c0+_t2n*(3*_c1+_t2n*(5*_c2+_t2n*(7*_c3+9*_c4*_t2n)))
                    _ok = np.abs(_drdt) > 1e-6
                    _thn = np.where(_ok, _thn+(_rnew-_rest)/np.where(_ok,_drdt,np.float32(1)), _thn)
                    _thn = np.clip(_thn, 0, 3.11)
                _sr = np.where(_rnew < 1e-6, np.float32(1), _rnew)
                xn = (np.sin(_thn) * np.where(_rnew < 1e-6, 0, _dx/_sr)).astype(np.float32)
                yn = (np.sin(_thn) * np.where(_rnew < 1e-6, 0, _dy/_sr)).astype(np.float32)
                zn = np.cos(_thn).astype(np.float32)
            else:
                # 3D rotation fallback when no KLNS calibration
                t = rs_time_map * readout_s
                xn = xr + yaw_c * t * zr - roll_c * t * yr
                yn = yr + roll_c * t * xr - pitch_c * t * zr
                zn = zr - yaw_c * t * xr + pitch_c * t * yr

            lat_n = np.arcsin(np.clip(yn, -1, 1))
            lon_n = np.arctan2(xn, zn)
            mx = np.clip((lon_n / np.float32(math.pi) + 0.5) * np.float32(half_w),
                         0, np.float32(half_w - 1)).astype(np.float32)
            my = np.clip((0.5 - lat_n / np.float32(math.pi)) * np.float32(h),
                         0, np.float32(h - 1)).astype(np.float32)
            return mx, my

        # ── .360 per-eye cross remap path (184.5° FOV) ─────────────────────
        # For .360 input: remap each eye directly from its own EAC cross.
        # This avoids the parallax seam at the lens boundary and gives each eye
        # access to the full 184.5° FOV of its own lens.
        if is_360 and hasattr(self, 'cached_crossA') and hasattr(self, 'cached_crossB'):
            _pv_crossA = self.cached_crossA
            _pv_crossB = self.cached_crossB

            # Track which projection the cached _pv_xyz grid is for. Both
            # equirect and fisheye VR180 per-eye previews are square, so the
            # (h, half_w) key isn't enough to distinguish them — without this
            # flag, switching fisheye→equirect would reuse the fisheye grid.
            _pv_want_mode = 'fisheye' if (self.output_format_combo.currentIndex() == 1
                                          and self.config.geoc_klns) else 'equirect'
            if getattr(self, '_pv_mode', None) != _pv_want_mode:
                # Invalidate the cached preview grid + all GPU upload buffers
                # so the branch below rebuilds from scratch in the new mode.
                for _attr in ('_pv_xyz_x', '_pv_xyz_y', '_pv_xyz_z',
                              '_pv_dims', '_pv_cuda_bufs', '_pv_wgpu_bufs',
                              '_pv_mlx_xyz_x', '_pv_mlx_xyz_y', '_pv_mlx_xyz_z',
                              '_pv_fish_invalid', '_pv_mx_r', '_pv_my_r',
                              '_pv_mx_l', '_pv_my_l'):
                    if hasattr(self, _attr):
                        delattr(self, _attr)
                self._pv_mode = _pv_want_mode

            # Fisheye SBS preview: use fisheye xyz grids instead of equirect
            if self.output_format_combo.currentIndex() == 1 and self.config.geoc_klns:
                _fish_pv_key = ('fisheye_xyz_pv', h)
                if _fish_pv_key not in self._preview_remap_cache:
                    _cal = self.config.geoc_cal_dim or 4216
                    _fish_out = 4096
                    # Preview scales from 4096 output, not 4216 sensor
                    _scale = h / _fish_out
                    _pv_sz = h  # square fisheye per eye
                    _offset = (_cal - _fish_out) / 2.0
                    _cx = (_cal / 2.0 - _offset) * _scale
                    _cy = (_cal / 2.0 - _offset) * _scale
                    _yy, _xx = np.mgrid[0:_pv_sz, 0:_pv_sz].astype(np.float32)
                    _r = np.sqrt((_xx - _cx)**2 + (_yy - _cy)**2)
                    _phi = np.arctan2(_yy - _cy, _xx - _cx)
                    _c0,_c1,_c2,_c3,_c4 = [np.float32(c) for c in self.config.geoc_klns]
                    _theta = _r / (_c0 * _scale)
                    for _ in range(5):
                        _t2 = _theta*_theta
                        _r_est = _theta*(_c0*_scale+_t2*((_c1*_scale)+_t2*((_c2*_scale)+_t2*((_c3*_scale)+_t2*(_c4*_scale)))))
                        _dr = _c0*_scale+_t2*(3*_c1*_scale+_t2*(5*_c2*_scale+_t2*(7*_c3*_scale+_t2*9*_c4*_scale)))
                        _theta -= (_r_est - _r) / np.maximum(_dr, 1e-6)
                        _theta = np.clip(_theta, 0, np.pi)
                    _sin_t=np.sin(_theta); _cos_t=np.cos(_theta)
                    _xn=_sin_t*np.cos(_phi); _yn=_sin_t*np.sin(_phi); _zn=_cos_t
                    _xn=_xn[::-1,:]; _yn=_yn[::-1,:]; _zn=_zn[::-1,:]
                    _max_r = klns_forward(np.float32(np.pi*184.5/360), self.config.geoc_klns) * _scale
                    _inv = (_r >= _max_r)[::-1, :]
                    _xn_f = _xn.ravel().copy(); _yn_f = _yn.ravel().copy(); _zn_f = _zn.ravel().copy()
                    self._preview_remap_cache[_fish_pv_key] = (
                        _xn_f, _yn_f, _zn_f, _pv_sz, _inv)
                    del _xx,_yy,_r,_phi,_theta,_sin_t,_cos_t,_xn,_yn,_zn,_inv

                _fxyz = self._preview_remap_cache[_fish_pv_key]
                self._pv_xyz_x = _fxyz[0]
                self._pv_xyz_y = _fxyz[1]
                self._pv_xyz_z = _fxyz[2]
                _pv_fish_sz = _fxyz[3]
                self._pv_fish_invalid = _fxyz[4] if len(_fxyz) > 4 else None
                self._pv_dims = (_pv_fish_sz, _pv_fish_sz)
                n_pv = len(self._pv_xyz_x)
                half_w = _pv_fish_sz
                h = _pv_fish_sz
                self._pv_mx_r = np.empty(n_pv, dtype=np.float32)
                self._pv_my_r = np.empty(n_pv, dtype=np.float32)
                self._pv_mx_l = np.empty(n_pv, dtype=np.float32)
                self._pv_my_l = np.empty(n_pv, dtype=np.float32)
                # Force re-init of GPU buffers
                if hasattr(self, '_pv_cuda_bufs'): delattr(self, '_pv_cuda_bufs')
                if hasattr(self, '_pv_wgpu_bufs'): delattr(self, '_pv_wgpu_bufs')
                if hasattr(self, '_pv_mlx_xyz_x'): delattr(self, '_pv_mlx_xyz_x')

            # Lazy-init precomputed grids + buffers for preview cross remap (same as render)
            if not hasattr(self, '_pv_xyz_x') or self._pv_dims != (h, half_w):
                u = np.linspace(0, 1, half_w, dtype=np.float32)
                v = np.linspace(0, 1, h, dtype=np.float32)
                ug, vg = np.meshgrid(u, v)
                lon = (ug - 0.5) * np.float32(math.pi)
                lat = (0.5 - vg) * np.float32(math.pi)
                cos_lat = np.cos(lat)
                self._pv_xyz_x = (cos_lat * np.sin(lon)).ravel()
                self._pv_xyz_y = np.sin(lat).ravel()
                self._pv_xyz_z = (cos_lat * np.cos(lon)).ravel()
                n = self._pv_xyz_x.shape[0]
                self._pv_mx_r = np.empty(n, dtype=np.float32)
                self._pv_my_r = np.empty(n, dtype=np.float32)
                self._pv_mx_l = np.empty(n, dtype=np.float32)
                self._pv_my_l = np.empty(n, dtype=np.float32)
                self._pv_dims = (h, half_w)
                # CUDA persistent preview buffers
                self._pv_cuda_bufs = None
                if HAS_NUMBA_CUDA:
                    try:
                        self._pv_cuda_bufs = {
                            'd_xx': _cuda.to_device(self._pv_xyz_x),
                            'd_yy': _cuda.to_device(self._pv_xyz_y),
                            'd_zz': _cuda.to_device(self._pv_xyz_z),
                            'd_zt': _cuda.to_device(np.zeros(n, dtype=np.float32)),
                        }
                    except Exception as e:
                        print(f"CUDA preview buffer init failed: {e}")
                        self._pv_cuda_bufs = None
                # wgpu preview buffers
                self._pv_wgpu_bufs = None
                if HAS_WGPU and _wgpu_init():
                    try:
                        self._pv_wgpu_bufs = {
                            'xyz_x': _wgpu_create_buffer(self._pv_xyz_x),
                            'xyz_y': _wgpu_create_buffer(self._pv_xyz_y),
                            'xyz_z': _wgpu_create_buffer(self._pv_xyz_z),
                            't_offset': _wgpu_create_buffer(np.zeros(n, dtype=np.float32)),
                            'out': _wgpu_create_empty_buffer(n * 3 * 4),
                            'n': n,
                        }
                    except Exception as e:
                        print(f"wgpu preview buffer init failed: {e}")
                        self._pv_wgpu_bufs = None
                # Precompute RS time offset for preview (identity rotation)
                if self.config.geoc_klns is not None:
                    from vr180_gui import klns_forward as _klns_fwd
                    _klns = self.config.geoc_klns
                    _cy = np.float32(self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctry)
                    _c0 = np.float32(_klns[0])
                    _cal_f = np.float32(self.config.geoc_cal_dim)
                    theta_id = np.arccos(np.clip(self._pv_xyz_z, -1, 1))
                    r_fish = _klns_fwd(theta_id, _klns)
                    sin_theta = np.sqrt(np.maximum(1.0 - self._pv_xyz_z**2, 0.0))
                    safe_sin = np.where(sin_theta < 1e-7, 1.0, sin_theta)
                    sensor_y = np.where(sin_theta < 1e-7,
                                        _cy - _c0 * self._pv_xyz_y,
                                        _cy - r_fish * self._pv_xyz_y / safe_sin)
                    self._pv_rs_t_offset = ((sensor_y / _cal_f) - np.float32(0.5)).astype(np.float32)
                    # Upload RS t_offset to wgpu
                    if self._pv_wgpu_bufs is not None:
                        _wgpu_device.queue.write_buffer(
                            self._pv_wgpu_bufs['t_offset'], 0, self._pv_rs_t_offset.tobytes())
                else:
                    self._pv_rs_t_offset = None

            # Build per-eye view adjustment matrices (global+stereo)
            x_right_yaw = global_yaw + stereo_yaw
            x_right_pitch = global_pitch + stereo_pitch
            x_right_roll = global_roll + stereo_roll
            x_left_yaw = global_yaw - stereo_yaw
            x_left_pitch = global_pitch - stereo_pitch
            x_left_roll = global_roll - stereo_roll
            has_right_view = abs(x_right_yaw) > 0.01 or abs(x_right_pitch) > 0.01 or abs(x_right_roll) > 0.01
            has_left_view = abs(x_left_yaw) > 0.01 or abs(x_left_pitch) > 0.01 or abs(x_left_roll) > 0.01
            R_view_right = _build_ypr_matrix(x_right_yaw, x_right_pitch, x_right_roll) if has_right_view else None
            R_view_left = _build_ypr_matrix(x_left_yaw, x_left_pitch, x_left_roll) if has_left_view else None

            R_cam_right = gyro_right.astype(np.float32) if (has_stabilizer and gyro_right is not None) else np.eye(3, dtype=np.float32)
            R_cam_left = gyro_left.astype(np.float32) if (has_stabilizer and gyro_left is not None) else np.eye(3, dtype=np.float32)

            R_left = (R_cam_left @ R_view_left) if R_view_left is not None else R_cam_left
            R_right = (R_cam_right @ R_view_right) if R_view_right is not None else R_cam_right

            rs_ms = self.rs_correction_slider.value()
            rs_on = self.rs_correction_checkbox.isChecked()
            rs_yaw_factor = self.rs_factor_slider.value()
            rs_pitch_factor = self.rs_pitch_factor_slider.value()
            rs_roll_factor = self.rs_roll_factor_slider.value()
            rs_active = (rs_on and rs_ms > 0.01
                         and hasattr(self, '_gyro_stabilizer')
                         and self._gyro_stabilizer is not None
                         and self.config.geoc_klns is not None)

            n_pv = self._pv_xyz_x.shape[0]
            _deg2rad = np.float32(np.pi / 180.0)

            # RIGHT EYE
            if rs_active and self._pv_rs_t_offset is not None:
                rs_time = (self.timeline_slider.value() / 1000.0) * self.video_duration if self.video_duration > 0 else 0
                _pv_use_cori = self.rs_use_cori_checkbox.isChecked()
                angular_vel = self._gyro_stabilizer.get_angular_velocity_at_time(rs_time, use_cori=_pv_use_cori)
                R_heading = self._gyro_stabilizer.get_heading_matrix_at_time(rs_time).astype(np.float32)
                R_sensor_right = (R_heading @ R_view_right) if R_view_right is not None else R_heading
                R_cross_right = self._gyro_stabilizer.get_iori_right_matrix_at_time(rs_time).astype(np.float32)
                has_Rc = np.max(np.abs(R_cross_right - np.eye(3, dtype=np.float32))) > 1e-6

                readout_s = np.float32(rs_ms / 1000.0)
                t_offset_scaled = self._pv_rs_t_offset * readout_s
                yaw_coeff = np.float32(angular_vel[2] * rs_yaw_factor) * _deg2rad
                pitch_coeff = np.float32(angular_vel[1] * rs_pitch_factor) * _deg2rad
                roll_coeff = np.float32(angular_vel[0] * rs_roll_factor) * _deg2rad
                R = R_sensor_right
                Rc = R_cross_right if has_Rc else np.eye(3, dtype=np.float32)

                _pv_wb = getattr(self, '_pv_wgpu_bufs', None)
                _pv_cb = getattr(self, '_pv_cuda_bufs', None)
                # Detect two-step mode for preview
                _pv_two_step = (abs(rs_yaw_factor) < 0.01 and
                               abs(rs_pitch_factor - 2.0) < 0.01 and
                               abs(rs_roll_factor - 2.0) < 0.01)
                if HAS_MLX and not HAS_NUMBA_CUDA:
                    # MLX Metal preview: uses sensor-space RS correction
                    if self.config.geoc_klns:
                        _kl = self.config.geoc_klns
                        _mlx_klns_params[:] = [
                            _kl[0], _kl[1], _kl[2], _kl[3], _kl[4],
                            self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctrx,
                            self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctry,
                            self.config.geoc_cal_dim, rs_ms / 1000.0
                        ]
                    if not hasattr(self, '_pv_mlx_xyz_x'):
                        self._pv_mlx_xyz_x = mx.array(self._pv_xyz_x)
                        self._pv_mlx_xyz_y = mx.array(self._pv_xyz_y)
                        self._pv_mlx_xyz_z = mx.array(self._pv_xyz_z)
                    _pv_mlx_t = mx.array(t_offset_scaled)

                    rs_c = np.array([yaw_coeff, pitch_coeff, roll_coeff], dtype=np.float32)
                    right_out = _mlx_process_eye(
                            _pv_crossA,
                            self._pv_mlx_xyz_x, self._pv_mlx_xyz_y, self._pv_mlx_xyz_z,
                            R, _pv_mlx_t, rs_c,
                            R_cross_right if has_Rc else None, has_Rc, n_pv,
                            pix_max=65535.0, two_step=False)
                    right = right_out.reshape(h, half_w, 3)
                elif HAS_NUMBA_CUDA and _pv_cb is not None:
                    rs_c = np.array([yaw_coeff, pitch_coeff, roll_coeff], dtype=np.float32)
                    _pv_cb['d_zt'].copy_to_device(t_offset_scaled)
                    right = _cuda_process_eye(
                        _pv_crossA, _pv_cb['d_xx'], _pv_cb['d_yy'], _pv_cb['d_zz'],
                        R, _pv_cb['d_zt'], rs_c,
                        R_cross_right if has_Rc else None, has_Rc,
                        n_pv, h, half_w, pix_max=65535)
                elif HAS_WGPU and _pv_wb is not None:
                    rs_c = np.array([yaw_coeff, pitch_coeff, roll_coeff], dtype=np.float32)
                    _wgpu_device.queue.write_buffer(
                        _pv_wb['t_offset'], 0, t_offset_scaled.tobytes())
                    right_out = _wgpu_process_eye(
                        _pv_crossA, _pv_wb['xyz_x'], _pv_wb['xyz_y'], _pv_wb['xyz_z'],
                        R, _pv_wb['t_offset'], rs_c,
                        R_cross_right if has_Rc else None, has_Rc,
                        _pv_wb['n'], _pv_wb['out'])
                    right = right_out.reshape(h, half_w, 3)
                elif HAS_NUMBA:
                    _pv_use_klns = self.config.geoc_klns is not None
                    _pv_kl = self.config.geoc_klns if _pv_use_klns else [0]*5
                    _nb_cross_remap_rs(
                        self._pv_xyz_x, self._pv_xyz_y, self._pv_xyz_z,
                        R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2],
                        t_offset_scaled, yaw_coeff, pitch_coeff, roll_coeff,
                        has_Rc,
                        Rc[0,0], Rc[0,1], Rc[0,2], Rc[1,0], Rc[1,1], Rc[1,2], Rc[2,0], Rc[2,1], Rc[2,2],
                        self._pv_mx_r, self._pv_my_r, n_pv,
                        np.float32(_pv_kl[0]), np.float32(_pv_kl[1]), np.float32(_pv_kl[2]),
                        np.float32(_pv_kl[3]), np.float32(_pv_kl[4]),
                        np.float32(self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctrx) if _pv_use_klns else np.float32(0),
                        np.float32(self.config.geoc_cal_dim / 2.0 + self.config.geoc_ctry) if _pv_use_klns else np.float32(0),
                        np.float32(self.config.geoc_cal_dim) if _pv_use_klns else np.float32(0),
                        np.float32(rs_ms / 1000.0),
                        False)  # 3D rotation (proven best)
                    r_mx = self._pv_mx_r.reshape(h, half_w)
                    r_my = self._pv_my_r.reshape(h, half_w)
                    right = cv2.remap(_pv_crossA, r_mx, r_my,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
                else:
                    r_mx, r_my = FrameExtractor.build_cross_remap_rs(
                        R_sensor_right, half_w, h, angular_vel, rs_ms,
                        rs_yaw_factor, rs_pitch_factor, rs_roll_factor,
                        self.config.geoc_klns, self.config.geoc_ctrx,
                        self.config.geoc_ctry, self.config.geoc_cal_dim,
                        R_cross=R_cross_right)
                    right = cv2.remap(_pv_crossA, r_mx, r_my,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
            else:
                # No RS: rotation-only cross remap
                R = R_right.astype(np.float32)
                _pv_wb = getattr(self, '_pv_wgpu_bufs', None)
                _pv_cb = getattr(self, '_pv_cuda_bufs', None)
                if HAS_NUMBA_CUDA and _pv_cb is not None:
                    right = _cuda_process_eye(
                        _pv_crossA, _pv_cb['d_xx'], _pv_cb['d_yy'], _pv_cb['d_zz'],
                        R, _pv_cb['d_zt'], None,
                        None, False, n_pv, h, half_w, pix_max=65535)
                elif HAS_WGPU and _pv_wb is not None:
                    right_out = _wgpu_process_eye(
                        _pv_crossA, _pv_wb['xyz_x'], _pv_wb['xyz_y'], _pv_wb['xyz_z'],
                        R, _pv_wb['t_offset'], None,
                        None, False, _pv_wb['n'], _pv_wb['out'])
                    right = right_out.reshape(h, half_w, 3)
                elif HAS_NUMBA:
                    _nb_cross_remap_rot(
                        self._pv_xyz_x, self._pv_xyz_y, self._pv_xyz_z,
                        R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2],
                        self._pv_mx_r, self._pv_my_r, n_pv)
                    r_mx = self._pv_mx_r.reshape(h, half_w)
                    r_my = self._pv_my_r.reshape(h, half_w)
                    right = cv2.remap(_pv_crossA, r_mx, r_my,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
                else:
                    r_mx, r_my = FrameExtractor.build_cross_remap(R_right, half_w, h)
                    right = cv2.remap(_pv_crossA, r_mx, r_my,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

            # LEFT EYE — RS when no_firmware_rs, else rotation-only
            R = R_left.astype(np.float32)
            _pv_left_rs = (self.config.no_firmware_rs and rs_active
                           and self._pv_rs_t_offset is not None and angular_vel is not None)
            _pv_wb = getattr(self, '_pv_wgpu_bufs', None)
            _pv_cb = getattr(self, '_pv_cuda_bufs', None)
            if _pv_left_rs:
                # Apply RS to left eye with same angular velocity (no-firmware-RS mode)
                _l_yaw_c = np.float32(angular_vel[2] * rs_yaw_factor) * _deg2rad
                _l_pitch_c = np.float32(angular_vel[1] * rs_pitch_factor) * _deg2rad
                _l_roll_c = np.float32(angular_vel[0] * rs_roll_factor) * _deg2rad
                if HAS_MLX and not HAS_NUMBA_CUDA:
                    _l_rs_c = np.array([_l_yaw_c, _l_pitch_c, _l_roll_c], dtype=np.float32)
                    left_out = _mlx_process_eye(
                        _pv_crossB,
                        self._pv_mlx_xyz_x, self._pv_mlx_xyz_y, self._pv_mlx_xyz_z,
                        R, _pv_mlx_t, _l_rs_c,
                        None, False, n_pv, pix_max=65535.0, two_step=False)
                    left = left_out.reshape(h, half_w, 3)
                elif HAS_NUMBA_CUDA and _pv_cb is not None:
                    _l_rs_c = np.array([_l_yaw_c, _l_pitch_c, _l_roll_c], dtype=np.float32)
                    left = _cuda_process_eye(
                        _pv_crossB, _pv_cb['d_xx'], _pv_cb['d_yy'], _pv_cb['d_zz'],
                        R, _pv_cb['d_zt'], _l_rs_c,
                        None, False, n_pv, h, half_w, pix_max=65535)
                elif HAS_NUMBA:
                    _nb_cross_remap_rs(
                        self._pv_xyz_x, self._pv_xyz_y, self._pv_xyz_z,
                        R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2],
                        t_offset_scaled,
                        _l_yaw_c, _l_pitch_c, _l_roll_c,
                        False,
                        1,0,0, 0,1,0, 0,0,1,
                        self._pv_mx_l, self._pv_my_l, n_pv)
                    l_mx = self._pv_mx_l.reshape(h, half_w)
                    l_my = self._pv_my_l.reshape(h, half_w)
                    left = cv2.remap(_pv_crossB, l_mx, l_my,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
                else:
                    l_mx, l_my = FrameExtractor.build_cross_remap(R_left, half_w, h)
                    left = cv2.remap(_pv_crossB, l_mx, l_my,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
            elif HAS_NUMBA_CUDA and _pv_cb is not None:
                left = _cuda_process_eye(
                    _pv_crossB, _pv_cb['d_xx'], _pv_cb['d_yy'], _pv_cb['d_zz'],
                    R, _pv_cb['d_zt'], None,
                    None, False, n_pv, h, half_w, pix_max=65535)
            elif HAS_WGPU and _pv_wb is not None:
                left_out = _wgpu_process_eye(
                    _pv_crossB, _pv_wb['xyz_x'], _pv_wb['xyz_y'], _pv_wb['xyz_z'],
                    R, _pv_wb['t_offset'], None,
                    None, False, _pv_wb['n'], _pv_wb['out'])
                left = left_out.reshape(h, half_w, 3)
            elif HAS_NUMBA:
                _nb_cross_remap_rot(
                    self._pv_xyz_x, self._pv_xyz_y, self._pv_xyz_z,
                    R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2],
                    self._pv_mx_l, self._pv_my_l, n_pv)
                l_mx = self._pv_mx_l.reshape(h, half_w)
                l_my = self._pv_my_l.reshape(h, half_w)
                left = cv2.remap(_pv_crossB, l_mx, l_my,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
            else:
                l_mx, l_my = FrameExtractor.build_cross_remap(R_left, half_w, h)
                left = cv2.remap(_pv_crossB, l_mx, l_my,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

            result = np.hstack([left, right])

            # Fisheye preview: mask outside circle.
            # Level B pipeline: everything downstream is uint16, so the
            # mask and bitwise_and operate at 16-bit. cv2.bitwise_and is
            # dtype-agnostic and works on uint16.
            if hasattr(self, '_pv_fish_invalid') and self._pv_fish_invalid is not None:
                _pv_mask_1ch = (~self._pv_fish_invalid).astype(np.uint16) * np.uint16(65535)
                _pv_mask_3ch = cv2.merge([_pv_mask_1ch]*3)
                cv2.bitwise_and(result[:, half_w:], _pv_mask_3ch, dst=result[:, half_w:])
                cv2.bitwise_and(result[:, :half_w], _pv_mask_3ch, dst=result[:, :half_w])

        # ── .mov / non-cross remap path ───────────────────────────────────
        else:
            # After base shift: frame[:, :half_w] = RightEye, frame[:, half_w:] = LeftEye
            # Consistent naming: right_eye/left_eye = actual eye content
            right_eye = frame[:, :half_w].copy()
            left_eye = frame[:, half_w:].copy()

            # Check RS activation
            rs_ms = self.rs_correction_slider.value()
            rs_on = self.rs_correction_checkbox.isChecked()
            rs_yaw_factor = self.rs_factor_slider.value()
            rs_pitch_factor = self.rs_pitch_factor_slider.value()
            rs_roll_factor = self.rs_roll_factor_slider.value()
            rs_active = (rs_on and rs_ms > 0.01
                         and hasattr(self, '_gyro_stabilizer')
                         and self._gyro_stabilizer is not None
                         and self.config.geoc_klns is not None)

            # Pre-build RS time map for fused remap (cached)
            _pv_rs_time_map = None
            _pv_rs_angular_vel = None
            if rs_active:
                rs_time = 0
                if self.video_duration > 0:
                    rs_time = (self.timeline_slider.value() / 1000.0) * self.video_duration
                _pv_rs_angular_vel = self._gyro_stabilizer.get_angular_velocity_at_time(rs_time, use_cori=self.rs_use_cori_checkbox.isChecked())
                # Build/cache RS time map from KLNS (same geometry as .360 path)
                _rs_cache_key = ('rs_tmap', h, half_w)
                if _rs_cache_key in self._preview_remap_cache:
                    _pv_rs_time_map = self._preview_remap_cache[_rs_cache_key]
                else:
                    _pv_rs_time_map = build_rs_time_map(h, half_w,
                                                         self.config.geoc_klns,
                                                         ctrx=self.config.geoc_ctrx,
                                                         ctry=self.config.geoc_ctry,
                                                         cal_dim=self.config.geoc_cal_dim)
                    self._preview_remap_cache[_rs_cache_key] = _pv_rs_time_map

            # Build per-eye view adjustment matrices (global+stereo)
            # Same convention as .360 cross path
            has_right_view = abs(right_yaw) > 0.01 or abs(right_pitch) > 0.01 or abs(right_roll) > 0.01
            has_left_view = abs(left_yaw) > 0.01 or abs(left_pitch) > 0.01 or abs(left_roll) > 0.01
            R_view_right = _build_ypr_matrix(right_yaw, right_pitch, right_roll) if has_right_view else None
            R_view_left = _build_ypr_matrix(left_yaw, left_pitch, left_roll) if has_left_view else None

            if gyro_left is not None or gyro_right is not None:
                # Gyro + view: bake user adjustments into gyro matrix (same as .360 path)
                R_right = gyro_right.astype(np.float32) if gyro_right is not None else np.eye(3, dtype=np.float32)
                R_left = gyro_left.astype(np.float32) if gyro_left is not None else np.eye(3, dtype=np.float32)
                if R_view_right is not None:
                    R_right = (R_right @ R_view_right).astype(np.float32)
                if R_view_left is not None:
                    R_left = (R_left @ R_view_left).astype(np.float32)

                # Right eye: fused RS + rotation remap in 3D (matches .360 kernel)
                if _pv_rs_time_map is not None and _pv_rs_angular_vel is not None:
                    rot_mx, rot_my = _build_halfequirect_remap_rs(
                        R_right, half_w, h, _pv_rs_angular_vel, _pv_rs_time_map,
                        rs_ms, rs_yaw_factor, rs_pitch_factor, rs_roll_factor)
                else:
                    rot_mx, rot_my = _build_halfequirect_remap(R_right, half_w, h)
                right_eye = cv2.remap(right_eye, rot_mx, rot_my,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

                # Left eye: RS when no_firmware_rs, else rotation only
                if self.config.no_firmware_rs and _pv_rs_time_map is not None and _pv_rs_angular_vel is not None:
                    rot_mx, rot_my = _build_halfequirect_remap_rs(
                        R_left, half_w, h, _pv_rs_angular_vel, _pv_rs_time_map,
                        rs_ms, rs_yaw_factor, rs_pitch_factor, rs_roll_factor)
                else:
                    rot_mx, rot_my = _build_halfequirect_remap(R_left, half_w, h)
                left_eye = cv2.remap(left_eye, rot_mx, rot_my,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
            else:
                # No gyro: apply manual adjustments directly
                if has_left_view:
                    l_mx, l_my = _build_halfequirect_remap(R_view_left, half_w, h)
                    left_eye = cv2.remap(left_eye, l_mx, l_my,
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
                if has_right_view:
                    r_mx, r_my = _build_halfequirect_remap(R_view_right, half_w, h)
                    right_eye = cv2.remap(right_eye, r_mx, r_my,
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))

            # Combine: left eye on left, right eye on right (standard SBS)
            result = np.hstack([left_eye, right_eye])

        # Apply color adjustments (Lift/Gamma/Gain) using LUT for speed.
        # Level B pipeline: 65536-entry uint16 LUT applied via numba's
        # _nb_apply_1d_lut (33× faster than numpy fancy indexing on uint16).
        # cv2.LUT doesn't support uint16 so it can't be used here.
        lift = self.lift_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0
        gain = self.gain_slider.value() / 100.0

        if abs(lift) > 0.01 or abs(gamma - 1.0) > 0.01 or abs(gain - 1.0) > 0.01:
            # Cache key includes dtype so a late switch between pipelines
            # wouldn't reuse a wrong-width LUT.
            color_key = (round(lift, 3), round(gamma, 3), round(gain, 3), 'u16')
            if not hasattr(self, '_color_lut_cache') or not hasattr(self, '_color_lut_cache_key') or self._color_lut_cache_key != color_key:
                lut_1d = np.arange(65536, dtype=np.float32) / 65535.0
                if abs(lift) > 0.01:
                    lut_1d = lut_1d + lift * (1.0 - lut_1d)
                if abs(gain - 1.0) > 0.01:
                    lut_1d = lut_1d * gain
                lut_1d = np.clip(lut_1d, 0.0, 1.0)
                if abs(gamma - 1.0) > 0.01:
                    lut_1d = np.power(lut_1d, 1.0 / gamma)
                self._color_lut_cache = np.clip(lut_1d * 65535.0, 0, 65535).astype(np.uint16)
                self._color_lut_cache_key = color_key

            # Apply via numba parallel fancy index (uint16) — no cv2.LUT.
            if HAS_NUMBA:
                lut_out = np.empty_like(result)
                _nb_apply_1d_lut(result.ravel(), self._color_lut_cache, lut_out.ravel())
                result = lut_out
            else:
                # Pure numpy fallback — slower but correct
                result = self._color_lut_cache[result]

        # NOTE: saturation is applied below, AFTER the 3D LUT and sharpen,
        # matching the export pipeline order (1D LUT → 3D LUT → sharpen → saturation).
        # Keeping the same order across preview and export ensures WYSIWYG.

        # Apply 3D LUT (uses same optimized path as render: MLX > Numba > numpy)
        # IMPORTANT: all LUT kernels assume BGR channel order (matching export).
        # The preview frame is RGB (from gbrp16le), so we flip R↔B before the
        # LUT lookup and flip back after. This is a view-only operation when
        # contiguous, and a cheap copy otherwise.
        lut_path = self.lut_path_edit.text()
        lut_intensity = self.lut_intensity_slider.value() / 100.0
        if lut_path and Path(lut_path).exists() and lut_intensity > 0.01:
            try:
                # Cache the loaded LUT
                if not hasattr(self, '_cached_lut_path') or self._cached_lut_path != lut_path:
                    self._cached_lut_3d = load_cube_lut(lut_path)
                    self._cached_lut_path = lut_path
                    # Invalidate MLX cache so it gets rebuilt
                    if hasattr(self, '_pv_mlx_lut_flat'):
                        del self._pv_mlx_lut_flat

                lut_3d = self._cached_lut_3d
                lut_size = lut_3d.shape[0]
                n_pixels = result.shape[0] * result.shape[1]

                # RGB → BGR so LUT kernels see the same channel layout as export
                result = np.ascontiguousarray(result[:, :, ::-1])

                # Level B: preview pipeline is uint16 throughout, so we
                # ask each backend to operate at pix_max=65535. cv2.addWeighted
                # works on uint16 for the intensity blend.
                if HAS_MLX:
                    if not hasattr(self, '_pv_mlx_lut_flat'):
                        self._pv_mlx_lut_flat = mx.array(lut_3d.ravel())
                    lut_result = _mlx_apply_lut_3d(result, self._pv_mlx_lut_flat, lut_size, n_pixels, pix_max=65535.0)
                    if lut_intensity < 1.0:
                        result = cv2.addWeighted(result, 1.0 - lut_intensity, lut_result, lut_intensity, 0)
                    else:
                        result = lut_result
                elif HAS_NUMBA_CUDA:
                    # _cuda_apply_lut_3d autodetects dtype from result.dtype
                    lut_result = _cuda_apply_lut_3d(result, lut_3d, lut_size)
                    if lut_intensity < 1.0:
                        result = cv2.addWeighted(result, 1.0 - lut_intensity, lut_result, lut_intensity, 0)
                    else:
                        result = lut_result
                elif HAS_WGPU and _wgpu_device is not None:
                    lut_result = _wgpu_apply_lut_3d(result, lut_3d.ravel(), lut_size, n_pixels)
                    if lut_intensity < 1.0:
                        result = cv2.addWeighted(result, 1.0 - lut_intensity, lut_result, lut_intensity, 0)
                    else:
                        result = lut_result
                elif HAS_NUMBA:
                    lut_result = np.empty_like(result)
                    _nb_apply_lut_3d(result, lut_3d, lut_result, lut_size, pix_max=65535)
                    if lut_intensity < 1.0:
                        result = cv2.addWeighted(result, 1.0 - lut_intensity, lut_result, lut_intensity, 0)
                    else:
                        result = lut_result
                else:
                    # Numpy fallback — full resolution trilinear interpolation
                    img = result.astype(np.float32) / 65535.0
                    b_idx = np.clip(img[:,:,0] * (lut_size - 1), 0, lut_size - 1.001)
                    g_idx = np.clip(img[:,:,1] * (lut_size - 1), 0, lut_size - 1.001)
                    r_idx = np.clip(img[:,:,2] * (lut_size - 1), 0, lut_size - 1.001)
                    b0 = b_idx.astype(np.int32); g0 = g_idx.astype(np.int32); r0 = r_idx.astype(np.int32)
                    b1 = np.minimum(b0+1, lut_size-1); g1 = np.minimum(g0+1, lut_size-1); r1 = np.minimum(r0+1, lut_size-1)
                    fb = (b_idx-b0)[:,:,np.newaxis]; fg = (g_idx-g0)[:,:,np.newaxis]; fr = (r_idx-r0)[:,:,np.newaxis]
                    c00 = lut_3d[b0,g0,r0]*(1-fr) + lut_3d[b0,g0,r1]*fr
                    c01 = lut_3d[b0,g1,r0]*(1-fr) + lut_3d[b0,g1,r1]*fr
                    c10 = lut_3d[b1,g0,r0]*(1-fr) + lut_3d[b1,g0,r1]*fr
                    c11 = lut_3d[b1,g1,r0]*(1-fr) + lut_3d[b1,g1,r1]*fr
                    lut_out = (c00*(1-fg)+c01*fg)*(1-fb) + (c10*(1-fg)+c11*fg)*fb
                    if lut_intensity < 1.0:
                        img_rgb = img[:,:,::-1]
                        lut_out = img_rgb + (lut_out - img_rgb) * lut_intensity
                    result = np.clip(lut_out[:,:,::-1] * 65535.0, 0, 65535).astype(np.uint16)

                # BGR → RGB back to preview colour space for display
                result = np.ascontiguousarray(result[:, :, ::-1])

            except Exception as e:
                self.status_bar.showMessage(f"LUT error: {e}")

        # Apply sharpening (preview) — stackBlur for speed.
        # In equirectangular output we use a cos(lat) per-row falloff to
        # keep pole pixels from over-sharpening (each pole pixel represents
        # a tiny angular region). In fisheye output the projection has
        # roughly uniform angular density, so we use a flat weight and let
        # the slider drive the full strength everywhere inside the circle.
        sharpen_amt = self.sharpen_slider.value() / 100.0
        sharpen_rad = self.sharpen_radius_slider.value() / 10.0 if hasattr(self, 'sharpen_radius_slider') else 1.5
        if sharpen_amt > 0.01 and result is not None:
            h_s, w_s = result.shape[:2]
            _pv_pix_max = 65535.0 if result.dtype == np.uint16 else 255.0
            _pv_is_fisheye = (self.output_format_combo.currentIndex() == 1
                              and getattr(self.config, 'is_360_input', False))

            # Preferred path: call _mlx_sharpen — the SAME Metal shader the
            # export uses. Sigma-based separable Gaussian + latitude-weighted
            # USM in pure float32 inside the kernel, byte-exact-equivalent
            # to the export's sharpen pass. Cache the per-resolution kernel
            # and lat weights so slider drags don't rebuild them.
            #
            # Fallback (no MLX): float32 cv2.stackBlur path. We MUST convert
            # to float32 before stackBlur because cv2.stackBlur on uint16
            # has a precision bug — its integer accumulator drops bits and
            # the blurred mean can be 3-4% off from the source mean at
            # ksize=5, which biases the unsharp diff and brightens the
            # output (~+14% at weight=2.0). The export's CPU fallback already
            # uses this same float32 workaround (see apply_equirect_sharpen
            # ~ line 6265).
            _used_mlx_sharpen = False
            if HAS_MLX and result.dtype == np.uint16:
                # Build kernel + lat-weights matching the export's recipe
                # exactly (sigma-based ksize, separable Gaussian, cos(lat) or
                # uniform-1.0 weights). Cached per (h, sigma_int_x10, mode).
                _sigma = float(sharpen_rad)
                _sigma_key = round(_sigma * 10)
                _mode_key = 'fisheye' if _pv_is_fisheye else 'equirect'
                _sharp_key = ('mlx_sharp', h_s, _sigma_key, _mode_key)
                _cached = self._preview_remap_cache.get(_sharp_key)
                if _cached is None:
                    _ksize = max(3, int(np.ceil(_sigma * 6)) | 1)
                    _x_k = np.arange(_ksize, dtype=np.float32) - _ksize // 2
                    _g1d = np.exp(-0.5 * (_x_k / _sigma) ** 2).astype(np.float32)
                    _g1d /= _g1d.sum()
                    if _pv_is_fisheye:
                        _lat_w = np.ones(h_s, dtype=np.float32)
                    else:
                        _lat_w = np.array([
                            float(np.cos((0.5 - y / h_s) * np.pi))
                            for y in range(h_s)
                        ], dtype=np.float32)
                        _lat_w = np.clip(_lat_w, 0.02, 1.0)
                    _cached = (_g1d, _lat_w, _ksize)
                    if len(self._preview_remap_cache) > 30:
                        self._preview_remap_cache.clear()
                    self._preview_remap_cache[_sharp_key] = _cached
                _g1d, _lat_w, _ksize = _cached
                # _mlx_sharpen takes (frame, g1d, lat_weights, amount, ksize, pix_max)
                # Same call signature the export uses at line 6207.
                try:
                    result = _mlx_sharpen(result, _g1d, _lat_w,
                                          float(sharpen_amt), _ksize,
                                          pix_max=_pv_pix_max)
                    _used_mlx_sharpen = True
                except Exception as _mlx_sharpen_err:
                    print(f"⚠ MLX sharpen failed ({_mlx_sharpen_err}), "
                          f"falling back to cv2 float32 path")

            if not _used_mlx_sharpen:
                # Float32 cv2 fallback
                sk = max(3, int(np.sqrt(12 * sharpen_rad**2 + 1) + 0.5) | 1)
                result_f = result.astype(np.float32)
                if hasattr(cv2, 'stackBlur'):
                    blurred_f = cv2.stackBlur(result_f, (sk, sk))
                else:
                    blurred_f = cv2.GaussianBlur(result_f, (0, 0), sigmaX=sharpen_rad)
                diff = result_f - blurred_f
                if _pv_is_fisheye:
                    _pv_w_scalar = float(np.clip(sharpen_amt, 0.02, 4.0))
                    result = np.clip(result_f + _pv_w_scalar * diff, 0, _pv_pix_max).astype(result.dtype)
                else:
                    _pv_rows = (np.arange(h_s, dtype=np.float32) + 0.5) / h_s
                    _pv_lat = (0.5 - _pv_rows) * np.float32(np.pi)
                    _pv_w = np.clip(np.cos(_pv_lat) * sharpen_amt, 0.02, 4.0).astype(np.float32)[:, np.newaxis, np.newaxis]
                    result = np.clip(result_f + _pv_w * diff, 0, _pv_pix_max).astype(result.dtype)

        # Apply saturation (post-LUT, post-sharpen — matches export pipeline)
        saturation = self.saturation_slider.value() / 100.0
        if abs(saturation - 1.0) > 0.01 and result is not None:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(gray_3ch, 1.0 - saturation, result, saturation, 0)

        # Apply circular edge mask per eye (after color/LUT).
        # Level B: pipeline is uint16, so we blend in uint32 arithmetic
        # (safe for uint16 * uint32 with a 16-bit shift). The cached mask
        # stores values in 0..65536 so the shift is a clean >>16 divide.
        mask_size = self.mask_size_slider.value()
        mask_feather = self.mask_feather_slider.value()
        if mask_size < 100.0:
            h_r, w_r = result.shape[:2]
            hw = w_r // 2
            # Include dtype in key so older 'u8' caches don't leak into
            # the new uint16 pipeline (and vice versa on future rollback).
            mask_key = (h_r, hw, mask_size, mask_feather, 'u16')
            if not hasattr(self, '_pv_mask_cache_key') or self._pv_mask_cache_key != mask_key:
                yy, xx = np.mgrid[:h_r, :hw]
                u = (xx + 0.5) / hw * 2.0 - 1.0
                v = (yy + 0.5) / h_r * 2.0 - 1.0
                lon = u * (np.pi / 2.0)
                lat = v * (np.pi / 2.0)
                cos_ang = np.clip(np.cos(lat) * np.cos(lon), -1.0, 1.0)
                r = np.arccos(cos_ang) / (np.pi / 2.0)
                r_inner = mask_size / 100.0
                r_feath = mask_feather / 100.0
                r_outer = r_inner + r_feath
                if r_feath > 0.001:
                    m = np.clip((r_outer - r) / r_feath, 0.0, 1.0)
                else:
                    m = (r <= r_inner).astype(np.float32)
                # Fixed-point mask: 0..65536 (17 bits) → clean >>16 blend.
                # 65535 * 65536 = 4294901760, fits in uint32 with no overflow.
                m_u32 = (m * 65536.0).astype(np.uint32)
                self._pv_mask_cache = np.stack([m_u32, m_u32, m_u32], axis=-1)
                self._pv_mask_cache_key = mask_key
            emask = self._pv_mask_cache
            result[:, hw:] = ((result[:, hw:].astype(np.uint32) * emask) >> 16).astype(np.uint16)
            result[:, :hw] = ((result[:, :hw].astype(np.uint32) * emask) >> 16).astype(np.uint16)

        # Upside-down mount: rotate 180° (flips image + swaps L/R eyes)
        if self.upside_down_checkbox.isChecked():
            result = cv2.rotate(result, cv2.ROTATE_180)

        self.original_frame = result
        self._update_preview()
        self.status_bar.showMessage("Ready")

    def _on_frame_extracted(self, frame):
        """Legacy callback for filtered frame extraction"""
        self.original_frame = frame
        self._update_preview()
    
    def _schedule_preview_update(self):
        # With OpenCV processing, we can update much faster
        # Use short debounce just to avoid too many updates while dragging
        self.preview_timer.start(50)

    def _on_preview_timer(self):
        if self.config.input_path and self.cached_raw_frame is not None:
            # Apply OpenCV filters directly to cached frame (instant)
            self._apply_preview_filters_to_cached_frame()
        elif self.config.input_path:
            # No cached frame yet, need to extract
            self._extract_frame(self.preview_timestamp)
    
    def _update_preview(self):
        if self.original_frame is None: return
        frame = self.original_frame.copy()
        h, w = frame.shape[:2]
        hw = w // 2
        mode = self.preview_mode_combo.currentData()
        left, right = frame[:, :hw], frame[:, hw:]

        # Level B: frame is uint16 (via the PyAV hwaccel + gbrp16le pipeline)
        # unless a caller has set original_frame to uint8 elsewhere. Preview
        # modes below branch on dtype so both work.
        _pv_dtype = frame.dtype
        _pix_max_int = 65535 if _pv_dtype == np.uint16 else 255

        if mode == PreviewMode.SIDE_BY_SIDE: preview = frame
        elif mode == PreviewMode.ANAGLYPH:
            preview = np.zeros_like(left)
            preview[:,:,0] = left[:,:,0]
            preview[:,:,1] = right[:,:,1]
            preview[:,:,2] = right[:,:,2]
        elif mode == PreviewMode.OVERLAY_50:
            preview = ((left.astype(np.float32) * 0.5 + right.astype(np.float32) * 0.5)).astype(_pv_dtype)
        elif mode == PreviewMode.SINGLE_EYE: preview = left if self.current_eye == "left" else right
        elif mode == PreviewMode.DIFFERENCE:
            preview = np.clip(np.abs(left.astype(np.float32) - right.astype(np.float32)) * 3, 0, _pix_max_int).astype(_pv_dtype)
        elif mode == PreviewMode.CHECKERBOARD:
            preview = np.zeros_like(left)
            bs = 64
            for y in range(0, h, bs):
                for x in range(0, hw, bs):
                    ye, xe = min(y+bs, h), min(x+bs, hw)
                    if ((y//bs) + (x//bs)) % 2 == 0: preview[y:ye, x:xe] = left[y:ye, x:xe]
                    else: preview[y:ye, x:xe] = right[y:ye, x:xe]
        else: preview = frame

        # Downconvert uint16 → uint8 for Qt display. QImage.Format_RGB888 is
        # 8-bit per channel so the 16→8 drop happens exactly at this boundary
        # (and nowhere else in the preview pipeline — all filters above run
        # on the full uint16 data). Rounded integer divide preserves precision
        # as well as an 8-bit display can show.
        if preview.dtype == np.uint16:
            preview_u8 = ((preview.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
        else:
            preview_u8 = preview
        if not preview_u8.flags['C_CONTIGUOUS']:
            preview_u8 = np.ascontiguousarray(preview_u8)

        ph, pw = preview_u8.shape[:2]
        qimg = QImage(preview_u8.data.tobytes(), pw, ph, 3*pw, QImage.Format.Format_RGB888)
        self.preview_widget.set_frame(QPixmap.fromImage(qimg))
    
    def _zoom(self, factor):
        old_zoom = self.preview_widget.zoom_level
        self.preview_widget.zoom_level = max(0.25, min(4.0, self.preview_widget.zoom_level * factor))
        self.zoom_label.setText(f"{int(self.preview_widget.zoom_level * 100)}%")

        # Update cursor based on zoom level
        if self.preview_widget.zoom_level > 1.0 and old_zoom <= 1.0:
            self.preview_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.preview_widget.zoom_level <= 1.0 and old_zoom > 1.0:
            self.preview_widget.setCursor(Qt.CursorShape.ArrowCursor)
            self.preview_widget.pan_offset_x = 0
            self.preview_widget.pan_offset_y = 0

        self.preview_widget._update_display()

    def _zoom_reset(self):
        self.preview_widget.zoom_level = 1.0
        self.preview_widget.pan_offset_x = 0
        self.preview_widget.pan_offset_y = 0
        self.zoom_label.setText("100%")
        self.preview_widget.setCursor(Qt.CursorShape.ArrowCursor)
        self.preview_widget._update_display()
    
    def _timeline_dragging(self, value):
        if self.video_duration > 0:
            self.time_label.setText(f"{self._format_time((value/1000)*self.video_duration)} / {self._format_time(self.video_duration)}")
    
    def _timeline_changed(self, value):
        if self.video_duration > 0:
            ts = (value / 1000) * self.video_duration
            self.time_label.setText(f"{self._format_time(ts)} / {self._format_time(self.video_duration)}")
            self._extract_frame(ts)
    
    def _format_time(self, s): return f"{int(s//60):02d}:{int(s%60):02d}"

    def _format_time_full(self, s):
        """Format time as HH:MM:SS.mmm"""
        hours = int(s // 3600)
        minutes = int((s % 3600) // 60)
        seconds = s % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _parse_time(self, time_str):
        """Parse time string (HH:MM:SS.mmm or seconds) to seconds"""
        time_str = time_str.strip()
        try:
            # Try parsing as plain seconds first
            return float(time_str)
        except ValueError:
            pass

        try:
            # Try parsing as HH:MM:SS.mmm or MM:SS.mmm or SS.mmm
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            elif len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 1:
                return float(parts[0])
        except (ValueError, IndexError):
            pass

        return None

    def _set_in_point(self):
        """Set trim start point at current timeline position"""
        if self.video_duration > 0:
            current_time = (self.timeline_slider.value() / 1000) * self.video_duration
            self.trim_start = current_time
            self.in_time_edit.setText(self._format_time_full(current_time))
            self._update_trim_duration()

    def _set_out_point(self):
        """Set trim end point at current timeline position"""
        if self.video_duration > 0:
            current_time = (self.timeline_slider.value() / 1000) * self.video_duration
            self.trim_end = current_time
            self.out_time_edit.setText(self._format_time_full(current_time))
            self._update_trim_duration()

    def _clear_trim(self):
        """Clear trim points"""
        self.trim_start = 0.0
        self.trim_end = 0.0
        self.in_time_edit.setText("00:00:00.000")
        self.out_time_edit.setText(self._format_time_full(self.video_duration) if self.video_duration > 0 else "00:00:00.000")
        self.timeline_slider.clearTrim()
        self._update_trim_duration()

    def _in_time_edited(self):
        """Handle manual edit of in time"""
        time_val = self._parse_time(self.in_time_edit.text())
        if time_val is not None:
            self.trim_start = max(0, min(time_val, self.video_duration if self.video_duration > 0 else time_val))
            self.in_time_edit.setText(self._format_time_full(self.trim_start))
            self._update_trim_duration()
        else:
            # Reset to current value
            self.in_time_edit.setText(self._format_time_full(self.trim_start))

    def _out_time_edited(self):
        """Handle manual edit of out time"""
        time_val = self._parse_time(self.out_time_edit.text())
        if time_val is not None:
            max_time = self.video_duration if self.video_duration > 0 else time_val
            self.trim_end = max(0, min(time_val, max_time))
            self.out_time_edit.setText(self._format_time_full(self.trim_end))
            self._update_trim_duration()
        else:
            # Reset to current value
            self.out_time_edit.setText(self._format_time_full(self.trim_end))

    def _update_trim_duration(self):
        """Update the trim duration label and timeline visualization"""
        if self.trim_end > self.trim_start:
            duration = self.trim_end - self.trim_start
            self.trim_duration_label.setText(self._format_time(duration))
            # Update timeline visualization
            if self.video_duration > 0:
                start_norm = self.trim_start / self.video_duration
                end_norm = self.trim_end / self.video_duration
                self.timeline_slider.setTrimRange(start_norm, end_norm, True)
        elif self.video_duration > 0 and self.trim_start == 0 and self.trim_end == 0:
            self.trim_duration_label.setText(self._format_time(self.video_duration))
            self.timeline_slider.clearTrim()
        else:
            self.trim_duration_label.setText("--:--")
            self.timeline_slider.clearTrim()

    def _inject_only_output(self):
        """Inject VR180 metadata into the input file and save as the output file.

        Bypasses all processing — just copies input to output with YouTube VR180 metadata.
        """
        if not self.config.input_path:
            QMessageBox.warning(self, "No Input File", "Please load an input video first.")
            return
        output_path = self.output_path_edit.text()
        if not output_path:
            QMessageBox.warning(self, "No Output File", "Please set an output file path first.")
            return
        input_file = Path(self.config.input_path)
        output_file = Path(output_path)
        try:
            self.statusBar().showMessage("Injecting VR180 metadata (no processing)...")
            from spatialmedia import metadata_utils

            metadata = metadata_utils.Metadata()
            metadata.stereo = "left-right"
            metadata.spherical = "equirectangular"
            metadata.clip_left_right = 1073741823
            metadata.orientation = {"yaw": 0, "pitch": 0, "roll": 0}

            metadata_utils.inject_metadata(str(input_file), str(output_file), metadata, lambda msg: None)

            self.statusBar().showMessage("VR180 metadata injected successfully")
            QMessageBox.information(self, "Done", f"VR180 metadata injected:\n{input_file.name} → {output_file.name}")
        except Exception as e:
            self.statusBar().showMessage(f"Injection failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to inject metadata:\n{str(e)}")

    def _reset_all(self):
        for w in [self.global_shift_slider, self.global_yaw, self.global_pitch, self.global_roll,
                  self.stereo_yaw_offset, self.stereo_pitch_offset, self.stereo_roll_offset]:
            w.setValue(0)
    
    def _start_processing(self):
        codec_map = {0: "auto", 1: "h265", 2: "prores"}
        prores_map = {0: "proxy", 1: "lt", 2: "standard", 3: "hq", 4: "4444", 5: "4444xq"}
        # Get LUT path if specified
        lut_path = None
        if self.lut_path_edit.text():
            lut_path = Path(self.lut_path_edit.text())

        # Determine trim end (0 means use full duration)
        trim_end = self.trim_end if self.trim_end > 0 else self.video_duration

        config = ProcessingConfig(
            input_path=self.config.input_path, output_path=Path(self.output_path_edit.text()),
            global_shift=self.global_shift_slider.value(),
            global_adjustment=PanomapAdjustment(self.global_yaw.value(), self.global_pitch.value(), self.global_roll.value()),
            stereo_offset=PanomapAdjustment(self.stereo_yaw_offset.value(), self.stereo_pitch_offset.value(), self.stereo_roll_offset.value()),
            output_codec=codec_map[self.codec_combo.currentIndex()], quality=self.quality_spinbox.value(),
            bitrate=self.bitrate_spinbox.value(), use_bitrate=self.use_bitrate_radio.isChecked(),
            prores_profile=prores_map[self.prores_combo.currentIndex()],
            encoder_type=self._encoder_map[self.encoder_combo.currentIndex()],
            lut_path=lut_path,
            lut_intensity=self.lut_intensity_slider.value() / 100.0,
            saturation=self.saturation_slider.value() / 100.0,
            lift=self.lift_slider.value() / 100.0,
            gamma=self.gamma_slider.value() / 100.0,
            gain=self.gain_slider.value() / 100.0,
            h265_bit_depth=10 if self.h265_bit_depth_combo.currentIndex() == 1 else 8,
            inject_vr180_metadata=self.vr180_metadata_checkbox.isChecked(),
            vision_pro_mode=["standard", "hvc1", "mvhevc"][self.vision_pro_combo.currentIndex()],
            trim_start=self.trim_start,
            trim_end=trim_end,
            gyro_data=self.config.gyro_data,
            gyro_smooth_ms=self.gyro_smooth_slider.value(),
            gyro_horizon_lock=self.gyro_horizon_lock_checkbox.isChecked(),
            gyro_max_corr_deg=self.gyro_max_corr_slider.value(),
            gyro_responsiveness=self.gyro_responsiveness_slider.value(),
            gyro_stabilize=self.gyro_stabilize_checkbox.isChecked(),
            is_360_input=self.config.is_360_input,
            rs_correction_ms=self.rs_correction_slider.value(),
            rs_correction_enabled=self.rs_correction_checkbox.isChecked(),
            rs_yaw_factor=self.rs_factor_slider.value(),
            rs_pitch_factor=self.rs_pitch_factor_slider.value(),
            rs_roll_factor=self.rs_roll_factor_slider.value(),
            rs_use_cori=self.rs_use_cori_checkbox.isChecked(),
            no_firmware_rs=self.config.no_firmware_rs,
            output_format=["equirect", "fisheye"][self.output_format_combo.currentIndex()],
            audio_ambisonics=self.audio_format_combo.currentData() == "ambisonic",
            geoc_klns=self.config.geoc_klns,
            geoc_ctrx=self.config.geoc_ctrx,
            geoc_ctry=self.config.geoc_ctry,
            geoc_cal_dim=self.config.geoc_cal_dim,
            mask_size=self.mask_size_slider.value(),
            mask_feather=self.mask_feather_slider.value(),
            edge_fill=True,
            eac_out_w=self.config.eac_out_w,
            eac_out_h=self.config.eac_out_h,
            sharpen_amount=self.sharpen_slider.value() / 100.0,
            sharpen_radius=self.sharpen_radius_slider.value() / 10.0,
            denoise_strength=self.denoise_slider.value() / 100.0,
            segment_paths=self.config.segment_paths,
            upside_down=self.upside_down_checkbox.isChecked())
        
        if not config.input_path or not config.input_path.exists():
            QMessageBox.warning(self, "Error", "Select input file"); return
        if config.output_path.exists():
            if QMessageBox.question(self, "Overwrite?", f"Overwrite {config.output_path}?") != QMessageBox.StandardButton.Yes: return

        # Warn Windows GPU users about 8640×4320 software fallback
        if (sys.platform == 'win32' and config.encoder_type != 'software'
                and max(config.eac_out_w, config.eac_out_h) > 8192
                and config.is_360_input):
            reply = QMessageBox.question(
                self, "Software Encoding Required",
                f"Output resolution {config.eac_out_w}×{config.eac_out_h} exceeds the GPU encoder limit (8192px).\n\n"
                "The export will use software encoding (libx265/libx264) which is significantly slower.\n\n"
                "Continue with software encoding?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.ffmpeg_output.setHtml('<span style="color:#d4d4d4;">Starting export...</span>')
        self.processor = VideoProcessor(config)
        self.processor.output_line.connect(self._append_ffmpeg_output)
        self.processor.status.connect(self.status_bar.showMessage)
        self.processor.finished_signal.connect(self._on_finished)
        self.processor.start()
    
    def _cancel_processing(self):
        if hasattr(self, 'processor'): self.processor.cancel()

    def _append_ffmpeg_output(self, line):
        """Update the info panel with export status (replaces content for live stats)."""
        if line.startswith("Export complete"):
            # Completion summary: append to keep visible
            self.ffmpeg_output.append(f'<span style="color:#4ec9b0;">{line}</span>')
        else:
            # Live stats: replace content
            self.ffmpeg_output.setHtml(f'<span style="color:#d4d4d4;">{line}</span>')


    def _on_finished(self, success, msg):
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success: QMessageBox.information(self, "Success", msg)
        else: QMessageBox.warning(self, "Failed", msg)
        self.status_bar.showMessage(msg)

    def _load_settings(self):
        """Load saved settings from previous session"""
        # Load stereo offset values
        self.stereo_yaw_offset.setValue(self.settings.value("stereo_yaw", 0.0, type=float))
        self.stereo_pitch_offset.setValue(self.settings.value("stereo_pitch", 0.0, type=float))
        self.stereo_roll_offset.setValue(self.settings.value("stereo_roll", 0.0, type=float))

        # Load global adjustment values
        self.global_yaw.setValue(self.settings.value("global_yaw", 0.0, type=float))
        self.global_pitch.setValue(self.settings.value("global_pitch", 0.0, type=float))
        self.global_roll.setValue(self.settings.value("global_roll", 0.0, type=float))

        # Load shift value
        self.global_shift_slider.setValue(self.settings.value("global_shift", 0, type=int))

        # Load encoder type setting
        self.encoder_combo.setCurrentIndex(self.settings.value("encoder_type", 0, type=int))

        # Load color adjustment settings (ASC CDL: Lift, Gamma, Gain)
        self.lift_slider.setValue(self.settings.value("lift", 0, type=int))
        self.gamma_slider.setValue(self.settings.value("gamma", 100, type=int))
        self.gain_slider.setValue(self.settings.value("gain", 100, type=int))
        self.saturation_slider.setValue(self.settings.value("saturation", 100, type=int))

        # Load VR180 metadata setting
        self.vr180_metadata_checkbox.setChecked(self.settings.value("vr180_metadata", False, type=bool))

        # Load Vision Pro mode setting
        self.vision_pro_combo.setCurrentIndex(self.settings.value("vision_pro_mode", 0, type=int))

        # Load H.265 bit depth setting
        self.h265_bit_depth_combo.setCurrentIndex(self.settings.value("h265_bit_depth", 0, type=int))

        # Load resolution and sharpen settings
        self.resolution_combo.setCurrentIndex(self.settings.value("resolution_idx", 0, type=int))
        self._on_resolution_changed()  # Apply to config
        self.sharpen_slider.setValue(self.settings.value("sharpen", 0, type=int))
        self.sharpen_radius_slider.setValue(self.settings.value("sharpen_radius", 13, type=int))
        self.denoise_slider.setValue(self.settings.value("denoise", 0, type=int))

        # Load LUT path. Sentinel "__UNSET__" lets us detect first launch vs. a
        # previous session where the user deliberately cleared the field.
        # On first launch we no longer eagerly pre-populate the bundled GP Log
        # LUT — it will be auto-loaded when a .360 file is opened instead, so
        # non-360 users don't get an inappropriate colour transform.
        saved_lut = self.settings.value("lut_path", "__UNSET__", type=str)
        if saved_lut == "__UNSET__":
            # First launch — leave empty; _browse_input / dropEvent will
            # auto-populate the bundled LUT when a .360 file is loaded.
            pass
        elif saved_lut:
            # Previous session had a LUT set — restore it if the file still
            # exists, otherwise fall back to the bundled LUT if available.
            if Path(saved_lut).exists():
                self.lut_path_edit.setText(saved_lut)
            else:
                bundled = get_bundled_lut_path()
                if bundled:
                    self.lut_path_edit.setText(bundled)
                    print(f"[LUT] Previously saved LUT missing, reverted to bundled: {bundled}")
                else:
                    self.lut_path_edit.clear()
        # else: previous session had an empty LUT path — honor that choice.
        # Load LUT intensity
        self.lut_intensity_slider.setValue(self.settings.value("lut_intensity", 100, type=int))

    def _save_settings(self):
        """Save current settings for next session"""
        # Save stereo offset values
        self.settings.setValue("stereo_yaw", self.stereo_yaw_offset.value())
        self.settings.setValue("stereo_pitch", self.stereo_pitch_offset.value())
        self.settings.setValue("stereo_roll", self.stereo_roll_offset.value())

        # Save global adjustment values
        self.settings.setValue("global_yaw", self.global_yaw.value())
        self.settings.setValue("global_pitch", self.global_pitch.value())
        self.settings.setValue("global_roll", self.global_roll.value())

        # Save shift value
        self.settings.setValue("global_shift", self.global_shift_slider.value())

        # Save encoder type setting
        self.settings.setValue("encoder_type", self.encoder_combo.currentIndex())

        # Save color adjustment settings (ASC CDL: Lift, Gamma, Gain)
        self.settings.setValue("lift", self.lift_slider.value())
        self.settings.setValue("gamma", self.gamma_slider.value())
        self.settings.setValue("gain", self.gain_slider.value())
        self.settings.setValue("saturation", self.saturation_slider.value())

        # Save VR180 metadata setting
        self.settings.setValue("vr180_metadata", self.vr180_metadata_checkbox.isChecked())

        # Save Vision Pro mode setting
        self.settings.setValue("vision_pro_mode", self.vision_pro_combo.currentIndex())

        # Save H.265 bit depth setting
        self.settings.setValue("h265_bit_depth", self.h265_bit_depth_combo.currentIndex())

        # Save resolution and sharpen settings
        self.settings.setValue("resolution_idx", self.resolution_combo.currentIndex())
        self.settings.setValue("sharpen", self.sharpen_slider.value())
        self.settings.setValue("sharpen_radius", self.sharpen_radius_slider.value())
        self.settings.setValue("denoise", self.denoise_slider.value())

        # Save LUT path + intensity so they survive across sessions
        self.settings.setValue("lut_path", self.lut_path_edit.text())
        self.settings.setValue("lut_intensity", self.lut_intensity_slider.value())

    def dragEnterEvent(self, event):
        """Handle drag enter event - accept video files

        IMPORTANT: Always accept drag events at the main window level.
        Qt's drag/drop system will automatically reject drops on child widgets
        unless we accept at the top level first.
        """
        try:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if len(urls) == 1:
                    file_path = urls[0].toLocalFile()
                    # Accept common video file extensions
                    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.mts', '.m2ts', '.osv', '.360')):
                        event.accept()
                        return
            # Always ignore drag events we don't want (this prevents child widgets from blocking)
            event.ignore()
        except Exception as e:
            print(f"Error in dragEnterEvent: {e}")
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event - load the video file

        This event fires when user releases the mouse button to drop a file.
        We accept the drop regardless of which child widget is under the cursor.
        """
        try:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if len(urls) == 1:
                    file_path = urls[0].toLocalFile()
                    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.mts', '.m2ts', '.osv', '.360')):
                        self.config.input_path = Path(file_path)
                        self.config.is_360_input = file_path.lower().endswith('.360')
                        # Detect multi-segment GoPro recordings and ask user
                        if self.config.is_360_input:
                            segments = detect_gopro_segments(file_path)
                            if len(segments) > 1:
                                seg_names = [Path(s).name for s in segments]
                                msg = QMessageBox(self)
                                msg.setWindowTitle("Multi-Segment Recording Detected")
                                msg.setText(
                                    f"Found {len(segments)} segments for this recording:\n"
                                    + "\n".join(f"  {name}" for name in seg_names)
                                )
                                msg.setInformativeText("Import all segments as one combined clip, or just this file?")
                                combine_btn = msg.addButton("Combined Clip", QMessageBox.ButtonRole.AcceptRole)
                                single_btn = msg.addButton("This File Only", QMessageBox.ButtonRole.RejectRole)
                                msg.setDefaultButton(combine_btn)
                                msg.exec()
                                if msg.clickedButton() == combine_btn:
                                    self.config.segment_paths = segments
                                    stem = Path(segments[0]).stem
                                    output = Path(file_path).parent / f"{stem}_combined.mov"
                                    self.input_path_edit.setText(f"{file_path} (+{len(segments)-1} segments)")
                                else:
                                    self.config.segment_paths = None
                                    output = Path(file_path).parent / f"{Path(file_path).stem}_adjusted.mov"
                            else:
                                self.config.segment_paths = None
                                output = Path(file_path).parent / f"{Path(file_path).stem}_adjusted.mov"
                        else:
                            self.config.segment_paths = None
                            output = Path(file_path).parent / f"{Path(file_path).stem}_adjusted.mov"
                        self.output_path_edit.setText(str(output))
                        self.config.output_path = output
                        self._load_video_info()
                        self.process_btn.setEnabled(True)
                        # Show/hide audio format option based on input type
                        self.audio_format_label.setVisible(self.config.is_360_input)
                        self.audio_format_combo.setVisible(self.config.is_360_input)
                        self.resolution_label.setVisible(self.config.is_360_input)
                        self.resolution_combo.setVisible(self.config.is_360_input)
                        self.output_format_label.setVisible(self.config.is_360_input)
                        self.output_format_combo.setVisible(self.config.is_360_input)
                        if not self.config.is_360_input:
                            self.audio_format_combo.setCurrentIndex(0)
                        # Auto-load GPMF gyro data from .360 file
                        if self.config.is_360_input:
                            self._auto_load_360_gyro(file_path)
                            # Auto-populate bundled LUT for .360 (GP Log) if none set
                            if not self.lut_path_edit.text():
                                bundled = get_bundled_lut_path()
                                if bundled:
                                    self.lut_path_edit.setText(bundled)
                            self._schedule_preview_update()
                        else:
                            # Clear bundled LUT for non-360 input (it's GP Log specific)
                            bundled = get_bundled_lut_path()
                            if bundled and self.lut_path_edit.text() == bundled:
                                self.lut_path_edit.clear()
                            self._schedule_preview_update()
                        event.accept()
                        return
            event.ignore()
        except Exception as e:
            print(f"Error in dropEvent: {e}")
            import traceback
            traceback.print_exc()
            event.ignore()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # I key - set in point
        if event.key() == Qt.Key.Key_I and not event.modifiers():
            self._set_in_point()
            return
        # O key - set out point
        if event.key() == Qt.Key.Key_O and not event.modifiers():
            self._set_out_point()
            return
        # Pass to parent for other keys
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Save settings and cleanup when window closes"""
        # Stop any running processing
        if hasattr(self, 'processor') and self.processor and self.processor.isRunning():
            self.processor.cancel()
            self.processor.wait(3000)  # Wait up to 3 seconds
            if self.processor.isRunning():
                self.processor.terminate()
                self.processor.wait(1000)

        # Stop any frame extraction
        if hasattr(self, 'extractor') and self.extractor and self.extractor.isRunning():
            self.extractor.cancel()  # Kill FFmpeg process properly
            self.extractor.wait(1000)

        self._save_settings()
        super().closeEvent(event)


_bt709_colorspace = None  # Cached NSColorSpace, loaded once

def _setup_bt709_window_colorspace(window):
    """Set the main NSWindow's color space to BT.709 for display-accurate preview."""
    global _bt709_colorspace
    if sys.platform != 'darwin':
        return
    try:
        from AppKit import NSApplication, NSColorSpace
        from Foundation import NSData
        if _bt709_colorspace is None:
            icc_path = '/System/Library/ColorSync/Profiles/ITU-709.icc'
            if not os.path.exists(icc_path):
                return
            with open(icc_path, 'rb') as f:
                icc_data = f.read()
            ns_data = NSData.dataWithBytes_length_(icc_data, len(icc_data))
            _bt709_colorspace = NSColorSpace.alloc().initWithICCProfileData_(ns_data)
            if _bt709_colorspace is None:
                return
            print("[ColorMgmt] BT.709 NSColorSpace loaded")
        for nswindow in NSApplication.sharedApplication().windows():
            nswindow.setColorSpace_(_bt709_colorspace)
    except ImportError:
        pass
    except Exception as e:
        print(f"[ColorMgmt] Failed: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VR180ProcessorGUI()
    window.show()
    # Set BT.709 BEFORE the first paint event. window.show() creates the
    # NSWindow synchronously but doesn't paint until app.exec() runs the
    # event loop. By setting the color space here (no processEvents!),
    # the very first paint is already under BT.709 — no sRGB widget cache
    # is ever created, so there's nothing stale to flicker against.
    _setup_bt709_window_colorspace(window)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

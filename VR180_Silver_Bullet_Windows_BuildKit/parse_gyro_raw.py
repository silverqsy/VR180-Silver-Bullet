#!/usr/bin/env python3
"""Parse raw GYRO + ACCL from GoPro .360 GPMF metadata.

Extracts high-rate gyroscope (angular velocity) and accelerometer data,
then integrates gyro to reconstruct camera orientation quaternions.

This replaces CORI-based stabilization when CORI is zeroed out.

IMPORTANT: GYRO data from GPMF is in **rad/s** after SCAL division (not deg/s).

ORIN="ZXY" body frame: raw sensor channels [0]=Z, [1]=X, [2]=Y.
Integration uses correct body frame: col0=bodyX←raw[1], col1=bodyY←raw[2], col2=bodyZ←raw[0].
CORI in file stores (w, x, Z, Y) — y/z slots swapped from standard.
After integration, quaternion Y↔Z components are swapped to match CORI convention.
(Quaternion multiply is NOT covariant under Y↔Z swap, so integration must
use correct body frame first, then transform output.)
"""
import sys, os, struct, subprocess, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vr180_gui import get_ffmpeg_path, get_ffprobe_path, get_subprocess_flags
except ImportError:
    from vr180_gui_fish import get_ffmpeg_path, get_ffprobe_path, get_subprocess_flags

FILE_NEW = "/Users/siyangqi/Downloads/vr180_processor/GS010173.360"
FILE_OLD = "/Users/siyangqi/Downloads/vr180_processor/GS010172.360"

# GoPro MAX GYRO axis mapping: ORIN="ZXY" → raw[0]=Z, raw[1]=X, raw[2]=Y
# Correct body frame for integration: bodyX←raw[1], bodyY←raw[2], bodyZ←raw[0]
# Post-integration Y↔Z swap is applied in gyro_to_cori_quats() to match CORI convention
# Verified: 2.67° RMS vs CORI on GS010161.360 (with STMP alignment)
GYRO_AXIS_MAP = (1, 2, 0)       # col0←raw[1]=bodyX, col1←raw[2]=bodyY, col2←raw[0]=bodyZ
GYRO_AXIS_SIGN = (1.0, 1.0, 1.0)  # all positive
# ACCL uses same ORIN="ZXY" and same body frame mapping
ACCL_AXIS_MAP = (1, 2, 0)
ACCL_AXIS_SIGN = (1.0, 1.0, 1.0)
# GRAV (gravity vector) uses different axis order: raw format is (X, Z, Y) in body frame
# bodyX←raw[0], bodyY←raw[2], bodyZ←raw[1]
# Verified by correlating GRAV with normalized ACCL across multiple .360 files
GRAV_AXIS_MAP = (0, 2, 1)
GRAV_AXIS_SIGN = (1.0, 1.0, 1.0)


def parse_gpmf_stream(file_path):
    """Extract raw GPMF metadata stream from .360 file."""
    result = subprocess.run([
        get_ffmpeg_path(), '-i', file_path, '-map', '0:3', '-c', 'copy',
        '-f', 'rawvideo', '-'
    ], capture_output=True, creationflags=get_subprocess_flags())
    if result.returncode != 0:
        raise Exception(f"Failed to extract GPMF: {result.stderr.decode()[:200]}")
    return result.stdout


def parse_gpmf_entries(data):
    """Walk through GPMF data and yield (fourcc, type_char, struct_size, repeat, payload_bytes, offset)."""
    pos = 0
    while pos + 8 <= len(data):
        fourcc = data[pos:pos+4]
        try:
            fourcc_str = fourcc.decode('ascii')
        except:
            pos += 4
            continue
        type_char = chr(data[pos+4])
        struct_size = data[pos+5]
        repeat = struct.unpack('>H', data[pos+6:pos+8])[0]
        payload_size = struct_size * repeat
        padded = (payload_size + 3) & ~3
        payload = data[pos+8:pos+8+payload_size]
        yield fourcc_str, type_char, struct_size, repeat, payload, pos
        if fourcc_str == 'DEVC' or fourcc_str == 'STRM':
            pos += 8  # container — step into children
        else:
            pos += 8 + padded


def parse_gyro_accl_full(file_path):
    """Parse all GYRO and ACCL blocks from the GPMF stream.

    Properly tracks DEVC/STRM nesting to associate STMP timestamps with
    each data stream. STMP (microsecond timestamps) are critical for correct
    temporal alignment between GYRO and CORI data.

    Returns dict with:
        'gyro_blocks': list of dicts, each with:
            'samples': np.array shape (N, 3) — raw axes, in **rad/s** after SCAL
            'scal': scale factor used
            'n_samples': N
            'stmp_us': STMP timestamp in microseconds (or None)
        'accl_blocks': list of dicts, same format but in m/s²
        'cori_stmps': list of (stmp_us, n_samples) for CORI blocks
        'fps': video fps
        'data_len': length of GPMF data
    """
    data = parse_gpmf_stream(file_path)

    # Get video info
    probe_result = subprocess.run([
        get_ffprobe_path(), '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,duration', '-of', 'json', file_path
    ], capture_output=True, text=True, creationflags=get_subprocess_flags())
    video_info = json.loads(probe_result.stdout)
    stream = video_info['streams'][0]
    fps_str = stream.get('r_frame_rate', '30/1')
    fps_parts = fps_str.split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    gyro_blocks = []
    accl_blocks = []
    grav_blocks = []
    mnor_blocks = []
    cori_stmps = []

    # Walk GPMF with proper DEVC/STRM nesting to track STMP per stream
    pos = 0
    current_scal = None
    current_stmp = None  # STMP for current STRM

    while pos + 8 <= len(data):
        fourcc = data[pos:pos+4]
        try:
            fourcc_str = fourcc.decode('ascii')
        except:
            pos += 1
            continue

        type_char = chr(data[pos+4])
        struct_size = data[pos+5]
        repeat = struct.unpack('>H', data[pos+6:pos+8])[0]
        payload_size = struct_size * repeat
        padded = (payload_size + 3) & ~3

        if fourcc_str == 'DEVC':
            pos += 8
            continue

        if fourcc_str == 'STRM':
            # New stream — reset per-stream state
            current_scal = None
            current_stmp = None
            pos += 8
            continue

        payload = data[pos+8:pos+8+payload_size] if pos+8+payload_size <= len(data) else b''

        if fourcc_str == 'STMP':
            # Parse STMP timestamp (microseconds)
            if type_char == 'J' and struct_size == 8 and len(payload) >= 8:
                current_stmp = struct.unpack('>Q', payload[:8])[0]
            elif type_char == 'L' and struct_size == 4 and len(payload) >= 4:
                current_stmp = struct.unpack('>I', payload[:4])[0]

        elif fourcc_str == 'SCAL':
            if type_char == 's' and struct_size == 2:
                vals = [struct.unpack('>h', payload[i*2:(i+1)*2])[0]
                        for i in range(min(repeat, len(payload)//2))]
                current_scal = vals[0] if len(vals) == 1 else vals
            elif type_char == 'S' and struct_size == 2:
                vals = [struct.unpack('>H', payload[i*2:(i+1)*2])[0]
                        for i in range(min(repeat, len(payload)//2))]
                current_scal = vals[0] if len(vals) == 1 else vals
            elif type_char == 'l' and struct_size == 4:
                vals = [struct.unpack('>i', payload[i*4:(i+1)*4])[0]
                        for i in range(min(repeat, len(payload)//4))]
                current_scal = vals[0] if len(vals) == 1 else vals

        elif fourcc_str == 'GYRO' and type_char == 's' and struct_size == 6:
            n = repeat
            scal = current_scal if current_scal else 1
            if isinstance(scal, list):
                scal = scal[0]
            samples = np.zeros((n, 3), dtype=np.float64)
            for j in range(n):
                off = j * 6
                if off + 6 <= len(payload):
                    gx, gy, gz = struct.unpack('>hhh', payload[off:off+6])
                    samples[j] = [gx / scal, gy / scal, gz / scal]
            gyro_blocks.append({
                'samples': samples,  # raw (x,y,z) in rad/s
                'scal': scal,
                'n_samples': n,
                'stmp_us': current_stmp,
            })

        elif fourcc_str == 'ACCL' and type_char == 's' and struct_size == 6:
            n = repeat
            scal = current_scal if current_scal else 1
            if isinstance(scal, list):
                scal = scal[0]
            samples = np.zeros((n, 3), dtype=np.float64)
            for j in range(n):
                off = j * 6
                if off + 6 <= len(payload):
                    ax, ay, az = struct.unpack('>hhh', payload[off:off+6])
                    samples[j] = [ax / scal, ay / scal, az / scal]
            accl_blocks.append({
                'samples': samples,
                'scal': scal,
                'n_samples': n,
                'stmp_us': current_stmp,
            })

        elif fourcc_str == 'GRAV' and type_char == 's' and struct_size == 6:
            n = repeat
            scal = current_scal if current_scal else 1
            if isinstance(scal, list):
                scal = scal[0]
            samples = np.zeros((n, 3), dtype=np.float64)
            for j in range(n):
                off = j * 6
                if off + 6 <= len(payload):
                    gx, gy, gz = struct.unpack('>hhh', payload[off:off+6])
                    samples[j] = [gx / scal, gy / scal, gz / scal]
            grav_blocks.append({
                'samples': samples,
                'scal': scal,
                'n_samples': n,
                'stmp_us': current_stmp,
            })

        elif fourcc_str == 'MNOR' and type_char == 's' and struct_size == 6:
            n = repeat
            scal = current_scal if current_scal else 1
            if isinstance(scal, list):
                scal = scal[0]
            samples = np.zeros((n, 3), dtype=np.float64)
            for j in range(n):
                off = j * 6
                if off + 6 <= len(payload):
                    mx, my, mz = struct.unpack('>hhh', payload[off:off+6])
                    samples[j] = [mx / scal, my / scal, mz / scal]
            mnor_blocks.append({
                'samples': samples,
                'scal': scal,
                'n_samples': n,
                'stmp_us': current_stmp,
            })

        elif fourcc_str == 'CORI':
            # Track CORI STMP for time-base alignment
            if current_stmp is not None:
                cori_stmps.append((current_stmp, repeat))

        pos += 8 + padded

    return {
        'gyro_blocks': gyro_blocks,
        'accl_blocks': accl_blocks,
        'grav_blocks': grav_blocks,
        'mnor_blocks': mnor_blocks,
        'cori_stmps': cori_stmps,
        'fps': fps,
        'data_len': len(data),
    }


def gyro_to_timestamps(gyro_blocks, fps, n_video_frames, cori_stmps=None):
    """Assign timestamps to gyro samples and apply axis remapping.

    Uses STMP timestamps when available for accurate temporal alignment.
    STMP timestamps are in microseconds from start of recording; they are
    converted to a frame-index time base (where frame 0 = t=0) by subtracting
    the first CORI STMP.

    Falls back to equal-block-duration linspace when STMPs are not available.

    Returns:
        times: (N,) timestamps in seconds
        gyro_rpy_rad: (N, 3) angular velocity in rad/s, remapped to (roll, pitch, yaw)
    """
    all_samples = []
    all_times = []

    # Check if STMP timestamps are available
    has_stmp = all(b.get('stmp_us') is not None for b in gyro_blocks)

    if has_stmp:
        # Use STMP-based timestamps for accurate alignment.
        # STMP is the timestamp of the first sample in each block (microseconds).
        #
        # Problem: STMP inter-block interval is ~1.001s (not 1.000s), so raw STMP
        # times drift from video frame-index times by ~1ms/second. Over a 35s clip,
        # this is 35ms of cumulative drift.
        #
        # Solution: Build a STMP→frame-index time mapping using CORI STMPs.
        # Each CORI block starts at a known STMP time and contains a known number
        # of frames. The frame-index time for each CORI block boundary is:
        #   frame_idx_time = cumulative_cori_count / fps
        # We interpolate GYRO STMP times through this mapping to convert them
        # to the frame-index time base.

        # Build STMP → frame-index mapping from CORI blocks
        if cori_stmps and len(cori_stmps) > 0:
            cori_start_us = cori_stmps[0][0]
            # Build mapping: (stmp_relative_s, frame_index_s)
            stmp_anchors = []  # (stmp_s, frame_s)
            cumulative_frames = 0
            for cs_us, cs_n in cori_stmps:
                stmp_s = (cs_us - cori_start_us) / 1e6
                frame_s = cumulative_frames / fps
                stmp_anchors.append((stmp_s, frame_s))
                cumulative_frames += cs_n

            # Add end-of-last-block anchor so GYRO samples after the last
            # block start don't clamp to a flat value
            last_cori_stmp_us, last_cori_n = cori_stmps[-1]
            end_stmp_s = (last_cori_stmp_us - cori_start_us) / 1e6 + last_cori_n / fps
            end_frame_s = cumulative_frames / fps
            stmp_anchors.append((end_stmp_s, end_frame_s))

            stmp_anchor_t = np.array([a[0] for a in stmp_anchors])
            frame_anchor_t = np.array([a[1] for a in stmp_anchors])
        else:
            # Fallback: no CORI STMPs — use identity mapping
            cori_start_us = gyro_blocks[0]['stmp_us']
            stmp_anchor_t = None
            frame_anchor_t = None

        for block_idx, block in enumerate(gyro_blocks):
            n = block['n_samples']
            stmp_s = (block['stmp_us'] - cori_start_us) / 1e6

            # Block end time in STMP space = next block's STMP
            if block_idx + 1 < len(gyro_blocks):
                next_stmp_s = (gyro_blocks[block_idx + 1]['stmp_us'] - cori_start_us) / 1e6
            else:
                # Last block: estimate from average block duration
                if block_idx > 0:
                    prev_stmp_s = (gyro_blocks[block_idx - 1]['stmp_us'] - cori_start_us) / 1e6
                    block_dur = stmp_s - prev_stmp_s
                else:
                    block_dur = n / 800.0  # ~800 Hz fallback
                next_stmp_s = stmp_s + block_dur

            # Generate timestamps in STMP space
            times_stmp = np.linspace(stmp_s, next_stmp_s, n, endpoint=False)

            # Convert from STMP space to frame-index space via interpolation
            if stmp_anchor_t is not None and len(stmp_anchor_t) >= 2:
                times = np.interp(times_stmp, stmp_anchor_t, frame_anchor_t)
            else:
                times = times_stmp

            all_times.append(times)

            # Apply axis remapping
            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = GYRO_AXIS_SIGN[ax] * raw[:, GYRO_AXIS_MAP[ax]]
            all_samples.append(remapped)
    else:
        # Fallback: equal-block-duration linspace (legacy behavior)
        total_duration = n_video_frames / fps
        n_blocks = len(gyro_blocks)
        block_duration = total_duration / n_blocks

        for block_idx, block in enumerate(gyro_blocks):
            n = block['n_samples']
            t_start = block_idx * block_duration
            t_end = (block_idx + 1) * block_duration

            times = np.linspace(t_start, t_end, n, endpoint=False)
            all_times.append(times)

            # Apply axis remapping: raw (x,y,z) → (roll, pitch, yaw)
            raw = block['samples']  # (n, 3) in rad/s
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = GYRO_AXIS_SIGN[ax] * raw[:, GYRO_AXIS_MAP[ax]]
            all_samples.append(remapped)

    if not all_times:
        return np.array([]), np.array([])

    return np.concatenate(all_times), np.concatenate(all_samples)


def accl_to_timestamps(accl_blocks, fps, n_video_frames, cori_stmps=None):
    """Assign timestamps to ACCL samples and apply axis remapping.

    Same STMP-based alignment as gyro_to_timestamps(), same ORIN="ZXY" remap.

    Returns:
        times: (M,) timestamps in seconds (frame-index time base)
        accl_body: (M, 3) acceleration in m/s², body frame [bodyX, bodyY, bodyZ]
    """
    all_samples = []
    all_times = []

    has_stmp = all(b.get('stmp_us') is not None for b in accl_blocks)

    if has_stmp:
        # Build STMP → frame-index mapping from CORI anchors (same as gyro_to_timestamps)
        if cori_stmps and len(cori_stmps) > 0:
            cori_start_us = cori_stmps[0][0]
            stmp_anchors = []
            cumulative_frames = 0
            for cs_us, cs_n in cori_stmps:
                stmp_s = (cs_us - cori_start_us) / 1e6
                frame_s = cumulative_frames / fps
                stmp_anchors.append((stmp_s, frame_s))
                cumulative_frames += cs_n
            last_cori_stmp_us, last_cori_n = cori_stmps[-1]
            end_stmp_s = (last_cori_stmp_us - cori_start_us) / 1e6 + last_cori_n / fps
            end_frame_s = cumulative_frames / fps
            stmp_anchors.append((end_stmp_s, end_frame_s))
            stmp_anchor_t = np.array([a[0] for a in stmp_anchors])
            frame_anchor_t = np.array([a[1] for a in stmp_anchors])
        else:
            cori_start_us = accl_blocks[0]['stmp_us']
            stmp_anchor_t = None
            frame_anchor_t = None

        for block_idx, block in enumerate(accl_blocks):
            n = block['n_samples']
            stmp_s = (block['stmp_us'] - cori_start_us) / 1e6
            if block_idx + 1 < len(accl_blocks):
                next_stmp_s = (accl_blocks[block_idx + 1]['stmp_us'] - cori_start_us) / 1e6
            else:
                if block_idx > 0:
                    prev_stmp_s = (accl_blocks[block_idx - 1]['stmp_us'] - cori_start_us) / 1e6
                    block_dur = stmp_s - prev_stmp_s
                else:
                    block_dur = n / 800.0
                next_stmp_s = stmp_s + block_dur
            times_stmp = np.linspace(stmp_s, next_stmp_s, n, endpoint=False)
            if stmp_anchor_t is not None and len(stmp_anchor_t) >= 2:
                times = np.interp(times_stmp, stmp_anchor_t, frame_anchor_t)
            else:
                times = times_stmp
            all_times.append(times)

            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = ACCL_AXIS_SIGN[ax] * raw[:, ACCL_AXIS_MAP[ax]]
            all_samples.append(remapped)
    else:
        total_duration = n_video_frames / fps
        n_blocks = len(accl_blocks)
        block_duration = total_duration / n_blocks
        for block_idx, block in enumerate(accl_blocks):
            n = block['n_samples']
            t_start = block_idx * block_duration
            t_end = (block_idx + 1) * block_duration
            times = np.linspace(t_start, t_end, n, endpoint=False)
            all_times.append(times)
            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = ACCL_AXIS_SIGN[ax] * raw[:, ACCL_AXIS_MAP[ax]]
            all_samples.append(remapped)

    if not all_times:
        return np.array([]), np.array([])

    return np.concatenate(all_times), np.concatenate(all_samples)


def grav_to_timestamps(grav_blocks, fps, n_video_frames, cori_stmps=None):
    """Assign timestamps to GRAV samples with STMP-based alignment.

    GRAV is GoPro's pre-computed gravity vector (unit normalized, ~200Hz or frame-rate).
    Same STMP structure as GYRO/ACCL. Same ORIN="ZXY" axis remap.

    Returns:
        times: (K,) timestamps in seconds (frame-index time base)
        grav_body: (K, 3) gravity direction in body frame (unit-ish, ~1.0 magnitude)
    """
    all_samples = []
    all_times = []

    has_stmp = all(b.get('stmp_us') is not None for b in grav_blocks)

    if has_stmp:
        if cori_stmps and len(cori_stmps) > 0:
            cori_start_us = cori_stmps[0][0]
            stmp_anchors = []
            cumulative_frames = 0
            for cs_us, cs_n in cori_stmps:
                stmp_s = (cs_us - cori_start_us) / 1e6
                frame_s = cumulative_frames / fps
                stmp_anchors.append((stmp_s, frame_s))
                cumulative_frames += cs_n
            last_cori_stmp_us, last_cori_n = cori_stmps[-1]
            end_stmp_s = (last_cori_stmp_us - cori_start_us) / 1e6 + last_cori_n / fps
            end_frame_s = cumulative_frames / fps
            stmp_anchors.append((end_stmp_s, end_frame_s))
            stmp_anchor_t = np.array([a[0] for a in stmp_anchors])
            frame_anchor_t = np.array([a[1] for a in stmp_anchors])
        else:
            cori_start_us = grav_blocks[0]['stmp_us']
            stmp_anchor_t = None
            frame_anchor_t = None

        for block_idx, block in enumerate(grav_blocks):
            n = block['n_samples']
            stmp_s = (block['stmp_us'] - cori_start_us) / 1e6
            if block_idx + 1 < len(grav_blocks):
                next_stmp_s = (grav_blocks[block_idx + 1]['stmp_us'] - cori_start_us) / 1e6
            else:
                if block_idx > 0:
                    prev_stmp_s = (grav_blocks[block_idx - 1]['stmp_us'] - cori_start_us) / 1e6
                    block_dur = stmp_s - prev_stmp_s
                else:
                    block_dur = n / 200.0
                next_stmp_s = stmp_s + block_dur
            times_stmp = np.linspace(stmp_s, next_stmp_s, n, endpoint=False)
            if stmp_anchor_t is not None and len(stmp_anchor_t) >= 2:
                times = np.interp(times_stmp, stmp_anchor_t, frame_anchor_t)
            else:
                times = times_stmp
            all_times.append(times)

            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = GRAV_AXIS_SIGN[ax] * raw[:, GRAV_AXIS_MAP[ax]]
            all_samples.append(remapped)
    else:
        total_duration = n_video_frames / fps
        n_blocks = len(grav_blocks)
        block_duration = total_duration / n_blocks
        for block_idx, block in enumerate(grav_blocks):
            n = block['n_samples']
            t_start = block_idx * block_duration
            t_end = (block_idx + 1) * block_duration
            times = np.linspace(t_start, t_end, n, endpoint=False)
            all_times.append(times)
            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = GRAV_AXIS_SIGN[ax] * raw[:, GRAV_AXIS_MAP[ax]]
            all_samples.append(remapped)

    if not all_times:
        return np.array([]), np.array([])

    return np.concatenate(all_times), np.concatenate(all_samples)


# MNOR (Magnetic North Orientation) axis mapping for VQF body frame.
# MNOR is a unit vector pointing to magnetic north, ORIN="ZXY".
# Uses GRAV_MAP (not GYRO_AXIS_MAP) — verified on all clips:
#   MNOR(GRAV_MAP) rotated by VQF gives heading std < 2.5° across all test files.
#   MNOR(GYRO_MAP) rotated by VQF gives heading std 65-130° (wrong frame).
# This is because MNOR and GRAV share the same body-frame convention in GoPro GPMF.
MNOR_VQF_MAP = GRAV_AXIS_MAP     # (0, 2, 1)
MNOR_VQF_SIGN = GRAV_AXIS_SIGN   # (1.0, 1.0, 1.0)


def mnor_to_timestamps(mnor_blocks, fps, n_video_frames, cori_stmps=None):
    """Assign timestamps to MNOR samples and remap to VQF body frame.

    MNOR is GoPro MAX2's firmware-calibrated magnetic north unit vector (~30Hz).
    Already hard-iron/soft-iron corrected internally. Scaled to ~50µT for VQF.

    Returns:
        times: (K,) timestamps in seconds
        mnor_body: (K, 3) magnetic north direction in VQF body frame, scaled to ~50µT
    """
    all_samples = []
    all_times = []

    has_stmp = all(b.get('stmp_us') is not None for b in mnor_blocks)

    if has_stmp:
        if cori_stmps and len(cori_stmps) > 0:
            cori_start_us = cori_stmps[0][0]
            stmp_anchors = []
            cumulative_frames = 0
            for cs_us, cs_n in cori_stmps:
                stmp_s = (cs_us - cori_start_us) / 1e6
                frame_s = cumulative_frames / fps
                stmp_anchors.append((stmp_s, frame_s))
                cumulative_frames += cs_n
            last_cori_stmp_us, last_cori_n = cori_stmps[-1]
            end_stmp_s = (last_cori_stmp_us - cori_start_us) / 1e6 + last_cori_n / fps
            end_frame_s = cumulative_frames / fps
            stmp_anchors.append((end_stmp_s, end_frame_s))
            stmp_anchor_t = np.array([a[0] for a in stmp_anchors])
            frame_anchor_t = np.array([a[1] for a in stmp_anchors])
        else:
            cori_start_us = mnor_blocks[0]['stmp_us']
            stmp_anchor_t = None
            frame_anchor_t = None

        for block_idx, block in enumerate(mnor_blocks):
            n = block['n_samples']
            stmp_s = (block['stmp_us'] - cori_start_us) / 1e6
            if block_idx + 1 < len(mnor_blocks):
                next_stmp_s = (mnor_blocks[block_idx + 1]['stmp_us'] - cori_start_us) / 1e6
            else:
                if block_idx > 0:
                    prev_stmp_s = (mnor_blocks[block_idx - 1]['stmp_us'] - cori_start_us) / 1e6
                    block_dur = stmp_s - prev_stmp_s
                else:
                    block_dur = n / 30.0
                next_stmp_s = stmp_s + block_dur
            times_stmp = np.linspace(stmp_s, next_stmp_s, n, endpoint=False)
            if stmp_anchor_t is not None and len(stmp_anchor_t) >= 2:
                times = np.interp(times_stmp, stmp_anchor_t, frame_anchor_t)
            else:
                times = times_stmp
            all_times.append(times)

            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = MNOR_VQF_SIGN[ax] * raw[:, MNOR_VQF_MAP[ax]]
            # Scale unit vector to ~50µT (Earth's field) for VQF magnitude checks
            remapped *= 50.0
            all_samples.append(remapped)
    else:
        total_duration = n_video_frames / fps
        n_blocks = len(mnor_blocks)
        block_duration = total_duration / n_blocks
        for block_idx, block in enumerate(mnor_blocks):
            n = block['n_samples']
            t_start = block_idx * block_duration
            t_end = (block_idx + 1) * block_duration
            times = np.linspace(t_start, t_end, n, endpoint=False)
            all_times.append(times)
            raw = block['samples']
            remapped = np.zeros_like(raw)
            for ax in range(3):
                remapped[:, ax] = MNOR_VQF_SIGN[ax] * raw[:, MNOR_VQF_MAP[ax]]
            remapped *= 50.0
            all_samples.append(remapped)

    if not all_times:
        return np.array([]), np.array([])

    return np.concatenate(all_times), np.concatenate(all_samples)


def grav_tilt_correction(gyro_times, gyro_body_rad, grav_times, grav_body,
                          alpha=0.02):
    """Fuse 800Hz gyro + GRAV gravity vector via complementary-style correction.

    GRAV is GoPro's pre-computed gravity vector (already filtered, no motion artifacts).
    Much cleaner than raw ACCL for tilt reference. Corrects roll+pitch drift;
    yaw still drifts (no horizontal reference without magnetometer).

    Args:
        gyro_times: (N,) timestamps in seconds
        gyro_body_rad: (N, 3) angular velocity in rad/s, body frame [X, Y, Z]
        grav_times: (K,) timestamps in seconds
        grav_body: (K, 3) gravity direction in body frame (unit-ish)
        alpha: correction gain per second (higher = trust GRAV more)
    Returns:
        quats: (N, 4) orientation quaternions (w, x, y, z)
    """
    n = len(gyro_times)
    quats = np.zeros((n, 4))

    # Upsample GRAV to GYRO timestamps
    grav_interp = np.zeros((n, 3))
    for ax in range(3):
        grav_interp[:, ax] = np.interp(gyro_times, grav_times, grav_body[:, ax])

    # Init from first GRAV samples (gravity → pitch+roll)
    n_init = min(10, n)
    g0 = grav_interp[:n_init].mean(axis=0)
    g0n = np.linalg.norm(g0)
    if g0n > 0.01:
        g0 = g0 / g0n
    roll0 = math.atan2(g0[1], g0[2])
    pitch0 = math.atan2(-g0[0], math.sqrt(g0[1]**2 + g0[2]**2))
    cr, sr = math.cos(roll0/2), math.sin(roll0/2)
    cp, sp = math.cos(pitch0/2), math.sin(pitch0/2)
    q0 = np.array([cr*cp, sr*cp, cr*sp, -sr*sp])
    q0 /= np.linalg.norm(q0)
    quats[0] = q0

    q = q0.copy()
    for i in range(1, n):
        dt = gyro_times[i] - gyro_times[i-1]
        if dt <= 0 or dt > 0.05:
            quats[i] = q
            continue

        # Gyro integration step (midpoint)
        omega = (gyro_body_rad[i-1] + gyro_body_rad[i]) / 2.0
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-12:
            axis = omega / np.linalg.norm(omega)
            ha = angle / 2
            dq = np.array([math.cos(ha), *(math.sin(ha) * axis)])
            q = _quat_multiply(q, dq)
        q /= np.linalg.norm(q)

        # GRAV tilt correction
        grav = grav_interp[i]
        grav_norm = np.linalg.norm(grav)
        if grav_norm > 0.5:
            grav_unit = grav / grav_norm

            # Expected gravity in body frame from current quaternion
            w, x, y, z = q
            g_est = np.array([
                2*(x*z - w*y),
                2*(y*z + w*x),
                w*w - x*x - y*y + z*z,
            ])

            # Cross product = rotation error between estimated and measured gravity
            error = np.cross(grav_unit, g_est)
            err_mag = np.linalg.norm(error)
            if err_mag > 1e-10:
                corr_angle = alpha * err_mag * dt
                corr_axis = error / err_mag
                ha_c = corr_angle / 2
                dq_c = np.array([math.cos(ha_c), *(math.sin(ha_c) * corr_axis)])
                q = _quat_multiply(q, dq_c)

        q /= np.linalg.norm(q)
        quats[i] = q

    return quats


# ─── Madgwick IMU Sensor Fusion ──────────────────────────────────────────────

def _quat_multiply(q1, q2):
    """Quaternion multiply (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _madgwick_step(q, gyr, acc, dt, beta):
    """One Madgwick IMU update step.

    Args:
        q: (4,) quaternion (w,x,y,z)
        gyr: (3,) angular velocity in rad/s, body frame
        acc: (3,) acceleration in m/s², body frame (will be normalized)
        dt: timestep in seconds
        beta: correction gain (higher = trust accel more)
    Returns:
        Updated quaternion (4,).
    """
    qw, qx, qy, qz = q
    gx, gy, gz = gyr
    ax, ay, az = acc

    a_norm = math.sqrt(ax*ax + ay*ay + az*az)
    # Gate: only apply accel correction when |a| is within 20% of gravity.
    # During motion, centripetal/linear acceleration contaminates the signal.
    g = 9.81
    accel_ok = (a_norm > 0.8 * g) and (a_norm < 1.2 * g)
    if a_norm < 1e-10 or not accel_ok:
        # Gyro-only update (no accel correction)
        qdot = 0.5 * _quat_multiply(q, np.array([0.0, gx, gy, gz]))
        q = q + qdot * dt
        return q / np.linalg.norm(q)

    ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm

    # Objective: align estimated gravity with measured accel
    # Estimated gravity in body frame from quaternion:
    #   g_est = [2(qx*qz - qw*qy), 2(qw*qx + qy*qz), 1 - 2(qx² + qy²)]
    f1 = 2.0*(qx*qz - qw*qy) - ax
    f2 = 2.0*(qw*qx + qy*qz) - ay
    f3 = 2.0*(0.5 - qx*qx - qy*qy) - az

    # Jacobian of objective
    J = np.array([
        [-2*qy,  2*qz, -2*qw,  2*qx],
        [ 2*qx,  2*qw,  2*qz,  2*qy],
        [ 0.0,  -4*qx, -4*qy,  0.0 ],
    ])
    grad = J.T @ np.array([f1, f2, f3])
    gn = np.linalg.norm(grad)
    if gn > 1e-10:
        grad /= gn

    # Gyro quaternion derivative - beta * gradient correction
    qdot = 0.5 * _quat_multiply(q, np.array([0.0, gx, gy, gz])) - beta * grad
    q = q + qdot * dt
    return q / np.linalg.norm(q)


def madgwick_imu_filter(gyro_times, gyro_body_rad, accl_times, accl_body_ms2,
                         beta=0.033):
    """Fuse 800Hz gyro + accel via Madgwick filter to produce orientation quats.

    Args:
        gyro_times: (N,) timestamps in seconds
        gyro_body_rad: (N, 3) angular velocity in rad/s, body frame [X, Y, Z]
        accl_times: (M,) timestamps in seconds
        accl_body_ms2: (M, 3) acceleration in m/s², body frame [X, Y, Z]
        beta: Madgwick gain (higher = more accel correction, less drift but more noise)
    Returns:
        quats: (N, 4) orientation quaternions (w, x, y, z)
    """
    n = len(gyro_times)
    quats = np.zeros((n, 4))

    # Upsample ACCL to GYRO timestamps via linear interpolation
    accl_interp = np.zeros((n, 3))
    for ax in range(3):
        accl_interp[:, ax] = np.interp(gyro_times, accl_times, accl_body_ms2[:, ax])

    # Initialize quaternion from first ~10 accel samples (gravity → pitch+roll)
    n_init = min(10, n)
    a0 = accl_interp[:n_init].mean(axis=0)
    a0n = a0 / np.linalg.norm(a0)
    roll0 = math.atan2(a0n[1], a0n[2])
    pitch0 = math.atan2(-a0n[0], math.sqrt(a0n[1]**2 + a0n[2]**2))
    cr, sr = math.cos(roll0/2), math.sin(roll0/2)
    cp, sp = math.cos(pitch0/2), math.sin(pitch0/2)
    q0 = np.array([cr*cp, sr*cp, cr*sp, -sr*sp])
    q0 /= np.linalg.norm(q0)
    quats[0] = q0

    q = q0.copy()
    for i in range(1, n):
        dt = gyro_times[i] - gyro_times[i-1]
        if dt <= 0 or dt > 0.05:
            quats[i] = q
            continue
        g = (gyro_body_rad[i-1] + gyro_body_rad[i]) / 2.0  # midpoint
        a = accl_interp[i]
        q = _madgwick_step(q, g, a, dt, beta)
        quats[i] = q

    return quats


def integrate_gyro_to_quats(times, gyro_rad_s):
    """Integrate angular velocity (rad/s) to orientation quaternions.

    Uses first-order quaternion integration with midpoint method:
        q(t+dt) = q(t) * delta_q
    where delta_q = [cos(|ω|dt/2), sin(|ω|dt/2) * ω/|ω|]

    Args:
        times: (N,) timestamps in seconds
        gyro_rad_s: (N, 3) angular velocity in **rad/s** (roll, pitch, yaw)

    Returns:
        quats: (N, 4) quaternions (w, x, y, z)
    """
    n = len(times)
    quats = np.zeros((n, 4), dtype=np.float64)
    quats[0] = [1, 0, 0, 0]

    for i in range(1, n):
        dt = times[i] - times[i-1]
        if dt <= 0:
            quats[i] = quats[i-1]
            continue

        # Midpoint angular velocity
        omega = (gyro_rad_s[i-1] + gyro_rad_s[i]) / 2.0
        angle = np.linalg.norm(omega) * dt

        if angle < 1e-12:
            quats[i] = quats[i-1]
            continue

        axis = omega / np.linalg.norm(omega)
        half_angle = angle / 2

        dq_w = math.cos(half_angle)
        dq_xyz = math.sin(half_angle) * axis

        q = quats[i-1]
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq_w, dq_xyz[0], dq_xyz[1], dq_xyz[2]

        quats[i] = [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ]

        norm = np.linalg.norm(quats[i])
        if norm > 0:
            quats[i] /= norm

    return quats


def quat_to_euler_batch(quats):
    """Convert (N, 4) quaternion array to (N, 3) euler angles in degrees.
    Order: (roll, pitch, yaw)."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi/2, sinp), np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])


def resample_quats_to_frames(times, quats, fps, n_frames, time_offset_s=0.0, window_s=0.0):
    """Resample orientation quaternions to video frame timestamps.

    Two sampling modes:

    - **Point sample** (window_s == 0): linearly interpolate the quaternion
      at each frame's sample time (i/fps + time_offset_s). This is the
      legacy behavior.

    - **Window average** (window_s > 0): for each frame, average all
      quaternion samples within [center - window_s/2, center + window_s/2],
      where center = i/fps + time_offset_s. This integrates gyro noise out
      over the rolling-shutter readout window — set window_s=srot_s and
      time_offset_s=srot_s/2 to make each frame's CORI represent the mean
      orientation over its physical readout interval.

    Args:
        times: (M,) gyro sample timestamps in seconds (frame-index time base).
        quats: (M, 4) quaternions at those times. Should be sign-continuous
            (consecutive samples have positive dot product).
        fps: video frame rate.
        n_frames: number of video frames to produce.
        time_offset_s: additive offset applied to each frame's sample time.
        window_s: averaging window width in seconds. 0 disables averaging.
    """
    frame_times = np.arange(n_frames) / fps + time_offset_s
    frame_quats = np.zeros((n_frames, 4), dtype=np.float64)

    if window_s > 0 and len(times) > 0:
        # Window-average mode. Sign-align the input quats first so adjacent
        # samples (which represent the same rotation under double-cover) don't
        # cancel out when summed.
        quats_aligned = np.asarray(quats, dtype=np.float64).copy()
        # In-place sign continuity pass (flip q[j] if dot(q[j], q[j-1]) < 0)
        for j in range(1, len(quats_aligned)):
            if (quats_aligned[j, 0] * quats_aligned[j - 1, 0]
                    + quats_aligned[j, 1] * quats_aligned[j - 1, 1]
                    + quats_aligned[j, 2] * quats_aligned[j - 1, 2]
                    + quats_aligned[j, 3] * quats_aligned[j - 1, 3]) < 0:
                quats_aligned[j] = -quats_aligned[j]

        half_w = window_s / 2.0
        for i in range(n_frames):
            t_lo = frame_times[i] - half_w
            t_hi = frame_times[i] + half_w
            lo_idx = int(np.searchsorted(times, t_lo, side='left'))
            hi_idx = int(np.searchsorted(times, t_hi, side='right'))
            if hi_idx > lo_idx:
                # Mean of quats in the window (un-normalized; renormalized below)
                frame_quats[i] = quats_aligned[lo_idx:hi_idx].mean(axis=0)
            else:
                # Fallback for edge frames: nearest available sample
                idx = max(0, min(lo_idx, len(quats_aligned) - 1))
                frame_quats[i] = quats_aligned[idx]
    else:
        # Point sampling via linear interpolation
        for c in range(4):
            frame_quats[:, c] = np.interp(frame_times, times, quats[:, c])

    # Re-normalize
    norms = np.linalg.norm(frame_quats, axis=1, keepdims=True)
    frame_quats /= np.where(norms > 0, norms, 1)

    return frame_times, frame_quats


def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles to quaternion (w, x, y, z)."""
    r, p, y = np.radians(roll_deg/2), np.radians(pitch_deg/2), np.radians(yaw_deg/2)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def get_gyro_angular_velocity(file_path, fps, n_video_frames):
    """Extract 800Hz GYRO angular velocity with STMP-aligned timestamps.

    Returns angular velocity ordered for fisheye_remap_rs_stab():
      [0] = bodyY → roll slot (rotation around optical axis)
      [1] = bodyX → pitch slot (tilt up/down → vertical shear)
      [2] = bodyZ → yaw slot (pan left/right → horizontal shear)

    Axis mapping verified empirically: co-moving test chart (GS010172.360)
    showed bodyZ rotation causes horizontal per-row shear in EAC→fisheye,
    confirming bodyZ = vertical axis = yaw.

    Returns:
        gyro_times: (N,) timestamps in seconds (frame-index time base)
        gyro_angvel: (N, 3) angular velocity in deg/s, RS correction order
    """
    result = parse_gyro_accl_full(file_path)
    gyro_blocks = result['gyro_blocks']
    cori_stmps = result.get('cori_stmps', [])

    if not gyro_blocks:
        return None, None

    # Get STMP-aligned timestamps and body-frame angular velocity (rad/s)
    # gyro_rpy columns: [0]=bodyX, [1]=bodyY, [2]=bodyZ
    times, gyro_rpy_rad = gyro_to_timestamps(gyro_blocks, fps, n_video_frames, cori_stmps)

    if len(times) == 0:
        return None, None

    # Convert to deg/s and reorder for RS correction slots:
    #   slot[0] = roll  = bodyY = gyro col1
    #   slot[1] = pitch = bodyX = gyro col0  (tilt → vertical shear)
    #   slot[2] = yaw   = bodyZ = gyro col2  (pan → horizontal shear)
    gyro_angvel_deg = np.degrees(gyro_rpy_rad[:, [1, 0, 2]])

    return times, gyro_angvel_deg


def gyro_to_cori_quats(file_path, n_video_frames=None):
    """Full pipeline: parse raw GYRO → integrated orientation quaternions per frame.

    Returns dict compatible with parse_gopro_gyro_data() output format:
        'fps': float
        'frames': list of dicts with 'cori_quat' (w,x,y,z) and 'cori_euler' (roll,pitch,yaw)
        'iori_quat'/'iori_euler' are zeroed.

    This function replaces CORI data when CORI is zeroed (ERS disabled firmware).
    """
    result = parse_gyro_accl_full(file_path)
    gyro_blocks = result['gyro_blocks']
    accl_blocks = result['accl_blocks']
    fps = result['fps']

    if not gyro_blocks:
        raise ValueError(f"No GYRO data found in {file_path}")

    # Get frame count if not provided
    if n_video_frames is None:
        from vr180_gui import parse_gopro_gyro_data
        gd = parse_gopro_gyro_data(file_path)
        n_video_frames = len(gd['frames'])

    # Get timestamped, remapped GYRO (rad/s, body frame order)
    cori_stmps = result.get('cori_stmps', [])
    times, gyro_rpy_rad = gyro_to_timestamps(gyro_blocks, fps, n_video_frames, cori_stmps)

    # Fuse gyro + accelerometer via Madgwick filter (drift-corrected)
    if accl_blocks:
        accl_times, accl_body = accl_to_timestamps(accl_blocks, fps, n_video_frames, cori_stmps)
        if len(accl_times) > 0:
            quats = madgwick_imu_filter(times, gyro_rpy_rad, accl_times, accl_body, beta=0.033)
            print(f"gyro_to_cori_quats: Madgwick fusion ({len(times)} gyro + {len(accl_times)} accl samples)")
        else:
            quats = integrate_gyro_to_quats(times, gyro_rpy_rad)
            print(f"gyro_to_cori_quats: pure gyro integration (no ACCL timestamps)")
    else:
        quats = integrate_gyro_to_quats(times, gyro_rpy_rad)
        print(f"gyro_to_cori_quats: pure gyro integration (no ACCL data)")

    # Resample to frame rate
    frame_times, frame_quats = resample_quats_to_frames(times, quats, fps, n_video_frames)

    # Swap Y↔Z quaternion components to match CORI's native convention.
    # CORI file stores (w, x, Z, Y) — integration produces standard (w, x, y, z),
    # so swap columns 2↔3 to match. (Quat multiply is not covariant under Y↔Z swap,
    # so we must integrate in correct body frame first, then transform.)
    frame_quats[:, [2, 3]] = frame_quats[:, [3, 2]]

    frame_eulers = quat_to_euler_batch(frame_quats)

    # Build output compatible with parse_gopro_gyro_data format
    frames = []
    zero_quat = (1.0, 0.0, 0.0, 0.0)
    zero_euler = (0.0, 0.0, 0.0)
    for i in range(n_video_frames):
        frames.append({
            'time': i / fps,
            'cori_quat': tuple(frame_quats[i]),
            'cori_euler': tuple(frame_eulers[i]),
            'iori_quat': zero_quat,
            'iori_euler': zero_euler,
        })

    return {
        'fps': fps,
        'srot_ms': 15.224,  # GoPro MAX sensor readout time
        'frames': frames,
        'source': 'madgwick_fusion' if accl_blocks else 'gyro_integration',
        'n_gyro_samples': len(times),
        'gyro_duration': times[-1],
    }


def vqf_to_cori_quats(file_path, n_video_frames=None):
    """VQF-based IMU fusion: parse raw GYRO + GRAV/ACCL (+ MNOR) → orientation quats.

    Uses the VQF (Versatile Quaternion Filter) algorithm with:
    - 800Hz GYRO for angular velocity
    - GRAV×9.81 as accelerometer input (cleaner than raw ACCL, pre-filtered
      by GoPro). Falls back to raw ACCL if GRAV is zeroed.
    - MNOR (firmware-calibrated magnetic north) for 9D fusion if available

    VQF provides automatic gyro bias estimation and drift-corrected pitch/roll.
    Yaw drifts ~0.1-0.2°/s without magnetometer (acceptable for stabilization
    with smoothing). With MNOR, yaw is anchored to magnetic north.

    Falls back to pure gyro integration if pyvqf is not available.

    Frame timing: each frame's CORI is the AVERAGE of all VQF orientation
    quaternions over the rolling-shutter readout window [i/fps, i/fps + srot].
    center = i/fps + srot/2, width = srot. GoPro MAX SROT = 15.224ms →
    ~12 samples averaged per frame to integrate out gyro/sensor noise.

    Returns dict compatible with parse_gopro_gyro_data() output format.
    """
    try:
        from pyvqf import PyVQF
    except ImportError:
        print("vqf_to_cori_quats: pyvqf not available, falling back to pure gyro integration")
        return _gyro_integration_to_cori(file_path, n_video_frames)

    result = parse_gyro_accl_full(file_path)
    gyro_blocks = result['gyro_blocks']
    accl_blocks = result['accl_blocks']
    grav_blocks = result.get('grav_blocks', [])
    mnor_blocks = result.get('mnor_blocks', [])
    fps = result['fps']

    if not gyro_blocks:
        raise ValueError(f"No GYRO data found in {file_path}")

    if n_video_frames is None:
        from vr180_gui import parse_gopro_gyro_data
        gd = parse_gopro_gyro_data(file_path)
        n_video_frames = len(gd['frames'])

    # Get STMP-aligned, axis-remapped gyro samples (rad/s, body frame)
    cori_stmps = result.get('cori_stmps', [])
    gyro_times, gyro_body_rad = gyro_to_timestamps(gyro_blocks, fps, n_video_frames, cori_stmps)
    if len(gyro_times) == 0:
        raise ValueError(f"No GYRO timestamps in {file_path}")

    duration = n_video_frames / fps
    gyro_dt = 1.0 / (len(gyro_body_rad) / duration)

    # Choose accelerometer source: GRAV×9.81 preferred, raw ACCL fallback
    acc_source = "unknown"
    acc_input = None
    if grav_blocks:
        grav_times, grav_body = grav_to_timestamps(grav_blocks, fps, n_video_frames, cori_stmps)
        grav_magnitude = np.linalg.norm(np.mean(grav_body[:min(30, len(grav_body))], axis=0))
        if grav_magnitude > 0.1:
            acc_input = grav_body * 9.81
            acc_source = "GRAV×9.81"

    if acc_source == "unknown" and accl_blocks:
        _, acc_input = accl_to_timestamps(accl_blocks, fps, n_video_frames, cori_stmps)
        acc_source = "raw ACCL"

    if acc_source == "unknown":
        print("vqf_to_cori_quats: no accelerometer data, falling back to gyro integration")
        return _gyro_integration_to_cori(file_path, n_video_frames)

    # Resample accelerometer to gyro rate
    acc_resampled = np.zeros((len(gyro_body_rad), 3))
    acc_dt = duration / len(acc_input) if len(acc_input) > 0 else 1.0
    for i in range(len(gyro_body_rad)):
        t = i * gyro_dt
        idx = min(int(t / acc_dt), len(acc_input) - 1)
        acc_resampled[i] = acc_input[idx]

    # Magnetometer: use MNOR (firmware-calibrated magnetic north) if available
    mag_source = "none"
    mag_resampled = None
    mnor_body = None
    if mnor_blocks:
        mnor_times, mnor_body = mnor_to_timestamps(mnor_blocks, fps, n_video_frames, cori_stmps)
        if len(mnor_times) > 0:
            mag_resampled = np.zeros((len(gyro_body_rad), 3))
            for i in range(len(gyro_body_rad)):
                idx = np.searchsorted(mnor_times, gyro_times[i], side='right') - 1
                idx = max(0, min(idx, len(mnor_body) - 1))
                mag_resampled[i] = mnor_body[idx]
            mag_source = "MNOR"

    # Run VQF (9D with MNOR if available, 6D otherwise)
    vqf_params = {}
    if mag_resampled is not None:
        vqf_params['magDistRejectionEnabled'] = False
        vqf_params['tauMag'] = 5.0

    vqf = PyVQF(gyro_dt, **vqf_params)
    if mag_resampled is not None:
        batch_result = vqf.updateBatch(gyro_body_rad, acc_resampled, mag_resampled)
        vqf_quats = batch_result['quat9D']
    else:
        batch_result = vqf.updateBatch(gyro_body_rad, acc_resampled)
        vqf_quats = batch_result['quat6D']

    bias_est = vqf.getBiasEstimate()
    bias_deg_s = bias_est[0] * 180.0 / np.pi

    # Resample to frame rate by AVERAGING VQF quats over the rolling-shutter
    # readout window: [i/fps, i/fps + srot]. center = i/fps + srot/2, width = srot.
    # GoPro MAX SROT = 15.224ms → ~12 samples averaged per frame.
    SROT_S = 15.224 / 1000.0
    _, frame_quats = resample_quats_to_frames(
        gyro_times, vqf_quats, fps, n_video_frames,
        time_offset_s=SROT_S / 2.0,  # window centered at mid-readout
        window_s=SROT_S)              # average across the entire readout window

    # Swap Y↔Z quaternion components to match CORI's native convention
    frame_quats[:, [2, 3]] = frame_quats[:, [3, 2]]

    frame_eulers = quat_to_euler_batch(frame_quats)

    # Build output compatible with parse_gopro_gyro_data format
    frames = []
    zero_quat = (1.0, 0.0, 0.0, 0.0)
    zero_euler = (0.0, 0.0, 0.0)
    for i in range(n_video_frames):
        frames.append({
            'time': i / fps,
            'cori_quat': tuple(frame_quats[i]),
            'cori_euler': tuple(frame_eulers[i]),
            'iori_quat': zero_quat,
            'iori_euler': zero_euler,
        })

    samples_per_frame = int(SROT_S * len(gyro_times) / gyro_times[-1])
    n_mag = len(mnor_body) if (mag_resampled is not None and mnor_body is not None) else 0
    print(f"vqf_to_cori_quats: VQF {'9D' if mag_resampled is not None else '6D'} "
          f"({acc_source}+{mag_source}), {len(gyro_body_rad)} gyro + {len(acc_input)} acc"
          f"{f' + {n_mag} mag' if mag_resampled is not None else ''} samples, "
          f"bias=[{bias_deg_s[0]:.3f}, {bias_deg_s[1]:.3f}, {bias_deg_s[2]:.3f}]°/s, "
          f"{SROT_S*1000:.2f}ms readout-window averaging (~{samples_per_frame} samples/frame)")

    return {
        'fps': fps,
        'srot_ms': 15.224,
        'frames': frames,
        'source': f'vqf_{mag_source}',
        'n_gyro_samples': len(gyro_times),
        'gyro_duration': gyro_times[-1],
        'gyro_bias_deg_s': tuple(bias_deg_s),
    }


def _gyro_integration_to_cori(file_path, n_video_frames=None):
    """Pure gyro integration fallback when VQF / pyvqf is unavailable.

    Same readout-window averaging as the VQF path, but starts the
    quaternion trajectory from identity and uses no accelerometer data.
    """
    result = parse_gyro_accl_full(file_path)
    gyro_blocks = result['gyro_blocks']
    fps = result['fps']

    if not gyro_blocks:
        raise ValueError(f"No GYRO data found in {file_path}")

    if n_video_frames is None:
        from vr180_gui import parse_gopro_gyro_data
        gd = parse_gopro_gyro_data(file_path)
        n_video_frames = len(gd['frames'])

    cori_stmps = result.get('cori_stmps', [])
    gyro_times, gyro_body_rad = gyro_to_timestamps(gyro_blocks, fps, n_video_frames, cori_stmps)
    if len(gyro_times) == 0:
        raise ValueError(f"No GYRO timestamps in {file_path}")

    quats = integrate_gyro_to_quats(gyro_times, gyro_body_rad)

    SROT_S = 15.224 / 1000.0
    _, frame_quats = resample_quats_to_frames(
        gyro_times, quats, fps, n_video_frames,
        time_offset_s=SROT_S / 2.0,
        window_s=SROT_S)

    frame_quats[:, [2, 3]] = frame_quats[:, [3, 2]]
    frame_eulers = quat_to_euler_batch(frame_quats)

    frames = []
    zero_quat = (1.0, 0.0, 0.0, 0.0)
    zero_euler = (0.0, 0.0, 0.0)
    for i in range(n_video_frames):
        frames.append({
            'time': i / fps,
            'cori_quat': tuple(frame_quats[i]),
            'cori_euler': tuple(frame_eulers[i]),
            'iori_quat': zero_quat,
            'iori_euler': zero_euler,
        })

    samples_per_frame = int(SROT_S * len(gyro_times) / gyro_times[-1])
    print(f"_gyro_integration_to_cori: pure gyro integration ({len(gyro_times)} samples, "
          f"{SROT_S*1000:.2f}ms readout-window averaging, ~{samples_per_frame} samples/frame)")

    return {
        'fps': fps,
        'srot_ms': 15.224,
        'frames': frames,
        'source': 'gyro_integration',
        'n_gyro_samples': len(gyro_times),
        'gyro_duration': gyro_times[-1],
    }


def vqf_to_cori_quats_multi_segment(segment_paths, n_chapter_frames_list):
    """Multi-segment version of vqf_to_cori_quats: aggregate raw gyro across
    all segments, integrate as ONE continuous stream, then resample to per-
    frame quats. This avoids any discontinuity at segment boundaries that
    per-segment integration + post-hoc quaternion chaining would introduce.

    Pipeline:
      1. For each segment, parse raw gyro (and grav/accl/mnor for VQF mode).
      2. Offset each segment's STMP-based gyro times by the cumulative time
         of the preceding segments (so they form a single continuous time
         axis spanning the entire recording).
      3. Concatenate the gyro / accl / mnor sample arrays.
      4. Run VQF (or pure gyro integration) on the combined stream.
      5. Resample to combined frame timestamps with mid-readout sampling
         and SROT-width window averaging.
      6. Y↔Z quat swap and build per-frame dicts.

    Args:
        segment_paths: list of .360 file paths in chapter order.
        n_chapter_frames_list: list of int, video frame count for each segment.
            Must be in the same order as segment_paths and have the same length.

    Returns dict with the same keys as vqf_to_cori_quats() plus 'cori_is_vqf'.
    """
    if len(segment_paths) != len(n_chapter_frames_list):
        raise ValueError("segment_paths and n_chapter_frames_list length mismatch")
    if not segment_paths:
        raise ValueError("no segment paths provided")

    # Try to use PyVQF for fusion; fall back to pure gyro integration if not available
    try:
        from pyvqf import PyVQF
        have_vqf = True
    except ImportError:
        have_vqf = False
        print("vqf_to_cori_quats_multi_segment: pyvqf not available, "
              "using pure gyro integration across segments")

    all_gyro_times = []
    all_gyro_body = []
    all_acc_input = []  # in m/s² (GRAV*9.81 or raw ACCL)
    all_acc_times = []
    all_mag = []
    all_mag_times = []
    fps = None

    cumulative_time = 0.0
    use_grav = True
    have_mag = True
    seg_segment_durations = []  # for diagnostic / boundary listing

    for seg_idx, seg_path in enumerate(segment_paths):
        result = parse_gyro_accl_full(seg_path)
        if fps is None:
            fps = result['fps']
        gyro_blocks = result['gyro_blocks']
        accl_blocks = result['accl_blocks']
        grav_blocks = result.get('grav_blocks', [])
        mnor_blocks = result.get('mnor_blocks', [])
        cori_stmps = result.get('cori_stmps', [])

        n_frames_seg = n_chapter_frames_list[seg_idx]
        gyro_times, gyro_body_rad = gyro_to_timestamps(gyro_blocks, fps, n_frames_seg, cori_stmps)
        if len(gyro_times) == 0:
            raise ValueError(f"No GYRO timestamps in segment: {seg_path}")

        # Offset gyro times by cumulative time. STMP-based gyro_times are in
        # the per-segment frame-index time base; adding cumulative_time puts
        # them on the GLOBAL frame-index time base.
        all_gyro_times.append(gyro_times + cumulative_time)
        all_gyro_body.append(gyro_body_rad)

        # Accelerometer source: prefer GRAV, fall back to raw ACCL
        if use_grav and grav_blocks:
            grav_times, grav_body = grav_to_timestamps(grav_blocks, fps, n_frames_seg, cori_stmps)
            grav_mag = np.linalg.norm(np.mean(grav_body[:min(30, len(grav_body))], axis=0))
            if grav_mag > 0.1:
                all_acc_times.append(grav_times + cumulative_time)
                all_acc_input.append(grav_body * 9.81)
            else:
                use_grav = False
        if not use_grav and accl_blocks:
            acc_times, acc_body = accl_to_timestamps(accl_blocks, fps, n_frames_seg, cori_stmps)
            all_acc_times.append(acc_times + cumulative_time)
            all_acc_input.append(acc_body)

        # Magnetometer
        if have_mag and mnor_blocks:
            mnor_times, mnor_body = mnor_to_timestamps(mnor_blocks, fps, n_frames_seg, cori_stmps)
            if len(mnor_times) > 0:
                all_mag_times.append(mnor_times + cumulative_time)
                all_mag.append(mnor_body)
            else:
                have_mag = False
        elif not mnor_blocks:
            have_mag = False

        seg_dur = n_frames_seg / fps
        seg_segment_durations.append(seg_dur)
        cumulative_time += seg_dur

    # Concatenate all sample arrays
    combined_gyro_times = np.concatenate(all_gyro_times)
    combined_gyro_body = np.concatenate(all_gyro_body, axis=0)
    total_n_frames = sum(n_chapter_frames_list)

    # ── Choose integration path: VQF (with grav/mnor) or pure gyro integration ──
    quats = None
    bias_deg_s = (0.0, 0.0, 0.0)
    acc_source = "none"
    mag_source = "none"

    if have_vqf and all_acc_input:
        # Run VQF on the combined stream. We need acc and (optionally) mag
        # resampled to gyro rate.
        combined_acc_times = np.concatenate(all_acc_times)
        combined_acc_input = np.concatenate(all_acc_input, axis=0)
        acc_source = "GRAV×9.81" if use_grav else "raw ACCL"

        # Resample acc to gyro times via linear interpolation
        acc_resampled = np.zeros((len(combined_gyro_body), 3))
        for c in range(3):
            acc_resampled[:, c] = np.interp(combined_gyro_times,
                                             combined_acc_times,
                                             combined_acc_input[:, c])

        mag_resampled = None
        if have_mag and all_mag:
            combined_mag_times = np.concatenate(all_mag_times)
            combined_mag = np.concatenate(all_mag, axis=0)
            mag_resampled = np.zeros((len(combined_gyro_body), 3))
            for c in range(3):
                mag_resampled[:, c] = np.interp(combined_gyro_times,
                                                 combined_mag_times,
                                                 combined_mag[:, c])
            mag_source = "MNOR"

        # Compute VQF dt from the gyro samples
        duration_s = combined_gyro_times[-1] - combined_gyro_times[0]
        gyro_dt = duration_s / max(1, len(combined_gyro_body) - 1)

        vqf_params = {}
        if mag_resampled is not None:
            vqf_params['magDistRejectionEnabled'] = False
            vqf_params['tauMag'] = 5.0
        vqf = PyVQF(gyro_dt, **vqf_params)
        if mag_resampled is not None:
            batch = vqf.updateBatch(combined_gyro_body, acc_resampled, mag_resampled)
            quats = batch['quat9D']
        else:
            batch = vqf.updateBatch(combined_gyro_body, acc_resampled)
            quats = batch['quat6D']
        bias_est = vqf.getBiasEstimate()
        bias_deg_s = tuple(bias_est[0] * 180.0 / np.pi)
    else:
        # Pure gyro integration on the combined stream (always continuous)
        quats = integrate_gyro_to_quats(combined_gyro_times, combined_gyro_body)

    # ── Resample to combined frame timestamps with mid-readout averaging ──
    SROT_S = 15.224 / 1000.0
    _, frame_quats = resample_quats_to_frames(
        combined_gyro_times, quats, fps, total_n_frames,
        time_offset_s=SROT_S / 2.0,
        window_s=SROT_S)

    # Y↔Z swap to match CORI's native convention
    frame_quats[:, [2, 3]] = frame_quats[:, [3, 2]]
    frame_eulers = quat_to_euler_batch(frame_quats)

    # Build per-frame dicts on the GLOBAL frame-index time base
    frames = []
    zero_quat = (1.0, 0.0, 0.0, 0.0)
    zero_euler = (0.0, 0.0, 0.0)
    for i in range(total_n_frames):
        frames.append({
            'time': i / fps,
            'cori_quat': tuple(frame_quats[i]),
            'cori_euler': tuple(frame_eulers[i]),
            'iori_quat': zero_quat,
            'iori_euler': zero_euler,
        })

    samples_per_frame = int(SROT_S * len(combined_gyro_times) / combined_gyro_times[-1])
    label = f"VQF {'9D' if mag_source == 'MNOR' else '6D'} ({acc_source}+{mag_source})" \
            if quats is not None and have_vqf and all_acc_input \
            else "pure gyro integration"
    boundary_str = ", ".join(f"{d:.2f}s" for d in seg_segment_durations)
    print(f"vqf_to_cori_quats_multi_segment: {label} across {len(segment_paths)} segments "
          f"[{boundary_str}], {len(combined_gyro_times)} combined gyro samples over "
          f"{combined_gyro_times[-1]:.2f}s, {SROT_S*1000:.2f}ms readout-window averaging "
          f"(~{samples_per_frame} samples/frame), bias=[{bias_deg_s[0]:.3f}, "
          f"{bias_deg_s[1]:.3f}, {bias_deg_s[2]:.3f}]°/s")

    return {
        'fps': fps,
        'srot_ms': 15.224,
        'frames': frames,
        'source': f'vqf_{mag_source}' if (have_vqf and all_acc_input) else 'gyro_integration',
        'n_gyro_samples': len(combined_gyro_times),
        'gyro_duration': float(combined_gyro_times[-1]),
        'gyro_bias_deg_s': bias_deg_s,
        'cori_is_vqf': True,
    }


def main():
    for label, path in [("GS010172 (old, ERS on)", FILE_OLD),
                         ("GS010173 (new, ERS off)", FILE_NEW)]:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        result = parse_gyro_accl_full(path)
        gyro_blocks = result['gyro_blocks']
        accl_blocks = result['accl_blocks']
        fps = result['fps']

        print(f"\n  FPS: {fps}")
        print(f"  GYRO blocks: {len(gyro_blocks)}")
        if gyro_blocks:
            total_gyro = sum(b['n_samples'] for b in gyro_blocks)
            avg_per_block = total_gyro / len(gyro_blocks)
            print(f"  Total GYRO samples: {total_gyro}")
            print(f"  Avg samples/block: {avg_per_block:.0f} (~{avg_per_block:.0f} Hz at 1 block/s)")
            print(f"  SCAL: {gyro_blocks[0]['scal']}")

            # Show raw values (rad/s)
            print(f"\n  First block, first 5 samples (RAD/S, raw axes):")
            for j in range(min(5, gyro_blocks[0]['n_samples'])):
                s = gyro_blocks[0]['samples'][j]
                print(f"    [{j}] x={s[0]:>8.4f}  y={s[1]:>8.4f}  z={s[2]:>8.4f} rad/s"
                      f"  ({np.degrees(s[0]):>7.1f}  {np.degrees(s[1]):>7.1f}  {np.degrees(s[2]):>7.1f} deg/s)")

        print(f"\n  ACCL blocks: {len(accl_blocks)}")
        if accl_blocks:
            total_accl = sum(b['n_samples'] for b in accl_blocks)
            print(f"  Total ACCL samples: {total_accl}")
            print(f"  SCAL: {accl_blocks[0]['scal']}")
            print(f"  First 3 ACCL samples (m/s²):")
            for j in range(min(3, accl_blocks[0]['n_samples'])):
                s = accl_blocks[0]['samples'][j]
                print(f"    [{j}] x={s[0]:>8.3f}  y={s[1]:>8.3f}  z={s[2]:>8.3f}")
            g_mag = np.linalg.norm(accl_blocks[0]['samples'][:10], axis=1)
            print(f"  |gravity| from first 10 samples: {g_mag.mean():.3f} m/s²")

        # ── Test gyro_to_cori_quats pipeline ──
        if gyro_blocks:
            from vr180_gui import parse_gopro_gyro_data as _parse
            _gd = _parse(path)
            n_video_frames = len(_gd['frames'])

            print(f"\n  Testing gyro_to_cori_quats pipeline ({n_video_frames} frames)...")
            gyro_data = gyro_to_cori_quats(path, n_video_frames)
            frames = gyro_data['frames']

            print(f"  Source: {gyro_data['source']}")
            print(f"  GYRO samples: {gyro_data['n_gyro_samples']}")
            print(f"  Duration: {gyro_data['gyro_duration']:.2f}s")

            # Show orientation
            eulers = np.array([f['cori_euler'] for f in frames])
            print(f"\n  Integrated orientation range:")
            print(f"    Roll:  {eulers[:,0].min():>8.2f}° to {eulers[:,0].max():>8.2f}°")
            print(f"    Pitch: {eulers[:,1].min():>8.2f}° to {eulers[:,1].max():>8.2f}°")
            print(f"    Yaw:   {eulers[:,2].min():>8.2f}° to {eulers[:,2].max():>8.2f}°")

            # Compare to CORI for old file
            if "172" in path:
                cori_euler = np.array([f['cori_euler'] for f in _gd['frames']])
                n_cmp = min(len(frames), len(_gd['frames']))

                # Align by subtracting mean offset
                gyro_euler = eulers[:n_cmp].copy()
                for ax in range(3):
                    offset = np.mean(cori_euler[:n_cmp, ax] - gyro_euler[:, ax])
                    gyro_euler[:, ax] += offset

                print(f"\n  COMPARISON vs CORI (after alignment):")
                for ax, name in [(0, 'Roll'), (1, 'Pitch'), (2, 'Yaw')]:
                    c = np.corrcoef(gyro_euler[:, ax], cori_euler[:n_cmp, ax])[0, 1]
                    rms = np.sqrt(np.mean((gyro_euler[:, ax] - cori_euler[:n_cmp, ax])**2))
                    print(f"    {name}: corr={c:.4f}, RMS error={rms:.2f}°")


if __name__ == "__main__":
    main()

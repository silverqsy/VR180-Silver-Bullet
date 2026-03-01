#!/usr/bin/env python3
"""
VR180 SBS Half-Equirectangular Video Processor - GUI Edition
"""

import sys
import subprocess
import json
import os
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

# Import select only on Unix/Mac (not available on Windows)
if sys.platform != 'win32':
    import select

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
    QMessageBox, QSplitter, QLineEdit, QStatusBar,
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


def get_spatial_path():
    """Get the path to bundled spatial or system spatial"""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check multiple possible locations
        base_path = Path(sys._MEIPASS)

        # Try _internal folder (Windows/Linux style)
        # On Windows, add .exe extension
        spatial_name = 'spatial.exe' if sys.platform == 'win32' else 'spatial'
        spatial_path = base_path / spatial_name
        if spatial_path.exists():
            return str(spatial_path)

        # Try macOS app bundle Resources folder
        if sys.platform == 'darwin':
            # Go up from _MEIPASS to find Resources
            resources_path = base_path.parent / 'Resources' / 'spatial'
            if resources_path.exists():
                return str(resources_path)

            # Try Frameworks folder
            frameworks_path = base_path.parent / 'Frameworks' / 'spatial'
            if frameworks_path.exists():
                return str(frameworks_path)

    # Check for spatial in system PATH
    spatial = shutil.which('spatial')
    if spatial:
        return spatial

    # Not found
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


def get_hw_encoder_args(codec: str, use_hw: bool, quality: int, bitrate: int, use_bitrate: bool, bit_depth: int = 8):
    """Get FFmpeg encoder arguments based on detected hardware.

    Args:
        codec: 'h264' or 'h265'
        use_hw: Whether to use hardware acceleration
        quality: CRF/CQ value
        bitrate: Target bitrate in Mbps
        use_bitrate: Whether to use bitrate mode instead of quality
        bit_depth: 8 or 10 bit

    Returns:
        List of FFmpeg encoder arguments
    """
    hw = detect_hardware_encoders()

    # Determine encoder
    hw_encoder = hw.get(codec) if use_hw else None

    if hw_encoder and use_hw:
        hw_type = hw['type']

        if hw_type == 'videotoolbox':
            # macOS VideoToolbox
            if use_bitrate:
                enc = ["-c:v", hw_encoder, "-b:v", f"{bitrate}M"]
            else:
                enc = ["-c:v", hw_encoder, "-q:v", str(min(100, quality * 2))]
            if codec == 'h265':
                enc.extend(["-tag:v", "hvc1"])
                if bit_depth == 10:
                    enc.extend(["-pix_fmt", "p010le"])

        elif hw_type == 'nvidia':
            # NVIDIA NVENC
            if use_bitrate:
                enc = ["-c:v", hw_encoder, "-preset", "p4", "-b:v", f"{bitrate}M"]
            else:
                enc = ["-c:v", hw_encoder, "-preset", "p4", "-cq", str(quality)]
            if codec == 'h265':
                enc.extend(["-tag:v", "hvc1"])
                if bit_depth == 10:
                    enc.extend(["-pix_fmt", "p010le"])

        elif hw_type == 'amd':
            # AMD AMF
            if use_bitrate:
                enc = ["-c:v", hw_encoder, "-quality", "balanced", "-b:v", f"{bitrate}M"]
            else:
                enc = ["-c:v", hw_encoder, "-quality", "balanced", "-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality)]
            if codec == 'h265':
                enc.extend(["-tag:v", "hvc1"])
                if bit_depth == 10:
                    enc.extend(["-pix_fmt", "p010le"])

        elif hw_type == 'intel':
            # Intel QuickSync
            if use_bitrate:
                enc = ["-c:v", hw_encoder, "-preset", "medium", "-b:v", f"{bitrate}M"]
            else:
                enc = ["-c:v", hw_encoder, "-preset", "medium", "-global_quality", str(quality)]
            if codec == 'h265':
                enc.extend(["-tag:v", "hvc1"])
                if bit_depth == 10:
                    enc.extend(["-pix_fmt", "p010le"])
        else:
            # Fallback to software
            hw_encoder = None

    # Software fallback
    if not hw_encoder:
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


def parse_gopro_cori_roll(file_path: str) -> list:
    """
    Parse GoPro .360 file to extract the APPLIED roll stabilization.

    GoPro stabilization works by:
    - CORI (Camera Orientation): Physical camera orientation (raw motion)
    - IORI (Image Orientation): How the image was rotated to counteract CORI

    The actual stabilization applied = IORI - CORI (the smoothing, not full leveling)
    To undo the baked-in stabilization, we need to apply the inverse of this difference.

    Returns: List of (timestamp_seconds, roll_degrees) tuples, one per frame.
             roll_degrees is the stabilization that was applied (IORI_roll - CORI_roll)
    """
    import struct
    import math

    # Extract GPMD stream using FFmpeg
    result = subprocess.run([
        get_ffmpeg_path(), '-i', file_path, '-map', '0:3', '-c', 'copy', '-f', 'rawvideo', '-'
    ], capture_output=True)

    if result.returncode != 0:
        raise Exception(f"Failed to extract GPMD data: {result.stderr.decode()}")

    data = result.stdout

    # Get video frame rate to calculate timestamps
    probe_result = subprocess.run([
        get_ffprobe_path(), '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,duration', '-of', 'json', file_path
    ], capture_output=True, text=True, creationflags=get_subprocess_flags())

    video_info = json.loads(probe_result.stdout)
    stream = video_info['streams'][0]

    # Parse frame rate (e.g., "30000/1001" or "30/1")
    fps_str = stream.get('r_frame_rate', '30/1')
    fps_parts = fps_str.split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    def quat_to_pitch(w, x, y, z):
        """Convert quaternion to pitch angle in degrees.

        IORI Pitch represents how GoPro's stabilization rotated the image.
        This is what we need to undo to fix the stereo roll offset.
        """
        # Extract pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        return math.degrees(pitch)

    def parse_iori_pitch(data):
        """Parse IORI (Image Orientation) pitch values from GPMF data."""
        pitch_samples = []
        idx = 0

        while True:
            idx = data.find(b'IORI', idx)
            if idx < 0:
                break

            # GPMF structure: FourCC(4) + Type(1) + StructSize(1) + Count(2) + Data
            if idx + 8 > len(data):
                break

            struct_size = data[idx + 5]  # Should be 8 (4 shorts = quaternion)
            count = struct.unpack('>H', data[idx + 6:idx + 8])[0]

            # Each quaternion is 8 bytes (4 x int16)
            for i in range(count):
                offset = idx + 8 + (i * struct_size)
                if offset + struct_size > len(data):
                    break

                # GoPro quaternions are Q15 fixed-point (divide by 32768)
                # Order is W, X, Z, Y in the file
                w, x, z, y = struct.unpack('>hhhh', data[offset:offset + 8])
                wn, xn, zn, yn = w / 32768.0, x / 32768.0, z / 32768.0, y / 32768.0

                # Get pitch value from IORI
                pitch = quat_to_pitch(wn, xn, yn, zn)
                pitch_samples.append(pitch)

            idx += 1

        return pitch_samples

    # Extract IORI pitch values (the image orientation applied by GoPro stabilization)
    iori_pitch_samples = parse_iori_pitch(data)

    if not iori_pitch_samples:
        raise Exception("No IORI data found in file")

    # Create timestamp-pitch pairs
    # IORI Pitch represents the stabilization applied - we'll invert this to undo it
    frame_duration = 1.0 / fps
    roll_data = []

    print(f"Found {len(iori_pitch_samples)} IORI pitch samples")
    for i, pitch in enumerate(iori_pitch_samples):
        timestamp = i * frame_duration
        roll_data.append((timestamp, pitch))

    print(f"Sample IORI pitch values (first 10): {[round(r[1], 2) for r in roll_data[:10]]}")

    return roll_data


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
    quality: int = 18
    bitrate: int = 200  # Mbps
    use_bitrate: bool = True  # If False, use CRF quality; if True, use bitrate
    prores_profile: str = "standard"
    use_hardware_accel: bool = True
    encoder_speed: str = "fast"  # fast, medium, slow
    lut_path: Optional[Path] = None  # Optional LUT file for color grading
    lut_intensity: float = 1.0  # LUT intensity 0.0 to 1.0
    gamma: float = 1.0  # Gamma: midtone adjustment via power function
    gain: float = 1.0  # Gain: overall brightness multiplier, affects highlights
    lift: float = 0.0  # Lift: raises/lowers black level (shadows only, preserves white point)
    h265_bit_depth: int = 8  # 8-bit or 10-bit for H.265
    inject_vr180_metadata: bool = False  # Inject VR180 metadata for YouTube
    vision_pro_mode: str = "standard"  # standard, hvc1, or mvhevc
    # Trim settings
    trim_start: float = 0.0  # Start time in seconds
    trim_end: float = 0.0  # End time in seconds (0 = use full duration)
    # GoPro roll stabilization fix
    gopro_roll_data: Optional[list] = None  # List of (timestamp, roll_degrees) tuples from .360 file
    roll_stabilized: bool = False  # VR180 Roll Stabilized: keep left eye fixed, apply correction to right eye only


class FrameExtractor(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    raw_frame_ready = pyqtSignal(np.ndarray)  # New signal for unprocessed frame
    error = pyqtSignal(str)

    def __init__(self, video_path: Path, timestamp: float = 0.0, filter_complex: str = None, extract_raw: bool = False):
        super().__init__()
        self.video_path = video_path
        self.timestamp = timestamp
        self.filter_complex = filter_complex
        self.extract_raw = extract_raw  # If True, extract without filters for caching
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

    def run(self):
        try:
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
        try:
            cfg = self.config

            # If we have GoPro roll data, use per-frame processing
            if cfg.gopro_roll_data:
                self._run_with_gopro_roll()
                return

            self.status.emit("Processing...")

            # Detect codec
            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                   "-show_entries", "stream=codec_name", "-of", "json",
                                   str(cfg.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            codec = json.loads(probe.stdout)["streams"][0]["codec_name"]
            if codec in ["hevc", "h265"]: codec = "h265"
            elif codec in ["prores", "prores_ks"]: codec = "prores"
            output_codec = codec if cfg.output_codec == "auto" else cfg.output_codec
            
            # Get video info
            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,pix_fmt,color_space,color_transfer,color_primaries",
                                   "-show_entries", "format=duration", "-of", "json",
                                   str(cfg.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            video_info = json.loads(probe.stdout)
            full_duration = float(video_info.get("format", {}).get("duration", 0))
            stream = video_info["streams"][0]

            # Calculate effective duration (considering trim)
            if cfg.trim_end > 0 and cfg.trim_end > cfg.trim_start:
                duration = cfg.trim_end - cfg.trim_start
                self.status.emit(f"Video duration: {full_duration:.2f}s (trimmed to {duration:.2f}s)")
            else:
                duration = full_duration
                self.status.emit(f"Video duration: {duration:.2f}s")

            # Check if input is 10-bit
            pix_fmt = stream.get("pix_fmt", "")
            is_10bit = "10le" in pix_fmt or "p010" in pix_fmt

            # Build filter
            filters = []

            # Only convert 10-bit to 8-bit if output is 8-bit H.265
            # For 10-bit H.265 or ProRes output, preserve 10-bit input
            need_8bit_conversion = is_10bit and output_codec == "h265" and cfg.h265_bit_depth == 8

            if need_8bit_conversion:
                self.status.emit("Converting 10-bit input to 8-bit for output...")
                filters.append("[0:v]format=yuv420p[input_8bit]")
                input_label = "[input_8bit]"
            else:
                if is_10bit and (output_codec == "prores" or (output_codec == "h265" and cfg.h265_bit_depth == 10)):
                    self.status.emit("Detected 10-bit input - preserving 10-bit for output...")
                input_label = "[0:v]"

            if cfg.global_shift != 0:
                shift = cfg.global_shift
                if shift > 0:
                    filters.extend([f"{input_label}split=2[sh_a][sh_b]", f"[sh_a]crop={shift}:ih:0:0[sh_right]",
                                   f"[sh_b]crop=iw-{shift}:ih:{shift}:0[sh_left]", f"[sh_left][sh_right]hstack=inputs=2[shifted]"])
                else:
                    abs_shift = abs(shift)
                    filters.extend([f"{input_label}split=2[sh_a][sh_b]", f"[sh_a]crop={abs_shift}:ih:iw-{abs_shift}:0[sh_left]",
                                   f"[sh_b]crop=iw-{abs_shift}:ih:0:0[sh_right]", f"[sh_left][sh_right]hstack=inputs=2[shifted]"])
                input_label = "[shifted]"
            
            filters.extend([f"{input_label}split=2[full1][full2]", "[full1]crop=iw/2:ih:0:0[left_in]", "[full2]crop=iw/2:ih:iw/2:0[right_in]"])

            left_yaw = cfg.global_adjustment.yaw + cfg.stereo_offset.yaw
            left_pitch = cfg.global_adjustment.pitch + cfg.stereo_offset.pitch
            left_roll = cfg.global_adjustment.roll + cfg.stereo_offset.roll
            right_yaw = cfg.global_adjustment.yaw - cfg.stereo_offset.yaw
            right_pitch = cfg.global_adjustment.pitch - cfg.stereo_offset.pitch
            right_roll = cfg.global_adjustment.roll - cfg.stereo_offset.roll

            # Check if we have GoPro roll data - if so, use per-frame processing
            if cfg.gopro_roll_data:
                # Per-frame processing is handled separately in _process_with_gopro_roll
                # This branch should not be reached - gopro processing uses different code path
                pass
            else:
                # No GoPro roll data - apply standard v360 transformation
                if any([left_yaw, left_pitch, left_roll]):
                    filters.append(f"[left_in]v360=input=hequirect:output=hequirect:yaw={left_yaw}:pitch={left_pitch}:roll={left_roll}:interp=lanczos[left_out]")
                else:
                    filters.append("[left_in]null[left_out]")

                if any([right_yaw, right_pitch, right_roll]):
                    filters.append(f"[right_in]v360=input=hequirect:output=hequirect:yaw={right_yaw}:pitch={right_pitch}:roll={right_roll}:interp=lanczos[right_out]")
                else:
                    filters.append("[right_in]null[right_out]")

            # Combine left and right
            filters.append("[left_out][right_out]hstack=inputs=2[stacked]")

            # Apply pre-LUT color adjustments (gamma, white point, black point)
            pre_lut_filters = []
            current_label = "[stacked]"

            # Check if any pre-LUT adjustments are needed (ASC CDL style: Lift, Gamma, Gain)
            has_adjustments = (abs(cfg.gamma - 1.0) > 0.01 or
                             abs(cfg.gain - 1.0) > 0.01 or
                             abs(cfg.lift) > 0.01)

            if has_adjustments:
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
                if abs(cfg.lift) > 0.01:
                    lut_expr = f"({lut_expr}+{cfg.lift}*(1-{lut_expr}))"

                # Apply Gain: multiply the result
                # This scales highlights while preserving black point (after lift adjustment)
                # gain range: 0.5 to 2.0 (1.0 = neutral)
                if abs(cfg.gain - 1.0) > 0.01:
                    lut_expr = f"({lut_expr}*{cfg.gain})"

                # Clamp before gamma to avoid pow() on negative values
                lut_expr = f"clip({lut_expr},0,1)"

                # Apply Gamma: power function
                # gamma range: 0.1 to 3.0 (1.0 = neutral)
                if abs(cfg.gamma - 1.0) > 0.01:
                    power = 1.0 / cfg.gamma
                    lut_expr = f"pow({lut_expr},{power})"

                # Final clamp and denormalize back to pixel values
                lut_expr = f"clip({lut_expr},0,1)*maxval"

                filters.append(f"{current_label}lutrgb=r='{lut_expr}':g='{lut_expr}':b='{lut_expr}'[color_adjusted]")
                current_label = "[color_adjusted]"

            # Apply LUT if specified - optimized for performance
            if cfg.lut_path and cfg.lut_path.exists():
                # Escape the path for FFmpeg filter syntax
                lut_path_str = str(cfg.lut_path).replace('\\', '/').replace(':', '\\:')

                if cfg.lut_intensity >= 0.99:
                    # Full intensity - apply LUT directly without blending (much faster)
                    # Use tetrahedral interpolation for best performance
                    # Format to RGB for LUT processing, then back to YUV for encoding
                    filters.append(f"{current_label}format=gbrp,lut3d=file='{lut_path_str}':interp=tetrahedral[lut_final]")
                    current_label = "[lut_final]"
                elif cfg.lut_intensity > 0.01:
                    # Partial intensity - use same blend formula as preview for consistency
                    # Linear interpolation: original * (1-intensity) + lut * intensity
                    filters.append(f"{current_label}format=gbrp,split[original][lut_input]")
                    filters.append(f"[lut_input]lut3d=file='{lut_path_str}':interp=tetrahedral[lut_output]")
                    # Use blend with custom expression for accurate color mixing (matches preview)
                    filters.append(f"[original][lut_output]blend=all_expr='A*(1-{cfg.lut_intensity})+B*{cfg.lut_intensity}'[lut_final]")
                    current_label = "[lut_final]"
                else:
                    # No intensity - skip LUT
                    pass  # current_label stays as is

            # Finalize filter chain - add format conversion for ProRes if needed
            # We need to determine the output format before finalizing filters
            needs_prores = cfg.vision_pro_mode == "mvhevc" or output_codec == "prores"

            if needs_prores and sys.platform != 'darwin':
                # For Windows ProRes encoding, ensure correct pixel format in filter chain
                if output_codec == "prores" and cfg.prores_profile in ["4444", "4444xq"]:
                    filters.append(f"{current_label}format=yuv444p10le[out]")
                else:
                    filters.append(f"{current_label}format=yuv422p10le[out]")
            else:
                filters.append(f"{current_label}null[out]")

            # If outputting to MV-HEVC, use lossless intermediate to avoid double lossy compression
            if cfg.vision_pro_mode == "mvhevc":
                # Use ProRes HQ as lossless intermediate for MV-HEVC workflow
                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", "3"]  # ProRes HQ
                else:
                    enc = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]  # ProRes HQ (format set in filter)
            # Encoder settings with hardware acceleration - auto-detect GPU type
            elif output_codec == "h265":
                enc = get_hw_encoder_args('h265', cfg.use_hardware_accel, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate, cfg.h265_bit_depth)
            elif output_codec == "prores":
                profile_map = {"proxy": "0", "lt": "1", "standard": "2", "hq": "3", "4444": "4", "4444xq": "5"}
                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    # macOS VideoToolbox ProRes encoding
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", profile_map.get(cfg.prores_profile, "3")]
                else:
                    # Software ProRes - pixel format is set in filter chain for Windows
                    enc = ["-c:v", "prores_ks", "-profile:v", profile_map.get(cfg.prores_profile, "3"),
                           "-vendor", "apl0"]
            elif output_codec in ["h264", "libx264"]:
                enc = get_hw_encoder_args('h264', cfg.use_hardware_accel, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate)
            else:
                enc = ["-c:v", output_codec]
            
            # Build FFmpeg command with hardware decode for HEVC input
            # CRITICAL: Disable hardware decode when using LUT or complex color filters
            # GPU→CPU transfers for filter processing kills performance
            has_lut = cfg.lut_path and cfg.lut_path.exists() and cfg.lut_intensity > 0.01
            decode_args = []
            if codec == "h265" and sys.platform == 'darwin' and not has_lut and not has_adjustments:
                # Use VideoToolbox hardware decoding for HEVC (much faster for 10-bit)
                # Only when NOT using CPU-bound filters (LUT, color grading)
                decode_args = ["-hwaccel", "videotoolbox"]
            elif has_lut or has_adjustments:
                # When using LUT/color grading, use software decode and optimize for CPU
                self.status.emit("Using optimized software decoding for LUT processing...")

            # Add multi-threading optimization
            import os
            cpu_count = os.cpu_count() or 4
            thread_args = ["-threads", str(cpu_count)]
            filter_thread_args = ["-filter_threads", str(cpu_count)]

            # Build trim arguments
            trim_args = []
            if cfg.trim_start > 0:
                trim_args.extend(["-ss", str(cfg.trim_start)])
            if cfg.trim_end > 0 and cfg.trim_end > cfg.trim_start:
                # Use -t (duration) instead of -to for accurate trimming with -ss
                trim_duration = cfg.trim_end - cfg.trim_start
                trim_args.extend(["-t", str(trim_duration)])
                self.status.emit(f"Trimming: {cfg.trim_start:.2f}s to {cfg.trim_end:.2f}s (duration: {trim_duration:.2f}s)")

            cmd = [get_ffmpeg_path(), "-y"] + decode_args + thread_args + trim_args + [
                   "-i", str(cfg.input_path)] + filter_thread_args + [
                   "-filter_complex", ";".join(filters),
                   "-map", "[out]", "-map", "0:a?", "-c:a", "copy"] + enc + \
                   ["-progress", "pipe:1", "-stats_period", "0.5", str(cfg.output_path)]

            self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1, creationflags=get_subprocess_flags())
            process = self._process

            # Thread to read stderr
            import threading
            import queue
            stderr_queue = queue.Queue()

            def read_stderr():
                try:
                    for line in process.stderr:
                        stderr_queue.put(line)
                except:
                    pass

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Monitor both stdout and stderr
            progress_data = {}
            while True:
                if self._cancelled:
                    process.terminate()
                    process.wait(timeout=5)
                    self.finished_signal.emit(False, "Cancelled")
                    return

                # Check if process finished
                if process.poll() is not None:
                    break

                # Try to read stderr (non-blocking)
                try:
                    while not stderr_queue.empty():
                        stderr_line = stderr_queue.get_nowait().strip()
                        if stderr_line:
                            self.output_line.emit(f"[stderr] {stderr_line}")
                except:
                    pass

                # Try to read stdout with timeout
                try:
                    if sys.platform != 'win32':
                        ready, _, _ = select.select([process.stdout], [], [], 0.5)
                        if not ready:
                            continue

                    line = process.stdout.readline()
                    if not line:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    # Emit every line to the output display
                    self.output_line.emit(line)

                    # Parse key=value for specific handling
                    if '=' in line:
                        key, value = line.split('=', 1)
                        progress_data[key] = value

                        # When we get 'progress' marker, calculate percentage and show FPS
                        if key == 'progress':
                            fps_str = ""
                            pct_str = ""

                            # Get FPS
                            if 'fps' in progress_data:
                                try:
                                    fps = float(progress_data['fps'])
                                    if fps > 0:
                                        fps_str = f"{fps:.1f} fps"
                                except:
                                    pass

                            # Calculate percentage from out_time
                            time_s = None

                            # Try different time formats
                            if 'out_time' in progress_data:
                                # Format: "00:00:47.500000"
                                try:
                                    time_str = progress_data['out_time']
                                    parts = time_str.split(':')
                                    if len(parts) == 3:
                                        hours = int(parts[0])
                                        minutes = int(parts[1])
                                        seconds = float(parts[2])
                                        time_s = hours * 3600 + minutes * 60 + seconds
                                except:
                                    pass
                            elif 'out_time_ms' in progress_data:
                                try:
                                    time_s = int(progress_data['out_time_ms']) / 1000.0
                                except:
                                    pass
                            elif 'out_time_us' in progress_data:
                                try:
                                    time_s = int(progress_data['out_time_us']) / 1000000.0
                                except:
                                    pass

                            # Calculate percentage
                            if time_s is not None and duration > 0:
                                try:
                                    pct = min(99, int((time_s / duration) * 100))
                                    pct_str = f"{pct}%"
                                except:
                                    pass

                            # Show status with both percentage and FPS
                            status_parts = []
                            if pct_str:
                                status_parts.append(pct_str)
                            if fps_str:
                                status_parts.append(fps_str)
                            if status_parts:
                                self.status.emit(f"Processing... {' - '.join(status_parts)}")

                            # Clear for next update
                            progress_data = {}
                except:
                    pass

            # Wait for process to finish - no timeout, let it run as long as needed
            process.wait()

            # Read any remaining stderr
            try:
                while not stderr_queue.empty():
                    stderr_line = stderr_queue.get_nowait().strip()
                    if stderr_line:
                        self.output_line.emit(f"[stderr] {stderr_line}")
            except:
                pass

            # Check return code - only 0 means success
            if process.returncode == 0:
                self.progress.emit(100)

                # Track completion messages
                completion_tags = []

                # Inject VR180 metadata for YouTube if requested
                if cfg.inject_vr180_metadata:
                    self.status.emit("Injecting VR180 metadata for YouTube...")
                    try:
                        self._inject_vr180_metadata(cfg.output_path)
                        completion_tags.append("YouTube VR180")
                    except Exception as meta_error:
                        completion_tags.append(f"VR180 metadata failed: {meta_error}")

                # Add Vision Pro processing if requested
                if cfg.vision_pro_mode == "hvc1":
                    self.status.emit("Adding hvc1 tag for Apple compatibility...")
                    try:
                        self._inject_hvc1_tag(cfg.output_path)
                        completion_tags.append("Apple compatible")
                    except Exception as meta_error:
                        completion_tags.append(f"hvc1 tag failed: {meta_error}")
                elif cfg.vision_pro_mode == "mvhevc":
                    self.status.emit("Converting to MV-HEVC for Vision Pro...")
                    try:
                        self._convert_to_mvhevc(cfg)
                        completion_tags.append("Vision Pro MV-HEVC")
                    except Exception as meta_error:
                        completion_tags.append(f"MV-HEVC conversion failed: {meta_error}")

                # Generate completion message
                if completion_tags:
                    self.finished_signal.emit(True, f"Complete! ({', '.join(completion_tags)})")
                else:
                    self.finished_signal.emit(True, "Complete!")
            else:
                # FFmpeg failed - read stderr for error details
                try:
                    stderr_output = process.stderr.read()
                    # Filter to show only error-related lines
                    error_lines = [line for line in stderr_output.split('\n')
                                  if any(keyword in line.lower() for keyword in
                                        ['error', 'failed', 'invalid', 'no such', 'cannot', 'unable'])]
                    if error_lines:
                        error_msg = '\n'.join(error_lines[:15])  # Show first 15 error lines
                    else:
                        # No specific errors found, show last part of output
                        error_msg = '\n'.join(stderr_output.split('\n')[-30:])
                    self.finished_signal.emit(False, f"FFmpeg error (return code {process.returncode}):\n{error_msg}")
                except:
                    self.finished_signal.emit(False, f"FFmpeg failed with return code {process.returncode}")

        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def _run_with_gopro_roll(self):
        """Process video with per-frame GoPro roll correction using OpenCV.

        Uses OpenCV for fast in-memory equirectangular roll transformation instead of
        spawning FFmpeg for each frame.
        """
        import bisect
        import math

        if not HAS_CV2:
            self.finished_signal.emit(False, "OpenCV (cv2) is required for GoPro roll correction")
            return

        cfg = self.config
        self.status.emit("Processing with GoPro roll correction (OpenCV)...")

        try:
            # Get video info
            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,r_frame_rate,pix_fmt,codec_name",
                                   "-show_entries", "format=duration", "-of", "json",
                                   str(cfg.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            video_info = json.loads(probe.stdout)
            stream = video_info["streams"][0]
            full_duration = float(video_info.get("format", {}).get("duration", 0))
            width = stream["width"]
            height = stream["height"]
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

            self.status.emit(f"Will process {total_frames} frames at {fps:.2f} fps using OpenCV")

            # Prepare roll data lookup
            roll_timestamps = [t for t, _ in cfg.gopro_roll_data]
            roll_values = [r for _, r in cfg.gopro_roll_data]

            def get_roll_at_time(t):
                """Interpolate roll value at given time"""
                if t <= roll_timestamps[0]:
                    return roll_values[0]
                if t >= roll_timestamps[-1]:
                    return roll_values[-1]
                idx = bisect.bisect_right(roll_timestamps, t) - 1
                if idx >= len(roll_timestamps) - 1:
                    return roll_values[-1]
                t0, t1 = roll_timestamps[idx], roll_timestamps[idx + 1]
                v0, v1 = roll_values[idx], roll_values[idx + 1]
                alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
                return v0 + alpha * (v1 - v0)

            # Base adjustments
            left_yaw = cfg.global_adjustment.yaw + cfg.stereo_offset.yaw
            left_pitch = cfg.global_adjustment.pitch + cfg.stereo_offset.pitch
            left_roll_base = cfg.global_adjustment.roll + cfg.stereo_offset.roll
            right_yaw = cfg.global_adjustment.yaw - cfg.stereo_offset.yaw
            right_pitch = cfg.global_adjustment.pitch - cfg.stereo_offset.pitch
            right_roll_base = cfg.global_adjustment.roll - cfg.stereo_offset.roll

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

                # Apply remapping with bilinear interpolation
                result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return result

            # Precompute remap tables for common roll values to speed up processing
            remap_cache = {}

            def get_remap_tables(h, w, roll_deg, tolerance=0.1):
                """Get or create remap tables for given roll, with caching."""
                # Round roll to nearest tolerance for caching
                cache_key = (h, w, round(roll_deg / tolerance) * tolerance)
                if cache_key in remap_cache:
                    return remap_cache[cache_key]

                roll_rad = math.radians(roll_deg)
                cos_r = math.cos(roll_rad)
                sin_r = math.sin(roll_rad)

                u = np.linspace(0, 1, w, dtype=np.float32)
                v = np.linspace(0, 1, h, dtype=np.float32)
                u_grid, v_grid = np.meshgrid(u, v)

                lon = (u_grid - 0.5) * math.pi
                lat = (0.5 - v_grid) * math.pi

                cos_lat = np.cos(lat)
                x = cos_lat * np.sin(lon)
                y = np.sin(lat)
                z = cos_lat * np.cos(lon)

                x_new = cos_r * x - sin_r * y
                y_new = sin_r * x + cos_r * y
                z_new = z

                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)

                u_new = (lon_new / math.pi) + 0.5
                v_new = 0.5 - (lat_new / math.pi)

                map_x = np.clip((u_new * w).astype(np.float32), 0, w - 1)
                map_y = np.clip((v_new * h).astype(np.float32), 0, h - 1)

                # Limit cache size
                if len(remap_cache) > 100:
                    remap_cache.clear()

                remap_cache[cache_key] = (map_x, map_y)
                return map_x, map_y

            # Load LUT if specified
            lut_3d = None
            if cfg.lut_path and cfg.lut_path.exists():
                try:
                    lut_3d = load_cube_lut(str(cfg.lut_path))
                    self.status.emit(f"Loaded LUT: {cfg.lut_path.name}")
                except Exception as e:
                    self.status.emit(f"Warning: Failed to load LUT: {e}")

            def apply_color_adjustments(frame, lift, gamma, gain):
                """Apply ASC CDL color adjustments using OpenCV."""
                if abs(lift) < 0.01 and abs(gamma - 1.0) < 0.01 and abs(gain - 1.0) < 0.01:
                    return frame

                # Convert to float for processing
                img = frame.astype(np.float32) / 255.0

                # Apply Lift: x + lift * (1-x)
                if abs(lift) > 0.01:
                    img = img + lift * (1.0 - img)

                # Apply Gain: x * gain
                if abs(gain - 1.0) > 0.01:
                    img = img * gain

                # Clamp before gamma
                img = np.clip(img, 0.0, 1.0)

                # Apply Gamma: x^(1/gamma)
                if abs(gamma - 1.0) > 0.01:
                    power = 1.0 / gamma
                    img = np.power(img, power)

                # Clamp and convert back to uint8
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                return img

            def apply_lut_3d(frame, lut, intensity=1.0):
                """Apply 3D LUT to frame at reduced resolution for speed.

                Downscales, applies trilinear LUT, then upscales back.
                """
                if lut is None or intensity < 0.01:
                    return frame

                h, w = frame.shape[:2]
                lut_size = lut.shape[0]

                # Downscale for LUT application (4x faster at half res)
                scale = 2
                small_h, small_w = h // scale, w // scale
                small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

                # Convert to float
                img = small.astype(np.float32) / 255.0

                # Compute LUT indices
                b_idx = np.clip(img[:, :, 0] * (lut_size - 1), 0, lut_size - 1.001)
                g_idx = np.clip(img[:, :, 1] * (lut_size - 1), 0, lut_size - 1.001)
                r_idx = np.clip(img[:, :, 2] * (lut_size - 1), 0, lut_size - 1.001)

                # Integer and fractional parts
                b0 = b_idx.astype(np.int32)
                g0 = g_idx.astype(np.int32)
                r0 = r_idx.astype(np.int32)
                b1 = np.minimum(b0 + 1, lut_size - 1)
                g1 = np.minimum(g0 + 1, lut_size - 1)
                r1 = np.minimum(r0 + 1, lut_size - 1)

                fb = (b_idx - b0)[:, :, np.newaxis]
                fg = (g_idx - g0)[:, :, np.newaxis]
                fr = (r_idx - r0)[:, :, np.newaxis]

                # Trilinear interpolation
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

                # Blend with original based on intensity
                if intensity < 1.0:
                    img_rgb = img[:, :, ::-1]
                    result_rgb = img_rgb + (result_rgb - img_rgb) * intensity

                # RGB to BGR and back to uint8
                result_small = np.clip(result_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)

                # Upscale back to original size
                result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LINEAR)
                return result

            # Precompute remap tables for yaw/pitch adjustments (these are constant per eye)
            yaw_pitch_remap_left = None
            yaw_pitch_remap_right = None

            def get_yaw_pitch_remap_tables(h, w, yaw_deg, pitch_deg):
                """Create remap tables for combined yaw and pitch rotation on half-equirectangular."""
                if abs(yaw_deg) < 0.01 and abs(pitch_deg) < 0.01:
                    return None

                yaw_rad = math.radians(yaw_deg)
                pitch_rad = math.radians(pitch_deg)

                cos_yaw = math.cos(yaw_rad)
                sin_yaw = math.sin(yaw_rad)
                cos_pitch = math.cos(pitch_rad)
                sin_pitch = math.sin(pitch_rad)

                u = np.linspace(0, 1, w, dtype=np.float32)
                v = np.linspace(0, 1, h, dtype=np.float32)
                u_grid, v_grid = np.meshgrid(u, v)

                # Convert to spherical angles (half equirectangular)
                lon = (u_grid - 0.5) * math.pi
                lat = (0.5 - v_grid) * math.pi

                # Convert to 3D cartesian
                cos_lat = np.cos(lat)
                x = cos_lat * np.sin(lon)
                y = np.sin(lat)
                z = cos_lat * np.cos(lon)

                # Apply pitch rotation (around X axis)
                y_pitch = cos_pitch * y - sin_pitch * z
                z_pitch = sin_pitch * y + cos_pitch * z
                x_pitch = x

                # Apply yaw rotation (around Y axis)
                x_new = cos_yaw * x_pitch + sin_yaw * z_pitch
                z_new = -sin_yaw * x_pitch + cos_yaw * z_pitch
                y_new = y_pitch

                # Convert back to spherical
                lat_new = np.arcsin(np.clip(y_new, -1, 1))
                lon_new = np.arctan2(x_new, z_new)

                u_new = (lon_new / math.pi) + 0.5
                v_new = 0.5 - (lat_new / math.pi)

                map_x = np.clip((u_new * w).astype(np.float32), 0, w - 1)
                map_y = np.clip((v_new * h).astype(np.float32), 0, h - 1)

                return (map_x, map_y)

            def process_frame_opencv(frame, left_roll, right_roll, global_shift,
                                    left_yaw, left_pitch, right_yaw, right_pitch):
                """Process a single SBS frame entirely using OpenCV.

                Applies: global shift, yaw/pitch per eye, roll per eye, color adjustments, LUT.
                """
                nonlocal yaw_pitch_remap_left, yaw_pitch_remap_right

                h, full_w = frame.shape[:2]
                half_w = full_w // 2

                # Apply global shift if needed
                if global_shift != 0:
                    frame = np.roll(frame, global_shift, axis=1)

                # Split into left and right
                left = frame[:, :half_w].copy()
                right = frame[:, half_w:].copy()

                # Apply yaw/pitch to each eye (precompute remap tables on first frame)
                if abs(left_yaw) >= 0.01 or abs(left_pitch) >= 0.01:
                    if yaw_pitch_remap_left is None:
                        yaw_pitch_remap_left = get_yaw_pitch_remap_tables(h, half_w, left_yaw, left_pitch)
                    if yaw_pitch_remap_left is not None:
                        left = cv2.remap(left, yaw_pitch_remap_left[0], yaw_pitch_remap_left[1],
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                if abs(right_yaw) >= 0.01 or abs(right_pitch) >= 0.01:
                    if yaw_pitch_remap_right is None:
                        yaw_pitch_remap_right = get_yaw_pitch_remap_tables(h, half_w, right_yaw, right_pitch)
                    if yaw_pitch_remap_right is not None:
                        right = cv2.remap(right, yaw_pitch_remap_right[0], yaw_pitch_remap_right[1],
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Apply roll to each eye
                if abs(left_roll) >= 0.01:
                    map_x, map_y = get_remap_tables(h, half_w, left_roll)
                    left = cv2.remap(left, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                if abs(right_roll) >= 0.01:
                    map_x, map_y = get_remap_tables(h, half_w, right_roll)
                    right = cv2.remap(right, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Combine back to SBS - swap order to match expected output format
                result = np.hstack([right, left])

                # Apply color adjustments (Lift/Gamma/Gain)
                result = apply_color_adjustments(result, cfg.lift, cfg.gamma, cfg.gain)

                # Apply LUT
                result = apply_lut_3d(result, lut_3d, cfg.lut_intensity)

                return result

            # Set up FFmpeg decode pipeline - simple raw decode, all processing in OpenCV
            decode_cmd = [
                get_ffmpeg_path(),
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(cfg.input_path),
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-v", "quiet",
                "-"
            ]

            # Set up FFmpeg encode pipeline
            # Determine encoder settings using hardware detection
            if output_codec == "h265":
                enc = get_hw_encoder_args('h265', cfg.use_hardware_accel, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate, cfg.h265_bit_depth)
            elif output_codec == "prores":
                profile_map = {"proxy": "0", "lt": "1", "standard": "2", "hq": "3", "4444": "4", "4444xq": "5"}
                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", profile_map.get(cfg.prores_profile, "3")]
                else:
                    enc = ["-c:v", "prores_ks", "-profile:v", profile_map.get(cfg.prores_profile, "3"), "-vendor", "apl0"]
            else:
                enc = get_hw_encoder_args('h264', cfg.use_hardware_accel, cfg.quality, cfg.bitrate,
                                         cfg.use_bitrate)

            encode_cmd = [
                get_ffmpeg_path(), "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-ss", str(start_time), "-t", str(duration),
                "-i", str(cfg.input_path),
                "-map", "0:v",
                "-map", "1:a?",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p"
            ] + enc + [str(cfg.output_path)]

            # Start decode and encode processes
            self.status.emit("Starting video decode...")
            decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          creationflags=get_subprocess_flags())
            encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, creationflags=get_subprocess_flags())

            frame_size = width * height * 3  # BGR24
            frame_count = 0
            last_update_frame = 0
            import time
            start_process_time = time.time()
            last_update_time = start_process_time

            try:
                while True:
                    if self._cancelled:
                        decode_proc.terminate()
                        encode_proc.terminate()
                        self.finished_signal.emit(False, "Cancelled")
                        return

                    # Read one frame
                    raw_frame = decode_proc.stdout.read(frame_size)
                    if len(raw_frame) < frame_size:
                        break

                    # Convert to numpy array (copy to make it writable)
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()

                    # Calculate roll for this frame
                    # Invert the IORI pitch to counter the stabilization that was applied
                    frame_time = start_time + (frame_count / fps)
                    gopro_roll = get_roll_at_time(frame_time)
                    roll_correction = -gopro_roll  # Negative to invert the stabilization

                    if cfg.roll_stabilized:
                        # Roll stabilized mode: keep left eye fixed, apply 2x to right eye
                        left_roll = left_roll_base
                        right_roll = right_roll_base - (2 * roll_correction)
                    else:
                        # Normal mode: split correction between both eyes
                        left_roll = left_roll_base + roll_correction
                        right_roll = right_roll_base - roll_correction

                    # Process frame entirely with OpenCV
                    processed = process_frame_opencv(frame, left_roll, right_roll, cfg.global_shift,
                                                    left_yaw, left_pitch, right_yaw, right_pitch)

                    # Write to encoder
                    try:
                        encode_proc.stdin.write(processed.tobytes())
                        # Flush periodically on Windows to prevent buffer blocking
                        if sys.platform == 'win32' and frame_count % 30 == 0:
                            encode_proc.stdin.flush()
                    except (BrokenPipeError, OSError) as e:
                        # Encoder process died unexpectedly
                        raise Exception(f"Encoder process terminated unexpectedly: {e}")

                    frame_count += 1

                    # Update progress periodically
                    current_time = time.time()
                    if frame_count - last_update_frame >= 10:
                        progress = int((frame_count / total_frames) * 90)
                        self.progress.emit(min(progress, 90))

                        # Calculate FPS
                        elapsed = current_time - start_process_time
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        percent = (frame_count / total_frames) * 100

                        self.status.emit(f"{percent:.1f}% - Frame {frame_count}/{total_frames} @ {fps_actual:.1f} fps")
                        last_update_frame = frame_count
                        last_update_time = current_time

                # Close pipes and wait for processes
                decode_proc.stdout.close()

                # Flush and close stdin properly - important for Windows
                try:
                    encode_proc.stdin.flush()
                except:
                    pass
                try:
                    encode_proc.stdin.close()
                except:
                    pass

                self.status.emit("Finalizing video encoding...")

                # Use communicate() with timeout to avoid hanging
                # This also drains stderr to prevent pipe buffer from filling up
                try:
                    _, stderr = encode_proc.communicate(timeout=120)  # Longer timeout for finalization
                except subprocess.TimeoutExpired:
                    encode_proc.kill()
                    encode_proc.communicate()
                    raise Exception("Encoding timed out during finalization")

                if encode_proc.returncode != 0:
                    stderr_text = stderr.decode() if stderr else ""
                    raise Exception(f"Encoding failed: {stderr_text[-500:]}")

                self.progress.emit(100)
                self.finished_signal.emit(True, f"Complete! (GoPro roll fix: {frame_count} frames processed with OpenCV)")

            except Exception as e:
                try:
                    decode_proc.terminate()
                except:
                    pass
                try:
                    encode_proc.stdin.close()
                except:
                    pass
                try:
                    encode_proc.terminate()
                except:
                    pass
                raise e

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"GoPro Roll Fix Error: {error_msg}")
            self.finished_signal.emit(False, str(e))

    def _inject_hvc1_tag(self, video_path: Path):
        """Set hvc1 tag for Apple device compatibility (fast, no re-encode)"""
        import tempfile
        import shutil

        # Create temporary file
        temp_file = Path(tempfile.mktemp(suffix='.mov'))

        try:
            # Re-mux the file with hvc1 tag (required for Apple devices)
            # This is a fast operation that doesn't re-encode
            cmd = [
                get_ffmpeg_path(),
                '-i', str(video_path),
                '-c', 'copy',  # Copy streams without re-encoding
                '-tag:v', 'hvc1',  # Use hvc1 instead of hev1 (Apple requirement)
                '-movflags', '+faststart',  # Optimize for streaming
                str(temp_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=get_subprocess_flags())
            if result.returncode != 0:
                raise Exception(f"hvc1 tag injection failed: {result.stderr}")

            # Replace original file
            shutil.move(str(temp_file), str(video_path))

        finally:
            # Clean up temp file if it still exists
            if temp_file.exists():
                temp_file.unlink()

    def _convert_to_mvhevc(self, cfg: ProcessingConfig):
        """Convert to MV-HEVC with APMP metadata for Vision Pro using spatial CLI tool"""
        import tempfile
        import shutil

        # Check if spatial CLI is available
        spatial_path = get_spatial_path()
        if not spatial_path:
            raise Exception(
                "spatial CLI tool not found!\n\n"
                "To enable MV-HEVC encoding for Vision Pro, install Mike Swanson's spatial tool:\n\n"
                "brew install mikeswanson/spatial/spatial-media-kit-tool\n\n"
                "After installation, restart the app."
            )

        # Create temporary file for MV-HEVC output
        temp_file = Path(tempfile.mktemp(suffix='_mvhevc.mov'))

        try:
            # Determine bitrate for MV-HEVC encoding
            if cfg.output_codec == "auto":
                # Auto mode: use 350 Mbps for MV-HEVC
                bitrate_mbps = 350
            elif cfg.use_bitrate:
                # User specified bitrate: use it
                bitrate_mbps = cfg.bitrate
            else:
                # CRF mode: convert to estimated bitrate (higher quality = higher bitrate)
                # CRF 18 ≈ 100 Mbps, CRF 23 ≈ 50 Mbps, CRF 28 ≈ 25 Mbps
                bitrate_mbps = int(150 - (cfg.quality * 4))
                bitrate_mbps = max(25, min(200, bitrate_mbps))  # Clamp between 25-200

            # Build spatial make command
            # spatial uses VideoToolbox for encoding with full bitrate control + Vision Pro metadata
            cmd = [
                spatial_path, 'make',
                '--input', str(cfg.output_path),
                '--output', str(temp_file),
                '--format', 'sbs',  # Side-by-side stereo input
                '--bitrate', f'{bitrate_mbps}M',  # Full bitrate control!
                '--cdist', '65',  # Camera baseline in mm (user's VR180 camera)
                '--hfov', '180',  # Horizontal field of view for VR180 (full 180°)
                '--hadjust', '0',  # Horizontal disparity adjustment
                '--projection', 'halfEquirect',  # VR180 projection
                '--hero', 'left',  # Primary eye for 2D playback
                '--faststart',  # Optimize for streaming
                '--overwrite'
            ]

            self.status.emit(f"Encoding MV-HEVC at {bitrate_mbps}Mbps with Vision Pro metadata...")

            # Run spatial with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=get_subprocess_flags()
            )

            # Monitor progress
            for line in process.stdout:
                # spatial outputs encoding progress
                if '%' in line or 'frame' in line.lower():
                    self.status.emit(f"MV-HEVC encoding: {line.strip()}")

            process.wait()

            if process.returncode != 0:
                raise Exception(f"MV-HEVC conversion with spatial failed with code {process.returncode}")

            # Replace original file with MV-HEVC version
            shutil.move(str(temp_file), str(cfg.output_path))

            self.status.emit(f"✓ MV-HEVC encoding complete at {bitrate_mbps}Mbps with Vision Pro metadata")

        finally:
            # Clean up temp file if it still exists
            if temp_file.exists():
                temp_file.unlink()

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
        self.setWindowTitle("VR180 Silver Bullet")
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
        shift_group = QGroupBox("① Global Horizontal Shift")
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
        global_group = QGroupBox("② Global Panomap Adjustment")
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
        controls_layout.addWidget(global_group)

        # Stereo offset
        offset_group = QGroupBox("③ Stereo Offset (Applied Oppositely)")
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

        # GoPro Roll Fix section
        gopro_group = QGroupBox("③.5 GoPro Roll Stabilization Fix")
        gopro_layout = QVBoxLayout(gopro_group)
        gopro_layout.addWidget(QLabel("Extract roll data from GoPro .360 file to undo\nroll stabilization during export."))

        gopro_btn_layout = QHBoxLayout()
        self.gopro_roll_btn = QPushButton("Load .360 File")
        self.gopro_roll_btn.setToolTip("Select a GoPro .360 file to extract roll stabilization data")
        gopro_btn_layout.addWidget(self.gopro_roll_btn)

        self.gopro_roll_clear_btn = QPushButton("Clear")
        self.gopro_roll_clear_btn.setToolTip("Clear loaded roll data")
        self.gopro_roll_clear_btn.setEnabled(False)
        gopro_btn_layout.addWidget(self.gopro_roll_clear_btn)

        self.seek_zero_btn = QPushButton("Seek 0 Correction Frame")
        self.seek_zero_btn.setToolTip("Jump to the next frame where IORI pitch correction is 0")
        self.seek_zero_btn.setEnabled(False)
        gopro_btn_layout.addWidget(self.seek_zero_btn)

        gopro_btn_layout.addStretch()
        gopro_layout.addLayout(gopro_btn_layout)

        self.gopro_roll_status = QLabel("No roll data loaded")
        self.gopro_roll_status.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        gopro_layout.addWidget(self.gopro_roll_status)

        # VR180 Roll stabilized option
        self.roll_stabilized_checkbox = QCheckBox("VR180 Roll Stabilized")
        self.roll_stabilized_checkbox.setToolTip("Enable if source video has roll stabilization baked in.\nKeeps left eye untouched, applies correction to right eye only.")
        self.roll_stabilized_checkbox.setChecked(False)
        gopro_layout.addWidget(self.roll_stabilized_checkbox)

        controls_layout.addWidget(gopro_group)

        # Output settings
        output_group = QGroupBox("④ Output Settings")
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

        # Hardware acceleration checkbox
        self.hw_accel_checkbox = QCheckBox("Hardware Acceleration")
        self.hw_accel_checkbox.setChecked(True)
        output_layout.addWidget(self.hw_accel_checkbox, 6, 0, 1, 2)

        # H.265 bit depth
        self.h265_bit_depth_label = QLabel("H.265 Bit Depth:")
        output_layout.addWidget(self.h265_bit_depth_label, 7, 0)
        self.h265_bit_depth_combo = QComboBox()
        self.h265_bit_depth_combo.addItems(["8-bit", "10-bit"])
        self.h265_bit_depth_combo.setToolTip("8-bit: Standard compatibility\n10-bit: Higher quality, better gradients")
        output_layout.addWidget(self.h265_bit_depth_combo, 7, 1)

        # Pre-LUT color adjustments (ASC CDL: Lift, Gamma, Gain)
        output_layout.addWidget(QLabel("Lift:"), 8, 0)
        self.lift_slider = SliderWithSpinBox(-100, 100, 0, decimals=0, step=1, suffix="")
        self.lift_slider.setToolTip("Lift (Offset): Raises/lowers black level, affects shadows most\nNegative = crush blacks, 0 = neutral, Positive = lift shadows")
        output_layout.addWidget(self.lift_slider, 8, 1)

        output_layout.addWidget(QLabel("Gamma:"), 9, 0)
        self.gamma_slider = SliderWithSpinBox(10, 300, 100, decimals=0, step=1, suffix="")
        self.gamma_slider.setToolTip("Gamma (Power): Adjusts midtones while preserving black/white points\n<100 = darker midtones, 100 = neutral, >100 = brighter midtones")
        output_layout.addWidget(self.gamma_slider, 9, 1)

        output_layout.addWidget(QLabel("Gain:"), 10, 0)
        self.gain_slider = SliderWithSpinBox(50, 200, 100, decimals=0, step=1, suffix="")
        self.gain_slider.setToolTip("Gain (Slope): Overall brightness multiplier, affects highlights most\n<100 = darker, 100 = neutral, >100 = brighter")
        output_layout.addWidget(self.gain_slider, 10, 1)

        # LUT file selection (spanning both columns for more space)
        output_layout.addWidget(QLabel("LUT File:"), 11, 0)
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
        output_layout.addLayout(lut_file_layout, 11, 1)

        # LUT intensity slider
        output_layout.addWidget(QLabel("LUT Intensity:"), 12, 0)
        self.lut_intensity_slider = SliderWithSpinBox(0, 100, 100, 0, 1, "%")
        self.lut_intensity_slider.setToolTip("Blend strength: 0% = no LUT, 100% = full LUT")
        output_layout.addWidget(self.lut_intensity_slider, 12, 1)

        # VR180 metadata checkbox for YouTube
        self.vr180_metadata_checkbox = QCheckBox("Inject VR180 Metadata for YouTube")
        self.vr180_metadata_checkbox.setToolTip("Add VR180 spherical metadata for YouTube upload\n"
                                                 "Uses Spherical Video V2 specification (fast, no re-encode)")
        output_layout.addWidget(self.vr180_metadata_checkbox, 13, 0, 1, 2)

        # Vision Pro / Apple compatibility mode (macOS only)
        if IS_MACOS:
            output_layout.addWidget(QLabel("Vision Pro Mode:"), 14, 0)
            self.vision_pro_combo = QComboBox()
            self.vision_pro_combo.addItems([
                "Standard (no special metadata)",
                "Apple Compatible (hvc1 tag - fast)",
                "Vision Pro MV-HEVC (full APMP - slow)"
            ])
            self.vision_pro_combo.setToolTip(
                "Standard: Normal video output\n"
                "Apple Compatible: Add hvc1 tag (fast, no re-encode)\n"
                "Vision Pro MV-HEVC: Full spatial video with APMP metadata (re-encodes to MV-HEVC)"
            )
            output_layout.addWidget(self.vision_pro_combo, 14, 1)
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

        # FFmpeg output text box
        self.ffmpeg_output = QTextEdit()
        self.ffmpeg_output.setReadOnly(True)
        self.ffmpeg_output.setMaximumHeight(100)
        self.ffmpeg_output.setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 11px; background-color: #1e1e1e; color: #d4d4d4; }")
        self.ffmpeg_output.setPlaceholderText("FFmpeg output will appear here...")
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
        self.browse_output_btn.clicked.connect(self._browse_output)
        self.preview_mode_combo.currentIndexChanged.connect(self._schedule_preview_update)
        self.zoom_in_btn.clicked.connect(lambda: self._zoom(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self._zoom(0.8))
        self.zoom_reset_btn.clicked.connect(self._zoom_reset)
        self.timeline_slider.sliderMoved.connect(self._timeline_dragging)
        self.timeline_slider.valueChanged.connect(self._timeline_changed)
        self.global_shift_slider.valueChanged.connect(self._schedule_preview_update)
        self.global_yaw.valueChanged.connect(self._schedule_preview_update)
        self.global_pitch.valueChanged.connect(self._schedule_preview_update)
        self.global_roll.valueChanged.connect(self._schedule_preview_update)
        self.stereo_yaw_offset.valueChanged.connect(self._schedule_preview_update)
        self.stereo_pitch_offset.valueChanged.connect(self._schedule_preview_update)
        self.stereo_roll_offset.valueChanged.connect(self._schedule_preview_update)
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
        self.lut_intensity_slider.slider.sliderReleased.connect(self._schedule_preview_update)
        self.vision_pro_combo.currentIndexChanged.connect(self._update_vision_pro_mode)
        self.eye_toggle_btn.clicked.connect(self._toggle_eye)
        self.preview_mode_combo.currentIndexChanged.connect(self._update_preview_mode_ui)

        # Trim controls
        self.set_in_btn.clicked.connect(self._set_in_point)
        self.set_out_btn.clicked.connect(self._set_out_point)
        self.clear_trim_btn.clicked.connect(self._clear_trim)
        self.in_time_edit.editingFinished.connect(self._in_time_edited)
        self.out_time_edit.editingFinished.connect(self._out_time_edited)

        # GoPro roll fix controls
        self.gopro_roll_btn.clicked.connect(self._load_gopro_roll_data)
        self.gopro_roll_clear_btn.clicked.connect(self._clear_gopro_roll_data)
        self.seek_zero_btn.clicked.connect(self._seek_zero_correction_frame)

        # Initial state
        self._update_codec_settings()
        self._update_vision_pro_mode()
        self._update_preview_mode_ui()
    
    def _browse_input(self):
        # Use last used folder or home directory
        last_folder = self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", last_folder, "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.mts *.m2ts *.osv)")
        if path:
            # Remember the folder for next time
            self.settings.setValue("last_input_folder", str(Path(path).parent))
            self.config.input_path = Path(path)
            self.input_path_edit.setText(path)
            output = Path(path).parent / f"{Path(path).stem}_adjusted{Path(path).suffix}"
            self.output_path_edit.setText(str(output))
            self.config.output_path = output
            self._load_video_info()
            self.process_btn.setEnabled(True)
    
    def _browse_output(self):
        # Use current output path, or last input folder, or home
        default_path = self.output_path_edit.text() or self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", default_path, "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.mts *.m2ts *.osv)")
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

    def _load_gopro_roll_data(self):
        """Load roll stabilization data from a GoPro .360 file"""
        last_folder = self.settings.value("last_input_folder", str(Path.home()), type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GoPro .360 File", last_folder,
            "GoPro 360 Files (*.360);;All Files (*.*)"
        )
        if not path:
            return

        try:
            self.gopro_roll_status.setText("Parsing CORI data...")
            self.gopro_roll_status.setStyleSheet("QLabel { color: #0066cc; font-style: italic; }")
            QApplication.processEvents()

            # Parse the roll data
            roll_data = parse_gopro_cori_roll(path)
            self.config.gopro_roll_data = roll_data

            # Calculate statistics
            rolls = [r for _, r in roll_data]
            min_roll = min(rolls)
            max_roll = max(rolls)
            duration = roll_data[-1][0] if roll_data else 0

            self.gopro_roll_status.setText(
                f"Loaded {len(roll_data)} samples ({duration:.1f}s)\n"
                f"IORI pitch range: {min_roll:.1f}° to {max_roll:.1f}°"
            )
            self.gopro_roll_status.setStyleSheet("QLabel { color: #009900; }")
            self.gopro_roll_clear_btn.setEnabled(True)
            self.seek_zero_btn.setEnabled(True)

            self.status_bar.showMessage(f"Loaded GoPro roll data: {len(roll_data)} samples")

        except Exception as e:
            self.gopro_roll_status.setText(f"Error: {str(e)}")
            self.gopro_roll_status.setStyleSheet("QLabel { color: #cc0000; }")
            self.config.gopro_roll_data = None
            QMessageBox.warning(self, "Error", f"Failed to parse GoPro file:\n{str(e)}")

    def _clear_gopro_roll_data(self):
        """Clear the loaded GoPro roll data"""
        self.config.gopro_roll_data = None
        self.gopro_roll_status.setText("No roll data loaded")
        self.gopro_roll_status.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        self.gopro_roll_clear_btn.setEnabled(False)
        self.seek_zero_btn.setEnabled(False)
        self.status_bar.showMessage("GoPro roll data cleared")

    def _seek_zero_correction_frame(self):
        """Jump to the next frame where IORI pitch correction is 0 (or closest to 0)"""
        if not self.config.gopro_roll_data:
            return

        # Get current time from timeline and video duration
        current_time = self.timeline_slider.value() / 1000.0
        max_time = self.timeline_slider.maximum() / 1000.0  # Video duration

        # Find the next frame with pitch closest to 0, starting from current position
        roll_data = self.config.gopro_roll_data
        best_idx = None
        best_pitch = float('inf')

        # Also track the minimum pitch in range as fallback
        min_idx = None
        min_pitch = float('inf')

        # First, look for frames after current time with pitch < 0.1
        for i, (timestamp, pitch) in enumerate(roll_data):
            if timestamp > max_time:  # Don't go beyond video duration
                break
            # Track minimum pitch as fallback
            if abs(pitch) < abs(min_pitch):
                min_pitch = pitch
                min_idx = i
            if timestamp > current_time + 0.01:  # Small offset to avoid finding same frame
                if abs(pitch) < 0.1:  # Must be very close to 0
                    best_pitch = pitch
                    best_idx = i
                    break

        # If no good frame found after current, wrap around to beginning
        if best_idx is None:
            for i, (timestamp, pitch) in enumerate(roll_data):
                if timestamp > max_time:  # Don't go beyond video duration
                    break
                if abs(pitch) < 0.1:  # Must be very close to 0
                    best_pitch = pitch
                    best_idx = i
                    break

        # If still not found, use the frame with minimum pitch value
        if best_idx is None and min_idx is not None:
            best_idx = min_idx
            best_pitch = min_pitch

        if best_idx is not None:
            target_time = roll_data[best_idx][0]
            target_ms = int(target_time * 1000)
            self.timeline_slider.setValue(target_ms)
            self._timeline_changed(target_ms)
            self.status_bar.showMessage(f"Jumped to frame {best_idx} at {target_time:.2f}s (IORI pitch: {best_pitch:.2f}°)")

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

        # Auto-update output file extension based on codec
        # ProRes MUST use .mov container, H.265 can use .mp4
        if self.output_path_edit.text():
            output_path = Path(self.output_path_edit.text())
            if is_prores and output_path.suffix.lower() != '.mov':
                # Change to .mov for ProRes
                new_path = output_path.with_suffix('.mov')
                self.output_path_edit.setText(str(new_path))
                self.config.output_path = new_path
            elif is_h265 and output_path.suffix.lower() == '.mov':
                # Change to .mp4 for H.265 (optional, user can keep .mov)
                new_path = output_path.with_suffix('.mp4')
                self.output_path_edit.setText(str(new_path))
                self.config.output_path = new_path

    def _update_encoding_mode(self):
        """Enable/disable quality or bitrate spinbox based on radio selection"""
        use_crf = self.use_crf_radio.isChecked()
        self.quality_spinbox.setEnabled(use_crf)
        self.bitrate_spinbox.setEnabled(not use_crf)

    def _update_vision_pro_mode(self):
        """Gray out YouTube metadata checkbox when MV-HEVC is selected (macOS only)"""
        if IS_MACOS:
            is_mvhevc = self.vision_pro_combo.currentIndex() == 2  # Vision Pro MV-HEVC
            self.vr180_metadata_checkbox.setEnabled(not is_mvhevc)
            if is_mvhevc:
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

    def _load_video_info(self):
        if not self.config.input_path: return
        try:
            result = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height", "-show_entries", "format=duration",
                                    "-of", "json", str(self.config.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
            info = json.loads(result.stdout)
            self.video_duration = float(info.get("format", {}).get("duration", 0))
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

            self.status_bar.showMessage(f"Loaded: {self.config.input_path.name} ({width}x{info['streams'][0]['height']})")
            self._extract_frame(0)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
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
            # Need to decode new frame - extract raw frame first for caching
            # OpenCV will apply all adjustments after frame is loaded
            self.status_bar.showMessage(f"Loading frame at {timestamp:.2f}s...")
            self.extractor = FrameExtractor(self.config.input_path, timestamp, extract_raw=True)
            self.extractor.raw_frame_ready.connect(self._on_raw_frame_extracted)
            self.extractor.error.connect(lambda e: self.status_bar.showMessage(f"Error: {e}"))
            self.extractor.start()
    
    def _build_preview_filter(self):
        shift = self.global_shift_slider.value()
        yaw, pitch, roll = self.global_yaw.value(), self.global_pitch.value(), self.global_roll.value()
        syaw, spitch, sroll = self.stereo_yaw_offset.value(), self.stereo_pitch_offset.value(), self.stereo_roll_offset.value()
        lut_path = self.lut_path_edit.text()
        lut_intensity = self.lut_intensity_slider.value() / 100.0

        # Get color adjustment values (ASC CDL: Lift, Gamma, Gain)
        lift = self.lift_slider.value() / 100.0  # -1.0 to 1.0
        gamma = self.gamma_slider.value() / 100.0  # 0.1 to 3.0
        gain = self.gain_slider.value() / 100.0  # 0.5 to 2.0

        # Check if any adjustments are needed
        has_adjustments = any([shift, yaw, pitch, roll, syaw, spitch, sroll])
        has_lut = lut_path and Path(lut_path).exists() and lut_intensity > 0.01
        has_color_adjustments = (abs(lift) > 0.01 or
                                abs(gamma - 1.0) > 0.01 or
                                abs(gain - 1.0) > 0.01)

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

            filters.append(f"{current_label}lutrgb=r='{lut_expr}':g='{lut_expr}':b='{lut_expr}'[color_adjusted]")
            current_label = "[color_adjusted]"

        # Apply LUT if specified - optimized for performance
        if has_lut:
            lut_path_str = lut_path.replace('\\', '/').replace(':', '\\:')
            if lut_intensity >= 0.99:
                # Full intensity - apply LUT directly without blending (much faster)
                filters.append(f"{current_label}format=gbrp,lut3d=file='{lut_path_str}':interp=tetrahedral[out]")
            elif lut_intensity > 0.01:
                # Partial intensity - use tetrahedral interpolation for better performance
                filters.append(f"{current_label}format=gbrp,split[original][lut_input]")
                filters.append(f"[lut_input]lut3d=file='{lut_path_str}':interp=tetrahedral[lut_output]")
                filters.append(f"[original][lut_output]blend=all_expr='A*(1-{lut_intensity})+B*{lut_intensity}'[out]")
            else:
                filters.append(f"{current_label}null[out]")
        else:
            filters.append(f"{current_label}null[out]")

        return ";".join(filters)
    
    def _on_raw_frame_extracted(self, frame):
        """Callback when raw frame is extracted - cache it and apply filters"""
        self.cached_raw_frame = frame
        self.cached_timestamp = self.preview_timestamp
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

        left_yaw = global_yaw + stereo_yaw
        left_pitch = global_pitch + stereo_pitch
        left_roll = global_roll + stereo_roll
        right_yaw = global_yaw - stereo_yaw
        right_pitch = global_pitch - stereo_pitch
        right_roll = global_roll - stereo_roll

        # Apply global shift
        if global_shift != 0:
            frame = np.roll(frame, global_shift, axis=1)

        # Split into left and right
        left = frame[:, :half_w].copy()
        right = frame[:, half_w:].copy()

        # Check if we need to recompute remap tables (cache them)
        left_key = (h, half_w, round(left_yaw, 2), round(left_pitch, 2), round(left_roll, 2))
        right_key = (h, half_w, round(right_yaw, 2), round(right_pitch, 2), round(right_roll, 2))

        if not hasattr(self, '_preview_remap_cache'):
            self._preview_remap_cache = {}

        def get_cached_remap_tables(key, yaw_deg, pitch_deg, roll_deg):
            if key in self._preview_remap_cache:
                return self._preview_remap_cache[key]

            if abs(yaw_deg) < 0.01 and abs(pitch_deg) < 0.01 and abs(roll_deg) < 0.01:
                return None

            h, w = key[0], key[1]
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            roll_rad = math.radians(roll_deg)

            cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
            cos_pitch, sin_pitch = math.cos(pitch_rad), math.sin(pitch_rad)
            cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)

            u = np.linspace(0, 1, w, dtype=np.float32)
            v = np.linspace(0, 1, h, dtype=np.float32)
            u_grid, v_grid = np.meshgrid(u, v)

            lon = (u_grid - 0.5) * math.pi
            lat = (0.5 - v_grid) * math.pi

            cos_lat = np.cos(lat)
            x = cos_lat * np.sin(lon)
            y = np.sin(lat)
            z = cos_lat * np.cos(lon)

            x1 = cos_roll * x - sin_roll * y
            y1 = sin_roll * x + cos_roll * y
            z1 = z

            y2 = cos_pitch * y1 - sin_pitch * z1
            z2 = sin_pitch * y1 + cos_pitch * z1
            x2 = x1

            x3 = cos_yaw * x2 + sin_yaw * z2
            z3 = -sin_yaw * x2 + cos_yaw * z2
            y3 = y2

            lat_new = np.arcsin(np.clip(y3, -1, 1))
            lon_new = np.arctan2(x3, z3)

            u_new = (lon_new / math.pi) + 0.5
            v_new = 0.5 - (lat_new / math.pi)

            map_x = np.clip((u_new * w).astype(np.float32), 0, w - 1)
            map_y = np.clip((v_new * h).astype(np.float32), 0, h - 1)

            result = (map_x, map_y)

            # Limit cache size
            if len(self._preview_remap_cache) > 20:
                self._preview_remap_cache.clear()
            self._preview_remap_cache[key] = result
            return result

        # Apply transformations to each eye
        remap_left = get_cached_remap_tables(left_key, left_yaw, left_pitch, left_roll)
        if remap_left is not None:
            left = cv2.remap(left, remap_left[0], remap_left[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        remap_right = get_cached_remap_tables(right_key, right_yaw, right_pitch, right_roll)
        if remap_right is not None:
            right = cv2.remap(right, remap_right[0], remap_right[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Combine back
        result = np.hstack([left, right])

        # Apply color adjustments (Lift/Gamma/Gain) using LUT for speed
        lift = self.lift_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0
        gain = self.gain_slider.value() / 100.0

        if abs(lift) > 0.01 or abs(gamma - 1.0) > 0.01 or abs(gain - 1.0) > 0.01:
            # Build 1D LUT for color adjustments (much faster than per-pixel math)
            color_key = (round(lift, 3), round(gamma, 3), round(gain, 3))
            if not hasattr(self, '_color_lut_cache') or not hasattr(self, '_color_lut_cache_key') or self._color_lut_cache_key != color_key:
                lut_1d = np.arange(256, dtype=np.float32) / 255.0
                if abs(lift) > 0.01:
                    lut_1d = lut_1d + lift * (1.0 - lut_1d)
                if abs(gain - 1.0) > 0.01:
                    lut_1d = lut_1d * gain
                lut_1d = np.clip(lut_1d, 0.0, 1.0)
                if abs(gamma - 1.0) > 0.01:
                    lut_1d = np.power(lut_1d, 1.0 / gamma)
                self._color_lut_cache = (lut_1d * 255).astype(np.uint8)
                self._color_lut_cache_key = color_key

            # Apply 1D LUT using cv2.LUT (very fast)
            result = cv2.LUT(result, self._color_lut_cache)

        # Apply 3D LUT
        lut_path = self.lut_path_edit.text()
        lut_intensity = self.lut_intensity_slider.value() / 100.0
        if lut_path and Path(lut_path).exists() and lut_intensity > 0.01:
            try:
                # Cache the loaded LUT
                if not hasattr(self, '_cached_lut_path') or self._cached_lut_path != lut_path:
                    self._cached_lut_3d = load_cube_lut(lut_path)
                    self._cached_lut_path = lut_path

                lut_3d = self._cached_lut_3d
                lut_size = lut_3d.shape[0]

                # Downscale for preview speed if image is large
                preview_scale = 1
                if h > 1000:
                    preview_scale = 2
                    result_small = cv2.resize(result, (full_w // preview_scale, h // preview_scale), interpolation=cv2.INTER_AREA)
                else:
                    result_small = result

                img = result_small.astype(np.float32) / 255.0

                b_idx = img[:, :, 0] * (lut_size - 1)
                g_idx = img[:, :, 1] * (lut_size - 1)
                r_idx = img[:, :, 2] * (lut_size - 1)

                b_idx = np.clip(b_idx, 0, lut_size - 1.001)
                g_idx = np.clip(g_idx, 0, lut_size - 1.001)
                r_idx = np.clip(r_idx, 0, lut_size - 1.001)

                b0, b1 = b_idx.astype(np.int32), np.minimum(b_idx.astype(np.int32) + 1, lut_size - 1)
                g0, g1 = g_idx.astype(np.int32), np.minimum(g_idx.astype(np.int32) + 1, lut_size - 1)
                r0, r1 = r_idx.astype(np.int32), np.minimum(r_idx.astype(np.int32) + 1, lut_size - 1)

                fb = (b_idx - b0)[:, :, np.newaxis]
                fg = (g_idx - g0)[:, :, np.newaxis]
                fr = (r_idx - r0)[:, :, np.newaxis]

                c000 = lut_3d[b0, g0, r0]
                c001 = lut_3d[b0, g0, r1]
                c010 = lut_3d[b0, g1, r0]
                c011 = lut_3d[b0, g1, r1]
                c100 = lut_3d[b1, g0, r0]
                c101 = lut_3d[b1, g0, r1]
                c110 = lut_3d[b1, g1, r0]
                c111 = lut_3d[b1, g1, r1]

                c00 = c000 * (1 - fr) + c001 * fr
                c01 = c010 * (1 - fr) + c011 * fr
                c10 = c100 * (1 - fr) + c101 * fr
                c11 = c110 * (1 - fr) + c111 * fr
                c0 = c00 * (1 - fg) + c01 * fg
                c1 = c10 * (1 - fg) + c11 * fg
                lut_result = c0 * (1 - fb) + c1 * fb

                if lut_intensity < 1.0:
                    img_rgb = img[:, :, ::-1]
                    lut_result = img_rgb + (lut_result - img_rgb) * lut_intensity

                result_small = np.clip(lut_result[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)

                # Upscale back if we downscaled
                if preview_scale > 1:
                    result = cv2.resize(result_small, (full_w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    result = result_small

            except Exception as e:
                self.status_bar.showMessage(f"LUT error: {e}")

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
        
        if mode == PreviewMode.SIDE_BY_SIDE: preview = frame
        elif mode == PreviewMode.ANAGLYPH:
            preview = np.zeros_like(left)
            preview[:,:,0] = left[:,:,0]
            preview[:,:,1] = right[:,:,1]
            preview[:,:,2] = right[:,:,2]
        elif mode == PreviewMode.OVERLAY_50: preview = ((left.astype(float) * 0.5 + right.astype(float) * 0.5)).astype(np.uint8)
        elif mode == PreviewMode.SINGLE_EYE: preview = left if self.current_eye == "left" else right
        elif mode == PreviewMode.DIFFERENCE: preview = np.clip(np.abs(left.astype(float) - right.astype(float)) * 3, 0, 255).astype(np.uint8)
        elif mode == PreviewMode.CHECKERBOARD:
            preview = np.zeros_like(left)
            bs = 64
            for y in range(0, h, bs):
                for x in range(0, hw, bs):
                    ye, xe = min(y+bs, h), min(x+bs, hw)
                    if ((y//bs) + (x//bs)) % 2 == 0: preview[y:ye, x:xe] = left[y:ye, x:xe]
                    else: preview[y:ye, x:xe] = right[y:ye, x:xe]
        else: preview = frame
        
        ph, pw = preview.shape[:2]
        qimg = QImage(preview.data.tobytes(), pw, ph, 3*pw, QImage.Format.Format_RGB888)
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
            use_hardware_accel=self.hw_accel_checkbox.isChecked(),
            lut_path=lut_path,
            lut_intensity=self.lut_intensity_slider.value() / 100.0,
            lift=self.lift_slider.value() / 100.0,
            gamma=self.gamma_slider.value() / 100.0,
            gain=self.gain_slider.value() / 100.0,
            h265_bit_depth=10 if self.h265_bit_depth_combo.currentIndex() == 1 else 8,
            inject_vr180_metadata=self.vr180_metadata_checkbox.isChecked(),
            vision_pro_mode=["standard", "hvc1", "mvhevc"][self.vision_pro_combo.currentIndex()],
            trim_start=self.trim_start,
            trim_end=trim_end,
            gopro_roll_data=self.config.gopro_roll_data,
            roll_stabilized=self.roll_stabilized_checkbox.isChecked())
        
        if not config.input_path or not config.input_path.exists():
            QMessageBox.warning(self, "Error", "Select input file"); return
        if config.output_path.exists():
            if QMessageBox.question(self, "Overwrite?", f"Overwrite {config.output_path}?") != QMessageBox.StandardButton.Yes: return
        
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.ffmpeg_output.clear()
        self.processor = VideoProcessor(config)
        self.processor.output_line.connect(self._append_ffmpeg_output)
        self.processor.status.connect(self.status_bar.showMessage)
        self.processor.finished_signal.connect(self._on_finished)
        self.processor.start()
    
    def _cancel_processing(self):
        if hasattr(self, 'processor'): self.processor.cancel()

    def _append_ffmpeg_output(self, line):
        """Append a line to the FFmpeg output text box"""
        self.ffmpeg_output.append(line)
        # Auto-scroll to bottom
        scrollbar = self.ffmpeg_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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

        # Load hardware acceleration setting
        self.hw_accel_checkbox.setChecked(self.settings.value("hardware_accel", True, type=bool))

        # Load color adjustment settings (ASC CDL: Lift, Gamma, Gain)
        self.lift_slider.setValue(self.settings.value("lift", 0, type=int))
        self.gamma_slider.setValue(self.settings.value("gamma", 100, type=int))
        self.gain_slider.setValue(self.settings.value("gain", 100, type=int))

        # Load VR180 metadata setting
        self.vr180_metadata_checkbox.setChecked(self.settings.value("vr180_metadata", False, type=bool))

        # Load Vision Pro mode setting
        self.vision_pro_combo.setCurrentIndex(self.settings.value("vision_pro_mode", 0, type=int))

        # Load H.265 bit depth setting
        self.h265_bit_depth_combo.setCurrentIndex(self.settings.value("h265_bit_depth", 0, type=int))

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

        # Save hardware acceleration setting
        self.settings.setValue("hardware_accel", self.hw_accel_checkbox.isChecked())

        # Save color adjustment settings (ASC CDL: Lift, Gamma, Gain)
        self.settings.setValue("lift", self.lift_slider.value())
        self.settings.setValue("gamma", self.gamma_slider.value())
        self.settings.setValue("gain", self.gain_slider.value())

        # Save VR180 metadata setting
        self.settings.setValue("vr180_metadata", self.vr180_metadata_checkbox.isChecked())

        # Save Vision Pro mode setting
        self.settings.setValue("vision_pro_mode", self.vision_pro_combo.currentIndex())

        # Save H.265 bit depth setting
        self.settings.setValue("h265_bit_depth", self.h265_bit_depth_combo.currentIndex())

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
                    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.mts', '.m2ts', '.osv')):
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
                    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.mts', '.m2ts', '.osv')):
                        self.config.input_path = Path(file_path)
                        self.input_path_edit.setText(file_path)
                        output = Path(file_path).parent / f"{Path(file_path).stem}_adjusted{Path(file_path).suffix}"
                        self.output_path_edit.setText(str(output))
                        self.config.output_path = output
                        self._load_video_info()
                        self.process_btn.setEnabled(True)
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


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VR180ProcessorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

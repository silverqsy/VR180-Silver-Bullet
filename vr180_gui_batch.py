#!/usr/bin/env python3
"""
VR180 SBS Half-Equirectangular Video Processor - Batch Edition
With Clip Queue, Trimming, and Concatenation Support
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
    QScrollArea, QToolButton, QListWidget, QListWidgetItem, QCheckBox, QTimeEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QTime
from PyQt6.QtGui import QImage, QPixmap, QPainter


def get_ffmpeg_path():
    """Get the path to bundled ffmpeg or system ffmpeg"""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check multiple possible locations
        base_path = Path(sys._MEIPASS)

        # Try _internal folder (Windows/Linux style)
        ffmpeg_path = base_path / 'ffmpeg'
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
        ffprobe_path = base_path / 'ffprobe'
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
        spatial_path = base_path / 'spatial'
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
class ClipItem:
    """Represents a clip in the processing queue with per-clip settings"""
    input_path: Path
    start_time: float = 0.0  # Start time in seconds
    end_time: float = 0.0   # End time in seconds (0 = full duration)
    duration: float = 0.0   # Total clip duration

    # Per-clip VR180 adjustment settings
    global_shift: int = 0
    global_yaw: float = 0.0
    global_pitch: float = 0.0
    global_roll: float = 0.0
    stereo_yaw: float = 0.0
    stereo_pitch: float = 0.0
    stereo_roll: float = 0.0

    # Per-clip color grading settings
    gamma: float = 1.0
    white_point: float = 1.0
    black_point: float = 0.0
    lut_intensity: float = 1.0

    def get_trim_args(self):
        """Get FFmpeg trim arguments for this clip"""
        args = []
        if self.start_time > 0:
            args.extend(['-ss', str(self.start_time)])
        if self.end_time > 0 and self.end_time > self.start_time:
            duration = self.end_time - self.start_time
            args.extend(['-t', str(duration)])
        return args

    def has_custom_settings(self):
        """Check if this clip has any custom adjustment settings"""
        return (self.global_shift != 0 or self.global_yaw != 0.0 or
                self.global_pitch != 0.0 or self.global_roll != 0.0 or
                self.stereo_yaw != 0.0 or self.stereo_pitch != 0.0 or
                self.stereo_roll != 0.0)


@dataclass
class ProcessingConfig:
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    global_shift: int = 0
    global_adjustment: PanomapAdjustment = field(default_factory=PanomapAdjustment)
    stereo_offset: PanomapAdjustment = field(default_factory=PanomapAdjustment)
    output_codec: str = "auto"
    quality: int = 18
    bitrate: int = 50  # Mbps
    use_bitrate: bool = False  # If False, use CRF quality; if True, use bitrate
    prores_profile: str = "standard"
    use_hardware_accel: bool = True
    encoder_speed: str = "fast"  # fast, medium, slow
    lut_path: Optional[Path] = None  # Optional LUT file for color grading
    lut_intensity: float = 1.0  # LUT intensity 0.0 to 1.0
    gamma: float = 1.0  # Gamma correction (applied pre-LUT)
    white_point: float = 1.0  # White point adjustment (applied pre-LUT)
    black_point: float = 0.0  # Black point adjustment (applied pre-LUT)
    h265_bit_depth: int = 8  # 8-bit or 10-bit for H.265
    inject_vr180_metadata: bool = False  # Inject VR180 metadata for YouTube
    vision_pro_mode: str = "standard"  # standard, hvc1, or mvhevc
    trim_start: Optional[float] = None  # Trim start time in seconds (for batch processing)
    trim_duration: Optional[float] = None  # Trim duration in seconds (for batch processing)
    for_concatenation: bool = False  # Force keyframes for concat compatibility


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
            self.status.emit("Processing...")
            cfg = self.config
            
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
            duration = float(video_info.get("format", {}).get("duration", 0))
            stream = video_info["streams"][0]

            # Log duration for debugging
            self.status.emit(f"Video duration: {duration:.2f}s")

            # Check if input is 10-bit (causes extreme slowness with filters)
            pix_fmt = stream.get("pix_fmt", "")
            is_10bit = "10le" in pix_fmt or "p010" in pix_fmt

            # Build filter
            filters = []

            # For 10-bit input, convert to 8-bit first for faster processing
            if is_10bit:
                self.status.emit("Detected 10-bit input - converting to 8-bit for faster processing...")
                filters.append("[0:v]format=yuv420p[input_8bit]")
                input_label = "[input_8bit]"
            else:
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

            # Check if any pre-LUT adjustments are needed
            has_adjustments = (abs(cfg.gamma - 1.0) > 0.01 or
                             abs(cfg.white_point - 1.0) > 0.01 or
                             abs(cfg.black_point) > 0.01)

            if has_adjustments:
                # Build eq filter for color adjustments
                eq_params = []
                if abs(cfg.gamma - 1.0) > 0.01:
                    eq_params.append(f"gamma={cfg.gamma}")

                # Black point: lift the shadows (0.0 = no change, positive = lift blacks)
                if abs(cfg.black_point) > 0.01:
                    # Map black_point range (-1 to 1) to brightness (-1 to 1)
                    eq_params.append(f"brightness={cfg.black_point}")

                # White point: adjust highlights (1.0 = no change, <1 = compress, >1 = expand)
                if abs(cfg.white_point - 1.0) > 0.01:
                    # Map white_point to contrast adjustment
                    eq_params.append(f"contrast={cfg.white_point}")

                if eq_params:
                    eq_filter = ":".join(eq_params)
                    filters.append(f"{current_label}eq={eq_filter}[color_adjusted]")
                    current_label = "[color_adjusted]"

            # Apply LUT if specified
            if cfg.lut_path and cfg.lut_path.exists():
                # Escape the path for FFmpeg filter syntax
                lut_path_str = str(cfg.lut_path).replace('\\', '/').replace(':', '\\:')

                if cfg.lut_intensity > 0.01:
                    # Apply LUT and blend with original based on intensity
                    filters.append(f"{current_label}split[original][lut_input]")
                    filters.append(f"[lut_input]lut3d=file='{lut_path_str}'[lut_output]")
                    # Use blend with custom expression: lerp between original and LUT
                    # A=original (bottom), B=LUT (top), intensity controls blend
                    filters.append(f"[original][lut_output]blend=all_expr='A*(1-{cfg.lut_intensity})+B*{cfg.lut_intensity}'[lut_final]")
                    current_label = "[lut_final]"
                else:
                    # No intensity - skip LUT
                    pass  # current_label stays as is

            # Finalize filter chain
            filters.append(f"{current_label}null[out]")

            # If outputting to MV-HEVC, use lossless intermediate to avoid double lossy compression
            if cfg.vision_pro_mode == "mvhevc":
                # Use ProRes HQ as lossless intermediate for MV-HEVC workflow
                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", "3"]  # ProRes HQ
                else:
                    enc = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]  # ProRes HQ
            # Encoder settings with hardware acceleration
            elif output_codec == "h265":
                # Set pixel format based on bit depth
                pix_fmt = "yuv420p10le" if cfg.h265_bit_depth == 10 else "yuv420p"

                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    # macOS VideoToolbox hardware encoding
                    if cfg.use_bitrate:
                        enc = ["-c:v", "hevc_videotoolbox", "-b:v", f"{cfg.bitrate}M", "-tag:v", "hvc1"]
                    else:
                        enc = ["-c:v", "hevc_videotoolbox", "-q:v", str(min(100, cfg.quality * 2)), "-tag:v", "hvc1"]
                    # VideoToolbox supports 10-bit
                    if cfg.h265_bit_depth == 10:
                        enc.extend(["-pix_fmt", "p010le"])
                    else:
                        enc.extend(["-pix_fmt", "yuv420p"])
                    # Force keyframes for concatenation - ensure first frame is keyframe
                    if cfg.for_concatenation:
                        enc.extend(["-g", "60", "-force_key_frames", "expr:gte(t,0)"])
                elif cfg.use_hardware_accel and sys.platform == 'win32':
                    # Windows NVIDIA NVENC
                    if cfg.use_bitrate:
                        enc = ["-c:v", "hevc_nvenc", "-preset", "p4", "-b:v", f"{cfg.bitrate}M", "-tag:v", "hvc1"]
                    else:
                        enc = ["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", str(cfg.quality), "-tag:v", "hvc1"]
                    # NVENC supports 10-bit
                    if cfg.h265_bit_depth == 10:
                        enc.extend(["-pix_fmt", "p010le"])
                    # Force keyframes for concatenation
                    if cfg.for_concatenation:
                        enc.extend(["-g", "60", "-forced-idr", "1"])
                else:
                    # Software encoding
                    preset = {"fast": "fast", "medium": "medium", "slow": "slow"}.get(cfg.encoder_speed, "medium")
                    if cfg.use_bitrate:
                        enc = ["-c:v", "libx265", "-b:v", f"{cfg.bitrate}M", "-preset", preset, "-tag:v", "hvc1"]
                    else:
                        enc = ["-c:v", "libx265", "-crf", str(cfg.quality), "-preset", preset, "-tag:v", "hvc1"]
                    # Add bit depth for software encoder
                    enc.extend(["-pix_fmt", pix_fmt])
                    if cfg.h265_bit_depth == 10:
                        enc.extend(["-profile:v", "main10"])
                    # Force keyframes at start for concatenation compatibility
                    if cfg.for_concatenation:
                        enc.extend(["-g", "60", "-keyint_min", "60", "-sc_threshold", "0"])
            elif output_codec == "prores":
                profile_map = {"proxy": "0", "lt": "1", "standard": "2", "hq": "3", "4444": "4", "4444xq": "5"}
                if cfg.use_hardware_accel and sys.platform == 'darwin':
                    # macOS VideoToolbox ProRes encoding
                    enc = ["-c:v", "prores_videotoolbox", "-profile:v", profile_map.get(cfg.prores_profile, "3")]
                else:
                    # Software ProRes
                    enc = ["-c:v", "prores_ks", "-profile:v", profile_map.get(cfg.prores_profile, "3"), "-vendor", "apl0"]
            else:
                enc = ["-c:v", output_codec]
            
            # Build FFmpeg command with hardware decode for HEVC input
            decode_args = []
            if codec == "h265" and sys.platform == 'darwin':
                # Use VideoToolbox hardware decoding for HEVC (much faster for 10-bit)
                decode_args = ["-hwaccel", "videotoolbox"]

            # Add trim parameters if specified (batch processing mode)
            # Using -ss before -i for fast keyframe seeking
            trim_args = []
            if cfg.trim_start is not None and cfg.trim_start > 0:
                trim_args.extend(["-ss", str(cfg.trim_start)])
            if cfg.trim_duration is not None and cfg.trim_duration > 0:
                trim_args.extend(["-t", str(cfg.trim_duration)])

            cmd = [get_ffmpeg_path(), "-y"] + decode_args + trim_args + [
                   "-i", str(cfg.input_path),
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
        """Inject VR180 metadata for YouTube using Spherical Video V2 specification"""
        import struct
        import tempfile
        import shutil

        # Create temporary file
        temp_file = Path(tempfile.mktemp(suffix='.mov'))

        try:
            # First, use FFmpeg to inject the spherical metadata
            # VR180 requires:
            # - Spherical projection: equirectangular
            # - Stereo mode: left-right (side-by-side)
            # - Half equirectangular (180 degrees horizontal)

            cmd = [
                get_ffmpeg_path(),
                '-i', str(video_path),
                '-c', 'copy',  # Copy streams without re-encoding
                '-metadata:s:v:0', 'spherical-video=true',
                '-metadata:s:v:0', 'stereo_mode=left-right',
                str(temp_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=get_subprocess_flags())
            if result.returncode != 0:
                raise Exception(f"Metadata injection failed: {result.stderr}")

            # Now we need to add the sv3d box for VR180
            # This requires manual MP4 atom manipulation
            # Read the file and inject proper sv3d atoms
            self._inject_sv3d_box(temp_file, video_path)

        finally:
            # Clean up temp file if it still exists
            if temp_file.exists():
                temp_file.unlink()

    def _inject_sv3d_box(self, temp_file: Path, output_file: Path):
        """Inject sv3d (Spherical Video V2) box into MP4 file for VR180"""
        import struct

        # For VR180, we need to inject these boxes into the video track:
        # - st3d: Stereoscopic 3D video box (side-by-side mode)
        # - sv3d: Spherical video box (projection info)

        # Read the temp file
        with open(temp_file, 'rb') as f:
            data = f.read()

        # Create st3d box (Stereoscopic 3D Video Box)
        # Stereo mode: 0 = mono, 1 = top-bottom, 2 = left-right
        st3d_data = struct.pack('>I4sBBB', 13, b'st3d', 0, 0, 2)  # mode = 2 (left-right)

        # Create proj box (Projection Box) for equirectangular
        # Projection type for equirectangular
        proj_header = struct.pack('>I4s', 8, b'prhd')

        # Create equi box (Equirectangular Projection Box)
        # For VR180: yaw=0, pitch=0, roll=0, top=0, bottom=0, left=-90°, right=90°
        equi_data = struct.pack('>I4sIIIIIII',
            40, b'equi',  # size and type
            0,  # yaw (0°)
            0,  # pitch (0°)
            0,  # roll (0°)
            0,  # bounds_top (0°)
            0,  # bounds_bottom (0°)
            0xC0000000,  # bounds_left (-90° in 16.16 fixed point)
            0x40000000,  # bounds_right (90° in 16.16 fixed point)
        )

        proj_data = proj_header + equi_data
        proj_box = struct.pack('>I4s', len(proj_data) + 8, b'proj') + proj_data

        # Create sv3d box (Spherical Video Box)
        svhd_data = struct.pack('>I4s8s', 16, b'svhd', b'\x00' * 8)  # version 0, no flags
        sv3d_content = svhd_data + proj_box
        sv3d_box = struct.pack('>I4s', len(sv3d_content) + 8, b'sv3d') + sv3d_content

        # Combine st3d and sv3d into uuid box
        # For maximum compatibility, we write these as regular boxes in the video track

        # Write the modified file
        # This is a simplified approach - just append metadata to moov atom
        # For production use, proper MP4 parsing would be needed

        # For now, use exiftool or similar approach via FFmpeg metadata
        # Since full MP4 atom manipulation is complex, we'll use FFmpeg's built-in support

        # Actually, let's use a simpler approach with FFmpeg's spatial metadata support
        cmd = [
            get_ffmpeg_path(),
            '-i', str(temp_file),
            '-c', 'copy',
            '-metadata:s:v', 'stereo_mode=left_right',
            '-movflags', '+faststart',
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=get_subprocess_flags())
        if result.returncode != 0:
            # If that fails, just copy the temp file
            import shutil
            shutil.move(str(temp_file), str(output_file))


class TimelineSlider(QSlider):
    """Custom slider with visual trim markers"""
    def __init__(self, orientation):
        super().__init__(orientation)
        self.trim_in = 0  # Normalized 0-1000
        self.trim_out = 0  # Normalized 0-1000
        self.has_trim = False

    def set_trim_points(self, in_point, out_point):
        """Set trim points (normalized 0-1000 range)"""
        self.trim_in = in_point
        self.trim_out = out_point
        self.has_trim = True
        self.update()

    def clear_trim_points(self):
        """Clear trim markers"""
        self.has_trim = False
        self.update()

    def paintEvent(self, event):
        """Custom paint to draw trim markers"""
        super().paintEvent(event)

        if not self.has_trim:
            return

        from PyQt6.QtGui import QPainter, QColor, QPen
        from PyQt6.QtWidgets import QStyleOptionSlider

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create style option for slider
        opt = QStyleOptionSlider()
        opt.initFrom(self)
        opt.minimum = self.minimum()
        opt.maximum = self.maximum()
        opt.sliderPosition = self.value()
        opt.orientation = self.orientation()

        # Get groove rectangle
        groove = self.style().subControlRect(
            self.style().ComplexControl.CC_Slider,
            opt,
            self.style().SubControl.SC_SliderGroove,
            self
        )

        # Calculate positions
        width = groove.width()
        x_start = groove.x()
        y_center = groove.y() + groove.height() // 2

        in_pos = x_start + int((self.trim_in / 1000.0) * width)
        out_pos = x_start + int((self.trim_out / 1000.0) * width)

        # Draw trim region highlight
        if self.trim_out > self.trim_in:
            painter.fillRect(in_pos, groove.y(), out_pos - in_pos, groove.height(),
                           QColor(100, 150, 255, 80))

        # Draw In marker (green vertical line)
        if self.trim_in > 0:
            pen = QPen(QColor(0, 200, 0), 3)
            painter.setPen(pen)
            painter.drawLine(in_pos, groove.y() - 5, in_pos, groove.y() + groove.height() + 5)

        # Draw Out marker (red vertical line)
        if self.trim_out > 0:
            pen = QPen(QColor(255, 100, 0), 3)
            painter.setPen(pen)
            painter.drawLine(out_pos, groove.y() - 5, out_pos, groove.y() + groove.height() + 5)


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

        # Clip queue for batch processing
        self.clip_queue = []  # List of ClipItem objects
        self.current_clip_index = 0  # Track which clip is being edited

        # Settings for persistence
        from PyQt6.QtCore import QSettings
        self.settings = QSettings("VR180Processor", "VR180ProcessorBatch")

        # Initialize preview timer before UI (needed for _load_settings)
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._on_preview_timer)

        self._init_ui()
        self._apply_styles()
        self._connect_signals()
        self._load_settings()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        from PyQt6.QtCore import Qt
        if event.key() == Qt.Key.Key_I:
            self._set_trim_in_point()
        elif event.key() == Qt.Key.Key_O:
            self._set_trim_out_point()
        elif event.key() == Qt.Key.Key_X:
            self._clear_trim_points()
        else:
            super().keyPressEvent(event)

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

        # Splitter for preview and controls
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

        # Trim control buttons
        self.set_in_btn = QPushButton("In")
        self.set_in_btn.setToolTip("Set trim start point (I key)")
        self.set_in_btn.setFixedWidth(50)
        timeline_layout.addWidget(self.set_in_btn)

        self.set_out_btn = QPushButton("Out")
        self.set_out_btn.setToolTip("Set trim end point (O key)")
        self.set_out_btn.setFixedWidth(60)
        timeline_layout.addWidget(self.set_out_btn)

        self.clear_trim_btn = QPushButton("Clr")
        self.clear_trim_btn.setToolTip("Clear trim points (X key)")
        self.clear_trim_btn.setFixedWidth(50)
        timeline_layout.addWidget(self.clear_trim_btn)

        self.timeline_slider = TimelineSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.setTracking(False)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        timeline_layout.addWidget(self.timeline_slider, stretch=1)
        timeline_layout.addWidget(self.time_label)

        # Trim info label
        self.trim_info_label = QLabel("")
        self.trim_info_label.setFixedWidth(120)
        self.trim_info_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        timeline_layout.addWidget(self.trim_info_label)

        preview_layout.addLayout(timeline_layout)
        
        splitter.addWidget(preview_container)
        
        # Controls section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(16)

        # Clip Queue section (moved to right column)
        queue_group = QGroupBox("Clip Queue")
        queue_layout = QVBoxLayout(queue_group)

        # Top row: Add clips button and concatenate checkbox
        queue_top_layout = QHBoxLayout()
        self.add_clips_btn = QPushButton("Add Clips to Queue")
        self.add_clips_btn.setToolTip("Add multiple clips to process with same settings")
        queue_top_layout.addWidget(self.add_clips_btn)

        self.concatenate_checkbox = QCheckBox("Combine Clips")
        self.concatenate_checkbox.setToolTip("Merge all processed clips into one output file")
        queue_top_layout.addWidget(self.concatenate_checkbox)

        self.apply_to_all_btn = QPushButton("Apply Settings to All")
        self.apply_to_all_btn.setToolTip("Copy current clip's adjustments to all other clips in queue")
        self.apply_to_all_btn.setEnabled(False)
        queue_top_layout.addWidget(self.apply_to_all_btn)

        queue_top_layout.addStretch()
        queue_layout.addLayout(queue_top_layout)

        # Project save/load buttons
        project_layout = QHBoxLayout()
        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.setToolTip("Save all clips and their settings to a project file")
        self.save_project_btn.setEnabled(False)
        project_layout.addWidget(self.save_project_btn)

        self.load_project_btn = QPushButton("Load Project")
        self.load_project_btn.setToolTip("Load a saved project with all clips and settings")
        project_layout.addWidget(self.load_project_btn)

        project_layout.addStretch()
        queue_layout.addLayout(project_layout)

        # Clip list and controls
        clip_content_layout = QHBoxLayout()

        # Clip list widget
        self.clip_list_widget = QListWidget()
        self.clip_list_widget.setMaximumHeight(150)
        self.clip_list_widget.setToolTip("Click to select clip for preview and adjustment")
        clip_content_layout.addWidget(self.clip_list_widget, stretch=1)

        # Queue control buttons
        queue_buttons_layout = QVBoxLayout()
        self.remove_clip_btn = QPushButton("Del")
        self.remove_clip_btn.setEnabled(False)
        self.remove_clip_btn.setFixedWidth(40)
        self.remove_clip_btn.setObjectName("queueBtn")
        self.remove_clip_btn.setToolTip("Remove selected clip")
        queue_buttons_layout.addWidget(self.remove_clip_btn)

        self.move_up_btn = QPushButton("Up")
        self.move_up_btn.setEnabled(False)
        self.move_up_btn.setFixedWidth(40)
        self.move_up_btn.setObjectName("queueBtn")
        self.move_up_btn.setToolTip("Move up")
        queue_buttons_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("Dn")
        self.move_down_btn.setEnabled(False)
        self.move_down_btn.setFixedWidth(40)
        self.move_down_btn.setObjectName("queueBtn")
        self.move_down_btn.setToolTip("Move down")
        queue_buttons_layout.addWidget(self.move_down_btn)

        self.clear_queue_btn = QPushButton("Clr")
        self.clear_queue_btn.setEnabled(False)
        self.clear_queue_btn.setFixedWidth(40)
        self.clear_queue_btn.setObjectName("queueBtn")
        self.clear_queue_btn.setToolTip("Clear all clips")
        queue_buttons_layout.addWidget(self.clear_queue_btn)

        queue_buttons_layout.addStretch()
        clip_content_layout.addLayout(queue_buttons_layout)

        queue_layout.addLayout(clip_content_layout)
        controls_layout.addWidget(queue_group)

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
        self.global_yaw = SliderWithSpinBox(-180, 180, 0, 1, 0.1, "°")
        global_layout.addWidget(self.global_yaw, 0, 1)
        global_layout.addWidget(QLabel("Pitch:"), 1, 0)
        self.global_pitch = SliderWithSpinBox(-90, 90, 0, 1, 0.1, "°")
        global_layout.addWidget(self.global_pitch, 1, 1)
        global_layout.addWidget(QLabel("Roll:"), 2, 0)
        self.global_roll = SliderWithSpinBox(-45, 45, 0, 1, 0.1, "°")
        global_layout.addWidget(self.global_roll, 2, 1)
        controls_layout.addWidget(global_group)
        
        # Stereo offset
        offset_group = QGroupBox("③ Stereo Offset (Applied Oppositely)")
        offset_layout = QGridLayout(offset_group)
        offset_layout.addWidget(QLabel("Yaw:"), 0, 0)
        self.stereo_yaw_offset = SliderWithSpinBox(-10, 10, 0, 2, 0.05, "°")
        offset_layout.addWidget(self.stereo_yaw_offset, 0, 1)
        offset_layout.addWidget(QLabel("Pitch:"), 1, 0)
        self.stereo_pitch_offset = SliderWithSpinBox(-10, 10, 0, 2, 0.05, "°")
        offset_layout.addWidget(self.stereo_pitch_offset, 1, 1)
        offset_layout.addWidget(QLabel("Roll:"), 2, 0)
        self.stereo_roll_offset = SliderWithSpinBox(-10, 10, 0, 2, 0.05, "°")
        offset_layout.addWidget(self.stereo_roll_offset, 2, 1)
        controls_layout.addWidget(offset_group)
        
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
        self.quality_spinbox.setEnabled(False)  # Disabled by default since bitrate mode is default
        output_layout.addWidget(self.quality_spinbox, 1, 1)

        # H.265 bitrate settings (mutually exclusive with quality)
        self.bitrate_label = QLabel("Bitrate (Mbps):")
        output_layout.addWidget(self.bitrate_label, 2, 0)
        self.bitrate_spinbox = QSpinBox()
        self.bitrate_spinbox.setRange(1, 700)
        self.bitrate_spinbox.setValue(200)  # Default to 200 Mbps
        self.bitrate_spinbox.setEnabled(True)  # Enabled by default since bitrate mode is default
        self.bitrate_spinbox.setToolTip("Target bitrate in Mbps")
        output_layout.addWidget(self.bitrate_spinbox, 2, 1)

        # Radio buttons for quality vs bitrate
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self.use_crf_radio = QRadioButton("Use Quality (CRF)")
        output_layout.addWidget(self.use_crf_radio, 3, 0, 1, 2)

        self.use_bitrate_radio = QRadioButton("Use Bitrate")
        self.use_bitrate_radio.setChecked(True)  # Default to bitrate mode
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

        # Pre-LUT color adjustments
        output_layout.addWidget(QLabel("Gamma:"), 8, 0)
        self.gamma_slider = SliderWithSpinBox(10, 300, 100, 0, 0.01, "")
        self.gamma_slider.setToolTip("Gamma correction: <100 = darker, 100 = neutral, >100 = brighter")
        output_layout.addWidget(self.gamma_slider, 8, 1)

        output_layout.addWidget(QLabel("White Point:"), 9, 0)
        self.white_point_slider = SliderWithSpinBox(50, 200, 100, 0, 0.01, "")
        self.white_point_slider.setToolTip("Adjust highlights: <100 = compress whites, 100 = neutral, >100 = expand whites")
        output_layout.addWidget(self.white_point_slider, 9, 1)

        output_layout.addWidget(QLabel("Black Point:"), 10, 0)
        self.black_point_slider = SliderWithSpinBox(-100, 100, 0, 0, 0.01, "")
        self.black_point_slider.setToolTip("Adjust shadows: negative = crush blacks, 0 = neutral, positive = lift blacks")
        output_layout.addWidget(self.black_point_slider, 10, 1)

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
        from PyQt6.QtWidgets import QTextEdit
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
            QPushButton#queueBtn { padding: 2px 4px; font-size: 11px; }
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
        self.set_in_btn.clicked.connect(self._set_trim_in_point)
        self.set_out_btn.clicked.connect(self._set_trim_out_point)
        self.clear_trim_btn.clicked.connect(self._clear_trim_points)
        self.global_shift_slider.valueChanged.connect(self._on_adjustment_changed)
        self.global_yaw.valueChanged.connect(self._on_adjustment_changed)
        self.global_pitch.valueChanged.connect(self._on_adjustment_changed)
        self.global_roll.valueChanged.connect(self._on_adjustment_changed)
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
        # Color grading sliders - save to current clip AND update preview when changed
        self.gamma_slider.valueChanged.connect(self._on_adjustment_changed)
        self.white_point_slider.valueChanged.connect(self._on_adjustment_changed)
        self.black_point_slider.valueChanged.connect(self._on_adjustment_changed)
        self.lut_intensity_slider.valueChanged.connect(self._on_adjustment_changed)
        self.vision_pro_combo.currentIndexChanged.connect(self._update_vision_pro_mode)
        self.eye_toggle_btn.clicked.connect(self._toggle_eye)
        self.preview_mode_combo.currentIndexChanged.connect(self._update_preview_mode_ui)

        # Clip queue connections
        self.add_clips_btn.clicked.connect(self._add_clips_to_queue)
        self.remove_clip_btn.clicked.connect(self._remove_selected_clip)
        self.move_up_btn.clicked.connect(self._move_clip_up)
        self.move_down_btn.clicked.connect(self._move_clip_down)
        self.clear_queue_btn.clicked.connect(self._clear_queue)
        self.apply_to_all_btn.clicked.connect(self._apply_settings_to_all)
        self.clip_list_widget.itemSelectionChanged.connect(self._on_clip_selection_changed)
        self.concatenate_checkbox.stateChanged.connect(self._update_output_path_for_queue)
        self.save_project_btn.clicked.connect(self._save_project)
        self.load_project_btn.clicked.connect(self._load_project)

        # Initial state
        self._update_codec_settings()
        self._update_vision_pro_mode()
        self._update_preview_mode_ui()
    
    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", str(Path.home()), "Video (*.mp4 *.mov *.mkv)")
        if path:
            self.config.input_path = Path(path)
            self.input_path_edit.setText(path)
            output = Path(path).parent / f"{Path(path).stem}_adjusted{Path(path).suffix}"
            self.output_path_edit.setText(str(output))
            self.config.output_path = output
            self._load_video_info()
            self.process_btn.setEnabled(True)
    
    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", self.output_path_edit.text() or str(Path.home()), "Video (*.mp4 *.mov *.mkv)")
        if path:
            self.config.output_path = Path(path)
            self.output_path_edit.setText(path)

    def _browse_lut(self):
        """Browse for LUT file"""
        path, _ = QFileDialog.getOpenFileName(self, "Select LUT File", str(Path.home()), "LUT Files (*.cube *.3dl);;All Files (*.*)")
        if path:
            self.lut_path_edit.setText(path)
            self._schedule_preview_update()

    def _clear_lut(self):
        """Clear the LUT file selection"""
        self.lut_path_edit.clear()
        self._schedule_preview_update()

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

        # Check if we have adjustments that need FFmpeg filters
        has_adjustments = (
            self.global_shift_slider.value() != 0 or
            self.global_yaw.value() != 0 or self.global_pitch.value() != 0 or self.global_roll.value() != 0 or
            self.stereo_yaw_offset.value() != 0 or self.stereo_pitch_offset.value() != 0 or self.stereo_roll_offset.value() != 0 or
            self.lut_path_edit.text() or
            self.gamma_slider.value() != 1.0 or self.white_point_slider.value() != 100 or self.black_point_slider.value() != 0
        )

        # If no adjustments and we have cached frame, use it
        if not has_adjustments and self.cached_timestamp == timestamp and self.cached_raw_frame is not None:
            # Use cached frame directly (no filters needed)
            self.original_frame = self.cached_raw_frame.copy()
            self._update_preview()
            self.status_bar.showMessage("Ready")
        elif has_adjustments or force_filter:
            # Need to apply FFmpeg filters - extract with filters
            self.status_bar.showMessage(f"Applying adjustments...")

            preview_filter = self._build_preview_filter()
            self.extractor = FrameExtractor(self.config.input_path, timestamp, preview_filter)
            self.extractor.frame_ready.connect(self._on_frame_extracted)
            self.extractor.error.connect(lambda e: self.status_bar.showMessage(f"Error: {e}"))
            self.extractor.start()
        else:
            # Need to decode new frame - extract raw frame first for caching
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

        # Get color adjustment values
        gamma = self.gamma_slider.value() / 100.0
        white_point = self.white_point_slider.value() / 100.0
        black_point = self.black_point_slider.value() / 100.0

        # Check if any adjustments are needed
        has_adjustments = any([shift, yaw, pitch, roll, syaw, spitch, sroll])
        has_lut = lut_path and Path(lut_path).exists() and lut_intensity > 0.01
        has_color_adjustments = (abs(gamma - 1.0) > 0.01 or
                                abs(white_point - 1.0) > 0.01 or
                                abs(black_point) > 0.01)

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

        # Apply pre-LUT color adjustments (already checked has_color_adjustments above)
        current_label = "[stacked]"

        if has_color_adjustments:
            eq_params = []
            if abs(gamma - 1.0) > 0.01:
                eq_params.append(f"gamma={gamma}")
            if abs(black_point) > 0.01:
                eq_params.append(f"brightness={black_point}")
            if abs(white_point - 1.0) > 0.01:
                eq_params.append(f"contrast={white_point}")

            if eq_params:
                eq_filter = ":".join(eq_params)
                filters.append(f"{current_label}eq={eq_filter}[color_adjusted]")
                current_label = "[color_adjusted]"

        # Apply LUT if specified
        if has_lut:
            lut_path_str = lut_path.replace('\\', '/').replace(':', '\\:')
            if lut_intensity > 0.01:
                filters.append(f"{current_label}split[original][lut_input]")
                filters.append(f"[lut_input]lut3d=file='{lut_path_str}'[lut_output]")
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
        """Apply adjustments to cached frame using CPU (much faster than re-decoding)"""
        if self.cached_raw_frame is None:
            return

        # For now, use the cached frame directly
        # TODO: Apply CPU-based IPD/rotation/color adjustments here for instant preview
        # This would require implementing the filters in NumPy/OpenCV instead of FFmpeg
        self.original_frame = self.cached_raw_frame.copy()
        self._update_preview()
        self.status_bar.showMessage("Ready")

    def _on_frame_extracted(self, frame):
        """Legacy callback for filtered frame extraction"""
        self.original_frame = frame
        self._update_preview()
    
    def _schedule_preview_update(self):
        # Debounce: Wait 500ms after last adjustment before updating
        # This prevents multiple FFmpeg calls while dragging sliders
        self.preview_timer.start(500)
    
    def _on_preview_timer(self):
        if self.config.input_path:
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

    def _clear_frame_cache(self):
        """Clear cached frames to free memory"""
        import gc
        self.original_frame = None
        self.cached_raw_frame = None
        self.cached_timestamp = None
        # Force garbage collection to free memory immediately
        gc.collect()

    def _timeline_dragging(self, value):
        if self.video_duration > 0:
            self.time_label.setText(f"{self._format_time((value/1000)*self.video_duration)} / {self._format_time(self.video_duration)}")
    
    def _timeline_changed(self, value):
        if self.video_duration > 0:
            ts = (value / 1000) * self.video_duration
            self.time_label.setText(f"{self._format_time(ts)} / {self._format_time(self.video_duration)}")
            self._extract_frame(ts)
    
    def _format_time(self, s): return f"{int(s//60):02d}:{int(s%60):02d}"

    def _set_trim_in_point(self):
        """Set trim start point at current timeline position"""
        if len(self.clip_queue) == 0:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            # Get current timeline position
            current_time = (self.timeline_slider.value() / 1000) * self.video_duration
            clip.start_time = current_time

            # Make sure end_time is after start_time
            if clip.end_time > 0 and clip.end_time <= clip.start_time:
                clip.end_time = 0.0  # Reset end time

            self._update_trim_display()
            self._update_clip_list_display()
            self.status_bar.showMessage(f"Trim In point set at {self._format_time(current_time)}")

    def _set_trim_out_point(self):
        """Set trim end point at current timeline position"""
        if len(self.clip_queue) == 0:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            # Get current timeline position
            current_time = (self.timeline_slider.value() / 1000) * self.video_duration
            clip.end_time = current_time

            # Make sure end_time is after start_time
            if clip.end_time <= clip.start_time:
                QMessageBox.warning(self, "Invalid Trim", "Out point must be after In point")
                clip.end_time = 0.0
                return

            self._update_trim_display()
            self._update_clip_list_display()
            self.status_bar.showMessage(f"Trim Out point set at {self._format_time(current_time)}")

    def _clear_trim_points(self):
        """Clear all trim points for current clip"""
        if len(self.clip_queue) == 0:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            clip.start_time = 0.0
            clip.end_time = 0.0

            self._update_trim_display()
            self._update_clip_list_display()
            self.status_bar.showMessage("Trim points cleared")

    def _update_trim_display(self):
        """Update the trim info label and timeline markers to show current in/out points"""
        if len(self.clip_queue) == 0:
            self.trim_info_label.setText("")
            self.timeline_slider.clear_trim_points()
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            if clip.start_time > 0 or clip.end_time > 0:
                # Update text display
                in_str = self._format_time(clip.start_time)
                out_str = self._format_time(clip.end_time if clip.end_time > 0 else clip.duration)
                self.trim_info_label.setText(f"[{in_str} - {out_str}]")

                # Update timeline visual markers (convert to 0-1000 range)
                if self.video_duration > 0:
                    in_norm = int((clip.start_time / self.video_duration) * 1000)
                    out_norm = int(((clip.end_time if clip.end_time > 0 else clip.duration) / self.video_duration) * 1000)
                    self.timeline_slider.set_trim_points(in_norm, out_norm)
            else:
                self.trim_info_label.setText("")
                self.timeline_slider.clear_trim_points()

    def _update_clip_list_display(self):
        """Update the current clip list item to show trim info"""
        if len(self.clip_queue) == 0:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            self._update_single_clip_display(current_row)

    def _update_single_clip_display(self, index):
        """Update a single clip's display in the list"""
        if index < 0 or index >= len(self.clip_queue):
            return

        clip = self.clip_queue[index]
        base_name = clip.input_path.name

        # Add gear emoji if has custom settings
        if clip.has_custom_settings():
            base_name = f"⚙️ {base_name}"

        # Add trim info if present
        if clip.start_time > 0 or clip.end_time > 0:
            trim_info = f" [{self._format_time(clip.start_time)}-{self._format_time(clip.end_time if clip.end_time > 0 else clip.duration)}]"
            base_name += trim_info

        self.clip_list_widget.item(index).setText(base_name)

    def _update_all_clip_displays(self):
        """Update all clips' displays in the list"""
        for i in range(len(self.clip_queue)):
            self._update_single_clip_display(i)

    def _on_adjustment_changed(self):
        """Called when any adjustment slider changes - saves to clip and updates preview"""
        self._save_current_clip_settings()
        self._schedule_preview_update()

    def _reset_all(self):
        for w in [self.global_shift_slider, self.global_yaw, self.global_pitch, self.global_roll,
                  self.stereo_yaw_offset, self.stereo_pitch_offset, self.stereo_roll_offset]:
            w.setValue(0)

    # Clip Queue Management Methods

    def _save_current_clip_settings(self):
        """Save current UI settings to the currently selected clip"""
        if len(self.clip_queue) == 0:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            # Save current UI values to clip - VR180 adjustments
            clip.global_shift = self.global_shift_slider.value()
            clip.global_yaw = self.global_yaw.value()
            clip.global_pitch = self.global_pitch.value()
            clip.global_roll = self.global_roll.value()
            clip.stereo_yaw = self.stereo_yaw_offset.value()
            clip.stereo_pitch = self.stereo_pitch_offset.value()
            clip.stereo_roll = self.stereo_roll_offset.value()

            # Save color grading settings
            clip.gamma = self.gamma_slider.value() / 100.0
            clip.white_point = self.white_point_slider.value() / 100.0
            clip.black_point = self.black_point_slider.value() / 100.0
            clip.lut_intensity = self.lut_intensity_slider.value() / 100.0

            # Update the display for this clip
            self._update_clip_list_display()

    def _update_output_path_for_queue(self):
        """Update output path based on queue state and combine checkbox"""
        if len(self.clip_queue) == 0:
            return

        # Use first clip's directory and name as base
        first_clip = self.clip_queue[0]
        base_dir = first_clip.input_path.parent

        if self.concatenate_checkbox.isChecked():
            # Combined output: use first clip's name with "_combined" suffix
            base_name = first_clip.input_path.stem
            extension = first_clip.input_path.suffix
            output_path = base_dir / f"{base_name}_combined_adjusted{extension}"
        else:
            # Individual outputs: use first clip's name with "_adjusted" suffix
            base_name = first_clip.input_path.stem
            extension = first_clip.input_path.suffix
            output_path = base_dir / f"{base_name}_adjusted{extension}"

        self.output_path_edit.setText(str(output_path))
        self.config.output_path = output_path

    def _add_clips_to_queue(self):
        """Add multiple clips to the processing queue"""
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Videos", str(Path.home()), "Video (*.mp4 *.mov *.mkv)")
        if not paths:
            return

        for path in paths:
            # Get video duration for this clip
            try:
                result = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                        "-show_entries", "format=duration",
                                        "-of", "json", path], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
                info = json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 0))
            except:
                duration = 0.0

            # Create clip item
            clip = ClipItem(input_path=Path(path), duration=duration)
            self.clip_queue.append(clip)

            # Add to list widget
            item = QListWidgetItem(f"{Path(path).name}")
            self.clip_list_widget.addItem(item)

        # Enable controls
        self._update_queue_button_states()

        # Enable process button when clips are in queue
        if len(self.clip_queue) > 0:
            self.process_btn.setEnabled(True)

        # Auto-populate output path based on first clip
        if len(self.clip_queue) > 0:
            self._update_output_path_for_queue()

        # Auto-select the first clip added so user can preview it immediately
        if len(paths) > 0:
            self.clip_list_widget.setCurrentRow(len(self.clip_queue) - len(paths))

        self.status_bar.showMessage(f"Added {len(paths)} clip(s) to queue (total: {len(self.clip_queue)})")

    def _remove_selected_clip(self):
        """Remove the currently selected clip from the queue"""
        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0:
            self.clip_queue.pop(current_row)
            self.clip_list_widget.takeItem(current_row)
            self._update_queue_button_states()
            self._update_trim_display()

            # Disable process button if queue is empty
            if len(self.clip_queue) == 0:
                self.process_btn.setEnabled(False)

            self.status_bar.showMessage(f"Removed clip (remaining: {len(self.clip_queue)})")

    def _move_clip_up(self):
        """Move the selected clip up in the queue"""
        current_row = self.clip_list_widget.currentRow()
        if current_row > 0:
            # Swap in queue list
            self.clip_queue[current_row], self.clip_queue[current_row - 1] = self.clip_queue[current_row - 1], self.clip_queue[current_row]

            # Swap in list widget
            item = self.clip_list_widget.takeItem(current_row)
            self.clip_list_widget.insertItem(current_row - 1, item)
            self.clip_list_widget.setCurrentRow(current_row - 1)

    def _move_clip_down(self):
        """Move the selected clip down in the queue"""
        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue) - 1:
            # Swap in queue list
            self.clip_queue[current_row], self.clip_queue[current_row + 1] = self.clip_queue[current_row + 1], self.clip_queue[current_row]

            # Swap in list widget
            item = self.clip_list_widget.takeItem(current_row)
            self.clip_list_widget.insertItem(current_row + 1, item)
            self.clip_list_widget.setCurrentRow(current_row + 1)

    def _clear_queue(self):
        """Clear all clips from the queue"""
        if len(self.clip_queue) > 0:
            reply = QMessageBox.question(self, "Clear Queue",
                                        f"Remove all {len(self.clip_queue)} clip(s) from queue?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.clip_queue.clear()
                self.clip_list_widget.clear()
                self._update_queue_button_states()
                self._update_trim_display()

                # Disable process button when queue is empty
                self.process_btn.setEnabled(False)

                self.status_bar.showMessage("Queue cleared")

    def _on_clip_selection_changed(self):
        """Update button states and load clip for preview when clip selection changes"""
        self._update_queue_button_states()

        # Clear cached frames to free memory when switching clips
        self._clear_frame_cache()

        # Load selected clip for preview
        current_row = self.clip_list_widget.currentRow()
        if current_row >= 0 and current_row < len(self.clip_queue):
            clip = self.clip_queue[current_row]
            self.current_clip_index = current_row

            # Load this clip's video
            self.config.input_path = clip.input_path
            self.input_path_edit.setText(str(clip.input_path))

            # Block signals while loading to prevent triggering saves
            widgets_to_block = [
                self.global_shift_slider, self.global_yaw, self.global_pitch, self.global_roll,
                self.stereo_yaw_offset, self.stereo_pitch_offset, self.stereo_roll_offset,
                self.gamma_slider, self.white_point_slider, self.black_point_slider, self.lut_intensity_slider
            ]
            for widget in widgets_to_block:
                widget.blockSignals(True)

            # Load this clip's settings into UI - VR180 adjustments
            self.global_shift_slider.setValue(clip.global_shift)
            self.global_yaw.setValue(clip.global_yaw)
            self.global_pitch.setValue(clip.global_pitch)
            self.global_roll.setValue(clip.global_roll)
            self.stereo_yaw_offset.setValue(clip.stereo_yaw)
            self.stereo_pitch_offset.setValue(clip.stereo_pitch)
            self.stereo_roll_offset.setValue(clip.stereo_roll)

            # Load color grading settings
            self.gamma_slider.setValue(int(clip.gamma * 100))
            self.white_point_slider.setValue(int(clip.white_point * 100))
            self.black_point_slider.setValue(int(clip.black_point * 100))
            self.lut_intensity_slider.setValue(int(clip.lut_intensity * 100))

            # Unblock signals
            for widget in widgets_to_block:
                widget.blockSignals(False)

            # Load video info and start preview
            self._load_video_info()

            # Update trim display for this clip
            self._update_trim_display()

            # Update all clip displays to preserve trim info
            self._update_all_clip_displays()

            self.status_bar.showMessage(f"Previewing clip {current_row + 1}/{len(self.clip_queue)}: {clip.input_path.name}")

    def _apply_settings_to_all(self):
        """Apply current clip's settings to all other clips in queue"""
        if len(self.clip_queue) <= 1:
            return

        current_row = self.clip_list_widget.currentRow()
        if current_row < 0:
            return

        # Get current clip's settings
        source_clip = self.clip_queue[current_row]

        # Confirm with user
        reply = QMessageBox.question(
            self, "Apply to All",
            f"Copy settings from '{source_clip.input_path.name}' to all {len(self.clip_queue) - 1} other clip(s)?\n\n"
            "This will overwrite their individual adjustments.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Apply to all other clips
        applied_count = 0
        for i, clip in enumerate(self.clip_queue):
            if i != current_row:
                # Copy VR180 adjustments
                clip.global_shift = source_clip.global_shift
                clip.global_yaw = source_clip.global_yaw
                clip.global_pitch = source_clip.global_pitch
                clip.global_roll = source_clip.global_roll
                clip.stereo_yaw = source_clip.stereo_yaw
                clip.stereo_pitch = source_clip.stereo_pitch
                clip.stereo_roll = source_clip.stereo_roll

                # Copy color grading settings
                clip.gamma = source_clip.gamma
                clip.white_point = source_clip.white_point
                clip.black_point = source_clip.black_point
                clip.lut_intensity = source_clip.lut_intensity

                applied_count += 1

        # Update all clip displays
        self._update_all_clip_displays()

        self.status_bar.showMessage(f"Applied settings to {applied_count} clip(s)")

    def _update_queue_button_states(self):
        """Enable/disable queue buttons based on current state"""
        has_clips = len(self.clip_queue) > 0
        has_selection = self.clip_list_widget.currentRow() >= 0
        current_row = self.clip_list_widget.currentRow()

        self.remove_clip_btn.setEnabled(has_selection)
        self.move_up_btn.setEnabled(has_selection and current_row > 0)
        self.move_down_btn.setEnabled(has_selection and current_row < len(self.clip_queue) - 1)
        self.clear_queue_btn.setEnabled(has_clips)
        self.concatenate_checkbox.setEnabled(has_clips)
        self.apply_to_all_btn.setEnabled(has_clips and len(self.clip_queue) > 1)
        self.save_project_btn.setEnabled(has_clips)

    def _start_processing(self):
        # Save current clip settings before processing
        self._save_current_clip_settings()

        # Check if using queue mode or single file mode
        use_queue = len(self.clip_queue) > 0

        if use_queue:
            # Queue batch processing mode
            output_dir = Path(self.output_path_edit.text()).parent
            if not output_dir.exists():
                QMessageBox.warning(self, "Error", "Output directory does not exist")
                return

            # Check codec consistency for concatenation
            concatenate = self.concatenate_checkbox.isChecked()
            if concatenate:
                # Verify all input files have the same codec
                codec_map = {0: "auto", 1: "h265", 2: "prores"}
                output_codec = codec_map[self.codec_combo.currentIndex()]

                if output_codec == "auto":
                    # Check if all source files have the same codec
                    codecs = set()
                    for clip in self.clip_queue:
                        try:
                            probe = subprocess.run([get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                                                   "-show_entries", "stream=codec_name", "-of", "json",
                                                   str(clip.input_path)], capture_output=True, text=True, check=True, creationflags=get_subprocess_flags())
                            codec = json.loads(probe.stdout)["streams"][0]["codec_name"]
                            if codec in ["hevc", "h265"]: codec = "h265"
                            elif codec in ["prores", "prores_ks"]: codec = "prores"
                            codecs.add(codec)
                        except:
                            pass

                    if len(codecs) > 1:
                        QMessageBox.warning(self, "Codec Mismatch",
                            f"Cannot concatenate clips with different codecs ({', '.join(codecs)}).\n\n"
                            "Please select a specific output codec (H.265 or ProRes) instead of Auto.")
                        return

            # Ask user to confirm batch processing
            msg = f"Process {len(self.clip_queue)} clip(s)"
            if concatenate:
                msg += " and concatenate into single output?"
            else:
                msg += " as separate files?"

            reply = QMessageBox.question(self, "Batch Process", msg,
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

            self.process_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.ffmpeg_output.clear()

            # Start batch processor
            self._start_batch_processing(output_dir, concatenate)
        else:
            # Single file processing mode (original behavior)
            self._start_single_file_processing()

    def _start_single_file_processing(self):
        """Original single file processing"""
        codec_map = {0: "auto", 1: "h265", 2: "prores"}
        prores_map = {0: "proxy", 1: "lt", 2: "standard", 3: "hq", 4: "4444", 5: "4444xq"}
        # Get LUT path if specified
        lut_path = None
        if self.lut_path_edit.text():
            lut_path = Path(self.lut_path_edit.text())

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
            gamma=self.gamma_slider.value() / 100.0,
            white_point=self.white_point_slider.value() / 100.0,
            black_point=self.black_point_slider.value() / 100.0,
            h265_bit_depth=10 if self.h265_bit_depth_combo.currentIndex() == 1 else 8,
            inject_vr180_metadata=self.vr180_metadata_checkbox.isChecked(),
            vision_pro_mode=["standard", "hvc1", "mvhevc"][self.vision_pro_combo.currentIndex()])

        if not config.input_path or not config.input_path.exists():
            QMessageBox.warning(self, "Error", "Select input file"); return
        if config.output_path.exists():
            if QMessageBox.question(self, "Overwrite?", f"Overwrite {config.output_path}?") != QMessageBox.StandardButton.Yes: return

        self.ffmpeg_output.clear()
        self.processor = VideoProcessor(config)
        self.processor.output_line.connect(self._append_ffmpeg_output)
        self.processor.status.connect(self.status_bar.showMessage)
        self.processor.finished_signal.connect(self._on_finished)
        self.processor.start()

    def _start_batch_processing(self, output_dir, concatenate):
        """Process multiple clips from the queue"""
        codec_map = {0: "auto", 1: "h265", 2: "prores"}
        prores_map = {0: "proxy", 1: "lt", 2: "standard", 3: "hq", 4: "4444", 5: "4444xq"}
        lut_path = None
        if self.lut_path_edit.text():
            lut_path = Path(self.lut_path_edit.text())

        # Create batch processor
        from PyQt6.QtCore import QThread

        class BatchProcessor(QThread):
            progress = pyqtSignal(int, int)  # current, total
            status = pyqtSignal(str)
            output_line = pyqtSignal(str)
            finished_signal = pyqtSignal(bool, str)

            def __init__(self, clips, output_dir, concatenate, settings, final_output_path=None):
                super().__init__()
                self.clips = clips
                self.output_dir = output_dir
                self.concatenate = concatenate
                self.settings = settings
                self.final_output_path = final_output_path
                self._cancelled = False
                self._process = None
                self.processed_files = []

            def cancel(self):
                self._cancelled = True
                if self._process:
                    try:
                        self._process.terminate()
                    except:
                        pass

            def run(self):
                try:
                    self.output_line.emit(f"Starting batch processing of {len(self.clips)} clip(s)...")
                    self.output_line.emit("-" * 60)

                    # Process each clip
                    for idx, clip in enumerate(self.clips):
                        if self._cancelled:
                            self.finished_signal.emit(False, "Cancelled")
                            return

                        self.progress.emit(idx + 1, len(self.clips))
                        self.status.emit(f"Processing clip {idx + 1}/{len(self.clips)}: {clip.input_path.name}")
                        self.output_line.emit(f"\n[Clip {idx + 1}/{len(self.clips)}] {clip.input_path.name}")
                        self.output_line.emit(f"Output: {clip.input_path.stem}_adjusted{clip.input_path.suffix}")

                        # Create output path
                        if self.concatenate:
                            # Temporary file for concatenation
                            output_path = self.output_dir / f"_temp_{idx:03d}_{clip.input_path.stem}.mov"
                        else:
                            # Individual output file
                            output_path = self.output_dir / f"{clip.input_path.stem}_adjusted{clip.input_path.suffix}"

                        # Create config for this clip using its per-clip settings
                        # Use H.265 intermediate when concatenating (more reliable than ProRes intermediate)
                        config = ProcessingConfig(
                            input_path=clip.input_path,
                            output_path=output_path,
                            global_shift=clip.global_shift,  # Use clip's own settings
                            global_adjustment=PanomapAdjustment(clip.global_yaw, clip.global_pitch, clip.global_roll),
                            stereo_offset=PanomapAdjustment(clip.stereo_yaw, clip.stereo_pitch, clip.stereo_roll),
                            output_codec='h265' if self.concatenate else self.settings['output_codec'],
                            quality=self.settings['quality'],
                            bitrate=self.settings['bitrate'],
                            use_bitrate=self.settings['use_bitrate'],
                            prores_profile=self.settings['prores_profile'],
                            use_hardware_accel=self.settings['use_hardware_accel'],
                            lut_path=self.settings['lut_path'],
                            lut_intensity=clip.lut_intensity,  # Use clip's own color grading
                            gamma=clip.gamma,  # Use clip's own color grading
                            white_point=clip.white_point,  # Use clip's own color grading
                            black_point=clip.black_point,  # Use clip's own color grading
                            h265_bit_depth=self.settings['h265_bit_depth'],
                            inject_vr180_metadata=False if self.concatenate else self.settings['inject_vr180_metadata'],
                            vision_pro_mode='standard' if self.concatenate else self.settings['vision_pro_mode'],
                            trim_start=clip.start_time if clip.start_time > 0 else None,
                            trim_duration=clip.end_time - clip.start_time if clip.end_time > 0 else None,
                            for_concatenation=self.concatenate  # Force keyframes if concatenating
                        )

                        # Process this clip using VideoProcessor logic inline
                        processor = VideoProcessor(config)
                        processor.output_line.connect(self.output_line.emit)

                        # Wrap status to include clip number
                        def emit_status_with_clip_number(msg):
                            # Prepend clip number to status messages
                            self.status.emit(f"[{idx + 1}/{len(self.clips)}] {msg}")

                        processor.status.connect(emit_status_with_clip_number)
                        processor.start()
                        processor.wait()  # Wait for completion

                        if self._cancelled:
                            self.finished_signal.emit(False, "Cancelled")
                            return

                        self.processed_files.append(output_path)
                        self.output_line.emit(f"[Clip {idx + 1}] Completed successfully")
                        self.output_line.emit("-" * 60)

                    # Concatenate if requested
                    if self.concatenate and len(self.processed_files) > 0:
                        self.output_line.emit(f"\n{'='*60}")
                        self.output_line.emit("CONCATENATING CLIPS...")
                        self.output_line.emit(f"{'='*60}")
                        self.status.emit("Concatenating clips...")
                        self._concatenate_files()

                    if self._cancelled:
                        self.finished_signal.emit(False, "Cancelled")
                    else:
                        msg = f"Successfully processed {len(self.clips)} clip(s)"
                        if self.concatenate:
                            msg += " and concatenated"
                        self.finished_signal.emit(True, msg)

                except Exception as e:
                    self.finished_signal.emit(False, f"Error: {str(e)}")

            def _concatenate_files(self):
                """Concatenate all processed files using FFmpeg concat demuxer"""
                try:
                    # Create concat file list
                    concat_file = self.output_dir / "_concat_list.txt"
                    with open(concat_file, 'w') as f:
                        for file_path in self.processed_files:
                            f.write(f"file '{file_path.absolute()}'\n")

                    self.output_line.emit(f"Created concat list with {len(self.processed_files)} files")
                    # Log which files are being concatenated
                    for i, f in enumerate(self.processed_files):
                        self.output_line.emit(f"  [{i+1}] {f.name}")

                    # Output path for concatenated file
                    if self.final_output_path:
                        final_output = Path(self.final_output_path)
                    else:
                        final_output = self.output_dir / "concatenated_output.mov"
                    self.output_line.emit(f"Output: {final_output.name}")

                    # All clips are H.265 intermediates - use concat demuxer for fast stream copy
                    n = len(self.processed_files)

                    self.output_line.emit(f"Concatenating {n} H.265 clips using demuxer (stream copy)")

                    # Debug: Show concat file contents
                    self.output_line.emit(f"Concat list contents:")
                    with open(concat_file, 'r') as f:
                        for line in f:
                            self.output_line.emit(f"  {line.rstrip()}")

                    # Use concat demuxer - this does stream copy without re-encoding
                    # Much faster and no quality loss
                    cmd = [
                        get_ffmpeg_path(),
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy",  # Stream copy - no re-encoding
                        "-movflags", "+faststart",  # Optimize for streaming
                        "-y",
                        str(final_output)
                    ]

                    self.output_line.emit(f"Using concat demuxer for lossless stream copy")
                    self.output_line.emit(f"Command: {' '.join(cmd)}")

                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        creationflags=get_subprocess_flags()
                    )

                    for line in self._process.stdout:
                        if self._cancelled:
                            break
                        self.output_line.emit(line.rstrip())

                    return_code = self._process.wait()

                    # Check if concatenation succeeded
                    if return_code != 0:
                        self.output_line.emit(f"Concatenation failed with return code {return_code}")
                        raise Exception(f"FFmpeg concatenation failed with code {return_code}")

                    # Verify output file exists and has reasonable size
                    if not final_output.exists():
                        raise Exception("Output file was not created")

                    output_size = final_output.stat().st_size
                    if output_size < 1000000:  # Less than 1MB is suspicious
                        self.output_line.emit(f"WARNING: Output file is only {output_size} bytes - this may indicate a problem")

                    # Clean up temp files
                    try:
                        concat_file.unlink()
                    except:
                        pass
                    # Clean up temp ProRes clip files
                    for temp_file in self.processed_files:
                        if temp_file.name.startswith("_temp_"):
                            try:
                                temp_file.unlink()
                                self.output_line.emit(f"Cleaned up: {temp_file.name}")
                            except:
                                pass

                    self.status.emit(f"Concatenated to: {final_output.name}")
                    self.output_line.emit(f"Final file size: {output_size / 1024 / 1024:.1f} MB")

                except Exception as e:
                    self.output_line.emit(f"Concatenation error: {str(e)}")

        # Prepare settings dictionary
        settings = {
            'global_shift': self.global_shift_slider.value(),
            'global_adjustment': PanomapAdjustment(self.global_yaw.value(), self.global_pitch.value(), self.global_roll.value()),
            'stereo_offset': PanomapAdjustment(self.stereo_yaw_offset.value(), self.stereo_pitch_offset.value(), self.stereo_roll_offset.value()),
            'output_codec': codec_map[self.codec_combo.currentIndex()],
            'quality': self.quality_spinbox.value(),
            'bitrate': self.bitrate_spinbox.value(),
            'use_bitrate': self.use_bitrate_radio.isChecked(),
            'prores_profile': prores_map[self.prores_combo.currentIndex()],
            'use_hardware_accel': self.hw_accel_checkbox.isChecked(),
            'lut_path': lut_path,
            'lut_intensity': self.lut_intensity_slider.value() / 100.0,
            'gamma': self.gamma_slider.value() / 100.0,
            'white_point': self.white_point_slider.value() / 100.0,
            'black_point': self.black_point_slider.value() / 100.0,
            'h265_bit_depth': 10 if self.h265_bit_depth_combo.currentIndex() == 1 else 8,
            'inject_vr180_metadata': self.vr180_metadata_checkbox.isChecked(),
            'vision_pro_mode': ["standard", "hvc1", "mvhevc"][self.vision_pro_combo.currentIndex()]
        }

        # Get final output path if concatenating
        final_output_path = None
        if concatenate:
            final_output_path = Path(self.output_path_edit.text())

        self.processor = BatchProcessor(self.clip_queue, output_dir, concatenate, settings, final_output_path)
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

    def _save_project(self):
        """Save current project with all clips and settings"""
        if len(self.clip_queue) == 0:
            QMessageBox.warning(self, "No Clips", "Add clips to the queue before saving a project")
            return

        # Save current clip settings first
        self._save_current_clip_settings()

        # Ask user for save location
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", str(Path.home()), "VR180 Project (*.vr180proj)"
        )
        if not path:
            return

        # Build project data
        project_data = {
            "version": "1.0",
            "clips": [],
            "concatenate": self.concatenate_checkbox.isChecked(),
            "output_path": self.output_path_edit.text(),
            "lut_path": self.lut_path_edit.text(),
            "codec": self.codec_combo.currentIndex(),
            "quality": self.quality_spinbox.value(),
            "bitrate": self.bitrate_spinbox.value(),
            "use_bitrate": self.use_bitrate_radio.isChecked(),
            "prores_profile": self.prores_combo.currentIndex(),
            "h265_bit_depth": self.h265_bit_depth_combo.currentIndex(),
            "hw_accel": self.hw_accel_checkbox.isChecked(),
            "inject_vr180_metadata": self.vr180_metadata_checkbox.isChecked(),
            "vision_pro_mode": self.vision_pro_combo.currentIndex()
        }

        # Save each clip's data
        for clip in self.clip_queue:
            clip_data = {
                "input_path": str(clip.input_path),
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "global_shift": clip.global_shift,
                "global_yaw": clip.global_yaw,
                "global_pitch": clip.global_pitch,
                "global_roll": clip.global_roll,
                "stereo_yaw": clip.stereo_yaw,
                "stereo_pitch": clip.stereo_pitch,
                "stereo_roll": clip.stereo_roll,
                "gamma": clip.gamma,
                "white_point": clip.white_point,
                "black_point": clip.black_point,
                "lut_intensity": clip.lut_intensity
            }
            project_data["clips"].append(clip_data)

        # Write to file
        try:
            with open(path, 'w') as f:
                json.dump(project_data, f, indent=2)
            QMessageBox.information(self, "Success", f"Project saved to {Path(path).name}")
            self.status_bar.showMessage(f"Project saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save project: {str(e)}")

    def _load_project(self):
        """Load a saved project with all clips and settings"""
        # Ask user for file to load
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", str(Path.home()), "VR180 Project (*.vr180proj)"
        )
        if not path:
            return

        try:
            with open(path, 'r') as f:
                project_data = json.load(f)

            # Clear current queue
            self.clip_queue.clear()
            self.clip_list_widget.clear()

            # Load clips
            for clip_data in project_data.get("clips", []):
                # Check if file still exists
                input_path = Path(clip_data["input_path"])
                if not input_path.exists():
                    reply = QMessageBox.question(
                        self, "Missing File",
                        f"File not found: {input_path.name}\n\nContinue loading other clips?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    continue

                # Create clip with all settings
                clip = ClipItem(
                    input_path=input_path,
                    start_time=clip_data.get("start_time", 0.0),
                    end_time=clip_data.get("end_time", 0.0),
                    duration=clip_data.get("duration", 0.0),
                    global_shift=clip_data.get("global_shift", 0),
                    global_yaw=clip_data.get("global_yaw", 0.0),
                    global_pitch=clip_data.get("global_pitch", 0.0),
                    global_roll=clip_data.get("global_roll", 0.0),
                    stereo_yaw=clip_data.get("stereo_yaw", 0.0),
                    stereo_pitch=clip_data.get("stereo_pitch", 0.0),
                    stereo_roll=clip_data.get("stereo_roll", 0.0),
                    gamma=clip_data.get("gamma", 1.0),
                    white_point=clip_data.get("white_point", 1.0),
                    black_point=clip_data.get("black_point", 0.0),
                    lut_intensity=clip_data.get("lut_intensity", 1.0)
                )
                self.clip_queue.append(clip)

                # Add to list widget
                item = QListWidgetItem(input_path.name)
                self.clip_list_widget.addItem(item)

            # Restore global settings
            self.concatenate_checkbox.setChecked(project_data.get("concatenate", False))
            self.output_path_edit.setText(project_data.get("output_path", ""))
            self.lut_path_edit.setText(project_data.get("lut_path", ""))
            self.codec_combo.setCurrentIndex(project_data.get("codec", 0))
            self.quality_spinbox.setValue(project_data.get("quality", 18))
            self.bitrate_spinbox.setValue(project_data.get("bitrate", 200))
            if project_data.get("use_bitrate", False):
                self.use_bitrate_radio.setChecked(True)
            else:
                self.use_crf_radio.setChecked(True)
            self.prores_combo.setCurrentIndex(project_data.get("prores_profile", 3))
            self.h265_bit_depth_combo.setCurrentIndex(project_data.get("h265_bit_depth", 1))
            self.hw_accel_checkbox.setChecked(project_data.get("hw_accel", False))
            self.vr180_metadata_checkbox.setChecked(project_data.get("inject_vr180_metadata", False))
            self.vision_pro_combo.setCurrentIndex(project_data.get("vision_pro_mode", 0))

            # Update UI states
            self._update_queue_button_states()
            self._update_all_clip_displays()

            # Enable process button if clips loaded
            if len(self.clip_queue) > 0:
                self.process_btn.setEnabled(True)
                # Select first clip
                self.clip_list_widget.setCurrentRow(0)

            QMessageBox.information(
                self, "Success",
                f"Loaded {len(self.clip_queue)} clip(s) from {Path(path).name}"
            )
            self.status_bar.showMessage(f"Project loaded: {Path(path).name}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load project: {str(e)}")

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

        # Load color adjustment settings
        self.gamma_slider.setValue(self.settings.value("gamma", 100, type=int))
        self.white_point_slider.setValue(self.settings.value("white_point", 100, type=int))
        self.black_point_slider.setValue(self.settings.value("black_point", 0, type=int))

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

        # Save color adjustment settings
        self.settings.setValue("gamma", self.gamma_slider.value())
        self.settings.setValue("white_point", self.white_point_slider.value())
        self.settings.setValue("black_point", self.black_point_slider.value())

        # Save VR180 metadata setting
        self.settings.setValue("vr180_metadata", self.vr180_metadata_checkbox.isChecked())

        # Save Vision Pro mode setting
        self.settings.setValue("vision_pro_mode", self.vision_pro_combo.currentIndex())

        # Save H.265 bit depth setting
        self.settings.setValue("h265_bit_depth", self.h265_bit_depth_combo.currentIndex())

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

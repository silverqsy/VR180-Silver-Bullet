#!/usr/bin/env python3
"""
VR180 SBS Half-Equirectangular Video Processor - CLI Edition

Command-line tool for batch processing VR180 videos with panomap adjustments.
"""

import subprocess
import argparse
import json
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


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


@dataclass
class PanomapAdjustment:
    """Rotation adjustments in degrees"""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class ProcessingConfig:
    input_path: Path
    output_path: Path
    global_shift: int = 0
    global_adjustment: PanomapAdjustment = None
    stereo_offset: Optional[PanomapAdjustment] = None  # Applied +/- to left/right eyes
    output_codec: Optional[str] = None
    quality: int = 18
    prores_profile: str = "hq"
    
    def __post_init__(self):
        if self.global_adjustment is None:
            self.global_adjustment = PanomapAdjustment()
        if self.stereo_offset is None:
            self.stereo_offset = PanomapAdjustment()


def detect_codec(input_path: Path) -> str:
    """Detect the codec of the input video"""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    codec = data["streams"][0]["codec_name"]
    
    if codec in ["hevc", "h265"]:
        return "h265"
    elif codec in ["prores", "prores_ks"]:
        return "prores"
    return codec


def get_video_info(input_path: Path) -> dict:
    """Get video metadata"""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,pix_fmt,color_space,color_transfer,color_primaries",
        "-show_entries", "format=duration",
        "-of", "json",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def build_filter_complex(config: ProcessingConfig) -> str:
    """Build the FFmpeg filter_complex string"""
    filters = []
    
    # Global horizontal shift using split/crop/hstack for efficient wrap-around
    if config.global_shift != 0:
        shift = config.global_shift
        if shift > 0:
            # Take 'shift' pixels from left, put on right
            filters.append(f"[0:v]split=2[sh_a][sh_b]")
            filters.append(f"[sh_a]crop={shift}:ih:0:0[sh_right]")
            filters.append(f"[sh_b]crop=iw-{shift}:ih:{shift}:0[sh_left]")
            filters.append(f"[sh_left][sh_right]hstack=inputs=2[shifted]")
        else:
            # Negative shift = move image left
            abs_shift = abs(shift)
            filters.append(f"[0:v]split=2[sh_a][sh_b]")
            filters.append(f"[sh_a]crop={abs_shift}:ih:iw-{abs_shift}:0[sh_left]")
            filters.append(f"[sh_b]crop=iw-{abs_shift}:ih:0:0[sh_right]")
            filters.append(f"[sh_left][sh_right]hstack=inputs=2[shifted]")
        input_label = "[shifted]"
    else:
        input_label = "[0:v]"
    
    # Split into left and right eyes
    filters.append(f"{input_label}split=2[full1][full2]")
    filters.append("[full1]crop=iw/2:ih:0:0[left_in]")
    filters.append("[full2]crop=iw/2:ih:iw/2:0[right_in]")
    
    # Calculate per-eye adjustments using stereo offset
    # Left eye: global + stereo offset
    # Right eye: global - stereo offset (opposite)
    stereo = config.stereo_offset or PanomapAdjustment()
    
    left_yaw = config.global_adjustment.yaw + stereo.yaw
    left_pitch = config.global_adjustment.pitch + stereo.pitch
    left_roll = config.global_adjustment.roll + stereo.roll
    
    right_yaw = config.global_adjustment.yaw - stereo.yaw
    right_pitch = config.global_adjustment.pitch - stereo.pitch
    right_roll = config.global_adjustment.roll - stereo.roll
    
    # Apply v360 adjustments using hequirect (half equirectangular) format
    if any([left_yaw, left_pitch, left_roll]):
        left_v360 = (f"v360=input=hequirect:output=hequirect:"
                    f"yaw={left_yaw}:pitch={left_pitch}:roll={left_roll}:"
                    f"interp=lanczos")
        filters.append(f"[left_in]{left_v360}[left_out]")
    else:
        filters.append("[left_in]null[left_out]")
    
    if any([right_yaw, right_pitch, right_roll]):
        right_v360 = (f"v360=input=hequirect:output=hequirect:"
                     f"yaw={right_yaw}:pitch={right_pitch}:roll={right_roll}:"
                     f"interp=lanczos")
        filters.append(f"[right_in]{right_v360}[right_out]")
    else:
        filters.append("[right_in]null[right_out]")
    
    # Stack back to SBS
    filters.append("[left_out][right_out]hstack=inputs=2[out]")
    
    return ";".join(filters)


def get_encoder_settings(codec: str, config: ProcessingConfig, video_info: dict) -> list:
    """Get FFmpeg encoder settings"""
    stream = video_info["streams"][0]
    
    if codec == "h265":
        settings = [
            "-c:v", "libx265",
            "-crf", str(config.quality),
            "-preset", "slow",
            "-tag:v", "hvc1",
        ]
        if "pix_fmt" in stream:
            settings.extend(["-pix_fmt", stream["pix_fmt"]])
        if stream.get("color_space"):
            settings.extend(["-colorspace", stream["color_space"]])
        if stream.get("color_transfer"):
            settings.extend(["-color_trc", stream["color_transfer"]])
        if stream.get("color_primaries"):
            settings.extend(["-color_primaries", stream["color_primaries"]])
            
    elif codec == "prores":
        profile_map = {
            "proxy": "0", "lt": "1", "standard": "2",
            "hq": "3", "4444": "4", "4444xq": "5"
        }
        settings = [
            "-c:v", "prores_ks",
            "-profile:v", profile_map.get(config.prores_profile, "3"),
            "-vendor", "apl0",
            "-pix_fmt", stream.get("pix_fmt", "yuv422p10le"),
        ]
    else:
        settings = ["-c:v", codec]
    
    return settings


def process_video(config: ProcessingConfig, verbose: bool = False):
    """Process a single video"""
    print(f"\n{'='*60}")
    print(f"Processing: {config.input_path.name}")
    print('='*60)
    
    # Detect codec
    input_codec = detect_codec(config.input_path)
    output_codec = config.output_codec or input_codec
    
    print(f"Input codec: {input_codec}")
    print(f"Output codec: {output_codec}")
    print(f"Global shift: {config.global_shift}px")
    print(f"Global adjustment: yaw={config.global_adjustment.yaw}°, "
          f"pitch={config.global_adjustment.pitch}°, "
          f"roll={config.global_adjustment.roll}°")
    
    if config.stereo_offset and any([config.stereo_offset.yaw, config.stereo_offset.pitch, config.stereo_offset.roll]):
        print(f"Stereo offset: yaw={config.stereo_offset.yaw}°, "
              f"pitch={config.stereo_offset.pitch}°, "
              f"roll={config.stereo_offset.roll}°")
    
    # Get video info
    video_info = get_video_info(config.input_path)
    
    # Build filter and encoder settings
    filter_complex = build_filter_complex(config)
    encoder_settings = get_encoder_settings(output_codec, config, video_info)
    
    # Build FFmpeg command
    cmd = [
        get_ffmpeg_path(),
        "-i", str(config.input_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:a", "copy",
    ]
    cmd.extend(encoder_settings)
    cmd.extend(["-y", str(config.output_path)])
    
    if verbose:
        print("\nFFmpeg command:")
        print(" ".join(cmd))
    
    # Run FFmpeg
    print("\nEncoding...")
    process = subprocess.run(cmd, capture_output=not verbose)
    
    if process.returncode != 0:
        if not verbose:
            print("Error:", process.stderr.decode() if process.stderr else "Unknown error")
        raise RuntimeError(f"FFmpeg failed with code {process.returncode}")
    
    print(f"✓ Output saved to: {config.output_path}")


def batch_process(input_dir: Path, output_dir: Path, config_template: ProcessingConfig,
                  extensions: Tuple[str, ...] = (".mp4", ".mov", ".mkv"),
                  verbose: bool = False):
    """Process all videos in a directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    print(f"Found {len(video_files)} video files")
    
    for i, input_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        
        output_path = output_dir / f"{input_path.stem}_adjusted{input_path.suffix}"
        
        config = ProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            global_shift=config_template.global_shift,
            global_adjustment=config_template.global_adjustment,
            stereo_offset=config_template.stereo_offset,
            output_codec=config_template.output_codec,
            quality=config_template.quality,
            prores_profile=config_template.prores_profile
        )
        
        try:
            process_video(config, verbose=verbose)
        except Exception as e:
            print(f"✗ Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Process VR180 SBS videos with panomap adjustments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix split-eye frame and adjust horizon
  %(prog)s input.mp4 output.mp4 --shift 3840 --roll -2

  # Full stereo alignment with offset
  %(prog)s input.mov output.mov --yaw 2 --pitch -1 --stereo-yaw 0.3

  # Batch process folder
  %(prog)s ./raw/ ./processed/ --shift 3840 --yaw 3

  # High quality H.265 output
  %(prog)s input.mp4 output.mp4 --codec h265 --quality 14
        """
    )
    
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    
    # Global shift
    parser.add_argument("-s", "--shift", type=int, default=0,
                        help="Global horizontal shift in pixels (for fixing split-eye frames)")
    
    # Global adjustments
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Global yaw adjustment in degrees")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Global pitch adjustment in degrees")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Global roll adjustment in degrees")
    
    # Stereo offset adjustments (applied +/- to left/right eyes)
    parser.add_argument("--stereo-yaw", type=float, default=0.0,
                        help="Stereo yaw offset (+ to left, - to right)")
    parser.add_argument("--stereo-pitch", type=float, default=0.0,
                        help="Stereo pitch offset (+ to left, - to right)")
    parser.add_argument("--stereo-roll", type=float, default=0.0,
                        help="Stereo roll offset (+ to left, - to right)")
    
    # Output settings
    parser.add_argument("-c", "--codec", choices=["h265", "prores", "auto"], default="auto",
                        help="Output codec (default: auto)")
    parser.add_argument("-q", "--quality", type=int, default=18,
                        help="Quality (CRF for H.265, default: 18)")
    parser.add_argument("--prores-profile",
                        choices=["proxy", "lt", "standard", "hq", "4444", "4444xq"],
                        default="hq", help="ProRes profile (default: hq)")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show FFmpeg output")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Build adjustments
    global_adj = PanomapAdjustment(args.yaw, args.pitch, args.roll)
    stereo_offset = PanomapAdjustment(args.stereo_yaw, args.stereo_pitch, args.stereo_roll)
    
    output_codec = None if args.codec == "auto" else args.codec
    
    if input_path.is_dir():
        # Batch mode
        config_template = ProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            global_shift=args.shift,
            global_adjustment=global_adj,
            stereo_offset=stereo_offset,
            output_codec=output_codec,
            quality=args.quality,
            prores_profile=args.prores_profile
        )
        batch_process(input_path, output_path, config_template, verbose=args.verbose)
    else:
        # Single file mode
        config = ProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            global_shift=args.shift,
            global_adjustment=global_adj,
            stereo_offset=stereo_offset,
            output_codec=output_codec,
            quality=args.quality,
            prores_profile=args.prores_profile
        )
        process_video(config, verbose=args.verbose)


if __name__ == "__main__":
    main()

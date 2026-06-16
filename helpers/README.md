# helpers/

macOS-native Swift binaries the Rust pipeline spawns as external
processes. They use `VideoToolbox` / `AVFoundation` / `AudioToolbox`
APIs that have no wgpu / ffmpeg-next equivalent, so we keep them
exactly as they are in the Python app.

## Helpers

| Helper | Purpose | Frameworks |
|---|---|---|
| `mvhevc_encode` | MV-HEVC spatial video encode via `VTCompressionSessionEncodeMultiImageFrame`. Bakes Apple spatial metadata into the format description (no post-encode transcode). | AVFoundation, VideoToolbox, CoreMedia, CoreVideo, Accelerate |
| `vt_denoise` | `VTTemporalNoiseFilter` with `64RGBALE` intermediate (true 10-bit through denoise). | AVFoundation, VideoToolbox, CoreMedia, CoreVideo |
| `apac_encode` | Apple Positional Audio Codec encode for Vision Pro spatial audio. Two modes: audio-only (`--input → --output`) or one-pass encode+mux with passthrough video (`--input + --video-input → --output`). The mux mode exists because ffmpeg's mov muxer drops APAC's `dapa` config atom on `-c:a copy`. | AVFoundation, CoreMedia, AudioToolbox |
| `braw_helper` | Blackmagic RAW decode/metadata/gyro/audio extraction. Modes: `--info <file>` (JSON metadata to stdout), `--decode <file> [--track N] [--16bit]` (BGRA/BGRA16 frames on stdout, JSON header on stderr), `--gyro <file>` (interleaved float32 `[gx gy gz ax ay az]` on stdout), `--audio <file>` (RIFF WAV on stdout). Multi-track (URSA Cine Immersive, Pyxis 12K stereo) is auto-side-by-side without `--track`. | BlackmagicRawAPI, CoreFoundation, CoreServices |

## Building

```sh
./helpers/build_swift.sh   # mvhevc_encode, vt_denoise, apac_encode
./helpers/braw/build_braw.sh   # braw_helper (requires Blackmagic RAW SDK installed)
```

Output binaries go to `helpers/bin/`, which is `.gitignore`d. The
sources in `helpers/swift/` and `helpers/braw/` are committed; the
binaries are not.

`braw_helper` is C++ (not Swift) because the Blackmagic RAW SDK ships
only C++ headers. It links against the user's installed `Blackmagic
RAW.framework` at build time (the SDK is not redistributable). Without
the SDK installed, the BRAW input path is unavailable — the rest of the
app still works.

## Why these stay Swift

These are Apple-only APIs with no portable equivalent. APAC in
particular requires `kAudioFormatAPAC` (introduced macOS 14) which
no Rust crate wraps. The Python app already proved the helper-
subprocess pattern works end-to-end; we keep it verbatim and just
spawn from Rust instead of Python.

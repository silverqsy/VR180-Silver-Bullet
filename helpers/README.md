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

## Building

```sh
./helpers/build_swift.sh
```

Output binaries go to `helpers/bin/`, which is `.gitignore`d. The
sources in `helpers/swift/` are committed; the binaries are not.

## Why these stay Swift

These are Apple-only APIs with no portable equivalent. APAC in
particular requires `kAudioFormatAPAC` (introduced macOS 14) which
no Rust crate wraps. The Python app already proved the helper-
subprocess pattern works end-to-end; we keep it verbatim and just
spawn from Rust instead of Python.

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
| `braw_helper` | Blackmagic RAW decode/metadata/gyro/audio extraction (SDK 6.0). Modes: `--info <file>` (JSON metadata to stdout: size, rate, frames, gyro, audio, readout from the sensor line time, sampled shutter, colour science, embedded LUT title, pipeline, resolution scales, and a `color` block: the SDK's clip attributes with their ranges/lists — colour science gen, gamma, gamut, tone curve, highlight recovery, gamut compression — plus frame 0's white balance, tint, exposure, ISO and the ISO list), `--decode <file> [--start N] [--count M] [--track N] [--16bit] [--scale full|half|quarter|eighth] [--look film|video|extended|rec709] [--pipeline auto|metal|cpu] [--step N]` plus the colour overrides `--csgen 4|5|6 --gamma S --gamut S --wb K --tint T --exposure EV --iso N --tc-contrast|--tc-saturation|--tc-midpoint|--tc-highlights|--tc-shadows|--tc-black-level|--tc-white-level F --tc-video-black 0|1 --highlight-recovery 0|1 --gamut-compression 0|1` (RGBA/RGBA16 frames on stdout — `"format": "rgba"` in the JSON header on stderr, with the PROCESSED size, the pipeline + GPU name and the step; `auto` decodes on the first Metal device the SDK offers, else the CPU; `--step N` emits every Nth frame and the header's `frames` is the emitted count; frames are pipelined up to 3 deep; `--zerocopy N [--mono]` hands frames over as N slots of global IOSurfaces — one RGBA8 surface per eye on the 2:1 canvas, filled by a Metal blit — reporting `F <frame> <slot>` / `K <gen>` (seek acknowledged) / `E` (end) on stdout and taking `R <slot>`, `S <frame> <gen>` (seek) and `P <flag> <value…>` (a colour override changed live, applied once the frames in flight are done) on stdin, surface IDs in the header; built from `braw_helper.cpp` (C++) + `braw_zc.mm` (Objective-C++, Metal + IOSurface)), `--gyro <file>` (interleaved float32 `[gx gy gz ax ay az]` on stdout), `--shutters <file> [--step N]` (per-frame shutter seconds), `--lut <file> <out.cube>` (the embedded post-3D-LUT), `--meta <file> [--frame N]` (every clip + frame metadata key), `--audio <file>` (RIFF WAV on stdout). Multi-track (URSA Cine Immersive) is auto-side-by-side without `--track`. | BlackmagicRawAPI, CoreFoundation, CoreServices |

## Building

```sh
./helpers/build_swift.sh   # mvhevc_encode, vt_denoise, apac_encode
./helpers/braw/build_braw.sh   # braw_helper (requires Blackmagic RAW SDK installed)
```

Output binaries go to `helpers/bin/`, which is `.gitignore`d. The
sources in `helpers/swift/` and `helpers/braw/` are committed; the
binaries are not.

### Keeping shipped helpers current

Because the binaries are not in git, a release bundle could carry a
helper built from older sources (it happened: the 2026-10-10 APAC fix).
Three guards now close that gap:

- Every Swift helper answers `--version` with `<name> <sha256 of its
  source>` — the build script compiles the source as `main.swift` next to
  a generated `version.swift` (`-D HELPER_VERSION_EMBEDDED`).
- `./helpers/build_swift.sh --check DIR` compares a directory's helpers
  against the current sources; `--install DIR` rebuilds and copies them.
  `scripts/release_mac.sh` always installs fresh helpers and refuses to
  continue on a mismatch.
- `crates/vr180-gui/build.rs` runs the build script on macOS whenever a
  helper source changes, so `helpers/bin/` (what the Dev bundles copy)
  stays current after a plain `cargo build`.

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

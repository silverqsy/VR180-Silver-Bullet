# Test fixtures

Raw GPMF (GoPro Metadata Format) byte streams extracted from real
.360 clips, used by `vr180-core::gyro` unit tests so we can validate
the parser without needing the multi-GB source files.

## Files

| File | Source | Size | Notes |
|---|---|---|---|
| `GS010172.gpmf` | `GS010172.360` (30s clip, GoPro Max) | 314 KB | 875 CORI + 875 IORI samples; CORI is firmware-stabilized (non-identity at t=0); IORI starts at exact identity. Used by `gpmf::tests::*` and `cori_iori::tests::*`. |

## Regenerating

To extract a GPMF fixture from any GoPro `.360` / `.mp4` file:

```sh
ffmpeg -loglevel error -i <input.360> -map 0:m:handler_name:'GoPro MET' \
    -c copy -f rawvideo <output>.gpmf
```

(Or `-map 0:3` for the GoPro Max — the GPMF stream is always
index 3 there, but the `gpmd` codec tag lookup we do in the Rust
pipeline is camera-model-agnostic.)

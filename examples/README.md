# Example configs for `vr180-render render --config`

These are the JSON schema exposed by Phase 0.9 (`vr180-render::config::ExportConfig`).

## Run

```sh
./target/release/vr180-render render --config examples/full_vision_pro.json
```

## Schema

`deny_unknown_fields` is on — typos in field names error out instead of
silently being ignored. Every field has an identity / off default, so a
minimal config can be just:

```json
{ "input": "in.360", "output": "out.mov" }
```

→ produces a 4K SBS HEVC with no color tools, no audio, libx265 encoder.

## Field reference

| Field | Type | Default | Meaning |
|---|---|---|---|
| `input`  | path | — required — | source `.360` |
| `output` | path | — required — | output `.mov` / `.mp4` |
| `eye_w` | u32 | 2048 | per-eye width; 2048 → 4K SBS, 4096 → 8K SBS |
| `frames` | u32 | 0 (= all) | limit number of frames (test-only) |
| `fps` | f32 \| null | null (= source) | output FPS override |
| `bitrate` | u32 | 12000 | HEVC kbps. ~12 Mbps at 4K, ~40 Mbps at 8K. |
| `encoder` | enum | `"auto"` | `"auto"` (VT on macOS, libx265 elsewhere), `"sw"`, `"vt"` |
| `hw_accel` | enum | `"auto"` | hardware **decode**: `"auto"`, `"sw"`, `"vt"` |
| `zero_copy` | bool | false | skip CPU-EAC-assemble (macOS only) |
| `zero_copy_encode` | bool | false | also skip readback (requires `zero_copy` + `encoder: "vt"`) |
| `cdl` | object | identity | `{ lift, gamma, gain, shadow, highlight }` |
| `grade` | object | identity | `{ temperature, tint, saturation }` |
| `sharpen` | object | off | `{ amount, sigma }` |
| `mid_detail` | object | off | `{ amount, sigma }` |
| `lut` | string \| null | null | `.cube` path or `"bundled"` for the GP-Log LUT |
| `lut_intensity` | f32 | 1.0 | LUT blend factor [0..1] |
| `apac_audio` | bool | false | embed APAC spatial audio (macOS only) |
| `apac_bitrate` | u32 | 384000 | APAC target bits/sec |
| `apmp` | bool | false | tag for Vision Pro VR180 recognition |

For the canonical schema definition (with code-level comments on each
field), see `crates/vr180-render/src/config.rs`.

## Spawn pattern (for the Python GUI)

```python
import json, subprocess
from pathlib import Path

cfg = {
    "input":  str(input_path),
    "output": str(output_path),
    "eye_w":  2048,
    "encoder": "vt",
    "zero_copy": True,
    "zero_copy_encode": True,
    "cdl": {
        "lift": user.lift, "gamma": user.gamma, "gain": user.gain,
        "shadow": user.shadow, "highlight": user.highlight,
    },
    # ... grade / sharpen / mid_detail / lut ...
    "apac_audio": user.apac, "apmp": True,
}
config_path = Path(tempfile.mkdtemp()) / "export.json"
config_path.write_text(json.dumps(cfg))

subprocess.run(
    ["./vr180-render", "render", "--config", str(config_path)],
    check=True,
)
```

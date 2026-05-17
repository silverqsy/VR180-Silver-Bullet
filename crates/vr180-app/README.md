# vr180-app

Tauri 2 desktop GUI for VR180 Silver Bullet Neo.

## What's here

This is a fresh scaffold — the goal of the first commit is to have a
runnable Tauri app that can open a `.360`, show probe info + a
single-frame preview, and surface the param panels we'll wire up over
the next several sessions.

| Area                     | Status                                  |
|--------------------------|-----------------------------------------|
| Tauri 2 shell + window   | ✅ — `cargo tauri dev` opens a window    |
| File picker (`Open .360`)| ✅ — via `tauri-plugin-dialog`           |
| Drag-drop a `.360`       | ✅ — via `tauri://drag-drop` event       |
| Probe clip → JSON        | ✅ — `probe_clip` command                |
| Multi-segment chain info | ✅ — `detect_segments` reports siblings  |
| SROT auto-detect display | ✅ — `lookup_srot_ms` command            |
| GEOC / KLNS probe        | ✅ — `probe_geoc` command                |
| Preview frame (first)    | ✅ — `extract_preview_frame` command     |
| Param sliders (UI only)  | ✅ — sliders update local readouts       |
| Param wiring to pipeline | ❌ — preview is identity-only for now    |
| Scrub timeline           | ❌ — slider exists but isn't time-mapped |
| Live playback @ 30 fps   | ❌                                       |
| Export button → pipeline | ❌ — UI in place; backend call to land   |
| Full Python parity       | ❌ — multi-session port                  |

## Run it

```sh
# From workspace root:
cargo tauri dev --manifest-path crates/vr180-app/Cargo.toml

# Or from this directory:
cd crates/vr180-app && cargo tauri dev
```

The app window should appear. Drop a `.360` on the window or click
**Open .360**, and the sidebar populates with the probe info + the
preview stage shows the first frame as a half-equirect SBS image.

`RUST_LOG=info cargo tauri dev` to see backend tracing in the
terminal.

## Layout

```text
crates/vr180-app/
├── Cargo.toml              — Tauri 2 + the rest of the workspace crates
├── build.rs                — tauri_build::build()
├── tauri.conf.json         — window + bundle config
├── capabilities/
│   └── default.json        — dialog / fs / scope permissions
├── icons/                  — placeholder PNG icons (TODO: real assets)
├── src/
│   ├── lib.rs              — tauri::Builder + run()
│   ├── main.rs             — thin bin wrapper
│   └── commands.rs         — every #[tauri::command] handler
└── ui/                     — frontend (vanilla HTML/JS/CSS)
    ├── index.html
    ├── style.css
    └── main.js
```

### Why vanilla HTML/JS for the frontend

Setting up React / Vue / Svelte adds a node-modules build step and
dev-server orchestration which slows down iteration when we're still
designing the layout. Vanilla HTML/JS:

- Renders directly out of `ui/` with zero build step.
- `cargo tauri dev` serves the static files and reloads on save.
- Easy to drop in React later: replace `main.js` with a Vite-built
  bundle and update `tauri.conf.json`'s `beforeDevCommand`.

This is the "ship it" choice. Plan to swap in React once the feature
set is stable and the JS file grows past ~1500 LOC.

## Adding a new Tauri command

1. Add a `#[tauri::command]` function in `src/commands.rs`.
   - Returns `Result<T, String>` (Tauri JSONifies the error string).
   - Args + return types `serde::{Serialize, Deserialize}` derivable.
2. Register the function name in the `tauri::generate_handler!`
   macro in `src/lib.rs`.
3. Call from JS:
   ```js
   const { invoke } = window.__TAURI__.core;
   const result = await invoke('your_command', { arg1: ..., arg2: ... });
   ```
   Arg names in the JS object **must match the Rust parameter names
   in snake_case**, otherwise the deserializer errors.

## Roadmap (planned, in order)

1. **Export button → backend pipeline call** with progress events.
   Reuse the same `export()` function the CLI uses.
2. **Scrub timeline → preview** — `time_s` parameter plumbed through
   the decoder seek, preview re-renders on scrub release.
3. **Param plumbing** — wire the sidebar sliders to the preview's
   ColorStackPlan + StabilizeParams + RsParams so users see their
   edits live (debounced ~250 ms).
4. **Live playback @ 30 fps** — decode + render + display loop.
   Either: render to canvas via `tauri.invoke` per-frame (simple, slow)
   or: emit PNG/JPEG over a websocket / `tauri_plugin_websocket`
   (faster). Likely the latter.
5. **Multi-segment chain UI** — show all detected siblings, allow
   user to start at a specific chapter.
6. **Trim range** — set in/out points on the timeline.
7. **APMP / APAC toggles** wired (UI already in place).
8. **Remaining Python parity** — edge mask, vignette, anaglyph
   preview, vision-pro modes, multi-camera, etc.

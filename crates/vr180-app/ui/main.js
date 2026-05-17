// VR180 Silver Bullet Neo — frontend.
//
// Vanilla JS (no framework yet — easy to swap in React/Vue later as
// features stabilize). Every backend call goes through `tauri.invoke`
// and lands in `crates/vr180-app/src/commands.rs`.

const { invoke } = window.__TAURI__.core;
const { open } = window.__TAURI__.dialog;

// ─── App state ─────────────────────────────────────────────────────

const state = {
  /** The currently loaded primary segment path. */
  path: null,
  /** Probe result from `probe_clip`. */
  clip: null,
  /** Last preview render in milliseconds (for status line). */
  lastPreviewMs: null,
  /** Debounce timer for preview refresh on slider change. */
  previewTimer: null,
};

// ─── Bootstrapping ─────────────────────────────────────────────────

(async function init() {
  try {
    const v = await invoke('version_info');
    document.getElementById('version-info').textContent = `v${v.app} · pipeline ${v.pipeline}`;
  } catch (e) {
    console.error('version_info failed:', e);
  }

  wireFilePicker();
  wireDragDrop();
  wireSliderReadouts();
  wirePreviewRefreshHooks();

  setStatus('Ready. Drop a .360 onto the window or click Open.');
})();

// ─── File picker ───────────────────────────────────────────────────

function wireFilePicker() {
  document.getElementById('btn-open').addEventListener('click', async () => {
    const selected = await open({
      multiple: false,
      filters: [
        { name: 'GoPro 360 / MP4', extensions: ['360', 'mp4', 'MP4', 'mov', 'MOV'] },
        { name: 'All files', extensions: ['*'] },
      ],
    });
    if (selected) await loadFile(selected);
  });
}

// Tauri v2 raises a drop event on the webview when a file is dropped on the window.
// We register through `window.__TAURI__.event.listen` if available.
async function wireDragDrop() {
  try {
    const { listen } = window.__TAURI__.event;
    await listen('tauri://drag-drop', async (event) => {
      const paths = event.payload?.paths || [];
      if (paths[0]) await loadFile(paths[0]);
    });
  } catch (e) {
    // Drag-drop event API not available; skip silently.
    console.warn('drag-drop wire failed:', e);
  }
}

async function loadFile(path) {
  setStatus(`Loading ${path}…`);
  state.path = path;
  try {
    const clip = await invoke('probe_clip', { path });
    state.clip = clip;
    renderSourceInfo(clip);
    document.getElementById('btn-export').disabled = false;
    document.getElementById('btn-refresh-preview').disabled = false;
    document.getElementById('rng-scrub').disabled = false;
    setStatus('Loaded.');
    statusDims(clip);
    statusChain(clip);
    await refreshSrot(path);
    await refreshPreview();
  } catch (e) {
    console.error('probe_clip failed:', e);
    setStatus(`Error: ${e}`);
  }
}

function renderSourceInfo(clip) {
  const el = document.getElementById('source-info');
  const fname = state.path.split('/').pop();
  const chain = clip.segments.length > 1
    ? `<div><strong>${clip.segments.length}-segment chain</strong> (${formatDuration(clip.chain_duration_sec)} total)</div>`
    : '';
  el.innerHTML = `
    <div class="filename" title="${state.path}">${fname}</div>
    <div class="kv">
      <span>Stream</span><span>${clip.width} × ${clip.height}</span>
      <span>FPS</span><span>${clip.fps.toFixed(3)}</span>
      <span>Duration</span><span>${formatDuration(clip.duration_sec)}</span>
      <span>Frames</span><span>${clip.frame_count.toLocaleString()}</span>
      <span>EAC tile</span><span>${clip.eac_tile_w} px</span>
    </div>
    ${chain}
  `;
}

async function refreshSrot(path) {
  try {
    const srotMs = await invoke('lookup_srot_ms', { path });
    const txt = srotMs ? `${srotMs.toFixed(3)} ms` : '—';
    document.getElementById('txt-srot').value = txt;
    document.getElementById('status-srot').textContent = `SROT ${txt}`;
  } catch (e) {
    console.warn('lookup_srot_ms failed:', e);
  }
}

// ─── Preview ───────────────────────────────────────────────────────

function wirePreviewRefreshHooks() {
  document.getElementById('btn-refresh-preview').addEventListener('click', refreshPreview);
  document.getElementById('rng-scrub').addEventListener('change', () => {
    // Debounced — for now we only render frame 0 anyway.
    if (state.previewTimer) clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(refreshPreview, 250);
  });
}

async function refreshPreview() {
  if (!state.path) return;
  setStatus('Rendering preview…');
  const t = performance.now();
  try {
    const eyeW = previewEyeW();
    const res = await invoke('extract_preview_frame', {
      req: {
        path: state.path,
        eye_w: eyeW,
        time_s: null,
        identity_only: true,
      },
    });
    showPreview(res);
    const ms = performance.now() - t;
    state.lastPreviewMs = ms;
    setStatus(`Preview rendered in ${ms.toFixed(0)} ms (backend ${res.elapsed_ms.toFixed(0)} ms).`);
  } catch (e) {
    console.error('extract_preview_frame failed:', e);
    setStatus(`Preview error: ${e}`);
  }
}

function showPreview(res) {
  const img = document.getElementById('preview-img');
  img.src = `data:image/png;base64,${res.png_base64}`;
  img.hidden = false;
  document.getElementById('preview-placeholder').style.display = 'none';
}

function previewEyeW() {
  // Lock preview to a moderate resolution for speed — full export uses
  // the user's "Eye width" selection but preview wants fast turnaround.
  return 768;
}

// ─── Slider value readouts ─────────────────────────────────────────

function wireSliderReadouts() {
  const pairs = [
    ['rng-temp', 'val-temp', v => (v >= 0 ? '+' : '') + v.toFixed(2)],
    ['rng-tint', 'val-tint', v => (v >= 0 ? '+' : '') + v.toFixed(2)],
    ['rng-sat', 'val-sat', v => v.toFixed(2)],
    ['rng-sharp', 'val-sharp', v => v.toFixed(2)],
  ];
  for (const [r, l, fmt] of pairs) {
    const rng = document.getElementById(r);
    const lbl = document.getElementById(l);
    const update = () => { lbl.textContent = fmt(parseFloat(rng.value)); };
    rng.addEventListener('input', update);
    update();
  }
}

// ─── Status / helpers ──────────────────────────────────────────────

function setStatus(msg) {
  document.getElementById('status-msg').textContent = msg;
}
function statusDims(clip) {
  document.getElementById('status-dims').textContent =
    `${clip.width}×${clip.height} @ ${clip.fps.toFixed(2)}`;
}
function statusChain(clip) {
  const n = clip.segments.length;
  document.getElementById('status-chain').textContent =
    n > 1 ? `${n} segments` : '1 segment';
}

function formatDuration(sec) {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  const r = (s - m * 60).toFixed(1);
  return `${m}:${r.padStart(4, '0')}`;
}

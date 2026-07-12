#!/usr/bin/env node
// Build (and MERGE) the auto-updater manifest `latest.json` for a release,
// signing the platform artifact with minisign in the same step.
//
// Produces a Tauri-updater-compatible manifest for VR180 Silver
// Bullet's native updater (crates/vr180-gui/src/updater.rs):
//
//   node scripts/make-latest-json.mjs --version 2.1.0 --notes-file RELEASE_NOTES.md
//
// Run once on EACH build machine (macOS + Windows); each run signs its own
// platform's artifact and merges its entry into release-staging/latest.json
// (copy the other machine's latest.json into release-staging/ first, or run
// on a synced checkout). Then upload to the GitHub release:
//
//   gh release upload v<version> release-staging/latest.json \
//       dist/VR180-Silver-Bullet-<version>-mac-arm64-app.zip           (mac)
//       dist/VR180-Silver-Bullet-<version>-windows-x64-setup.exe      (win)
//
// The app polls the STABLE URL
//   https://github.com/silverqsy/VR180-Silver-Bullet/releases/latest/download/latest.json
// so latest.json must be an asset on every release.
//
// Signing key: ~/.vr180-updater/vr180-updater.key (passwordless minisign
// key, generated with `minisign -G -W`; the key FILE is the secret — it
// lives outside the repo and must exist on both build machines). The
// matching pubkey is baked into updater.rs. Lose the key → no more
// updates, ever.
//
// Options:
//   --version <semver>      REQUIRED. Must match the release tag v<version>.
//   --notes <text> | --notes-file <path>   Release notes (markdown).
//   --artifact <path>       Override the artifact to sign+publish.
//   --out <dir>             Staging dir (default release-staging/).
//   --repo <owner/name>     Default silverqsy/VR180-Silver-Bullet.
//   --date <iso>            Override pub_date (default: now).

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
}
const die = (m) => { console.error("error: " + m); process.exit(1); };

const version = args.version || die("--version required (e.g. --version 2.1.0)");
const repo = args.repo || "silverqsy/VR180-Silver-Bullet";
const outDir = args.out || "release-staging";
const notes = args.notes
  ?? (args["notes-file"] ? fs.readFileSync(args["notes-file"], "utf8") : "");
const pubDate = args.date || new Date().toISOString();

// Platform key must match updater.rs PLATFORM_KEY.
const platformKey =
  process.platform === "darwin"
    ? "darwin-aarch64"
    : process.platform === "win32"
      ? "windows-x86_64"
      : die(`unsupported build platform ${process.platform}`);

// Default artifact per platform (the UPDATE vehicle, not the fresh-install
// one): mac = zip of the notarized .app; win = the Inno setup.exe.
const defaultArtifact =
  process.platform === "darwin"
    ? `dist/VR180-Silver-Bullet-${version}-mac-arm64-app.zip`
    : `dist/VR180-Silver-Bullet-${version}-windows-x64-setup.exe`;
const artifact = args.artifact || defaultArtifact;
if (!fs.existsSync(artifact)) {
  die(`artifact not found: ${artifact}
  mac: build + sign + notarize the .app, then:  ditto -c -k --keepParent "dist/VR180 Silver Bullet.app" "${defaultArtifact}"
  win: ISCC.exe installer\\windows.iss`);
}

// ── Sign with minisign (passwordless key → no prompt) ──
const keyPath = path.join(os.homedir(), ".vr180-updater", "vr180-updater.key");
if (!fs.existsSync(keyPath)) die(`signing key not found at ${keyPath}`);
const sigPath = artifact + ".minisig";
fs.rmSync(sigPath, { force: true });
execFileSync("minisign", ["-S", "-s", keyPath, "-m", artifact], { stdio: "inherit" });
const signature = Buffer.from(fs.readFileSync(sigPath, "utf8"), "utf8").toString("base64");

// ── Build / merge the manifest ──
fs.mkdirSync(outDir, { recursive: true });
const manifestPath = path.join(outDir, "latest.json");

// RESPIN SAFETY: the other platform's entry may exist only in the
// RELEASE's latest.json (uploaded from the other machine). Sync it down
// before merging so a regenerate + --clobber upload can't drop it.
// Best-effort — a missing release/tag/gh just means a fresh manifest.
try {
  execFileSync("gh", [
    "release", "download", `v${version}`,
    "--pattern", "latest.json", "--clobber", "--dir", outDir,
  ], { stdio: "pipe" });
  console.log(`✓ synced existing latest.json from release v${version}`);
} catch { /* no release yet / no gh — start fresh */ }
let manifest = { version, notes, pub_date: pubDate, platforms: {} };
if (fs.existsSync(manifestPath)) {
  try {
    const prev = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    if (prev.version === version) {
      // Same release built on the other machine — keep its entries.
      manifest = prev;
      manifest.notes = notes || prev.notes;
      manifest.pub_date = pubDate;
    }
  } catch { /* corrupt staging manifest — rebuild from scratch */ }
}
manifest.platforms[platformKey] = {
  signature,
  url: `https://github.com/${repo}/releases/download/v${version}/${path.basename(artifact)}`,
};
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");

// Stage a copy of the artifact next to the manifest so the upload step
// grabs everything from one folder.
fs.copyFileSync(artifact, path.join(outDir, path.basename(artifact)));

console.log(`✓ signed ${path.basename(artifact)} (${platformKey})`);
console.log(`✓ ${manifestPath}:`);
console.log(`  version   ${manifest.version}`);
console.log(`  platforms ${Object.keys(manifest.platforms).join(", ")}`);
console.log(`\nUpload:  gh release upload v${version} ${manifestPath} ${path.join(outDir, path.basename(artifact))}`);

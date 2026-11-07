#!/usr/bin/env bash
# Cut the macOS release bundle of VR180 Silver Bullet — the recipe that used
# to live in notes, as one script, with the helper-staleness guard built in.
#
#   scripts/release_mac.sh                build + bundle + sign + verify
#   scripts/release_mac.sh --notarize     … then notarize, staple, zip, DMG
#   scripts/release_mac.sh --check-only   just verify dist/ (helpers current, signature valid)
#   scripts/release_mac.sh --notarize-only   skip the build: verify the existing bundle, then
#                                            notarize / staple / zip / DMG (a rerun after a
#                                            notarization-side failure, e.g. an unsigned agreement)
#
# Steps (see CLAUDE.md "Release (macOS)"):
#   1. cargo build --release -p vr180-gui   (build.rs rebuilds helpers/bin on source change)
#   2. helpers/build_swift.sh --install     ALWAYS rebuilds the 3 Swift helpers from source
#      and copies them next to the app binary — the release can no longer ship a
#      helper built from older sources (every helper answers `--version` with its
#      source hash; `--check` compares it, and this script refuses to continue
#      on a mismatch).
#   2b. helpers/braw/build_braw.sh     rebuilds braw_helper against the installed Blackmagic
#      RAW SDK and copies it next to the app binary. It is signed with the
#      disable-library-validation entitlement (helpers/braw/braw_helper.entitlements):
#      it loads Blackmagic's own framework from the SDK install, signed by another team,
#      which the hardened runtime otherwise refuses ("different Team IDs") — verified.
#   3. copy vr180-gui, bump Info.plist to the Cargo version
#   4. dylibbundler (fresh binary → /opt/homebrew refs → ~14 dylibs into Frameworks/)
#   5. sign inside-out with Developer ID (dylibs → helpers → binary → bundle),
#      `codesign --verify --deep --strict`, helper check, `--version` smoke
#   6. (--notarize) ditto zip → notarytool (keychain profile vr180-notary) → staple →
#      update zip → DMG with /Applications link → sign/notarize/staple DMG → spctl.
#      The updater manifest + GitHub upload stay manual (printed at the end).
#
# Env: VR180_SIGN_IDENTITY (default: the Developer ID below), VR180_NOTARY_PROFILE
# (default vr180-notary). Never type credentials here — the keychain holds them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VER="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
APP="$ROOT/dist/VR180 Silver Bullet.app"
MACOS="$APP/Contents/MacOS"
FRAMEWORKS="$APP/Contents/Frameworks"
PLIST="$APP/Contents/Info.plist"
IDENTITY="${VR180_SIGN_IDENTITY:-Developer ID Application: Yuanming Ni (655S25MMH5)}"
PROFILE="${VR180_NOTARY_PROFILE:-vr180-notary}"
MODE="${1:-build}"
HELPERS=(mvhevc_encode vt_denoise apac_encode)
BRAW_HELPER="$MACOS/braw_helper"
BRAW_ENTITLEMENTS="$ROOT/helpers/braw/braw_helper.entitlements"
BRAW_SDK_LIBS="/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac/Libraries"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { echo "✗ $*" >&2; exit 1; }

[[ "$(uname)" == "Darwin" ]] || die "macOS only"
[[ -d "$APP" && -f "$PLIST" ]] || die "dist skeleton missing: $APP (Info.plist + assets/icon.icns expected)"

verify_bundle() {
    say "verify: helpers built from the current sources"
    bash helpers/build_swift.sh --check "$MACOS" || die "a bundled helper is stale — refusing to ship it"
    say "verify: code signature"
    codesign --verify --deep --strict --verbose=2 "$APP" || die "signature invalid"
    say "verify: smoke (helpers answer --version from inside the bundle)"
    for h in "${HELPERS[@]}"; do "$MACOS/$h" --version; done
    say "verify: braw_helper (built from the current sources, entitled, starts with the SDK framework)"
    local stale braw_out
    [[ -x "$BRAW_HELPER" ]] || die "braw_helper missing from the bundle"
    stale="$(find helpers/braw -type f \( -name '*.cpp' -o -name '*.mm' -o -name '*.h' -o -name '*.sh' \) -newer "$BRAW_HELPER" | head -1)"
    [[ -z "$stale" ]] || die "bundled braw_helper is older than $stale — rebuild it (helpers/braw/build_braw.sh)"
    codesign -d --entitlements - "$BRAW_HELPER" 2>&1 | grep -q 'disable-library-validation' \
        || die "braw_helper lacks the disable-library-validation entitlement (it could not load Blackmagic's framework)"
    braw_out="$("$BRAW_HELPER" 2>&1 || true)"
    grep -q '^Usage:' <<<"$braw_out" || die "braw_helper did not start: $braw_out"
    echo "bundle version: $(plutil -extract CFBundleShortVersionString raw "$PLIST")"
}

if [[ "$MODE" == "--check-only" ]]; then verify_bundle; exit 0; fi
if [[ "$MODE" == "--notarize-only" ]]; then
    [[ "$(plutil -extract CFBundleShortVersionString raw "$PLIST")" == "$VER" ]] || die "bundle is not $VER — run a full build first"
    verify_bundle
fi

if [[ "$MODE" != "--notarize-only" ]]; then
say "1. cargo build --release ($VER)"
cargo build --release -p vr180-gui

say "2. Swift helpers: rebuild from source and install into the bundle"
bash helpers/build_swift.sh --install "$MACOS"

say "2b. BRAW helper: rebuild against the installed Blackmagic RAW SDK and install into the bundle"
[[ -d "$BRAW_SDK_LIBS/BlackmagicRawAPI.framework" ]] \
    || die "Blackmagic RAW SDK not installed ($BRAW_SDK_LIBS) — the release ships braw_helper, so the build Mac needs the SDK (see helpers/braw/build_braw.sh)"
bash helpers/braw/build_braw.sh >/dev/null || die "braw_helper build failed"
cp -f helpers/bin/braw_helper "$BRAW_HELPER"

say "3. app binary + Info.plist $VER"
cp -f target/release/vr180-gui "$MACOS/vr180-gui"
plutil -replace CFBundleShortVersionString -string "$VER" "$PLIST"
plutil -replace CFBundleVersion -string "$VER" "$PLIST"

say "4. dylibbundler"
command -v dylibbundler >/dev/null || die "dylibbundler not installed (brew install dylibbundler)"
dylibbundler -od -b -x "$MACOS/vr180-gui" -d "$FRAMEWORKS" -p @executable_path/../Frameworks/ >/dev/null
echo "bundled $(ls "$FRAMEWORKS" | wc -l | tr -d ' ') dylibs"
if otool -L "$MACOS/vr180-gui" | grep -q /opt/homebrew; then die "binary still references /opt/homebrew"; fi

say "5. sign inside-out: $IDENTITY"
for f in "$FRAMEWORKS"/*.dylib; do codesign --force --options runtime --timestamp -s "$IDENTITY" "$f"; done
for h in "${HELPERS[@]}"; do codesign --force --options runtime --timestamp -s "$IDENTITY" "$MACOS/$h"; done
# braw_helper loads Blackmagic's framework from the SDK install, signed by another team, which the
# hardened runtime refuses ("different Team IDs") unless library validation is disabled for it.
codesign --force --options runtime --timestamp --entitlements "$BRAW_ENTITLEMENTS" -s "$IDENTITY" "$BRAW_HELPER"
codesign --force --options runtime --timestamp -s "$IDENTITY" "$MACOS/vr180-gui"
codesign --force --options runtime --timestamp -s "$IDENTITY" "$APP"
verify_bundle
fi  # (build + sign)

if [[ "$MODE" != "--notarize" && "$MODE" != "--notarize-only" ]]; then
    say "done (not notarized). Next: scripts/release_mac.sh --notarize"
    exit 0
fi

say "6. notarize + staple the app"
STAGE="$ROOT/dist/release-$VER"
rm -rf "$STAGE"; mkdir -p "$STAGE/dmgroot"
ditto -c -k --keepParent "$APP" "$STAGE/app-for-notary.zip"
xcrun notarytool submit "$STAGE/app-for-notary.zip" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP"
spctl -a -vv "$APP"
UPDATE_ZIP="$ROOT/dist/VR180-Silver-Bullet-$VER-mac-arm64-app.zip"
rm -f "$UPDATE_ZIP"; ditto -c -k --keepParent "$APP" "$UPDATE_ZIP"
echo "update asset: $UPDATE_ZIP"

say "7. DMG"
DMG="$ROOT/dist/VR180-Silver-Bullet-$VER-mac-arm64.dmg"
cp -R "$APP" "$STAGE/dmgroot/"
ln -s /Applications "$STAGE/dmgroot/Applications"
rm -f "$DMG"
hdiutil create -volname "VR180 Silver Bullet $VER" -srcfolder "$STAGE/dmgroot" -ov -format UDZO "$DMG" >/dev/null
codesign --force --timestamp -s "$IDENTITY" "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"
spctl -a -vv -t open --context context:primary-signature "$DMG"
rm -rf "$STAGE"

say "done. Remaining manual steps"
cat <<EOM
  node scripts/make-latest-json.mjs --version $VER --notes-file <release-notes.md>
  gh release create v$VER (or upload --clobber): "$UPDATE_ZIP", latest.json, "$DMG"
  (NOT the .minisig: its contents are already embedded in latest.json's "signature" field, which is what the updater verifies)
EOM

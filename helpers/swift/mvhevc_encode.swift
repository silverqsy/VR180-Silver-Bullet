// mvhevc_encode — VideoToolbox MV-HEVC direct encoder helper
//
// Reads raw BGR48LE side-by-side stereo frames from stdin and writes a
// spatial-video .mov with MV-HEVC (Main10 profile) + all the VideoToolbox
// spatial metadata baked into the format description (hero eye, stereo
// baseline, horizontal FOV, half-equirect projection). The output is
// directly consumable by Apple Vision Pro, the Files.app spatial preview,
// and the Photos.app spatial video lane — no post-encode transcode and no
// external `spatial` CLI dependency.
//
// Usage (pipe mode only):
//   mvhevc_encode --width <per-eye-w> --height <h> --fps <fps>
//                 --bitrate <bps, e.g. 100000000>
//                 --baseline-mm <63>
//                 --hfov-deg <180>
//                 --hero <left|right>
//                 --output <file.mov>
//
// Stdin:
//   Each frame = 2 × per_eye_w × h × 6 bytes (uint16 BGR, little-endian).
//   Left half comes first, then right half (host SBS layout). Helper splits
//   internally. EOF terminates the encode and finalizes the file.
//
// Build:
//   swiftc -O -o mvhevc_encode mvhevc_encode.swift \
//       -framework AVFoundation -framework VideoToolbox \
//       -framework CoreMedia -framework CoreVideo -framework Accelerate

import Foundation
import AVFoundation
import VideoToolbox
import CoreMedia
import CoreVideo

// MARK: - Argument parsing

var perEyeW: Int = 0
var perEyeH: Int = 0
var fps: Double = 29.97
var bitrateBps: Int = 100_000_000  // 100 Mbps default
var baselineMm: Double = 65.0
var hfovDeg: Double = 180.0
var heroEye: String = "left"
var outputPath: String = ""

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "--width":      argIdx += 1; perEyeW = Int(args[argIdx]) ?? 0
    case "--height":     argIdx += 1; perEyeH = Int(args[argIdx]) ?? 0
    case "--fps":        argIdx += 1; fps = Double(args[argIdx]) ?? 29.97
    case "--bitrate":    argIdx += 1; bitrateBps = Int(args[argIdx]) ?? 100_000_000
    case "--baseline-mm":argIdx += 1; baselineMm = Double(args[argIdx]) ?? 65.0
    case "--hfov-deg":   argIdx += 1; hfovDeg = Double(args[argIdx]) ?? 180.0
    case "--hero":       argIdx += 1; heroEye = args[argIdx]
    case "--output":     argIdx += 1; outputPath = args[argIdx]
    default:
        fputs("WARN: unknown argument \(arg)\n", stderr)
    }
    argIdx += 1
}

guard perEyeW > 0, perEyeH > 0, !outputPath.isEmpty else {
    fputs("Usage: mvhevc_encode --width W --height H --fps F --bitrate B --output FILE\n", stderr)
    fputs("       [--baseline-mm 65] [--hfov-deg 180] [--hero left|right]\n", stderr)
    exit(1)
}

// MARK: - Platform checks

guard VTIsStereoMVHEVCEncodeSupported() else {
    fputs("ERROR: VTIsStereoMVHEVCEncodeSupported() == false. MV-HEVC encoding requires Apple Silicon + macOS 14+.\n", stderr)
    exit(1)
}

fputs("mvhevc_encode: per-eye=\(perEyeW)x\(perEyeH) fps=\(fps) bitrate=\(bitrateBps)bps baseline=\(baselineMm)mm hfov=\(hfovDeg)° hero=\(heroEye)\n", stderr)
fputs("mvhevc_encode: output = \(outputPath)\n", stderr)

// MARK: - I/O pixel buffer setup

let frameBytesPerEye = perEyeW * perEyeH * 6  // BGR48LE
let sbsFrameBytes = frameBytesPerEye * 2

let bufAttrs: [String: Any] = [
    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
]

// Intermediate: 16-bit RGBA host-LE for BGR48 → 10-bit YUV transfer.
// Same trick we use in vt_denoise.swift — 64RGBALE is the only 16-bit
// host-endian RGB format VTPixelTransferSession supports as a source to
// the VT video pipeline.
let rgbaFmt: OSType = kCVPixelFormatType_64RGBALE

var leftRGBA: CVPixelBuffer?
var rightRGBA: CVPixelBuffer?
CVPixelBufferCreate(kCFAllocatorDefault, perEyeW, perEyeH, rgbaFmt, bufAttrs as CFDictionary, &leftRGBA)
CVPixelBufferCreate(kCFAllocatorDefault, perEyeW, perEyeH, rgbaFmt, bufAttrs as CFDictionary, &rightRGBA)
guard let leftRGBA, let rightRGBA else {
    fputs("ERROR: can't create 64RGBALE I/O buffers\n", stderr); exit(1)
}

// Color tagging for VTPixelTransferSession → it needs to know the source
// colorspace to do RGB→YCbCr correctly. BT.709 matches HEVC main10 from
// GoPro MAX, which is the typical upstream source.
for pb in [leftRGBA, rightRGBA] {
    CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey,
                          kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey,
                          kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
}

// Encoder input: 10-bit YUV 4:2:0 bi-planar (HEVC main10 native format).
//
// The encoder is asynchronous — it retains the pixel buffers we submit
// until it's done with them (could be several frames later due to GOP /
// reference window). If we reuse a single pair of YUV buffers across
// frames, we'll corrupt the encoder's references and get
// kVTPixelTransferNotSupportedErr (-12905) on the next transfer.
//
// Solution: allocate each frame's YUV buffers from a CVPixelBufferPool,
// which auto-recycles buffers once the encoder releases them. The pool
// minimum-count is sized to cover the encoder's reference window plus a
// margin for in-flight frames.
let yuvFmt: OSType = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

let yuvPoolAttrs: [String: Any] = [
    kCVPixelBufferPoolMinimumBufferCountKey as String: 16   // plenty for HEVC GOP window
]
let yuvBufAttrs: [String: Any] = [
    kCVPixelBufferWidthKey as String: perEyeW,
    kCVPixelBufferHeightKey as String: perEyeH,
    kCVPixelBufferPixelFormatTypeKey as String: yuvFmt,
    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
]
var yuvPool: CVPixelBufferPool?
CVPixelBufferPoolCreate(kCFAllocatorDefault, yuvPoolAttrs as CFDictionary,
                        yuvBufAttrs as CFDictionary, &yuvPool)
guard let yuvPool else {
    fputs("ERROR: can't create 10-bit YUV pool\n", stderr); exit(1)
}

// Helper to fetch a fresh YUV buffer from the pool + tag color metadata
func poolYUVBuffer() -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, yuvPool, &pb)
    if status != 0 || pb == nil {
        fputs("WARN: yuvPool.createPixelBuffer failed (\(status))\n", stderr)
        return nil
    }
    CVBufferSetAttachment(pb!, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(pb!, kCVImageBufferTransferFunctionKey,
                          kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(pb!, kCVImageBufferYCbCrMatrixKey,
                          kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    return pb
}

// Pixel transfer session (BGRA16 → YUV10). HW-accelerated on Apple Silicon.
var transferSession: VTPixelTransferSession?
VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
guard let transferSession else {
    fputs("ERROR: can't create VTPixelTransferSession\n", stderr); exit(1)
}

// MARK: - VTCompressionSession setup

var compressionSession: VTCompressionSession?
let srcPBAttrs: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: yuvFmt,
    kCVPixelBufferWidthKey as String: perEyeW,
    kCVPixelBufferHeightKey as String: perEyeH,
    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
]

let createStatus = VTCompressionSessionCreate(
    allocator: kCFAllocatorDefault,
    width: Int32(perEyeW),
    height: Int32(perEyeH),
    codecType: kCMVideoCodecType_HEVC,
    encoderSpecification: nil,
    imageBufferAttributes: srcPBAttrs as CFDictionary,
    compressedDataAllocator: nil,
    outputCallback: nil,          // use per-frame block handler instead
    refcon: nil,
    compressionSessionOut: &compressionSession
)
guard createStatus == 0, let compressionSession else {
    fputs("ERROR: VTCompressionSessionCreate failed (\(createStatus))\n", stderr); exit(1)
}

// Helper to set a property + log failures
@discardableResult
func setProp(_ key: CFString, _ value: CFTypeRef) -> OSStatus {
    let s = VTSessionSetProperty(compressionSession, key: key, value: value)
    if s != 0 {
        fputs("WARN: VTSessionSetProperty('\(key as String)') failed (\(s))\n", stderr)
    }
    return s
}

// --- MV-HEVC multi-view layer configuration ---
setProp(kVTCompressionPropertyKey_MVHEVCVideoLayerIDs, [0, 1] as CFArray)
setProp(kVTCompressionPropertyKey_MVHEVCViewIDs, [0, 1] as CFArray)
setProp(kVTCompressionPropertyKey_MVHEVCLeftAndRightViewIDs, [0, 1] as CFArray)

// --- Per-view presence flags ---
setProp(kVTCompressionPropertyKey_HasLeftStereoEyeView, kCFBooleanTrue)
setProp(kVTCompressionPropertyKey_HasRightStereoEyeView, kCFBooleanTrue)

// --- Hero eye (used for 2D fallback playback) ---
if heroEye == "right" {
    setProp(kVTCompressionPropertyKey_HeroEye, kVTHeroEye_Right)
} else {
    setProp(kVTCompressionPropertyKey_HeroEye, kVTHeroEye_Left)
}

// --- Stereo baseline (in micrometers per the header spec: "uint32") ---
let baselineMicrometers = Int(baselineMm * 1000.0 + 0.5)
setProp(kVTCompressionPropertyKey_StereoCameraBaseline, NSNumber(value: baselineMicrometers))

// --- Horizontal field of view (in millidegrees per the header spec) ---
let hfovMillideg = Int(hfovDeg * 1000.0 + 0.5)
setProp(kVTCompressionPropertyKey_HorizontalFieldOfView, NSNumber(value: hfovMillideg))

// --- Horizontal disparity adjustment (0 = centered) ---
setProp(kVTCompressionPropertyKey_HorizontalDisparityAdjustment, NSNumber(value: 0))

// --- Projection kind — HalfEquirectangular for VR180 ---
if #available(macOS 26.0, *) {
    setProp(kVTCompressionPropertyKey_ProjectionKind, kVTProjectionKind_HalfEquirectangular)
} else {
    fputs("WARN: ProjectionKind=halfEquirect requires macOS 26+; skipping\n", stderr)
}

// --- Codec profile and bitrate ---
setProp(kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_HEVC_Main10_AutoLevel)
setProp(kVTCompressionPropertyKey_AverageBitRate, NSNumber(value: bitrateBps))

// --- Encoder latency controls (CRITICAL) ---
//
// VideoToolbox's HEVC encoder, with default settings + AllowFrameReordering,
// will happily buffer an ENTIRE multi-GOP window of input pixel buffers
// before it emits any output samples. This is fine for pure offline batch
// encoding of a small clip, but for a pipe-fed streaming encode it means
// the upstream pool grows without bound (the retained input buffers prevent
// recycling) and memory explodes — on a 300-frame 2K test the encoder held
// every frame in memory simultaneously (~14 GB RSS).
//
// The fix is two-pronged:
//   1. `MaxFrameDelayCount` — hard cap on how many source frames the
//      encoder may hold before emitting its first output. This directly
//      bounds the encoder's internal pipeline depth.
//   2. `RealTime = true` — a strong hint that emissions should be as
//      prompt as possible (the hardware encoder stops doing GOP-wide
//      lookahead).
//
// We keep `AllowFrameReordering = true` so B-frames are still available
// for compression efficiency, but the reorder window is capped to
// `MaxFrameDelayCount` frames — so the whole pipeline (submit → encode →
// output callback → AVAssetWriter append) never accumulates more than
// ~4 frames of input pixel buffers. Combined with the in-flight semaphore
// below, memory is deterministically bounded.
setProp(kVTCompressionPropertyKey_RealTime, kCFBooleanTrue)
setProp(kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanTrue)

// --- Color metadata on the bitstream ---
setProp(kVTCompressionPropertyKey_ColorPrimaries, kCVImageBufferColorPrimaries_ITU_R_709_2)
setProp(kVTCompressionPropertyKey_TransferFunction, kCVImageBufferTransferFunction_ITU_R_709_2)
setProp(kVTCompressionPropertyKey_YCbCrMatrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2)

// --- Keyframe interval / GOP (~1 second for more frequent sync points) ---
let gopFrames = max(1, Int(fps * 1.0 + 0.5))
setProp(kVTCompressionPropertyKey_MaxKeyFrameInterval, NSNumber(value: gopFrames))

// --- Prepare session ---
let prepStatus = VTCompressionSessionPrepareToEncodeFrames(compressionSession)
if prepStatus != 0 {
    fputs("ERROR: VTCompressionSessionPrepareToEncodeFrames failed (\(prepStatus))\n", stderr)
    exit(1)
}

// MARK: - AVAssetWriter setup (lazy-init on first encoded sample buffer)

let outputURL = URL(fileURLWithPath: outputPath)
try? FileManager.default.removeItem(at: outputURL)

var writer: AVAssetWriter? = nil
var writerInput: AVAssetWriterInput? = nil
var writerReady: Bool = false
let writerInitLock = NSLock()
var writerInitFailed = false

func ensureWriter(formatDesc: CMFormatDescription, firstPTS: CMTime) -> Bool {
    writerInitLock.lock()
    defer { writerInitLock.unlock() }
    if writerReady { return true }
    if writerInitFailed { return false }

    do {
        let w = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video,
                                       outputSettings: nil,
                                       sourceFormatHint: formatDesc)
        input.expectsMediaDataInRealTime = false
        guard w.canAdd(input) else {
            fputs("ERROR: writer cannot add input\n", stderr)
            writerInitFailed = true
            return false
        }
        w.add(input)
        guard w.startWriting() else {
            fputs("ERROR: writer.startWriting failed: \(w.error?.localizedDescription ?? "unknown")\n", stderr)
            writerInitFailed = true
            return false
        }
        w.startSession(atSourceTime: firstPTS)
        writer = w
        writerInput = input
        writerReady = true
        fputs("mvhevc_encode: writer initialized\n", stderr)
        return true
    } catch {
        fputs("ERROR: AVAssetWriter init threw: \(error)\n", stderr)
        writerInitFailed = true
        return false
    }
}

// MARK: - BGR48LE → 64RGBALE fill helper
//
// IMPORTANT: the upstream Python render loop produces a single
// SBS frame as a numpy `(H, 2*perEyeW, 3)` uint16 array, i.e. the
// memory layout is ROW-INTERLEAVED:
//
//     row 0: [L0_0, L0_1, ..., L0_{W-1}, R0_0, R0_1, ..., R0_{W-1}]
//     row 1: [L1_0, L1_1, ..., L1_{W-1}, R1_0, R1_1, ..., R1_{W-1}]
//     ...
//
// where L = left eye, R = right eye, W = perEyeW, and each sample
// is 3 uint16 values (BGR). The row stride in uint16 samples is
// `2 * perEyeW * 3`, not `perEyeW * 3`.
//
// The left eye starts at pointer offset 0 (beginning of each row),
// the right eye starts at offset `perEyeW * 3` (half a row in).
// Both eyes walk forward using the FULL SBS row stride to step to
// the next row.
//
// A previous version of this helper assumed a CONTIGUOUS layout
// (left eye packed, then right eye packed) and split the SBS byte
// buffer in half with `base.advanced(by: perEyeW * perEyeH * 3)`.
// That turned out to actually read the TOP half of the frame as
// "left" and the BOTTOM half as "right", producing a visibly
// doubled / ghosted output with the city skyline split across
// the eyes (traffic lights multiplied, buildings with doubled
// rooflines, etc.). Fixed 2026-04-09.

let sbsRowStrideUInt16 = 2 * perEyeW * 3  // full_w * channels

func fillRGBAFromBGR48(_ pb: CVPixelBuffer, _ raw: UnsafePointer<UInt16>) {
    CVPixelBufferLockBaseAddress(pb, [])
    let base = CVPixelBufferGetBaseAddress(pb)!
    let stride = CVPixelBufferGetBytesPerRow(pb)
    for row in 0..<perEyeH {
        let dst = base.advanced(by: row * stride).bindMemory(to: UInt16.self, capacity: perEyeW * 4)
        // Step through the source using the SBS row stride, NOT the
        // per-eye row stride — see the comment block above.
        let srcRow = raw.advanced(by: row * sbsRowStrideUInt16)
        for col in 0..<perEyeW {
            let si = col * 3
            let b = srcRow[si + 0]
            let g = srcRow[si + 1]
            let r = srcRow[si + 2]
            let di = col * 4
            dst[di + 0] = r
            dst[di + 1] = g
            dst[di + 2] = b
            dst[di + 3] = 65535
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, [])
}

// MARK: - Tag collections for left/right views
//
// We use Swift-native CMTag builders:
//   CMTag.stereoView(.leftEye)   — CMTypedTag<CMStereoViewComponents>, value = kCMStereoView_LeftEye
//   CMTag.stereoView(.rightEye)  — same, right eye
//   CMTag.videoLayerID(0) / (1)  — CMTypedTag<Int64>, matching
//                                  kVTCompressionPropertyKey_MVHEVCVideoLayerIDs = [0, 1]
//
// The tags + pixel buffers go into CMTaggedBuffer instances which are then
// passed as an array to VTCompressionSessionEncodeMultiImageFrame (the Swift-
// refined signature — no manual CMTagCollection / CMTaggedBufferGroup
// construction needed).

let leftEyeTag = CMTag.stereoView(.leftEye)
let rightEyeTag = CMTag.stereoView(.rightEye)
let layer0Tag = CMTag.videoLayerID(0)
let layer1Tag = CMTag.videoLayerID(1)

// MARK: - Main encode loop

let stdinFD = FileHandle.standardInput
let fpsTimescale: Int32 = 30000
let frameDurationValue = Int64(Double(fpsTimescale) / fps + 0.5)
let frameDuration = CMTimeMake(value: frameDurationValue, timescale: fpsTimescale)

// Pre-allocated scratch data for reading SBS frame + splitting halves
var sbsBuffer = Data(count: sbsFrameBytes)

// In-flight throttle. VTCompressionSessionEncodeMultiImageFrame is async —
// the block handler fires some frames LATER (HEVC with AllowFrameReordering
// typically holds 2-5 frames in its reference window before emitting). Without
// a cap on submissions-minus-drains, the main thread races ahead, pulls fresh
// YUV buffers out of the pool for every new frame, and memory grows without
// bound until the system starts swapping. Symptom the user saw: "stuck in
// the middle, huge RAM". The fix is a DispatchSemaphore that bounds the
// number of frames between submission and drain.
//
// Sizing: 6 gives HEVC room to keep its B-frame reference window full
// (typically 4-5 frames) with a small margin for the async callback, and
// caps YUV memory at ~6 × 2 eyes × frame_size ≈ 300-800 MB on an 8K export.
// This also naturally back-pressures the Python side: when the semaphore
// blocks, stdin reads block, and the upstream pipe fills until Python's
// encode_proc.stdin.write() blocks too.
let maxInFlight = 6
let inFlightSem = DispatchSemaphore(value: maxInFlight)

var pts = CMTime.zero
var framesEncoded = 0
var appendErrors = 0
// `autoreleasepool { }` uses an escaping closure that can't `break` the
// outer while loop — we use explicit flags to signal outer-loop exit.
var eofReached = false
var loopShouldBreak = false
// Diagnostic: how many times has the output callback fired?
// If this stays at 0 until after the encode loop exits, the encoder
// is batch-buffering every submitted frame before emitting any output.
// Access from block handler requires a lock (simple atomic counter)
let handlerFiredLock = NSLock()
var handlersFired = 0
var firstHandlerAt: Date? = nil
let encodeStartTime = Date()

// The per-iteration autoreleasepool here is CRITICAL. Without it, every
// CVPixelBuffer, CMTaggedBuffer, CMSampleBuffer, Data, etc. we touch ends
// up on Swift's bridged NSAutoreleasePool and only gets drained when the
// runloop returns — which never happens in a tight stdin-read loop. The
// result is that EVERY submitted frame's input YUV buffers stay retained
// for the duration of the encode and memory grows linearly with frame
// count (measured 8.7 GB for 180 2K frames = every frame still live).
// The semaphore back-pressure was bounding the encoder's internal queue
// correctly — the leak was on the autorelease side, not the encoder side.
// Wrapping the loop body in `autoreleasepool` forces the pool to drain at
// the end of each iteration, so temporaries get freed immediately.
while true { autoreleasepool {
    // Read one SBS frame from stdin
    let chunk = stdinFD.readData(ofLength: sbsFrameBytes)
    if chunk.count < sbsFrameBytes {
        // Set flag to exit outer loop after this pool drains
        eofReached = true
        return
    }

    // Fill RGBA buffers from the left/right halves of the SBS frame.
    // The layout is ROW-INTERLEAVED (see the big comment on
    // fillRGBAFromBGR48 for details):
    //   - Left eye starts at the beginning of every row (offset 0 within the row)
    //   - Right eye starts `perEyeW * 3` uint16 samples into every row
    // Both walk forward using the full SBS row stride.
    chunk.withUnsafeBytes { rawPtr in
        let base = rawPtr.bindMemory(to: UInt16.self).baseAddress!
        fillRGBAFromBGR48(leftRGBA,  base)
        fillRGBAFromBGR48(rightRGBA, base.advanced(by: perEyeW * 3))
    }

    // ── Acquire an in-flight slot. Blocks (and therefore blocks the stdin
    // read on the next iteration, propagating back-pressure to Python) when
    // `maxInFlight` frames are already in the encoder pipeline waiting to
    // drain. The matching signal lives at the end of the block handler
    // below; any error path before the submit must signal here too.
    inFlightSem.wait()

    // Get fresh YUV buffers from the pool (auto-recycled once encoder
    // releases the previous frame's references)
    guard let leftYUV = poolYUVBuffer(), let rightYUV = poolYUVBuffer() else {
        fputs("ERROR: can't fetch YUV buffers from pool at frame \(framesEncoded)\n", stderr)
        inFlightSem.signal()
        loopShouldBreak = true
        return
    }

    // Transfer RGBA16 → YUV10 on GPU
    let tlStatus = VTPixelTransferSessionTransferImage(transferSession, from: leftRGBA, to: leftYUV)
    let trStatus = VTPixelTransferSessionTransferImage(transferSession, from: rightRGBA, to: rightYUV)
    if tlStatus != 0 || trStatus != 0 {
        fputs("ERROR: transfer failed (l=\(tlStatus) r=\(trStatus)) at frame \(framesEncoded)\n", stderr)
        inFlightSem.signal()
        loopShouldBreak = true
        return
    }

    // Build tagged buffers for each eye and encode via the Swift-refined
    // multi-image encode entry point (takes an [CMTaggedBuffer] array
    // directly, no manual CMTaggedBufferGroup construction).
    let leftTagged = CMTaggedBuffer(tags: [leftEyeTag, layer0Tag], pixelBuffer: leftYUV)
    let rightTagged = CMTaggedBuffer(tags: [rightEyeTag, layer1Tag], pixelBuffer: rightYUV)

    var infoFlags: VTEncodeInfoFlags = []
    let encStatus = VTCompressionSessionEncodeMultiImageFrame(
        compressionSession,
        taggedBuffers: [leftTagged, rightTagged],
        presentationTimeStamp: pts,
        duration: frameDuration,
        frameProperties: nil,
        infoFlagsOut: &infoFlags
    ) { status, flags, sampleBuffer in
        // Every exit path from this block MUST signal the in-flight
        // semaphore — that's what releases the slot back to the main thread.
        defer { inFlightSem.signal() }

        // Diagnostic: track handler firings
        handlerFiredLock.lock()
        handlersFired += 1
        if firstHandlerAt == nil {
            firstHandlerAt = Date()
            let dt = firstHandlerAt!.timeIntervalSince(encodeStartTime)
            fputs("mvhevc_encode: first output handler fired at t=\(String(format: "%.2f", dt))s\n", stderr)
        }
        handlerFiredLock.unlock()

        if status != 0 {
            fputs("WARN: encode output status \(status) at pts \(pts.value)\n", stderr)
            return
        }
        guard let sb = sampleBuffer else { return }
        guard let fmt = CMSampleBufferGetFormatDescription(sb) else {
            fputs("WARN: no format description on sample buffer\n", stderr)
            return
        }
        let firstPTS = CMSampleBufferGetPresentationTimeStamp(sb)
        if !ensureWriter(formatDesc: fmt, firstPTS: firstPTS) {
            return
        }
        // Bounded back-pressure on writer readiness. If the writer wedges
        // (bad state, disk full, etc.) we don't want to livelock here —
        // cap the wait at 5 seconds and drop the frame.
        var waitedMs = 0
        while writerInput?.isReadyForMoreMediaData == false {
            Thread.sleep(forTimeInterval: 0.002)
            waitedMs += 2
            if waitedMs > 5000 {
                fputs("WARN: writerInput not ready after 5s — dropping frame (writer.status=\(writer?.status.rawValue ?? -1))\n", stderr)
                appendErrors += 1
                return
            }
        }
        if let input = writerInput, !input.append(sb) {
            appendErrors += 1
            fputs("WARN: writerInput.append() returned false (status=\(writer?.status.rawValue ?? -1), error=\(String(describing: writer?.error)))\n", stderr)
        }
    }

    if encStatus != 0 {
        // The block handler does NOT fire on submission failure, so we
        // must manually release the permit we acquired above.
        fputs("ERROR: VTCompressionSessionEncodeMultiImageFrame failed (\(encStatus)) at frame \(framesEncoded)\n", stderr)
        inFlightSem.signal()
        loopShouldBreak = true
        return
    }

    pts = CMTimeAdd(pts, frameDuration)
    framesEncoded += 1
    if framesEncoded % 30 == 0 {
        handlerFiredLock.lock()
        let fired = handlersFired
        handlerFiredLock.unlock()
        fputs("mvhevc_encode: submitted=\(framesEncoded) drained=\(fired) in-flight=\(framesEncoded - fired)\n", stderr)
    }
} // end autoreleasepool
    if eofReached || loopShouldBreak { break }
}

// MARK: - Finalize

fputs("mvhevc_encode: flushing encoder (submitted \(framesEncoded) frames)...\n", stderr)
VTCompressionSessionCompleteFrames(compressionSession, untilPresentationTimeStamp: .invalid)

writerInput?.markAsFinished()

if let w = writer {
    let finishSem = DispatchSemaphore(value: 0)
    w.finishWriting {
        finishSem.signal()
    }
    finishSem.wait()

    if w.status != .completed {
        fputs("ERROR: writer.status = \(w.status.rawValue), error = \(String(describing: w.error))\n", stderr)
        exit(1)
    }
}

if appendErrors > 0 {
    fputs("WARN: \(appendErrors) append errors during encode\n", stderr)
}

fputs("mvhevc_encode: done, \(framesEncoded) frames encoded → \(outputPath)\n", stderr)
exit(0)

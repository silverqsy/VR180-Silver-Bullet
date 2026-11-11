// mvhevc_encode — VideoToolbox MV-HEVC encoder for the spatial video export
//
// Reads side-by-side 10-bit 4:2:0 frames (P010 — the encoder's own input
// format) from stdin, splits each into its two eye views and writes an
// MV-HEVC (Main10) movie carrying Apple's spatial metadata: both eye views,
// the hero eye, the camera baseline, the horizontal field of view, the
// disparity adjustment and the projection. Nothing is converted on the way:
// the planes the host rendered are the planes VideoToolbox encodes, once.
//
// Usage:
//   mvhevc_encode --width <per-eye w> --height <h> --fps-num <n> --fps-den <d>
//                 --bitrate <bits per second> --output <file.mov>
//                 [--hfov-deg 90] [--baseline-mm 65] [--disparity 0.02]
//                 [--projection rect|hequ] [--hero left|right]
//
//   --disparity is a fraction of the image width (0.02 = 2 %); positive
//   moves the scene away from the viewer.
//
// Stdin, per frame: the Y plane of the side-by-side frame (2w × h samples,
// 16-bit little-endian with the 10 bits in the high end), then its
// interleaved CbCr plane (w pairs × h/2 rows, same sample format). Video
// range, BT.709. The left eye is the left half. EOF ends the movie.
//
// Build: helpers/build_swift.sh (never a bare swiftc for a shipped binary —
// the script embeds the source hash that `--version` reports).

import Foundation
import AVFoundation
import VideoToolbox
import CoreMedia
import CoreVideo

// Build identity. `helpers/build_swift.sh` compiles this file as main.swift
// next to a generated version.swift that defines HELPER_SOURCE_HASH (the
// sha256 of this source) under `-D HELPER_VERSION_EMBEDDED`; a bare `swiftc`
// build reports "unknown". `--version` lets the release script prove that a
// bundled helper was built from the current source (`build_swift.sh --check`).
#if !HELPER_VERSION_EMBEDDED
let HELPER_SOURCE_HASH = "unknown"
#endif
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--version" {
    print("mvhevc_encode \(HELPER_SOURCE_HASH)")
    exit(0)
}

func die(_ message: String) -> Never {
    fputs("ERROR: \(message)\n", stderr)
    exit(1)
}

func note(_ message: String) {
    fputs("mvhevc_encode: \(message)\n", stderr)
}

// MARK: - Arguments

struct Options {
    var eyeW = 0
    var eyeH = 0
    var fpsNum: Int32 = 30000
    var fpsDen: Int32 = 1001
    var bitrate = 50_000_000
    var hfovDeg = 90.0
    var baselineMm = 65.0
    var disparity = 0.02
    var projection = "rect"
    var hero = "left"
    var output = ""

    static func parse(_ args: [String]) -> Options {
        var o = Options()
        var i = 1
        while i < args.count {
            let key = args[i]
            guard i + 1 < args.count else { die("missing value for \(key)") }
            let value = args[i + 1]
            switch key {
            case "--width":       o.eyeW = Int(value) ?? 0
            case "--height":      o.eyeH = Int(value) ?? 0
            case "--fps-num":     o.fpsNum = Int32(value) ?? 0
            case "--fps-den":     o.fpsDen = Int32(value) ?? 0
            case "--bitrate":     o.bitrate = Int(value) ?? 0
            case "--hfov-deg":    o.hfovDeg = Double(value) ?? 0
            case "--baseline-mm": o.baselineMm = Double(value) ?? 0
            case "--disparity":   o.disparity = Double(value) ?? .nan
            case "--projection":  o.projection = value
            case "--hero":        o.hero = value
            case "--output":      o.output = value
            default: die("unknown argument \(key)")
            }
            i += 2
        }
        guard o.eyeW > 0, o.eyeH > 0, o.eyeW % 2 == 0, o.eyeH % 2 == 0 else {
            die("--width and --height must be positive and even (got \(o.eyeW) × \(o.eyeH))")
        }
        guard o.fpsNum > 0, o.fpsDen > 0 else { die("--fps-num / --fps-den must be positive") }
        guard o.bitrate > 0 else { die("--bitrate must be positive") }
        guard o.hfovDeg > 0, o.hfovDeg <= 360 else { die("--hfov-deg out of range") }
        guard o.baselineMm > 0 else { die("--baseline-mm must be positive") }
        guard o.disparity.isFinite, abs(o.disparity) <= 1 else { die("--disparity must be within ±1") }
        guard o.projection == "rect" || o.projection == "hequ" else { die("--projection must be rect or hequ") }
        guard o.hero == "left" || o.hero == "right" else { die("--hero must be left or right") }
        guard !o.output.isEmpty else { die("--output is required") }
        return o
    }
}

let opt = Options.parse(CommandLine.arguments)

guard VTIsStereoMVHEVCEncodeSupported() else {
    die("this Mac has no MV-HEVC encoder (it needs Apple silicon)")
}

let fpsValue = Double(opt.fpsNum) / Double(opt.fpsDen)
note("\(opt.eyeW) × \(opt.eyeH) per eye, \(String(format: "%.3f", fpsValue)) fps, \(opt.bitrate / 1000) kbps, "
    + "hfov \(opt.hfovDeg)°, baseline \(opt.baselineMm) mm, disparity \(opt.disparity), \(opt.projection), hero \(opt.hero)")

// MARK: - Eye buffers

let pixelFormat = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

/// One eye's row, in bytes, in either plane: w luma samples, or w/2 CbCr pairs.
let eyeRowBytes = opt.eyeW * 2
/// A row of the side-by-side frame, in bytes, in either plane.
let sbsRowBytes = opt.eyeW * 4
let frameBytes = sbsRowBytes * opt.eyeH + sbsRowBytes * (opt.eyeH / 2)

// The encoder keeps the buffers it is given until it is done with them
// (several frames later with reordering), so every frame takes fresh ones
// from a pool, which recycles them as the encoder lets go.
let pool: CVPixelBufferPool = {
    let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 8]
    let bufferAttrs: [String: Any] = [
        kCVPixelBufferWidthKey as String: opt.eyeW,
        kCVPixelBufferHeightKey as String: opt.eyeH,
        kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    var out: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary,
                                         bufferAttrs as CFDictionary, &out)
    guard status == kCVReturnSuccess, let pool = out else { die("pixel buffer pool: \(status)") }
    return pool
}()

/// Copy one eye (0 = left half, 1 = right half) out of the side-by-side
/// planes into a fresh pool buffer.
func eyeBuffer(_ frame: UnsafeRawPointer, eye: Int) -> CVPixelBuffer? {
    var out: CVPixelBuffer?
    guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess,
          let buffer = out else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    // The CbCr plane follows the Y plane, and the eye's offset inside a row
    // is the same in both.
    var src = frame + eye * eyeRowBytes
    for plane in 0..<2 {
        guard let dst = CVPixelBufferGetBaseAddressOfPlane(buffer, plane) else { return nil }
        let dstStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
        let rows = plane == 0 ? opt.eyeH : opt.eyeH / 2
        for row in 0..<rows {
            memcpy(dst + row * dstStride, src + row * sbsRowBytes, eyeRowBytes)
        }
        src += rows * sbsRowBytes
    }
    CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey,
                          kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey,
                          kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    return buffer
}

// MARK: - Compression session

let session: VTCompressionSession = {
    let sourceAttrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
        kCVPixelBufferWidthKey as String: opt.eyeW,
        kCVPixelBufferHeightKey as String: opt.eyeH,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    var out: VTCompressionSession?
    let status = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(opt.eyeW), height: Int32(opt.eyeH),
        codecType: kCMVideoCodecType_HEVC,
        encoderSpecification: nil,
        imageBufferAttributes: sourceAttrs as CFDictionary,
        compressedDataAllocator: nil,
        outputCallback: nil,     // a block per frame instead
        refcon: nil,
        compressionSessionOut: &out)
    guard status == noErr, let session = out else {
        die("VideoToolbox refused an MV-HEVC session at \(opt.eyeW) × \(opt.eyeH) per eye (\(status))")
    }
    return session
}()

/// A property the movie is wrong without: stop if the encoder refuses it.
func require(_ key: CFString, _ value: CFTypeRef) {
    let status = VTSessionSetProperty(session, key: key, value: value)
    if status != noErr { die("the encoder refused \(key as String) (\(status))") }
}

/// A tuning property: carry on without it.
func prefer(_ key: CFString, _ value: CFTypeRef) {
    let status = VTSessionSetProperty(session, key: key, value: value)
    if status != noErr { note("warning: \(key as String) not accepted (\(status))") }
}

// Two layers, one view each: layer 0 is the left eye.
require(kVTCompressionPropertyKey_MVHEVCVideoLayerIDs, [0, 1] as CFArray)
require(kVTCompressionPropertyKey_MVHEVCViewIDs, [0, 1] as CFArray)
require(kVTCompressionPropertyKey_MVHEVCLeftAndRightViewIDs, [0, 1] as CFArray)
require(kVTCompressionPropertyKey_HasLeftStereoEyeView, kCFBooleanTrue)
require(kVTCompressionPropertyKey_HasRightStereoEyeView, kCFBooleanTrue)
// The eye a 2D player shows. (The values are CoreMedia's constants, which
// exist on every macOS that has the key; VideoToolbox's own names for the
// same strings arrived in macOS 26.)
require(kVTCompressionPropertyKey_HeroEye,
        opt.hero == "right" ? kCMFormatDescriptionHeroEye_Right : kCMFormatDescriptionHeroEye_Left)
// Apple's units: micrometres, thousandths of a degree, ten-thousandths of
// the image width.
require(kVTCompressionPropertyKey_StereoCameraBaseline,
        NSNumber(value: UInt32((opt.baselineMm * 1000).rounded())))
if #available(macOS 14.4, *) {
    require(kVTCompressionPropertyKey_HorizontalFieldOfView,
            NSNumber(value: UInt32((opt.hfovDeg * 1000).rounded())))
} else {
    die("spatial video needs macOS 14.4 or later (the encoder cannot record the field of view)")
}
require(kVTCompressionPropertyKey_HorizontalDisparityAdjustment,
        NSNumber(value: Int32((opt.disparity * 10000).rounded())))
if opt.projection == "hequ" {
    if #available(macOS 26.0, *) {
        require(kVTCompressionPropertyKey_ProjectionKind, kCMFormatDescriptionProjectionKind_HalfEquirectangular)
    } else {
        die("a half-equirectangular projection needs macOS 26")
    }
} else if #available(macOS 15.0, *) {
    // Rectilinear is also what a file without the box means, so an older
    // system simply leaves it out.
    prefer(kVTCompressionPropertyKey_ProjectionKind, kCMFormatDescriptionProjectionKind_Rectilinear)
}

require(kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_HEVC_Main10_AutoLevel)
require(kVTCompressionPropertyKey_AverageBitRate, NSNumber(value: opt.bitrate))
prefer(kVTCompressionPropertyKey_ExpectedFrameRate, NSNumber(value: fpsValue))
// An export, not a live stream: quality over latency, B-frames allowed.
prefer(kVTCompressionPropertyKey_RealTime, kCFBooleanFalse)
prefer(kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanTrue)
// A key frame every second keeps seeking quick in the headset.
prefer(kVTCompressionPropertyKey_MaxKeyFrameInterval, NSNumber(value: max(1, Int(fpsValue.rounded()))))
prefer(kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, NSNumber(value: 1.0))
// What the host's planes are: video-range BT.709.
prefer(kVTCompressionPropertyKey_ColorPrimaries, kCVImageBufferColorPrimaries_ITU_R_709_2)
prefer(kVTCompressionPropertyKey_TransferFunction, kCVImageBufferTransferFunction_ITU_R_709_2)
prefer(kVTCompressionPropertyKey_YCbCrMatrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2)

do {
    let status = VTCompressionSessionPrepareToEncodeFrames(session)
    if status != noErr { die("VTCompressionSessionPrepareToEncodeFrames failed (\(status))") }
}

// MARK: - Movie writer

/// The movie file. Created from the first encoded sample (its format
/// description carries the parameter sets and the spatial extensions) and
/// fed from the encoder's output queue.
final class MovieWriter {
    private let url: URL
    private let timescale: CMTimeScale
    private let lock = NSLock()
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var failure: String?
    private var appended = 0

    init(path: String, timescale: CMTimeScale) {
        url = URL(fileURLWithPath: path)
        self.timescale = timescale
        try? FileManager.default.removeItem(at: url)
    }

    func append(_ sample: CMSampleBuffer) {
        lock.lock()
        defer { lock.unlock() }
        if failure != nil { return }
        if writer == nil {
            guard let format = CMSampleBufferGetFormatDescription(sample) else {
                failure = "the encoder's output has no format description"
                return
            }
            do {
                let w = try AVAssetWriter(outputURL: url, fileType: .mov)
                let i = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: format)
                i.expectsMediaDataInRealTime = false
                i.mediaTimeScale = timescale
                guard w.canAdd(i) else {
                    failure = "the movie writer refused the video track"
                    return
                }
                w.add(i)
                guard w.startWriting() else {
                    failure = "startWriting: \(w.error?.localizedDescription ?? "unknown error")"
                    return
                }
                w.startSession(atSourceTime: .zero)
                writer = w
                input = i
            } catch {
                failure = "movie writer: \(error.localizedDescription)"
                return
            }
        }
        guard let input, let writer else { return }
        // The writer drains to disk on its own queue; wait for room.
        var waited = 0.0
        while !input.isReadyForMoreMediaData {
            if writer.status != .writing || waited > 30 {
                failure = "the movie writer stopped taking samples: "
                    + (writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")
                return
            }
            Thread.sleep(forTimeInterval: 0.002)
            waited += 0.002
        }
        if input.append(sample) {
            appended += 1
        } else {
            failure = "append failed: \(writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")"
        }
    }

    func fail(_ message: String) {
        lock.lock()
        if failure == nil { failure = message }
        lock.unlock()
    }

    var error: String? {
        lock.lock()
        defer { lock.unlock() }
        return failure
    }

    /// Close the file. Returns the number of samples written.
    func finish() -> Int {
        lock.lock()
        let (w, i, count) = (writer, input, appended)
        lock.unlock()
        guard let w, let i else { return 0 }
        i.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        w.finishWriting { done.signal() }
        done.wait()
        if w.status != .completed {
            fail("finishing the movie: \(w.error?.localizedDescription ?? "status \(w.status.rawValue)")")
        }
        return count
    }
}

/// Bounds the frames between submission and output. The encoder is
/// asynchronous and holds on to its input buffers until a frame comes out;
/// without a bound the read loop outruns it and memory grows with the clip.
/// Blocking here also stalls the stdin reads, so the host's writes block and
/// the whole chain runs at the encoder's pace.
final class Flow {
    private let condition = NSCondition()
    private var inFlight = 0
    private var limit = 8

    func enter() {
        condition.lock()
        while inFlight >= limit {
            // The encoder may legitimately want more frames than the limit
            // before its first output (look-ahead): after a second without
            // one, let it have them rather than wait forever.
            if !condition.wait(until: Date().addingTimeInterval(1.0)) && inFlight >= limit {
                limit += 4
                note("the encoder holds \(inFlight) frames before output — limit raised to \(limit)")
            }
        }
        inFlight += 1
        condition.unlock()
    }

    func leave() {
        condition.lock()
        inFlight -= 1
        condition.signal()
        condition.unlock()
    }
}

let movie = MovieWriter(path: opt.output, timescale: opt.fpsNum)
let flow = Flow()

// MARK: - Encode loop

/// Fill `buffer` from stdin. False at a clean EOF (no bytes of a new frame);
/// a frame cut short is an error.
func readFrame(into buffer: UnsafeMutableRawPointer) -> Bool {
    var got = 0
    while got < frameBytes {
        let n = read(STDIN_FILENO, buffer + got, frameBytes - got)
        if n > 0 {
            got += n
        } else if n == 0 {
            if got != 0 { movie.fail("the input ended inside a frame (\(got) of \(frameBytes) bytes)") }
            return false
        } else if errno != EINTR {
            movie.fail("reading stdin: \(String(cString: strerror(errno)))")
            return false
        }
    }
    return true
}

let staging = UnsafeMutableRawPointer.allocate(byteCount: frameBytes, alignment: 64)
let leftTags: [CMTag] = [.stereoView(.leftEye), .videoLayerID(0)]
let rightTags: [CMTag] = [.stereoView(.rightEye), .videoLayerID(1)]
let frameDuration = CMTime(value: CMTimeValue(opt.fpsDen), timescale: opt.fpsNum)
var submitted: Int64 = 0
let started = Date()

while movie.error == nil {
    // The pool drains per iteration: without it every buffer touched here
    // waits on an autorelease pool that a tight read loop never empties.
    let more: Bool = autoreleasepool {
        guard readFrame(into: staging) else { return false }
        guard let left = eyeBuffer(staging, eye: 0), let right = eyeBuffer(staging, eye: 1) else {
            movie.fail("no pixel buffer for frame \(submitted)")
            return false
        }
        flow.enter()
        let pts = CMTime(value: submitted * CMTimeValue(opt.fpsDen), timescale: opt.fpsNum)
        let status = VTCompressionSessionEncodeMultiImageFrame(
            session,
            taggedBuffers: [CMTaggedBuffer(tags: leftTags, pixelBuffer: left),
                            CMTaggedBuffer(tags: rightTags, pixelBuffer: right)],
            presentationTimeStamp: pts,
            duration: frameDuration,
            frameProperties: nil,
            infoFlagsOut: nil
        ) { status, _, sample in
            defer { flow.leave() }
            if status != noErr {
                movie.fail("the encoder failed on a frame (\(status))")
            } else if let sample {
                movie.append(sample)
            }
        }
        if status != noErr {
            // No output block runs for a frame that was not accepted.
            flow.leave()
            movie.fail("the encoder refused frame \(submitted) (\(status))")
            return false
        }
        submitted += 1
        return true
    }
    if !more { break }
}

// MARK: - Finish

VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid)
let written = movie.finish()
staging.deallocate()

if let error = movie.error {
    die(error)
}
if written == 0 {
    die("no frames arrived")
}
if Int64(written) != submitted {
    die("\(submitted) frames went in but \(written) came out")
}
let seconds = Date().timeIntervalSince(started)
note("done: \(written) frames in \(String(format: "%.1f", seconds)) s "
    + "(\(String(format: "%.1f", Double(written) / max(seconds, 0.001))) fps)")
exit(0)

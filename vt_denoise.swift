// vt_denoise — VideoToolbox Temporal Noise Filter helper
// Two modes:
//   File mode: Reads video via AVAssetReader, applies VTTemporalNoiseFilter, outputs raw BGR48LE to stdout.
//   Pipe mode: Reads raw BGR48LE from stdin, converts to CVPixelBuffer, denoises, outputs BGR48LE to stdout.
//
// Usage:
//   File mode: vt_denoise <input> [--stream <idx>] [--strength <0.0-1.0>] [--skip <n>] [--max <n>]
//   Pipe mode:  vt_denoise --pipe --width <w> --height <h> [--strength <0.0-1.0>] [--max <n>]
//
// Build: swiftc -O -o vt_denoise vt_denoise.swift -framework AVFoundation -framework VideoToolbox -framework CoreMedia -framework CoreVideo -framework Accelerate

import Foundation
import AVFoundation
import VideoToolbox
import CoreMedia
import CoreVideo
import Accelerate

// MARK: - Argument parsing

var inputPath = ""
var streamIndex = 0
var strength: Float = 0.5
var skipFrames = 0
var maxFrames = Int.max
var outputBGR48 = true  // true = bgr48le (uint16), false = bgr24 (uint8)
var pipeMode = false
var pipeWidth = 0
var pipeHeight = 0

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    switch arg {
    case "--pipe": pipeMode = true
    case "--width": i += 1; pipeWidth = Int(CommandLine.arguments[i]) ?? 0
    case "--height": i += 1; pipeHeight = Int(CommandLine.arguments[i]) ?? 0
    case "--stream": i += 1; streamIndex = Int(CommandLine.arguments[i]) ?? 0
    case "--strength": i += 1; strength = Float(CommandLine.arguments[i]) ?? 0.5
    case "--skip": i += 1; skipFrames = Int(CommandLine.arguments[i]) ?? 0
    case "--max": i += 1; maxFrames = Int(CommandLine.arguments[i]) ?? Int.max
    case "--format":
        i += 1
        outputBGR48 = CommandLine.arguments[i] == "bgr48le"
    default:
        if inputPath.isEmpty { inputPath = arg }
    }
    i += 1
}

guard !inputPath.isEmpty || pipeMode else {
    fputs("Usage:\n  File: vt_denoise <input> [--stream <idx>] [--strength <0.0-1.0>] [--skip <n>] [--max <n>]\n  Pipe: vt_denoise --pipe --width <w> --height <h> [--strength <0.0-1.0>] [--max <n>]\n", stderr)
    exit(1)
}

// ═══════════════════════════════════════════════════════════════════════
// PIPE MODE: read raw BGR48LE from stdin → denoise → output BGR48LE to stdout
// ═══════════════════════════════════════════════════════════════════════
if pipeMode {
    guard pipeWidth > 0 && pipeHeight > 0 else {
        fputs("ERROR: --pipe requires --width and --height\n", stderr); exit(1)
    }
    guard VTTemporalNoiseFilterConfiguration.isSupported else {
        fputs("ERROR: VTTemporalNoiseFilter not supported\n", stderr); exit(1)
    }

    let pw = pipeWidth, ph = pipeHeight
    let frameBytes = pw * ph * 6  // BGR48LE = 6 bytes/pixel
    fputs("vt_denoise pipe: \(pw)x\(ph) strength=\(strength) max=\(maxFrames)\n", stderr)

    // VTTemporalNoiseFilter requires one of Apple's internal lossless
    // compressed pixel formats as the source format. The supported list
    // contains 16 entries on macOS 26 — 8 are 8-bit and 8 are 10-bit. The
    // names use Apple's four-char codes where the second character encodes
    // the bit depth ('8' = 8-bit, 'x' = 10-bit). Pick the first 10-bit
    // format so the denoise filter operates at the source's full precision
    // — the previous version of this helper picked supportedSourcePixelFormats.first
    // unconditionally, which is 8-bit, and forced an unnecessary
    // BGR48 → BGRA8 truncation (256 distinct levels per channel instead
    // of 1024). On macOS 26 with this fix the working precision is
    // genuinely 10-bit end-to-end through the denoise pass.
    let supportedFmts: [OSType] = VTTemporalNoiseFilterConfiguration.supportedSourcePixelFormats
    var targetFmt: OSType = supportedFmts.first!
    var targetBpc: Int = 0
    for fmt in supportedFmts {
        if let desc = CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, fmt) {
            var bpc: Int = 0
            if let v = CFDictionaryGetValue(desc, Unmanaged.passUnretained(kCVPixelFormatBitsPerComponent).toOpaque()) {
                bpc = Unmanaged<NSNumber>.fromOpaque(v).takeUnretainedValue().intValue
            }
            if bpc == 10 {
                targetFmt = fmt
                targetBpc = 10
                break
            }
        }
    }
    if targetBpc == 0 {
        // No 10-bit format available — record bpc of the fallback for the log line below.
        if let desc = CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, targetFmt),
           let v = CFDictionaryGetValue(desc, Unmanaged.passUnretained(kCVPixelFormatBitsPerComponent).toOpaque()) {
            targetBpc = Unmanaged<NSNumber>.fromOpaque(v).takeUnretainedValue().intValue
        }
    }
    let _fmtB0 = UInt8((targetFmt >> 24) & 0xff)
    let _fmtB1 = UInt8((targetFmt >> 16) & 0xff)
    let _fmtB2 = UInt8((targetFmt >> 8) & 0xff)
    let _fmtB3 = UInt8(targetFmt & 0xff)
    let _fmtFourcc = [_fmtB0, _fmtB1, _fmtB2, _fmtB3]
        .map { (32...126).contains($0) ? String(UnicodeScalar($0)) : "." }
        .joined()
    fputs("vt_denoise pipe: lossless source fmt = '\(_fmtFourcc)' (\(targetBpc)-bit, 0x\(String(format: "%08x", targetFmt)))\n", stderr)

    guard let config = VTTemporalNoiseFilterConfiguration(frameWidth: pw, frameHeight: ph, sourcePixelFormat: targetFmt) else {
        fputs("ERROR: Config failed for \(pw)x\(ph)\n", stderr); exit(1)
    }
    let prevCount = config.previousFrameCount ?? 1
    let nextCount = config.nextFrameCount ?? 2
    fputs("vt_denoise pipe: config prev=\(prevCount) next=\(nextCount)\n", stderr)

    let processor = VTFrameProcessor()
    do { try processor.startSession(configuration: config) }
    catch { fputs("ERROR: session: \(error)\n", stderr); exit(1) }

    // VTPixelTransferSession does the 16-bit RGB → 10-bit lossless YCbCr
    // conversion (color matrix + chroma subsample) on the GPU.
    var transferToCompressed: VTPixelTransferSession?
    VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferToCompressed)
    var transferFromCompressed: VTPixelTransferSession?
    VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferFromCompressed)

    let stdinFD = FileHandle.standardInput
    let stdoutFD = FileHandle.standardOutput

    // Use 16-bit-per-channel RGBA as the I/O intermediate so we can carry
    // the full 10 bits of source precision through to / back from the
    // lossless 10-bit compressed format.
    //
    // kCVPixelFormatType_64RGBALE is 16-bit RGBA in **little-endian**, host
    // byte order — no byte swapping needed, we just write uint16s directly.
    //
    // Why not 64ARGB? VTPixelTransferSession returns
    // kVTPixelTransferNotSupportedErr (-12905) when transferring 64ARGB
    // (big-endian) → the lossless 10-bit destination format on macOS 26.
    // The transfer also fails for 48RGB (BE). Probed on 2026-04-09; the
    // working 16-bit RGB-side intermediates are 64RGBALE (this one,
    // RGBA host-endian) and 4444AYpCbCr16 (16-bit YCbCr 4:4:4 w/ alpha).
    // 64RGBALE is the cleanest choice — same channel ordering as our
    // existing BGR48LE input (with alpha tacked on the end), and zero
    // byte-swapping cost on Apple Silicon.
    //
    // (The previous helper used kCVPixelFormatType_32BGRA which is 8 bits
    // per channel and silently truncated the input from 16-bit to 8-bit,
    // losing 2 of the source's 10 actual bits of precision.)
    let ioAttrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]]
    let ioFmt: OSType = kCVPixelFormatType_64RGBALE
    let windowSize = prevCount + 1 + nextCount

    // Reusable 16-bit RGBA I/O buffer (single buffer, reused every frame)
    var ioPB: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, pw, ph, ioFmt, ioAttrs as CFDictionary, &ioPB)
    guard let ioPB else { fputs("ERROR: can't create 64RGBALE I/O buffer\n", stderr); exit(1) }

    // Tag color metadata so VTPixelTransferSession knows how to convert
    // RGB → YCbCr when transferring into the lossless compressed format.
    // BT.709 video range matches HEVC main10 from GoPro MAX which is the
    // typical source piped through this helper.
    CVBufferSetAttachment(ioPB, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(ioPB, kCVImageBufferTransferFunctionKey,
                          kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(ioPB, kCVImageBufferYCbCrMatrixKey,
                          kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)

    // CVPixelBufferPool for compressed source/dest buffers — automatically recycles!
    // This is how GoPro Player caps memory: the pool reuses buffers after they're consumed.
    let poolAttrs: [String: Any] = [
        kCVPixelBufferPoolMinimumBufferCountKey as String: windowSize + 4  // ring + dest + margin
    ]
    let bufAttrs: [String: Any] = [
        kCVPixelBufferWidthKey as String: pw,
        kCVPixelBufferHeightKey as String: ph,
        kCVPixelBufferPixelFormatTypeKey as String: targetFmt,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
    ]
    var compressedPool: CVPixelBufferPool?
    CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary,
                            bufAttrs as CFDictionary, &compressedPool)
    guard let compressedPool else { fputs("ERROR: can't create buffer pool\n", stderr); exit(1) }

    func poolBuffer() -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, compressedPool, &pb)
        return pb
    }

    // Reusable output data buffer
    var outputData = Data(count: frameBytes)

    var framesRead = 0

    // Helper: read BGR48LE frame from stdin → fill reusable 64RGBALE CVPixelBuffer.
    // Source layout (host LE BGR48): [B G R]   per pixel as 3 uint16s (6 bytes)
    // Dest layout   (CV LE 64RGBALE): [R G B A] per pixel as 4 uint16s (8 bytes)
    // Both are host-endian uint16 — no byte swapping, just channel reordering.
    // No precision loss — every uint16 channel is preserved bit-exact.
    func readFrameFromStdin() -> (CVPixelBuffer, CMTime)? {
        let raw = stdinFD.readData(ofLength: frameBytes)
        if raw.count < frameBytes { return nil }

        CVPixelBufferLockBaseAddress(ioPB, [])
        let baseAddr = CVPixelBufferGetBaseAddress(ioPB)!
        let stride = CVPixelBufferGetBytesPerRow(ioPB)
        raw.withUnsafeBytes { srcPtr in
            let src = srcPtr.bindMemory(to: UInt16.self)  // host-LE BGR48
            for row in 0..<ph {
                let dst = baseAddr.advanced(by: row * stride).bindMemory(to: UInt16.self, capacity: pw * 4)
                for col in 0..<pw {
                    let si = (row * pw + col) * 3
                    let b = src[si + 0]
                    let g = src[si + 1]
                    let r = src[si + 2]
                    let di = col * 4
                    dst[di + 0] = r
                    dst[di + 1] = g
                    dst[di + 2] = b
                    dst[di + 3] = 65535   // alpha (opaque)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(ioPB, [])

        // Transfer 16-bit RGBA → 10-bit lossless compressed (hardware-accelerated)
        guard let compressedPB = poolBuffer(), let session = transferToCompressed else { return nil }
        let xferStatus = VTPixelTransferSessionTransferImage(session, from: ioPB, to: compressedPB)
        if xferStatus != 0 {
            fputs("vt_denoise pipe: transfer-to-lossless failed (status=\(xferStatus))\n", stderr)
            return nil
        }

        let pts = CMTimeMake(value: Int64(framesRead), timescale: 30)
        framesRead += 1
        return (compressedPB, pts)
    }

    // Helper: write denoised compressed CVPixelBuffer → BGR48LE to stdout
    func writeFrameToStdout(_ pb: CVPixelBuffer) {
        // Transfer 10-bit lossless compressed → 16-bit RGBA I/O buffer
        guard let session = transferFromCompressed else { return }
        let xferStatus = VTPixelTransferSessionTransferImage(session, from: pb, to: ioPB)
        if xferStatus != 0 {
            fputs("vt_denoise pipe: transfer-from-lossless failed (status=\(xferStatus))\n", stderr)
            return
        }

        // Convert 64RGBALE (host LE) → BGR48LE (host LE) into output buffer
        CVPixelBufferLockBaseAddress(ioPB, .readOnly)
        let baseAddr = CVPixelBufferGetBaseAddress(ioPB)!
        let stride = CVPixelBufferGetBytesPerRow(ioPB)

        outputData.withUnsafeMutableBytes { dstPtr in
            let dst = dstPtr.bindMemory(to: UInt16.self)
            for row in 0..<ph {
                let src = baseAddr.advanced(by: row * stride).bindMemory(to: UInt16.self, capacity: pw * 4)
                for col in 0..<pw {
                    let si = col * 4
                    let r = src[si + 0]
                    let g = src[si + 1]
                    let b = src[si + 2]
                    // skip alpha (src[si + 3])
                    let di = (row * pw + col) * 3
                    dst[di + 0] = b
                    dst[di + 1] = g
                    dst[di + 2] = r
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(ioPB, .readOnly)
        stdoutFD.write(outputData)
    }

    struct PipeFrame {
        let pb: CVPixelBuffer
        let frame: VTFrameProcessorFrame
    }

    let sem = DispatchSemaphore(value: 0)
    var framesOutput = 0
    let chunkSize = 50  // new processor every 50 frames — caps memory at ~500MB for 5952x1920

    // Read-ahead buffer: pre-read frames for the first chunk
    var readAhead: [(CVPixelBuffer, CMTime)] = []

    while framesOutput < maxFrames {
        autoreleasepool {
            // Read chunk + lookahead frames
            let framesToRead = chunkSize + nextCount
            var chunkData: [(CVPixelBuffer, CMTime)] = []

            // Use any leftover from previous chunk's lookahead
            chunkData.append(contentsOf: readAhead)
            readAhead.removeAll()

            while chunkData.count < framesToRead {
                guard let entry = readFrameFromStdin() else { break }
                chunkData.append(entry)
            }
            if chunkData.isEmpty { return }

            // Build VTFrameProcessorFrame objects
            let frames = chunkData.compactMap { (pb, pts) -> PipeFrame? in
                guard let f = VTFrameProcessorFrame(buffer: pb, presentationTimeStamp: pts) else { return nil }
                return PipeFrame(pb: pb, frame: f)
            }

            // Create NEW processor for this chunk (old one gets deallocated → memory freed)
            let chunkProcessor = VTFrameProcessor()
            do { try chunkProcessor.startSession(configuration: config) }
            catch { fputs("ERROR: chunk session: \(error)\n", stderr); return }

            // Process frames in this chunk
            let processCount = min(frames.count, chunkSize)
            for ci in 0..<processCount {
                if framesOutput >= maxFrames { break }
                let ps = max(0, ci - prevCount)
                let ne = min(frames.count, ci + nextCount + 1)
                let pf = Array(frames[ps..<ci].map { $0.frame })
                let nf = Array(frames[(ci+1)..<ne].map { $0.frame })
                if pf.isEmpty && nf.isEmpty { continue }

                guard let destPB = poolBuffer(),
                      let df = VTFrameProcessorFrame(buffer: destPB,
                          presentationTimeStamp: frames[ci].frame.presentationTimeStamp)
                else { continue }

                guard let params = VTTemporalNoiseFilterParameters(
                    sourceFrame: frames[ci].frame, nextFrames: nf, previousFrames: pf,
                    destinationFrame: df, filterStrength: strength,
                    hasDiscontinuity: framesOutput == 0 && ci == 0
                ) else { continue }

                var err: Error? = nil
                chunkProcessor.process(parameters: params) { _, e in err = e; sem.signal() }
                sem.wait()
                if err != nil { fputs("vt_denoise: err frame \(framesOutput): \(err!)\n", stderr); continue }

                writeFrameToStdout(destPB)
                framesOutput += 1
            }

            // Save lookahead frames for next chunk's temporal window
            if frames.count > chunkSize {
                readAhead = Array(frames[chunkSize...].map { ($0.pb, $0.frame.presentationTimeStamp) })
            }

            chunkProcessor.endSession()
            // chunkProcessor + frames go out of scope → autoreleasepool frees everything
        }

        if framesOutput % 100 == 0 { fputs("vt_denoise pipe: \(framesOutput) frames\n", stderr) }
        if readAhead.isEmpty && framesOutput > 0 { break }  // EOF
    }

    processor.endSession()
    fputs("vt_denoise pipe: done, \(framesOutput) frames\n", stderr)
    exit(0)
}

// ═══════════════════════════════════════════════════════════════════════
// FILE MODE: original behavior — read from video file via AVAssetReader
// ═══════════════════════════════════════════════════════════════════════

// MARK: - Setup

let asset = AVURLAsset(url: URL(fileURLWithPath: inputPath))
let videoTracks = asset.tracks(withMediaType: .video)

// Map absolute stream index to video track index (for .360 files with multiple tracks)
var targetTrack: AVAssetTrack?
if streamIndex == 0 {
    targetTrack = videoTracks.first
} else {
    // Find track matching the absolute stream index
    for t in asset.tracks {
        if t.trackID == Int32(streamIndex + 1) || (t.mediaType == .video && videoTracks.firstIndex(of: t) == streamIndex) {
            targetTrack = t
            break
        }
    }
    // Fallback: use video track by position
    if targetTrack == nil && streamIndex < videoTracks.count {
        targetTrack = videoTracks[streamIndex]
    }
    if targetTrack == nil { targetTrack = videoTracks.last }
}

guard let track = targetTrack else {
    fputs("ERROR: No video track found (stream \(streamIndex))\n", stderr)
    exit(1)
}

let w = Int(track.naturalSize.width)
let h = Int(track.naturalSize.height)
fputs("vt_denoise: \(w)x\(h) stream=\(streamIndex) strength=\(strength) skip=\(skipFrames) max=\(maxFrames) fmt=\(outputBGR48 ? "bgr48le" : "bgr24")\n", stderr)

// MARK: - AVAssetReader with lossless compressed format (required by VTTemporalNoiseFilter)

guard VTTemporalNoiseFilterConfiguration.isSupported else {
    fputs("ERROR: VTTemporalNoiseFilter not supported on this system\n", stderr)
    exit(1)
}

let losslessFmt = VTTemporalNoiseFilterConfiguration.supportedSourcePixelFormats.first!
guard let reader = try? AVAssetReader(asset: asset) else {
    fputs("ERROR: Can't create AVAssetReader\n", stderr); exit(1)
}

let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: losslessFmt
])
output.alwaysCopiesSampleData = false
reader.add(output)
reader.startReading()

// MARK: - Configure VTTemporalNoiseFilter

guard let config = VTTemporalNoiseFilterConfiguration(frameWidth: w, frameHeight: h, sourcePixelFormat: losslessFmt) else {
    fputs("ERROR: VTTemporalNoiseFilterConfiguration failed\n", stderr); exit(1)
}

let prevCount = config.previousFrameCount ?? 1
let nextCount = config.nextFrameCount ?? 2
fputs("vt_denoise: config prev=\(prevCount) next=\(nextCount)\n", stderr)

let processor = VTFrameProcessor()
do {
    try processor.startSession(configuration: config)
} catch {
    fputs("ERROR: startSession failed: \(error)\n", stderr); exit(1)
}

// MARK: - Frame ring buffer for temporal window

struct FrameEntry {
    let sample: CMSampleBuffer
    let frame: VTFrameProcessorFrame
    let pixelBuffer: CVPixelBuffer
}

var ringBuffer: [FrameEntry] = []
let windowSize = prevCount + 1 + nextCount  // total frames needed in window

// Read initial window
func readNextFrame() -> FrameEntry? {
    guard let sample = output.copyNextSampleBuffer(),
          let pb = CMSampleBufferGetImageBuffer(sample),
          let frame = VTFrameProcessorFrame(buffer: pb,
              presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sample))
    else { return nil }
    return FrameEntry(sample: sample, frame: frame, pixelBuffer: pb)
}

// Skip frames
for _ in 0..<skipFrames {
    _ = output.copyNextSampleBuffer()
}

// Fill initial window
for _ in 0..<windowSize {
    guard let entry = readNextFrame() else { break }
    ringBuffer.append(entry)
}

fputs("vt_denoise: window loaded (\(ringBuffer.count) frames), processing...\n", stderr)

// MARK: - CVPixelBuffer → BGR raw data conversion via VTPixelTransferSession

// Use VTPixelTransferSession for hardware-accelerated format conversion
// from lossless compressed → uncompressed BGRA, then reorder to BGR
var transferSession: VTPixelTransferSession?
VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)

func pixelBufferToBGRData(_ pb: CVPixelBuffer, bgr48: Bool) -> Data {
    let pw = CVPixelBufferGetWidth(pb)
    let ph = CVPixelBufferGetHeight(pb)

    // Create uncompressed BGRA destination
    let destFmt: OSType = kCVPixelFormatType_32BGRA
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]]
    var destPB: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, pw, ph, destFmt, attrs as CFDictionary, &destPB)
    guard let destPB else { return Data() }

    // Transfer (hardware-accelerated format conversion)
    if let session = transferSession {
        VTPixelTransferSessionTransferImage(session, from: pb, to: destPB)
    }

    // Read BGRA data
    CVPixelBufferLockBaseAddress(destPB, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(destPB, .readOnly) }

    let baseAddr = CVPixelBufferGetBaseAddress(destPB)!
    let stride = CVPixelBufferGetBytesPerRow(destPB)

    if bgr48 {
        // BGRA8 → BGR48LE (scale 0-255 → 0-65535 via *257)
        var bgrData = Data(count: pw * ph * 6)
        bgrData.withUnsafeMutableBytes { bgrPtr in
            let bgr = bgrPtr.bindMemory(to: UInt16.self)
            for row in 0..<ph {
                let rowPtr = baseAddr.advanced(by: row * stride).bindMemory(to: UInt8.self, capacity: pw * 4)
                for col in 0..<pw {
                    let p = row * pw + col
                    let s = col * 4
                    bgr[p * 3 + 0] = UInt16(rowPtr[s + 0]) * 257  // B
                    bgr[p * 3 + 1] = UInt16(rowPtr[s + 1]) * 257  // G
                    bgr[p * 3 + 2] = UInt16(rowPtr[s + 2]) * 257  // R
                }
            }
        }
        return bgrData
    } else {
        // BGRA8 → BGR24 (drop alpha)
        var bgrData = Data(count: pw * ph * 3)
        bgrData.withUnsafeMutableBytes { bgrPtr in
            let bgr = bgrPtr.bindMemory(to: UInt8.self)
            for row in 0..<ph {
                let rowPtr = baseAddr.advanced(by: row * stride).bindMemory(to: UInt8.self, capacity: pw * 4)
                for col in 0..<pw {
                    let p = row * pw + col
                    let s = col * 4
                    bgr[p * 3 + 0] = rowPtr[s + 0]  // B
                    bgr[p * 3 + 1] = rowPtr[s + 1]  // G
                    bgr[p * 3 + 2] = rowPtr[s + 2]  // R
                }
            }
        }
        return bgrData
    }
}

// MARK: - Process and output

let sem = DispatchSemaphore(value: 0)
var framesOutput = 0
var sourceIdx = 0  // index into ringBuffer of the current source frame

// The source frame is at position min(prevCount, available_prev_frames) in the ring buffer
// We slide the window: process center frame, output it, then read next frame and shift window

let stdoutFD = FileHandle.standardOutput

while framesOutput < maxFrames {
    // Determine source position in ring buffer
    let srcPos = min(prevCount, sourceIdx)
    if srcPos >= ringBuffer.count { break }

    // Build previous and next frame arrays
    let prevFrames = Array(ringBuffer[0..<srcPos].map { $0.frame })
    let nextEnd = min(ringBuffer.count, srcPos + nextCount + 1)
    let nextFrames = Array(ringBuffer[(srcPos+1)..<nextEnd].map { $0.frame })

    if prevFrames.isEmpty && nextFrames.isEmpty { break }

    // Create destination pixel buffer
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]]
    var destPB: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, w, h, losslessFmt, attrs as CFDictionary, &destPB)
    guard let destPB,
          let destFrame = VTFrameProcessorFrame(buffer: destPB,
              presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(ringBuffer[srcPos].sample))
    else { break }

    guard let params = VTTemporalNoiseFilterParameters(
        sourceFrame: ringBuffer[srcPos].frame,
        nextFrames: nextFrames,
        previousFrames: prevFrames,
        destinationFrame: destFrame,
        filterStrength: strength,
        hasDiscontinuity: framesOutput == 0
    ) else { break }

    // Process synchronously via async + semaphore
    var processError: Error? = nil
    processor.process(parameters: params) { _, error in
        processError = error
        sem.signal()
    }
    sem.wait()

    if let err = processError {
        fputs("vt_denoise: frame \(framesOutput) error: \(err)\n", stderr)
        break
    }

    // Convert denoised pixel buffer to BGR and write to stdout
    let bgrData = pixelBufferToBGRData(destPB, bgr48: outputBGR48)
    stdoutFD.write(bgrData)
    framesOutput += 1

    // Slide window: remove oldest, read next
    sourceIdx += 1
    if sourceIdx > prevCount {
        ringBuffer.removeFirst()
        sourceIdx -= 1
        if let next = readNextFrame() {
            ringBuffer.append(next)
        }
    }

    if framesOutput % 100 == 0 {
        fputs("vt_denoise: \(framesOutput) frames\n", stderr)
    }
}

processor.endSession()
reader.cancelReading()
fputs("vt_denoise: done, \(framesOutput) frames output\n", stderr)

// apac_encode — Apple Positional Audio Codec encoder helper
//
// Reads a 4-channel ambisonic PCM audio file (typically a temp WAV produced
// by ffmpeg from the GoPro MAX 2 .360 4ch ambisonic track at stream 0:5)
// and writes either:
//   (a) a standalone .mp4 with a single APAC audio track, or
//   (b) a final muxed .mov with the supplied video track copied through
//       (passthrough — sample data + format description preserved bit-exact)
//       PLUS the encoded APAC audio.
//
// Mode (b) exists because ffmpeg's mov muxer does NOT understand APAC and
// drops the codec-specific `dapa` configuration atom on `-c:a copy`. That
// would leave an invalid APAC track Vision Pro cannot decode. AVAssetWriter
// constructs the sample description correctly for APAC, so doing the encode
// and the mux in one helper sidesteps the ffmpeg limitation entirely.
//
// APAC is "Apple Positional Audio Codec" (kAudioFormatAPAC, FourCC 'apac'),
// available on macOS 14+ / iOS 18+ / visionOS 1+. It is the only audio
// codec for which Apple's spatial audio renderer in visionOS does true
// head-tracked spatialisation. SA3D-tagged ambisonic AAC works for
// YouTube VR / Quest Browser but Vision Pro ignores SA3D and falls back
// to plain stereo.
//
// Usage:
//   apac_encode --input <pcm.wav>
//               --output <file.mp4>           # audio-only output
//               [--video-input <video.mov>]   # mux mode: copy video + APAC
//               [--bitrate 384000]            # VBR target, default 384 kbps
//
// Audio input must be 4 channels @ 48 kHz (AmbiX W,Y,Z,X — what GoPro records).
// Video input (if supplied) is copied through bit-exact — the format
// description, sample data, and any spatial metadata atoms (e.g. MV-HEVC
// stereo extensions, APMP vexu/proj/pack/cams atoms) are preserved.
//
// Output container is .mov when --video-input is supplied (so we can carry
// arbitrary video sample types) and .mp4 otherwise. .m4a does NOT accept
// APAC at the AVAssetWriter level (canAdd returns false).
//
// Build:
//   swiftc -O -o apac_encode apac_encode.swift \
//       -framework AVFoundation -framework CoreMedia -framework AudioToolbox

import Foundation
import AVFoundation
import CoreMedia
import AudioToolbox

// MARK: - Argument parsing

var inputPath: String = ""
var videoInputPath: String = ""
var outputPath: String = ""
var bitrateBps: Int = 384_000  // 384 kbps default (Apple's recommended target)

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "--input":       argIdx += 1; inputPath = args[argIdx]
    case "--video-input": argIdx += 1; videoInputPath = args[argIdx]
    case "--output":      argIdx += 1; outputPath = args[argIdx]
    case "--bitrate":     argIdx += 1; bitrateBps = Int(args[argIdx]) ?? 384_000
    default:
        fputs("WARN: unknown argument \(arg)\n", stderr)
    }
    argIdx += 1
}

guard !inputPath.isEmpty, !outputPath.isEmpty else {
    fputs("Usage: apac_encode --input PCM.wav --output FILE.mp4 [--video-input VIDEO.mov] [--bitrate 384000]\n", stderr)
    exit(1)
}

let muxMode = !videoInputPath.isEmpty

// MARK: - Platform check

if #available(macOS 14.0, *) {
    // OK — APAC is available
} else {
    fputs("ERROR: APAC encoding requires macOS 14 (Sonoma) or newer.\n", stderr)
    exit(1)
}

let inputURL = URL(fileURLWithPath: inputPath)
let outputURL = URL(fileURLWithPath: outputPath)

// Remove existing output to avoid AVAssetWriter complaining
if FileManager.default.fileExists(atPath: outputPath) {
    try? FileManager.default.removeItem(at: outputURL)
}

// MARK: - Audio reader setup

let audioAsset = AVURLAsset(url: inputURL)
let audioTracks = audioAsset.tracks(withMediaType: .audio)
guard let audioTrack = audioTracks.first else {
    fputs("ERROR: no audio tracks in \(inputPath)\n", stderr)
    exit(1)
}

guard let audioFormatDescriptions = audioTrack.formatDescriptions as? [CMFormatDescription],
      let audioFirstFormat = audioFormatDescriptions.first,
      let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(audioFirstFormat) else {
    fputs("ERROR: can't read source audio format\n", stderr)
    exit(1)
}
let asbd = asbdPtr.pointee
let srcChannels = Int(asbd.mChannelsPerFrame)
let srcRate = asbd.mSampleRate

fputs("apac_encode: input audio = \(inputPath)\n", stderr)
fputs("apac_encode: source = \(srcChannels) channels @ \(srcRate) Hz\n", stderr)
if muxMode {
    fputs("apac_encode: input video = \(videoInputPath) (passthrough mux)\n", stderr)
}

guard srcChannels == 4 else {
    fputs("ERROR: APAC encoder expects 4-channel ambisonic input, got \(srcChannels) channels.\n", stderr)
    fputs("       For GoPro MAX 2: extract stream 0:5 to a 4ch WAV via\n", stderr)
    fputs("       'ffmpeg -i input.360 -map 0:5 -c:a pcm_s24le out.wav'.\n", stderr)
    exit(1)
}

let audioReader: AVAssetReader
do {
    audioReader = try AVAssetReader(asset: audioAsset)
} catch {
    fputs("ERROR: AVAssetReader init failed (audio): \(error)\n", stderr)
    exit(1)
}

// Decode to 32-bit float interleaved PCM @ original rate. APAC's encoder
// path expects float PCM input; the converter handles bit-depth promotion
// from s24le → float.
let decodeSettings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVLinearPCMBitDepthKey: 32,
    AVLinearPCMIsFloatKey: true,
    AVLinearPCMIsBigEndianKey: false,
    AVLinearPCMIsNonInterleaved: false,
    AVNumberOfChannelsKey: 4,
    AVSampleRateKey: srcRate,
]
let audioReaderOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: decodeSettings)
audioReaderOutput.alwaysCopiesSampleData = false
guard audioReader.canAdd(audioReaderOutput) else {
    fputs("ERROR: audio reader can't add output\n", stderr); exit(1)
}
audioReader.add(audioReaderOutput)

// MARK: - Video reader setup (mux mode only)

var videoReader: AVAssetReader? = nil
var videoReaderOutput: AVAssetReaderTrackOutput? = nil
var videoTrackForWriter: AVAssetTrack? = nil
var videoFormatHint: CMFormatDescription? = nil

if muxMode {
    let videoURL = URL(fileURLWithPath: videoInputPath)
    let videoAsset = AVURLAsset(url: videoURL)
    let videoTracks = videoAsset.tracks(withMediaType: .video)
    guard let vTrack = videoTracks.first else {
        fputs("ERROR: no video tracks in \(videoInputPath)\n", stderr)
        exit(1)
    }
    videoTrackForWriter = vTrack
    if let formats = vTrack.formatDescriptions as? [CMFormatDescription],
       let first = formats.first {
        videoFormatHint = first
    }
    do {
        videoReader = try AVAssetReader(asset: videoAsset)
    } catch {
        fputs("ERROR: AVAssetReader init failed (video): \(error)\n", stderr)
        exit(1)
    }
    // outputSettings: nil = passthrough — we get the original encoded
    // sample buffers exactly as they sit in the source file.
    let vOut = AVAssetReaderTrackOutput(track: vTrack, outputSettings: nil)
    vOut.alwaysCopiesSampleData = false
    guard videoReader!.canAdd(vOut) else {
        fputs("ERROR: video reader can't add passthrough output\n", stderr); exit(1)
    }
    videoReader!.add(vOut)
    videoReaderOutput = vOut
}

// MARK: - Writer setup with APAC + ambisonic layout

let writer: AVAssetWriter
do {
    // .m4a does NOT accept APAC (AVAssetWriter rejects the settings dict
    // with canAdd:false). When carrying video too, .mov is the most
    // permissive container. Otherwise default to .mp4.
    let lowered = outputPath.lowercased()
    let fileType: AVFileType
    if muxMode {
        fileType = lowered.hasSuffix(".mp4") ? .mp4 : .mov
    } else {
        fileType = lowered.hasSuffix(".mov") ? .mov : .mp4
    }
    writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
} catch {
    fputs("ERROR: AVAssetWriter init failed: \(error)\n", stderr)
    exit(1)
}

// 1st-order ambisonic channel layout tag.
//   kAudioChannelLayoutTag_HOA_ACN_SN3D = (190 << 16)
//   The low 16 bits encode the channel count.
//   For 1st order: 4 channels (W, Y, Z, X in ACN ordering).
//   SN3D normalisation = AmbiX standard (matches GoPro/YouTube/Apple).
var layout = AudioChannelLayout()
layout.mChannelLayoutTag = kAudioChannelLayoutTag_HOA_ACN_SN3D | 4
layout.mChannelBitmap = AudioChannelBitmap(rawValue: 0)
layout.mNumberChannelDescriptions = 0
let layoutData = withUnsafeBytes(of: &layout) { Data($0) }

// APAC encoder settings. AVFormatIDKey wants Int from the FourCC.
// AVEncoderBitRateKey is honored by the APAC encoder as a VBR target hint.
let encodeSettings: [String: Any] = [
    AVFormatIDKey: Int(kAudioFormatAPAC),
    AVSampleRateKey: 48000.0,           // APAC outputs 48 kHz natively
    AVNumberOfChannelsKey: 4,
    AVEncoderBitRateKey: bitrateBps,
    AVChannelLayoutKey: layoutData,
]

let audioWriterInput = AVAssetWriterInput(mediaType: .audio, outputSettings: encodeSettings)
audioWriterInput.expectsMediaDataInRealTime = false

guard writer.canAdd(audioWriterInput) else {
    fputs("ERROR: writer can't add APAC input — settings may be unsupported on this OS.\n", stderr)
    exit(1)
}
writer.add(audioWriterInput)

// Video passthrough input (mux mode only)
var videoWriterInput: AVAssetWriterInput? = nil
if muxMode, let vTrack = videoTrackForWriter {
    // outputSettings: nil = passthrough; sourceFormatHint carries the
    // original sample description (so MV-HEVC stereo data, APMP atoms,
    // colorspace tags, etc. are preserved bit-exact).
    let vIn = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: videoFormatHint)
    vIn.expectsMediaDataInRealTime = false
    vIn.transform = vTrack.preferredTransform
    if writer.canAdd(vIn) {
        writer.add(vIn)
        videoWriterInput = vIn
    } else {
        fputs("ERROR: writer can't add passthrough video input\n", stderr)
        exit(1)
    }
}

// MARK: - Encode loop

guard writer.startWriting() else {
    fputs("ERROR: writer.startWriting() failed: \(String(describing: writer.error))\n", stderr)
    exit(1)
}
writer.startSession(atSourceTime: .zero)

guard audioReader.startReading() else {
    fputs("ERROR: audioReader.startReading() failed: \(String(describing: audioReader.error))\n", stderr)
    exit(1)
}
if muxMode, let vr = videoReader {
    guard vr.startReading() else {
        fputs("ERROR: videoReader.startReading() failed: \(String(describing: vr.error))\n", stderr)
        exit(1)
    }
}

let audioQueue = DispatchQueue(label: "apac.encode.audio")
let videoQueue = DispatchQueue(label: "apac.encode.video")
let group = DispatchGroup()
var encodeError: Error? = nil
var audioSamplesWritten: Int64 = 0
var videoSamplesWritten: Int64 = 0
let errorLock = NSLock()
func setError(_ e: Error) {
    errorLock.lock(); if encodeError == nil { encodeError = e }; errorLock.unlock()
}

group.enter()
audioWriterInput.requestMediaDataWhenReady(on: audioQueue) {
    while audioWriterInput.isReadyForMoreMediaData {
        if audioReader.status == .reading, let buf = audioReaderOutput.copyNextSampleBuffer() {
            let n = CMSampleBufferGetNumSamples(buf)
            audioSamplesWritten += Int64(n)
            if !audioWriterInput.append(buf) {
                if let e = writer.error { setError(e) }
                fputs("ERROR: audioWriterInput.append failed: \(String(describing: writer.error))\n", stderr)
                audioWriterInput.markAsFinished()
                group.leave()
                return
            }
        } else {
            audioWriterInput.markAsFinished()
            if audioReader.status == .failed, let e = audioReader.error {
                setError(e)
                fputs("ERROR: audio reader failed: \(e)\n", stderr)
            }
            group.leave()
            return
        }
    }
}

if let videoIn = videoWriterInput, let vReaderOut = videoReaderOutput, let vReader = videoReader {
    group.enter()
    videoIn.requestMediaDataWhenReady(on: videoQueue) {
        while videoIn.isReadyForMoreMediaData {
            if vReader.status == .reading, let buf = vReaderOut.copyNextSampleBuffer() {
                videoSamplesWritten += 1
                if !videoIn.append(buf) {
                    if let e = writer.error { setError(e) }
                    fputs("ERROR: videoWriterInput.append failed: \(String(describing: writer.error))\n", stderr)
                    videoIn.markAsFinished()
                    group.leave()
                    return
                }
            } else {
                videoIn.markAsFinished()
                if vReader.status == .failed, let e = vReader.error {
                    setError(e)
                    fputs("ERROR: video reader failed: \(e)\n", stderr)
                }
                group.leave()
                return
            }
        }
    }
}

group.wait()

let finishSemaphore = DispatchSemaphore(value: 0)
writer.finishWriting {
    finishSemaphore.signal()
}
finishSemaphore.wait()

if writer.status == .failed {
    fputs("ERROR: writer.finishWriting failed: \(String(describing: writer.error))\n", stderr)
    exit(1)
}
if let err = encodeError {
    fputs("ERROR: encode aborted: \(err)\n", stderr)
    exit(1)
}

let durSec = Double(audioSamplesWritten) / srcRate
fputs("apac_encode: wrote \(audioSamplesWritten) PCM frames (\(String(format: "%.2f", durSec))s) → \(outputPath)\n", stderr)
if muxMode {
    fputs("apac_encode: muxed \(videoSamplesWritten) video samples (passthrough)\n", stderr)
}
exit(0)

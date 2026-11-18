// aivu_wrap — several Lens Space clips, already joined, as ONE Apple Immersive Video file
//
// A Lens Space export states its lens in its own file (the parametric
// immersive projection: one lens for the whole movie). Clips of different
// recording modes or cameras have different lenses, so their join cannot be
// described that way. An Apple Immersive Video Universal file (`.aivu`) can
// hold them: it carries a VENUE (a list of cameras, each one the mesh its
// picture is drawn on) and, on a metadata track, the camera every frame is
// shown through.
//
// This helper takes the file the join wrote (the parts' video packets one
// after the other, and the sound) and the parts themselves, and writes that
// file again as an `.aivu`:
//   - the video and the sound packet for packet, nothing encoded again (the
//     picture stays side by side: an Apple Immersive Video file need not be
//     MV-HEVC);
//   - the video's sample entry rebuilt with the projection kind "Apple
//     Immersive Video" in place of the one lens it stated;
//   - a venue with one camera per part: a mesh of the part's picture, built
//     from the lens the part's own file states;
//   - one metadata sample per frame, naming that frame's camera.
// Apple's own validator then reads the result; a file it refuses is removed.
//
// Usage:
//   aivu_wrap --joined <file> --output <file.aivu>
//             --part <file> --frames <n> [--part <file> --frames <n> ...]
//
//   --part / --frames, in the order of the join: a part's file (for its
//   lens) and the number of video frames it gave to the joined file.
//
// Progress goes to stderr as "progress <frames written> <frames in all>".
//
// Needs macOS 26 (the ImmersiveMediaSupport framework).
//
// Build: helpers/build_swift.sh (never a bare swiftc for a shipped binary —
// the script embeds the source hash that `--version` reports).

import Foundation
import AVFoundation
import CoreMedia
import ImageIO
import ImmersiveMediaSupport
import ModelIO
import simd

// Build identity. `helpers/build_swift.sh` compiles this file as main.swift
// next to a generated version.swift that defines HELPER_SOURCE_HASH (the
// sha256 of this source) under `-D HELPER_VERSION_EMBEDDED`; a bare `swiftc`
// build reports "unknown".
#if !HELPER_VERSION_EMBEDDED
let HELPER_SOURCE_HASH = "unknown"
#endif
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--version" {
    print("aivu_wrap \(HELPER_SOURCE_HASH)")
    exit(0)
}

func die(_ message: String) -> Never {
    fputs("ERROR: \(message)\n", stderr)
    exit(1)
}

func note(_ message: String) {
    fputs("aivu_wrap: \(message)\n", stderr)
}

// MARK: - Arguments

struct Part {
    var path: String
    var frames: Int
}

struct Options {
    var joined = ""
    var output = ""
    var parts: [Part] = []

    static func parse(_ args: [String]) -> Options {
        var o = Options()
        var i = 1
        func value() -> String {
            guard i + 1 < args.count else { die("\(args[i]) needs a value") }
            i += 1
            return args[i]
        }
        while i < args.count {
            switch args[i] {
            case "--joined": o.joined = value()
            case "--output": o.output = value()
            case "--part": o.parts.append(Part(path: value(), frames: 0))
            case "--frames":
                guard !o.parts.isEmpty, let n = Int(value()), n > 0 else { die("--frames follows a --part and is a count of frames") }
                o.parts[o.parts.count - 1].frames = n
            default: die("unknown argument \(args[i])")
            }
            i += 1
        }
        if o.joined.isEmpty || o.output.isEmpty || o.parts.isEmpty { die("usage: aivu_wrap --joined <file> --output <file.aivu> --part <file> --frames <n> ...") }
        if let p = o.parts.first(where: { $0.frames == 0 }) { die("\(p.path): no --frames") }
        return o
    }
}

// MARK: - The lens of a part

/// The one centred lens a Lens Space file states for both its eyes, in the
/// terms of its lens collection: the Mei-Rives model (a unit ray moved along
/// the axis by `xi`, two radial terms, the focal length in pixels of the
/// eye's frame).
struct Lens {
    var width = 0.0, height = 0.0
    var fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0
    var xi = 0.0, k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0
    var limitDeg = 90.0

    /// Pixels from the centre of a ray `theta` radians off the axis.
    func radius(_ theta: Double) -> Double {
        let m = sin(theta) / (cos(theta) + xi)
        return fx * m * (1 + k1 * m * m + k2 * m * m * m * m)
    }

    /// The angle a radius comes from, not past the stated limit.
    func angle(_ r: Double) -> Double {
        var lo = 0.0, hi = limitDeg * .pi / 180
        if radius(hi) <= r { return hi }
        for _ in 0..<60 {
            let mid = 0.5 * (lo + hi)
            if radius(mid) < r { lo = mid } else { hi = mid }
        }
        return 0.5 * (lo + hi)
    }
}

func lens(of path: String) async throws -> Lens {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    guard let track = try await asset.loadTracks(withMediaType: .video).first,
          let format = try await track.load(.formatDescriptions).first else { die("\(path): no video") }
    let ext = (CMFormatDescriptionGetExtensions(format) as? [String: Any]) ?? [:]
    guard let lenses = ext[kCMFormatDescriptionExtension_CameraCalibrationDataLensCollection as String] as? [[String: Any]], lenses.count == 2,
          ext[kCMFormatDescriptionExtension_ViewPackingKind as String] != nil else {
        die("\(path): not a side-by-side Lens Space file with a lens for each eye")
    }
    // Both eyes are drawn through one lens: the first says it.
    let l = lenses[0]
    var out = Lens()
    guard let k = l[kCMFormatDescriptionCameraCalibration_IntrinsicMatrix as String] as? Data, k.count == MemoryLayout<simd_float3x3>.size else {
        die("\(path): its lens has no intrinsic matrix")
    }
    let m = k.withUnsafeBytes { $0.loadUnaligned(as: simd_float3x3.self) }
    (out.fx, out.fy, out.cx, out.cy) = (Double(m.columns.0.x), Double(m.columns.1.y), Double(m.columns.2.x), Double(m.columns.2.y))
    let dims = l[kCMFormatDescriptionCameraCalibration_IntrinsicMatrixReferenceDimensions as String] as? [String: Any]
    out.width = (dims?["Width"] as? NSNumber)?.doubleValue ?? 0
    out.height = (dims?["Height"] as? NSNumber)?.doubleValue ?? 0
    // The file stores the projection offset with the other sign.
    out.xi = -((l[kCMFormatDescriptionCameraCalibration_IntrinsicMatrixProjectionOffset as String] as? NSNumber)?.doubleValue ?? 0)
    let d = (l[kCMFormatDescriptionCameraCalibration_LensDistortions as String] as? [NSNumber])?.map { $0.doubleValue } ?? []
    if d.count >= 4 { (out.k1, out.k2, out.p1, out.p2) = (d[0], d[1], d[2], d[3]) }
    out.limitDeg = (l[kCMFormatDescriptionCameraCalibration_RadialAngleLimit as String] as? NSNumber)?.doubleValue ?? 90
    guard out.width > 0, out.height > 0, out.fx > 0 else { die("\(path): its lens is incomplete") }
    // What the mesh below draws: a lens that is the same all the way round.
    guard abs(out.fy - out.fx) <= 1e-6 * out.fx, out.p1 == 0, out.p2 == 0 else {
        die("\(path): its lens is not the same all the way round (two focal lengths, or tangential terms)")
    }
    return out
}

// MARK: - The camera of a part

/// Pixels at the frame's border that hold no picture (the rim a Lens Space
/// frame is written with).
let frameRimPx = 8.0
/// Pixels kept clear of the lens's limit, where the picture ends in black.
let limitRimPx = 1.5
/// The mesh: rings from the axis outward, and spokes around it, the same
/// number to each side of the frame so that the frame's corners are spokes.
let meshRings = 128
let meshSpokesPerSide = 64

/// The picture of a part as the mesh it is drawn on.
///
/// An Apple Immersive camera is a mesh for each eye: a piece of the unit
/// sphere around the viewer (x to the right, y up, the lens looking along
/// -z), every vertex with the place in the eye's frame it shows (from the
/// frame's bottom left, 0 to 1 either way). The framework can build that
/// mesh from a lens profile, but the one it builds runs to 121.7 degrees
/// whatever the lens's limit, on past the frame's edge and past where this
/// lens model folds back on itself, and leaves it to a mask to end the
/// picture; and a mask of control points comes out wrong where its outline
/// goes over the zenith, as a full fisheye's does (macOS 27.2 draws a band
/// between two heights). So the mesh is built here and ends where the
/// picture does: along every spoke at the frame's rim or at the lens's
/// limit, whichever comes first. Nothing outside the picture is drawn, and
/// nothing is left for a mask to hide.
///
/// A triangle is flat between its vertices: at 4096 pixels an eye the
/// picture of a full fisheye is off by up to a quarter of a pixel at its
/// rim, and by less than a tenth on average (measured against the lens).
struct PictureMesh {
    /// x, y, z of every vertex.
    var positions: [Float] = []
    /// The place in the frame every vertex shows.
    var coordinates: [Float] = []
    /// Three vertices to a triangle.
    var indices: [UInt32] = []
}

func pictureMesh(_ lens: Lens) -> PictureMesh {
    // The frame's sides from the lens's centre, y up (the frame's top row is
    // pixel 0), and the lens's limit as a radius.
    let (left, right) = (-(lens.cx - frameRimPx), lens.width - lens.cx - frameRimPx)
    let (down, up) = (-(lens.height - lens.cy - frameRimPx), lens.cy - frameRimPx)
    let limit = lens.radius(lens.limitDeg * .pi / 180) - limitRimPx
    /// Pixels from the centre to where the picture ends, along an azimuth.
    func reach(_ phi: Double) -> Double {
        let (c, s) = (cos(phi), sin(phi))
        var r = limit
        if c > 1e-9 { r = min(r, right / c) } else if c < -1e-9 { r = min(r, left / c) }
        if s > 1e-9 { r = min(r, up / s) } else if s < -1e-9 { r = min(r, down / s) }
        return r
    }
    // Spokes: from each corner of the frame round to the next.
    var corners = [atan2(up, right), atan2(up, left), atan2(down, left), atan2(down, right)]
    for i in 1..<4 { while corners[i] <= corners[i - 1] { corners[i] += 2 * .pi } }
    corners.append(corners[0] + 2 * .pi)
    var spokes: [(phi: Double, theta: Double)] = []
    for side in 0..<4 {
        for i in 0..<meshSpokesPerSide {
            let phi = corners[side] + (corners[side + 1] - corners[side]) * Double(i) / Double(meshSpokesPerSide)
            spokes.append((phi, lens.angle(reach(phi))))
        }
    }
    var mesh = PictureMesh()
    func add(_ phi: Double, _ theta: Double) {
        let r = lens.radius(theta)
        mesh.positions += [Float(sin(theta) * cos(phi)), Float(sin(theta) * sin(phi)), Float(-cos(theta))]
        mesh.coordinates += [Float((lens.cx + r * cos(phi)) / lens.width), Float(1 - (lens.cy - r * sin(phi)) / lens.height)]
    }
    add(0, 0)
    for ring in 1...meshRings {
        for spoke in spokes { add(spoke.phi, spoke.theta * Double(ring) / Double(meshRings)) }
    }
    // Triangles, counter-clockwise as seen from the centre (the way the
    // framework's own meshes are wound).
    let n = spokes.count
    func at(_ ring: Int, _ spoke: Int) -> UInt32 { UInt32(1 + (ring - 1) * n + spoke % n) }
    for j in 0..<n { mesh.indices += [0, at(1, j), at(1, j + 1)] }
    for i in 1..<meshRings {
        for j in 0..<n {
            mesh.indices += [at(i, j), at(i + 1, j), at(i + 1, j + 1)]
            mesh.indices += [at(i, j), at(i + 1, j + 1), at(i, j + 1)]
        }
    }
    return mesh
}

func crc32(_ data: Data) -> UInt32 {
    var table = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 {
        var c = UInt32(i)
        for _ in 0..<8 { c = c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        table[i] = c
    }
    var crc: UInt32 = 0xFFFF_FFFF
    for byte in data { crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8) }
    return crc ^ 0xFFFF_FFFF
}

/// One file as a USDZ package: a zip archive that stores the file as it is,
/// its data on a 64-byte boundary (the padding goes in the entry's extra
/// field).
func package(_ data: Data, as name: String) -> Data {
    func le16(_ v: Int) -> [UInt8] { [UInt8(v & 255), UInt8(v >> 8 & 255)] }
    func le32(_ v: Int) -> [UInt8] { [UInt8(v & 255), UInt8(v >> 8 & 255), UInt8(v >> 16 & 255), UInt8(v >> 24 & 255)] }
    let nameBytes = Array(name.utf8)
    let crc = Int(crc32(data))
    var pad = (64 - (30 + nameBytes.count) % 64) % 64
    if pad > 0 && pad < 4 { pad += 64 }
    let extra: [UInt8] = pad == 0 ? [] : [0x86, 0x19] + le16(pad - 4) + [UInt8](repeating: 0, count: pad - 4)
    // No compression, no flags; the date is the format's first day.
    let common = le16(0) + le16(0) + le16(0) + le16(0x21) + le32(crc) + le32(data.count) + le32(data.count)
    var out = Data()
    out.append(contentsOf: [0x50, 0x4B, 0x03, 0x04] + le16(10) + common + le16(nameBytes.count) + le16(extra.count))
    out.append(contentsOf: nameBytes)
    out.append(contentsOf: extra)
    out.append(data)
    let directory = out.count
    out.append(contentsOf: [0x50, 0x4B, 0x01, 0x02] + le16(10) + le16(10) + common
               + le16(nameBytes.count) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(0))
    out.append(contentsOf: nameBytes)
    let size = out.count - directory
    out.append(contentsOf: [0x50, 0x4B, 0x05, 0x06] + le16(0) + le16(0) + le16(1) + le16(1) + le32(size) + le32(directory) + le16(0))
    return out
}

/// The mesh as the package a venue takes: one USD file holding a mesh for
/// each eye (the same one: a Lens Space file draws both eyes through one
/// lens). The framework knows the eyes' meshes by their names, `_0_Left`
/// and `_0_Right`, and refuses a package that calls them anything else.
func usdz(_ mesh: PictureMesh) throws -> Data {
    let allocator = MDLMeshBufferDataAllocator()
    func eye(_ name: String) -> MDLMesh {
        let positions = allocator.newBuffer(with: mesh.positions.withUnsafeBufferPointer { Data(buffer: $0) }, type: .vertex)
        let coordinates = allocator.newBuffer(with: mesh.coordinates.withUnsafeBufferPointer { Data(buffer: $0) }, type: .vertex)
        let indices = allocator.newBuffer(with: mesh.indices.withUnsafeBufferPointer { Data(buffer: $0) }, type: .index)
        let layout = MDLVertexDescriptor()
        layout.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        layout.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate, format: .float2, offset: 0, bufferIndex: 1)
        layout.layouts[0] = MDLVertexBufferLayout(stride: 12)
        layout.layouts[1] = MDLVertexBufferLayout(stride: 8)
        let triangles = MDLSubmesh(indexBuffer: indices, indexCount: mesh.indices.count, indexType: .uInt32, geometryType: .triangles, material: nil)
        let out = MDLMesh(vertexBuffers: [positions, coordinates], vertexCount: mesh.positions.count / 3, descriptor: layout, submeshes: [triangles])
        out.name = name
        return out
    }
    let asset = MDLAsset(bufferAllocator: allocator)
    asset.add(eye("_0_Left"))
    asset.add(eye("_0_Right"))
    // Model I/O writes a USD file to disk only.
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("aivu_wrap-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("mesh.usdc")
    try asset.export(to: file)
    return package(try Data(contentsOf: file), as: "mesh.usdc")
}

/// The mask every camera gets: none is wanted, the mesh ends the picture. A
/// file from Apple's own tools carries a white image where it wants no mask
/// (8640 x 4320, the two eyes side by side); so does this one.
func whiteMask() -> Data? {
    let (width, height) = (8640, 4320)
    guard let provider = CGDataProvider(data: Data(repeating: 255, count: width * height) as CFData),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil,
                              shouldInterpolate: false, intent: .defaultIntent) else { return nil }
    let out = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(out, "public.png" as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination) ? out as Data : nil
}

// MARK: - The file

/// A sample buffer of the joined file as the new file takes it: the same
/// data and timing, described by `format` when one is given.
func carried(_ sample: CMSampleBuffer, as format: CMFormatDescription?) -> CMSampleBuffer? {
    guard let format else { return sample }
    guard let data = CMSampleBufferGetDataBuffer(sample) else { return nil }
    var timingCount: CMItemCount = 0
    CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &timingCount)
    var timings = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: timingCount)
    CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: timingCount, arrayToFill: &timings, entriesNeededOut: &timingCount)
    var sizeCount: CMItemCount = 0
    CMSampleBufferGetSampleSizeArray(sample, entryCount: 0, arrayToFill: nil, entriesNeededOut: &sizeCount)
    var sizes = [Int](repeating: 0, count: sizeCount)
    CMSampleBufferGetSampleSizeArray(sample, entryCount: sizeCount, arrayToFill: &sizes, entriesNeededOut: &sizeCount)
    var out: CMSampleBuffer?
    CMSampleBufferCreateReady(allocator: nil, dataBuffer: data, formatDescription: format, sampleCount: CMSampleBufferGetNumSamples(sample),
                              sampleTimingEntryCount: timingCount, sampleTimingArray: &timings, sampleSizeEntryCount: sizeCount, sampleSizeArray: &sizes, sampleBufferOut: &out)
    guard let out else { return nil }
    // What says which samples are keyframes travels with them.
    if let from = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false) as? [[String: Any]],
       let to = CMSampleBufferGetSampleAttachmentsArray(out, createIfNecessary: true) {
        for (i, attachments) in from.enumerated() where i < CFArrayGetCount(to) {
            let target = unsafeBitCast(CFArrayGetValueAtIndex(to, i), to: NSMutableDictionary.self)
            for (key, value) in attachments { target[key] = value }
        }
    }
    return out
}

func run() async throws {
    let opt = Options.parse(CommandLine.arguments)
    let outURL = URL(fileURLWithPath: opt.output)
    try? FileManager.default.removeItem(at: outURL)

    let joined = AVURLAsset(url: URL(fileURLWithPath: opt.joined))
    guard let videoTrack = try await joined.loadTracks(withMediaType: .video).first,
          let videoFormat = try await videoTrack.load(.formatDescriptions).first else { die("\(opt.joined): no video") }
    let rate = try await videoTrack.load(.nominalFrameRate)
    let frameRate = max(1, Int(rate.rounded()))
    let total = opt.parts.reduce(0) { $0 + $1.frames }

    // The venue: a camera for every part.
    let venue = VenueDescriptor()
    guard let white = whiteMask() else { die("the white mask image could not be made") }
    let mask = ImmersiveCameraMask.image(ImmersiveImageMask(name: "no_mask", maskData: white))
    var cameras: [String] = []
    for (n, part) in opt.parts.enumerated() {
        let l = try await lens(of: part.path)
        let id = "clip-\(n + 1)"
        let mesh = pictureMesh(l)
        let calibration = ImmersiveCameraCalibration(name: id, type: .usdzMesh(ImmersiveCameraMeshCalibration(name: id, usdzData: try usdz(mesh))), mask: mask)
        try await venue.addCamera(ImmersiveCamera(id: id, calibration: calibration, presentationFrameRate: frameRate))
        // The framework's own reading of the mesh, before anything is written.
        guard let model = await venue.cameraViewModel(for: id), model.leftEye.vertexCount == mesh.positions.count / 3,
              model.rightEye.vertexCount == mesh.positions.count / 3 else { die("\(part.path): the venue does not take its mesh") }
        cameras.append(id)
        note(String(format: "%@: %.0f x %.0f an eye, focal %.2f px, xi %.4f, k1 %.4f, k2 %.4f, to %.1f deg; %d frames",
                    id, l.width, l.height, l.fx, l.xi, l.k1, l.k2, l.limitDeg, part.frames))
    }
    guard let aime = try await venue.aimeData else { die("the venue gave no data") }

    // The video's sample entry: the projection kind in place of the one
    // lens. (The stored entry as read would be copied as it is, whatever
    // is changed here: it goes.)
    var ext = (CMFormatDescriptionGetExtensions(videoFormat) as? [String: Any]) ?? [:]
    for key in [kCMFormatDescriptionExtension_VerbatimImageDescription, kCMFormatDescriptionExtension_VerbatimISOSampleEntry,
                kCMFormatDescriptionExtension_VerbatimSampleDescription, kCMFormatDescriptionExtension_CameraCalibrationDataLensCollection,
                kCMFormatDescriptionExtension_HorizontalFieldOfView, kCMFormatDescriptionExtension_StereoCameraBaseline,
                kCMFormatDescriptionExtension_HorizontalDisparityAdjustment] {
        ext.removeValue(forKey: key as String)
    }
    ext[kCMFormatDescriptionExtension_ProjectionKind as String] = kCMFormatDescriptionProjectionKind_AppleImmersiveVideo
    ext[kCMFormatDescriptionExtension_ViewPackingKind as String] = kCMFormatDescriptionViewPackingKind_SideBySide
    ext[kCMFormatDescriptionExtension_HasLeftStereoEyeView as String] = true
    ext[kCMFormatDescriptionExtension_HasRightStereoEyeView as String] = true
    ext[kCMFormatDescriptionExtension_HeroEye as String] = kCMFormatDescriptionHeroEye_Left
    let dims = CMVideoFormatDescriptionGetDimensions(videoFormat)
    var entry: CMVideoFormatDescription?
    CMVideoFormatDescriptionCreate(allocator: nil, codecType: CMFormatDescriptionGetMediaSubType(videoFormat), width: dims.width, height: dims.height,
                                   extensions: ext as CFDictionary, formatDescriptionOut: &entry)
    guard let entry else { die("no sample entry for the video") }

    // Reader: the video and every sound track, as stored.
    let reader = try AVAssetReader(asset: joined)
    let videoOut = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: nil)
    reader.add(videoOut)
    // Writer.
    let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)
    let venueItem = AVMutableMetadataItem()
    venueItem.identifier = .quickTimeMetadataAIMEData
    venueItem.dataType = kCMMetadataBaseDataType_RawData as String
    venueItem.value = aime as NSData
    writer.metadata = [venueItem]
    let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: entry)
    videoIn.expectsMediaDataInRealTime = false
    guard writer.canAdd(videoIn) else { die("the writer does not take the video") }
    writer.add(videoIn)
    var sound: [(AVAssetReaderTrackOutput, AVAssetWriterInput)] = []
    for track in try await joined.loadTracks(withMediaType: .audio) {
        guard let format = try await track.load(.formatDescriptions).first else { continue }
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: format)
        input.expectsMediaDataInRealTime = false
        guard reader.canAdd(out), writer.canAdd(input) else { die("the writer does not take the sound") }
        reader.add(out); writer.add(input)
        sound.append((out, input))
    }
    let spec: [String: Any] = [
        kCMMetadataFormatDescriptionMetadataSpecificationKey_Identifier as String: AVMetadataIdentifier.quickTimeMetadataPresentationImmersiveMedia.rawValue,
        kCMMetadataFormatDescriptionMetadataSpecificationKey_DataType as String: kCMMetadataBaseDataType_RawData as String,
    ]
    var hint: CMFormatDescription?
    CMMetadataFormatDescriptionCreateWithMetadataSpecifications(allocator: nil, metadataType: kCMMetadataFormatType_Boxed, metadataSpecifications: [spec] as CFArray, formatDescriptionOut: &hint)
    let metaIn = AVAssetWriterInput(mediaType: .metadata, outputSettings: nil, sourceFormatHint: hint)
    metaIn.expectsMediaDataInRealTime = false
    let metaAdaptor = AVAssetWriterInputMetadataAdaptor(assetWriterInput: metaIn)
    guard writer.canAdd(metaIn) else { die("the writer does not take the metadata") }
    writer.add(metaIn)
    metaIn.addTrackAssociation(withTrackOf: videoIn, type: AVAssetTrack.AssociationType.metadataReferent.rawValue)

    guard reader.startReading() else { die("reading \(opt.joined): \(String(describing: reader.error))") }
    guard writer.startWriting() else { die("writing \(opt.output): \(String(describing: writer.error))") }
    writer.startSession(atSourceTime: .zero)

    // The writer keeps its tracks in step by its own rule: it takes no
    // more of one while it waits for another. So only the video is
    // waited for; the sound is read well ahead of it and, like the
    // metadata, handed over whenever its input takes some. A track whose
    // data has all gone in is closed at once, or the writer waits for
    // more of it near the end.
    var pendingMeta: [AVTimedMetadataGroup] = []
    var pendingSound: [[CMSampleBuffer]] = Array(repeating: [], count: sound.count)
    var soundRead: [CMTime] = Array(repeating: .zero, count: sound.count)
    var soundEnded: [Bool] = Array(repeating: false, count: sound.count)
    var soundClosed: [Bool] = Array(repeating: false, count: sound.count)
    /// Read every sound track up to `until`.
    func readSound(until: CMTime) {
        for i in sound.indices where !soundEnded[i] {
            while CMTimeCompare(soundRead[i], until) < 0 {
                guard let sample = sound[i].0.copyNextSampleBuffer() else { soundEnded[i] = true; break }
                guard CMSampleBufferGetNumSamples(sample) > 0 else { continue }
                pendingSound[i].append(sample)
                let end = CMTimeAdd(sample.presentationTimeStamp, sample.duration.isValid ? sample.duration : .zero)
                if CMTimeCompare(end, soundRead[i]) > 0 { soundRead[i] = end }
            }
        }
    }
    /// Hand over what the inputs take. `true`: something moved.
    func hand() -> Bool {
        var moved = false
        while let group = pendingMeta.first, metaIn.isReadyForMoreMediaData {
            guard metaAdaptor.append(group) else { die("metadata: \(String(describing: writer.error))") }
            pendingMeta.removeFirst(); moved = true
        }
        for i in sound.indices {
            while let sample = pendingSound[i].first, sound[i].1.isReadyForMoreMediaData {
                guard sound[i].1.append(sample) else { die("sound: \(String(describing: writer.error))") }
                pendingSound[i].removeFirst(); moved = true
            }
            if soundEnded[i], pendingSound[i].isEmpty, !soundClosed[i] {
                sound[i].1.markAsFinished(); soundClosed[i] = true; moved = true
            }
        }
        return moved
    }

    // The frames. Packets come in decoding order; a part's packets are
    // together, so the n-th packet's part is known by counting. A frame's
    // metadata sample names its part's camera, and is held back until no
    // earlier frame can still come (the video is not in display order).
    var starts: [Int] = []
    var at = 0
    for part in opt.parts { starts.append(at); at += part.frames }
    var partStart: [CMTime?] = Array(repeating: nil, count: opt.parts.count)
    var held: [(time: CMTime, duration: CMTime, part: Int)] = []
    func metadata(for frame: (time: CMTime, duration: CMTime, part: Int)) throws -> AVTimedMetadataGroup {
        let command = PresentationCommand.setCamera(SetCameraCommand(id: frame.part + 1, time: partStart[frame.part] ?? frame.time, cameraID: cameras[frame.part]))
        let item = AVMutableMetadataItem()
        item.identifier = .quickTimeMetadataPresentationImmersiveMedia
        item.dataType = kCMMetadataBaseDataType_RawData as String
        item.value = try JSONEncoder().encode([command]) as NSData
        item.time = frame.time
        item.duration = frame.duration
        return AVTimedMetadataGroup(items: [item], timeRange: CMTimeRange(start: frame.time, duration: frame.duration))
    }
    /// Release the held frames shown before `before` (all of them when nil), in display order.
    func release(before: CMTime?) throws {
        held.sort { CMTimeCompare($0.time, $1.time) < 0 }
        while let first = held.first, before.map({ CMTimeCompare(first.time, $0) < 0 }) ?? true {
            pendingMeta.append(try metadata(for: first))
            held.removeFirst()
        }
    }

    let soundLead = CMTime(value: 2, timescale: 1)
    var frame = 0
    var part = 0
    var lastReport = Date.distantPast
    while let sample = videoOut.copyNextSampleBuffer() {
        let count = CMSampleBufferGetNumSamples(sample)
        guard count > 0, CMSampleBufferGetDataBuffer(sample) != nil else { continue }
        guard count == 1 else { die("a video sample buffer of \(count) samples") }
        guard frame < total else { die("the joined file holds more frames than its parts gave (\(total))") }
        while part + 1 < opt.parts.count, frame >= starts[part + 1] { part += 1 }
        let time = sample.presentationTimeStamp
        let duration = sample.duration.isValid && sample.duration.value > 0 ? sample.duration : CMTime(value: 1, timescale: CMTimeScale(frameRate))
        // A part's first packet is its first picture.
        if partStart[part] == nil { partStart[part] = time }
        let decode = sample.decodeTimeStamp.isValid ? sample.decodeTimeStamp : time
        readSound(until: CMTimeAdd(decode, soundLead))
        guard let packet = carried(sample, as: entry) else { die("frame \(frame): its packet could not be carried over") }
        var idle = 0
        _ = hand()
        while !videoIn.isReadyForMoreMediaData {
            if !hand() {
                try await Task.sleep(nanoseconds: 2_000_000)
                idle += 1
                // Still waiting: the writer may want sound from further on.
                if idle % 250 == 0 { readSound(until: CMTimeAdd(decode, CMTimeMultiply(soundLead, multiplier: Int32(1 + idle / 250)))) }
                if idle > 15_000 { die("the writer stopped taking video at frame \(frame): \(String(describing: writer.error))") }
            } else {
                idle = 0
            }
        }
        guard videoIn.append(packet) else { die("frame \(frame): \(String(describing: writer.error))") }
        held.append((time: time, duration: duration, part: part))
        // Nothing later is shown before this packet's decoding time.
        try release(before: decode)
        frame += 1
        if Date().timeIntervalSince(lastReport) > 0.25 {
            fputs("progress \(frame) \(total)\n", stderr)
            lastReport = Date()
        }
    }
    if reader.status == .failed { die("reading \(opt.joined): \(String(describing: reader.error))") }
    guard frame == total else { die("the joined file holds \(frame) frames, its parts gave \(total)") }
    videoIn.markAsFinished()
    try release(before: nil)
    readSound(until: .positiveInfinity)
    var idle = 0
    while !pendingMeta.isEmpty || pendingSound.contains(where: { !$0.isEmpty }) || soundClosed.contains(false) {
        if hand() { idle = 0; continue }
        try await Task.sleep(nanoseconds: 2_000_000)
        idle += 1
        if idle > 15_000 { die("the writer stopped taking metadata or sound: \(String(describing: writer.error))") }
    }
    metaIn.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else { die("finishing \(opt.output): \(String(describing: writer.error))") }
    fputs("progress \(total) \(total)\n", stderr)

    // Apple's own reading of the result.
    do {
        guard try await AIVUValidator.validate(url: outURL) else { throw NSError(domain: "aivu_wrap", code: 1, userInfo: [NSLocalizedDescriptionKey: "not a valid file"]) }
    } catch {
        try? FileManager.default.removeItem(at: outURL)
        die("the Apple Immersive Video validator refuses the file: \(error.localizedDescription)")
    }
    note("\(total) frames, \(cameras.count) cameras, \(sound.count) sound track(s): valid")
}

do { try await run() } catch { die("\(error)") }

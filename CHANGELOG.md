# Changelog

## Unreleased

### Insta360 X6: an archive copy of a raw take

A raw take is large: 40 to 80 GB a minute. The camera has to write a
gigabyte a second, and packs its samples with what a processor can do at
that pace. Under **RAW Footage Archiving**, below a raw take's facts in the
Source panel, **Compressed archive** writes the take's folder again with a
smaller capture, and the app opens that archive copy like the take: every
frame with its number and its timing, the gyro log, the lens record and the
sound beside it, the same development, stabilization and exports. There are
three sizes to pick from.

- **Lossless.** Every sample of the picture is kept as the camera recorded
  it. Most takes come out a quarter to two fifths smaller (23 to 42 % on
  24 takes of every frame size, a third in the middle). A take that is
  noise for the most part, shot at a high gain at night or holding every
  value of fourteen bits, gains little, and the camera has packed some as
  well as can be done: the copy looks at a few frames first, and where it
  would not come out smaller none is made and the panel says so. Each
  sample is predicted from its neighbours, and what is left of it is
  stored with the odds it has where the picture is that restless. Before a
  frame is written it is read back and compared with the frame it was made
  from, sample for sample; the frames developed from a copy of a whole
  take are the frames developed from the take, byte for byte, stabilized
  or not. A frame of the copy is read in about a quarter more time than
  the take's own (13 ms against 10.5 ms for a pair of 5760 × 5760 on an M5
  Max).
- **High quality** aims at 1 bit a pixel, where a take holds about 3: 13 to
  16 GB a minute for the frames the camera records. **Custom** takes any
  size from 0.5 to 1 bit a pixel. The panel shows what the copy comes to.
- **What the two smaller ones lose.** They are lossy. A sample's value is
  first turned into one of 1024 codes that follow the sensor's noise,
  measured on the take: two values the noise cannot tell apart share a
  code, two it can are several codes apart, in the shadows as in the
  highlights. Each of a frame's four colour planes is then stored as a
  picture by the graphics hardware's HEVC encoder. At 1 bit a pixel a
  sample comes back off by about its own noise or less: mid-tones look as
  they did, and a deep shadow lifted hard keeps a grain, a coarser one.
  Towards 0.5 bit lifted shadows turn to flat patches and fine texture
  softens. Saturated samples stay saturated, and none becomes one.
- **The corners outside the image circle.** A frame that holds a lens's
  whole view is larger than the lens's picture: its four corners, 13 to
  16 % of its samples, lie outside the image circle, where no light of the
  scene falls, only stray light and the sensor's noise. A copy of either
  kind stores them as black. The circle is the lens's own: its centre from
  the take's lens record, its radius measured on takes of three frame sizes
  (the picture fades out 4050 to 4080 calibration pixels from the centre,
  on both lenses alike; black begins a little past that). A lossless copy
  of a 5504, 5760 or 6000 frame comes out 8 to 10 % smaller for it, a
  lossy one 6 to 14 % smaller at the same setting; a 3840 frame gains 1 to
  9 %, and a frame that is the middle of the sensor has no such corners.
  The panel says how much of a frame they are before the copy is made.
- **A stretch, or the whole take.** A trimmed take is copied from In to
  Out, or whole: leaving out what is not wanted of a take often saves more
  than the packing does. Before the copy is made the panel says how much
  of the take goes into it and how much is left out before In and after
  Out, and what the take, the part copied and the copy come to. A copy of
  a stretch is a shorter clip: what the app takes from a whole clip (the
  tone statistics, the stabilization's reference) it then takes from that
  stretch.
- **What is kept.** Every picture stands alone: the copy seeks and plays
  like the take. The take is not touched, and a copy is made from a
  camera's take, never from another copy. A clip that is a copy says so
  under its name: the bits a pixel it came to, or that every sample of the
  picture is kept, that its corners are black, and which frames of its
  take it holds when that is not all of them.
- **The take first, its copy when the take is gone.** A clip is its take.
  Once a copy is written the panel says that the take can be moved away
  or deleted: by you, the app moves and deletes nothing. When a take in
  the list or in a project is no longer where it was, the app reads its
  copy in its place: a folder beside where the take was, or in the output
  folder, named after the take and saying that it was made of it (of
  several, one that keeps every sample first, then the one that holds the
  most of the take). The clip keeps its settings, its trim lies in a copy
  of a stretch where the stretch lay in the take, and the clip's facts say
  which take the copy stands in for. A take that comes back is read
  again. A copy holds every file of its take's folder, and goes on
  reading the take's folder of camera original DNGs where that is beside
  it, so frames enhanced for the take are still used.
- **The file.** The copy's capture is a file of records like the take's, a
  record a pair (the format is in `x6b.rs`'s notes, the lossless pictures
  in `raw_pack.rs`'s); earlier versions of the app refuse it as a kind
  they do not know.

### Insta360 X6: a camera mounted upside down

Checked for the first time on footage that was shot upside down. The
picture, the order of the eyes and the stabilization were right with the
**Upside-down mount** switch on; four things around it were not.

- **The switch sets itself.** An X6 clip or raw take says which way up the
  camera was: its accelerometer reads one g toward the top of the picture
  or toward its bottom. The switch follows that when a clip joins the list
  or comes back with a project, and says so. Set by hand it stays as you
  left it (**Follow the accelerometer** puts it back), and it is no longer
  passed on to the next clip. A still has no such record: its switch is
  set by hand as before.
- **The stereo alignment stays with its lens.** The stereo offsets are a
  rig's alignment, each lens's own small rotation, and upside down the two
  eyes change lenses. The offsets used to stay with the output eye, so
  each lens got the other's: on the footage the eyes came out 0.4° apart
  in roll and 0.1° vertically, twice the rig's own error. They now go with
  the lens (0.03°, as upright), and one alignment serves both ways a
  camera can be mounted. **Auto align** asked for pitch and yaw the wrong
  way round upside down; it now gives the same values either way up.
- **The view angles act as you see them.** Pano yaw and pitch, and the
  reframed view's pan and tilt, turned the view the opposite way upside
  down.
- **The sound is turned with the picture.** The microphones are turned
  over with the camera, so left and right, and up and down, were reversed
  against the picture. Stereo is written with its sides changed, the
  360° Audio field turned half a turn about its front axis (Ambisonic is
  encoded again for that instead of copied; APAC is made from the turned
  field), and the preview plays it the same way. This follows from the
  geometry and was checked on the files' channels, not by ear.

### Insta360 X6: raw video

- **The take itself is written from the Source panel.** An archive copy
  and the **Camera original DNGs** (every frame as the sensor recorded it,
  until now a kind of DNG sequence in Export format) are not views of the
  take: no size, grade, stabilization or noise reduction applies to them,
  and what they write is read again as a source. Each has its own button
  under the take's facts, under the heading RAW Footage Archiving
  (**Compressed archive** and **Camera original DNGs**). A run is of the
  take on screen, and its start button says so; takes ticked in the list
  besides it have a start button of their own under it, each with what
  would be written of its takes. A run goes from In to Out or whole,
  through the export queue (progress, Skip and Stop in the export bar),
  and leaves what was exported or waits to be as it found it. Export
  format keeps the take's views: a video, or
  the developed view as a DNG sequence, so Export never writes anything
  else. A project saved with Camera original picked exports a video
  again.
- **A take opened together with its archive copy is opened alone.** A
  folder that holds both gives the take; the copy is left out and named in
  the note. A copy opened by itself opens as a clip of its own.
- **A 5760 × 5760 frame stored to within one code value.** A take with a
  record for each eye may store its 5760 × 5760 frames that way too, the
  whole view or the middle of the sensor. 2.6.0 read that frame only with
  every value kept, and refused such a take as one it did not know. It
  opens now, like a take of 6000 × 6000.
- **Enhanced frames in the Lens Space view.** A take whose frames were
  enhanced in a raw converter has the two larger sizes in the Lens Space
  view as well, **13120 × 6560** and **16384 × 8192**, under the choice
  the VR180 view keeps for them. Until now a lens frame was written at
  the take's own size whatever its frames held. At either size the frame
  is the take's frame scaled to it, the lens in the file scaled with it,
  and a stabilized frame keeps the size with the stabilizer's margin
  inside it. A stretch without enhanced frames exports at the size picked
  without them, and the export bar says so. Checked on a take, stabilized
  and not: both sizes lie on the frame of the take's own size to a
  twentieth of a pixel, and hold detail finer than that frame can; 0.53
  and 0.63 s a frame pair on an M5 Max.

### Joined files: enhanced frames in every clip or in none

- **All of the clips, or none.** A file joined from several clips reads a
  raw take's enhanced frames only when every clip of the run has some for
  its stretch. One clip without them (a take whose frames were not
  enhanced, a stretch past the enhanced ones, a clip of another camera)
  leaves them out of the whole file: every clip is then read from the
  capture, at the size picked without enhanced frames, so the file keeps
  one look and one frame size from its first clip to its last. The run
  starts by looking through the takes' folders (the export bar says so,
  for a second or two), and where enhanced frames are left out although
  some clips have them, the bar and the run's summary name the first clip
  that has none.
- **The frame each clip is really written at.** Before a joined run starts
  its clips are compared at the size their exports write, not at the size
  asked for. That differs for a Lens Space clip through the camera's lens,
  whose frame is 5 % larger on each side when the clip is stabilized, and
  for a take read from enhanced frames at a size picked for those. Two
  such clips used to pass the check and were found to differ only when
  the finished parts were put together, after every clip had been
  exported; they are now named before anything is written, and for two
  Lens Space clips of which one is stabilized the note says that this is
  what sets them apart.

## 2.6.0 — 2026-11-24

### Insta360 X6: raw video

A raw take of the X6 is a folder: the sensor's own 14-bit Bayer data of both
lenses for every frame, the gyro log, the factory lens record and the sound.
The app opens such a folder as one clip (add the folder, the capture file in
it, or a folder of takes) and treats it like any other video: preview, trim,
stabilization, eye alignment, every output format. What is new is the
developing: a raw clip has no look of its own, so the Color panel gets a raw
converter's controls for it, and every frame can be had as a DNG.

- **Other frame sizes.** A take whose eyes are 5504 × 5504, 4912 × 4912
  or 3968 × 3968 opens like one of 3840 × 3840: the same preview,
  stabilization, development and exports, the Camera original DNGs
  included. The 5504 frame holds the lens's whole view. The 4912 and the
  3968 frame hold the middle of it at the sensor's own pixels: outside the
  frame the picture is black, the Crop mask applies, and the Resolution
  list offers the native-sharpness size, as for a Crop video clip. Every
  frame is placed in the lens's view, and its rows are timed, by the
  take's own sensor record.
- **A record for each eye.** A take that stores each eye's frame as a
  record of its own opens like the others: eyes of 5760 × 5760 or
  6000 × 6000 at 30 fps, or 4608 × 3968 at 60 fps, with every sensor
  value kept or stored to within one code value. The two sensors' frames
  are paired by when their exposures started; a frame whose partner was
  never written is left out, and the gap is filled like any dropped
  frame. The two square frames hold either the lens's whole view or the
  middle of it at the sensor's own pixels, and the 4608 × 3968 frame is a
  wider than tall part of the middle: a frame that is part of the view
  gets the Crop mask and the native-sharpness size. Every record's
  checksum is checked as the record is read, so damaged data stops an
  export instead of ending up in it.
- **The camera's white balance.** A take that records the white balance
  the camera had found when it began is developed for that white: As shot
  is then the take's own, and each eye's Camera original DNGs carry its
  sensor's neutral. A take without that record keeps the reference
  neutral, as before.
- **Raw development.** Temperature and Tint (or As shot), Exposure,
  Contrast, Highlights, Shadows, Whites, Blacks, in a raw converter's own
  terms. The rendering follows Adobe Camera Raw's, control by control, so a
  frame graded here and the same frame's DNG opened there with the same
  numbers look alike. It was measured there, on charts and on frames of
  sample takes, not taken from a description:
  - with nothing moved the two differ by a third of an 8-bit step on
    average, and a white balance by a tenth to seven tenths;
  - Shadows and Highlights are local, as there: they lift or hold back an
    object as a whole and leave the texture inside it alone, they belong to
    the picture (what counts as a shadow follows the clip's own levels), and
    moved together they do not simply add. Half the slider is within about
    one step, the whole slider within one and a half to two and a half;
  - Blacks and Whites below zero are within half a step to one; Contrast
    within about one step at half the slider and two at its ends; Whites
    above zero within one to two and a half steps up to the middle of the
    slider and coarser beyond it;
  - a grade with every slider in use lands within one to two steps for
    moderate amounts, three to five at the extremes;
  - a highlight keeps its colour where one of the sensor's channels clips.
    A clipped channel is rebuilt from the other two, and how was first
    taken from neutral highlights only: a clipped green was raised to the
    red beside it. On a neutral that is right. On a sunlit wall, orange
    under the white balance in use, it turned the brightest parts into
    flat, hard-edged patches of pure yellow. The rule is now the one
    Camera Raw itself goes by, measured on charts of coloured patches
    stored clipped (three white balances, a chart whose red clips first,
    a mosaic and a linear file, per-pixel noise: 0.7 % rms over a thousand
    patches): a channel's own reading counts for less the nearer it is to
    its clip point, the channel that clips first rises toward the mean of
    the other two, and of those two the lower rises toward the higher. On
    the take that showed it, the clipped-yellow pixels of a frame went
    from about a thousand to none, and the wall's brightest parts are
    within one step of Camera Raw's with the exposure raised.
  - leaves against a sky keep their contrast when Highlights is pulled
    down. The gain went pixel by pixel: every bit of sky between the
    leaves was pulled down as far as the open sky and the leaves were
    left where they were, so a tree's edge turned into a pale, flat haze
    in which the lens's colour fringes were the only thing left to see.
    Camera Raw does not do that, and what it does was measured on charts
    of bright and dark mixed at sizes from one pixel to sixteen and of
    steps placed at every position: its gain is held on cells of four
    pixels of the picture (whatever the picture's size), fades toward an
    edge on the side the slider moves and is exactly zero on the other.
    The gain is now smoothed the same way: where bright and dark are
    finely mixed the bright pixels are pulled down by about the share of
    bright among them, the dark ones stay, and the mixture keeps its
    contrast; a large area is treated as before. Shadows gains the
    mirror of it. On the charts the gain is a third nearer Camera Raw's
    (0.12 to 0.08 stops for Highlights, 0.28 to 0.17 for Shadows); on
    the frame that showed it the sky between the leaves went from 6 to
    10 steps too dark to within 3; on four other frames Highlights is
    as near as it was and Shadows a little nearer. A clip without the
    two sliders pays nothing; with them the whole local stage is about
    8 ms of a frame at 8448×4224.
- **Demosaic.** The step that makes a full-colour picture of the sensor's
  mosaic is a new one, shaped after Camera Raw's: the same frames were
  developed there and here, and the method that reproduces its result was
  kept. Edges come out clean, with no alternating pattern along them and
  no purple and yellow in fine lines; the preview, which used to merge
  2×2 sites and showed colour fringes for it, now shows the full method's
  picture reduced. Developed, a frame differs from Camera Raw's by a
  quarter of an 8-bit step on average where it was one to three before.
- **Colour noise reduction.** A raw converter also smooths the picture's
  colour by default, and that is what frees fine lines of the last colour
  a demosaic leaves in them and takes the coloured speckle out of noise.
  The Color panel has it now, under Noise reduction: one slider, 0 to 100,
  starting at 25 as Camera Raw's does. Luminance is left exactly as it is;
  a real colour edge is not smeared. It was fitted to Camera Raw's own
  result at 10, 25, 50 and 100: it takes out 0.9 to 0.95 of what Camera
  Raw takes out up to 50 and the two changes correlate at 0.88 to 0.95;
  at 100 Camera Raw reaches further (it flattens faint colour over tens
  of pixels). Costs about 10 ms a frame at 8192×4096. The setting travels
  in the DNG. A clip graded before this gets the default 25. A bright
  pixel no longer lends its colour to a much darker one beside it: the
  smoothing weighed its neighbours by their values, so a sky coloured
  the edge of every leaf in front of it. Neighbours now count by how
  near their brightness is as well (measured on Camera Raw with edges
  of light: nothing crosses a step of a stop, and the mixing dies away
  over about a fifth of a stop).
- **Sharpening.** The detail panel's four controls, Amount, Radius, Detail
  and Masking, on the developed picture, done the way Camera Raw does it:
  on the luminance alone, fading out toward black and toward white, held
  back at strong edges. The kernel was read off Camera Raw itself for 1386
  combinations of Amount, Radius and Detail. Low Detail holds halos back
  as it does there: what the sharpening adds beyond the range of the
  picture around a pixel is cut to a third at Detail 0, less and less up
  to Detail 25 or so, while the steepening of an edge between its two
  sides is kept whole. Measured on frames of sample takes as the change
  each makes to the same frame:
  - from Detail 25 up the two changes correlate at 0.96 to 0.99 and differ
    by 12 to 34 % of their size (least at high Detail), with the overall
    strength within a tenth;
  - at Detail 0 to 20 they differ by 16 to 36 % with the strength within a
    twentieth (a Radius of 3 at Detail 0: about 50 %);
  - with Masking in use, 23 to 40 %;
  - with any Amount above zero Camera Raw also smooths the finest texture a
    little before it sharpens, most at low Detail and large radii. That
    part is not reproduced, and it is most of what is left at low Detail:
    the finest texture and noise stay a little crisper here.

  The radius is in pixels of the exported frame, so the preview shows the
  export's picture reduced: judge sharpening in the zoom view or on a
  snapshot. It adds about half to the export time at full size. The
  settings travel in the DNG.
- **One development for the whole clip.** The rendering is derived from
  the picture: its brightest and darkest levels and the spread of its
  levels. A raw converter reads them off each picture it is given. A clip
  developed that way follows its own content: when a pan lets more sky in,
  everything else is rendered differently, and a clip shot at one fixed
  exposure brightens and darkens by itself. (On a test take with
  Highlights at −100, a patch of sky whose recorded values did not move
  went from 130 to 169 of 255 within seven seconds, most of it in one.)
  So they are read once per clip, off sixteen frames spread over it and
  taken together as one picture, and every frame is developed by them: the
  same patch then stays within one step (148 to 149). The preview, a
  snapshot and the export read the same sixteen frames and agree; a change
  of white balance is read off them again without a frame being decoded.
  Sampling takes a few seconds when a clip is first opened or exported;
  until it is in, the preview measures each frame by itself. The DNG of a
  frame opened in a raw converter is developed from that frame alone, so
  it matches the clip's frame where the frame is like the rest of the clip.
- **Automatic exposure.** Under Exposure, **Auto exposure** lets the clip
  set its own exposure, frame by frame. A frame gets the exposure at which
  its picture comes out normally bright as the clip is developed, with
  every other control where it stands: the picture is metered as it will
  be shown (white balance, Highlights, Shadows and the rest included),
  weighted toward where the lens looks. Pull the highlights down and the
  frames are exposed up to keep the picture where it was. Two controls go
  with it. **EV** adds stops to what it finds. **Smoothness** is how many
  seconds of the clip, on either side, a frame's exposure is averaged
  over: 0 follows the light as it changes, a few seconds follow it the way
  an eye does, a large value gives the whole clip one exposure. Because
  the whole clip is known, the exposure moves ahead of a change as much as
  after it, and never jumps. The exposure is the only thing that moves:
  the development stays one for the whole clip, a fixed rendering of
  exposed values, as if the camera had exposed each frame so. A clip
  given one exposure this way is developed exactly as that exposure set by
  hand develops it. The clip is metered at half-second steps when the
  control is first switched on (ten to twenty seconds for a take of two
  and a half minutes on a network share; the panel says how far it is).
  After that every control answers at once. The Exposure control shows the
  exposure of the frame on screen, the preview and the export read the
  same exposures, and a frame's DNG carries its own. At EV 0 the target is
  the middle of ten daylight takes graded by hand, which themselves lie up
  to a stop either side of it: a meter does not know what a picture is
  meant to look like, and EV is for that. A frame the lens saw nothing in
  is not lifted more than three stops beyond the clip's middle.
- **One grade, one look, in every view.** A reframed view shows a part of
  the scene and a Lens Space frame has a black border, so measured on their
  own they would be developed differently from the VR180 picture (with
  Shadows and Highlights in use, by as much as 13 steps). They are developed
  from the half-equirect of the same frame instead, and the three views
  agree to within about half a step. A raw converter given the DNG of such a view develops
  it from that picture alone, so there the views still differ; the DNG of
  the VR180 frame is the one that matches.
- **Every frame is a DNG.** Snapshot writes a linear DNG beside the JPEG:
  the projected frame in the camera's own linear colour, with the panel's
  settings inside, so another raw converter opens it as it looks here. In
  the export bar a raw clip can be written as a **DNG sequence** (a folder
  of numbered frames) or as video; video is 10-bit.
- **The camera's own frames as a DNG sequence.** The DNG sequence has a
  second form, **Camera original**: every frame exactly as the sensor
  recorded it, one 3840×3840 Bayer DNG per eye (`CAM0/000000094.dng`,
  `CAM1/000000094.dng`, named by the camera's frame number), with no
  projection, no stabilization and no grading in it. It is for work that
  has to happen on the raw frames before anything else, such as running
  every frame through a raw converter's enhancement. The files are
  losslessly packed (about 12 MB a frame per eye at 30 fps, as much as the
  capture itself), carry the camera's colour description, and open in
  Adobe Camera Raw and in macOS. The take's small files (lens record,
  gyro, sound) and a list of the frames with their exposure times are
  written beside them. Trim points apply. From a fast disk a one-minute
  30 fps take (1800 frame pairs, 43 GB) writes in about half a minute.
- **One frame as the camera recorded it.** Beside Snapshot, a raw clip has
  **Original DNG** (Shift+S): the frame on screen as its two Bayer DNGs,
  the files the Camera original sequence holds for that frame. They go
  where the snapshot goes and under its name, one per eye:
  `<clip>_frame<N>_CAM0.dng` and `…_CAM1.dng` in the export folder (beside
  the clip when none is set). A frame written again takes the next free
  name, as a snapshot does, so a file adjusted in a raw converter is never
  written over. Playback pauses on the frame.
- **Frames enhanced in a raw converter come back in.** Camera Raw's
  Super Resolution can be run on the Camera original DNGs, and the export
  then uses its result. Camera Raw 18 enhances a DNG in place: the file
  keeps its mosaic and gains the enhanced picture, the frame demosaiced
  and enlarged twofold (7680×7680 an eye) in the camera's own linear
  values. With **Use enhanced frames** on (Export format, on by default)
  and the folder of Camera original DNGs beside the take, every frame of
  an export is taken from that picture where the folder has one, and from
  the capture, enlarged to match, where it has not. An export whose
  stretch of the take holds no enhanced frame runs on the capture as
  before, and the export bar says when enhanced frames are in use.
  Enhanced frames hold more than 8192×4096 shows, so a take that has
  them gets two larger sizes, **13120×6560** and **16384×8192**, with a
  choice of their own: Export format lists them once the frames are
  found and says what it found, and a stretch without enhanced frames
  exports at the size picked without them. Both sizes encode in hardware
  on Apple silicon (0.60 and 0.69 s a frame pair, 5.7 and 7.2 GB); like
  every size above 8192×4096 they are beyond what most players decode.
  Preview, grading and stabilization stay on the capture; lens, timing
  and sound are the take's. So the whole round trip is: export Camera
  original, enhance the folder in Camera Raw, export the video. Read from
  a frame enhanced by Camera Raw 18.7; its values agree with Camera Raw's
  own demosaic of the frame to a part in a thousand. About 0.45 s a frame
  pair and 4 GB of memory at 8192×4096. macOS only for now: the enhanced
  picture is JPEG XL, which this build decodes through the system. On
  Windows the option is replaced by a line that says so and a take exports
  from the capture; the settings Camera Raw left in the files are read
  there as on a Mac.
- **Windows.** A raw take opens, previews and exports there as well
  (checked on an NVIDIA RTX 4090, on Vulkan and on Direct3D 12), and the
  Camera original DNGs it writes are byte for byte the ones a Mac writes.
  An 8192×4096 export runs at about 10 frames a second with tone and
  sharpening in use and 15 with the default development, through NVENC
  or as ProRes.
- **Rolling shutter and the accelerometer.** A take that records the
  sensor's row timing and the accelerometer is stabilized with both.
  Each frame's pose is taken where its centre row starts exposing (the
  frame's stamp plus half the readout the take states: 14.6 ms at 30 and
  60 fps, 15.4 ms at 50), every row is corrected for its own time, top
  row first, and the poses are built with the accelerometer as the
  camera's own video files' are, which keeps a long take from tilting.
  Measured on six hand-held takes against the picture itself: the readout
  the take states is the one the picture shows, and on the hardest of
  them the far-field shake of a locked export fell from 1.5 to 0.4 pixels
  a frame at 1024 pixels an eye (2.6 without stabilization). A take
  without the two stabilizes as before.
- **Auto align.** The View adjustment panel's Auto align works on raw
  takes: the two eyes' camera-linear frames are turned into a picture the
  matcher can read (white-balanced, lifted, on a display curve) and
  measured as any other clip's. On two takes of one camera it gives the
  same pitch and roll to 0.01°, and a known offset put on the sliders
  comes back in the measurement to 0.005°.
- **Settings from Camera Raw.** Raw development has **Read from Camera
  Raw**: the white balance, the six tone controls, the sharpening and the
  colour noise reduction a frame was left with in Camera Raw become the
  clip's. They are read from the Camera original DNG of the frame on
  screen, or of the nearest frame that has settings, in the folder beside
  the take. Whatever else was set there (texture, clarity, a curve, the
  colour mixer, masks, a crop, another profile and so on) is named under
  the button and stays in Camera Raw. The section also no longer turns an
  as-shot white balance into a custom one of the same value just by being
  shown.
- **Noise reduction.** The Noise Reduction section (macOS) works on raw
  takes. The filter is a video filter, 10-bit and with colour at half
  resolution, so a take's frames pass it white-balanced and on a
  logarithmic curve and come back as the camera's linear values: tone and
  colour stay where they were (within 0.1 of 255 on a dusk take), grain in
  flat areas halves at strength 0.7, the strongest edges keep 97 % of
  their contrast and a clear sky gets no banding. It costs about 0.3 s a
  frame pair.
- **Shutter time.** A take that lists each frame's integration time
  (`exposure.csv`) has its poses taken at the centre row's mid-exposure:
  half the frame's own shutter time later than where the row starts. Its
  frames are also timed by when the sensor read them out, which is even,
  and not by when their exposures began, which moves with every change of
  shutter: a take whose shutter dropped from 16.4 to 7.9 ms in one step
  no longer reads as having lost a frame there. Takes without the list
  are sampled where the row starts, as before, which on a hand-held take
  leaves the 0.4 pixels above (0.06 with the 3 ms those takes turn out to
  need).

### Color: Exposure, Contrast, Highlights, Shadows, Whites and Blacks for every clip

The six tone controls a raw clip has in its development are now in the
Color panel of every video clip and still, under **Tone**. Each of them is a
change of exposure made in linear light, ahead of Lift / Gamma / Gain and
the LUT, and not a curve laid over the picture the LUT left.

- **Why ahead of the LUT.** A log clip's LUT presses the top stops of the
  recording into a narrow band below white. A control that works on the
  picture after it can only darken that band. Here Highlights takes the
  light itself down and the LUT then renders what the file holds. Two
  highlights one stop apart in the scene, both on the shoulder of the X6's
  own LUT, are shown 1.5 % of the range apart; with Highlights at −100
  they are shown 4.2 % apart, where the same pull made on the picture
  after the LUT leaves 0.9 %.
- **Which light.** An Insta360 I-Log clip is decoded with the camera's log
  curve: the light is the scene's and a stop is a stop of exposure. The
  ETTR recovery of such a clip is made by the same stage. Every other clip
  (D-Log M, GP-Log, HLG, plain video, a still) is worked on in the light
  its picture shows: the stage looks through the clip's own Lift / Gamma /
  Gain and LUT to know what a value is displayed as, changes that light,
  and writes the value that displays the result. A neutral tone then moves
  exactly as the displayed light says, whatever LUT is loaded.
- **The controls are the raw development's.** Shadows and Highlights are
  local: they lift or hold back an area as a whole and leave the texture
  inside it alone (leaves against a sky keep their contrast when
  Highlights is pulled down: the gain is smoothed over cells of four
  pixels of the exported frame, as in the raw development), and what
  counts as a shadow follows the picture's own levels. Whites, Blacks and
  Contrast follow those levels too. For a video
  clip they are the levels of the whole clip, read once off sixteen frames
  spread over it (both eyes together, so the two eyes are toned alike): a
  control set for the clip then does the same to every frame, and the
  picture does not brighten or darken with what each frame happens to
  hold. A still is measured by itself. The numbers are not Camera Raw's
  for a video clip: the picture is rendered by the camera's LUT, not by a
  raw converter's curve.
- **Nothing moved is nothing run.** With the six at zero the stage is not
  part of a clip's rendering at all: the 65 stored exports and 169 stored
  preview frames of the test set came out identical to the bytes they had
  before the controls existed.
- **Lift / Gamma / Gain** are as they were. The Shadow and Highlight
  sliders that sat with them are replaced by the two above; a clip graded
  with them keeps its values, shown as "earlier", until that section is
  reset.
- **Cost.** Exposure alone costs nothing. A control that goes by the
  levels has the clip sampled once, a few seconds before the first frame;
  no frame is measured after that.
- **Not offered** for a raw clip (its development has the six), for
  Blackmagic RAW (the SDK's own controls), and for a still written as a
  linear DNG (a DNG has no place for them).
- **Limits.** A reframed view and a Lens Space frame are measured as they
  are drawn, not from the whole half-sphere. Black outside the picture (a
  Crop mask, a lens frame's border) is lifted by Blacks above zero, as Lift
  lifts it. The Matching Eyes exposure trim acts after the stage.
- **Windows.** Checked there too (an NVIDIA RTX 4090): X6, Osmo 360 and
  GoPro `.360` clips through the hardware and the software export paths
  and as ProRes, 10 and 8 bit.

### Insta360 X6: dropped frames shown

The X6 leaves a hole in a clip where its sensor lost frames, in its video
files and in its raw takes alike. The app has held the frame before each
hole since 2.5.5, in the preview and in exports; it now also says where
they are.

- **File info** has a **Dropped frames** line for an X6 clip: `none`, or
  how many frames in how many places (`9 in 9 places`), in orange. Pointing
  at it lists the places with their times.
- **The timeline** marks every place in orange, as wide as the hole; the
  pointer on a mark says when and how many.
- Read from the clip's own frame times (the video track's table, a raw
  take's frame stamps), with the rest of the file info, at no cost to
  loading. Checked on the takes at hand against the camera's own count of
  frames it never saved: nine in one take, none in the others, as it says.

### Insta360 X6: ETTR recovery for I-Log clips

A clip exposed brighter than normal on purpose (exposing to the right: more
light on the sensor, cleaner shadows) has to come back down before its LUT.
Doing that with Gamma or Gain bends the colours, because those act on the
I-Log code values. The Color panel now has **ETTR recovery (I-Log)**: pick
how far over the clip was shot (**0 / +1 / +2 / +3** stops, or type an
amount in between) and that much exposure is taken back out in the light the
sensor recorded, ahead of the tone controls and the LUT.

- **An exposure change, not a curve.** Each channel is decoded with
  Insta360's published I-Log transfer function, scaled, and encoded again:
  the three channels keep their proportions, black stays where it is, and
  the LUT (the built-in one or your own) sees what a normal exposure would
  have given it. No extra LUT files are involved, and with no LUT the export
  stays I-Log, that much lower, for grading elsewhere.
- **Per clip, off by default.** A file does not record how it was exposed,
  so every clip starts at 0. "Apply settings to all" carries the choice to
  the other I-Log clips. Only X6 clips recorded in I-Log show the control.
- **Clipped highlights stay clipped**: what the sensor did not record does
  not come back, it only gets darker.
- **Checked**: the 10-bit path follows the formula to half a 16-bit step for
  every input code, and exports through each path (10- and 8-bit, hardware
  and software encoder, ProRes) land within one 10-bit code of it on
  average.

### Lens Space: the camera's own lens, for Apple Vision Pro

A VR180 file is a 180° canvas. A fisheye that sees more than that, as the
lenses of a 360 camera do, loses what lies past the canvas's edge. A Crop
clip of the Insta360 X6, which covers about 97° or 126°, sits in the middle
of a canvas that is half to two thirds black, and the canvas has to be far
larger than the recording to keep its detail: 13120 × 6560 for a recording
of 9824 × 4912. The **Fisheye** output format is now **Lens Space**, and
writes each eye the way the camera recorded it: in the camera lens's own
projection, in the recording's own frame, with that lens stated in the file
(Apple Projected Media Profile, parametric immersive projection), so the
headset shows every pixel at its own angle.

- **Every video source**: Insta360 X6 in all of its recording modes, DJI
  Osmo 360, QooCam 3 Ultra, Blackmagic RAW, GoPro `.360`, and a
  side-by-side file whose halves are fisheye images. A side-by-side file
  that is already VR180 has no lens, and exports as VR180.
- **The lens's whole view.** A full fisheye frame keeps its field past
  180°: about 200° across the frame of an X6 at 8K or of an Osmo 360, 213°
  for an X6 at 10K, and more towards the frame's corners. A GoPro `.360`
  holds 184.5° per lens; its frame is a fisheye of that field at the
  file's own pixel density. Around the picture the frame is black.
- **No black canvas for a Crop clip.** The whole frame is picture, at the
  density it was recorded: 9824 × 4912 for a 10K Crop clip (56 % of the
  pixels of its 180° canvas), 7424 × 3712 for a 7.4K Crop clip (32 %), each
  with all of its detail.
- **The clip's own size.** A lens frame is as large as the recording
  (7680 × 3840 for an 8K X6 or an Osmo 360) and is never enlarged; the 8K
  choice writes a reduced copy of a frame that is larger than that.
- **Everything else as before**: stabilization, rolling-shutter correction,
  eye alignment, the crop mask and colour are applied the same way. Both
  eyes are rendered into one common lens with the camera lens's own radial
  law, so the picture is resampled at very nearly one to one. The profile
  describes one family of lenses; a camera whose lens record is of another
  kind gets the lens of that family that follows it most closely (within
  three pixels over the whole field of an Osmo 360).
- **Stabilization has no spare canvas here**, so a stabilized frame is 5 %
  larger on each side than the recording: 8448 × 4224 for an 8K X6 or an
  Osmo 360 clip, 8160 × 4080 for a 7.4K Crop clip, 10800 × 5400 for a 10K
  Crop clip, the same for every clip of a mode. Where the picture does not
  reach, the frame is black, as around the picture of a VR180 export; a
  shift beyond the margin is cut at the frame's edge (under 4 seconds of a
  rough one-minute hand-held walk on a 7.4K Crop clip). With stabilization
  off the frame is the recorded one. An 8K copy keeps its size and scales
  the picture instead. A crop mask is measured from the frame's own edge
  (the larger one, on a stabilized clip): its window can be as wide as the
  frame, and gives a steady edge once it is small enough to stay inside
  the moving picture. Camera lock suits a camera that stays put.
- **On Windows**, the NVIDIA encoder takes frames up to 8192 pixels a
  side. A lens frame without stabilization fits for an 8K clip (7680 ×
  3840, about 30 frames a second on an RTX 4090); the stabilized frame of
  the same clip is 8448 × 4224 (8656 × 4328 for a GoPro `.360`), which it
  refuses. So on Windows the 8K choice, the default, writes that frame
  scaled to **8192 × 4096**: the whole stabilized field, 3 % smaller than
  recorded (5 % for a `.360`), through NVENC at about 25 frames a second.
  The clip's own size is still there and keeps every pixel: as H.265 it is
  encoded in software, at about 2 frames a second, which the export bar
  and Export format say; as ProRes it stays on the GPU (about 15 frames a
  second). On a Mac the hardware encoder takes the frame and nothing
  changes.
- **The preview is the export.** In Lens Space the preview draws the
  frame that is written: its shape, the stabilizer's margin, the crop mask
  where the export has it.
- **Apple Vision Pro only** (visionOS 26 or later). No other headset and
  not YouTube read this form, so VR180 stays the default format and is
  unchanged: export a clip in both and you get `…_SBS` and `…_Lens`.
- **The equidistant disc is still there.** Lens Space has a **Lens**
  choice per clip: **Camera lens** (the default, everything above) or
  **Equidistant**, the 195° disc in a square frame (185° for a GoPro
  `.360`) that players and tools for fisheye video expect, as the Fisheye
  format wrote it. With the Apple Vision Pro metadata on, a disc now
  states its own lens too: it used to be tagged as a half-equirect, which
  the headset showed with the wrong geometry. YouTube's metadata has no
  fisheye form, so nothing is written for it on a disc. A clip whose
  settings were saved in the Fisheye format opens with the camera's lens:
  pick Equidistant to get the disc back.
- **Checked**: for every source kind, the system's own playback mesh,
  built from the lens in the file, lands within a thousandth of a pixel of
  what the app rendered, out to 113° off the axis; the picture agrees with
  the VR180 export of the same frames, re-projected independently, to 0.2
  of a pixel; the lens boxes are byte for byte what Apple's writer
  produces. Every existing export is frame-identical, and so is every
  preview outside Lens Space; a Lens Space preview lies on the exported
  frame, margin and mask included. Crop clips have been watched on the
  headset; a frame wider than 180° has not yet.

### macOS: the preview can match QuickTime

On a Mac, an export played in QuickTime, in Safari or on Apple Vision Pro
looks a little brighter than the app's preview of the same frame. Neither is
wrong about the file. Both are colour-matched to the display; the difference
is the tone curve the picture is read with. Apple's players read a Rec.709
video with their own Rec.709 curve, and the preview window is treated as
sRGB. The mid-tones end up about 10 of 255 apart.

**Match QuickTime**, in the toolbar, shows the preview the way those players
show the export.

- **A video frame is read with the curve the system uses for Rec.709
  video** (a power of 1.961 with a short straight toe near black) on its
  way to the screen. Photos are sRGB already and are shown as before.
- **Checked against the system's own compositor**: Core Animation's
  renderer, drawing into a display's colour space, puts a decoded Rec.709
  video frame and the matched preview of it within 0.6 of 255 of each other
  on every one of 96 test colours (greys from black to white, primaries,
  skin tones). With the switch off they are up to 11 apart.
- **Only the preview changes.** The export, and the values in it, are the
  same with the switch on or off.
- **Off by default**, and remembered. Off is the preview as it always was,
  which is closer to how most players on Windows show the file.
- macOS only.

### Clip list: several clips as one file, projects, the list kept between launches

The clip list was a queue of separate exports, and it was gone when the app
closed. It is now also a simple timeline, and it is kept.

- **Join into one file.** With that box ticked in the export bar, "Join
  all" (or "Join selected") exports the clips in the order of the list and
  writes them as one file, one after the other, with hard cuts. Every clip
  keeps its own trim, stabilization, grade and lens settings. A clip is
  dragged to another place in the list by the handle at its left.
  - Each clip is exported once, the usual way, and the results are put
    together without encoding the picture again: a cut costs no quality
    and falls on a keyframe. The sound is cut to each clip's own picture to
    the sample, faded over 5 ms at a cut and encoded once for the whole
    file, so it stays on the picture however many clips there are.
  - The file's sound is the first clip's Audio choice. Stereo takes any
    clips. Where every clip has four-channel sound (the X6's 360° Audio, a
    `.360`, the QooCam's four capsules) the joined file can carry the
    Ambisonic track or, on a Mac, Apple Positional Audio: the clips' sound
    fields are put together channel for channel and encoded once. A clip
    without such sound in the run is named before anything is exported.
  - The clips have to come out alike: the same frame size and frame rate,
    the same codec and bit depth, the same view. One size for all of them
    is picked in Format (8192 × 4096 takes clips of any camera). What does
    not fit is said, with the clips' names, before anything is exported.
  - The file is named after the first clip (`<name>_Joined_SBS.mp4`) and
    carries the projection metadata a single export carries.
  - On Windows the NVIDIA encoder is set up alike on every export path. A
    camera file, a raw take and a Blackmagic RAW clip reach it by different
    paths, and those asked for two presets (p4 on the GPU-resident path, p5
    on the others). The preset is written into a file's parameter sets, so
    clips that had taken different paths were refused at the end of the run
    as "encoded with different parameters". Every path uses p4 now, the
    GPU-resident path's value from the start, and such clips join; an
    H.265 export through the other paths changes by that much.
  - One file or none: when a clip fails or the run is stopped no joined
    file is written, and the clips exported by then stay in a folder beside
    where it would have been.
  - **Lens Space clips become an Apple Immersive Video file** (macOS 26 or
    later). A Lens Space file states its clip's lens, and a file described
    that way has one lens, so clips of different recording modes or cameras
    could not share one. An Apple Immersive Video Universal file (`.aivu`)
    holds a camera for every clip, a mesh made from that clip's lens that
    ends where its picture does, and says frame by frame which one a frame
    is shown through.
    The clips' side-by-side pictures and their sound go in packet for
    packet: nothing is encoded again, and the file is written in about the
    time it takes to copy it. A sound field is written as Apple Positional
    Audio. The file is named `<name>_Joined_Lens.aivu`.
    - Checked with what the system offers: Apple's own validator passes the
      files, and the system's preview renderer draws every clip through its
      own lens, both eyes (against the same view drawn from each lens:
      0.00 px with the clip's own, 10 to 127 px with another clip's), for a
      full-view clip, a Crop clip and a clip of another mode in one file.
      Looking to the sides, straight up and down and behind, it draws the
      picture to its edge and nothing beyond it.
    - For Apple Vision Pro. The clips still have to come out at one frame
      size and frame rate (the 8K size in Format does that for every mode).
    - Played on Apple Vision Pro: a file of three clips in three recording
      modes, as the app writes it, with stereo sound and with Apple
      Positional Audio. Not available on Windows: the camera description is
      written by a framework of macOS.
  - Not joined: stills, DNG sequences, spatial videos; and, where no Apple
    Immersive Video file can be written, the Lens Space view through the
    camera's own lens.
- **The list comes back.** The clips, their order, every clip's trim and
  settings, the ticks, the output folder and the Format choices are kept as
  you work and are there again the next time the app starts, with the clip
  that was on screen. A clip whose file is not found (a drive that is not
  connected) stays in the list, marked, with its settings: click it once
  the file is back.
- **Projects.** The toolbar's Project menu saves the same things as a file
  (`.vr180proj`) and opens one: New, Open, Save, Save as; Cmd/Ctrl+S saves.
  A project holds no media, only where the clips are, and that also from
  its own folder: a project moved together with its clips still finds them.
  On Windows that includes a project and its clips on one drive or network
  share with no folder in common: the disk may come up under another
  letter.

### Windows: faster exports of sources without a fast path

Sources that have no zero-copy path on Windows (Insta360 X6 raw video,
Blackmagic RAW, and any clip whose fast path is not available, such as on a
card with too little memory for hardware decode) export on the general
loop. For 10-bit H.265 through NVENC and for ProRes that loop took each
frame off the GPU as a full RGB picture (201 MB at 8192×4096), converted
it on the CPU and only then encoded it, one step after the other. It now
has the GPU write the encoder's own planes, reads those back (half to a
third of the bytes) and hands them to an encoder that runs alongside the
next frame's work; a raw take is also read from disk ahead of the frame
being worked on. On an RTX 4090 at 8192×4096: X6 raw with tone and
sharpening 3.5 → 10 frames a second, with the default development 4.2 →
15; a Blackmagic RAW clip at 4096×2048 4 → 10 (ProRes 16). The picture is
the one the fast paths write (the same GPU conversion); software H.265
and 8-bit output are unchanged.

### Fixes

- **Exports slowed down whenever the app was not in front (macOS).** macOS
  rests an app whose window is covered or that is behind another app, and
  an export is when the app is left alone: with the same settings a run
  went at 10.3 frames a second with the window in front and at 5.4 behind
  another app, and a clip with noise reduction at 1.65 against 0.46. The
  app now tells the system it is at work for as long as a run lasts, from
  its first clip to its last (a joined file included): it keeps its speed
  wherever its window is, and the Mac does not go to sleep in the middle of
  a long batch. Its display may still sleep.
  **On Windows** the same notice is given in Windows's terms. A PC goes to
  sleep once its power plan's idle time is over whatever a program is
  computing: for as long as a run lasts it now stays awake (its display may
  still turn off). And a program whose windows are all minimised or covered
  can be run at reduced speed on battery power: the app is exempt during a
  run. On a desktop PC on mains power a minimised run measures the same as
  one in front (two 8192 × 4096 clips at 31 frames a second either way), so
  there the gain is the sleep.
- **The 3D LUT was applied at 8-bit precision.** The cube was stored on the
  GPU with 8 bits a channel and read through a hardware sampler, so a 10-bit
  export sat up to two codes away from the LUT's own values, by a different
  amount in each channel: a faint tint in neutrals that changed from one
  lattice cell to the next. The cube is now stored at 16 bits and
  interpolated exactly, within 0.02 of a 10-bit code. Every preview and
  export that uses a LUT changes by up to those two codes.
- **10-bit exports through the hardware encoders were cut off, not
  rounded.** The encoder is handed 16-bit words whose top ten bits are the
  code, and it reads only those: the fraction the app left below them was
  dropped, so luma and both chroma planes came out about 0.4 of a 10-bit
  code low. The codes are rounded to the nearest one now, and the chroma
  coefficients are written to six places (they were four). HEVC Main10 and
  ProRes through VideoToolbox, and the Windows paths that share the compose.
- **Exports through a software colour conversion were off, in three
  ways.** FFmpeg's converter was left to its defaults wherever the app
  hands it a picture: when an encoder is fed RGB (libx265, the 8-bit and
  the fallback H.265 paths and software ProRes on Windows) and when a
  source is decoded on the CPU (the portable path, GoPro `.360` without the
  GPU decode).
  - It converted with the **BT.601 matrix** in files tagged BT.709: greys
    were right and every saturated colour was not (a 75 % red 4 % of the
    range too bright in luma). A graded GoPro export decoded and encoded
    that way.
  - Its **16-bit pictures are on another scale** than the GPU's (white is
    65280, not 65535), so a picture that crossed from one to the other
    once came out 0.39 % dark or bright.
  - Its fast **8-bit conversions run 2 of 255 dark**: an 8-bit software
    export of a 10-bit source was 1.8 codes darker than the hardware one.
  - Its **full range stops short above 8 bits**: for a full-range 10-bit
    source (a QooCam 3 Ultra clip) it takes code 1020 for white, not 1023,
    so such a clip decoded on the CPU was 0.3 % brighter than the same
    clip decoded on the GPU.

  Every such conversion now states its matrix, puts 16-bit pictures on one
  scale, and uses FFmpeg's exact arithmetic on a threaded context. Flat
  test colours come back code for code through every 10-bit path and
  through the 8-bit software one, and the software exports are faster for
  it (10-bit H.265 in software: 1.8×).

  On macOS the live preview and the VideoToolbox exports decode and encode
  on the GPU and never went through any of this. The **paused
  full-resolution still** did (the magnifier's picture, and what Snapshot
  saves): it is decoded on the CPU for every camera, and for a 10-bit
  source it was 1.8 of 255 darker than the live preview of the same
  moment. The two now agree to 0.02.

  What remains: an 8-bit export through VideoToolbox is 0.4 of a code low,
  in Apple's own RGB conversion; and the working-size preview of a
  CPU-decoded source still uses the fast conversion.
- **A queued export waited for the window to be seen.** With the app's
  window minimised, or completely covered by another one, the clip being
  exported finished, but the next one in the queue did not start, and the
  finished notification did not come, until the window was shown again: an
  unattended batch sat idle for as long as the window stayed covered. The
  hand-over to the next clip now runs whether or not the window is visible.
- **A Crop clip's export size followed the other clips' choice.** The
  export resolution was one choice for the whole list, so a list that held
  X6 Crop clips next to full-view clips had no setting that gave each its
  best: a full-view clip's own size put the Crop clips on the smaller canvas
  of their own size, and a Crop clip's native sharpness sent the others to
  8K. A Crop clip now keeps a choice of its own. The Format window edits it
  while a Crop clip is open and the general one otherwise, and a batch
  exports each clip at the size picked for its sort.

## 2.5.6 — 2026-11-13

### Insta360 X6: 7.4K, 10K and their Crop recording modes

Four more dual-fisheye recording modes of the X6 open like the 8K one: the
factory lens model from the file, the same gyro stabilizer, and each mode's
own rolling-shutter time.

| Mode | Each eye | Rate | What the frame holds |
|---|---|---|---|
| 7.4K | 3712 × 3712 | 59.94 | the whole lens view — the 8K mode's sensor window, scaled |
| 7.4K Crop | 3712 × 3712 | 59.94 | native sensor pixels from the centre of the lens, about 97° across |
| 10K | 4864 × 4864 | 29.97 | the whole lens view from the higher-resolution readout |
| 10K Crop | 4912 × 4912 | 29.97 | native sensor pixels, about 125° across (88° off axis in the corners) |

- **The mode is read from the file**: each eye's encoded size plus the
  trailer's sensor-window record. The clip header and the Source panel show
  its name. The two 3712² modes encode the same size at the same rate, so a
  clip whose window record is missing (stripped by a re-mux) cannot say
  which it is: it opens as "7.4K60 · view to be confirmed" with a **View:
  Wide / Crop** choice under Source. The older 3808² crop and 4864² "9.7K"
  signatures are recognized too; an unknown one opens through the generic
  rule and says so.
- **Each mode has its own geometry.** The wide modes scale their sensor
  window to the frame; the Crop modes are native pixels cut from the centre
  of the sensor, so the lens model applies unscaled and only shifted. The
  window record is a signature, not a decode size or a crop rectangle —
  reading it as one put the 7.4K Crop 24 % off in scale.
- **Checked on footage, per mode and per eye**: feature motion predicted
  from the gyro through the lens model agrees with the tracked motion to the
  same scale as the 8K mode (within 0.2 %), at the same time offset, with
  the readout the file states. Stabilized exports leave about 0.05 px of
  frame-to-frame shake at 1024 px per eye in all four modes (1.2 – 2 px
  without), and scaling the rolling-shutter correction away from the file's
  value in either direction makes it worse.
- **Rolling shutter per mode**: the file's readout is used as recorded
  (14.56 / 16.04 / 31.65 / 21.29 ms). The 7.4K Crop encodes 3712 of the
  3968 rows the sensor reads, so its frame spans that share of the readout.
- **A Crop frame keeps every pixel it recorded.** It holds part of the
  lens' view, so in the VR180 and fisheye outputs the directions outside it
  are black — corners included, where the frame reaches furthest. (The
  full-view modes still repeat their rim past the image circle; that hides
  the gaps stabilization can open at the edge.)
- **The reframed (Flat 3D) view repeats a Crop frame's edge** past the
  recorded picture, like the full-view modes do at their rim: the viewport
  sits inside the recorded view, so what shows there is the border the
  stabilizer exposes, and a repeated edge covers it where black would not.
  On a stabilized 92° flat view of a 7.4K Crop clip, 81 of 180 frames
  showed a black border before; none do now.
- **Crop mask** (Output panel, Crop clips): a black window over the picture
  with **Mask size**, **Feather** and **Corner roundness** — from the
  recorded frame's own outline through a rounded rectangle to an ellipse,
  hard-edged or faded. It is fixed in the output and identical in both
  eyes, so it stays still while the stabilized picture moves under it. Off
  by default; the reframed view does not use it (its sliders gray out).
- **The I-Log LUT follows the file's picture profile**, not the frame size:
  it is applied when the clip says I-Log (or says nothing, as before).
- Native export sizes follow the mode: 7424 × 3712, 9728 × 4864 and
  9824 × 4912.
- **Native sharpness for Crop clips: 13120 × 6560.** A Crop frame is native
  sensor pixels over part of the lens' view (36 pixels per degree at the
  centre, twice the 8K mode), so a 180° canvas of the frame's own size
  scales it down — the 10K Crop by a quarter, the 7.4K Crop by almost half.
  The Resolution list of a Crop clip gains an entry that renders the canvas
  at the frame's centre density, taken from the file's lens calibration
  (6560 pixels per eye). H.265 encodes it in hardware on Apple Silicon
  (about 10 fps on an M5 Max); it is beyond HEVC Level 6.2, so most players
  will not decode it, and ProRes at this size uses the software encoder.
  On Windows it is the other way round: NVIDIA's encoder stops at 8192
  pixels a side, so H.265 at this size is encoded in software (about half a
  frame per second on a fast desktop), while ProRes stays on the GPU (about
  4 fps on an RTX 4090).

### Spatial video export (macOS)

The reframed (Flat 3D) view can be exported as an **Apple spatial video**:
choose **Spatial video (MV-HEVC)** under Codec in the export format. The
two eyes go into one MV-HEVC Main10 stream in a `.mov`, with the metadata
that makes a Vision Pro show it as a spatial video (the soft-edged window,
the immersive view) instead of a flat 3D one. Other players show the left
eye in 2D.

- **The field of view written is the view you framed**: the reframed
  view's own horizontal angle, at 1:1 (up to 3840 × 3840 per eye) or 16:9
  (up to 3840 × 2160 per eye). A
  Crop-mode clip becomes a spatial video camera whose field of view you
  choose afterwards.
- **Life size has a ceiling in the headset.** In its immersive view a
  Vision Pro shows a spatial video at life size only up to 90° wide and 80°
  tall. A wider frame is scaled down whole to fit, so everything in it
  looks smaller: that is past 80° for a 1:1 view and past 90° for 16:9. The
  export format says so when the view is past the limit, with the field of
  view it will be shown at and the Zoom that stays at life size. Checked on
  a headset (visionOS 27) with charts that differed only in the field of
  view written to the file.
- **Camera baseline** (65 mm by default) and **Disparity adjustment** (2 %
  by default: the far background sits just behind the window) are in the
  export format.
- **One encode, from the rendered frames.** The frame the GPU composed for
  the encoder is what the MV-HEVC encoder gets — no side-by-side
  intermediate, no second generation. Stabilization, colour, trims and the
  audio formats (stereo, ambisonic, APAC) work as in any other export.
- **A spatial video is rectilinear**, so the option needs Defish at 0. With
  Defish above 0, or on a VR180 frame, a clip in the queue exports as H.265
  side by side instead; the Codec choice stays for the clips that can use
  it.
- Measured on an M5 Max from a 7.4K Crop clip (59.94 fps, stabilized):
  2160 × 2160 per eye at 46 fps, 3840 × 2160 per eye at 32 fps; from a 10K
  Crop clip (29.97 fps), 3840 × 3840 per eye at 20 fps.
- Checked with the system's own tests on macOS 27: the files read as
  spatial video to AVFoundation and to Photos, and VideoToolbox decodes both
  eyes of every frame. A side-by-side file with the same metadata does not
  qualify, on the Mac or in the headset, which is why this is MV-HEVC.
- Needs macOS 14.4 or later. Not on Windows: there is no MV-HEVC encoder
  there.

### Reframed (Flat 3D): 3840 × 3840 per eye

The 1:1 view has a larger video size: **3840 × 3840 per eye**, a 7680 × 3840
side-by-side frame (or a spatial video of that size). The 16:9 view stays at
3840 × 2160 per eye; a size the current view does not offer drops to the
largest one below it.

### File info: bit rate

The file info (Source panel) and the clip header show the video bit rate
read from the file. A dual-stream recording lists each eye's stream next to
the total (`252.7 Mbps (126.6 + 126.0)`); where the container states none
(Blackmagic RAW) the figure is the file's size over its duration, marked ≈.

### Fixes

- **Windows: an export too large for the NVIDIA encoder says so.** Above
  8192 pixels a side (a Crop clip's 13120 × 6560 canvas, a 12K Blackmagic
  RAW clip at its own size) NVENC refuses the frame and the software H.265
  encoder takes over, many times slower. That used to happen silently: the
  note under the Resolution list was the Mac's ("encodes in hardware") and
  the export bar showed nothing. The note now describes what the encoder
  on this platform does, and the export bar names the fallback while it
  runs.
- **A clip that lost only a few frames exported slightly slow.** The
  nominal frame rate replaced the average only when the two differed by more
  than 0.5 %; a clip missing 8 frames in a minute sits inside that, so its
  export ran 0.4 % slow and ended a quarter of a second behind its audio.
  Any missing frame now selects the nominal rate.
- An `.insv` whose trailer cannot be read still opens with the generic X6
  lens profile placed for its recording mode (no gyro, so no stabilization)
  instead of being stretched to the full lens.
- **The zoomed-in still of a source whose eyes are not square.** It was
  projected at the source frame's own shape, and the preview is laid out
  from the picture it is given. A side-by-side file of 8192 × 4320 (two
  4096 × 4320 eyes) changed proportions when the still replaced the live
  preview. A single-lens 12K Blackmagic RAW clip (one 12288 × 6480 eye)
  asked for a 24576-pixel-wide picture, more than the GPU holds, so its
  still never appeared in the side-by-side view. Each eye of the still is
  square now, like the preview and the export, and no larger than two can
  sit side by side on the GPU (8192 pixels on current hardware). Stereo
  Blackmagic RAW clips were not affected: their eyes are already square.
- **VR180 metadata no longer carries a camera baseline**, and its slider
  is gone from the Vision Pro option. The box the app wrote was not where
  Apple's format puts it nor in its layout, so readers skipped it: the
  slider never reached the file. The baseline is optional for VR180 and the
  player does not use it, so files play exactly as before — what a Vision
  Pro reads from them (both eyes, side by side, half-equirectangular, 180°)
  is unchanged. The baseline setting now belongs to the spatial video
  export, where it is required and written by the encoder.

## 2.5.5 — 2026-11-07

### Footage with dropped frames (Insta360 X6)

An X6 clip in which the camera dropped frames — the video track keeps real
time, so it has holes where frames are missing and plays at a lower average
rate than the recording mode — lost its stabilization: the app looked up
each frame's pose by `round(pts / dt)` with `dt` taken from the average
rate, which lands tens of frames early long before the first hole.

- **Frames are now located by the file's own sample table.** Each decoded
  frame finds its row in the per-frame tables (stabilization, rolling
  shutter, exposure, eye matching) by its position in the video track, in
  the preview, the zoom still and every export path. The clip's frame rate
  is the nominal one (the camera's recording mode), not the average; a clip
  without holes is unchanged.
- **Exports stay in sync with the audio.** The output is constant-rate at
  the nominal frame rate: the frame before each hole is held for the missing
  slots, so the picture never drifts against the sound (verified: a 2 s
  window with 91 source frames exports as 100 frames at 50 fps, 2.000 s).
- **The preview plays on the audio's clock too.** Playback is paced by each
  frame's own time: a hole holds the previous frame for the missing slots
  instead of running through it, so the picture no longer drifts ahead of
  the sound on a dropped-frame clip.
- The clip info shows the frames actually in the track; the frame counter
  counts output frames while exporting.

### Blackmagic RAW: rebuilt on the current stabilization path

Blackmagic cameras with a VR180 lens (tested on a PYXIS 12K, 12288×6480
and 12288×5112 clips with their gyro) now get the same treatment as the
other cameras. The camera records no lens calibration, so the lens stays a
manual entry — the app seeds it from the image circles it finds in the
frame.

- **Decoding.** Single-track clips are split into their two eyes (a
  dual-fisheye lens on one sensor); two-track clips are composed by the
  helper as before. Each eye is fitted on a square canvas, the pair making
  a 2:1 frame, the way the fisheye-centre web tool and VR players treat a
  non-square eye: a 2.4:1 clip (12288×5112) is padded black top and
  bottom, a 17:9 clip (12288×6480) is cropped top and bottom, and the lens
  parameters, the Override panel and the rolling-shutter timing all refer
  to that canvas. 10-bit and ProRes exports decode at 16 bits; the preview
  decodes at a fraction of the recorded size; seeking works.
- **Stabilization.** The clip's gyro drives the same stabilizer as the
  other cameras (camera lock, soft-stab, Response, max correction), with
  per-scanline rolling-shutter correction timed from the file: the sensor
  line time and the per-frame shutter (which the camera's auto exposure
  moves within a take) are read from the clip. The gyro-to-camera
  orientation was measured on the sample footage.
- **Lens.** On load the app finds the two image circles and seeds each
  eye's principal point and focal from them; FOV, distortion (k1–k5, p1/p2)
  and the centres are then set in the Override panel. Camera presets and
  the Gyroflow profile loader remain available for this camera.
- **Colour and audio.** Clips decode in the Film log they were shot in and
  the camera's own embedded "Gen 5 Film to Extended Video" LUT is applied
  automatically; the PCM stereo track is encoded to AAC and muxed. A
  channel-layout quirk that silently dropped WAV audio in the muxer is
  fixed.
- **GPU decoding.** The helper decodes on the Mac's GPU (the SDK's Metal
  pipeline) when one is available, falling back to the CPU otherwise, and
  keeps the next frames in flight while the current one is handed over —
  about 1.7× the preview-size throughput of the CPU path with half the CPU
  time, and a third of the CPU time at half and full size, which is what
  keeps playback smooth when the machine is busy. The decode log line and
  the clip probe report which pipeline is in use. The helper also hands
  frames over in the app's own pixel order (no per-pixel conversion
  afterwards), the preview decodes at the SDK's smallest size that still
  covers the preview (a 12K clip at 50/60 fps: 1024-pixel eyes; the
  reframed view keeps 2048), and a >30 fps clip is thinned before the
  decode rather than after it — the camera's files are intra-only, so the
  skipped frames cost nothing.
- **Smoother preview on macOS, all cameras.** The window now redraws when
  the decoder hands it a frame instead of on a fast timer. The timer was
  making the UI wait for the display most of the time, which on Metal
  blocks the decoder's own GPU work for a whole refresh per step — the
  "render-bound" frame skips during playback. The BRAW audio track now
  plays (its resampler rejected every frame before), and the zoom still no
  longer decodes extra full-size frames in the background.
- **Zero-copy BRAW preview (macOS).** The helper now hands decoded frames
  to the app as shared GPU surfaces instead of pushing pixels through a
  pipe: no copy on the CPU at any size, so the reframed view decodes the
  full 12K frame and every zoom level samples native pixels. Measured
  throughput 96 fps at full 12K (26 through the pipe), 328 at quarter.
- **Blackmagic RAW colour controls.** The Color panel gains a Blackmagic
  RAW section with the SDK's own development of the sensor data: colour
  science generation, gamma and gamut (the camera's full lists), ISO,
  exposure, white balance and tint, highlight recovery, gamut compression,
  and the tone curve (saturation, contrast, midpoint, highlights, shadows,
  black and white level, video black level). Every control starts at the
  clip's own value, a dot marks what was changed, and "Reset to clip"
  clears the overrides. The app's Tone and White balance stages are hidden
  for this camera (they duplicate the SDK's); the embedded Film→video LUT
  switches off automatically under a non-Film gamma and back on for Film.
  The SDK applies its tone curve only under the "Blackmagic Design Custom"
  gamma, so moving a tone-curve control switches the gamma to it, as
  DaVinci Resolve does; the controls start from the curve the current
  gamma is built from (the SDK's own preset for the camera), so the
  picture continues from where it was and only the moved control changes.
  Highlight and shadow rolloff are named as such, with help text: they
  shape the knee and toe of the curve only (a subtle effect by design);
  midpoint, contrast and the black / white levels are the level controls.
  The colour science generation decides what gamma and gamut exist: Gen 6
  has none of the Blackmagic Design gammas and decodes a Film clip to
  DaVinci Intermediate / DaVinci Wide Gamut, so the panel lists each
  generation's own options, drops a choice the new generation rejects, and
  disables the tone curve where there is no Custom gamma.
  The preview re-decodes the frame on screen as a control moves (the
  zero-copy helper takes the change live), and exports and zoom stills use
  the same settings.
- **H.265 bitrate up to 1000 Mbps.** The VR180 bitrate slider's ceiling
  rises from 500 to 1000 Mbps for native 12K exports (logarithmic, so the
  usual 50–300 Mbps band keeps its fine steps).
- **Full-resolution export.** The Resolution picker's native entry now
  shows the loaded clip's own size (a 12K BRAW: 12288 × 6144) instead of a
  fixed 7680 × 3840 label. HEVC at that size encodes in hardware; the Mac's
  hardware ProRes encoder refuses frames above 8192 × 4096, so ProRes
  exports beyond it switch to the software encoder automatically (slower;
  a hint says so). Such files exceed HEVC Level 6.2 and need a capable
  player.
- **BRAW on slow storage.** Play no longer waits for the per-frame shutter
  scan (one full frame per sample, about 2 GB on a long 12K clip): the
  live view uses the clip's shutter range from the probe, and the export
  reads the exact per-frame table. The clip probe is cached, and the zoom
  still starts its decoder only when a frame is asked for.
- **Needs Blackmagic RAW installed.** The decoding helper ships inside the
  app, but Blackmagic's RAW library is not redistributable: install the
  free "Blackmagic RAW" package from Blackmagic's support site (macOS; the
  SDK is part of it) or Blackmagic RAW 6.x with the SDK component
  (Windows). The helper is built against SDK 6.0; without the library a
  `.braw` clip cannot be opened.

### Kandao QooCam 3 Ultra: video and stills

The QooCam 3 Ultra joins the Osmo 360, the X6 and the Max 2 as a VR180-mod
camera, with the same coverage as the X6: the factory lens model, gyro
stabilization with per-scanline rolling-shutter correction timed from the
file, stills, and the camera's own colour handling.

- **Video (`.mp4`).** The camera writes one HEVC stream with both eyes
  stacked top/bottom (3840×7680 at 30 fps) and two proprietary boxes after
  `moov` — the factory lens record, the 1 kHz gyro, the per-frame timestamps
  and exposures. The app reads all of it. The lens model is the one QooCam
  Studio's own stitcher runs (a 4-coefficient Kannala-Brandt lens per eye
  from the record, scaled to the stream); the top half is the left eye
  after the mod (measured on near-field parallax). Stabilization fuses the
  gyro exactly as the X6 path does; the readout (19.4 ms with HDR off,
  24.6 ms with HDR on, top row first) and the frame content time
  (`PTS + readout/2 − shutter/2` plus a per-mode delay) were measured from
  image motion against the gyro on two cameras and both sensors, over
  shutters from 0.2 to 10 ms — indoor clips, where the camera holds a
  10 ms shutter, depend on getting the shutter term right. Decoding
  takes the zero-copy VideoToolbox path (each eye sampled in place from the
  stacked frame).
- **HLG clips.** The camera records HLG BT.2020, full range (whether or
  not its HDR readout mode is on). The decode path now carries the
  stream's YCbCr matrix (BT.2020 instead of the fixed BT.709) and a
  built-in HLG → Rec.709 conversion (BT.2100 inverse OETF, 1000 nit OOTF,
  BT.2408 reference-white scaling with a soft highlight shoulder) is
  applied on load like the other cameras' log LUTs, keyed on the stream's
  transfer tag; a Rec.709-tagged clip needs nothing. The conversion is
  scaled for the way this camera uses the HLG range (ordinary scenes run
  to full scale rather than stopping at the HDR reference white), so
  skies and bright surfaces keep their detail instead of clipping.
- **HDR readout mode.** The camera's HDR setting is a slower sensor scan
  for more dynamic range; the app reads the flag from the file and uses
  that mode's measured readout time and frame-stamp timing for the
  rolling-shutter correction and stabilization.
- **Stills (`.dng`).** The 13888×6944 RGGB dual-fisheye DNG opens like the
  X6's: the lens record (the DNG is its own calibration canvas) and the
  per-lens 17×13 shading grids come from the EXIF `UserComment` and feed
  the same raw stage, preview tiers and PNG / JPEG / linear DNG export. The
  stereo offset is shared between the QooCam's video and stills, as for
  the X6.
- **Audio.** The 4-channel track is the camera's four microphone capsules
  (not a sound field; Kandao's spatial audio is Fraunhofer upHear, a
  parametric process the app cannot replicate). The capsule layout was
  measured from the recordings — one capsule on each of the front, left,
  back and right faces, all in phase — and the app encodes the array to
  first-order ambisonics itself (the capsule differences are integrated
  and phase-matched to the pressure, so directions carry level, not just
  phase), so this camera gets the same **Audio** choices as the X6:
  **Stereo** (a front-facing pair from the sound field), **Ambisonic**
  (AmbiX + SA3D) and **APAC** (macOS). Each export window is normalized to
  a safe peak (the capsules are recorded low and vary between clips); a
  capsule channel the camera records late (one of the two test units
  delays its front channel by 0.25–0.45 ms, differently per clip) is
  re-aligned, and capsules whose level or low-frequency response differ
  (up to 13 dB between neighbours on that unit) are balanced pair by
  pair, both from a measurement on the file itself. Playback uses the
  same encoder.
- **Calibration.** The lens model form was checked against QooCam
  Studio's own VR180 export of the sample clip (0.8 px across the whole
  eye once Studio's own principal-point shift is accounted for). The app
  renders from the camera's lens record as written, for video and stills
  alike; it does not copy Studio's adjustments.
- **Windows.** The QooCam takes the Windows zero-copy preview and export
  arms (NVDEC → D3D11 → Vulkan) like the other cameras: the stacked frame
  is converted whole and split on the GPU, HEVC exports run at the NVENC
  rate, and the D3D11 converter now applies the stream's own YCbCr matrix
  (HLG clips were slightly desaturated on the Windows fast path, which
  had BT.709 built in). A single-lens QooCam recording (`Q180_…`, one
  3840×3840 circle) is reported as an unsupported format instead of being
  split into two half-eyes.
### Also in this release

- **Preview no longer dies silently on a network read error.** A stalled
  read from a network share reported as end-of-stream, so the preview
  decoder could exit after zero frames and leave the window on "Click ▶
  Play" with Play doing nothing. The decoders now tell a read error from
  the end of the clip (the error is logged), and the app restarts a
  decoder once by itself when it stops before its first frame.
- **Blackmagic RAW on Windows.** The BRAW helper now builds and runs on
  Windows (`helpers/braw/build_braw.ps1`, needs Blackmagic RAW 6.x with the
  SDK component installed): preview and export of PYXIS clips with the
  gyro stabilization, the embedded LUT and the audio, decoding on the SDK's
  CUDA pipeline when an NVIDIA GPU is present. The preview hand-off is
  GPU-resident on Windows too: the helper fills shareable D3D11 textures
  the app maps directly (the Windows twin of the macOS IOSurface path), so
  no frame pixels cross the pipe during playback. In the reframed output
  mode the preview decodes the full 12K frame on the GPU and hands it over
  with a CUDA → D3D11 copy, so a zoomed viewport shows native pixels as on
  macOS, at the full playback rate from local storage. Exports take the
  readback path. The helper runs without a console window (opening a
  BRAW clip used to pop up a terminal window on Windows).
- **Larger reframed stills.** A photo's reframed (Flat 3D) export goes up
  to 5760×3240 per eye at 16:9 and 4320×4320 at 1:1; the 2160-line cap
  that keeps video encodes sane now applies to video only.
- **The previous launch's log is kept** as `vr180-gui.prev.log` next to
  the current one, so a problem that shows up after a relaunch still has
  a trail to read.

## 2.5.2 — 2026-10-10

### Windows: APAC is no longer offered
- APAC (Apple Positional Audio) is encoded by a macOS helper and has no
  Windows encoder; picking it on Windows only ever produced a stereo track
  with a log warning. The Windows export-track picker now offers Stereo and
  Ambisonic only, and a previously saved APAC choice is treated as Ambisonic
  — the spatial option Windows can actually write.

### Fixed on Windows: X6 360° Audio stereo export
- On Windows the new stereo downmix of the X6's 360° Audio track silently
  failed to open its temporary file and fell back to muxing the raw
  4-channel track, so a "stereo" export still carried a 4.0 track. The
  FFmpeg build bundled with the Windows app has no `ipod` muxer, which is
  what a `.m4a` name resolves to; the audio writer now names its container
  explicitly instead of guessing from the extension. Windows exports of such
  clips now get a real stereo track, at the source's level.
- The stereo decode of an X6 360° Audio clip now renders only the export
  window instead of the whole clip (a 127 s clip cost ~7 s up front even for
  a 6 s export), and it starts sample-accurately at the trim point rather
  than at the next audio packet.

### Insta360 X6: automatic per-frame eye-exposure matching

The X6's two sensors auto-expose independently — shutter *and* gain — so the
two eyes of a VR180 clip differ in brightness, and the difference moves with
the light: on real post-mod footage the eyes sat up to about a stop apart
for seconds at a time, then drifted back. The static "Matching Eyes"
exposure trim cannot follow that.

- **Auto-match eye exposure (per frame)** — a new switch in the Colour
  panel's Matching Eyes section, shown for X6 clips. Every frame, the
  brightness of the two eyes is measured on the GPU over the central region
  both eyes share (a luma histogram; the mean shift of five quantiles across
  the rank range both eyes keep once their clipped tails are cut is robust
  to the content only one eye sees), and the difference is
  cancelled with a log-domain code offset before the LUT — on I-Log that is
  an exposure gain, so colour is untouched. Clipped highlights and crushed
  blacks are cut out by rank, jointly for both eyes, so sky in the frame
  does not water the correction down. The frame loop never waits on the
  GPU: the export reads nothing back, and the preview's readout arrives
  through a small asynchronous copy ring. A reference picker chooses
  whether the eyes meet in the middle (default) or one eye follows the
  other; a readout shows the measured difference and the offsets applied,
  in EV.
- Temporal behaviour: the camera's AE steps are taken at once (a lagged
  step would leave the eyes mismatched for several frames); the small
  frame-to-frame jitter is low-passed. A seek resets the smoothing.
- The static Eye Exposure trim still composes on top, and the whole thing
  is off for other cameras, whose sensors expose together.
- Also fixed in the preview: in the single-eye "right" view the per-eye
  trims (CT / tint / exposure) were applied to the wrong eye; they now
  follow the eye they belong to.

### GoPro .360: stabilization timed to each frame's exposure, and to the frames the camera actually dropped

Two timing gaps in the GoPro path, found by rigging a GoPro MAX, a DJI
Osmo 360 and an Insta360 X6 together and stabilizing the same walk and
run through all three.

- **Per-frame exposure.** The orientation sample now sits at the centre
  row's mid-exposure — frame start + readout/2 − shutter/2, from the
  per-frame shutter the camera records (`SHUT`) — the rule the DJI and X6
  paths use, and the rolling-shutter correction is centred on that same
  instant. In daylight nothing changes (a 1 ms shutter moves the sample
  by half a millisecond; the old midpoint measured right to within
  ~1.5 ms); in low light it matters: at a 1/30 s shutter the pose was
  sampled 16 ms late, shake that followed the motion. On a dark running
  section the locked residual drops 12 % and its motion-correlated part
  disappears.
- **Dropped frames.** When the camera's encoder falls behind it drops a
  sensor frame and repeats the previous image to keep 30 fps; the gyro
  and the telemetry block timestamps keep running, so that block is one
  frame period short. The gyro timeline used to be stretched evenly over
  the block's 30 video frames, mis-timing every frame after the repeat by
  up to a frame period — a ~0.7 s burst of jerks per drop. The camera
  marks these in its `MSKP` ("frame skip") stream; the gyro is now
  re-anchored per frame from it, and a repeated frame receives the pose
  and RS correction of the image it copies. The test clips had one to two
  drops each, both under heavy motion.

With both in place the GoPro's frame-to-frame jitter on the 30 fps test
clip matches the Osmo and X6 second by second; the GoPro still looks
worse than the 50 fps cameras simply because each 30 fps frame carries
1.7× the motion.

### Insta360 X6 stills: open a `.dng`, export a VR180 still — as PNG, JPEG or a linear DNG

The X6 writes its photos as one 15520×7760 DNG with both fisheye circles
side by side, 14-bit Bayer, the factory lens model in the EXIF and an
Insta360 trailer. The app now opens them like any other source: the raw
stage (the camera's per-channel gain maps, a 5×5 linear demosaic, black /
white levels) runs on the CPU, each eye goes through the same exact
UCM fisheye projection the X6 video path uses, and the result is one
side-by-side half-equirect still — 8192×4096 or the full 15520×7760.

Three containers in the export bar:

- **PNG (16-bit)** and **JPEG** — developed: as-shot white balance, the
  DNG colour matrix, the camera's baseline exposure, a filmic tone curve.
- **DNG (linear raw)** — the camera-native linear data, demosaiced and
  reprojected but not white-balanced or tone-mapped, with the source's
  colour matrix, neutral, baseline exposure and EXIF carried over. Raw
  developers (Lightroom / Camera Raw / Apple's) open it as the X6's own
  raw and apply their profile, so the VR180 still can be graded with the
  full raw latitude. Lossless-JPEG tiles (the layout Adobe's own
  converter writes); validated with exiftool, macOS ImageIO and Adobe
  DNG Converter.

Stills have no timeline or stabilization (those sections are hidden);
view adjust, stereo alignment, eye swap and the colour stack apply as for
video. The grade bakes into PNG / JPEG; the **linear DNG carries it as
metadata** instead: white balance → `AsShotNeutral`, the lift / gamma /
gain / shadow / highlight tone → `ProfileToneCurve` (the app's tone curve
composed with the CDL, honoured by Adobe and Apple), saturation →
`ProfileLookTable` (Adobe), the raw exposure offset → `BaselineExposure`;
the per-eye "Matching Eyes" trims are baked exactly in camera-linear
space (one DNG has one profile). A 3D LUT and sharpening cannot travel as
DNG metadata and are left out of the DNG (the export panel says so).

With a still loaded, everything that cannot act on one photo is hidden
rather than greyed: the transport row (play / stop / snapshot / trim),
the timeline, stabilization, rolling shutter, noise reduction, Auto align
and the built-in log LUT buttons. With DNG output selected, the 3D LUT and
Detail sections go too, and the preview drops them so it shows exactly
what the DNG will carry. The Output section's Fisheye and Reframed modes
and the eye swap / upside-down toggles now work for stills, and the lens
Override is honoured by the still export as it already was by the preview.

Fixed: exporting the loaded still left the photo frozen — the export
stopped the preview thread (as it does for a video that shares the file)
and a still has no Play button to start it again, so the sidebar and the
pan drag no longer did anything until the clip was reloaded. The still's
preview now stays live through its own export (it holds only textures;
the export re-reads the file), and a still whose preview did stop gets a
"Restart preview" button. Found alongside: Stop before the first frame
left Play dead until a reload; the snapshot key on a still waited out a
15 s timeout and saved the working-res preview (it now points at Export);
Space on a still spun the playback repaint cadence; a restart of the same
clip re-seeded hand-tuned lens Override values from the file; and the
audio probe opened the whole DNG on the UI thread.

The still preview is no longer the video's 768-px working size: a photo
has one frame, so once the sliders rest it renders at the source's
working resolution (7760×3880 side by side on the X6; the reframed view
at 3880 px per eye), and the zoom magnifier looks into that image. While
a slider or the pan is moving, quick renders at the preview size keep it
fluid; the full one follows ~120 ms after the last change. The "Preview
size" menu is not offered for photos.

Beyond that, the unbinned 7760² halves are used where they can be seen:
the reframed view always samples them (a narrow viewport shows a small
part of the lens, so the source's resolution is what it shows), and when
the magnifier is zoomed past what the binned size can feed, the photo is
rendered at its native size (15520×7760 side by side). The native pair is
developed on first need (~1 GB of GPU textures, kept while the photo is
open). The GPU device now asks for the adapter's own texture-size limit
(16384 on every desktop GPU) instead of the 8192 default.

The X6's stereo offset (yaw / pitch / roll between the eyes) is now one
value shared by its `.insv` video and `.dng` stills: the same two lenses,
the same alignment. Tune it on either and the other kind picks it up the
next time a clip is activated or added; the sidebar says so.

### Insta360 X6 "360° Audio": the camera's ambisonics, decoded instead of copied

With the camera's audio mode set to 360° Audio the X6 writes a 4-channel
AAC track with no channel semantics in the file. It is first-order
ambisonics (AmbiX W, Y, Z, X — matched against Insta360 Studio's SA3D-tagged
export, which only rotates it by the stabilisation yaw). Copied as is, the
standard layout downmix every player applies cancels two anti-phase
channels and plays ~9 dB left-heavy; that is what an export of such a clip
sounded like until now.

The X6 now gets the same Audio section as the `.360`: **Stereo** (default)
decodes a front-facing pair of virtual cardioids (±45°, the camera's front
= the rig's forward) to a 192 kb/s AAC; **Ambisonic** copies the 4-channel
track and tags it SA3D for YouTube VR / Quest head-tracking, exactly as
Studio does; **APAC** re-encodes it to Apple Positional Audio for Vision
Pro (macOS). The preview player decodes the same stereo pair. Clips
recorded in the Auto / Stereo audio modes are unchanged (stereo AAC
passthrough).
(The first cut of the APAC path handed the helper a WAV tagged as a
positional 4.0 layout, which AVFoundation "converted" to ambisonics as
silence; the WAV now carries the ambisonic channel order, as the GoPro
path's always did.) The APAC helper also labels its input as HOA ACN/SN3D
now: without that label AVFoundation wrote the track as "Quadraphonic"
instead of ambisonics — the `.360` APAC exports had been tagged that way
all along, so Vision Pro saw a 4-channel surround track, not a spatial
sound field. Both sources now produce the HOA-tagged APAC track.

### GoPro .360: rolling-shutter correction audited and tightened

The GoPro rolling-shutter term was measured directly for the first time:
raw (unstabilized) frames exported with the correction off, on, doubled
and negated, and the residual inter-frame warp regressed against the
change in angular rate. The result: the correction has the right sign and
the right axis (the sensor is read top to bottom, on both lenses, and the
left eye behaves exactly like the right), and each unit of correction
removes 15.3 ms of readout — the file's 15.224 ms within 1 %. Three
things that were assumptions are now derived or physical:

- **Readout time from the file.** The per-row RS window and the
  mid-exposure pose instant now take the sensor readout from the
  recording's `SROT` setting; 15.224 ms is only the fallback for a file
  without one.
- **The sensor row of a pixel is computed from the ray the lens actually
  imaged** (the camera-frame direction after the stabilization rotation),
  not from the output direction before it. The old choice was a
  small-rotation approximation inherited from the reference pipeline; it
  was off by the whole applied rotation, which in camera-lock mode is
  tens of degrees.
- **Each row group carries the camera's mean rate between the reference
  row and that row**, so the per-row correction is the integrated
  rotation rather than the instantaneous rate at the row — the two differ
  by half the angular acceleration times the row offset squared, which
  only matters in jolts.

### The app is just "VR180 Silver Bullet" now
- The Windows installer, its Start-menu and desktop shortcuts, the Add/Remove
  Programs entry and the executable's own name (Task Manager, Properties) all
  dropped the "2.0" — the product name is no longer version-specific; the
  version lives in the version field. Upgrading replaces the old-named
  shortcuts; settings and the install location are untouched.

## 2.5.1 — 2026-09-25

### macOS: side-by-side sources — 8-bit joins the zero-copy path, and it uses far less memory

Two changes to the zero-copy path for generic side-by-side `.mp4` / `.mov`
sources on macOS.

- **8-bit sources now take the fast path.** Until now only 10-bit HEVC/H.264
  qualified; an 8-bit clip fell back to the CPU decoder. VideoToolbox hands
  8-bit frames back as NV12; those planes are now wrapped as 8-bit textures
  and the kernels expand the range from a per-source constant instead of a
  hard-coded 10-bit one. On an 8192×4096 8-bit clip a 120-frame export went
  from **21.95 s to 3.73 s** and peak memory from **3.14 GB to 1.03 GB**.
  Colour checked against the BT.709 formula applied to the raw codes: the
  new path lands within 0.28/255 on flat areas (the CPU path it replaces
  was 0.78/255 dark); the preview within 0.01/255. The range tag is
  honoured too — a full-range (`pc`) source no longer gets limited-range
  constants — on both 8- and 10-bit, and on the CPU fallback as well (its
  scaler now passes the tag to swscale), so the preview, the paused still
  and every export agree.
- **No more whole-frame intermediate.** The side-by-side path used to
  convert the entire frame to RGBA16 and then copy out two per-eye halves
  before projecting: at 8K that is a 268 MB texture plus two 134 MB halves
  per frame, and two full-frame passes. The projection kernels now sample
  each eye straight from the decoded planes, the same way the DJI and X6
  paths always have, and the preview resolves each eye from its own half.
  Peak memory for the 10-bit 8K export: **2.20 GB → 1.65 GB**. The DJI and
  X6 paths are bit-identical to before. A side effect worth knowing: the old
  path put the left eye's last column exactly on the chroma boundary with the
  right eye and blended it in — one column of wrong colour along the seam,
  in both preview and export. Sampling per eye fixes that.

Accepted: 8- and 10-bit H.264/HEVC in 4:2:0. ProRes and 4:2:2 profiles still
use the CPU path.

### Insta360 X6: eye order and stabilization direction fixed for real post-mod footage

The X6 path was wired before any modded camera existed, on an assumption about
which way the mod turns the back lens. The first post-mod footage showed the
assumption was mirrored: the right eye landed on the left of the frame, and
stabilization pushed the wrong way on both eyes — every correction added shake
instead of removing it ("twice as shaky").

- **Eye order.** Stream 0 (the back lens) is now the RIGHT eye by default,
  the same convention as the DJI Osmo 360. Nothing in the shared pipeline
  changed — the X6 loader simply labels stream 0 as lens A, as DJI does, and
  the iterator, calibration, per-eye rotation and rolling-shutter rows all
  follow. If you had turned **Swap eyes** on as a workaround, turn it off.
- **Stabilization.** The IMU→camera basis was measured on a stock camera
  against the back lens. The mod turns that lens 180° about the body's
  vertical axis to face the screen-lens side, so both lenses now share the
  stock frame with x and z reversed. Measured on three post-mod clips: with
  the old basis a camera-lock left 1.6–1.9× the *unstabilised* frame-to-frame
  motion; with the corrected basis 0.00× (a perfect lock) on windows with
  pitch/roll jitter, and soft-stab removed 74–100 % of the jitter on every
  window tested while the old basis made each one worse.

Pre-mod (stock, 360°) X6 files are not a target of this app, so no behaviour
was kept for them.

### Fixed: Windows machines with two GPUs
- On a hybrid Windows machine (an integrated GPU alongside a discrete one,
  which is most laptops) the hardware decoder could land on a different GPU
  from the renderer. The zero-copy preview would then hang before the first
  frame, stuck on "Loading stabilization data…", and a stopgap that shipped
  briefly dropped every such machine to the slow CPU path instead. The decoder
  is now created on the same GPU the renderer chose and its identity verified
  before use, so hybrid machines keep the GPU fast path. Verified on both GPUs
  of a 4090 + Intel UHD box. Contributed by @Norman3D.

### macOS: side-by-side sources get the zero-copy GPU path

`.360`, `.osv` and `.insv` clips have decoded straight into GPU memory on
macOS for a while: VideoToolbox hands back an IOSurface, the app wraps it as
a Metal texture, and the frame never touches the CPU. Generic side-by-side
`.mp4` / `.mov` sources were the one format left out — they still decoded on
the CPU and uploaded every frame. That gap is now closed, for both preview
and export.

- **Side-by-side export is ~3.7x faster and uses ~6x less memory.** On a
  3840x1920 10-bit HEVC clip a 120-frame export went from **4.73 s / 697 MB**
  to **1.29 s / 118 MB**. The whole frame is resolved once on the GPU and the
  two eyes are split with a subregion copy, so there is no swscale pass and no
  per-frame upload.
- **Colour is closer to the source, not just faster.** Measured against a
  known-good reference the new path lands at a mean error of 0.46/255 where
  the CPU path measures 0.72/255 — the CPU path's swscale conversion rounds
  slightly dark. Eye order, trim and frame-accurate seek all match the old
  path exactly.
- Requires a **10-bit H.264/HEVC** source, which is what VideoToolbox can
  hand over as P010. Anything else — 8-bit, ProRes, a multi-segment clip, or
  an export with temporal noise reduction on — falls back to the existing CPU
  path automatically, with the reason in the log. Windows is untouched.
- **Fixed:** the source bit depth was read from `bits_per_raw_sample`, which
  is frequently `0` even for genuine 10-bit files (any x265-muxed clip, for
  one). The depth now falls back to the stream's pixel format, so the fast
  path actually engages instead of silently declining every clip.

### macOS: side-by-side sources decode several times faster
- **Generic side-by-side `.mp4` / `.mov` sources now decode multi-threaded
  on macOS.** libavcodec defaults a decoder to ONE thread and nothing in
  the app ever set otherwise, so these files — the catch-all path for any
  clip that isn't `.360` / `.osv` / `.insv` — decoded on a single core.
  A 301-frame reframe export of an 8256x4128 10-bit HEVC clip went from
  **114.6 s to 52.1 s (2.2x)**, and preview decode from 4.3 to ~18 fps.
  Output is byte-identical to before. Thread count is capped at 6 (the
  knee of the speed/memory curve) and is skipped entirely when a hardware
  decoder is driving, which leaves Windows exactly as it was.
- Hardware (VideoToolbox) decode for these sources is available behind
  `VR180_SBS_VT=1` but is **off by default**: it measured *slower* than
  threaded software decode here, because the frame has to be copied back
  from the GPU for the CPU pipeline, which costs more than the decode it
  saves. (On Windows the equivalent frames stay on the GPU, which is why
  it pays there.) It is restricted to H.264/HEVC and skipped for 8-bit
  full-range clips, both cases where it would otherwise change colour.
- **Fixed:** the cached scaler was built once from the first frame and
  reused forever. It now revalidates per frame, so a decoder that changes
  pixel format mid-stream rebuilds instead of failing every subsequent
  frame, and a frame that changes *size* mid-stream is now a clear error
  rather than a mis-split image.

### Fixed: 8-bit side-by-side input on Windows
- **8-bit SBS sources failed on the Windows GPU fast path.** The hardware
  decoder's frame format follows the source bit depth — 8-bit H.264/HEVC
  decodes to NV12, 10-bit to P010 — but the D3D11 YCbCr→RGB converter asked
  for 16-bit plane views unconditionally, which an NV12 surface rejects
  outright. The path was chosen on a GPU capability rather than on the
  source's actual format, so it committed to the fast path and then died on
  the first frame, with no fallback left. Since the fast path only started
  taking generic SBS files two days ago, this never shipped.
- The converter now reads the source's format and picks matching plane views
  and range constants, so **8-bit side-by-side input gets the same GPU fast
  path 10-bit already had** rather than merely falling back. Output is RGBA16
  either way, so nothing downstream changes, and 10-bit output is unchanged
  bit-for-bit. Anything the converter genuinely cannot sample is now declined
  up front, where falling back to the portable path still works.
- The same converter now honours the clip's **colour range tag**: a
  full-range (`pc`-tagged) side-by-side clip renders identically on the GPU
  fast path and the CPU fallback, using the same four expansion constants
  the macOS path uses. Previously the fast path always assumed limited range.

### Matching Eyes: exposure
- The **Matching Eyes** panel gains an **Eye Exposure (±EV)** slider
  alongside Eye CT and Eye Tint. Like them it applies oppositely to the two
  eyes — left `+`, right `−`, in stops — so a brightness difference between
  the two lenses can be dialled out without changing the overall level. A
  lens-to-lens exposure mismatch is a common cause of binocular rivalry (the
  eyes "fighting" instead of fusing), and until now only its colour could be
  corrected. Range ±0.5 stop, 0 = off; ↑/↓ nudge by 0.01.
- **Fixed:** the Matching Eyes trim was silently dropped on **8-bit exports**.
  That arm grades the composed side-by-side frame in one pass, so both eyes
  got the same (un-trimmed) plan, while every 10-bit and zero-copy arm applied
  it per eye. 8-bit now grades each half with its own plan. Output with no
  trim set is byte-identical to before.

### Snapshot
- **📷 Snapshot** button in the transport bar (`S`; Stop no longer has a
  hotkey): saves the
  frame on screen — grade, stabilization, reframe and all — as a 92 %
  JPEG in the export output folder (next to the source when none is
  set), named `<source>_frame<N>.jpg` (a second shot of the same frame
  gets `-2`, `-3`, …, so several reframed compositions of one frame all
  keep). Rendered at the **export's resolution** — native / 8K for
  VR180, the chosen per-eye size for Reframed — through the full-detail
  still path; a snapshot mid-playback pauses on that frame first.

### 3D display output (new)
- **"3D display" toggle** in the toolbar: shows the stereo pair on a
  side-by-side 3D monitor or AR glasses that appear to the OS as one wide
  screen (e.g. 3840×1080) — left eye in the left half, right eye in the
  right, on a chrome-less fullscreen output that follows the live preview
  (pan / zoom / grade / stabilization included). When a 3840×1080-class
  side-by-side screen is connected (macOS and Windows) the output goes
  fullscreen on it automatically; the preview switches to SBS and each eye
  renders at half the screen width so a 16:9 reframed eye fills its half
  1:1 (other aspects are letterboxed inside each half). `Esc` closes it.
  With no such screen connected a movable window opens instead — drag it
  onto the 3D screen and press `F` for fullscreen. macOS: honours
  "Displays have separate Spaces" (native fullscreen on the glasses when
  on, a borderless window at the screen bounds when off — each is the
  only mode that works under that setting).

### Reframed (Flat 3D) export bitrate
- Reframed exports now have their **own H.265 bitrate** (10–300 Mbps),
  separate from the VR180 rate, **seeded from a per-size recommendation**
  — about 0.28 bits per pixel per frame at 30 fps, ×1.5 at 60 fps:
  3840×1080 → 35, 5120×1440 → 60, 7680×2160 → 140 Mbps at 30 fps
  (20 / 35 / 80 for the 2:1 sizes). The Format window shows the
  recommendation for the current size and clip frame rate; dragging the
  slider pins a custom value, "Use recommended" un-pins it. Previously
  the shared 200 Mbps VR180 default applied to Flat 3D too — 5–15× more
  than those frames can use.

### Generic side-by-side input
- **"Dewarp fisheye input" toggle** (Source panel) for plain `.mp4` /
  `.mov` side-by-side sources. Off by default: the file is taken as an
  already-dewarped VR180 half-equirect SBS and sampled directly, so a
  finished VR180 export can be reframed to Flat 3D (or re-aligned /
  re-graded) without the fisheye dewarp distorting it. Turn it on for raw
  dual-fisheye SBS recordings, which then use the Fisheye lens settings
  as before. `.360` / `.osv` / `.insv` are unaffected — their lens model
  always comes from the file.
- Fixed: 10-bit (H.265 10-bit / ProRes) exports of a plain side-by-side
  source failed at the first frame with a wgpu validation error — the
  SBS decoder always produced 8-bit frames while the 10-bit arms expect
  16-bit ones. The SBS decoder now follows the output bit depth, so a
  10-bit SBS source also keeps its precision.
- **Laptops with two GPUs no longer crash on the fast path.** When the
  video decoder and the renderer ended up on different GPUs (common on
  laptops with both an integrated and a discrete chip), the accelerated
  path could take the app down. It now detects the mismatch and uses the
  compatible path instead — slower, but it works.
- **Failures are now reported instead of looking like success.** An export
  that lost its decoder mid-run used to finalize a short file and report
  "done"; it now fails with the reason (cancelling still keeps the partial
  file, as before). A crash inside the preview decoder no longer leaves
  Play/Pause toggling a dead worker. And a GPU error is caught and shown
  as a dismissable warning instead of taking the whole app down.
- **Windows: hardware decode now checks free GPU memory first, and falls
  back to software when it will not fit.** A hardware video decoder needs
  roughly 240 MB of GPU memory per megapixel of video, so an 8K clip wants
  about 8 GB and a dual-lens camera about 7 GB. On a card that cannot spare
  that, the app now decodes in software automatically — slower, but it
  works — instead of the preview silently freezing or an export stopping
  early. The export bar says why ("not enough GPU memory for hardware
  decode (7.9 GB needed, 5.9 GB free)"). Cards with room are unaffected.
  `VR180_NO_HW_DECODE=1` forces software decode; `VR180_FORCE_HW_DECODE=1`
  skips the check.
- **Windows: side-by-side sources are now GPU-accelerated end to end** —
  hardware (NVDEC) decode, GPU eye split, projection, color and encode,
  with the audio muxed inline. Exports measured 5 → 37 fps for a 4K
  H.265 export (~7×; ProRes rides the GPU encoder the same way), and
  **playback now uses the same zero-copy path** instead of downloading
  and converting every frame on the CPU — the SBS decoder had never used
  hardware decode at all, which hit 8K side-by-side files hardest.

## 2.5.0

### Reframed output mode (new)
- **Format → "Reframed (Flat 3D)"**: a pinhole-style side-by-side view
  of each eye instead of the VR180 half-equirect — zoom (horizontal FOV),
  pan / tilt / roll, a **Defish** blend from rectilinear to a fisheye look,
  and a 1:1 or 16:9 per-eye frame. Stabilization, stereo offsets, per-row
  rolling-shutter correction and the lens override all still apply.
  Available for DJI OSMO, Insta360 X6 and GoPro sources.
- Drag the preview to pan, scroll or pinch to zoom, double-click to
  recenter. The preview renders from the native frame and a paused frame
  shows the native-resolution still.
- Export writes the exact viewport at 1080 / 1440 / 2160 lines per eye —
  2:1 square or **32:9 side-by-side** (3840×1080 … 7680×2160) for AR
  glasses — with no VR180 metadata. The BeyondVR hack does not apply.

### Stabilization
- **Timing is derived from the file, no more IMU phase slider.** Each
  frame's pose is sampled at the centre row's mid-exposure using the
  file's sensor readout, exposure record and timestamps, so dark and
  bright clips and every frame rate are timed right automatically
  (verified on 25 / 30 / 50 fps DJI clips, consistent with DJI Studio's
  output).
- **Insta360 X6 (`.insv`)**: factory lens model, gyro stabilization
  matched to Insta360 Studio's output, per-sensor exposure timing, and
  Insta360's official X6 I-Log→Rec.709 LUT (v2, 65-point) bundled and
  auto-applied like the DJI and GoPro ones.
- **OSMO 360 II** support and **Auto align** stereo alignment; GoPro
  chapter-safe firmware-RS detection.
- **Camera lock** is now an explicit toggle in every stabilization panel
  (the GoPro `.360` panel gains one; OSV/INSV already had it). It locks the
  view to the first frame and ignores the smoothing *and* max-correction
  controls — fully locked no matter what the camera does. The old
  "Smooth = 0 means lock" convention is gone; Smooth is now a pure
  smoothing amount and grays out (with Max corr / Response) under the lock.

### App / UX
- Removing the currently loaded clip from the clip list now unloads it:
  the app activates the next remaining clip, or returns to the empty
  "no clip loaded" state when the list is emptied.
- Export progress / ETA fixed: totals now honour each clip's trim (a
  trimmed export used to stall short of 100% with an inflated ETA), and
  the rate is measured from the first written frame instead of the run
  start, so load / encoder start-up no longer counts as encode time.
- **Windows: GoPro `.360` ProRes (and software H.265) exports now run on
  the GPU fast path** — GPU decode, assembly, projection, color and 4:2:2
  compose feeding the GPU ProRes encoder (~5× at 8K, was a serial CPU
  loop with the GPU idle). Merged multi-chapter recordings included.
- Windows: the app now explicitly prefers the Vulkan GPU backend (all
  fast export paths require it), and when an export does land on a slow
  path the export bar says why (e.g. "CPU export path — ProRes GPU
  encoder unavailable") instead of just being slow.
- **Exports no longer stall at 100%**: stereo-audio exports on every
  path on both platforms — Windows hardware H.265 (NVENC) and the macOS
  GoPro `.360` zero-copy arm included — now mux the audio inline while
  encoding, straight into the final file (merged
  multi-segment recordings too) — previously the whole encoded video was
  rewritten afterwards to add the audio, which at ProRes bitrates could
  take longer than the encode itself. Paths that still need the second
  pass (ambisonic / APAC) now show "muxing audio…" in the export bar
  instead of sitting silently at 100%.

### Since 2.0.0
- Seamless auto-update (2.1.0), `.360` lens calibration override, ProRes
  4:2:2 with GPU compose, 8K export default, Matching Eyes white-balance
  trim, BeyondVR Hack (VR180 output only), frame-exact multi-segment seams.

## 2.0.0

The `2.0` clean-room rewrite of VR180 Silver Bullet — a native Rust + wgpu
application replacing the Python/PyQt6 app. **The headline addition is full
support for the OSMO VR180 Mod** (`.osv`). One self-contained binary
per platform, no Python runtime, no system `ffmpeg`. Runs on **macOS (Apple
Silicon)** and **Windows (NVIDIA)**.

### Cameras & formats
- **OSMO VR180 Mod** (`.osv`) — **the headline of 2.0.** Exact
  per-lens factory dewarp loaded from the file (5-coefficient Kannala-Brandt
  + Brown-Conrady tangential), with output on par with DJI Studio.
- **GoPro Max 2 VR180 Mod** (`.360`, EAC) — full GPU pipeline: zero-copy decode,
  noise reduction, and **automatic firmware vs no-firmware rolling-shutter
  detection** from the CORI stream (manual override retained).

### Engine
- GPU-first: `wgpu` compute (Metal / DX12 / Vulkan) with WGSL shaders.
- In-process video I/O via `ffmpeg-next` 8.1 (no subprocess).
- **macOS:** VideoToolbox zero-copy P010 decode/encode through IOSurface,
  HEVC and hardware ProRes.
- **Windows:** GPU-resident export — NVDEC → wgpu → CUDA → NVENC, no CPU
  readback (~36 fps @ 8K on a 4090), libx265 fallback.
- 10-bit end-to-end (Rgba16Unorm intermediates) when 10-bit output is
  selected — decode, projection, color stack, and encode all hold ≥10-bit.

### Stabilization & rolling shutter
- Camera-lock and velocity-dampened soft-stab (adaptive smoothing with a
  **Response** slider and a soft elastic correction limit).
- Per-scanline rolling-shutter correction from measured sensor-readout
  timing; gravity/horizon alignment.
- Precise OSV IMU stabilization + rolling-shutter timing (SROT, IMU phase), matched to DJI Studio.
- **New:** GoPro Max 2 VR180 Mod (`.360`) firmware-RS mode is auto-detected
  per clip from the CORI signal (the toggle still overrides).

### Color
- CDL, 3D LUT (DJI D-LogM→Rec.709 bundled + autoloaded), white balance,
  saturation, sharpen, mid-detail — identical stack in preview and export,
  matched to the Python app.

### Noise reduction
- Temporal NR via Apple `VTTemporalNoiseFilter`, ported **in-process** (objc2
  FFI, no Swift helper), fully 10-bit, GPU-resident zero-copy for OSV and
  `.360`. Export-only; macOS-only (auto-hidden where unsupported).

### Output & delivery
- Half-equirect VR180 SBS, or a normalized equidistant fisheye SBS matched
  to the lens — **195°** for the OSMO VR180 Mod, **185°** for the
  GoPro Max 2 VR180 Mod.
- Native or **8192×4096 (8K)** resolution.
- H.265 or ProRes; **Vision Pro (APMP)** and **YouTube VR180** metadata
  injection; **APAC spatial** / ambisonic / stereo audio; OSV audio
  passthrough.
- Trim-accurate exports (video + audio aligned to the trim range).

### App / UX
- Native desktop GUI (eframe/egui + egui-wgpu + wgpu 29).
- **Unified batch + export:** one queue for single or many clips, a
  persistent bottom export bar with overall progress + ETA, per-clip
  multi-select, and a completion notification. (The separate batch and
  export-options windows were removed.)
- Preview modes (SBS / anaglyph / 50% overlay / single eye), zoom magnifier
  with a native-resolution still, per-eye view adjustment, upside-down mount.
- **Localized UI: English / 简体中文** (live toggle, bundled CJK font).
- Settings persist per-OS; RS mode and IMU phase are per-clip.

### Packaging
- macOS: signed + notarized `.app` / `.dmg`.
- Windows: Inno Setup installer (per-user, Start Menu shortcut).

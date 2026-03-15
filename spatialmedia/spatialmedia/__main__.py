#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spatial Media Metadata Injector 

Tool for examining and injecting spatial media metadata in MP4/MOV files.
"""

import argparse
import os
import re
import sys

path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, '..')
sys.path.insert(0, path)
from spatialmedia import metadata_utils


def console(contents):
  print(contents)


def main():
  """Main function for printing and injecting spatial media metadata."""

  parser = argparse.ArgumentParser(
      usage=
      "%(prog)s [options] [files...]\n\nBy default prints out spatial media "
      "metadata from specified files.")
  parser.add_argument(
      "-i",
      "--inject",
      action="store_true",
      help=
      "injects spatial media metadata into the first file specified (.mp4 or "
      ".mov) and saves the result to the second file specified")

  video_group = parser.add_argument_group("Spherical Video")
  video_group.add_argument(
      "-s",
      "--stereo",
      action="store",
      dest="stereo_mode",
      metavar="STEREO-MODE",
      choices=["none", "top-bottom", "left-right", "custom"],
      default=None,
      help="stereo mode (none | top-bottom | left-right | custom)")
  video_group.add_argument(
      "-m",
      "--projection",
      action="store",
      dest="projection",
      metavar="SPHERICAL-MODE",
      choices=["equirectangular", "cubemap", "mesh", "full-frame", "equi-mesh"],
      default=None,
      help="projection (equirectangular | cubemap | mesh | full-frame | equi-mesh)")
  video_group.add_argument(
      "-y",
      "--yaw",
      action="store",
      dest="yaw",
      metavar="YAW",
      default=0,
      help="yaw")
  video_group.add_argument(
      "-p",
      "--pitch",
      action="store",
      dest="pitch",
      metavar="PITCH",
      default=0,
      help="pitch")
  video_group.add_argument(
      "-r",
      "--roll",
      action="store",
      dest="roll",
      metavar="ROLL",
      default=0,
      help="roll")
  video_group.add_argument(
      "-d",
      "--degrees",
      action="store",
      dest="degrees",
      metavar="DEGREES",
      choices=["180", "360"],
      default=180,
      help="degrees")
  video_group.add_argument(
       "-c",
       "--correction",
       action="store",
       dest="fisheye_correction",
       metavar="FISHEYE-CORRECTION",
       default="0:0:0:0",
       help="polynomial fisheye lens correction (n1:n2:n3:n4) e.g 0.5:-0.1:0.2:-0.0005")
  video_group.add_argument(
       "-v",
       "--view",
       action="store",
       dest="field_of_view",
       metavar="FIELD-OF-VIEW",
       default="0x0",
       help="Field of view for equi_mesh or full frame. e.g. 180x180 or 16x9")
       
  audio_group = parser.add_argument_group("Spatial Audio")
  audio_group.add_argument(
      "-a",
      "--spatial-audio",
      action="store_true",
      help=
      "spatial audio. First-order periphonic ambisonics with ACN channel "
      "ordering and SN3D normalization")
  parser.add_argument("file", nargs="+", help="input/output files")

  args = parser.parse_args()

  if args.inject:
    if len(args.file) != 2:
      console("Injecting metadata requires both an input file and output file.")
      return

    metadata = metadata_utils.Metadata()

    if args.stereo_mode:
      metadata.stereo = args.stereo_mode

    if args.projection:
      metadata.spherical = args.projection
      if metadata.spherical == "equirectangular":
          metadata.clip_left_right = 0 if args.degrees == "360" else 1073741823

    if args.spatial_audio:
      parsed_metadata = metadata_utils.parse_metadata(args.file[0], console)
      if not metadata.audio:
        spatial_audio_description = metadata_utils.get_spatial_audio_description(
            parsed_metadata.num_audio_channels)
        if spatial_audio_description.is_supported:
          metadata.audio = metadata_utils.get_spatial_audio_metadata(
              spatial_audio_description.order,
              spatial_audio_description.has_head_locked_stereo)
        else:
          console("Audio has %d channel(s) and is not a supported "
                  "spatial audio format." % (parsed_metadata.num_audio_channels))
          return


    if args.fisheye_correction:
        metadata.fisheye_correction = [float(x) for x in args.fisheye_correction.split(':')]

    if args.field_of_view:
      metadata.fov = [float(x) for x in args.field_of_view.split('x')]
      if metadata.fov[0] == 0 or metadata.fov[1] == 0 :
        if args.projection == "full-frame" :
           metadata.fov[0] = 16.0
           metadata.fov[1] = 9.0       
        else :
           metadata.fov[0] = 180 
           metadata.fov[1] = 180;       


    if metadata.stereo or metadata.spherical or metadata.audio:
      metadata.orientation = {"yaw": args.yaw, "pitch": args.pitch, "roll": args.roll}
      metadata_utils.inject_metadata(args.file[0], args.file[1], metadata,
                                     console)
    else:
      console("Failed to generate metadata.")
    return

  if len(args.file) > 0:
    for input_file in args.file:
      if args.spatial_audio:
        parsed_metadata = metadata_utils.parse_metadata(input_file, console)
        metadata.audio = metadata_utils.get_spatial_audio_description(
            parsed_metadata.num_channels)
      metadata_utils.parse_metadata(input_file, console)
    return

  parser.print_help()
  return


if __name__ == "__main__":
  main()

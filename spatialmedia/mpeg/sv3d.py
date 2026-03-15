#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Vimeo. All rights reserved.
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

"""MPEG sv3d box processing classes.

Enables the injection of an sv3d MPEG-4. The sv3d box specification
conforms to that outlined in docs/spherical-video-v2-rfc.md
"""

import struct

from spatialmedia.mpeg import box
from spatialmedia.mpeg import constants
from spatialmedia.mpeg import mesh_projection


def load(fh, position=None, end=None):
    """ Loads the sv3d box located at position in an mp4 file.

    Args:
      fh: file handle, input file handle.
      position: int or None, current file position.

    Returns:
      new_box: box, sv3d box loaded from the file location or None.
    """
    if position is None:
        position = fh.tell()

    fh.seek(position)
    new_box = sv3dBox()
    new_box.position = position
    size = struct.unpack(">I", fh.read(4))[0]
    name = fh.read(4).decode('latin1')

    if (name != constants.TAG_SV3D):
        print ("Error: box is not an sv3d box.")
        return None

    if (position + size > end):
        print ("Error: sv3d box size exceeds bounds.")
        return None

    size = struct.unpack(">I", fh.read(4))[0]
    fh.read(size - 4) #svhd

    fh.read(4 + 4) #proj
    fh.read(4 + 4 + 4) #prhd
    new_box.yaw = struct.unpack(">I", fh.read(4))[0] / 65536
    new_box.pitch = struct.unpack(">I", fh.read(4))[0] / 65536
    new_box.roll = struct.unpack(">I", fh.read(4))[0] / 65536

    projection_position = fh.tell()

    fh.read(4) #size
    proj = fh.read(4).decode('latin1')
    if proj == "equi":
        new_box.projection = "equirectangular"
        tmp = struct.unpack(">I", fh.read(4))[0]
        new_box.clip_top = struct.unpack(">I", fh.read(4))[0]
        new_box.clip_bottom = struct.unpack(">I", fh.read(4))[0]
        new_box.clip_left_right = struct.unpack(">I", fh.read(4))[0]
        new_box.clip_right = struct.unpack(">I", fh.read(4))[0]
    elif proj == "cbmp":
        new_box.projection = "cubemap"
    elif proj == "ytmp":
        new_box.projection = "mesh"
        new_box.projection_box = mesh_projection.load(fh, projection_position, end)
    elif proj == "mshp":
        new_box.projection = "mesh"
        new_box.projection_box = mesh_projection.load(fh, projection_position, end)
    else:
        print ("Unknown projection type. " + new_box.projection)
        return None

    return new_box


class sv3dBox(box.Box):
    def __init__(self):
        box.Box.__init__(self)
        self.name = constants.TAG_SV3D
        self.header_size = 8
        self.proj_size = 0
        self.content_size = 0
        self.projection = ""
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.clip_left_right = 0;
        self.clip_right = 0;
        self.clip_top = 0;
        self.clip_bottom = 0;
        self.projection_box=None;

    @staticmethod
    def create(metadata):
        new_box = sv3dBox()
        new_box.header_size = 8
        new_box.name = constants.TAG_SV3D
        new_box.projection = metadata.spherical
        if new_box.projection == "equirectangular":
            new_box.content_size = 81 - new_box.header_size
            new_box.proj_size = 60
        elif new_box.projection == "cubemap":
            new_box.content_size = 73 - new_box.header_size
            new_box.proj_size = 52
        if new_box.projection == "mesh" or new_box.projection == "full-frame" or new_box.projection == "equi-mesh" :
            new_box.projection_box = mesh_projection.mshpBox.create(metadata)
            new_box.proj_size = new_box.projection_box.content_size + new_box.projection_box.header_size + 8 + 24 # mesh projection, proj and  prhd boxes
            new_box.content_size = new_box.proj_size + 21 - new_box.header_size  # 21 is size of svhd & proj boxes
        else:
            new_box.projection_box=None;
        new_box.yaw = float(metadata.orientation["yaw"])
        new_box.pitch = float(metadata.orientation["pitch"])
        new_box.roll = float(metadata.orientation["roll"])
        new_box.clip_left_right = metadata.clip_left_right;
        new_box.clip_right = metadata.clip_left_right;

        return new_box

    def print_box(self, console):
        """ Prints the contents of this spherical (sv3d) box to the
            console.
        """
        console("\t\tSpherical Mode: %s" % self.projection)
        if self.projection_box:
            self.projection_box.print_box(console)
        console("\t\t    [Yaw: %.02f, Pitch: %.02f, Roll: %.02f]" % (self.yaw, self.pitch, self.roll))
        console("\t\t    [Clip Top: %d, Bottom: %d, Left: %d Right: %d]" % (self.clip_top, self.clip_bottom, self.clip_left_right, self.clip_right))

    def get_metadata_string(self):
        """ Outputs a concise single line audio metadata string. """
        return "Spherical mode: %s (%f,%f,%f) (%d,%d,%d,%d)" % (self.projection, self.yaw, self.pitch, self.roll, self.clip_top, self.clip_bottom, self.clip_left_right, self.clip_right)

    def save(self, in_fh, out_fh, delta):

        if (self.header_size == 16):
            out_fh.write(struct.pack(">I", 1))
            out_fh.write(struct.pack(">Q", self.size()))
            out_fh.write(self.name.encode('latin1'))
        elif(self.header_size == 8):
            out_fh.write(struct.pack(">I", self.content_size + self.header_size))
            out_fh.write(self.name.encode('latin1'))

        #svhd
        out_fh.write(struct.pack(">I", 13))     # size
        out_fh.write("svhd".encode('latin1'))           # tag
        out_fh.write(struct.pack(">I", 0))      # version+flags
        out_fh.write(struct.pack(">B", 0))      # metadata

        #proj
        out_fh.write(struct.pack(">I", self.proj_size)) #size
        out_fh.write("proj".encode('latin1'))                   # proj

        #prhd
        out_fh.write(struct.pack(">I", 24))     # size
        out_fh.write("prhd".encode('latin1'))           # tag
        out_fh.write(struct.pack(">I", 0))      # version+flags
        out_fh.write(struct.pack(">I", int(self.yaw * 65536)))   # yaw
        out_fh.write(struct.pack(">I", int(self.pitch * 65536))) # pitch
        out_fh.write(struct.pack(">I", int(self.roll * 65536)))  # roll

        #cmbp or equi
        if self.projection == "equirectangular":
            out_fh.write(struct.pack(">I", 28)) # size
            out_fh.write("equi".encode('latin1'))       # tag
            out_fh.write(struct.pack(">I", 0))  # version+flags
            out_fh.write(struct.pack(">I", 0))
            out_fh.write(struct.pack(">I", 0))
            out_fh.write(struct.pack(">I", self.clip_left_right))
            out_fh.write(struct.pack(">I", self.clip_left_right))
        elif self.projection == "cubemap":
            out_fh.write(struct.pack(">I", 20)) # size
            out_fh.write("cbmp".encode('latin1'))       # tag
            out_fh.write(struct.pack(">I", 0))  # version+flags
            out_fh.write(struct.pack(">I", 0))  # layout
            out_fh.write(struct.pack(">I", 0))  # padding
        elif self.projection == "mesh" or self.projection == "full-frame" or self.projection == "equi-mesh" :
            self.projection_box.save(in_fh, out_fh, delta)



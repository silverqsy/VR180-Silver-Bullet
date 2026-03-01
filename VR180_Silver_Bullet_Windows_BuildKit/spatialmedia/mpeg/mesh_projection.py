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

"""MPEG sv3d mesh processing classes.

Enables the injection of an sv3d mesh projection box.
The mesh projection box specification conforms to that outlined in docs/spherical-video-v2-rfc.md
"""

import struct
import os
import io
import zlib

from spatialmedia.mpeg import box
from spatialmedia.mpeg import constants
from spatialmedia.mpeg import mesh


def load(fh, position=None, end=None):
    """ Loads the mesh projection box located at position in an mp4 file.

    Args:
      fh: file handle, input file handle.
      position: int or None, current file position.

    Returns:
      new_box: box, mesh projection box loaded from the file location or None.
    """
    if position is None:
        position = fh.tell()

    fh.seek(position)
    new_box = mshpBox()
    new_box.position = position
    size = struct.unpack(">I", fh.read(4))[0]
    name = fh.read(4).decode('latin1')

    if (name != 'ytmp' and name != 'mshp'):
        print ("Error: box is not an mesh projection box. " + name)
        return None

    if (position + size > end):
        print ("Error: mesh projection box size exceeds bounds.")
        return None

    tmp = struct.unpack(">I", fh.read(4))[0]
    crc32 = struct.unpack(">I", fh.read(4))[0]
    encoding = fh.read(4).decode('latin1')

    if encoding == 'dfl8':
         mesh_data = zlib.decompress(fh.read(size - 16), -15)
    elif encoding == 'raw ':
         mesh_data = fh.read(size - 16)

    new_box.meshbox = mesh.meshBox()

    meshfh = io.BytesIO(mesh_data)
    meshbox = mesh.load(meshfh, 0,  len(mesh_data))
    new_box.meshes.append(meshbox)

    if meshbox.content_size < len(mesh_data):
        meshbox = mesh.load(meshfh, meshfh.tell(), len(mesh_data))
        new_box.meshes.append(meshbox)

    new_box.meshbox.meshes = len (new_box.meshes)

    return new_box


class mshpBox(box.Box):
    def __init__(self):
        box.Box.__init__(self)
        self.name = 'mshp'
        self.header_size = 16
        self.content_size = 0
        self.meshes = [];
        self.crc32 = 0;
        """
        self.encoding = 'dfl8';
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, 'mesh_projection.dat')
        with open(filename, 'rb') as content_file:
            self.content = content_file.read();
            self.content_size = len(self.content) - 8; # file has flags and crc32 included in header
        """
    @staticmethod
    def create(metadata):
        new_box = mshpBox()
        new_box.projection = metadata.spherical
        new_box.meshbox = mesh.meshBox.create(metadata);
        new_box.encoding = b'dfl8';
        compobj = zlib.compressobj(9, zlib.DEFLATED, -15)
        new_box.contents = compobj.compress(new_box.meshbox.contents) + compobj.flush(zlib.Z_FINISH)
        new_box.content_size = len(new_box.contents) + 4

        return new_box

    def print_box(self, console):
        """ Prints the contents of this spherical (mshp) box to the
            console.
        """
        console("\t\tMesh Projection:" )
        self.meshbox.print_box(console)

    def get_metadata_string(self):
        """ Outputs a concise single line audio metadata string. """
        return "Mesh Projection: " + self.meshbox.get_metadata_string();

    def save(self, in_fh, out_fh, delta):
        
        """
             just write in a standard projection with oredefined meshes for now
        """
        
        out_fh.write(struct.pack(">I", self.content_size + self.header_size))
        out_fh.write(self.name.encode('latin1'))
        out_fh.write(struct.pack(">I", 0))               # version+flags
        out_fh.write(struct.pack(">I", zlib.crc32(self.encoding +  self.contents)& 0xffffffff))
        out_fh.write(self.encoding)
        out_fh.write(self.contents)

        """
            something like this for future use
        
        

        self.crc32 = binascii.crc32(self.encoding, self.crc32)
        for meshbox in self.meshes:
            self.crc32 = binascii.crc32(meshbox.meshdata,self.crc32)
            self.header_size += len(meshbox.meshdata)

        out_fh.write(struct.pack(">I", self.header_size + 16))
        out_fh.write(self.name.encode('latin1'))
        out_fh.write(struct.pack(">I", 0))               # version+flags
        out_fh.write(struct.pack(">I", self.crc32))      # crc32
        
        for meshbox in self.meshes:
            out_fh.write(meshbox.meshdata)           # mesh boxs

        """

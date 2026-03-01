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

"""Spatial Media Metadata Injector GUI 

GUI application for examining/injecting spatial media metadata in MP4/MOV files.
"""

import ntpath
import os
import sys
from sys import version_info
if version_info.major == 2:
    import tkFileDialog
    import tkMessageBox
elif version_info.major == 3:
    import tkinter.filedialog as tkFileDialog    
    import tkinter.messagebox as tkMessageBox

import traceback

try:
    if version_info.major == 2:
        from Tkinter import *
    elif version_info.major == 3:
        from tkinter import *
except ImportError:
    print("Tkinter library is not available.")
    exit(0)

path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, '..')
sys.path.insert(0, path)
from spatialmedia import metadata_utils 

SPATIAL_AUDIO_LABEL = "My video has spatial audio (ambiX ACN/SN3D format)"
HEAD_LOCKED_STEREO_LABEL = "with head-locked stereo"

class Console(object):
    def __init__(self):
        self.log = []

    def append(self, text):
        print(text.encode('latin1'))
        self.log.append(text)


class Application(Frame):
    def action_open(self):
        """Triggers open file diaglog, reading a new file's metadata."""
        tmp_in_file = tkFileDialog.askopenfilename(**self.open_options)
        if not tmp_in_file:
            return
        self.in_file = tmp_in_file

        self.set_message("Current 360 video: %s" % ntpath.basename(self.in_file))

        console = Console()
        parsed_metadata = metadata_utils.parse_metadata(self.in_file,
                                                        console.append)

        metadata = None
        audio_metadata = None
        if parsed_metadata:
            metadata = parsed_metadata.video
            audio_metadata = parsed_metadata.audio
            stereo = parsed_metadata.stereo

        for line in console.log:
            if "Error" in line:
                self.set_error("Failed to load file %s"
                               % ntpath.basename(self.in_file))
                self.disable_state()
                self.button_open.configure(state="normal")
                return

        self.enable_state()
        #        self.checkbox_spherical.configure(state="normal")

        infile = os.path.abspath(self.in_file)
        file_extension = os.path.splitext(infile)[1].lower()

        #   self.var_spherical.set(1)
        self.spatial_audio_description = metadata_utils.get_spatial_audio_description(
	            parsed_metadata.num_audio_channels)

        # print("self.spatial_audio_description", self.spatial_audio_description)
        # print("parsed_metadata.num_audio_channels", parsed_metadata.num_audio_channels)

        if not metadata:
            #    self.var_3d.set(0)
            self.char_degrees.set("180")
            self.char_layout.set("left-right")

        if not audio_metadata:
            self.var_spatial_audio.set(0)

        if metadata:
            # metadata here is an dict with a sv3d box

            print(metadata)
            metadata = list(metadata.values())[0]
            if metadata.clip_left_right > 0 or metadata.projection == "mesh":
                self.char_degrees.set("180")
            else:
                self.char_degrees.set("360")

            if metadata.projection == "mesh":
                self.char_format.set("fisheye")
            elif metadata.projection == "equirectangular":
                self.char_format.set("equi-rectangular")
            else:
                self.char_format.set(metadata.projection)

            """if metadata.get("Spherical", "") == "true":
                self.var_spherical.set(1)
            else:
                self.var_spherical.set(0)

            if metadata.get("StereoMode", "") == "top-bottom":
                self.var_3d.set(1)
            else:
                self.var_3d.set(0)
            """

        if stereo:
            stereo = list(stereo.values())[0]
            self.char_layout.set(stereo.stereo_mode_name());

        if audio_metadata:
            self.var_spatial_audio.set(1)
            print (audio_metadata.get_metadata_string())

        self.update_state()

    def action_inject_delay(self):
        
        
        stereo = self.char_layout.get()
        spherical = self.char_format.get()
        degrees = self.char_degrees.get()
        
        if spherical == "fisheye":
            spherical = "mesh"
            stereo = "left-right"
            degrees = "180"
        elif spherical == "equi-rectangular":
            spherical = "equirectangular"
        
        if stereo == 'mono':
            if degrees == "180":
                stereo = "left-right"
            else:
                stereo = "none"
            
        metadata = metadata_utils.Metadata()
        
        metadata.stereo = stereo
        metadata.spherical = spherical
        if degrees == "180":
            metadata.clip_left_right = 1073741823
        else:
            metadata.clip_left_right = 0

        if self.var_spatial_audio.get():
          metadata.audio = metadata_utils.get_spatial_audio_metadata(
              self.spatial_audio_description.order,
              self.spatial_audio_description.has_head_locked_stereo)

        if spherical == "full-frame" :
            metadata.fov = [16.0, 9.0]
            
        console = Console()
    
        if metadata.stereo or metadata.spherical or metadata.audio:
            metadata.orientation = {"yaw": 0, "pitch": 0, "roll": 0}
            metadata_utils.inject_metadata(self.in_file, self.save_file, metadata, console.append)
            self.set_message("Successfully saved file to %s\n"
                     % ntpath.basename(self.save_file))
            self.button_open.configure(state="normal")
            self.update_state()
        else:
            console("Failed to generate metadata.")
        return


    def action_inject(self):
        """Inject metadata into a new save file."""
        split_filename = os.path.splitext(ntpath.basename(self.in_file))
        base_filename = split_filename[0]
        extension = split_filename[1]
        self.save_options["initialfile"] = (base_filename
                                            + "_injected" + extension)
        self.save_file = tkFileDialog.asksaveasfilename(**self.save_options)
        if not self.save_file:
            return

        self.set_message("Saving file to %s" % ntpath.basename(self.save_file))

        # Launch injection on a separate thread after disabling buttons.
        self.disable_state()
        self.master.after(100, self.action_inject_delay)

    def action_set_spherical(self):
        self.update_state()

    def action_set_spatial_audio(self):
        self.update_state()

    def action_set_3d(self):
        self.update_state()

    def enable_state(self):
        self.button_open.configure(state="normal")

    def disable_state(self):
        #     self.checkbox_spherical.configure(state="disabled")
        #     self.checkbox_3D.configure(state="disabled")
        self.checkbox_spatial_audio.configure(state="disabled")
        self.button_inject.configure(state="disabled")
        self.button_open.configure(state="disabled")

    def update_state(self):
        """
        self.checkbox_spherical.configure(state="normal")
        if self.var_spherical.get():
            self.checkbox_3D.configure(state="normal")
            self.button_inject.configure(state="normal")
            if self.enable_spatial_audio:
                self.checkbox_spatial_audio.configure(state="normal")
        else:
            self.checkbox_3D.configure(state="disabled")
            self.button_inject.configure(state="disabled")
            self.checkbox_spatial_audio.configure(state="disabled")
        """
        if self.spatial_audio_description.is_supported:
            self.checkbox_spatial_audio.configure(state="normal")
        self.button_inject.configure(state="normal")

        if self.spatial_audio_description.has_head_locked_stereo:
            self.label_spatial_audio.configure(
                        text='{}\n{}'.format(
	                SPATIAL_AUDIO_LABEL, HEAD_LOCKED_STEREO_LABEL))
        else:
            self.label_spatial_audio.configure(text=SPATIAL_AUDIO_LABEL)
	


    def set_error(self, text):
        self.label_message["text"] = text
        self.label_message.config(fg="red")

    def set_message(self, text):
        self.label_message["text"] = text
        self.label_message.config(fg="blue")

    def create_widgets(self):
        """Sets up GUI contents."""

        row = 0
        column = 0

        PAD_X = 10

        row = row + 1
        column = 0
        self.label_message = Label(self)
        self.label_message["text"] = "Click Open to open your video."
        self.label_message.grid(row=row, column=column, rowspan=1,
                                columnspan=2, padx=PAD_X, pady=10, sticky=W)

        row = row + 1
        separator = Frame(self, relief=GROOVE, bd=1, height=2, bg="white")
        separator.grid(columnspan=row, padx=PAD_X, pady=4, sticky=N+E+S+W)

        # video format
        row += 1
        self.label_degrees = Label(self, anchor=W)
        self.label_degrees["text"] = "My video is foramtted as...."
        self.label_degrees.grid(row=row, column=column, padx=PAD_X, pady=7, sticky=W)
        column += 1
        
        self.char_format = StringVar()
        self.char_format.set("equi-rectangular")
        self.format_menu = OptionMenu(self, self.char_format, "equi-rectangular", "fisheye", "full-frame", "cubemap")
        self.format_menu.grid(row=row, column=column,  padx=PAD_X, pady=7, sticky=W)


        # 180 or 360
        column = 0
        row += 1
        self.label_degrees = Label(self, anchor=W)
        self.label_degrees["text"] = "My video is 180 or 360 degree..."
        self.label_degrees.grid(row=row, column=column, padx=PAD_X, pady=7, sticky=W)
        column += 1
      
        self.var_degrees = IntVar()
        
        
        self.char_degrees = StringVar()
        self.char_degrees.set(180)

        self.degee_menu = OptionMenu(self, self.char_degrees, "180", "360")
        self.degee_menu.grid(row=row, column=column,  padx=PAD_X, pady=7, sticky=W)
        
        """
        self.var_degrees.set(1073741823)
        """
        # video layout
        column = 0
        row += 1
        self.label_layout = Label(self, anchor=W)
        self.label_layout["text"] = "My video layout is..."
        self.label_layout.grid(row=row, column=column, padx=PAD_X, pady=7, sticky=W)
        column += 1
        
        
        self.char_layout = StringVar()
        self.char_layout.set("left-right")
    
        self.degee_menu = OptionMenu(self, self.char_layout, "left-right", "top-bottom", "mono")
        self.degee_menu.grid(row=row, column=column,  padx=PAD_X, pady=7, sticky=W)


        # Spatial Audio Checkbox
        row += 1
        column = 0
        self.label_spatial_audio = Label(self, anchor=W, justify=LEFT)
        self.label_spatial_audio["text"] = SPATIAL_AUDIO_LABEL
        self.label_spatial_audio.grid(row=row, column=column, padx=PAD_X, pady=7, sticky=W)
    
        column += 1
        self.var_spatial_audio = IntVar()
        self.checkbox_spatial_audio = \
        Checkbutton(self, variable=self.var_spatial_audio)
        self.checkbox_spatial_audio["command"] = self.action_set_spatial_audio
        self.checkbox_spatial_audio.grid(row=row, column=column, padx=0, pady=0)
    
        row = row + 1
        separator = Frame(self, relief=GROOVE, bd=1, height=2, bg="white")
        separator.grid(columnspan=row, padx=PAD_X, pady=10, sticky=N+E+S+W)


        # Button Frame
        column = 0
        row = row + 1
        buttons_frame = Frame(self)
        buttons_frame.grid(row=row, column=0, columnspan=3, padx=PAD_X, pady=10)

        self.button_open = Button(buttons_frame)
        self.button_open["text"] = "Open"
        self.button_open["fg"] = "black"
        self.button_open["command"] = self.action_open
        self.button_open.grid(row=0, column=0, padx=14, pady=2)

        self.button_inject = Button(buttons_frame)
        self.button_inject["text"] = "Inject metadata"
        self.button_inject["fg"] = "black"
        self.button_inject["command"] = self.action_inject
        self.button_inject.grid(row=0, column=1, padx=14, pady=2)

    def __init__(self, master=None):
        master.wm_title("Spatial Media Metadata Injector")
        master.config(menu=Menu(master))
        self.title = "Spatial Media Metadata Injector"
        self.open_options = {}
        self.open_options["filetypes"] = [("Videos", ("*.mov", "*.mp4"))]

        self.save_options = {}

        Frame.__init__(self, master)
        self.create_widgets()
        self.pack()

        self.in_file = None
        self.disable_state()
        self.enable_state()
        master.attributes("-topmost", True)
        master.focus_force()
        self.after(50, lambda: master.attributes("-topmost", False))
        self.enable_spatial_audio = False

def report_callback_exception(self, *args):
    exception = traceback.format_exception(*args)
    tkMessageBox.showerror("Error", exception)

def main():
    root = Tk()
    Tk.report_callback_exception = report_callback_exception
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()

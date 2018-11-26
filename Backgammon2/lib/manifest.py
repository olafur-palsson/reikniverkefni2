#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from lib.utils import load_file_as_json, save_json_to_file, does_file_exist
from lib.json_manipulator import JsonManipulator

class Manifest(JsonManipulator):

    def __init__(self, filename=None, eager=False, obj={}):
        JsonManipulator.__init__(self, obj)
        self.filename = filename
        self.eager = eager

    def load(self, filename = None):
        if filename is not None:
            if does_file_exist(filename):
                self.obj = load_file_as_json(filename)
        elif self.filename is not None:
            if does_file_exist(self.filename):
                self.obj = load_file_as_json(self.filename)
        

    def save(self, filename = None):
        if filename is not None:
            save_json_to_file(filename, self.obj, overwrite=True)
        elif self.filename is not None:
            save_json_to_file(self.filename, self.obj, overwrite=True)
    
    def get(self, path):
        if self.eager:
            self.load(self.filename)
        return super(Manifest, self).get(path)
    
    def set(self, path, value=None):
        if self.eager:
            self.load(self.filename)
        super(Manifest, self).set(path, value)

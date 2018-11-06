#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


class JsonManipulator:
    """
    A utility to handle JSON-like object.  E.g. if you have a 
    JSON object o = {} and we want to set o["3"][5] = True, then
    we have to check if object o has property "3", if not it needs to
    create that property with the correct type (array, object, or some
    primitve), all the way down.  In the case of o["3"][5] we don't know
    whether [5] is accessing an object or an array.

    The syntax here is,

        a{}b{}c[]+

    Which would mean 

        obj["a"]["b"]["c"].append(...)
    
    """

    def __init__(self, obj = {}):
        self.obj = obj

    def get_obj(self):
        return self.obj

    def set_obj(self, obj):
        self.obj = obj

    def parse(self, path):
        in_object_access = False
        in_array_access = False
        buffer = ""
        parts = []
        for i in range(len(path)):
            symbol = path[i]

            if in_object_access:
                if symbol == "}":
                    in_object_access = False
                    buffer = buffer.strip()
                    assert len(buffer) == 0
                    parts += ["{}"]
                else:
                    buffer += symbol
            elif in_array_access:
                if symbol == "]":
                    in_array_access = False
                    buffer = buffer.strip()
                    assert len(buffer) == 0
                    parts += ["[]"]
                else:
                    buffer += symbol
            else:
                if symbol == "{":
                    in_object_access = True
                    buffer = buffer.strip()
                    if len(buffer) > 0:
                        parts += [buffer]
                        buffer = ""
                elif symbol == "[":
                    in_array_access = True
                    buffer = buffer.strip()
                    if len(buffer) > 0:
                        parts += [buffer]
                        buffer = ""
                else:
                    buffer += symbol
        
        buffer = buffer.strip()
        if len(buffer) > 0:
            parts += [buffer]
            buffer = ""
        
        for i in range(1, len(parts), 2):
            assert parts[i] == "{}" or parts[i] == "[]"

        return parts
        
    
    def get(self, path):
        # MODIFIED FROM set
        parts = self.parse(path)
        node = self.obj
        n = len(parts)
        # Access part
        for i in range(1, len(parts), 2):
            n_name = parts[i - 1]
            n_type = parts[i]
            if isinstance(node, dict):
                if n_type == "{}":
                    if n_name not in node:
                        node[n_name] = {}
                    node = node[n_name] 
                elif n_type == "[]":
                    if n_name not in node:
                        node[n_name] = []
                    node = node[n_name]
                else:
                    raise Exception("Error")
            elif isinstance(node, list):
                idx = None
                if n_name == "-" or n_name == "+":
                    if n_name == "-":
                        if n_type == "{}":
                            node = node[0]
                        elif n_type == "[]":
                            node = node[0]
                        else:
                            raise Exception("Error")
                        raise Exception("Not supported: -")
                    elif n_name == "+":
                        if n_type == "{}":
                            node = node[-1]
                        elif n_type == "[]":
                            node = node[-1]
                        else:
                            raise Exception("Error")
                    else:
                        raise Exception("Error")
                else:
                    idx = int(n_name)
                    if n_type == "{}":
                        node = node[idx]
                    elif n_type == "[]":
                        node = node[idx]
                    else:
                        raise Exception("Error")
            else:
                raise Exception("Error!")
        if n % 2 == 1:
            part = parts[-1]
            n_name = part
            if isinstance(node, dict):
                return node[n_name]
            elif isinstance(node, list):
                idx = None
                if n_name == "-" or n_name == "+":
                    if n_name == "-":
                        if n_type == "{}":
                            return node[0]
                        elif n_type == "[]":
                            return node[0]
                        else:
                            raise Exception("Error")
                    elif n_name == "+":
                        if n_type == "{}":
                            return node[-1]
                        elif n_type == "[]":
                            return node[-1]
                        else:
                            raise Exception("Error")
                    else:
                        raise Exception("Error")
                else:
                    idx = int(n_name)
                    if n_type == "{}":
                        return node[idx]
                    elif n_type == "[]":
                        return node[idx]
                    else:
                        raise Exception("Error")
        else:
            # Just tryin'...
            return node
    
    def set(self, path, value=None):
        parts = self.parse(path)
        node = self.obj
        n = len(parts)
        if n % 2 == 0 and value is not None:
            raise Exception("Value cannot be inserted: " + path)
        # Access part
        for i in range(1, len(parts), 2):
            n_name = parts[i - 1]
            n_type = parts[i]
            if isinstance(node, dict):
                if n_type == "{}":
                    if n_name not in node:
                        node[n_name] = {}
                    node = node[n_name] 
                elif n_type == "[]":
                    if n_name not in node:
                        node[n_name] = []
                    node = node[n_name]
                else:
                    raise Exception("Error")
            elif isinstance(node, list):
                idx = None
                if n_name == "-" or n_name == "+":
                    if n_name == "-":
                        if n_type == "{}":
                            node.insert(0, {})
                            node = node[0]
                        elif n_type == "[]":
                            node.insert(0, [])
                            node = node[0]
                        else:
                            raise Exception("Error")
                        raise Exception("Not supported: -")
                    elif n_name == "+":
                        if n_type == "{}":
                            node.append({})
                            node = node[-1]
                        elif n_type == "[]":
                            node.append([])
                            node = node[-1]
                        else:
                            raise Exception("Error")
                    else:
                        raise Exception("Error")
                else:
                    idx = int(n_name)
                    if n_type == "{}":
                        node = node[idx]
                    elif n_type == "[]":
                        node = node[idx]
                    else:
                        raise Exception("Error")
            else:
                raise Exception("Error!")
        if n % 2 == 1:
            part = parts[-1]
            n_name = part
            if isinstance(node, dict):
                node[n_name] = value
            elif isinstance(node, list):
                idx = None
                if n_name == "-" or n_name == "+":
                    if n_name == "-":
                        if n_type == "{}":
                            node.insert(0, value)
                        elif n_type == "[]":
                            node.insert(0, value)
                        else:
                            raise Exception("Error")
                    elif n_name == "+":
                        if n_type == "{}":
                            node.append(value)
                        elif n_type == "[]":
                            node.append(value)
                        else:
                            raise Exception("Error")
                    else:
                        raise Exception("Error")
                else:
                    idx = int(n_name)
                    if n_type == "{}":
                        node[idx] = value
                    elif n_type == "[]":
                        node[idx] = value
                    else:
                        raise Exception("Error")
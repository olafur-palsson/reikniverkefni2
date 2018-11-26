#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import json
from pathlib import Path

import os
import os.path
import copy

import hashlib
import random

import zipfile
import time

import json_stable_stringify_python as json_stable


def does_file_exist(filepath):
    return os.path.isfile(filepath)


def load_file_as_strings(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    return lines


def save_strings_to_file(filepath, strings, overwrite=False):
    if not overwrite:
        if does_file_exist:
            print("Warning: " + str(filepath) + " already exists!")
            return
    f = open(filepath, 'w')
    f.writelines(strings)
    f.close()


def rename_file_to_content_addressable(filename, alg="sha1", ignore_extension = False, extension = ""):
    """
    Renames the file `addressable` so that it becomes content adddressable
    by using hashing algorithm `alg`.  I.e. the name of the file will be it's
    hash digest in hexadecimal.

    Args:
        filename: ...
        alg: ..
        ignore_extension: 
        extension: 

    Returns:
        Returns a string of the new filename
    """

    hash_digest = hash_file(filename, alg)

    base = os.path.basename(filename)

    parts = list(os.path.splitext(base))
    parts[0] = hash_digest

    part_dir = os.path.dirname(filename)


    old_filename = filename

    new_filename = None

    if ignore_extension:
        new_filename = os.path.join(part_dir, parts[0])
    else:
        new_filename = os.path.join(part_dir, ''.join(parts))

    new_filename += extension

    os.rename(old_filename, new_filename)

    return new_filename


def load_file_as_string(filepath):
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    return text


def save_string_to_file(filepath, string, overwrite=False):
    if not overwrite:
        if does_file_exist:
            print("Warning: " + str(filepath) + " already exists!")
            return
    f = open(filepath, 'w')
    f.write(string)
    f.close()


def load_file_as_json(filepath):
    json_dump = load_file_as_string(filepath)
    json_object = json.loads(json_dump)
    return json_object


def save_json_to_file(filepath, json_object, overwrite=False):
    """
    
    """
    if not overwrite:
        if does_file_exist:
            print("Warning: " + str(filepath) + " already exists!")
            return
    json_dump = json.dumps(json_object)
    save_string_to_file(filepath, json_dump, overwrite)


def merge_dict(from_dict, to_dict, deep_copy=False):
    """
    Merges dictionary `from_dict` into dictionary `to_dict` such that properties
    from `from_dict` overwrite properties from `to_dict`.

    NOTE: this modifies the dictionary `to_dict`.

    Args:
        from_dict: dictionary
        to_dict: dictionary
        deep_copy: deep copies from `from_dict`.  Default `False`.
    """
    # Apply "Robustness principle".
    if isinstance(from_dict, dict) and isinstance(from_dict, dict):
        # NOTE: do not put the if statement in the for body, as that is
        # much slower.
        if deep_copy:
            for key in from_dict:
                to_dict[key] = copy.deepcopy(from_dict[key])
        else:
            for key in from_dict:
                to_dict[key] = from_dict[key]


def merge_dict_sk(from_dict, to_dict, exclude_none=False, deep_copy=False):
    """
    Merges those properties from dictionary `from_dict` into dictionary `to_dict`
    whos keys exist in both dictionaries.

    NOTE: this modifies the dictionary `to_dict`.
    NOTE: merge dictionary `from_dict` into `to_dict` where they share keys (sk).

    Args:
        from_dict: dictionary
        to_dict: dictionary
        exclude_none: 
        deep_copy: deep copies from `from_dict`.  Default `False`.
    """
    # Apply "Robustness principle".
    if isinstance(from_dict, dict) and isinstance(from_dict, dict):
        # NOTE: do not put the if statement in the for body, as that is
        # much slower.
        if deep_copy:
            for key in from_dict:
                if key in to_dict:
                    v = from_dict[key]
                    if not(exclude_none and v is None):
                        to_dict[key] = copy.deepcopy(v)
                    
        else:
            for key in from_dict:
                if key in to_dict:
                    v = from_dict[key]
                    if not(exclude_none and v is None):
                        to_dict[key] = v


def get_merged_dict(from_dict, to_dict):
    """
    Creates a merged dictionary from dictionaries `from_dict` and `to_dict` 
    such that properties from `from_dict` overwrite properties from `to_dict`.

    Args:
        from_dict: dictionary
        to_dict: dictionary

    Returns:
        A merged dictionary.
    """
    merged_dict = { **to_dict, **from_dict }
    return merged_dict


def hash_string(string, alg="sha1"):
    """
    Hashes string `string` with algorithm `alg` and returns the digest

    Args:
        string: the string to hash
        alg: the hashing algorith, default `"sha1"`.

    Returns:
        The digest (string) of the string.
    """

    if alg == "sha1":
        hash_object = hashlib.sha1(string.encode())
        hex_digest = hash_object.hexdigest()
        return hex_digest
    
    raise Exception("Algorithm not supported: " + str(alg))


def hash_json(obj, alg="sha1"):
    string = json_stable.stringify(obj)
    return hash_string(string, alg)



def archive_files(archive_filename, filenames, cleanup = False):
    """
    Archives files `filenames` into ZIP file `archive_filename`.  If `cleanup`
    is `True` then the original files are removed, and only the archive 
    remains.

    Args:
        archive_filename: name of archive
        filenames: file names of files to archive
        cleanup (bool): whether to remove the files after archiving, keeping only
                        the archive.  Default `False`.
    
    Returns:
        name of archive
    """

    # Before zip-ing check if filenames exist and archive_filename does not
    # exist

    ok = True

    if does_file_exist(archive_filename):
        ok = False
        raise Exception("Archive filename already exists: " + str(archive_filename))
    
    for filename in filenames:
        if not does_file_exist(filename):
            ok = False
            raise Exception("Filename doesn't exist: " + str(filename))

    if ok:
        zip_file = zipfile.ZipFile(archive_filename, 'w')

        for filename in filenames:
            zip_file.write(filename)

        zip_file.close()

        if cleanup:
            for filename in filenames:
                os.remove(filename)

        return archive_filename
    
    return None


def print_json(json_object):
    string = json.dumps(json_object, sort_keys=True, indent=2, separators=(',', ':'))
    print(string)


def unarchive_archive(archive_filename, cleanup = False):
    """
    Unarchives archive `archive_filename` into the same directory that the
    archive is in.  If `cleanup` is `True` then the archive is removed after
    unarchiving.

    Args:
        archive_filename (str): archive
        cleanup (bool): whether to remove archive after unarchiving.  Default `False`.

    Returns:
        Returns filenames.
    """

    ok = True

    if not does_file_exist(archive_filename):
        ok = False
        raise Exception("Archive filename doesn't exists: " + str(archive_filename))

    filenames = None

    if ok:
        dirname = os.path.dirname(archive_filename)
        zip_file = zipfile.ZipFile(archive_filename, 'r')
        filenames = zip_file.namelist()
        zip_file.extractall('.')
        zip_file.close()

        for filename in filenames:
            assert does_file_exist(filename), "Extracted file doesn't exist: " + str(filename)

        if cleanup:
            os.remove(archive_filename)
        return filenames
    
    return None


def hash_file(filename, alg="sha1"):
    """
    Hash file `filename` and with algorithm `alg` and returns the digest.

    Args:
        filename: the file name of the file to hash
        alg: the hashing algorithm, default `"sha1"`.

    Returns:
        The digest (string) of the file.
    """

    hasher = None

    if alg == "sha1":
        hasher = hashlib.sha1()
    else:
        raise Exception("Algorithm not supported: " + str(alg))
    
    with open(filename, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    
    hex_digest = hasher.hexdigest()

    return hex_digest


def timestamp():
    """
    Returns number of milliseconds since January 1, 1970.

    Returns:
        integer
    """
    return int(round(time.time() * 1000))


def get_random_string(n = 10, alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"):
    """
    Returns a random string.

    Args:
        n (int): length of string
        alphabet (str): the alphabet to pick from, default `"0123456789abcdefghijklmnopqrstuvwxyz"`.
    """

    return ''.join(random.choice(alphabet) for _ in range(n))



if __name__ == "__main__":
    pass

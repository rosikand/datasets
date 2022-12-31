"""
File: build_file_list.py
------------------
Simple script which scrapes all of the image raw urls 
from the specified directory and wrirte them to a file. 
"""

import os
from glob import glob
import sys


def get_file_list(dirpath, extension=None):
    """
    Builds a list of files in the specified directory.
    If an extension is specified, only files with that extension
    are included in the list.
    """

    if extension is not None:
        dirpath = dirpath + "/*." + extension
    else:
        dirpath = dirpath + "/*"
    
    sub_set = glob(dirpath)

    new_paths = []

    for elem in sub_set:
        full_raw_path = "https://github.com/rosikand/datasets/blob/main/" + elem + "?raw=true"
        new_paths.append(full_raw_path)
    
    return new_paths


def get_all_valid_subdirs():
    """
    Returns a list of all valid subdirectories in the current directory.
    """
    sub_dirs = glob("*")
    valid_sub_dirs = []
    for elem in sub_dirs:
        if os.path.isdir(elem):
            valid_sub_dirs.append(elem)
    return valid_sub_dirs


def main():
    user_arguments = sys.argv[1:]
    assert len(user_arguments) > 0, "Please specify, at least, a directory path."
    assert type(user_arguments[0]) == str, "Please specify a directory path that is a valid string."
    assert len(user_arguments) < 3, "Please specify a directory path and, optionally, a file extension."

    all_valid_subdirs = get_all_valid_subdirs()
    assert user_arguments[0] in all_valid_subdirs, f"Please specify a valid directory path. Valid paths are: {all_valid_subdirs}"

    if len(user_arguments) == 2:
        assert type(user_arguments[1]) == str, "Please specify a file extension that is a valid string."
        extension = user_arguments[1]
    else:
        extension = None

    file_list = get_file_list(user_arguments[0], extension)


    with open(f"file_lists/{user_arguments[0]}.txt", "w") as f:
        for elem in file_list:
            f.write(elem + "\n")

    
    print("File list successfully written to file at " + f"'file_lists/{user_arguments[0]}.txt'")


if __name__ == "__main__":
    main()

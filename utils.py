import os
from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter


def include_patterns(*patterns):
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns


def copy_code(path):
    path = os.path.join(path, 'src')
    os.mkdir(path)
    py_files_path = os.path.dirname(os.path.realpath(__file__))
    copytree(py_files_path, path, ignore=include_patterns('*.py'))


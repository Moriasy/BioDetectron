import os
import errno
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.exposure import rescale_intensity

from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from biodetectron.datasets import *


def scale_box(box, scale_factor=0.1):
    deltah = (box[2] - box[0]) * scale_factor / 2
    deltaw = (box[3] - box[1]) * scale_factor / 2

    x1 = box[1] - deltaw
    x2 = box[3] - deltaw
    y1 = box[0] - deltah
    y2 = box[2] - deltah

    scaled_box = [y1,x1,y2,x2]

    return scaled_box


def include_patterns(*patterns):
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns


def remove_empty_dirs(output_folder):
    dirs = [x[0] for x in os.walk(output_folder, topdown=False)]
    for dir in dirs:
        try:
            os.rmdir(dir)
        except Exception as e:
            if e.errno == errno.ENOTEMPTY:
                print("Directory: {0} not empty".format(dir))


def copy_code(path):
    path = os.path.join(path, 'src')
    py_files_path = os.path.dirname(os.path.realpath(__file__))
    copytree(py_files_path, path, ignore=include_patterns('*.py', '*.yaml'))
    remove_empty_dirs(path)


def get_mean_std(folder):
    imglist = glob(os.path.join(folder, '*.jpg')) + \
              glob(os.path.join(folder, '*.tif')) + \
              glob(os.path.join(folder, '*.png'))

    mean = []
    means = []

    std = []
    stds = []

    for idx, path in enumerate(imglist):
        image = imread(path)

        if len(image.shape) > 3:
            image = np.max(image, axis=0)
        if len(image.shape) > 2:
            image = image[:,:,0]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        for n in range(image.shape[-1]):
            if idx == 0:
                means.append([])
                stds.append([])

            img = image[:, :, n]

            means[n].append(np.mean(img))
            stds[n].append(np.std(img))

    for n in range(image.shape[-1]):
        mean.append(float(np.round(np.mean(means[n]), 2)))
        std.append(float(np.round(np.mean(stds[n]), 2)))

    return mean, std

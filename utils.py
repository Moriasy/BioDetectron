import os
import errno
import numpy as np

from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter

from pandas import DataFrame
import pycocotools.coco as coco
from pycocotools.coco import COCO


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


def coco2csv(dataDir, dataType, annFile, mask=False):
    coco = COCO(annFile)
    imgIds = coco.getImgIds(catIds=[])

    for n in imgIds:
        annIds = coco.getAnnIds(imgIds=[n], catIds=[])
        anns = coco.loadAnns(annIds)

        img = coco.loadImgs(["{}".format(n)])[0]
        path = img['file_name']

        for ann in anns:
            ann["bbox"] = np.asarray(ann["bbox"]).clip(0)
            ann['bbox'] = np.round(ann["bbox"])
            ann['x1'] = ann["bbox"][0]
            ann['y1'] = ann["bbox"][1]
            ann['x2'] = ann["bbox"][0] + ann["bbox"][2]
            ann['y2'] = ann["bbox"][1] + ann["bbox"][3]

        df = DataFrame(anns)

        df = df.drop(['area', 'id', 'image_id', 'iscrowd', 'bbox', 'height', 'width'], axis=1)

        if not mask:
            df = df.drop('segmentation', axis=1)

        df.to_csv(os.path.join(dataDir, dataType, os.path.splitext(path)[0] + '.csv'))
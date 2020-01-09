import os
import errno
import numpy as np
from glob import glob
from skimage.io import imread

from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter

from pandas import DataFrame
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


def box2csv(boxes, labels, scores, path):
    df = {'category_id': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'score': []}

    for n in range(len(labels)):
        df['x1'].append(int(boxes[n][0]))
        df['y1'].append(int(boxes[n][1]))
        df['x2'].append(int(boxes[n][2]))
        df['y2'].append(int(boxes[n][3]))

        df['category_id'].append(labels[n])
        df['score'].append(scores[n])

    df = DataFrame(df)
    df.to_csv(path)
import os
import errno
import ffmpeg
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.exposure import rescale_intensity

from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from pandas import DataFrame
from pycocotools.coco import COCO

import datasets
from data import get_csv


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


def vidwrite(fn, images, framerate=2, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=1, s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def csv2video(file_out, path_in, dataset=None, suffix='_predict', do_mapping=False):
    dict_list = get_csv(path_in, dataset, suffix=suffix, do_mapping=do_mapping)
    metadata = MetadataCatalog.get(dataset)

    imgstack = []
    for dic in dict_list:
        image = imread(dic["file_name"])
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        if image.shape[0] < image.shape[-1]:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        image = rescale_intensity(image, out_range=(0, 255))
        image = image.astype(np.uint8)

        viz = Visualizer(image, metadata)
        viz = viz.draw_dataset_dict(dic)

        imgstack.append(viz.get_image())

    print(len(imgstack))

    vidwrite(file_out, imgstack, framerate=2)

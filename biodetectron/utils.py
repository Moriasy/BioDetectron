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

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from pandas import DataFrame
from pycocotools.coco import COCO

from biodetectron.datasets import *
from biodetectron.data import get_csv


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


def box2csv(path, boxes, labels=None, scores=None):
    df = {'x1': [], 'y1': [], 'x2': [], 'y2': []}

    if scores is not None:
        df['score'] = []

    if labels is not None:
        df['category_id'] = []

    for n in range(len(labels)):
        df['x1'].append(int(boxes[n][0]))
        df['y1'].append(int(boxes[n][1]))
        df['x2'].append(int(boxes[n][2]))
        df['y2'].append(int(boxes[n][3]))
        
        if labels is not None:
            df['category_id'].append(labels[n])
        
        if scores is not None: 
            df['score'].append(scores[n])

    df = DataFrame(df)
    df.to_csv(path)


class ColorVisualizer(Visualizer):
    def draw_dataset_dict(self, dic, colors=None):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]

            if colors is not None:
                try:
                    assigned_colors = [colors[x] for x in labels]
                except:
                    raise IndexError("Number of colors set in metadata less than predicted classes!")
            else:
                assigned_colors = None

            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=assigned_colors)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = imread(dic["sem_seg_file_name"])
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output


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

    try:
        colors = metadata.thing_classes_color
    except:
        colors = None

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

        viz = ColorVisualizer(image, metadata)
        viz = viz.draw_dataset_dict(dic, colors=colors)

        imgstack.append(viz.get_image())

    print(len(imgstack))

    vidwrite(file_out, imgstack, framerate=2)

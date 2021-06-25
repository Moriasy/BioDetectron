import os
import copy
import math
import torch
import json
import numpy as np
import pandas as pd
from glob import glob
from pycocotools import mask as cocomask

from skimage.io import imread
from skimage.measure import regionprops
from skimage.measure import label as label_binary
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from scipy.ndimage import gaussian_filter

from tifffile import memmap as quickread

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetMapper, MetadataCatalog, detection_utils as utils

from biodetectron.datasets import get_custom_augmenters
from biodetectron.utils import scale_box

def get_multi_masks(root_dir):
    imglist = glob(os.path.join(root_dir, '*.jpg')) + \
              glob(os.path.join(root_dir, '*.tif')) + \
              glob(os.path.join(root_dir, '*.png'))

    imglist = [path for path in imglist if 'instance' not in path and 'mask' not in path]
    imglist.sort()

    dataset_dicts = []
    for idx, filename in enumerate(imglist):
        record = {}
        objs = []

        ### THIS IS INEFFICIENT
        shapecheck = quickread(filename)

        if len(shapecheck.shape) == 2:
            height, width = quickread(filename).shape
        elif len(shapecheck.shape) == 3:
            height, width = quickread(filename).shape[0:2]
        elif len(shapecheck.shape) == 4:
            height, width = quickread(filename).shape[1:3]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        name = filename.replace('.tif', '_mask.tif')

        mask = imread(name).astype(np.uint16)
        with open(name.replace('_mask.tif', '_detections.json'), 'r') as file:
            points = json.load(file)

        fullmask = np.zeros((mask.shape[0], mask.shape[1], 7), dtype=np.uint16)
        fullmask[:,:,0] = mask

        for thing in points['things']:
            if thing['class'] == 1:
                val = np.max(fullmask[:,:,1]) + 1
                
                points = thing['points']

                idx = mask[int(points[0][0]), int(points[0][1])]
                fullmask[:,:,1][mask == idx] = val
                fullmask[:,:,5][mask == idx] = val

                idx = mask[int(points[1][0]), int(points[1][1])]
                fullmask[:,:,1][mask == idx] = val
                fullmask[:,:,5][mask == idx] = val

                idx = mask[int(points[2][0]), int(points[2][1])]
                fullmask[:,:,2][mask == idx] = val
                fullmask[:,:,5][mask == idx] = val

            elif thing['class'] == 2 or thing['class'] == 4:
                val = np.max(fullmask[:,:,3]) + 1
                
                points = thing['points']

                idx = mask[int(points[0][0]), int(points[0][1])]
                fullmask[:,:,3][mask == idx] = val
                fullmask[:,:,6][mask == idx] = val

                idx = mask[int(points[1][0]), int(points[1][1])]
                fullmask[:,:,4][mask == idx] = val
                fullmask[:,:,6][mask == idx] = val

            elif thing['class'] == 3:
                val = np.max(fullmask[:,:,1]) + 1
                
                points = thing['points']

                idx = mask[int(points[0][0]), int(points[0][1])]
                fullmask[:,:,1][mask == idx] = val
                fullmask[:,:,5][mask == idx] = val

                idx = mask[int(points[1][0]), int(points[1][1])]
                fullmask[:,:,1][mask == idx] = val
                fullmask[:,:,5][mask == idx] = val

        for n in range(fullmask.shape[2]):
            if n == 1 or n == 2 or n == 3 or n == 4:
                continue

            if n == 0:
                category_id = 0
            elif n == 5 or n == 7:
                category_id = 1
            elif n == 6:
                category_id = 2

            boxes = regionprops(fullmask[:,:,n])
            for rp in boxes:
                box = rp.bbox
                box = [box[1], box[0], box[3], box[2]]

                obj = {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": category_id,
                    "iscrowd": 0
                }
                objs.append(obj)
                    
        record["annotations"] = objs
        record["sem_seg"] = fullmask
        dataset_dicts.append(record)

    return dataset_dicts

class MaskDetectionLoader(DatasetMapper):
    def __init__(self, cfg, is_train=True, mask_format="bitmask"):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg
        self.mask_format=mask_format

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Read image and reshape it to always be [h, w, 3].
        image = imread(dataset_dict["file_name"])

        image = np.squeeze(image)

        ### NOT GENERALIZED YET!
        if len(image.shape) > 3:
            #image = np.max(image, axis=0)
            image = image[image.shape[0]//2 + np.random.randint(-2,2)]
            #image = image[np.random.randint(0, image.shape[0]-1)]

        if len(image.shape) > 2:
            if image.shape[0] < image.shape[-1]:
                image = np.transpose(image, (1, 2, 0))

            image = image[:,:,0]

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]  # h, w

        #image = equalize_adapthist(image)

        if not self.is_train:
            dataset_dict['gt_image'] = image

        segmap = SegmentationMapsOnImage(dataset_dict['sem_seg'].astype(np.uint16), shape=image.shape)

        # Define augmentations.
        seq = get_custom_augmenters(
            self.cfg.DATASETS.TRAIN,
            self.cfg.INPUT.MAX_SIZE_TRAIN,
            self.is_train,
            image_shape
        )

        if self.is_train:
            image, segmap = seq(image=image, segmentation_maps=segmap)
        #else:
        #    image, _, _ = seq(image=image, bounding_boxes=boxes, segmentation_maps=segmap)

        image = image.astype(np.float32)
        if 0 in self.cfg.MODEL.PIXEL_MEAN and 1 in self.cfg.MODEL.PIXEL_STD:
            image = rescale_intensity(image)
            # image = image - np.mean(image)
            # image = image / np.std(image)
            
        # Convert image to tensor for pytorch model.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        # Convert masks back to detectron2 annotation format.
        segmap = segmap.get_arr()

        # Convert boxes back to detectron2 annotation format.
        annos = []

        for n in range(segmap.shape[2]):
            if n == 1 or n == 2 or n == 3 or n == 4:
                    continue

            boxes = regionprops(segmap[:,:,n])
            for rp in boxes:
                singlemask = np.zeros((segmap.shape[0], segmap.shape[1]), dtype=np.uint8)

                if n == 0:
                    singlemask[segmap[:,:,n] == rp.label] = 1
                    category_id = 0
                elif n == 5:
                    singlemask[int(box[0]):int(box[2]),int(box[1]):int(box[3])][segmap[:,:,n][int(box[0]):int(box[2]),int(box[1]):int(box[3])] > 0] = 1
                    singlemask[segmap[:,:,1] == rp.label] = 2
                    singlemask[segmap[:,:,2] == rp.label] = 3
                    category_id = 1
                elif n == 6:
                    singlemask[int(box[0]):int(box[2]),int(box[1]):int(box[3])][segmap[:,:,n][int(box[0]):int(box[2]),int(box[1]):int(box[3])] > 0] = 1
                    singlemask[segmap[:,:,3] == rp.label] = 4
                    singlemask[segmap[:,:,4] == rp.label] = 5
                    category_id = 2
                elif n == 7:
                    singlemask[int(box[0]):int(box[2]),int(box[1]):int(box[3])][segmap[:,:,n][int(box[0]):int(box[2]),int(box[1]):int(box[3])] > 0] = 1
                    singlemask[segmap[:,:,1] == rp.label] = 1
                    category_id = 3

                box = rp.bbox
                #box = scale_box(box, 0.2)

                obj = {
                    "bbox": [box[1], box[0], box[3], box[2]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": singlemask,
                    "category_id":  category_id,
                    "iscrowd": 0
                }
                annos.append(obj)

        # for n in range(segmap.shape[2]):
        #     boxes = regionprops(segmap[:,:,n])
        #     for rp in boxes:
        #         singlemask = np.zeros_like(segmap[:,:,n], dtype=np.uint8)
        #         singlemask[segmap[:,:,n] == rp.label] = 1 

        #         box = rp.bbox

        #         obj = {
        #             "bbox": [box[1], box[0], box[3], box[2]],
        #             "bbox_mode": BoxMode.XYXY_ABS,
        #             "segmentation": cocomask.encode(np.asfortranarray(singlemask)),
        #             "category_id":  n,
        #             "iscrowd": 0
        #         }
        #         annos.append(obj)


        # Convert bounding box annotations to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.mask_format
        )
        
        dataset_dict["instances"] = utils.filter_empty_instances(instances, by_mask=False)

        return dataset_dict


import os
import copy
import torch
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from skimage.exposure import rescale_intensity

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetMapper, MetadataCatalog, detection_utils as utils

from datasets import get_custom_augmenters


def get_csv(root_dir, dataset):
    imglist = glob(os.path.join(root_dir, '*.jpg')) + \
                    glob(os.path.join(root_dir, '*.tif')) + \
                    glob(os.path.join(root_dir, '*.png'))

    dataset_dicts = []
    for idx, filename in enumerate(imglist):
        record = {}

        ### THIS IS UNEFFICIENT
        height, width = imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        targets = pd.read_csv(imglist[idx].replace('jpg', 'csv').replace('tif', 'csv').replace('png', 'csv'))

        try:
            mapping = MetadataCatalog.get(dataset).thing_dataset_id_to_contiguous_id
        except:
            mapping = None

        objs = []
        for row in targets.itertuples():
            category_id = mapping[row.category_id] if mapping is not None else row.category_id

            obj = {
                "bbox": [row.x1, row.y1, row.x2, row.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id":  category_id,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


class BoxDetectionLoader(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.cfg = cfg

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
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        if image.shape[0] < image.shape[-1]:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]  # h, w

        ### CUSTOM NORMALIZATION ?
        ### CHECK DATATYPE HERE !

        # Convert bounding boxes to imgaug format for augmentation.
        boxes = []
        for anno in dataset_dict["annotations"]:
            # if iscrowd == 0 ?
            boxes.append(BoundingBox(
                x1=anno["bbox"][0], x2=anno["bbox"][2],
                y1=anno["bbox"][1], y2=anno["bbox"][3], label=anno["category_id"]))

        boxes = BoundingBoxesOnImage(boxes, shape=image_shape)

        # Define augmentations.
        seq = get_custom_augmenters(
            self.cfg.DATASETS.TRAIN,
            self.cfg.INPUT.MAX_SIZE_TRAIN,
            self.is_train,
            image_shape
        )

        image, boxes = seq(image=image, bounding_boxes=boxes)

        # Convert image to tensor for pytorch model.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        # Convert boxes back to detectron2 annotation format.
        annos = []
        for box in boxes:
            obj = {
                "bbox": [box.x1, box.y1, box.x2, box.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id":  box.label,
                "iscrowd": 0
            }
            annos.append(obj)

        # Convert bounding box annotations to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.mask_format
        )

        # Create a tight bounding box from masks, useful when image is cropped
        if self.crop_gen and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances, by_mask=False)

        return dataset_dict




import os
import cv2
import copy
import torch
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from skimage.exposure import rescale_intensity

from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import DatasetEvaluator


def get_csv(root_dir):
    imglist = glob(os.path.join(root_dir, '*.jpg')) + \
                    glob(os.path.join(root_dir, '*.tif')) + \
                    glob(os.path.join(root_dir, '*.png'))

    dataset_dicts = []
    for idx, filename in enumerate(imglist):
        record = {}

        ### THIS IS UNEFFICIENT
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        targets = pd.read_csv(imglist[idx].replace('jpg', 'csv').replace('tif', 'csv').replace('png', 'csv'))

        objs = []
        for row in targets.itertuples():
            obj = {
                "bbox": [row.x1, row.y1, row.x2, row.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id": row.category_id - 1,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def skimread(path, format=None):
    img = imread(path)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    # REMOVE WHEN PIL DEPENDENCIES ARE REMOVED
    img = np.repeat(img, 3, axis=-1)

    return img


class SKImageLoader(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)

        #utils.read_image = skimread

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        dataset_dict['ori_image'] = image

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        #dataset_dict["image"] = torch.as_tensor(image)

        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        # if self.load_proposals:
        #     utils.transform_proposals(
        #         dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
        #     )

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0 ### MAYBE HAS TO CHANGE FOR LEONIES DATASET
            ]
            #annos = [obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        # if "sem_seg_file_name" in dataset_dict:
        #     with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
        #         sem_seg_gt = Image.open(f)
        #         sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        #     sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        #     sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        #     dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict




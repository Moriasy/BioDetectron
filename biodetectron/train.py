from biodetectron.ops import paste_masks_in_image
from biodetectron.masks import BitMasks
import detectron2

detectron2.layers.paste_masks_in_image = paste_masks_in_image
detectron2.structures.BitMasks = BitMasks

import os
import numpy as np
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import default_argument_parser, DefaultTrainer, launch, default_setup
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper

from biodetectron.data import MaskDetectionLoader
from biodetectron.datasets import register_custom_datasets
from biodetectron.utils import copy_code, get_mean_std
from biodetectron.eval import GenericEvaluator

from biodetectron.models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluators = [GenericEvaluator(dataset_name, cfg, cfg.OUTPUT_DIR)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.MASK_ON:
            return build_detection_test_loader(cfg, dataset_name, mapper=MaskDetectionLoader(cfg, False))
        else:
            pass

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.MASK_ON:
            return build_detection_train_loader(cfg, mapper=MaskDetectionLoader(cfg, True))
        else:
            pass


def setup(args):
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_MASK_CLASSES = None

    cfg.merge_from_file(args.config_file)

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    if comm.get_rank() == 0:
        copy_code(cfg.OUTPUT_DIR)

    if "None" in cfg.MODEL.PIXEL_MEAN or "None" in cfg.MODEL.PIXEL_STD:
        mean, std = get_mean_std(path_dict[cfg.DATASETS.TRAIN[0]])
        cfg.MODEL.PIXEL_MEAN = mean
        cfg.MODEL.PIXEL_STD = std

    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg


def main(args):
    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load()

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main,
           num_gpus_per_machine=args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args, ),
           )
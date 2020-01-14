import os
import numpy as np
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import GeneralizedRCNN
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import default_argument_parser, DefaultTrainer, launch, default_setup
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper

from datasets import register_custom_datasets
from utils import copy_code, get_mean_std
from data import BoxDetectionLoader
from eval import GenericEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluators = [GenericEvaluator(dataset_name, cfg, cfg.OUTPUT_DIR)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=BoxDetectionLoader(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=BoxDetectionLoader(cfg, True))

    @classmethod
    def build_model(cls, cfg):
        return VisRCNN(cfg)


class VisRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.max_vis_props = cfg.MAX_VIS_PROPS

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), self.max_vis_props)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)


def setup(args):
    cfg = get_cfg()

    ### Own cfg values
    cfg.MAX_VIS_PROPS = 200

    cfg.merge_from_file(args.config_file)

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    path_dict = register_custom_datasets()

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

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res

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
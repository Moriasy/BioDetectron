import os
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluators
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


def setup(args):
    cfg = get_cfg()
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
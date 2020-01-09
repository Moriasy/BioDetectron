from imgaug import augmenters as iaa
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_custom_augmenters(name, max_size, is_train, image_shape):
    if image_shape[0] > image_shape[1]:
        resize = iaa.Resize({"height": max_size, "width":"keep-aspect-ratio"})
    else:
        resize = iaa.Resize({"height": "keep-aspect-ratio", "width": max_size})

    ####### OSMAN DATA
    if name == "osman":
        if is_train:
            seq = iaa.Sequential([
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1),
                iaa.Sometimes(1, iaa.Rot90(k=(0, 3))),
            ])

        else:
            seq = iaa.Sequential([
                resize,
            ])


    ####### WEN DATA
    elif name == "wen":
        if is_train:
            seq = iaa.Sequential([
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1),
                iaa.Sometimes(1, iaa.Rot90(k=(0, 3))),
            ])

        else:
            seq = iaa.Sequential([
                resize,
            ])


    ####### WING DATA
    elif name == "wings":
        if is_train:
            seq = iaa.Sequential([
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1),
                iaa.Sometimes(0.25, iaa.Rot90(k=(0, 3))),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.3, 2))),
                iaa.Sometimes(0.33, iaa.GaussianBlur(sigma=(0.25, 1)))
            ])

        else:
            seq = iaa.Sequential([
                resize,
            ])

    else:
        if is_train:
            seq = iaa.Sequential([
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1),
                iaa.Sometimes(1, iaa.Rot90(k=(0, 3))),
            ])

        else:
            seq = iaa.Sequential([
                resize,
            ])

    return seq


class DictGetter:
    def __init__(self, dataset, train_path=None, val_path=None):
        self.dataset = dataset
        self.train_path = train_path
        self.val_path = val_path

    def get_train_dicts(self):
        from data import get_csv
        if self.train_path:
            return get_csv(self.train_path, self.dataset)
        else:
            raise ValueError("Training data path is not set!")

    def get_val_dicts(self):
        from data import get_csv
        if self.val_path:
            return get_csv(self.val_path, self.dataset)
        else:
            raise ValueError("Validation data path is not set!")


def register_custom_datasets():
    path_dict = {}

    ####### OSMAN DATA
    dict_getter = DictGetter("osman", train_path='/scratch/bunk/osman/mating_cells/COCO/DIR/train',
                             val_path='/scratch/bunk/osman/mating_cells/COCO/DIR/val')

    path_dict["osman"] = dict_getter.train_path

    DatasetCatalog.register("osman", dict_getter.get_train_dicts)
    MetadataCatalog.get("osman").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3}

    DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2,}

    ####### WEN DATA
    dict_getter = DictGetter("wen", train_path='/scratch/bunk/wen/COCO/DIR/train2014',
                             val_path='/scratch/bunk/wen/COCO/DIR/val2014')

    path_dict["wen"] = dict_getter.train_path

    DatasetCatalog.register("wen", dict_getter.get_train_dicts)
    MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
    MetadataCatalog.get("wen").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

    DatasetCatalog.register("wen_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
    MetadataCatalog.get("wen_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

    ####### WING DATA
    dict_getter = DictGetter("wings", train_path='/scratch/bunk/wings/images/COCO/DIR/train2014',
                             val_path='/scratch/bunk/wings/images/COCO/DIR/val2014')

    path_dict["wings"] = dict_getter.train_path

    DatasetCatalog.register("wings", dict_getter.get_train_dicts)
    MetadataCatalog.get("wings").thing_classes = ["wing"]
    MetadataCatalog.get("wings").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:0}

    DatasetCatalog.register("wings_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("wings_val").thing_classes = ["wing"]
    MetadataCatalog.get("wings_val").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:0}

    return path_dict

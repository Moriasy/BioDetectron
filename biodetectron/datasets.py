from imgaug import augmenters as iaa
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_custom_augmenters(name, max_size, is_train, image_shape):
    if image_shape[0] > image_shape[1]:
        resize = iaa.Resize({"height": max_size, "width":"keep-aspect-ratio"})
        crop = iaa.CropToFixedSize(width=None, height=image_shape[1])
    else:
        resize = iaa.Resize({"height": "keep-aspect-ratio", "width": max_size})
        crop = iaa.CropToFixedSize(width=image_shape[1], height=None)

    ######## OSMAN DATA
    if "osman" in name:
        if is_train:
            seq = iaa.Sequential([
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.1),
                iaa.Rot90(k=(0, 3)),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
            ])

        else:
            seq = iaa.Sequential([
                resize,
            ])

    ######## ISRAEL YEAST DATA
    elif "israel" in name:
        if is_train:
            seq = iaa.Sequential([

            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## WEN DATA
    elif "wen" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(width=max_size, height=max_size),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.Affine(
                    rotate=(-45, 45),
                    backend='skimage'
                ),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.75, 1.25))),
            ])

        else:
            seq = iaa.Sequential([

            ])


    ######## WING DATA
    elif "wings" in name:
        if is_train:
            seq = iaa.Sequential([
                crop,
                resize,
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.Affine(
                    rotate=(-45, 45),
                    backend='skimage'
                ),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.75, 1.25))),
                iaa.Sometimes(0.33, iaa.GaussianBlur(sigma=(0.25, 1)))
            ])
        else:
            seq = iaa.Sequential([
                resize,
            ])


    ######## ELSE
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
    def __init__(self, dataset, train_path=None, val_path=None, mask=False):
        self.dataset = dataset
        self.train_path = train_path
        self.val_path = val_path
        self.mask = mask

    def get_train_dicts(self):
        from data import get_csv, get_masks
        if self.train_path:
            if self.mask:
                return get_masks(self.train_path)
            else:
                return get_csv(self.train_path, self.dataset)
        else:
            raise ValueError("Training data path is not set!")

    def get_val_dicts(self):
        from data import get_csv, get_masks
        if self.val_path:
            if self.mask:
                return get_masks(self.val_path)
            else:
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
    MetadataCatalog.get("osman").thing_classes = ["mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:1, 4:2}

    DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman_val").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:1, 4:2}

    ######## ISRAEL YEAST DATA
    dict_getter = DictGetter("israel", train_path='/scratch/bunk/osman/israel/Rootfolder/Added_Slices/train',
                             val_path='/scratch/bunk/osman/israel/Rootfolder/Added_Slices/val', mask=True)

    path_dict["israel"] = dict_getter.train_path

    DatasetCatalog.register("israel", dict_getter.get_train_dicts)
    MetadataCatalog.get("israel").thing_classes = ["motherATP", "motherKate", "daughter"]
    MetadataCatalog.get("israel").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

    DatasetCatalog.register("israel_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("israel_val").thing_classes = ["motherATP", "motherKate", "daughter"]
    MetadataCatalog.get("israel_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

    ####### WEN DATA
    dict_getter = DictGetter("wen", train_path='/scratch/bunk/wen/COCO/DIR/train2014',
                             val_path='/scratch/bunk/wen/COCO/DIR/val2014')

    path_dict["wen"] = dict_getter.train_path

    DatasetCatalog.register("wen", dict_getter.get_train_dicts)
    MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
    MetadataCatalog.get("wen").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    MetadataCatalog.get("wen").thing_classes_color = [
        (1.000, 0.000, 0.000),
        (0.0666, 0.000, 1.000),
        (0.969, 0.969, 0.075),
        (0.270, 0.714, 0.357),
        (1.000, 1.000, 1.000),
        (1.000, 0.620, 0.043),
        (0.816, 0.800, 0.588),
        (0.965, 0.129, 0.827)
    ]

    DatasetCatalog.register("wen_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
    MetadataCatalog.get("wen_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    MetadataCatalog.get("wen_val").thing_classes_color = [
        (1.000, 0.000, 0.000),
        (0.0666, 0.000, 1.000),
        (0.969, 0.969, 0.075),
        (0.270, 0.714, 0.357),
        (1.000, 1.000, 1.000),
        (1.000, 0.620, 0.043),
        (0.816, 0.800, 0.588),
        (0.965, 0.129, 0.827)
    ]

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


register_custom_datasets()

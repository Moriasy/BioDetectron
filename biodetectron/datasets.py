from imgaug import augmenters as iaa
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_custom_augmenters(name, max_size, is_train):  
    seq = iaa.Sequential([])

    if "yeastmate" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(width=400, height=400),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                # iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                # iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
                # iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=1.0))
            ])

        else:
            seq = iaa. Sequential([

            ])

    return seq


class DictGetter:
    def __init__(self, dataset, train_path=None, val_path=None, mask=False):
        self.dataset = dataset
        self.train_path = train_path
        self.val_path = val_path
        self.mask = mask

    def get_train_dicts(self):
        from data import get_multi_masks
        if self.train_path:
            if self.mask:
                return get_multi_masks(self.train_path)
            else:
                pass
        else:
            raise ValueError("Training data path is not set!")

    def get_val_dicts(self):
        from data import get_csv, get_multi_masks
        if self.val_path:
            if self.mask:
                return get_multi_masks(self.val_path)
            else:
                pass
        else:
            raise ValueError("Validation data path is not set!")


def register_custom_datasets():
    path_dict = {}

    ####### YEASTMATE DATA
    dict_getter = DictGetter("yeastmate", train_path='/scratch/bunk/osman/budding/paper_all_mid/train',
                             val_path='/scratch/bunk/osman/budding/paper_all_mid/val', mask=True)

    path_dict["yeastmate"] = dict_getter.train_path

    DatasetCatalog.register("yeastmate", dict_getter.get_train_dicts)
    MetadataCatalog.get("yeastmate").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}
 
    DatasetCatalog.register("yeastmate_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("yeastmate_val").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

register_custom_datasets()

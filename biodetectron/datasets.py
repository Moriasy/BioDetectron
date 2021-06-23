from imgaug import augmenters as iaa
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_custom_augmenters(name, max_size, is_train, image_shape):  
    seq = iaa.Sequential([])

    if "yeastmate_budding" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(width=max_size, height=max_size),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
            ])

        else:
            seq = iaa. Sequential([

            ])


class DictGetter:
    def __init__(self, dataset, train_path=None, val_path=None, mask=False, dna=False):
        self.dataset = dataset
        self.train_path = train_path
        self.val_path = val_path
        self.mask = mask
        self.dna = dna

    def get_train_dicts(self):
        from data import get_csv, get_multi_masks_new_format, get_dna, get_neuro_txt
        if self.train_path:
            if self.mask:
                return get_multi_masks_new_format(self.train_path)
            elif self.dna:
                return get_dna(self.train_path)
            else:
                return get_neuro_txt(self.train_path, self.dataset)
        else:
            raise ValueError("Training data path is not set!")

    def get_val_dicts(self):
        from data import get_csv, get_multi_masks_new_format, get_dna, get_neuro_txt
        if self.val_path:
            if self.mask:
                return get_multi_masks_new_format(self.val_path)
            elif self.dna:
                return get_dna(self.val_path)
            else:
                return get_neuro_txt(self.val_path, self.dataset)
        else:
            raise ValueError("Validation data path is not set!")


def register_custom_datasets():
    path_dict = {}

    ####### YEASTMATE DATA
    dict_getter = DictGetter("yeastmate_budding", train_path='/scratch/bunk/osman/budding/paper/train',
                             val_path='/scratch/bunk/osman/budding/paper/val', mask=True)

    path_dict["yeastmate_budding"] = dict_getter.train_path

    DatasetCatalog.register("yeastmate_budding", dict_getter.get_train_dicts)
    MetadataCatalog.get("yeastmate_budding").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate_budding").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}
 
    DatasetCatalog.register("yeastmate_budding_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("yeastmate_budding_val").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate_budding_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

register_custom_datasets()

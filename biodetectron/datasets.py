from imgaug import augmenters as iaa
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_custom_augmenters(name, max_size, is_train, image_shape):
    if image_shape[0] > image_shape[1]:
        resize = iaa.Resize({"height": max_size, "width":"keep-aspect-ratio"})
        crop = iaa.CropToFixedSize(width=None, height=image_shape[1])
    else:
        resize = iaa.Resize({"height": "keep-aspect-ratio", "width": max_size})
        crop = iaa.CropToFixedSize(width=image_shape[1], height=None)

    
    seq = iaa.Sequential([])

    ######## enes DATA
    if "enes" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=1.0))
            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## enes DATA
    if "yeastmate_budding" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(width=max_size, height=max_size),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                # iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                # iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
                # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=1.0))
            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## neuro
    if "neuro" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(width=max_size, height=max_size),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                # iaa.Rot90(k=(0, 3)),
                # iaa.Sometimes(0.25, iaa.Affine(
                #     rotate=(-45, 45),
                #     backend='skimage'
                # )),
                #iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                #iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 2))),
                #iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=2.0))
            ])

        else:
            seq = iaa. Sequential([

            ])


    ######## yeastmate mother DATA
    if "osman_mother" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                # iaa.Sometimes(0.25, iaa.Affine(
                #     rotate=(-45, 45),
                #     backend='skimage'
                # )),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 2))),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=2.0))
            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## OSMAN DATA
    # if "osman" in name:
    #     if is_train:
    #         seq = iaa.Sequential([
    #             iaa.CropToFixedSize(height=max_size, width=max_size),
    #             iaa.Fliplr(0.5),
    #             iaa.Flipud(0.5),
    #             iaa.Rot90(k=(0, 3))
    #             # iaa.Affine(
    #             #     rotate=(-45, 45),
    #             #     backend='skimage'
    #             # ),
    #             # iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
    #             # iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
    #         ])
    #     else:
    #         seq = iaa.Sequential([
    #             resize,
    #         ])

    ######## dna DATA
    if "dna" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=1.0))
            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## ISRAEL YEAST DATA
    if "israel" in name:
        if is_train:
            seq = iaa.Sequential([

            ])

        else:
            seq = iaa. Sequential([

            ])

    ######## yeastmate YEAST DATA
    # if "yeastmate" in name:
    #     if is_train:
    #         seq = iaa.Sequential([
    #             iaa.Fliplr(0.5),
    #             iaa.Flipud(0.5),
    #             iaa.Rot90(k=(0, 3)),
    #             iaa.Sometimes(0.25, iaa.Affine(
    #                 rotate=(-45, 45),
    #                 backend='skimage'
    #             )),
    #             iaa.Sometimes(0.33, iaa.GammaContrast(gamma=(0.8, 1.2))),
    #             iaa.Sometimes(0.5, iaa.Multiply(mul=(0.5, 1.5))),
    #             iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=1.0))
    #         ])

    #     else:
    #         seq = iaa. Sequential([

    #         ])

    ######## WEN DATA
    if "wen" in name:
        if is_train:
            seq = iaa.Sequential([
                iaa.CropToFixedSize(height=max_size, width=max_size),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.Affine(
                    rotate=(-45, 45),
                    backend='skimage'
                ),
                iaa.Sometimes(0.5, iaa.GammaContrast(gamma=(0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.Multiply(mul=(0.75, 1.25))),
            ])

        else:
            seq = iaa.Sequential([

            ])


    ######## WING DATA
    if "wings" in name:
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

    return seq


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

    ####### YEASTMATE BUDDING DATA
    dict_getter = DictGetter("yeastmate_budding", train_path='/scratch/bunk/osman/budding/paper/train',
                             val_path='/scratch/bunk/osman/budding/paper/val', mask=True)

    path_dict["yeastmate_budding"] = dict_getter.train_path

    # DatasetCatalog.register("yeastmate_budding", dict_getter.get_train_dicts)
    # MetadataCatalog.get("yeastmate_budding").thing_classes = ["single_cell", "mating_mother", "mating_daughter", "budding_mother", "budding_daughter", "mating", "budding"]
    # MetadataCatalog.get("yeastmate_budding").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
 
    # DatasetCatalog.register("yeastmate_budding_val", dict_getter.get_val_dicts)
    # MetadataCatalog.get("yeastmate_budding_val").thing_classes = ["single_cell", "mating_mother", "mating_daughter", "budding_mother", "budding_daughter", "mating", "budding"]
    # MetadataCatalog.get("yeastmate_budding_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}

    DatasetCatalog.register("yeastmate_budding", dict_getter.get_train_dicts)
    MetadataCatalog.get("yeastmate_budding").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate_budding").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}
 
    DatasetCatalog.register("yeastmate_budding_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("yeastmate_budding_val").thing_classes = ["single_cell", "mating", "budding"]
    MetadataCatalog.get("yeastmate_budding_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

    ####### ENES DATA
    dict_getter = DictGetter("enes", train_path='/scratch/bunk/DAPI_stainings_2daysold/train',
                             val_path='/scratch/bunk/DAPI_stainings_2daysold/val', dna=True)

    path_dict["enes"] = dict_getter.train_path

    DatasetCatalog.register("enes", dict_getter.get_train_dicts)
    MetadataCatalog.get("enes").thing_classes = ["E_DS_DNMT_TKO", "E_DS_DNMT_WT", "E_Ehmt2_KO", "E_Ehmt2_WT",'E_J1_DNMT_TKO','E_J1_DNMT_WT','E_SUV_DKO','E_SUV_WT','N_DS_DNMT_TKO','N_DS_DNMT_WT','N_Ehmt2_KO','N_Ehmt2_WT','N_J1_DNMT_TKO','N_J1_DNMT_WT','N_SUV_DKO','N_SUV_WT']
 
    DatasetCatalog.register("enes_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("enes_val").thing_classes = ["E_DS_DNMT_TKO", "E_DS_DNMT_WT", "E_Ehmt2_KO", "E_Ehmt2_WT",'E_J1_DNMT_TKO','E_J1_DNMT_WT','E_SUV_DKO','E_SUV_WT','N_DS_DNMT_TKO','N_DS_DNMT_WT','N_Ehmt2_KO','N_Ehmt2_WT','N_J1_DNMT_TKO','N_J1_DNMT_WT','N_SUV_DKO','N_SUV_WT']

    ####### NEURO DATA
    dict_getter = DictGetter("neuro", train_path='/scratch/bunk/neuro/pretraining/train',
                             val_path='/scratch/bunk/neuro/pretraining/val')

    path_dict["neuro"] = dict_getter.train_path

    DatasetCatalog.register("neuro", dict_getter.get_train_dicts)
    MetadataCatalog.get("neuro").thing_classes = ["target"]
    MetadataCatalog.get("neuro").thing_dataset_id_to_contiguous_id = {0:0}
 
    DatasetCatalog.register("neuro_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("neuro_val").thing_classes = ["target"]
    MetadataCatalog.get("neuro_val").thing_dataset_id_to_contiguous_id = {0:0}

    ####### OSMAN DATA
    dict_getter = DictGetter("osman", train_path='/scratch/bunk/osman/israelset/train',
                             val_path='/scratch/bunk/osman/israelset/val')

    path_dict["osman"] = dict_getter.train_path

    DatasetCatalog.register("osman", dict_getter.get_train_dicts)
    MetadataCatalog.get("osman").thing_classes = ["mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman").thing_dataset_id_to_contiguous_id = {0:0, 2:1, 3:2}
 
    DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["mating", "single_cell", "crowd"]
    MetadataCatalog.get("osman_val").thing_dataset_id_to_contiguous_id = {0:0, 2:1, 3:2}

    ####### DNA DATA
    dict_getter = DictGetter("dna", train_path='/scratch/bunk/dna_sir_ageing/examples_tiff_50intensity_8bit/train',
                             val_path='/scratch/bunk/dna_sir_ageing/examples_tiff_50intensity_8bit/val', dna=True)

    path_dict["dna"] = dict_getter.train_path

    DatasetCatalog.register("dna", dict_getter.get_train_dicts)
    MetadataCatalog.get("dna").thing_classes = ["untreated_old", "untreated_young", "young3d", "young6d", "young9d"]
    MetadataCatalog.get("dna").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3, 4:4}
 
    DatasetCatalog.register("dna_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("dna_val").thing_classes = ["untreated_old", "untreated_young", "young3d", "young6d", "young9d"]
    MetadataCatalog.get("dna_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3, 4:4}

    ####### MOTHER/DAUGHTER DATA
    dict_getter = DictGetter("osman_mother", train_path='/scratch/bunk/osman/israelset/train',
                             val_path='/scratch/bunk/osman/israelset/val', mask=True)

    path_dict["osman_mother"] = dict_getter.train_path

    DatasetCatalog.register("osman_mother", dict_getter.get_train_dicts)
    MetadataCatalog.get("osman_mother").thing_classes = ["single_cell", "mother", "daughter", "mating"]
    MetadataCatalog.get("osman_mother").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3}
 
    DatasetCatalog.register("osman_mother_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("osman_mother_val").thing_classes = ["single_cell", "mother", "daughter", "mating"]
    MetadataCatalog.get("osman_mother_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2, 3:3}

    ####### LEONIE DATA
    dict_getter = DictGetter("leonie", train_path='/scratch/bunk/osman/mating_cells/COCO/DIR/train',
                             val_path='/scratch/bunk/osman/mating_cells/COCO/DIR/val')

    path_dict["leonie"] = dict_getter.train_path

    DatasetCatalog.register("leonie", dict_getter.get_train_dicts)
    MetadataCatalog.get("leonie").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
    MetadataCatalog.get("leonie").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3}

    DatasetCatalog.register("leonie_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("leonie_val").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
    MetadataCatalog.get("leonie_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3}

    ######## ISRAEL YEAST DATA
    dict_getter = DictGetter("israel", train_path='/scratch/bunk/osman/israel/train',
                             val_path='/scratch/bunk/osman/israel/val', mask=True)

    path_dict["israel"] = dict_getter.train_path

    DatasetCatalog.register("israel", dict_getter.get_train_dicts)
    MetadataCatalog.get("israel").thing_classes = ["motherATP", "motherKate", "daughter"]
    MetadataCatalog.get("israel").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

    DatasetCatalog.register("israel_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("israel_val").thing_classes = ["motherATP", "motherKate", "daughter"]
    MetadataCatalog.get("israel_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1, 2:2}

    ######## YEASTMATE DATA
    dict_getter = DictGetter("yeastmate", train_path='/scratch/bunk/osman/israelset/train',
                             val_path='/scratch/bunk/osman/israelset/val', mask=True)

    path_dict["yeastmate"] = dict_getter.train_path

    DatasetCatalog.register("yeastmate", dict_getter.get_train_dicts)
    MetadataCatalog.get("yeastmate").thing_classes = ["singlecell", "mating"]
    MetadataCatalog.get("yeastmate").thing_dataset_id_to_contiguous_id = {0:0, 1:1}

    DatasetCatalog.register("yeastmate_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("yeastmate_val").thing_classes = ["singlecell", "mating"]
    MetadataCatalog.get("yeastmate_val").thing_dataset_id_to_contiguous_id = {0:0, 1:1}

    ####### WEN DATA
    dict_getter = DictGetter("wen", train_path='/scratch/bunk/wen/COCO/DIR/train2014',
                             val_path='/scratch/bunk/wen/COCO/DIR/val2014')

    path_dict["wen"] = dict_getter.train_path

    DatasetCatalog.register("wen", dict_getter.get_train_dicts)
    MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
    #MetadataCatalog.get("wen").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
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
    #MetadataCatalog.get("wen_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
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

#
# Project DeepHealth, UC5 "Deep Image Annotation"
#
# Franco Alberto Cardillo, ILC-CNR (UNITO) 
# francoalberto.cardillo@ilc.cnr.it
#

import numpy as np
import pyecvl.ecvl as ecvl


chest_iu = "chest-iu"
mimic_cxr = "mimic_cxr"

drop_last = {"training": True, "validation": False, "test": False}
drop_last_rnn_gpu = {"training": True, "validation": True, "test": True}
drop_last_rnn_pred = {"training": True, "validation": True, "test": False}
def get_augmentations(dataset_name, img_size=224):
    if dataset_name == chest_iu:
        return get_augmentations_chest_iu(img_size)
    elif dataset_name == mimic_cxr:
        return get_augmentations_mimic_cxr(img_size)
    else:
        raise f"Unknown dataset name: {dataset_name}"

def get_augmentations_chest_iu(img_size=224):
    mean = [0.48197903, 0.48197903, 0.48197903]
    std = [0.26261734, 0.26261734, 0.26261734]
    
    train_augs = ecvl.SequentialAugmentationContainer([
                    ecvl.AugResizeDim([300, 300]),
                    ecvl.AugRotate([-5,5]),
                    ecvl.AugToFloat32(divisor=255.0),
                    ecvl.AugNormalize(mean, std),
                    ecvl.AugRandomCrop([img_size, img_size])
            ])

    test_augs =  ecvl.SequentialAugmentationContainer([
                        ecvl.AugResizeDim([300, 300]),
                        ecvl.AugToFloat32(divisor=255.0),
                        ecvl.AugNormalize(mean, std),
                        ecvl.AugCenterCrop([img_size, img_size])
                    ])
    return train_augs, test_augs


def get_augmentations_mimic_cxr(img_size=224):
    mean = [ 0.4722, 0.4722, 0.4722 ]
    std = [ 0.3024, 0.3024, 0.3024 ]     
    train_augs = ecvl.SequentialAugmentationContainer([
                    ecvl.AugResizeDim([300, 300]),
                    ecvl.AugRotate([-5,5]),
                    ecvl.AugToFloat32(divisor=255.0),
                    ecvl.AugNormalize(mean, std),
                    ecvl.AugRandomCrop([img_size, img_size])
            ])

    test_augs =  ecvl.SequentialAugmentationContainer([
                        ecvl.AugResizeDim([300, 300]),
                        ecvl.AugToFloat32(divisor=255.0),
                        ecvl.AugNormalize(mean, std),
                        ecvl.AugCenterCrop([img_size, img_size])
                    ])
    return train_augs, test_augs



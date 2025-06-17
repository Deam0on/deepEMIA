from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


def custom_mapper(dataset_dict, augment=False):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    annos = dataset_dict.get("annotations", [])

    if augment:
        aug = T.AugmentationList(
            [
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomRotation(angle=[-20, 20]),
                T.RandomBrightness(0.8, 1.2),
            ]
        )
        aug_input = T.AugInput(image)
        transforms = aug(aug_input)
        image = aug_input.image
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in annos
        ]
    # Convert image to tensor
    dataset_dict["image"] = utils.convert_image_to_tensor(image, "BGR")
    dataset_dict["annotations"] = annos
    return dataset_dict

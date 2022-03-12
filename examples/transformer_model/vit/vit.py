from datasets import load_dataset
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomResizedCrop, ToTensor)
from transformers import AutoFeatureExtractor

if __name__ == '__main__':

    dataset = load_dataset('food101', split='train[:100]')
    dataset[0]['image']

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224')

    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    _transforms = Compose([
        RandomResizedCrop(feature_extractor.size),
        ColorJitter(brightness=0.5, hue=0.5),
        ToTensor(), normalize
    ])

    def transforms(examples):
        examples['pixel_values'] = [
            _transforms(image.convert('RGB')) for image in examples['image']
        ]
        return examples

    dataset.set_transform(transforms)

    dataset[0]['image']

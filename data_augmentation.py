import torch
from torchvision.transforms import v2 # PyTorch image transformations
from PIL import Image # pillow library for opening images

import matplotlib.pyplot as plt

# Without Augmentations
transforms_non_augmented = v2.Compose([
    v2.ToTensor(),
	v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# With Augmentations
transforms_augmented = v2.Compose([
    v2.ToTensor(),

    # flip image
    v2.RandomHorizontalFlip(p=0.5),
	v2.RandomVerticalFlip(p=0.5),

    # rotate image by 90 degrees 1, 2, or 3 times
    v2.RandomChoice([
        v2.Lambda(lambda image: torch.rot90(image, 1, [1, 2])), # 90 degrees 
        v2.Lambda(lambda image: torch.rot90(image, 2, [1, 2])), # 180 degrees
        v2.Lambda(lambda image: torch.rot90(image, 3, [1, 2])), # 270 degrees
    ]),

	v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_image = Image.open("Anemic_RBC_dataset_resized/Anemic/0001_01_a.png")

transformed_test_image = transforms_augmented(test_image)
v2.ToPILImage(transformed_test_image)

# convert from (C, H, W) to (H, W, C) for displaying in matplotlib
img_np = transformed_test_image.permute(1, 2, 0).numpy()

# display augmented image
plt.imshow(img_np)
plt.axis('off')
plt.show()

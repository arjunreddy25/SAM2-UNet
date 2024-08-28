import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio

class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        return {'image': image, 'label': label}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        print("Sample image paths:", self.images[:5])
        print("Sample ground truth paths:", self.gts[:5])

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

        self._check_files()

    def _check_files(self):
        missing_files = []
        for img_path in self.images:
            if not os.path.isfile(img_path):
                missing_files.append(img_path)

        for gt_path in self.gts:
            if not os.path.isfile(gt_path):
                missing_files.append(gt_path)

        if missing_files:
            print("Missing files:", missing_files)

    def __getitem__(self, idx):
        if idx >= len(self.images) or idx >= len(self.gts):
            raise IndexError(f"Index {idx} out of range. Images length: {len(self.images)}, GTs length: {len(self.gts)}")

        image_path = self.images[idx]
        gt_path = self.gts[idx]
        
        # Ensure the base filenames match
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        gt_name = os.path.splitext(os.path.basename(gt_path))[0]
        
        if image_name != gt_name:
            raise ValueError(f"Mismatch between image and ground truth names at index {idx}")

        image = self.rgb_loader(image_path)
        label = self.binary_loader(gt_path)
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        mat_contents = sio.loadmat(path)
        print(f"Keys in .mat file: {mat_contents.keys()}")  # This will print all available keys
    
    # Try to find the correct key for the mask data
        mask_keys = [key for key in mat_contents.keys() if key not in ['__header__', '__version__', '__globals__']]
    
        if not mask_keys:
            raise ValueError(f"No valid mask data found in {path}")
    
        if len(mask_keys) > 1:
            print(f"Multiple possible mask keys found: {mask_keys}. Using the first one.")
    
        mask_key = mask_keys[0]
        mask = mat_contents[mask_key]
    
    # Ensure the mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()  # Remove singleton dimensions
    
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape {mask.shape} in {path}")
    
        return Image.fromarray(mask.astype(np.uint8))

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        print("Sample test image paths:", self.images[:5])
        print("Sample test ground truth paths:", self.gts[:5])

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = os.path.basename(self.images[self.index])

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        mat_contents = sio.loadmat(path)
        # Assuming the mask is stored under the key 'mask' in the .mat file
        # You might need to adjust this key based on your .mat file structure
        mask = mat_contents['mask']
        return Image.fromarray(mask.astype(np.uint8))

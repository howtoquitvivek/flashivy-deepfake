import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.images = []
        self.labels = []

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, cls)
            for img in os.listdir(folder):
                self.images.append(os.path.join(folder, img))
                self.labels.append(label)

        split_idx = int(0.8 * len(self.images))
        if split == "train":
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

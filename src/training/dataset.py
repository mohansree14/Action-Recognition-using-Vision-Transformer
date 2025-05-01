import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HMDBDataset(Dataset):
    def __init__(self, root_dir, clip_size=8, transform=None, frame_rate=32):
        self.root_dir = root_dir
        self.clip_size = clip_size
        self.transform = transform
        self.frame_rate = frame_rate
        self.data = self._load_data()

    def _load_data(self):
        data = []
        subfolders = os.listdir(self.root_dir)
        
        # Ensure there are 25 subfolders
        if len(subfolders) != 25:
            raise ValueError(f"Dataset must contain exactly 25 subfolders. Found {len(subfolders)}.")

        for label, action in enumerate(subfolders):
            action_path = os.path.join(self.root_dir, action)
            if not os.path.isdir(action_path):  # Skip if not a directory
                continue
            for video_folder in os.listdir(action_path):
                video_path = os.path.join(action_path, video_folder)
                if os.path.isdir(video_path):  # Ensure it's a folder containing images
                    all_frames = sorted(os.listdir(video_path))
                    sampled_frames = all_frames[::self.frame_rate]
                    if len(sampled_frames) >= self.clip_size:  # Only include valid folders
                        data.append((video_path, label))
                    else:
                        print(f"Skipping {video_path}: Not enough frames. Expected at least {self.clip_size}, got {len(sampled_frames)}.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self._load_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames), label

    def _load_frames(self, video_path):
        """
        Load frames from a video folder, sampling at a rate of 1/32 to create a clip of size 8.
        """
        all_frames = sorted(os.listdir(video_path))  # Sort to maintain frame order
        sampled_frames = all_frames[::self.frame_rate]  # Sample frames at a rate of 1/32

        # Pad frames if there are not enough
        if len(sampled_frames) < self.clip_size:
            print(f"Padding {video_path}: Not enough frames. Expected at least {self.clip_size}, got {len(sampled_frames)}.")
            while len(sampled_frames) < self.clip_size:
                sampled_frames.append(sampled_frames[-1])  # Repeat the last frame

        # Select the first `clip_size` frames
        selected_frames = sampled_frames[:self.clip_size]

        # Load the selected frames as PIL images
        frames = [Image.open(os.path.join(video_path, frame)) for frame in selected_frames]
        return frames

def get_dataloader(root_dir, batch_size=8, clip_size=8, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = HMDBDataset(root_dir, clip_size=clip_size, transform=transform, frame_rate=32)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
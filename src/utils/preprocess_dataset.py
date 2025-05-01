import os
import shutil
import cv2

def augment_frames(video_path, clip_size, frame_rate):
    """
    Augment frames to ensure the folder has at least `clip_size` frames.
    Args:
        video_path (str): Path to the folder containing video frames.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    """
    all_frames = sorted(os.listdir(video_path))
    sampled_frames = all_frames[::frame_rate]

    if len(sampled_frames) >= clip_size:
        return sampled_frames  # No augmentation needed

    print(f"Augmenting frames for {video_path}: Expected {clip_size}, found {len(sampled_frames)}.")

    # Duplicate frames to reach the required clip size
    while len(sampled_frames) < clip_size:
        sampled_frames.append(sampled_frames[-1])  # Duplicate the last frame

    # Save augmented frames back to the folder
    for i, frame_name in enumerate(sampled_frames):
        src_path = os.path.join(video_path, frame_name)
        dst_path = os.path.join(video_path, f"augmented_frame_{i:04d}.jpg")
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

    return sampled_frames

def preprocess_dataset(root_dir, clip_size=8, frame_rate=32):
    """
    Preprocess the dataset by ensuring each folder has at least `clip_size` frames.
    Args:
        root_dir (str): Path to the dataset root directory.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    """
    subfolders = os.listdir(root_dir)
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for video_folder in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_folder)
            if os.path.isdir(video_path):
                all_frames = sorted(os.listdir(video_path))
                sampled_frames = all_frames[::frame_rate]
                if len(sampled_frames) < clip_size:
                    # Use augmentation instead of removing the folder
                    augment_frames(video_path, clip_size, frame_rate)
                else:
                    print(f"Valid folder: {video_path}")
                
                # Verification: Check the total number of frames after preprocessing
                total_frames = len(os.listdir(video_path))
                print(f"Folder: {video_path}, Total Frames After Preprocessing: {total_frames}")

if __name__ == "__main__":
    # Update the dataset path as needed
    dataset_path = "/content/HMDB_simp"
    preprocess_dataset(dataset_path)
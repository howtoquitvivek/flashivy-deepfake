import cv2
import os
import random

def extract_frames(video_dir, output_dir, frames_per_video=100, max_images=2500):
    os.makedirs(output_dir, exist_ok=True)

    videos = [v for v in os.listdir(video_dir) if v.endswith(".mp4")]
    random.shuffle(videos)

    saved_count = 0

    for video in videos:
        if saved_count >= max_images:
            break

        cap = cv2.VideoCapture(os.path.join(video_dir, video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < frames_per_video:
            cap.release()
            continue

        frame_ids = random.sample(range(total_frames), frames_per_video)
        frame_ids = set(frame_ids)

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx in frame_ids:
                out_path = os.path.join(
                    output_dir,
                    f"{video}_{idx}.jpg"
                )
                cv2.imwrite(out_path, frame)
                saved_count += 1

                if saved_count >= max_images:
                    break

            idx += 1

        cap.release()

    print(f"Saved {saved_count} frames to {output_dir}")

# CHANGE THESE PATHS IF NEEDED
extract_frames(
    "ff_data/original_sequences/youtube/c23/videos",
    "data/raw_frames/real"
)

extract_frames(
    "ff_data/manipulated_sequences/Deepfakes/c23/videos",
    "data/raw_frames/fake"
)

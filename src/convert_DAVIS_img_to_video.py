# Combine DAVIS frames into video
# Combine videos where set: test in eval/DAVIS/Annotations/db_info.yml

import os
import glob
import subprocess
import yaml


def main():
    base_dir = os.path.join('evaluation', 'dataset', 'DAVIS')
    db_info_path = os.path.join(base_dir, 'Annotations', 'db_info.yml')
    output_dir = os.path.join(base_dir, 'eval_video_1080p')
    os.makedirs(output_dir, exist_ok=True)

    with open(db_info_path, 'r') as f:
        info = yaml.safe_load(f)
    sequences = info.get('sequences', [])
    for seq in sequences:
        if seq.get('set') == 'test':
            name = seq.get('name')
            img_dir = os.path.join(base_dir, 'JPEGImages', '1080p', name)
            if not os.path.isdir(img_dir):
                print(f"Skipping {name}, directory not found: {img_dir}")
                continue
            img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
            if not img_files:
                print(f"No images found for {name} in {img_dir}")
                continue

            # Create video using ffmpeg
            fps = 24
            pattern = os.path.join(img_dir, '%05d.jpg')
            out_path = os.path.join(output_dir, f"{name}.mp4")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', pattern,
                '-c', 'copy',
                out_path
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Created video for {name}: {out_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to create video for {name}: {e}")


if __name__ == '__main__':
    main()

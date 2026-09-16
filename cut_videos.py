import os
from moviepy import VideoFileClip

input_dir = "omniflash_edit"
output_dir = "omniflash_edit_cut"
max_duration = 9

os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith(".mp4"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    clip = VideoFileClip(input_path)
    if clip.duration > max_duration:
        cut_clip = clip.subclipped(0, max_duration)
        print(f"{filename}: {clip.duration:.1f}s -> {max_duration}s")
    else:
        cut_clip = clip
        print(f"{filename}: {clip.duration:.1f}s (already <= {max_duration}s, copying as-is)")

    cut_clip.write_videofile(output_path, logger=None)
    clip.close()
    cut_clip.close()

print(f"\nDone. Output in {output_dir}/")

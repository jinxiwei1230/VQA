import os
from moviepy.editor import VideoFileClip

def convert_avi_to_mp4(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有 AVI 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".avi"):
            # 构建输入和输出文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".avi", ".mp4"))

            # 使用 moviepy 进行转换
            clip = VideoFileClip(input_path)
            clip.write_videofile(output_path, codec='libx264')

            print(f"Converted {input_path} to {output_path}")

if __name__ == "__main__":
    input_dir = "/home/disk2/four_avi"  # 替换为你的 AVI 文件夹路径
    output_dir = "/home/disk2/four_mp4"  # 替换为你的 MP4 输出文件夹路径

    convert_avi_to_mp4(input_dir, output_dir)
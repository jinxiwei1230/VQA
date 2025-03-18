import argparse
import json
import subprocess
from logging import exception
import os
import re
import time
import traceback
import sys

import cv2
from hdfs import InsecureClient
from kafka import KafkaConsumer
import sys
from omegaconf import OmegaConf

sys.path.append('/home/disk2/dachuang1-23/mist/')
from main_agqa_v2 import main
# 该代码从Kafka主题中消费消息，通过HDFS下载相应的视频文件到本地，再调用VideoProcessor类进行视频处理。处理完成后，更新处理状态。
sys.path.append('/home/disk2/dachuang1-23/v2s-portal')
sys.path.append('/home/disk2/dachuang1-23')
from zimu import process_audio_from_video

import vs_common
import common_utils
from process import VideoProcessor

# 前端上传 ➔ 后端接收视频文件。
# 后端上传 ➔ 将视频文件存储到 HDFS。
# 后端通知 ➔ 生成包含视频路径和问题信息的消息并发送到 Kafka。
# 消费者执行 ➔ 消费者获取消息并从 HDFS 下载视频进行处理。



def upload_directory_to_hdfs(local_dir, hdfs_dir):
    # 确保HDFS目标文件夹存在
    try:
        hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')
        hdfs_client.makedirs(hdfs_dir)

    except Exception as e:
        print(f"Failed to create HDFS directory {hdfs_dir}: {e}")

    # 视频文件扩展名
    video_extensions = {'.mp4'}

    # 记录上传的视频文件数量
    uploaded_count = 0

    # 遍历本地文件夹中的文件并上传
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            # 获取文件的扩展名
            _, ext = os.path.splitext(file)

            # 只处理.avi视频文件
            if ext.lower() == '.mp4':
                # 上传前1到500个视频文件
                if uploaded_count >= 1471:
                    print("Reached the limit of 500 files, stopping upload.")
                    return

                # 获取本地文件路径
                local_file_path = os.path.join(root, file)

                # 计算目标HDFS路径，直接使用文件名
                hdfs_file_path = os.path.join(hdfs_dir, file)

                # 删除 HDFS 目录下所有已存在的视频文件
                # try:
                #     # 获取目录下的所有文件，删除扩展名为 .avi 的文件
                #     files_in_hdfs = hdfs_client.list(hdfs_dir)  # 列出 HDFS 目录下的文件
                #     for file in files_in_hdfs:
                #         _, ext = os.path.splitext(file)
                #         if ext.lower() in video_extensions:  # 只删除 .avi 文件
                #             hdfs_file_path = os.path.join(hdfs_dir, file)
                #             try:
                #                 hdfs_client.delete(hdfs_file_path)  # 删除文件
                #                 print(f"Deleted existing file {hdfs_file_path} from HDFS.")
                #             except Exception as e:
                #                 print(f"Failed to delete {hdfs_file_path}: {e}")
                # except Exception as e:
                #     print(f"Failed to list files in HDFS directory {hdfs_dir}: {e}")


                # 上传文件到 HDFS
                try:
                    with open(local_file_path, 'rb') as local_file:
                        hdfs_client.write(hdfs_file_path, local_file, overwrite=True)
                    print(f"Uploaded {local_file_path} to {hdfs_file_path}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"Failed to upload {local_file_path} to {hdfs_file_path}: {e}")


    if uploaded_count == 0:
        print("No .avi video files found in the directory.")
    else:
        print(f"Successfully uploaded {uploaded_count} video files to HDFS.")




def download_video_from_hdfs(hdfs_dir, local_dir):
    # 确保本地存储目录存在
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    try:
        # 连接到HDFS
        hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')

        # 获取HDFS目录中的所有文件
        hdfs_files = hdfs_client.list(hdfs_dir)
        video_extensions = {'.avi'}

        # 下载每个视频文件
        for file in hdfs_files:
            # 获取文件扩展名
            _, ext = os.path.splitext(file)

            # 只处理.avi视频文件
            if ext.lower() in video_extensions:
                # 计算目标本地路径
                local_file_path = os.path.join(local_dir, file)

                # 下载文件
                try:
                    hdfs_file_path = os.path.join(hdfs_dir, file)
                    # 修正: 直接传递本地文件路径作为参数
                    hdfs_client.download(hdfs_file_path, local_file_path)
                    print(f"Downloaded {file} from HDFS to {local_file_path}")
                except Exception as e:
                    print(f"Failed to download {file} from HDFS: {e}")

    except Exception as e:
        print(f"Failed to connect to HDFS: {e}")


def delete_avi_files_from_hdfs(hdfs_dir):
    try:
        # 连接到HDFS
        hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')

        # 获取HDFS目录中的所有文件
        hdfs_files = hdfs_client.list(hdfs_dir)
        video_extensions = {'.avi'}

        # 删除每个.avi文件
        for file in hdfs_files:
            # 获取文件扩展名
            _, ext = os.path.splitext(file)

            # 只删除.avi视频文件
            if ext.lower() in video_extensions:
                try:
                    hdfs_file_path = os.path.join(hdfs_dir, file)
                    # 删除文件
                    hdfs_client.delete(hdfs_file_path)
                    print(f"Deleted {file} from HDFS")
                except Exception as e:
                    print(f"Failed to delete {file} from HDFS: {e}")

    except Exception as e:
        print(f"Failed to connect to HDFS: {e}")


if __name__ == '__main__':

    try:
        # 下载指定HDFS路径的视频文件到本地服务器
        # download_video_from_hdfs('/bdap/students/public/four_mp4', '/home/disk2/four_mp4')

        #上传
        # upload_directory_to_hdfs('/home/disk2/four_mp4', '/bdap/students/public/four_mp4')

        #删除avi
        delete_avi_files_from_hdfs('/bdap/students/2023110788/videos')
    # 处理过程中发生异常，捕获异常并打印出错误信息，包括导致失败的消息内容
    except Exception as e:
        print(f"Exception occurred: {e}")

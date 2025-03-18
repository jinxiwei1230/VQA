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
sys.path.append('/home/disk2/dachuang1-23/HMMC')
sys.path.append('/home/disk2/dachuang1-23/HMMC/v3s-portal')
from omegaconf import OmegaConf

import vs_common
import common_utils
from process import VideoProcessor

from main_task_retrieval import main

bootstrapServers = ['10.92.64.241:9092']
topic = 'video-text'
groupId = 'vqa3'
video_store_dir = vs_common.local_video_store_dir


class GateConsumer:
    def __init__(self):
        print('-' * 20)
        print('gate消费者启动')
        print('-' * 20)

        # 初始化属性
        self.kafkaTopic = topic  # 存储Kafka主题的名称，消费者将从这个主题中读取消息
        self.bootstrapServers = bootstrapServers  # Kafka服务器的地址，用于建立连接
        self.groupId = groupId  # 消费者组的ID，Kafka允许消费者以组的形式进行消费，以便在多个消费者之间共享工作
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')  # 创建一个与 HDFS 的连接客户端，后续可能会用来上传处理结果或视频数据

        # 创建 KafkaConsumer
        self.consumer = KafkaConsumer(
            self.kafkaTopic,  # 指定要消费的Kafka主题
            group_id=self.groupId,  # 指定消费者组，允许多个消费者共同消费消息
            bootstrap_servers=self.bootstrapServers,  # Kafka服务器地址
            # 一个消息最多消费50分钟，超出则会 rebalance 重新消费
            max_poll_interval_ms=3000000,  # 最大轮询间隔，允许消费者在处理消息时的最长等待时间
            connections_max_idle_ms=3000000000,  # 最大空闲时间，如果没有活动，连接会关闭
            request_timeout_ms=3050000,  # 请求超时时间，如果请求在此时间内没有得到响应，将引发超时异常
            session_timeout_ms=100000,  # 会话超时，决定消费者在多长时间内没有心跳信号将被认为失效
            heartbeat_interval_ms=30000,  # 心跳间隔，用于确保消费者在活动状态
        )
        # 创建一个 VideoProcessor 的实例，后续将用来处理消费者从Kafka获取到的视频数据
        self.processor = VideoProcessor()

        print('-' * 20)
        print('gate消费者启动完成！！')
        print('-' * 20)

    def consume_data(self):  # 用于逐个消费来自 Kafka 主题的消息
        try:
            # 一个迭代器，持续监听 Kafka 主题并接收消息。每当有新消息到达时，这一行就会读取到消息。
            for message in self.consumer:
                #  yield 关键字使得这个方法成为一个生成器，每当有新消息时，就会返回一条消息。
                yield message
        except Exception as e:
            print(repr(e))

    # 从 HDFS 下载视频，并调用一个处理器 (self.processor) 来处理视频数据
    def process_video(self, obj):
    # def process_video(self, obj):
        id = obj['id']
        try:
            # 获取与视频 ID 相关的处理记录
            print("11111111111111111")
            record = common_utils.getProcessRecordById(id)
            print(type(record))
            print(record)
            userid = record[1]
            print("userid:", userid)
            print("22222222222222222")
        except Exception as e:
            print("33333333333333333")
            # 如果出现异常，更新处理记录为失败，并输出异常信息。
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] get record failed')
            print("get record Exception! ", repr(e))
            return

        # 检查记录是否存在,如果未找到处理记录，则更新状态为失败并返回
        if len(record) == 0:
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] can not find record. id = {}'.format(id))
            print("can not find record. id = " + id)
            return

        # 保存字幕文本信息
        try:
            question = record[16]
            # 初始化结果字典
            result = {}
            for i in range(1, 1472):
            # for i in range(1, 11):
                result[str(i)] = {
                    "chCap": [question]
                }
            print("999999999999999999999999999999999")
            output_file = f"/home/disk2/dachuang1-23/text/kafka_result/{id}/vatex_data.json"

            # 检查路径是否存在，不存在则创建
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # 创建目录
            # 保存为 JSON 文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"JSON 文件已保存到 {output_file}")
        except Exception as e:
            print(repr(e))
            print("问题文本信息保存失败！")


        #调用视频文本匹配算法
        try:
            main(id, userid)
            print("视频文本匹配算法运行完成！")
        except Exception as e:
            print(repr(e))
            # 打印异常类型和消息
            print(f"异常类型: {type(e).__name__}")
            print(f"异常消息: {e}")

            # 打印完整的堆栈信息
            print("详细的错误信息：")
            print(traceback.format_exc())
            print("视频文本匹配算法调用出错！")

        #self.processor.forward(id, video_path)
        

        local_dir1 = vs_common.local_result_store_dir.format(id)
        print("-----------------------local_dir1-------------------")
        print(local_dir1)
        local_dir = os.path.dirname(local_dir1)
        print("-----------------------local_dir-------------------")
        print(local_dir)
        hdfs_dir = vs_common.hdfs_result_store_path.format(id)
        print("---------------------hdfs_dir---------------------")
        print(hdfs_dir)

        self.upload_directory_to_hdfs(local_dir, hdfs_dir)
        print("文件上传成功！")
        print("-----------------------------------------------------------------------------------------")

    def upload_directory_to_hdfs(self, local_dir, hdfs_dir):
        # 确保HDFS目标文件夹存在
        try:
            self.hdfs_client.makedirs(hdfs_dir)

        except Exception as e:
            print(f"Failed to create HDFS directory {hdfs_dir}: {e}")

        # 定义视频文件的扩展名
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv'}  # 可以根据需要添加其他视频格式

        # 遍历本地文件夹中的文件并上传
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # 获取文件的扩展名
                _, ext = os.path.splitext(file)
                # 检查是否为视频文件，如果是则跳过上传
                if ext.lower() in video_extensions:
                    print(f"Skipped video file: {local_file_path}")
                    continue

                # 计算在HDFS中的目标路径
                # 获取 local_file_path 相对于 local_dir 的相对路径
                relative_path = os.path.relpath(local_file_path, local_dir)
                print(f"---------------Relative path----------------")
                print(relative_path)
                # 将目标路径与 hdfs_dir 拼接
                hdfs_file_path = os.path.join(hdfs_dir, relative_path)
                print("-------------hdfs_file_path-----------------")
                print(hdfs_file_path)
                # 确保HDFS中的子目录存在
                try:
                    self.hdfs_client.makedirs(os.path.dirname(hdfs_file_path))
                except Exception as e:
                    print(f"Failed to create HDFS subdirectory for {hdfs_file_path}: {e}")

                # 上传文件到HDFS
                self.hdfs_client.write(hdfs_file_path, open(local_file_path, 'rb'), overwrite=True)
                print(f"Uploaded {local_file_path} to {hdfs_file_path}")


if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        print(f"当前 GPU 编号: 0")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    else:
        print("没有可用的 GPU 设备。")

    consumer = GateConsumer()
    # 调用 consume_data 方法开始从 Kafka 主题中消费消息。这是一个生成器，持续监听来自 Kafka 的消息。
    messages = consumer.consume_data()
    for message in consumer.consumer:
        # 当有新消息到达时，会将消息内容解析为 JSON 对象，存储在 obj 中
        obj = json.loads(message.value)
        try:
            print("处理消息！")
            consumer.process_video(obj)
            # consumer.process_video(obj)

        # 处理过程中发生异常，捕获异常并打印出错误信息，包括导致失败的消息内容
        except Exception as e:
            print("process video failed, message:{}".format(message), repr(e))

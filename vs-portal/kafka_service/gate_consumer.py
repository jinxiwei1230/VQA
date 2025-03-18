import json
import subprocess
from logging import exception
import os
import re
import time
import traceback
import sys
from hdfs import InsecureClient
from kafka import KafkaConsumer
import sys

from omegaconf import OmegaConf

# # 将 evaluate.py 和其他模块所在的路径添加到 sys.path
sys.path.append('/home/dachuang/vqa/sevila')  # 添加 evaluate.py 所在的路径
from model_load import load_model_from_config
from ConfigUpdate import update_config
from model_load import load_cfg_from_config
sys.path.append('/home/disk2/dachuang1-23')
from zimu import process_audio_from_video


# 该代码从Kafka主题中消费消息，通过HDFS下载相应的视频文件到本地，再调用VideoProcessor类进行视频处理。处理完成后，更新处理状态。
sys.path.append('/home/disk2/dachuang1-23/vs-portal')


import vs_common
import common_utils
from process import VideoProcessor

# 前端上传 ➔ 后端接收视频文件。
# 后端上传 ➔ 将视频文件存储到 HDFS。
# 后端通知 ➔ 生成包含视频路径和问题信息的消息并发送到 Kafka。
# 消费者执行 ➔ 消费者获取消息并从 HDFS 下载视频进行处理。

bootstrapServers = ['10.92.64.242:9092']
topic = 'test'
groupId = 'vqa'
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
    def process_video(self, obj, model):
    # def process_video(self, obj):
        id = obj['id']
        try:
            # 获取与视频 ID 相关的处理记录
            record = common_utils.getProcessRecordById(id)
        except Exception as e:
            # 如果出现异常，更新处理记录为失败，并输出异常信息。
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] get record failed')
            print("get record Exception! ", repr(e))
            return

        # 检查记录是否存在,如果未找到处理记录，则更新状态为失败并返回
        if len(record) == 0:
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] can not find record. id = {}'.format(id))
            print("can not find record. id = " + id)
            return

        # 从记录中提取视频路径和用户ID
        print("get record success. record:{}".format(record))
        video_path = record[2]
        user_id = record[1]

        # 设置存储路径,构建 HDFS 路径和本地存储路径。如果存储路径不存在，则创建它。
        video_full_name = video_path.split('/')[-1]  # 将 video_path 按照斜杠 (/) 分割成多个部分，并获取最后一个部分，也就是视频文件的名称
        hdfs_path = vs_common.hdfs_video_store_path.format(user_id) + '/{}'.format(video_path)
        store_path = video_store_dir.format(id)
        video_path = store_path + '/{}'.format(video_full_name)
        print(hdfs_path)
        print(store_path)
        print(video_path)
        # / bdap / students / 2023110788 // videos / test.mp4
        # / home / disk2 / dachuang1 - 23 / kafka_result / 265
        # / home / disk2 / dachuang1 - 23 / kafka_result / 265 / test.mp4

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        try:
            # 从 HDFS 下载视频到本地，并记录下载耗时。
            # 下载很快 测试160MB左右的视频 1s左右即可从HDFS下载到本地
            print("start download")
            start = time.time()
            self.hdfs_client.download(hdfs_path, store_path, overwrite=True, n_threads=3)
            end = time.time()
            print("download {}/{} finish!, wasted {} ms".format(id, video_full_name, end - start))
            # 下载完成后，更新处理状态为“预处理”,修改 process state
            common_utils.updateProcessStateById(id, vs_common.IN_PREPROCESSING)

        except Exception as e:
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] download video or update record failed')
            print("download video or get record Exception! ", repr(e))
            return

        try:
            record = common_utils.getProcessRecordById(id)
            video_id = record[2]
            file_name = os.path.basename(video_id)
            process_audio_from_video(id, file_name)
            print("字幕提取成功！")

        except Exception as e:
            # 获取详细的错误信息
            error_detail = traceback.format_exc()
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] 字幕提取失败')
            print("字幕提取失败！ ", repr(e))
            print(error_detail)
            # return

        try:
            try:
                # 运行 .sh 脚本
                script_path = '/home/dachuang/vqa/sevila/load_model.sh'  # 替换为实际的 .sh 文件路径
                # 传递 id 变量到脚本中
                subprocess.run(['bash', script_path, str(id)], check=True)
                print("id替换完成！")
            except subprocess.CalledProcessError as e:
                # 处理运行脚本时的异常
                print("运行脚本时发生异常（替换id）:", e)
                print("错误输出:", e.stderr)
                return

            # 调用 self.processor.forward 进行视频处理
            result_dir = "/home/disk2/dachuang1-23/kafka_result/" + str(id)
            exp_name = 'nextqa_infer'
            ckpt = "/home/disk2/dachuang1-23/results/nextqa_ft/sevila_pretrained.pth"

            cfg_path = "/home/dachuang/vqa/sevila/lavis/projects/sevila/eval/nextqa_eval.yaml"
            options = [
                f'run.output_dir={result_dir}/{exp_name}',
                'model.frame_num=4',
                'datasets.nextqa.vis_processor.eval.n_frms=32',
                'run.batch_size_eval=2',
                'model.task=qvh_freeze_loc_freeze_qa_vid',
                f"model.finetuned={ckpt}",
                'run.task=videoqa'
            ]

            cfg = load_cfg_from_config(cfg_path, options)
            print(vars(cfg))

            self.processor.forward(id, video_path, cfg, model)
            # self.processor.forward(id, video_path)

        # 如果发生异常，则更新记录为失败并打印异常信息
        except Exception as e:
            common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] process video failed. {}'.format(repr(e)))
            print("preprocess video or relation extraction Exception! ", repr(e))
            info = traceback.format_exc()
            print(info)
            return
        # 修改状态.如果处理成功，更新状态为成功
        common_utils.updateProcessStateById(id, vs_common.SUCCEED)

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
                # print(f"---------------Relative path----------------")
                # print(relative_path)
                # 将目标路径与 hdfs_dir 拼接
                hdfs_file_path = os.path.join(hdfs_dir, relative_path)
                # print("-------------hdfs_file_path-----------------")
                # print(hdfs_file_path)
                # 确保HDFS中的子目录存在
                try:
                    self.hdfs_client.makedirs(os.path.dirname(hdfs_file_path))
                except Exception as e:
                    print(f"Failed to create HDFS subdirectory for {hdfs_file_path}: {e}")

                # 上传文件到HDFS
                self.hdfs_client.write(hdfs_file_path, open(local_file_path, 'rb'), overwrite=True)
                # print(f"Uploaded {local_file_path} to {hdfs_file_path}")


def run_model_loading():
    result_dir = "/home/disk2/dachuang1-23/kafka_result/400"
    exp_name = 'nextqa_infer'
    ckpt = "/home/disk2/dachuang1-23/results/nextqa_ft/sevila_pretrained.pth"
    # ckpt = "/home/xuyuanxing-22/VideoQA/sevila/sevila_checkpoints/sevila_pretrained.pth"

    cfg_path = "/home/dachuang/vqa/sevila/lavis/projects/sevila/eval/nextqa_eval.yaml"
    options = [
        f'run.output_dir={result_dir}/{exp_name}',
        'model.frame_num=4',
        'datasets.nextqa.vis_processor.eval.n_frms=32',
        'run.batch_size_eval=2',
        'model.task=qvh_freeze_loc_freeze_qa_vid',
        f"model.finetuned={ckpt}",
        'run.task=videoqa'
    ]

    # 调用 model_load.py 中的函数，加载模型、数据集和任务
    cfg, model = load_model_from_config(cfg_path, options)
    return cfg, model

if __name__ == '__main__':
    import torch

    print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
    print(torch.cuda.is_available())  # 检查 CUDA 是否可用

    # 运行模型加载并获取返回值
    cfg, model = run_model_loading()

    consumer = GateConsumer()
    # 调用 consume_data 方法开始从 Kafka 主题中消费消息。这是一个生成器，持续监听来自 Kafka 的消息。
    messages = consumer.consume_data()
    for message in consumer.consumer:
        # 当有新消息到达时，会将消息内容解析为 JSON 对象，存储在 obj 中
        obj = json.loads(message.value)
        try:
            print("处理消息！")
            consumer.process_video(obj, model)
            # consumer.process_video(obj)

        # 处理过程中发生异常，捕获异常并打印出错误信息，包括导致失败的消息内容
        except Exception as e:
            print("process video failed, message:{}".format(message), repr(e))

# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import os
import time
import traceback
import sys

import requests
import random
import json
from hashlib import md5
# Set your own appid/appkey.
appid = '20240716002101367'
appkey = '1lRfgIX4PCbSnrhMB4kh'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'zh'
to_lang =  'en'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

from hdfs import InsecureClient
from kafka import KafkaConsumer
import sys
sys.path.append('/home/disk2/dachuang1-23/HMMC')
sys.path.append('/home/disk2/dachuang1-23/HMMC/v3s-portal')
sys.path.append('/home/disk2/dachuang1-23/qwen2.5-vl-3B')

import vs_common
import common_utils
from process import VideoProcessor

from video_answer import describe_video
bootstrapServers = ['10.92.64.241:9092']
topic = 'qwen'  # qwen
groupId = 'vqa4'
# video_store_dir = vs_common.local_video_store_dir
video_store_dir = '/home/disk2/dachuang1-23/qwen_kafka_result/{}'
local_result_store_dir = '/home/disk2/dachuang1-23/qwen_kafka_result/{}/process_results'

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def baidu_api(query,from_lang,to_lang):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    return result["trans_result"][0]['dst']

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
            video_path = record[2]
            user_id = record[1]
            q = record[16]
            a0 = record[17]
            a1 = record[18]
            a2 = record[19]
            a3 = record[20]
            a4 = record[21]
            print(a0)
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

        # 设置存储路径,构建 HDFS 路径和本地存储路径。如果存储路径不存在，则创建它。
        video_full_name = video_path.split('/')[-1]  # 将 video_path 按照斜杠 (/) 分割成多个部分，并获取最后一个部分，也就是视频文件的名称
        hdfs_path = vs_common.hdfs_video_store_path.format(user_id) + '/{}'.format(video_path)
        store_path = video_store_dir.format(id)
        video_path = store_path + '/{}'.format(video_full_name)
        print("video_full_name:", video_full_name)
        print(hdfs_path)
        print(store_path)
        print(video_path)
        # /home/disk2/dachuang1-23/text/kafka_result/970
        # /home/disk2/dachuang1-23/text/kafka_result/970/954.mp4

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

        #调用视频文本匹配算法
        try:
            # main(id, userid)
            print("video_path:", video_path)
            q_trans = baidu_api(q, from_lang, to_lang)  # 中-》英
            if (a0!=None and a1!=None and a2!=None and a3!=None and a4!=None):
                a0_trans = baidu_api(a0, from_lang, to_lang)
                a1_trans = baidu_api(a1, from_lang, to_lang)
                a2_trans = baidu_api(a2, from_lang, to_lang)
                a3_trans = baidu_api(a3, from_lang, to_lang)
                a4_trans = baidu_api(a4, from_lang, to_lang)
                question = f"{q_trans}+The answer simply returns the number corresponding to one option.\nOptions:\n0. {a0_trans}\n1. {a1_trans}\n2. {a2_trans}\n3. {a3_trans}\n4. {a4_trans}"
                print("多选组合后问题：", question)
            else:
                question = q_trans
                print("开放式问题：", question)

            # 将视频和问题传入模型
            description = describe_video(video_path, question)
            print(description)
            print("qwen运行完成！")

            # 将结果存入 JSON 文件
            result = {
                "qwen_answer": description  # 保存生成的答案
            }
            json_file_path = os.path.join(store_path, "qwen_result.json")
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(result, json_file, ensure_ascii=False, indent=4)
            print(f"结果已保存到 {json_file_path}")

        except Exception as e:
            print(repr(e))
            print(f"异常消息: {e}")
            print(traceback.format_exc())
            print("视频文本匹配算法调用出错！")

        local_dir1 = local_result_store_dir.format(id)
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
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # print('$' * 20)
    # processor = AutoProcessor.from_pretrained(model_path)

    consumer = GateConsumer()
    # 调用 consume_data 方法开始从 Kafka 主题中消费消息。这是一个生成器，持续监听来自 Kafka 的消息。
    messages = consumer.consume_data()
    for message in consumer.consumer:
        # 当有新消息到达时，会将消息内容解析为 JSON 对象，存储在 obj 中
        obj = json.loads(message.value)
        try:
            print("处理消息！")
            consumer.process_video(obj)

        # 处理过程中发生异常，捕获异常并打印出错误信息，包括导致失败的消息内容
        except Exception as e:
            print("process video failed, message:{}".format(message), repr(e))

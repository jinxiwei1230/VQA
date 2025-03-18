# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

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


import os
import traceback

import common_utils

import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont
from hdfs import InsecureClient
import sys

import vs_common
from common_utils import Timer
# from face_cluster import FaceCluster
# from face_detect import FaceRecognition
# from face_feature_visualization import FaceFeatureVisualization
# from object_det import ObjDetector
# from relation_extraction import RelationExtraction
import subprocess
sys.path.append('/home/dachuang/vqa/sevila')  # 添加 evaluate.py 所在的路径
import evaluate
from model_load import load_datasets_tasks
import translators as ts

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
    print(json.dumps(result, indent=4, ensure_ascii=False))
    return result["trans_result"][0]['dst']


#该类整合了视频处理的各个步骤，包括目标检测、人脸识别、人脸聚类、关系抽取等
class VideoProcessor:
    def __init__(self):
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='video_sys')  # 连接HDFS客户端

    def forward(self, id, video_path, cfg, model):
    # def forward(self, id, video_path):
        # Timer 是一个上下文管理器，用于测量整个处理过程所用的时间
        with Timer('All preprocess step'):
            source = cv2.VideoCapture(video_path)  # 使用OpenCV读取视频文件

            # 获取视频的宽度、高度、帧率和总帧数
            frame_width = source.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = source.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = source.get(cv2.CAP_PROP_FPS)
            fcount = source.get(cv2.CAP_PROP_FRAME_COUNT)
            font_style = self.get_font_style(frame_width, frame_height) # 根据视频的分辨率设置字体样式

            file_info = os.stat(video_path)  # 获取视频文件的元数据
            file_size = file_info.st_size  # 获取视频文件大小

            # 将视频信息更新到数据库或记录中
            common_utils.updateVideoInfoById(id, fps=int(fps),
                width=frame_width, height=frame_height,
                duration=(int)(fcount/fps), size=file_size)

            # 修改状态.如果处理成功，更新状态为成功
            common_utils.updateProcessStateById(id, vs_common.IN_RELATION_EXTRACTING)

            # 打印出修改后的整条记录
            try:
                print("-------------------- 更新后的记录 --------------------")
                # 获取与视频 ID 相关的处理记录
                updated_record = common_utils.getProcessRecordById(id)
                # 获取保存json文件的目录路径
                qa_annos_dir = os.path.join(os.path.dirname(video_path), "qa_annos")
                print("qa_annos_dir:")
                print(qa_annos_dir)
                # 检查目录是否存在，不存在则创建
                if not os.path.exists(qa_annos_dir):
                    os.makedirs(qa_annos_dir)

                # 存放 JSON 数据的列表
                json_data_list = []

                # 翻译为英文
                q = updated_record[16]
                a0 = updated_record[17]
                a1 = updated_record[18]
                a2 = updated_record[19]
                a3 = updated_record[20]
                a4 = updated_record[21]
                q_trans = baidu_api(q, from_lang, to_lang)
                a0_trans = baidu_api(a0, from_lang, to_lang)
                a1_trans = baidu_api(a1, from_lang, to_lang)
                a2_trans = baidu_api(a2, from_lang, to_lang)
                a3_trans = baidu_api(a3, from_lang, to_lang)
                a4_trans = baidu_api(a4, from_lang, to_lang)
                print("q:", q)
                print("a0:", a0)
                print("a1:", a1)
                print("a2:", a2)
                print("a3:", a3)
                print("a4:", a4)
                print("q_trans:",q_trans)
                print("a0_trans:",a0_trans)
                print("a1_trans:",a1_trans)
                print("a2_trans:",a2_trans)
                print("a3_trans:",a3_trans)
                print("a4_trans:",a4_trans)

                # 生成 JSON 格式的数据
                json_data = {
                    "video_id": "/".join(video_path.split('/')[-2:-1]) + '/' + video_path.split('/')[-1].split('.')[0],
                    # 提取出路径前缀 'videos/test'
                    "num_option": 5,  # 题目选项数
                    "qid": f"_5_{updated_record[0]}",  # 构建 QID
                    "a0": a0_trans,  # 选项 0
                    "a1": a1_trans,  # 选项 1
                    "a2": a2_trans,  # 选项 2
                    "a3": a3_trans,  # 选项 3
                    "a4": a4_trans,  # 选项 4
                    "answer": updated_record[3],  # 答案
                    "question": q_trans  # 问题
                }

                # 将生成的 JSON 数据添加到列表中
                json_data_list.append(json_data)

                # 将整个列表写入 JSON 文件，使用缩进格式
                json_string = json.dumps(json_data_list, ensure_ascii=False, indent=4)
                output_file = os.path.join(qa_annos_dir, "output.json")
                print(json_string)
                with open(output_file, "w") as json_file:
                    json_file.write(json_string)
                print(f"数据已保存到 {output_file}")

            except Exception as e:
                # 如果出现异常，更新处理记录为失败，并输出异常信息
                common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] 获取更新记录失败')
                print("获取更新记录时异常: ", repr(e))
                print("详细报错信息:")
                traceback.print_exc()  # 打印完整的异常堆栈信息
                return

            # 检查记录是否存在, 如果未找到处理记录，则更新状态为失败并返回
            if len(updated_record) == 0:
                common_utils.updateProcessRecordById(id, vs_common.FAILED, '[ALG] 找不到记录. id = {}'.format(id))
                print("找不到记录. id = " + str(id))
                return

            try:
                datasets, task = load_datasets_tasks(cfg)
                output_dir = "/home/disk2/dachuang1-23/kafka_result/" + str(id)
                evaluate.main(cfg, model, datasets, task, output_dir)
                print("模型调用完成！！！！")

            except subprocess.CalledProcessError as e:
                print("调用时发生异常:", e)
                print("错误输出:", e.stderr)
                return

            # try:
            #     # 运行 .sh 脚本
            #     script_path = '/home/dachuang/vqa/sevila/run_scripts/sevila/inference/nextqa_infer.sh'  # 替换为实际的 .sh 文件路径
            #     # 传递 id 变量到脚本中
            #     #subprocess.run(['bash', script_path, str(id)], check=True, capture_output=True, text=True)
            #     subprocess.run(['bash', script_path, str(id)], check=True)
            #     print("模型调用完成！！！！")
            # except subprocess.CalledProcessError as e:
            #     # 处理运行脚本时的异常
            #     print("运行脚本时发生异常:", e)
            #     print("错误输出:", e.stderr)
            #     return

    # python kafka_service/gate_consumer.py >> /home/disk2/dachuang1-23/kafka_result/output.log 2>&1 &

    # 根据视频的分辨率设定字体大小，以便在可视化时使用
    def get_font_style(self, frame_width, frame_height):
        frame_max = max(frame_height, frame_width)  # 找到最大边
        # 根据分辨率设置字体大小
        font_size = 55 if frame_max > 1500 else 35 if frame_max < 960 else 45
        #  定义打印文字的字体
        fontStyle = ImageFont.truetype("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc", font_size, encoding="utf-8")
        return fontStyle



if __name__ == '__main__':
    process = VideoProcessor()
    # process.visualize_face_cluster.forward(process.hdfs_client, 17)
    process.forward(45, '/home/shared/video/45/hoc_crowded_trim.mp4')



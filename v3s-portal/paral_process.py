import os
import sys
import time
import json

import numpy as np
from moviepy.editor import AudioFileClip
from scipy.io import wavfile
import cv2
import matplotlib.pyplot as plt
from hdfs import InsecureClient
import common_utils
import vs_common
from face_detect import FaceRecognition
from kafka_service.producer import Kafka_producer
sys.path.append('/home/zhangyuxuan-23/vs-portal')

class VideoProcessor:
    def __init__(self):
        # 创建了一个用于topic=face_recognition的生产者
        self.producer_face_recognition = Kafka_producer('face_recognition')

        # 创建了一个用于topic=asr的生产者
        self.producer_asr = Kafka_producer('asr')

        self.face_recognition = FaceRecognition()
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')

    def forward(self, id, video_path):
        #with Timer('All preprocess step'):
        source = cv2.VideoCapture(video_path)

        frame_width = source.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = source.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = source.get(cv2.CAP_PROP_FPS)
        fcount = source.get(cv2.CAP_PROP_FRAME_COUNT)
        # font_style = self.get_font_style(frame_width, frame_height)

        file_info = os.stat(video_path)
        file_size = file_info.st_size
        common_utils.updateVideoInfoById(id, fps=int(fps),
            width=frame_width, height=frame_height,
            duration=(int)(fcount/fps), size=file_size)

        hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(id) + '/origin'
        if not os.path.exists(hdfs_frame_output_dir):
            self.hdfs_client.makedirs(hdfs_frame_output_dir, 777)

        # 设置阈值 防止切分长度太短
        situation_threshold = fps * 3
        # 每秒均匀抽2帧
        skip = (int)(fps / 2)
        last_frame = None
        last_frame_id = 0
        frame_id = -1
        situation = 0
        last_situation = 0

        second_to_situation = {}
        while source.isOpened():
            ret, frame = source.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % skip != 0:
                continue
            # 测试用
            # if frame_id > 3000: break

            #with Timer('frame_{} process step'.format(frame_id)):
                # 分镜
            if last_frame is not None:
                n = self.calculate(last_frame, frame)
                if n < 0.6:  # 判断是否切分场景
                    if frame_id - last_frame_id >= situation_threshold:
                        situation += 1
                        last_frame_id = frame_id
            second_to_situation[frame_id / fps] = situation
            # 保存分镜结果
            self.save_situation_frame(hdfs_frame_output_dir, situation, frame_id, frame)

            if situation != last_situation:
                message = {}
                # 每次分完一个镜头就通知这个镜头下的所有帧可以进行人脸识别
                message['movie_id'] = id
                message['scene_id'] = situation - 1
                self.producer_face_recognition.sendMessage(message, partition=situation % 3)
                # print("发送了场景号：", situation-1)

            # 记录上一帧
            last_frame = frame
            last_situation = situation
        # 保存共有多少个situation用于后续进度控制
        common_utils.updateVideoSituationNum(id, situation+1)
        message = {}
        # 通知最后一个镜头
        message['movie_id'] = id
        message['scene_id'] = situation
        self.producer_face_recognition.sendMessage(message, partition=situation % 3)

        # 记录frame_id => situation_id 动态图对应
        hdfs_second_to_situation_dir = vs_common.hdfs_result_store_path.format(id) + '/process/relation'
        if not os.path.exists(hdfs_second_to_situation_dir):
            self.hdfs_client.makedirs(hdfs_second_to_situation_dir, 777)

        hdfs_relation_output_path = hdfs_second_to_situation_dir + '/second_to_situation.json'
        # 允许overwrite 是防止消息消费失败自动重试的场景
        self.hdfs_client.write(hdfs_relation_output_path, json.dumps(second_to_situation).encode(), overwrite=True)

        # 提取视频的音频保存为.wav 文件到hdfs文件系统，完成之后通知asr消费者进行音频到文本的提取
        # 暂时下线
        # my_audio_clip = AudioFileClip(video_path)
        # # 保存至本地
        # local_audio_output_path = vs_common.local_video_store_dir.format(id) + "/audio/"
        # if not os.path.exists(local_audio_output_path):
        #     os.makedirs(local_audio_output_path)
        # my_audio_clip.write_audiofile(local_audio_output_path + "audio_file.wav")
        # my_audio_clip.close()
        # # 读取音频文件: 这里将音频文件读取问numpy数组，并保存为txt文件，然后上传到hdfs
        # rate, sig = wavfile.read(local_audio_output_path + "audio_file.wav")
        # np.savetxt(local_audio_output_path + "audio_numpy_file.txt", X=sig)
        # hdfs_audio_file_output_path = vs_common.hdfs_result_store_path.format(id) + '/process/audio/audio_file.txt'
        # with open(local_audio_output_path + "audio_numpy_file.txt", 'r') as f:
        #     self.hdfs_client.write(hdfs_audio_file_output_path, f, overwrite=True)
        # # 发送消息给ASR消费者，消息内容包括：{"id": 视频id，"rate": 音频采样帧率}
        # message_asr = {}
        # message_asr["movie_id"] = id
        # message_asr["rate"] = rate
        # message_asr["audio_path"] = hdfs_audio_file_output_path
        # self.producer_asr.sendMessage(message_asr, partition=id % 5)


    # 通过比较RGB每个通道的直方图计算相似度 https://blog.csdn.net/ad_yangang/article/details/120857199
    def classify_hist_with_split(self, image1, image2, size=(256, 256)):
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        image1 = cv2.resize(image1, size)
        image2 = cv2.resize(image2, size)
        plt.imshow(image1)
        plt.show()
        plt.axis('off')

        plt.imshow(image2)
        plt.show()
        plt.axis('off')  # 把两张图分别显示出来

        sub_image1 = cv2.split(image1)
        sub_image2 = cv2.split(image2)  # 把单通道取出来
        print(type(sub_image1))
        sub_data = 0

        for im1, im2 in zip(sub_image1, sub_image2):  # 打包可迭代的参数
            sub_data += self.calculate(im1, im2)
        sub_data = sub_data / 3
        return sub_data

    # 计算单通道的直方图的相似值
    def calculate(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])  # 通道0，1，2 对应B,G,R
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # plt.plot(hist1, color="r")  # 画出直方图
        # plt.plot(hist2, color="g")
        # plt.show()
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1  # 统计相似
        degree = degree / len(hist1)
        return degree

    def save_situation_frame(self, hdfs_frame_output_dir, situation_id, frame_id, frame):
        hdfs_situation_frame_path = hdfs_frame_output_dir + '/{}'.format(situation_id)
        if not os.path.exists(hdfs_situation_frame_path):
            self.hdfs_client.makedirs(hdfs_situation_frame_path, 777)

        hdfs_frame_dst_path = hdfs_situation_frame_path + "/image_{}.jpg".format(frame_id)
        self.hdfs_client.write(hdfs_frame_dst_path, cv2.imencode('.jpg', frame)[1].tobytes(), overwrite=True)


if __name__ == '__main__':
    process = VideoProcessor()

    start = time.time()
    print("开始时间为", start)
    process.forward(86, '/home/huyibo-21/shared/video/86/3min_540p.mp4')

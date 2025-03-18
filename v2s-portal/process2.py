import json
import os
import common_utils

import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont
from hdfs import InsecureClient

import vs_common
from common_utils import Timer
from face_cluster import FaceCluster
from face_detect import FaceRecognition
from face_feature_visualization import FaceFeatureVisualization
from object_det import ObjDetector
from relation_extraction import RelationExtraction
import subprocess
import sys



#该类整合了视频处理的各个步骤，包括目标检测、人脸识别、人脸聚类、关系抽取等
class VideoProcessor:
    def __init__(self):
        self.obj_detector = ObjDetector()   # 初始化目标检测器
        self.face_recognition = FaceRecognition()  # 初始化人脸识别器
        self.face_cluster = FaceCluster()  # 初始化人脸聚类器
        self.relation_extraction = RelationExtraction()  # 初始化关系抽取器
        self.visualize_face_cluster = FaceFeatureVisualization()  # 初始化人脸特征可视化工具
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='video_sys')  # 连接HDFS客户端

    def forward(self, id, video_path):
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


            # 设置结果存放路径
            frame_output_dir = vs_common.local_result_store_dir.format(id) + '/origin'
            # 抽出的帧存放文件夹路径
            hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(id) + '/origin'
            if not os.path.exists(frame_output_dir):
                os.makedirs(frame_output_dir)
                self.hdfs_client.makedirs(hdfs_frame_output_dir, 777)


            # 设置阈值 防止切分长度太短
            situation_threshold = fps * 2  # 设置场景切分阈值（2秒）
            skip = (int)(fps / 2) # 每秒均匀抽2帧
            last_frame = None
            last_frame_id = 0
            frame_id = -1
            situation = 0

            # 人脸识别记录所需变量定义
            # 记录未识别出的人脸特征 用于后续聚类
            unknown_feature_matrix = []  # 用于存储未识别的人脸特征
            # 特征行号 id => (situation_id, frame_id, bbox)
            rowid_to_face_dict = {}  # 行号映射到人脸信息
            # name => [{situation_id, frame_id, bbox, feature}]
            name_to_face_dict = {}  # 姓名映射到人脸信息
            # frame_name => [bbox]
            frame_to_bboxes = {}  # 每帧的边界框信息
            # frame_id => situation_id
            second_to_situation = {}  # 每秒的场景ID映射

            while source.isOpened():  # 逐帧处理视频
                ret, frame = source.read()
                if not ret: # 如果没有帧了，结束循环
                    break
                frame_id += 1
                if frame_id % skip != 0:  # 按照设定的帧间隔进行处理（在 frame_id 能被 skip 整除时）
                    continue
                # 测试用
                # if frame_id > 3000: break

                with Timer('frame_{} process step'.format(frame_id)):
                    # 分镜
                    if last_frame is not None:
                        n = self.calculate(last_frame, frame)  # 计算场景相似度
                        if n < 0.6:  # 判断是否切分场景
                            # 如果相似度低于阈值且时间差大于 situation_threshold，则认为场景发生了变化
                            if frame_id - last_frame_id >= situation_threshold:
                                situation += 1  # 增加场景计数
                                last_frame_id = frame_id
                    second_to_situation[frame_id / fps] = situation  # 记录时间与场景的映射
                    # 保存分镜结果
                    self.save_situation_frame(frame_output_dir, hdfs_frame_output_dir, situation, frame_id, frame)  # 保存场景帧
                    # 记录上一帧
                    last_frame = frame

                    # 目标检测：调用目标检测器的 forward 方法进行目标检测
                    self.obj_detector.forward(id, situation, frame_id, frame, font_style, self.hdfs_client)

                    # 人脸识别：调用人脸识别器的 forward 方法进行人脸识别，传递必要的参数
                    self.face_recognition.forward(
                        id,
                        situation,
                        frame_id,
                        frame,
                        font_style,
                        name_to_face_dict,
                        unknown_feature_matrix,
                        rowid_to_face_dict,
                        self.hdfs_client,
                        frame_to_bboxes
                    )

            print("video {} start face cluster".format(id))

            # 人脸聚类
            label2idx = self.face_cluster.forward(unknown_feature_matrix)

            # 将聚类结果保存到数据库或存储系统。
            self.face_cluster.save_cluster_result(
                id,
                label2idx,
                name_to_face_dict,
                rowid_to_face_dict,
                unknown_feature_matrix,
                font_style,
                self.hdfs_client
            )

            # 将边界框信息保存到 JSON 文件中
            frame_to_bboxes_path = vs_common.local_result_store_dir.format(id) + '/process/face/bbox_dict.json'
            with open(frame_to_bboxes_path, 'w') as file:
                json.dump(frame_to_bboxes, file)

            # 创建存储每秒与场景ID映射的目录
            # 记录frame_id => situation_id 动态图对应
            second_to_situation_dir = vs_common.local_result_store_dir.format(id) + '/process/relation'
            hdfs_second_to_situation_dir = vs_common.hdfs_result_store_path.format(id) + '/process/relation'
            if not os.path.exists(second_to_situation_dir):
                os.makedirs(second_to_situation_dir)
                self.hdfs_client.makedirs(hdfs_second_to_situation_dir, 777)

            # 将每秒与场景ID的映射保存到 JSON 文件，并上传到 HDFS
            with open(second_to_situation_dir + '/second_to_situation.json', 'w') as file:
                json.dump(second_to_situation, file)  # 保存场景ID映射
            hdfs_relation_output_path = hdfs_second_to_situation_dir + '/second_to_situation.json'
            # 允许overwrite 是防止消息消费失败自动重试的场景
            self.hdfs_client.write(hdfs_relation_output_path, json.dumps(second_to_situation).encode(), overwrite=True)

            # 人脸特征: 可视化人脸聚类结果
            self.visualize_face_cluster.forward(self.hdfs_client, id)

            # 修改状态: 更新处理状态
            common_utils.updateProcessStateById(id, vs_common.IN_RELATION_EXTRACTING)

            print("video {} start relation extraction".format(id))

            # 关系抽取
            self.relation_extraction.forward(id, self.hdfs_client)


    # 根据视频的分辨率设定字体大小，以便在可视化时使用
    def get_font_style(self, frame_width, frame_height):
        frame_max = max(frame_height, frame_width)  # 找到最大边
        # 根据分辨率设置字体大小
        font_size = 55 if frame_max > 1500 else 35 if frame_max < 960 else 45
        #  定义打印文字的字体
        fontStyle = ImageFont.truetype("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc", font_size, encoding="utf-8")
        return fontStyle

    # 通过比较RGB每个通道的直方图计算相似度 https://blog.csdn.net/ad_yangang/article/details/120857199
    def classify_hist_with_split(self, image1, image2, size=(256, 256)):
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        image1 = cv2.resize(image1, size)  # 将输入的图像调整到指定的大小
        image2 = cv2.resize(image2, size)
        plt.imshow(image1)  # 显示调整后的图像
        plt.show()
        plt.axis('off')  # 关闭坐标轴显示

        plt.imshow(image2)
        plt.show()
        plt.axis('off')  # 把两张图分别显示出来

        sub_image1 = cv2.split(image1)  # 将每张图像分离成 R、G、B 三个通道
        sub_image2 = cv2.split(image2)
        print(type(sub_image1))
        sub_data = 0

        for im1, im2 in zip(sub_image1, sub_image2):  # 使用 zip 打包这两个图像的三个通道，以便进行逐通道比较
            sub_data += self.calculate(im1, im2)  # 计算每个通道的相似度.并将相似度累加
        sub_data = sub_data / 3
        return sub_data   # 返回三个通道相似度的平均值

    # 计算单通道的直方图的相似值
    def calculate(self, image1, image2):
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])  # 通道0，1，2 对应B,G,R
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        plt.plot(hist1, color="r")  # 画出直方图
        plt.plot(hist2, color="g")
        plt.show()
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1  # 统计相似
        degree = degree / len(hist1)
        return degree


    # frame_output_dir：本地保存帧的目录。
    # hdfs_frame_output_dir：HDFS保存帧的目录。
    # situation_id：表示当前场景的ID。
    # frame_id：当前帧的ID。
    # frame：要保存的图像帧。
    def save_situation_frame(self, frame_output_dir, hdfs_frame_output_dir, situation_id, frame_id, frame):
        situation_frame_path = frame_output_dir + '/{}'.format(situation_id)
        hdfs_situation_frame_path = hdfs_frame_output_dir + '/{}'.format(situation_id)
        if not os.path.exists(situation_frame_path):
            os.makedirs(situation_frame_path)
            self.hdfs_client.makedirs(hdfs_situation_frame_path, 777)

        frame_dst_path = situation_frame_path + "/image_{}.jpg".format(frame_id)
        hdfs_frame_dst_path = hdfs_situation_frame_path + "/image_{}.jpg".format(frame_id)
        # 使用cv2.imwrite保存图像帧到本地目录
        cv2.imwrite(frame_dst_path, frame)
        # 使用 HDFS 客户端将图像帧写入 HDFS
        self.hdfs_client.write(hdfs_frame_dst_path, cv2.imencode('.jpg', frame)[1].tobytes(), overwrite=True)


if __name__ == '__main__':
    process = VideoProcessor()
    # process.visualize_face_cluster.forward(process.hdfs_client, 17)
    process.forward(45, '/home/shared/video/45/hoc_crowded_trim.mp4')

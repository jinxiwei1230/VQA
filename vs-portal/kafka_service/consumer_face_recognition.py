import json
import os
import sys
import threading
import cv2
import numpy as np
from hdfs import InsecureClient
from PIL import ImageFont

sys.path.append('/home/zhangyuxuan-23/vs-portal')
from kafka_service.producer import Kafka_producer
import common_utils
import vs_common
from face_detect import FaceRecognition
from kafka import KafkaConsumer

bootstrapServers = ['10.92.64.241:9092']

class face_recognition_consumer:
    def __init__(self, groupId):
        threading.Thread.__init__(self)
        self.consumer = KafkaConsumer(
            group_id=groupId,
            bootstrap_servers=bootstrapServers,
            enable_auto_commit=True, # 如果设置为True可能会重复消费，因为此时消费者是完成该项任务之后再进行commit
            max_poll_interval_ms=86400000, # 增加最大的消费时间，防止消费者处理时间太长就被自动挂起不消费了
            api_version=(3, 2, 0),
            max_poll_records=30,
            session_timeout_ms=100000
        )
        self.face_recognition = FaceRecognition()
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')
        self.producer_face_cluster = Kafka_producer('face_cluster')

    def subscribe_topic(self, topic):
        self.consumer.subscribe(topic)

    def consume_data(self):
        print("人脸识别消费者启动")

        for message in self.consumer:
            try:
                # self.consumer.poll()  for循环每次取监听partion的一条信息，因此不用再poll
                # self.consumer.commit()
                obj = json.loads(message.value)
                id = str(obj['movie_id'])
                scene_id = str(obj['scene_id'])
                print("正在处理场景", scene_id)


                # TODO:太多小文件写入速度太慢，是否能直接读入内存进行处理
                frame_output_dir = vs_common.local_result_store_dir.format(id) + '/origin'
                hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(id) + '/origin/' + scene_id
                # 根据movie_id和scene_id 一次性读取所有帧并存入本地
                if not os.path.exists(frame_output_dir):
                    os.makedirs(frame_output_dir)
                self.hdfs_client.download(hdfs_frame_output_dir, frame_output_dir, overwrite=True, n_threads=3)


                situation = scene_id
                unknown_feature_matrix = []
                # 特征行号 id => (situation_id, frame_id, bbox)
                rowid_to_face_dict = {}
                # name => [{situation_id, frame_id, bbox, feature}]
                name_to_face_dict = {}
                # frame_name => [bbox]
                frame_to_bboxes = {}
                # frame_id => situation_id

                # 读取数据进行人脸识别
                for file_name in os.listdir(frame_output_dir + '/' + scene_id):
                    frame = cv2.imread(frame_output_dir + '/' + scene_id + '/' + file_name)
                    frame_id = file_name.split('_')[1].split('.')[0]
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]
                    font_style = self.get_font_style(frame_width, frame_height)
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
                # 存中间结果给cluster用
                print(456)
                self.save_face_recognition_temp(id, scene_id, unknown_feature_matrix, name_to_face_dict, rowid_to_face_dict, frame_to_bboxes)
                # 完成之后增加数据库中已经处理过的situation数量
                # common_utils.updateFaceRecogSituationNum(id)
                common_utils.updateFaceRecogSituationNum(id)

                print(scene_id, "已经完成人脸识别")

                situation_nums = common_utils.getProcessRecordById(id)[13]
                # processed_num = common_utils.getProcessRecordById(id)[14]
                processed_num = common_utils.getFaceRecogProcessed(id)

                # 如果已经处理的situation数量等于所有的，发送消息给人脸聚类topic
                if situation_nums == processed_num:
                    message_cluster = {"movie_id": id}
                    self.producer_face_cluster.sendMessage(message_cluster, partition=int(id) % 5)
                    print("发送消息给人脸聚类producer")
                    # 删除处理后在redis中的缓存，防止一直存储爆缓存
                    common_utils.delRedisinfo(id)
            except Exception as e:
                common_utils.updateProcessRecordById(id, vs_common.FAILED,
                                                     '[ALG] FACE_RECOGNITION_CONSUMER failed')

    def save_face_recognition_temp(self, id, scene_id, unknown_feature_matrix, name_to_face_dict, rowid_to_face_dict, frame_to_bboxes):
        name_to_face_dict_dir = vs_common.hdfs_result_store_path.format(id) + '/process/temp/name_to_face_dict'
        rowid_to_face_dict_dir = vs_common.hdfs_result_store_path.format(
            id) + '/process/temp/rowid_to_face_dict'
        unknown_feature_matrix_dir = vs_common.hdfs_result_store_path.format(
            id) + '/process/temp/unknown_feature_matrix'
        frame_to_bboxes_dir = vs_common.hdfs_result_store_path.format(
            id) + '/process/temp/frame_to_bboxes'

        if not os.path.exists(name_to_face_dict_dir):
            self.hdfs_client.makedirs(name_to_face_dict_dir, 777)
        if not os.path.exists(rowid_to_face_dict_dir):
            self.hdfs_client.makedirs(rowid_to_face_dict_dir, 777)
        if not os.path.exists(unknown_feature_matrix_dir):
            self.hdfs_client.makedirs(unknown_feature_matrix_dir, 777)
        if not os.path.exists(unknown_feature_matrix_dir):
            self.hdfs_client.makedirs(frame_to_bboxes_dir, 777)

        temp = {"0": np.array(unknown_feature_matrix).tolist()}

        self.hdfs_client.write(unknown_feature_matrix_dir + '/{}.json'.format(scene_id), json.dumps(temp).encode(), overwrite=True)
        self.hdfs_client.write(name_to_face_dict_dir + '/{}.json'.format(scene_id), json.dumps(name_to_face_dict).encode(), overwrite=True)
        self.hdfs_client.write(rowid_to_face_dict_dir + '/{}.json'.format(scene_id), json.dumps(rowid_to_face_dict).encode(), overwrite=True)
        self.hdfs_client.write(frame_to_bboxes_dir + '/{}.json'.format(scene_id),
                               json.dumps(frame_to_bboxes).encode(), overwrite=True)

    def get_font_style(self, frame_width, frame_height):
        frame_max = max(frame_height, frame_width)
        # 根据分辨率设置字体大小
        font_size = 55 if frame_max > 1500 else 35 if frame_max < 960 else 45
        #  定义打印文字的字体
        fontStyle = ImageFont.truetype("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc", font_size, encoding="utf-8")
        return fontStyle


if __name__ == '__main__':

    consumer = face_recognition_consumer('face_recognition_group')
    consumer.subscribe_topic('face_recognition')
    consumer.consume_data()









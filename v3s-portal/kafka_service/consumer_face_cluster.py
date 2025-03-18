import json
import os
import sys
import threading
import time
from hdfs import InsecureClient
from PIL import ImageFont
sys.path.append('/home/zhangyuxuan-23/vs-portal')
from face_feature_visualization import FaceFeatureVisualization
import common_utils
from face_cluster import FaceCluster
from kafka_service.producer import Kafka_producer
import vs_common
from kafka import KafkaConsumer

bootstrapServers = ['10.92.64.241:9092']

class face_cluster_consumer:
    def __init__(self, groupId):
        threading.Thread.__init__(self)
        self.consumer = KafkaConsumer(
            group_id=groupId,
            bootstrap_servers=bootstrapServers,
            enable_auto_commit=True, # 如果设置为True可能会重复消费，因为此时消费者是完成该项任务之后再进行commit
            max_poll_interval_ms=86400000 # 增加最大的消费时间，防止消费者处理时间太长就被自动挂起不消费了
        )
        self.face_cluster = FaceCluster()
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')
        self.visualize_face_cluster = FaceFeatureVisualization()
        self.producer_relation_extraction = Kafka_producer('relation_extraction')

    def subscribe_topic(self, topic):
        self.consumer.subscribe(topic)

    def consume_data(self):
        print("人脸聚类消费者启动")

        for message in self.consumer:
            try:
                # self.consumer.poll(timeout_ms=3000, max_records=10)
                obj = json.loads(message.value)
                id = str(obj['movie_id'])
                print(id)

                # 将所有的中间数据下载到本地

                frame_output_dir = vs_common.local_result_store_dir.format(id) + '/temp'

                hdfs_unknown_feature_matrix_dir = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/unknown_feature_matrix'
                hdfs_rowid_to_face_dict_dir = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/rowid_to_face_dict'
                hdfs_name_to_face_dict_dir = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/name_to_face_dict'
                frame_to_bboxes_dir = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/frame_to_bboxes'


                if not os.path.exists(frame_output_dir):
                    os.makedirs(frame_output_dir)
                self.hdfs_client.download(hdfs_unknown_feature_matrix_dir, frame_output_dir, overwrite=True)
                self.hdfs_client.download(hdfs_rowid_to_face_dict_dir, frame_output_dir, overwrite=True)
                self.hdfs_client.download(hdfs_name_to_face_dict_dir, frame_output_dir, overwrite=True)
                self.hdfs_client.download(frame_to_bboxes_dir, frame_output_dir, overwrite=True)

                name_to_face_dict = {}
                rowid_to_face_dict = {}
                unknown_feature_matrix = []
                frame_to_bboxes = {}

                unknown_feature_matrix_list = os.listdir(frame_output_dir + '/unknown_feature_matrix')
                unknown_feature_matrix_list.sort()

                rowid_to_face_dict_list = os.listdir(frame_output_dir + '/rowid_to_face_dict')
                rowid_to_face_dict_list.sort()

                name_to_face_dict_list = os.listdir(frame_output_dir + '/name_to_face_dict')
                name_to_face_dict_list.sort()

                frame_to_bboxes_list = os.listdir(frame_output_dir + '/frame_to_bboxes')
                frame_to_bboxes_list.sort()

                for file_name in unknown_feature_matrix_list:
                    with open(frame_output_dir + '/unknown_feature_matrix/' + file_name) as f:
                        temp_unknown_feature_matrix = json.load(f)['0']
                        for each in temp_unknown_feature_matrix:
                            unknown_feature_matrix.append(each)
                for file_name in rowid_to_face_dict_list:
                    with open(frame_output_dir + '/rowid_to_face_dict/' + file_name) as f:
                        temp_rowid_to_face_dict = json.load(f)
                        # 此时的temp_rowid_to_face_dict的keys为'0','1'....对不上，所以要更新keys
                        temp_rowid_to_face_dict_new = {}
                        have_frames_num = len(rowid_to_face_dict)
                        for k, v in temp_rowid_to_face_dict.items():
                            new_k = int(k) + have_frames_num
                            temp_rowid_to_face_dict_new[str(new_k)] = v
                        rowid_to_face_dict.update(temp_rowid_to_face_dict_new)
                for file_name in name_to_face_dict_list:
                    with open(frame_output_dir + '/name_to_face_dict/' + file_name) as f:
                        temp_name_to_face_dict = json.load(f)
                        name_to_face_dict.update(temp_name_to_face_dict)
                for file_name in frame_to_bboxes_list:
                    with open(frame_output_dir + '/frame_to_bboxes/' + file_name) as f:
                        temp_frame_to_bboxes = json.load(f)
                        frame_to_bboxes.update(temp_frame_to_bboxes)

                frame_width = common_utils.getProcessRecordById(id)[9]
                frame_height = common_utils.getProcessRecordById(id)[10]
                font_style = self.get_font_style(frame_width, frame_height)
                label2idx = self.face_cluster.forward(unknown_feature_matrix)
                self.face_cluster.save_cluster_result(
                    id,
                    label2idx,
                    name_to_face_dict,
                    rowid_to_face_dict,
                    unknown_feature_matrix,
                    font_style,
                    self.hdfs_client
                )

                # 保存到hdfs, 关系抽取再从里面下载
                frame_to_bboxes_path = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/bbox_dict.json'
                self.hdfs_client.write(frame_to_bboxes_path,
                                       json.dumps(frame_to_bboxes).encode(), overwrite=True)


                frame_to_bboxes_path = vs_common.local_result_store_dir.format(id) + '/process/face/bbox_dict.json'
                with open(frame_to_bboxes_path, 'w') as file:
                    json.dump(frame_to_bboxes, file)

                self.visualize_face_cluster.forward(self.hdfs_client, id)


                # 总类别数，传给关系识别作为最大person
                keys_len = len(list(label2idx.keys()))

                end = time.time()

                print("视频{}号在{}时间处理完毕".format(id, end))

                # 发送消息给关系抽取，后续应该还有结合文本和音频，所以这个topic是必要的

                message_relation_extraction = {}
                message_relation_extraction['movie_id'] = id
                message_relation_extraction['max_person'] = keys_len

                self.producer_relation_extraction.sendMessage(message_relation_extraction, partition=int(id) % 5)
            except Exception as e:
                common_utils.updateProcessRecordById(id, vs_common.FAILED,
                                                     '[ALG] FACE_CLUSTER_CONSUMER failed')



    def get_font_style(self, frame_width, frame_height):
        frame_max = max(frame_height, frame_width)
        # 根据分辨率设置字体大小
        font_size = 55 if frame_max > 1500 else 35 if frame_max < 960 else 45
        #  定义打印文字的字体
        fontStyle = ImageFont.truetype("/usr/share/fonts/wqy-microhei/wqy-microhei.ttc", font_size, encoding="utf-8")
        return fontStyle


if __name__ == '__main__':

    consumer = face_cluster_consumer('face_cluster_group')
    consumer.subscribe_topic('face_cluster')
    consumer.consume_data()









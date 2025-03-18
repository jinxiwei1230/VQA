import json
import os
import sys
import time
from hdfs import InsecureClient
from kafka import KafkaConsumer

sys.path.append('/home/zhangyuxuan-23/vs-portal')
import common_utils
import vs_common
from relation_extraction import RelationExtraction

bootstrapServers = ['10.92.64.241:9092']

class consumer_rel_extraction():
    def __init__(self, groupId):
        self.consumer = KafkaConsumer(
            group_id=groupId,
            bootstrap_servers=bootstrapServers,
            enable_auto_commit=True, # 如果设置为True可能会重复消费，因为此时消费者是完成该项任务之后再进行commit
            max_poll_interval_ms=86400000 # 增加最大的消费时间，防止消费者处理时间太长就被自动挂起不消费了
        )
        self.relation_extraction = RelationExtraction()
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')

    def subscribe_topic(self, topic):
        self.consumer.subscribe(topic)

    def consume_data(self):
        print("关系识别消费者启动")

        for message in self.consumer:
            try:
                start = time.time()
                # self.consumer.poll(timeout_ms=3000, max_records=10)
                obj = json.loads(message.value)
                id = str(obj['movie_id'])
                max_person = int(obj['max_person'])
                print("{}号视频开始关系抽取".format(id))

                # 下载中间数据

                bbox_dict_local_path = vs_common.local_result_store_dir.format(id) + '/process/face/bbox_dict.json'

                if not os.path.exists(vs_common.local_result_store_dir.format(id) + '/process/face'):
                    os.makedirs(vs_common.local_result_store_dir.format(id) + '/process/face')

                frame_to_bboxes_path = vs_common.hdfs_result_store_path.format(
                    id) + '/process/temp/bbox_dict.json'
                self.hdfs_client.download(frame_to_bboxes_path, bbox_dict_local_path, overwrite=True)
                bboxes_coordinates = json.load(open(bbox_dict_local_path))
                for _, value in bboxes_coordinates.items():
                    max_person = max(len(value), max_person)


                self.relation_extraction.forward(id, self.hdfs_client, max_person)
                end = time.time()
                print("{}号视频在{}时间完成关系抽取,关系抽取花费时间{}".format(id, end, end-start))
                # 修改状态
                common_utils.updateProcessStateById(id, vs_common.SUCCEED)
            except Exception as e:
                common_utils.updateProcessRecordById(id, vs_common.FAILED,
                                                     '[ALG] REL_EXTRA_CONSUMER failed')

if __name__ == '__main__':

    consumer = consumer_rel_extraction('relation_extraction_group')
    consumer.subscribe_topic('relation_extraction')
    consumer.consume_data()

import json
import os
import sys
import threading
import numpy as np
from hdfs import InsecureClient
from scipy.io import wavfile

sys.path.append('/home/zhangyuxuan-23/vs-portal')
from kafka_service.producer import Kafka_producer
import common_utils
import ASR
import vs_common
from kafka import KafkaConsumer

bootstrapServers = ['10.92.64.241:9092']

class consumer_asr:
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
        self.hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')
        self.producer_face_cluster = Kafka_producer('ASR')

    def subscribe_topic(self, topic):
        self.consumer.subscribe(topic)

    def consume_data(self):
        print("ASR消费者启动")

        for message in self.consumer:
            try:
                obj = json.loads(message.value)
                id = str(obj['movie_id'])
                audio_path = obj['audio_path']
                rate = obj['rate']
                print("正在处理场视频", id)

                # 从hdfs下载音频的numpy至本地机器
                local_audio_numpy_output_path = vs_common.local_video_store_dir.format(id) + '/audio/'
                try:
                    self.hdfs_client.download(audio_path, local_audio_numpy_output_path, overwrite=True)
                except Exception as e:
                    common_utils.updateProcessRecordById(id, vs_common.FAILED,
                                                         '[ALG] download ASR_NUMPY failed')


                with open(local_audio_numpy_output_path + "audio_numpy_file.txt", 'r') as audio_numpy:
                    f = np.loadtxt(audio_numpy)
                    wavfile.write(local_audio_numpy_output_path + "new_audio_file.wav", rate=rate, data=f)
                # 转写音频并生成json文件，保存至hdfs文件系统：这一步用于展示
                api = ASR.RequestApi(appid=vs_common.appid, secret_key=vs_common.secret_key,
                                     upload_file_path=local_audio_numpy_output_path + "new_audio_file.wav")
                result = api.all_api_request()
                result_json = json.loads(result['data'])
                subtitle_path = vs_common.hdfs_result_store_path.format(id) + "/subtitle/"
                if not os.path.exists(subtitle_path):
                    self.hdfs_client.makedirs(subtitle_path, 777)
                self.hdfs_client.write(subtitle_path + "asr_result.json", json.dumps(result_json, ensure_ascii=False),
                                       encoding='utf-8', overwrite=True)

                print("字幕提取完成，已保存至{}".format(subtitle_path + "asr_result.json"))
            except Exception as e:
                common_utils.updateProcessRecordById(id, vs_common.FAILED,
                                                     '[ALG] ASR_CONSUMER failed')
            # TODO: 提取文本特征

            # TODO: 通知关系识别算法 文本特征已经准备就绪

    def tt(self):
        print("Hello, world!")
        return None
        say("Bye~")


if __name__ == '__main__':

    consumer = consumer_asr('asr_group')
    consumer.subscribe_topic('asr')
    consumer.consume_data()










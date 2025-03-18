import json
import sys

from kafka import KafkaConsumer

from process import VideoProcessor

sys.path.append('/home/yanch/vs-alg-portal')


class Kafka_consumer:
    def __init__(self, bootstrapServers, kafkaTopic, groupId):
        self.kafkaTopic = kafkaTopic
        self.bootstrapServers = bootstrapServers
        self.groupId = groupId
        self.consumer = KafkaConsumer(
            self.kafkaTopic,
            group_id=self.groupId,
            bootstrap_servers=self.bootstrapServers
        )
        self.processor = VideoProcessor()

    def consume_data(self):
        try:
            for message in self.consumer:
                yield message
        except BaseException as e:
            print(e)


if __name__ == '__main__':
    bootstrapServers = ['10.105.222.250:9092']
    topicStr = 'test'

    print('-' * 20)
    print('消费者')
    print('-' * 20)

    groupId = 'group-yanch'
    consumer = Kafka_consumer(bootstrapServers, topicStr, groupId)
    messages = consumer.consume_data()
    for message in messages:
        obj = json.loads(message.value)
        print(obj)
        consumer.processor.forward(obj['path'])
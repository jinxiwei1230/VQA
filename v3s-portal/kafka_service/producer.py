import json

from kafka import KafkaProducer
from kafka.errors import KafkaError

bootstrapServers = ['10.92.64.241:9092']


class Kafka_producer:
    def __init__(self, kafkaTopic):
        self.bootstrapServers = bootstrapServers
        self.kafkaTopic = kafkaTopic
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrapServers,
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      api_version=(3, 2, 0))

    def sendMessage(self, message, partition):
        try:
            producer = self.producer
            producer.send(topic=self.kafkaTopic, value=message, partition=partition)
            producer.flush()
        except KafkaError as e:
            print(e)




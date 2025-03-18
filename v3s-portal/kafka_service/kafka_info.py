from kafka import KafkaAdminClient
from kafka.admin import NewTopic

# bootstrapServers = ['10.105.222.250:9092']
from kafka.cluster import ClusterMetadata

bootstrapServers = ['10.92.64.241:9092']

class kafka_admin():
    def __init__(self):
        self.admin = KafkaAdminClient(bootstrap_servers=bootstrapServers, api_version=(3, 2, 0))
        self.ClusterMetaData = ClusterMetadata(bootstrap_servers=bootstrapServers)

    def getBrokers(self):
        return self.ClusterMetaData.brokers()

    def getConsumerGroup(self, group_id):
        return self.admin.describe_consumer_groups(group_id), self.admin.list_consumer_group_offsets(group_id)

    def getTopics(self):
        return self.admin.list_topics();

    def create_topic(self, topic_name, num_partitions, replication_factor):
        topic = NewTopic(name=topic_name,
                         num_partitions=num_partitions,
                         replication_factor=replication_factor)
        self.admin.create_topics([topic])

    def deleteTopics(self, topics):
        self.admin.delete_topics(topics)

    def describe_topics(self, topic):
        return self.admin.describe_topics(topic)

    def create_partitions(self, topic_partitions):
        self.admin.create_partitions(topic_partitions)


if __name__ == '__main__':

    admin = kafka_admin()
    #admin.create_partitions({'test':'1'})

    # admin.create_topic('asr', 5, 1)
    # admin.deleteTopics(['face_recognition'])
    # a, b = admin.getConsumerGroup('face_recognition_group')
    print(admin.getBrokers())
    # print(a)
    # # for i in b.keys():
    # #     print(type(i))
    # print(len(b))
    # print(admin.describe_topics(['face_recognition']))
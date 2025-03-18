from kafka_service.producer import Kafka_producer

if __name__ == '__main__':
    for i in range(49, 50):

    # message_cluster = {"movie_id": '89', "max_person": '15'}
        message_cluster = {"id": i}
        #producer_face_cluster = Kafka_producer('face_cluster')
        producer_face_cluster = Kafka_producer('video_input')
        producer_face_cluster.sendMessage(message_cluster, partition=i % 5)


    # frame_id = 'image_5432.jpg'.split('_')[1].split('.')[0]
    # print(frame_id)



    # message_cluster = {"movie_id": '45', "max_person": '15'}
    # # message_cluster = {"id": i}
    # #producer_face_cluster = Kafka_producer('face_cluster')
    # producer_face_cluster = Kafka_producer('video_input')
    # producer_face_cluster.sendMessage(message_cluster, partition=i % 5)
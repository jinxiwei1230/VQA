#!/bin/bash

for i in {1..20}; do
  nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_face_recognition.py &>nohup_input_face.out&
done

nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/gate_consumer.py &>nohup_input_video1.out&
nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/gate_consumer.py &>nohup_input_video2.out&

nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_face_cluster.py &>nohup_face_cluster1.out&
nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_face_cluster.py &>nohup_face_cluster2.out&

nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_rel_extraction.py &>nohup_rel_extraction1.out&
#nohup /home/huyibo-21/miniconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_rel_extraction.py &>nohup_rel_extraction2.out&

nohup /home/zhangyuxuan-23/anaconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_asr.py &>nohup_asr1.out&
#nohup /home/huyibo-21/miniconda3/envs/video/bin/python3.8 -u ./kafka_service/consumer_asr.py &>nohup_asr2.out&
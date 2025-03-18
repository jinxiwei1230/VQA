ps -ef | grep ./kafka_service/consumer_face_recognition.py | grep -v  grep |awk '{print $2}' | xargs kill -9
ps -ef | grep ./kafka_service/consumer_rel_extraction.py | grep -v  grep |awk '{print $2}' | xargs kill -9
ps -ef | grep ./kafka_service/consumer_face_cluster.py | grep -v  grep |awk '{print $2}' | xargs kill -9
ps -ef | grep ./kafka_service/gate_consumer.py | grep -v  grep |awk '{print $2}' | xargs kill -9
ps -ef | grep ./kafka_service/consumer_asr.py | grep -v  grep |awk '{print $2}' | xargs kill -9
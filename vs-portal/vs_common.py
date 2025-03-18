# hdfs config


HDFS_HOST = "http://10.92.64.241:14000"
# hdfs_result_store_path = "/bdap/students/{}/video_system/{}/process_results"
# hdfs_video_store_path = 'bdap/students/{}/video_system/videos'
hdfs_result_store_path = "/bdap/students/video_result/{}"
hdfs_video_store_path = '/bdap/students/{}'

# pretrained model config
# 面部轮廓特征提取模型
face_shape_model_path = "/home/shared/model/shape_predictor_68_face_landmarks.dat"
# 人脸特征提取模型
face_resnet_model_path = "/home/shared/model/dlib_face_recognition_resnet_model_v1.dat"
resnet101_roi_model_path = "/home/shared/model/resnet101-5d3b4d8f.pth"
# 关系识别模型 todo: 模型名可以规范
SRModel_path = "/home/shared/model/model-0019.pth"

# 人脸库路径
known_people_dir = "/home/shared/known_people"
# local store dir
# local_result_store_dir = '/home/wanghaorui-22/video_system/{}/{}/process_results'
# local_video_store_dir = '/home/wanghaorui-22/video_system/{}/{}/video'
local_result_store_dir = '/home/disk2/dachuang1-23/kafka_result/{}/process_results'
local_video_store_dir = '/home/disk2/dachuang1-23/kafka_result/{}'

# 目标检测label
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

# mysql配置
mysql_conf = {
        "host": '10.92.64.242',
        "port": 3306,
        "user": 'video_sys',
        "password": 'video_sys_dssc',
        "db": 'video_sys',
        "charset": "utf8",
    }

# 处理状态
BEFORE_PROCESSING = 1
IN_PREPROCESSING = 2
IN_RELATION_EXTRACTING = 3
SUCCEED = 4
FAILED = 5
STOP = 6
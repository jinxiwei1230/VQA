# videosystem-algorithm
视频人物关系图构建系统--算法部分

```
-- message
     |- gate_consumer.py 接收业务端发来的处理消息，做好处理的前期工作（删除原文件夹、下载视频到本地等），调用process.py处理
     |- test_consumer.py 接收业务端发来的处理消息，调用process.py处理
-- common_utils.py 工具类
-- vs_knn.py 人脸聚类需要的knn算法
-- vs_common.py 常量存放
-- face_cluster.py 人脸聚类相关算法
-- face_detect.py 人脸识别相关算法
-- face_feature_visualization.py 人脸特征可视化相关算法
-- object_det.py 目标检测相关算法
-- relation_extraction.py 关系抽取相关算法
-- GRRN.py 关系抽取模型
-- resnet_roi.py ResNet模型
-- RIG.py RIG模型
-- SRDataset_video_ver.py Dataset数据集构建
-- relation_extraction_utils 关系抽取相关工具类
-- process.py 综合的处理流程，分镜-目标检测-人脸识别-人脸聚类-关系抽取-构建关系图-人脸特征可视化-持久化存储 
-- 字幕文件保存位置： /bdap/students/video_result_parallel/49/subtitle/asr_result.json
```
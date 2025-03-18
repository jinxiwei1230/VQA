from mtcnn import MTCNN
import cv2

# 加载视频帧或图片
image = cv2.imread('/home/disk2/dachuang1-23/results/data/9873067604/9873067604_frm_14.png')

# 初始化 MTCNN 模型
detector = MTCNN()

# 检测人脸
results = detector.detect_faces(image)

for result in results:
    # 获取边框和关键点
    bounding_box = result['box']
    keypoints = result['keypoints']

    # 在图像上绘制边框
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    # 绘制关键点
    for key, point in keypoints.items():
        cv2.circle(image, point, 2, (255, 0, 0), -1)

# 保存带有边框和关键点的图像
cv2.imwrite('/home/disk2/dachuang1-23/results/data/9873067604/000.png', image)

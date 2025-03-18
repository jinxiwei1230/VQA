import json

import numpy as np
import dlib
import cv2
import os

import vs_common


class FaceFeatureVisualization:
    def __init__(self):
        self.FACIAL_LANDMARKS_68_IDXS = dict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])
        # 加载人脸检测与关键点定位
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(vs_common.face_shape_model_path)
        self.hdfs_feature_output_dir = vs_common.hdfs_result_store_path + '/process/for_frontend/face_feature'
        self.hdfs_face_cluster_output_dir = vs_common.hdfs_result_store_path + '/process/for_frontend/face_cluster'
        self.image_dir = vs_common.local_result_store_dir + '/process/face'
        self.feature_output_dir = vs_common.local_result_store_dir + '/process/for_frontend/face_feature'
        self.face_cluster_output_dir = vs_common.local_result_store_dir + '/process/for_frontend/face_cluster'
        self.bboxes_list = vs_common.local_result_store_dir + '/process/face/bbox_dict.json'

    def shape_to_np(self, shape, dtype="int"):
        # 创建68*2
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # 遍历每一个关键点
        # 得到坐标
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)  # 第i个关键点的横纵坐标。
        return coords

    def visualize_facial_landmarks(self, image, shape, colors=None, alpha=0.75):
        # 创建两个copy
        # overlay and one for the final output image
        overlay = image.copy()
        output = image.copy()
        # 设置一些颜色区域
        if colors is None:
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                      (168, 100, 168), (158, 163, 32),
                      (163, 38, 32), (180, 42, 220)]
        # 遍历每一个区域
        for (i, name) in enumerate(self.FACIAL_LANDMARKS_68_IDXS.keys()):
            # 得到每一个点的坐标
            (j, k) = self.FACIAL_LANDMARKS_68_IDXS[name]
            pts = shape[j:k]
            # 检查位置
            if name == "jaw":
                # 用线条连起来
                for line in range(1, len(pts)):
                    ptA = tuple(pts[line - 1])
                    ptB = tuple(pts[line])
                    cv2.line(overlay, ptA, ptB, colors[i], 2)
            # 计算凸包
            else:
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        # 叠加在原图上，可以指定比例
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output

    def forward(self, hdfs_client, id):
        hdfs_feature_output_dir = self.hdfs_feature_output_dir.format(id)
        hdfs_face_cluster_output_dir = self.hdfs_face_cluster_output_dir.format(id)
        image_dir = self.image_dir.format(id)
        feature_output_dir = self.feature_output_dir.format(id)
        face_cluster_output_dir = self.face_cluster_output_dir.format(id)
        bboxes_list = self.bboxes_list.format(id)

        if not os.path.exists(feature_output_dir):
            os.makedirs(feature_output_dir)
            hdfs_client.makedirs(hdfs_feature_output_dir, 777)
        if not os.path.exists(face_cluster_output_dir):
            os.makedirs(face_cluster_output_dir)
            hdfs_client.makedirs(hdfs_face_cluster_output_dir, 777)

        bboxes_coordinates = json.load(open(bboxes_list))
        for root, dirs, files in os.walk(image_dir):
            for single_dir in dirs:
                person_id = single_dir.split('_')[1]
                files = os.listdir(os.path.join(image_dir, single_dir))
                files.sort(key=lambda element: int(element.split('.')[0].split('_')[1]), reverse=False)

                # find the frame with the least person nu,ber
                files.sort(key=lambda element: int(len(bboxes_coordinates[element.split('.')[0].split('_')[1]])),
                           reverse=False)

                for file in files:
                    # 读取输入数据，预处理
                    image = cv2.imread(os.path.join(image_dir, single_dir, file))
                    (h, w) = image.shape[:2]
                    width = 500
                    r = width / float(w)
                    dim = (width, int(h * r))
                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # 人脸检测
                    rects = self.detector(gray, 1)
                    if len(rects) > 0:
                        # 遍历检测到的框
                        output = image.copy()
                        for (_, rect) in enumerate(rects):
                            # 对人脸框进行关键点定位
                            shape = self.predictor(gray, rect)
                            shape = self.shape_to_np(shape)

                            # 遍历每一个部分
                            for (name, (i, j)) in self.FACIAL_LANDMARKS_68_IDXS.items():
                                clone = image.copy()
                                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (0, 0, 255), 2)

                                # 根据位置画点
                                for (x, y) in shape[i:j]:
                                    cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

                                # 提取ROI区域
                                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                                x = max(x, 0)
                                y = max(y, 0)
                                roi = image[y:y + h, x:x + w]
                                (h, w) = roi.shape[:2]
                                width = 250
                                r = width / float(w)
                                dim = (width, int(h * r))
                                cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
                                output = self.visualize_facial_landmarks(output, shape)

                        # 展示所有区域
                        cv2.imwrite(face_cluster_output_dir + '/{}.jpg'.format(person_id),
                                    np.array(image))
                        hdfs_client.write(hdfs_face_cluster_output_dir + '/{}.jpg'.format(person_id),
                                          cv2.imencode('.jpg', np.array(image))[1].tobytes(), overwrite=True)

                        cv2.imwrite(feature_output_dir + '/{}.jpg'.format(person_id),
                                    np.array(output))
                        hdfs_client.write(hdfs_feature_output_dir + '/{}.jpg'.format(person_id),
                                          cv2.imencode('.jpg', np.array(output))[1].tobytes(), overwrite=True)
                        break

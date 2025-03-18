import dlib
import os
import cv2
import numpy as np
import vs_common
from PIL import Image, ImageDraw


class FaceRecognition:
    def __init__(self):
        # 加载人脸检测所需模型
        # 人脸框检测器
        self.detector = dlib.get_frontal_face_detector()  # 加载检测器
        # 面部轮廓特征提取模型
        self.face_shape_model = dlib.shape_predictor(vs_common.face_shape_model_path)
        # 人脸特征提取模型
        self.face_resnet_model = dlib.face_recognition_model_v1(vs_common.face_resnet_model_path)

        self.local_face_store_dir = vs_common.local_result_store_dir + '/process/face'
        self.hdfs_face_store_dir = vs_common.hdfs_result_store_path + '/process/face'

        # 加载人脸库
        self.known_list = {}
        # for path in os.listdir(vs_common.known_people_dir):
        #     img = cv2.imread(vs_common.known_people_dir + "/" + path)
        #     name = path.split('.')[0]
        #
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     det_img = self.detector(gray, 0)
        #     shape = self.face_shape_model(img, det_img[0])
        #     # 可优化 可以存一个Json 没必要每次生成 定时任务更新
        #     # 参数3 为采样次数 默认为1 参数4为 padding比例
        #     face_encode = self.face_resnet_model.compute_face_descriptor(img, shape, 10, 0.15)
        #     self.known_list[name] = face_encode

    def forward(self, id, situation_id, frame_id, frame, fontStyle, name_to_face_dict, unknown_feature_matrix,
                rowid_to_face_dict, hdfs_client, frame_to_bboxes):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 将彩色帧转换为灰度图像
        det_face = self.detector(gray, 1) # 在灰度图像中检测人脸，并返回检测到的人脸框。 # 参数2，表示是否将原始图像进行放大 这里放大1倍
        face_id = 0  # denote the index_id of a detected face in the same frame
        frame_to_bboxes[frame_id] = []
        for face in det_face:
            shape = self.face_shape_model(frame, face) # 获取人脸的形状
            face_encode = self.face_resnet_model.compute_face_descriptor(frame, shape, 10, 0.15) # 计算人脸描述符
            match_name = "unknown"
            score = 1
            for key, feature in self.known_list.items():
                eu_dis = self.Eu_distance(face_encode, feature)
                if eu_dis < 0.45 and eu_dis < score:
                    match_name = key
                    score = eu_dis

            face_bbox = [face.left(), face.top(), face.right(), face.bottom()]
            info_dict = {'situation_id': situation_id, 'frame_id': frame_id, 'bbox': face_bbox, 'face_id': face_id}
            face_id += 1
            frame_to_bboxes[frame_id].append(face_bbox)

            if match_name != "unknown":
                # 对于每一个检测出的人脸都要单独存图片
                im = Image.fromarray(frame.astype(np.uint8))
                draw = ImageDraw.Draw(im)
                draw.rectangle(face_bbox, width=3, outline=(0, 0, 255))  # 画框
                draw.text((face_bbox[0], face_bbox[3]), match_name, (0, 255, 0), font=fontStyle)  # 写入label

                # 持久化存储
                face_det_output_path = self.local_face_store_dir.format(id) + '/{}'.format(match_name)
                hdfs_face_det_output_path = self.hdfs_face_store_dir.format(id) + '/{}'.format(match_name)
                if not os.path.exists(face_det_output_path):
                    os.makedirs(face_det_output_path)
                    hdfs_client.makedirs(hdfs_face_det_output_path, 777)
                marked_face_save_path = face_det_output_path + "/image_{}.jpg".format(frame_id)
                hdfs_marked_face_save_path = hdfs_face_det_output_path + "/image_{}.jpg".format(frame_id)
                cv2.imwrite(marked_face_save_path, np.array(im))
                hdfs_client.write(hdfs_marked_face_save_path, cv2.imencode('.jpg', np.array(im))[1].tobytes(), overwrite=True)

                # 已识别的人脸 feature 存到内存中，后续写入json
                if match_name not in name_to_face_dict:
                    name_to_face_dict[match_name] = {}
                if situation_id not in name_to_face_dict[match_name]:
                    name_to_face_dict[match_name][situation_id] = []
                info_dict['feature'] = np.array(face_encode).tolist()
                name_to_face_dict[match_name][situation_id].append(info_dict)
            else:
                # 未知的人脸信息先存储，后续再聚类处理
                feat_id = len(unknown_feature_matrix)
                unknown_feature_matrix.append(face_encode)
                rowid_to_face_dict[feat_id] = info_dict

    def Eu_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=2)

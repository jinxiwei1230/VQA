import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

import vs_common


class ObjDetector:
    def __init__(self):
        # 加载目标检测所需模型
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval()

        # 导入ResNet50的预训练模型，用于提取物体图像的特征
        self.resnet_50_model = models.resnet50(pretrained=True)
        # 重新定义最后一层
        self.resnet_50_model.fc = nn.Linear(2048, 256)
        # 将二维tensor初始化为单位矩阵
        torch.nn.init.eye(self.resnet_50_model.fc.weight)
        for param in self.resnet_50_model.parameters():
            param.requires_grad = False

        #  定义串联多个图片变换的操作
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 缩放
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor()]  # 转换成Tensor
        )
        self.local_obj_store_dir = vs_common.local_result_store_dir + '/process/objdet'
        self.hdfs_obj_store_dir = vs_common.hdfs_result_store_path + '/process/objdet'

    def forward(self, id, situation_id, frame_id, frame, fontStyle, hdfs_client):
        # 目标检测
        frame_obj_feat_list = []

        obj_det_output_path = self.local_obj_store_dir.format(id) + '/{}'.format(situation_id)
        hdfs_obj_det_output_path = self.hdfs_obj_store_dir.format(id) + '/{}'.format(situation_id)
        if not os.path.exists(obj_det_output_path):
            os.makedirs(obj_det_output_path)
            hdfs_client.makedirs(hdfs_obj_det_output_path, 777)

        # 将frame转换为PIL图像对象（Image）
        im = Image.fromarray(frame.astype(np.uint8)) if isinstance(frame, np.ndarray) else frame
        # 创建一个ImageDraw对象以在图像上绘制标注
        draw = ImageDraw.Draw(im)

        obj_det_res = self.yolo_model(frame) # 返回目标检测结果
        for result in obj_det_res.tolist():
            pred = result.pred
            pred_n = torch.tensor([item.cpu().detach().numpy() for item in pred]).cuda()
            if pred_n.shape[0]:
                for *box, conf, cls in pred[0]:  # xyxy, confidence, class
                    # 置信度过低的不要
                    if conf.item() < 0.3:
                        continue
                    label = '%s %.2f ' % (vs_common.names[int(cls)], conf.item())  # 查找 label
                    draw.rectangle(box, width=3, outline=(255, 0, 0))  # 画框
                    draw.text((box[0], box[3]), label, (255, 0, 0), font=fontStyle)  # 写入label

                    # resnet提取特征
                    # obj_array = np.array(frame)[(int)(box[1]):(int)(box[3]), (int)(box[0]):(int)(box[2])]
                    # obj_tensor = self.transform(Image.fromarray(obj_array))
                    # x = Variable(torch.unsqueeze(obj_tensor, dim=0).float(), requires_grad=False)
                    # obj_feat = self.resnet_50_model(x).data.numpy()

                    # obj_dict = {}
                    # box = [i.item() for i in box]
                    # obj_dict['bbox'] = box
                    # obj_dict['label'] = label
                    # obj_dict['feature'] = obj_feat.tolist()
                    # frame_obj_feat_list.append(obj_dict)

        # 保存标记图像
        marked_image_save_path = obj_det_output_path + "/image_{}.jpg".format(frame_id)
        hdfs_marked_image_save_path = hdfs_obj_det_output_path + "/image_{}.jpg".format(frame_id)
        cv2.imwrite(marked_image_save_path, np.array(im))
        hdfs_client.write(hdfs_marked_image_save_path, cv2.imencode('.jpg', np.array(im))[1].tobytes(), overwrite=True)

        # 保存目标信息 frame_obj_feat_list
        filename = obj_det_output_path + '/image_{}.json'.format(frame_id)
        with open(filename, 'w') as file:
            json.dump(frame_obj_feat_list, file)

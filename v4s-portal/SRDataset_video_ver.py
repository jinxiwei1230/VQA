import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from torchvision import datasets, transforms, models


class SRDataset_video_ver(data.Dataset):
    def __init__(self, max_person, image_dir, bboxes_list, image_size,
                 input_transform=None):
        super(SRDataset_video_ver, self).__init__()
        self.max_person = max_person
        self.image_dir = image_dir
        self.image_size = image_size
        self.input_transform = input_transform
        self.names = []
        self.bboxes = dict()
        self.character_relations_mask = dict()
        self.character_relations = dict()

        # image_dir: /image_{frame_id}.jpg
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                info = {'name': f}
                self.names.append(info)

        bboxes_coordinates = json.load(open(bboxes_list))
        #
        # for _, value in bboxes_coordinates.items():
        #     self.max_person = max(len(value), self.max_person)
        #
        # print(self.max_person)
        for key, value in bboxes_coordinates.items():
            self.bboxes[key] = []
            self.character_relations_mask[key] = np.zeros((self.max_person, self.max_person),
                                                          dtype=np.int32)
            frame_person_count = 0
            # print("value",value)
            for bbox_coordinate in value:
                self.bboxes[key].append(bbox_coordinate)
                frame_person_count += 1
            
            # print("maxperson",self.max_person)
            # print("frame_person_count",frame_person_count)

            for i in range(frame_person_count):
                for j in range(frame_person_count):
                    # print("i=",i,"j=",j)
                    # print("key=",key)
                    self.character_relations_mask[key][i][j] = 1

    def __getitem__(self, index):
        image_name = str(self.names[index]['name']).split('_')[1].split('.')[0]
        img = Image.open(os.path.join(self.image_dir, self.names[index]['name'])).convert('RGB')  # convert gray to rgb
        (w, h) = img.size
        full_mask = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        image_bboxes = np.zeros((self.max_person, 4), dtype=np.float32)

        try:
            bbox_np = np.array(self.bboxes[image_name])

            image_bboxes[:, 0] = 0
            image_bboxes[:, 1] = 0
            image_bboxes[:, 2] = w - 1
            image_bboxes[:, 3] = h - 1

            bbox_num = len(self.bboxes[image_name])
            if bbox_num == 0:
                img, image_bboxes = self.input_transform(img, torch.from_numpy(image_bboxes))
                return image_name, img, image_bboxes, full_mask
            image_bboxes[0:bbox_num, :] = bbox_np[:, :]
            image_bboxes = torch.from_numpy(image_bboxes)

            if self.input_transform:
                img, image_bboxes = self.input_transform(img, image_bboxes)
        except Exception as e:
            image_bboxes = torch.from_numpy(image_bboxes).long()
            img, image_bboxes = self.input_transform(img, image_bboxes)

        try:
            full_mask = np.logical_or(self.character_relations_mask[str(image_name)],
                                      self.character_relations_mask[str(image_name)].T)
            full_mask = torch.from_numpy(full_mask).long()
        except KeyError:
            full_mask = torch.from_numpy(full_mask).long()

        if not isinstance(img, torch.Tensor):
            print(image_name)
            print('Not Tensor' + str(type(self.input_transform)))

        return image_name, img, image_bboxes, full_mask

    def __len__(self):
        return len(self.names)

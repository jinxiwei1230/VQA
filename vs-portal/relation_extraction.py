import json
import os
import sys
import numpy
import torch
from torch.autograd import Variable

sys.path.append('/home/zhangyuxuan-23/vs-portal')
import RIG
import SRDataset_video_ver
import vs_common
from relation_extraction_utils import transforms


class RelationExtraction:
    def __init__(self):
        # self.max_person = 16
        self.categories = 6

        self.cache_size = 256
        self.image_size = 224
        self.transform_test = transforms.Compose([
            transforms.Resize((self.cache_size, self.cache_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.relation_dict = {'friends': 1, 'friend': 1, 'family': 2, 'couple': 3, 'professional': 4, 'commercial': 5,
                              'no relation': 6}
        self.relation_dict = {value: key for key, value in self.relation_dict.items()}

        self.SRModel = RIG.RIG(num_class=6, hidden_dim=2048, time_step=1, node_num=16)
        self.SRModel.load_state_dict(torch.load(vs_common.SRModel_path,map_location='cpu'), False)


    def forward(self, id, hdfs_client, max_person):
        result = numpy.zeros((1, max_person, max_person, self.categories), dtype=numpy.float32)
        # result = numpy.zeros((1, self.max_person, self.max_person, self.categories), dtype=numpy.float32)

        frame_face_to_cluster_dir = vs_common.hdfs_result_store_path.format(
            id) + '/process/temp/frame_face_to_cluster.json'
        if not os.path.exists(vs_common.local_result_store_dir.format(id) + '/process/face'):
            os.makedirs(vs_common.local_result_store_dir.format(id) + '/process/face')
        hdfs_client.download(frame_face_to_cluster_dir, vs_common.local_result_store_dir.format(id) + '/process/face/', overwrite=True, n_threads=3)
        frame_face_to_cluster = json.load(
            open(vs_common.local_result_store_dir.format(id) + '/process/face/frame_face_to_cluster.json'))

        # TODO:这里又一次将所有帧下载到本地，是否能够规定人脸聚类和关系抽取在同一台机器？
        check_frame_output_dir = vs_common.local_result_store_dir.format(id)
        hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(id) + '/origin/'
        hdfs_client.download(hdfs_frame_output_dir, check_frame_output_dir, overwrite=True, n_threads=5)

        for root, dirs, files in os.walk(vs_common.local_result_store_dir.format(id) + '/origin'):
            dirs.sort(key=lambda element: int(element), reverse=False)
            for d in dirs:
                image_dir = os.path.join(root, d)
                trainset = SRDataset_video_ver.SRDataset_video_ver(max_person=max_person, image_dir=image_dir,
                                                                   bboxes_list=vs_common.local_result_store_dir.format(id) 
                                                                        + '/process/face/bbox_dict.json',
                                                                   image_size=224,
                                                                   input_transform=self.transform_test)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
                                                          num_workers=1,
                                                          worker_init_fn=numpy.random.seed(
                                                              numpy.random.randint(1, 10000)))

                for batch_idx, (frame, img, image_bboxes, full_mask) in enumerate(trainloader):
                    frame, img, image_bboxes, full_mask = frame, Variable(img), Variable(image_bboxes), Variable(
                        full_mask)
                    output = self.SRModel(img, image_bboxes, full_mask)
                    output = output[1]
                    output = output.detach().numpy()
                    # map face_id in frame to clustered person_id in video
                    if not frame_face_to_cluster.__contains__(frame[0]):
                        continue
                    mapping = frame_face_to_cluster[frame[0]]
                    temp_result = numpy.zeros((1, max_person, max_person, self.categories),
                                              dtype=numpy.float32)
                    for i in range(max_person):
                        for j in range(max_person):
                            if mapping.__contains__(str(i)) and mapping.__contains__(str(j)):
                                cluster_i = int(mapping[str(i)].split('_')[1])
                                cluster_j = int(mapping[str(j)].split('_')[1])
                                temp_result[0, cluster_i, cluster_j, :] = output[0, i, j, :]
                    result += temp_result

                relations = []
                for i in range(max_person):
                    for j in range(int(max_person / 2)):
                        if not numpy.count_nonzero(result[0][i][j]) == 0 and i != j:
                            relation = {"value": self.relation_dict[numpy.argmax(result[0][i][j], axis=0)], "source": str(i),
                                        "target": str(j)}
                            relations.append(relation)

                # 本地机器不用再写入了
                relation_output_dir = vs_common.local_result_store_dir.format(id) + '/process/relation'
                hdfs_relation_output_dir = vs_common.hdfs_result_store_path.format(id) + '/process/relation'
                if not os.path.exists(relation_output_dir):
                    os.makedirs(relation_output_dir)
                    hdfs_client.makedirs(hdfs_relation_output_dir, 777)

                # filename = relation_output_dir + '/link{}.json'.format(d)
                # with open(filename, 'w') as file:
                #     json.dump(relations, file)

                hdfs_relation_output_path = hdfs_relation_output_dir + '/link{}.json'.format(d)
                hdfs_client.write(hdfs_relation_output_path, json.dumps(relations).encode(), overwrite=True)

import vs_common
import numpy as np
from tqdm import tqdm
# import infomap
import os
import json
import cv2
from PIL import Image, ImageDraw
from common_utils import Timer
from vs_knn import knn_faiss


class FaceCluster:
    def __init__(self):
        self.local_face_store_dir = vs_common.local_result_store_dir + '/process/face'
        self.hdfs_face_store_dir = vs_common.hdfs_result_store_path + '/process/face'
        self.local_frame_store_dir = vs_common.local_result_store_dir + '/origin'

        self.knn_method = 'faiss-gpu'
        self.metrics = ['pairwise', 'bcubed', 'nmi']
        self.min_sim = 0.48
        self.k = 30

    def forward(self, feature_matrix):
        return self.face_cluster(feature_matrix)

    def l2norm(self, vec):
        # 求每行向量的范数，并进行除法实现归一化
        vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
        return vec

    # def intdict2ndarray(d, default_val=-1):
    #     arr = np.zeros(len(d)) + default_val
    #     for k, v in d.items():
    #         arr[k] = v
    #     return arr

    # def read_meta(fn_meta, start_pos=0, verbose=True):
    #     """
    #     idx2lb：每一个顶点对应一个类
    #     lb2idxs：每个类对应一个id
    #     """
    #     lb2idxs = {}
    #     idx2lb = {}
    #     with open(fn_meta) as f:
    #         for idx, x in enumerate(f.readlines()[start_pos:]):
    #             lb = int(x.strip())
    #             if lb not in lb2idxs:
    #                 lb2idxs[lb] = []
    #             lb2idxs[lb] += [idx]
    #             idx2lb[idx] = lb

    #     inst_num = len(idx2lb)
    #     cls_num = len(lb2idxs)
    #     if verbose:
    #         print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    #     return lb2idxs, idx2lb

    def knns2ordered_nbrs(self, knns, sort=True):
        if isinstance(knns, list):
            knns = np.array(knns)
        nbrs = knns[:, 0, :].astype(np.int32)
        dists = knns[:, 1, :]
        if sort:
            # sort dists from low to high
            nb_idx = np.argsort(dists, axis=1)
            idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
            dists = dists[idxs, nb_idx]
            nbrs = nbrs[idxs, nb_idx]
        return dists, nbrs

    # 构造边
    def get_links(self, single, links, nbrs, dists):
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                # 排除本身节点
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - self.min_sim:
                    count += 1
                    links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            # 统计孤立点
            if count == 0:
                single.append(i)
        return single, links

    def cluster_by_infomap(self, nbrs, dists):
        """
        基于infomap的聚类
        :param nbrs: 
        :param dists: 
        :return: 
        """
        single = []
        links = {}
        with Timer('get links', verbose=True):
            single, links = self.get_links(single=single, links=links, nbrs=nbrs, dists=dists)

        infomapWrapper = infomap.Infomap("--two-level --directed")
        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)
        # 聚类运算
        infomapWrapper.run()

        label2idx = {}
        idx2label = {}

        # 聚类结果统计
        for node in infomapWrapper.iterTree():
            # node.physicalId 特征向量的编号
            # node.moduleIndex() 聚类的编号
            idx2label[node.physicalId] = node.moduleIndex()
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        for k, v in label2idx.items():
            if k == 0:
                node_count += len(v[2:])
                label2idx[k] = v[2:]
                # print(k, v[2:])
            else:
                node_count += len(v[1:])
                label2idx[k] = v[1:]
                # print(k, v[1:])

        # print(node_count)
        # 孤立点个数
        print("孤立点数：{}".format(len(single)))

        keys_len = len(list(label2idx.keys()))
        # print(keys_len)

        # 孤立点放入到结果中
        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1

        print("总类别数：{}".format(keys_len))

        idx_len = len(list(idx2label.keys()))
        print("总节点数：{}".format(idx_len))
        return label2idx

    def get_dist_nbr(self, features, k=80, knn_method='faiss-cpu'):
        features = np.array(features)
        features = features.reshape(-1, 128)
        features = self.l2norm(features)

        index = knn_faiss(feats=features, k=k, knn_method=knn_method)
        knns = index.get_knns()
        dists, nbrs = self.knns2ordered_nbrs(knns)
        return dists, nbrs

    def face_cluster(self, feature_matrix):
        with Timer('All face cluster step'):
            dists, nbrs = self.get_dist_nbr(features=feature_matrix, k=self.k, knn_method=self.knn_method)
            print(dists.shape, nbrs.shape)
            return self.cluster_by_infomap(nbrs, dists)

    def save_cluster_result(self, id, label2idx, name_to_face_dict, rowid_to_face_dict, unknown_feature_matrix,
                            font_style, hdfs_client):
        frame_output_dir = self.local_frame_store_dir.format(id)
        face_det_output_dir = self.local_face_store_dir.format(id)
        frame_face_to_cluster = {}
        # 下载所有帧到本地
        check_frame_output_dir = vs_common.local_result_store_dir.format(id)
        hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(id) + '/origin/'
        hdfs_client.download(hdfs_frame_output_dir, check_frame_output_dir, overwrite=True, n_threads=5)
        for label, idx_list in label2idx.items():
            person_name = "unknown_{}".format(label)
            name_to_face_dict[person_name] = {}

            # 去重
            record = set()
            # print(idx_list)
            # print(rowid_to_face_dict)
            for idx in idx_list:
                info = rowid_to_face_dict[idx]
                frame_id = info['frame_id']
                face_id = info['face_id']
                if frame_id in record:
                    continue
                record.add(frame_id)
                info['feature'] = np.array(unknown_feature_matrix[idx]).tolist()

                situation_id = info['situation_id']
                if situation_id not in name_to_face_dict[person_name]:
                    name_to_face_dict[person_name][situation_id] = []
                name_to_face_dict[person_name][situation_id].append(info)

                # 生成图片
                image_path = frame_output_dir + "/{}/image_{}.jpg".format(info['situation_id'], frame_id)
                img = cv2.imread(image_path)
                im = Image.fromarray(img.astype(np.uint8))
                draw = ImageDraw.Draw(im)
                face_bbox = info['bbox']
                draw.rectangle(face_bbox, width=3, outline=(0, 0, 255))  # 画框
                draw.text((face_bbox[0], face_bbox[3]), person_name, (0, 255, 0), font=font_style)  # 写入label

                # 持久化存储
                face_det_output_path = face_det_output_dir + '/{}'.format(person_name)
                hdfs_face_det_output_path = self.hdfs_face_store_dir.format(id) + '/{}'.format(person_name)
                if not os.path.exists(face_det_output_path):
                    os.makedirs(face_det_output_path)
                    hdfs_client.makedirs(hdfs_face_det_output_path, 777)
                marked_face_save_path = face_det_output_path + "/image_{}.jpg".format(frame_id)
                hdfs_marked_face_save_path = hdfs_face_det_output_path + "/image_{}.jpg".format(frame_id)
                cv2.imwrite(marked_face_save_path, np.array(im))
                if not frame_face_to_cluster.__contains__(frame_id):
                    frame_face_to_cluster[frame_id] = {}
                frame_face_to_cluster[frame_id][face_id] = person_name
                hdfs_client.write(hdfs_marked_face_save_path, cv2.imencode('.jpg', np.array(im))[1].tobytes(), overwrite=True)

        filename = face_det_output_dir + '/appear.json'
        with open(filename, 'w') as file:
            json.dump(name_to_face_dict, file)
        filename = face_det_output_dir + '/frame_face_to_cluster.json'

        frame_face_to_cluster_dir = vs_common.hdfs_result_store_path.format(
            id) + '/process/temp/frame_face_to_cluster.json'
        with open(filename, 'w') as file:
            json.dump(frame_face_to_cluster, file)
            hdfs_client.write(frame_face_to_cluster_dir ,
                                   json.dumps(frame_face_to_cluster).encode(), overwrite=True)

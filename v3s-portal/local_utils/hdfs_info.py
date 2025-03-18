import os
import sys
import json
from hdfs import InsecureClient


sys.path.append('/home/huyibo-21/vs-portal')
import vs_common


hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')

def getFileInfo(dir):
    return hdfs_client.list(dir)

def deleFile(dir):
    hdfs_client.delete(dir)

def readFile(dir):
    hdfs_client.read(dir)

def download(origin,output):
    hdfs_client.download(origin, output)


if __name__ == '__main__':
    # name_to_face_dict = {}
    # #hdfs_client.write('/bdap/students/video_result/45/process/temp/name_to_face_dict/1.json', json.dumps(name_to_face_dict).encode(),
    #                        overwrite=True)
    # deleFile(dir='/bdap/students/video_result/45/process/temp/rowid_to_face_dict/5.json')
    download('/bdap/students/DefaultUserId/videos/Jerry_Maguire.mp4', '/home/huyibo-21/shared/video/86')
    # hdfs_audio_file_output_path = vs_common.hdfs_result_store_path.format(49) + '/process/audio/audio_file.txt'
    # local_audio_numpy_output_path = vs_common.local_video_store_dir.format(49) + '/audio/'
    # hdfs_client.download("/bdap/students/video_result_parallel/97/process/relation/link5.json", "/home/huyibo-21", overwrite=True)
    #print(getFileInfo(dir="/bdap/students/video_result_parallel/97/process/relation"))

    # hdfs_client.makedirs("/bdap/students/video_result_parallel/119/process/relation", 777)
    # # hdfs_client.makedirs("/bdap/students/video_result_parallel/119/subtitle", 777)
    # hdfs_client.makedirs("/bdap/students/video_result_parallel/118/process/relation", 777)
    # # hdfs_client.makedirs("/bdap/students/video_result_parallel/118/subtitle", 777)
    # hdfs_client.makedirs("/bdap/students/video_result_parallel/117/process/relation", 777)
    # hdfs_client.makedirs("/bdap/students/DefaultUserId/videos", 777)





    # print(getFileInfo(dir="/bdap/students/video_result_parallel/117"))
    # print(getFileInfo(dir="/bdap/students/video_result_parallel/118"))
    # print(getFileInfo(dir="/bdap/students/video_result_parallel/118/process"))



    # hdfs_client.upload("/bdap/students/DefaultUserId/videos", "/home/huyibo-21/shared/video/Jerry_Maguire.mp4", 5)
    # hdfs_client.upload("/bdap/students/DefaultUserId/videos", "/home/huyibo-21/shared/video/Horrible_Bosses.mp4", 5)
    # hdfs_client.upload("/bdap/students/DefaultUserId/videos", "/home/huyibo-21/shared/video/Crazy_Stupid_Love.mp4", 5)

    # print(getFileInfo(dir="/bdap/students/DefaultUserId/videos"))


    # print(type(readFile('/bdap/students/video_result/45/process/temp/rowid_to_face_dict/3.json')))

    # check_frame_output_dir = vs_common.local_result_store_dir.format(82)
    # hdfs_frame_output_dir = vs_common.hdfs_result_store_path.format(82) + '/origin/'

    # check_frame_output_dir = '/home/huyibo-21/shared/video_parallel/81/process_results'
    # hdfs_frame_output_dir = '/bdap/students/video_result_parallel/81/origin/'
    # hdfs_client.download(hdfs_frame_output_dir, check_frame_output_dir, overwrite=True)
    # local_audio_output_path = vs_common.local_video_store_dir.format(98) + "/test/"
    # if not os.path.exists(local_audio_output_path):
    #     os.makedirs(local_audio_output_path)
    #
    # for root, dirs, files in os.walk(vs_common.local_result_store_dir.format(97) + '/origin'):
    #     print(root)
    #     print(dirs)
    #     print(files)
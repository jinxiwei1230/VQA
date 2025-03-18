import pickle
import numpy as np
from numpy import ndarray
from scipy.io import wavfile
import sys
from hdfs import InsecureClient

sys.path.append('/home/huyibo-21/vs-portal')
import vs_common

hdfs_client = InsecureClient(vs_common.HDFS_HOST, user='yanch')
rate, sig = wavfile.read('/home/huyibo-21/shared/video/86/3min_540p.wav')
# test = ndarray.dumps(sig)

# print(sig.shape)
print(type(sig))
np.savetxt('/home/huyibo-21/shared/video/86/audio/audio_temp.txt', X=sig)
# print(test)
hdfs_audiofile_output_path = vs_common.hdfs_result_store_path.format(id) + '/process/audio/audio_file.txt'
# sig.dump('/home/huyibo-21/shared/video/86/audio/audio_dump_file')
with open('/home/huyibo-21/shared/video/86/audio/audio_temp.txt', 'r') as f:
    hdfs_client.write(hdfs_audiofile_output_path, f, overwrite=True)
#
audio_numpy = hdfs_client.download(hdfs_audiofile_output_path, '/home/huyibo-21/shared/video/86/audiofromhdfs', overwrite=True)
with open('/home/huyibo-21/shared/video/86/audiofromhdfs/audio_file.txt', 'r') as ff:
    f = np.loadtxt(ff)
    print(type(f))
    # print(f == sig)
    wavfile.write('/home/huyibo-21/shared/video/86/audio/newaudio_file.wav', data=f)

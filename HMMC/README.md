<!-- 中文编码器的模型参数 -->

"/home/disk2/DATA/MSRVTT/hfl/chinese-roberta-wwm-ext"
/home/disk2/DATA/MSRVTT/hfl/chinese-roberta-wwm-ext

<!-- 第一步：视频预处理 -->

运行tools/frame2lmdb.py文件
修改视频路径、json路径、输出路径
最终获得mdb文件和json文件

<!-- 第二步：Test -->
<!-- 1.英文 -->

使用CHVTT数据集预训练，使用MSRVTT微调后的模型文件
/home/zhangyuxuan-23/baseline/MSRVTT/model_eng/pytorch_model.bin

修改dataloader.py中的dataloader_msrvtt_test函数的root路径和csv路径，分别是视频预处理文件和文本文件
root="/home/zhangyuxuan-23/baseline/MSRVTT/lmdb",
csv_path="/home/zhangyuxuan-23/baseline/MSRVTT/test.csv",

<!-- 2.中文 -->

使用CHVTT数据集预训练，使用VATEX微调后的模型文件
/home/zhangyuxuan-23/baseline/MSRVTT/model_chinese/pytorch_model.bin

修改test_list.txt
修改dataloader.py中的dataloader_vatex_test函数的root路径和csv路径，分别是视频预处理文件、list文件、文本文件
root='/home/zhangyuxuan-23/baseline/VATEX/lmdb',
data_path='/home/zhangyuxuan-23/baseline/VATEX',

<!-- 第三步：Fine tuning -->
<!-- 中文 -->
预训练模型：/home/zhangyuxuan-23/baseline/MSRVTT/model_chinese/pytorch_model.bin

修改train_list.txt
修改dataloader.py中的dataloader_vatex_train函数的root路径和csv路径，分别是视频预处理文件、list文件、文本文件
root='/home/zhangyuxuan-23/baseline/VATEX/lmdb',
data_path='/home/zhangyuxuan-23/baseline/VATEX',

注意batchsize的大小要和数据集的总大小相匹配


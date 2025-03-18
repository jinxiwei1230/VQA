# Self-Chained Image-Language Model for Video Localization and Question Answering


## Dataset Preparation
We test our model on:
+ [NExT-QA](https://doc-doc.github.io/docs/nextqa.html)

+ [STAR](https://star.csail.mit.edu/)

+ [How2QA](https://value-benchmark.github.io/index.html)

+ [TVQA](https://tvqa.cs.unc.edu/)

+ [VLEP](https://value-benchmark.github.io/index.html)

+ [QVHighlights](https://github.com/jayleicn/moment_detr)

[//]: # (NExT-QA: 用于多模态视频问答任务的数据集。)

[//]: # (STAR: 可能涉及情景理解或动作识别的视频问答数据集。)

[//]: # (How2QA: 主要用于评估模型在视频理解上的问答能力。)

[//]: # (TVQA: 包含来自电视节目片段的视频问答数据集。)

[//]: # (VLEP: 专注于评估视频中的逻辑推理能力。)

[//]: # (QVHighlights: 可能用于从视频中提取重要片段，并针对这些片段提出问答任务。)

We re-format original json/csv/jsonl files in different dataset to the same json format via jupyter script.

Please set your own dataset/video path in running scripts or in dataset config files. For example:

#这些数据集以不同的格式（如json、csv、jsonl等）提供，需要通过Jupyter脚本将它们重新格式化为统一的json格式。
* Option 1: change in running scripts

```bash
result_dir="YOUR_PATH"
train_path="YOUR_PATH"
val_path="YOUR_PATH"
video_path="YOUR_PATH"

exp_name='nextqa_infer'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path lavis/projects/sevila/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
datasets.nextqa.build_info.annotations.train.storage=${train_path} \
datasets.nextqa.build_info.annotations.val.storage=${val_path} \
datasets.nextqa.build_info.annotations.test.storage=${val_path} \
datasets.nextqa.build_info.videos.storage=${video_path} \
model.frame_num=4 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_eval=8 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

```

* Option 2: change in dataset config file:

change [config files](../lavis/configs/datasets/nextqa/defaults_qa.yaml)

[//]: # (方法1：在运行脚本中设置路径)
[//]: # (通过Bash脚本，用户可以在运行模型评估时指定数据集路径、视频路径、以及其他参数。)
[//]: # (例如：)
[//]: # (result_dir="YOUR_PATH")
[//]: # (train_path="YOUR_PATH")
[//]: # (val_path="YOUR_PATH")
[//]: # (video_path="YOUR_PATH")
[//]: # (脚本中还指定了模型的实验名称、预训练检查点路径、要使用的CUDA设备等。)

[//]: # (方法2：在数据集配置文件中设置路径)
[//]: # (直接在模型配置文件（如 nextqa_eval.yaml）中修改路径设置。)
[//]: # (文档提到可以在配置文件中找到相应的设置位置（lavis/configs/datasets/nextqa/defaults_qa.yaml），然后根据需求更改路径。)



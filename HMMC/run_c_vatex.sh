CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.run --nproc_per_node=1 --master_port 12000 main_task_retrieval.py --do_eval --num_thread_reader=8 --epochs=1 --batch_size=2 --n_display=1 --output_dir ckpts/val --max_frames 12 --frame_sample random --use_temp --use_frame_fea --lr 1e-4 --text_lr 3e-5 --coef_lr 8e-1 --batch_size_val 256  --task retrieval --dataset vatex --language chinese --init_model /home/disk2/dachuang1-23/HMMC/ckpts/val/pytorch_model.bin.1

#--task retrieval:设定任务类型为“检索”，意味着模型的目标是进行检索任务，可能是从一组视频中找到与查询文本相关的视频。
#--dataset vatex:设置数据集为 vatex，一个常见的视频-文本匹配数据集。
#--language chinese:选择使用中文作为语言。

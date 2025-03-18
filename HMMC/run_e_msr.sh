CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.run --nproc_per_node=1 --master_port 12000 main_task_retrieval.py --do_eval --num_thread_reader=8 --epochs=1 --batch_size=1 --n_display=1 --output_dir ckpts/val --max_frames 12 --frame_sample random --use_temp --use_frame_fea --lr 1e-4 --text_lr 3e-5 --coef_lr 8e-1 --batch_size_val 256  --task retrieval --dataset msrvtt --language english --init_model /home/zhangyuxuan-23/baseline/MSRVTT/model_eng/pytorch_model.bin

#--task retrieval:任务类型是 "retrieval"（检索），即模型用于视频-文本检索任务。
#--dataset msrvtt:数据集为 MSR-VTT，这是一个常用的视频-文本数据集，常用于视频检索任务。
#--language english:设置使用英文作为语言。
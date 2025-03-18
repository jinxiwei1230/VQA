# parameters
result_dir="/home/disk2/dachuang1-23/results"
exp_name='nextqa_ft'
ckpt='/home/disk2/dachuang1-23/results/nextqa_sr/checkpoint_best.pth'
#ckpt='sevila_checkpoints/sevila_pretrained_refined_nextqa.pth'

CUDA_VISIBLE_DEVICES=0 python train.py \
--cfg-path 'lavis/projects/sevila/train/nextqa.yaml' \
--options run.output_dir=${result_dir}/${exp_name} \
model.frame_num=4 \
datasets.nextqa.vis_processor.train.n_frms=32 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_train=4 \
run.batch_size_eval=4 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='qvh_freeze_locin_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa' \
run.log_freq=20 \
# run.resume_ckpt_path='/root/VideoQA/sevila/lavis/results/nextqa_ft/checkpoint_0.pth'
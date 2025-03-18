# parameters
result_dir="/home/disk2/dachuang1-23/results"
exp_name='star_sr'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
#ckpt='/home/disk2/dachuang1-23/results/star_sr/checkpoint_best.pth'


CUDA_VISIBLE_DEVICES=0 python train.py \
--cfg-path 'lavis/projects/sevila/train/star.yaml' \
--options run.output_dir=${result_dir}/${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.train.n_frms=4 \
datasets.star.vis_processor.eval.n_frms=32 \
run.batch_size_train=8 \
run.batch_size_eval=2 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=500 \
run.accum_grad_iters=1 \
model.task='qvh_train_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
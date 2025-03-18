id=368  # 获取第一个参数并赋值给 id
echo "The id is: $id"

# 切换到脚本所在的目录
cd /home/dachuang/vqa/sevila || exit

# 更新配置文件中的 ID
python /home/dachuang/vqa/sevila/update_config.py --id "${id}" \
    --cfg-path "/home/dachuang/vqa/sevila/lavis/projects/sevila/eval/nextqa_eval.yaml" \
    --dataset-cfg-path "/home/dachuang/vqa/sevila/lavis/configs/datasets/nextqa/defaults_qa.yaml"

echo "The id is: ${id}"

# 运行 evaluate.py 脚本
result_dir="/home/disk2/dachuang1-23/kafka_result/${id}"
exp_name='nextqa_infer'
ckpt="/home/disk2/dachuang1-23/results/nextqa_ft/checkpoint_best.pth"
CUDA_VISIBLE_DEVICES=0 python /home/dachuang/vqa/sevila/evaluate.py \
    --cfg-path "lavis/projects/sevila/eval/nextqa_eval.yaml" \
    --options run.output_dir=${result_dir}/${exp_name} \
    model.frame_num=4 \
    datasets.nextqa.vis_processor.eval.n_frms=32 \
    run.batch_size_eval=2 \
    model.task='qvh_freeze_loc_freeze_qa_vid' \
    model.finetuned=${ckpt} \
    run.task='videoqa'



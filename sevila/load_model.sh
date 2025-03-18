# 获取原始工作目录
original_dir=$(pwd)
echo "The original_dir is: $original_dir"
id=$1
echo "The id is: $id"
# 切换到脚本所在的目录
cd /home/dachuang/vqa/sevila || exit

# 更新配置文件中的 ID
python /home/dachuang/vqa/sevila/update_config.py --id "${id}" \
    --cfg-path "/home/dachuang/vqa/sevila/lavis/projects/sevila/eval/nextqa_eval.yaml" \
    --dataset-cfg-path "/home/dachuang/vqa/sevila/lavis/configs/datasets/nextqa/defaults_qa.yaml"

echo "当前目录： $(pwd)"
echo "The id is: ${id}"
# 返回到原始工作目录
cd "$original_dir" || exit
echo "当前目录： $(pwd)"


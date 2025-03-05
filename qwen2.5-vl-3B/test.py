# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5
# Set your own appid/appkey.
appid = '20240716002101367'
appkey = '1lRfgIX4PCbSnrhMB4kh'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def baidu_api(query,from_lang,to_lang):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    return result["trans_result"][0]['dst']

 
 
model_path = '/home/dachuang/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct'   # 修改为本地模型下载地址
video_path = "/home/disk2/dachuang1-23/kafka_result/958/87.mp4" # 输入视频地址
question = "Describe this video."
 
# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
 
# 视频输入
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": question},
        ],
    }
]
 
# 输入配置
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    # fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")
 
# 推理
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)

output_text_trans = baidu_api(output_text[0], from_lang, to_lang)
print(output_text_trans)
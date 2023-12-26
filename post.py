import json
import requests
from base64 import b64encode



with open(r'/root/autodl-tmp/visualglm-6b/bicycle.jpg', 'rb') as f:  # 此处不能使用encoding='utf-8'， 否则报错
    image = b64encode(f.read())  # b64encode是编码
image = str(image, encoding="utf-8")

req = {"image" : image,
       "text" : "please describe this picture briefly.",
       "history" : []}

response = requests.post("http://127.0.0.1:8080", json=req)
j = json.loads(response.text)
print(j["result"])


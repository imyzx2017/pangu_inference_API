import json
import time

import requests
import ast

url = "http://192.168.204.68:8899/inference_api"

payload = {
        'input_sentence': '近日有消息称，北京新东方（EDU.US）成立素质教育成长中心，下设艺术创作学院、人文发展学院、语商素养学院、自然科创空间站、智体运动训练馆、优质父母智慧馆六大板块，专注学生德智体美劳五育目标发展要求。'
                          '8月11日，红星资本局致电北京新东方，工作人员向红星资本局确认了“优质父母智慧馆”的存在。'
                          '工作人员称，',
        'topK': 0,
        'topP': 0.9,
        'result_len': 128
    }

########################## step1: 发送用户自定义输入payload，等待模型推理 ################################
response = requests.request("POST", url, data=payload)#, headers=headers)
if response.status_code == 200:
    print(response.text)
else:
    raise RuntimeError('请检查输入url的正确性，并重试(默认为: /inference_api)')
############################ step2: 获取当前请求的UID ##################################
get_current_queue_response = requests.request("GET", url, data="")
if get_current_queue_response.status_code == 200:
    step2_response = get_current_queue_response.text
    print(step2_response)
    _predict_timeConsume = int(step2_response.split(',')[1].split(': ')[-1][:-1])
    _request_id = int(step2_response.split(':')[-1][1:])
else:
    raise RuntimeError('请检查输入url的正确性，并重试(默认为: /inference_api)')

# ########################## step3: sleep等待模型推理完成，获取模型推理结果 ################################
time.sleep(_predict_timeConsume)
result_url = url.replace('_api', '_result')
get_result_response = requests.request("POST", result_url, data="{}".format(_request_id))
if get_result_response.status_code == 200:
    output_sentence = get_result_response.text
    print("output_sentence is :{}".format(output_sentence))
else:
    raise RuntimeError('请检查输入url的正确性，并重试(默认为: /inference_result_url)')

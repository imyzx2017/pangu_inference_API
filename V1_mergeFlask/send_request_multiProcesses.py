import time
import requests

def send_your_request2pangu_inference_API_url(payload, url='http://192.168.204.68:8899/inference_api'):
    ########################## step1: 发送用户自定义输入payload，等待模型推理 ################################
    response = requests.request("POST", url, data=payload)#, headers=headers)
    if response.status_code == 200:
        res_text = response.text
        print(res_text)
        _predict_timeConsume = int(res_text.split(',')[1].split(': ')[-1][:-1])
        _request_id = int(res_text.split(':')[-1][1:])
    elif response.status_code == 404:
        raise RuntimeError('请检查输入url的正确性，并重试(默认为: /inference_api)')
    elif response.status_code == 300:
        res_text = response.text
        raise RuntimeError(res_text)
    # ########################## step2: sleep等待模型推理完成，获取模型推理结果 ################################
    assert _predict_timeConsume >= 0
    time.sleep(_predict_timeConsume)

    get_result_response = requests.request("GET", url, params="{}".format(_request_id))
    if get_result_response.status_code == 200:
        output_sentence = get_result_response.text
        # print("output_sentence is :{}".format(output_sentence))
        return output_sentence
    elif get_result_response.status_code == 404:
        raise RuntimeError('请检查输入url的正确性，并重试(默认为: /inference_result_url)')
    elif get_result_response.status_code == 300:
        raise RuntimeError(get_result_response.text)


if __name__ == '__main__':
    # url = "http://192.168.204.68:8899/inference_api"

    payload = {
        'input_sentence': '近日有消息称，北京新东方（EDU.US）成立素质教育成长中心，下设艺术创作学院、人文发展学院、语商素养学院、自然科创空间站、智体运动训练馆、优质父母智慧馆六大板块，专注学生德智体美劳五育目标发展要求。'
                          '8月11日，红星资本局致电北京新东方，工作人员向红星资本局确认了“优质父母智慧馆”的存在。'
                          '工作人员称，',
        'topK': 0,
        'topP': 0.9,
        'result_len': 122
    }
    output_sentence = send_your_request2pangu_inference_API_url(payload)
    print("output_sentence is :{}".format(output_sentence))
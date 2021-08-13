# pangu-alpha 13B inference API
###################################################################
# 说明：本项目处于前沿探索阶段，体验功能仅供学术测试使用。请勿输入违反法律内容，同时，
# 未经许可，禁止分享，传播输入及生成文本内容。感谢理解！
###################################################################
import requests
def send_your_request2pangu_inference_API_url(payload, url='https://pangu-alpha.openi.org.cn/query?'):
    res = requests.get(url, params=payload)
    return res.json()['rsvp'][-1]

if __name__ == '__main__':
    # url = "https://pangu-alpha.openi.org.cn/query?"
    payload = {
        'u': '近日有消息称，北京新东方（EDU.US）成立素质教育成长中心，下设艺术创作学院、人文发展学院、语商素养学院、自然科创空间站、智体运动训练馆、优质父母智慧馆六大板块，专注学生德智体美劳五育目标发展要求。'
                          '8月11日，红星资本局致电北京新东方，工作人员向红星资本局确认了“优质父母智慧馆”的存在。'
                          '工作人员称，',
        'top_k': 4,
        'top_p': 0.9,
        'result_len': 50
    }
    output_sentence = send_your_request2pangu_inference_API_url(payload)
    print("output_sentence is :{}".format(output_sentence))
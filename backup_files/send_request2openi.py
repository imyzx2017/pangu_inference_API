import json
import requests
import ast

url = "https://git.openi.org.cn/user/login"

payload = {
        'user_name': 'imyzx',
        'password': 'yzx4495449s5'
    }

response = requests.request("POST", url, data=payload)#, headers=headers)
print(response.text)
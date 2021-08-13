# coding=UTF-8
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import json
import threading
import flask

import obs
from flask_apscheduler import APScheduler
from process_sentence import pre_process, post_process

import sentencepiece as spm
import jieba
import math

import random
import re
# import requests

############ setting redis ################
import redis
r = redis.StrictRedis(host='192.168.204.68', port=6379, db=0)
queue_r = r #redis.StrictRedis(host='192.168.204.68', port=6379, db=0)
###########################################
time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
random_num = random.randrange(100000, 999999)
GLOBAL_DIR_NAME = str(time_stamp) + "_" + str(random_num) + '_data-13B-releaseV3-test03'


app = flask.Flask(__name__)
MAX_MESSAGE_LENGTH = 0x7fffffff

OFFLOAD = '02_02'
QUEUE_NAME = 'queue{}'.format(OFFLOAD)
TIMECONSUME_NAME = 'timeConsume{}'.format(OFFLOAD)

hf_obs_dir_format = lambda x: x if x.endswith('/') else (x+'/')

SCAN_SLEEP_TIME = 0.1 #every 0.5 seconds scan
SCAN_DIRECTORY = 's3://pcl-verify/yizx/serving_logs_13B_increment_flaskPassParams_releaseV2_autoid_codesTest_offload{}'.format(OFFLOAD)
bucket_name = SCAN_DIRECTORY[5:].split('/',1)[0]

hf_obs_file = lambda *x: '/'.join([SCAN_DIRECTORY, *x])
hf_work_file = lambda *x: os.path.join(GLOBAL_DIR_NAME, *x)

if not os.path.exists(hf_work_file('request')):
    os.makedirs(hf_work_file('request'))
if not os.path.exists(hf_work_file('reply')):
    os.makedirs(hf_work_file('reply'))
if not os.path.exists(hf_work_file('stars')):
    os.makedirs(hf_work_file('stars'))
if not os.path.exists(hf_work_file('autoPrompt')):
    os.makedirs(hf_work_file('autoPrompt'))

def restart_webFront_clearRedis(redi):
    ###### clear all info ############
    # for item in redi.keys():
    #     redi.delete(item)
    # ##################################
    # tmp_list = ['queue', 'queue01', 'queue02', 'queue03', 'queue00_01', 'queue00_02', 'queue01_01', 'queue01_02'
    #             , 'queue02_01', 'queue02_02',
    #             'timeConsume', 'timeConsume01', 'timeConsume02', 'timeConsume03'
    #             'timeConsume00_01', 'timeConsume00_02', 'timeConsume01_01', 'timeConsume01_02',
    #             'timeConsume02_01', 'timeConsume02_02']

    #######################################################################
    tmp_list = ['queue{}'.format(OFFLOAD), 'timeConsume{}'.format(OFFLOAD)]

    for item in tmp_list:
        redi.delete(item)


def obs_path_to_key(obs_path, bucket_name=bucket_name):
    tmp0 = 's3://'+bucket_name
    assert obs_path.startswith(tmp0)
    ret = obs_path[(len(tmp0)+1):]
    return ret

def parse_obs_path(obs_path):
    # print(obs_path)
    assert obs_path.startswith('s3://')
    obs_path = obs_path[5:]
    bucket_name,key = obs_path.split('/',1)
    return bucket_name, key

def obs_isfile(obs_client, obs_path):
    bucket,key = parse_obs_path(obs_path)
    assert not key.endswith('/')
    ret = obs_client.getObjectMetadata(bucket, key)['status']==200
    return ret

class JIEBATokenizer:
    def __init__(self, vocab_file, model_file, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        # self.encoder = json.load(open(vocab_file))
        with open(vocab_file, 'r', encoding='utf-8') as fid:
            self.decoder = dict((x,y.strip().split('\t',1)[0]) for x,y in enumerate(fid))
            self.encoder = {v:k for k,v in self.decoder.items()}
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.eod_id = self.encoder['<eod>']
        self.eot_id = self.encoder['<eot>']
        self.pad_id = self.encoder['<pad>']

    def cut(self, text):
        ret = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        return ret

    def spm_cut(self, text):
        ret = [self.decode(x) for x in self.encode(text)]
        return ret

    def encode(self, text):
        seg_list = self.cut(text) #['他', '是', '苹果公司', '的', '创始人']
        new_seg = " ".join(seg_list)
        ids = self.sp.encode(new_seg) #[26, 16, 2339, 775, 11, 7352]
        return ids

    def decode(self, ids):
        text = self.sp.decode(ids)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text


tokenizer = JIEBATokenizer('bpe_4w_pcl/vocab.vocab', 'bpe_4w_pcl/vocab.model')
z0 = '他是苹果公司的创始人'
tokenizer.spm_cut(z0)

global_varible_lock = threading.Lock()

global_varible = {
    'UID': 0,
}

MAX_QUEUE_LENS = 2
STOP_FLAG = True
LOOP_COUNT = 0

restart_webFront_clearRedis(queue_r)

def get_sentence_UID():
    try:
        global_varible_lock.acquire()
        ret = global_varible['UID']
        global_varible['UID'] += 1
    finally:
        global_varible_lock.release()
    return ret


access_key_id = 'T5SLNRWNNNINAQBBNFJ6'
secret_access_key = 'sVEDyI8KlNKTkVBuRkgwjVyTXlygiq21pzQG9qUu'
server = '112.95.163.82'
bucket_name = SCAN_DIRECTORY[5:].split('/', 1)[0]
obs_client = obs.ObsClient(access_key_id=access_key_id, secret_access_key=secret_access_key, server=server)

# in_data = {
#     'uid': 1,
#     'session_id': 1,
#     'time_stamp': 1,
#     'query': '问：中国四大发明\n答：',
#     "query_type": "自由生成",
#     "parameters": {}
# }
# in_data = json.dumps(in_data)
# responseTMP = requests.post('http://192.168.202.129:7777/prompt_reco/', data=in_data)


def obs_upload_file(obs_client, local_path, obs_path):
    obs_path = obs_path.rstrip('/')
    bucket_name,key = parse_obs_path(obs_path)
    response = obs_client.uploadFile(bucket_name, key, local_path)
    if response.status!=200:
        print(f'[{response.status}] fail for uploading "{local_path}"')
    return response.status


def obs_download_file(obs_client, obs_path, local_path):
    obs_path = obs_path.rstrip('/')
    bucket_name,key = parse_obs_path(obs_path)
    if os.path.isdir(local_path):
        local_path = os.path.join(local_path, obs_path.rsplit('/')[-1])
    response = obs_client.downloadFile(bucket_name, key, local_path)
    if response.status!=200:
        print(f'[{response.status}] fail for downloading "{obs_path}"')


def obs_listdir(obs_client, obs_path):
    bucket_name,key = parse_obs_path(obs_path)
    key = hf_obs_dir_format(key)
    assert len(key)>1 #cannot be the root directory
    tmp0 = len(key)
    ret = [x['key'][tmp0:].split('/',1)[0] for x in obs_client.listObjects(bucket_name, key)['body']['contents']]
    ret = sorted({x for x in ret if x})
    return ret


def update_queue_startUp(redi, queue_name, time_name):
    # 获取当前queue长度
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)   # redi.get(queue_name)

    # get queue RealTime TimeConsume !!
    redisTimeConsume = redi.get(time_name)
    if redisTimeConsume == None:
        current_timeConsume = 0
    else:
        current_timeConsume = int(redisTimeConsume.decode('utf-8'))
    ################################################################

    assert len(queue_list) <= MAX_QUEUE_LENS

    # 队列长度为0
    if len(queue_list) == 0:
        thisUID = get_sentence_UID()
        redi.rpush(queue_name, thisUID)
        return thisUID, thisUID, len(queue_list) + 1

    # 队列长度已满，或队列排队时间过满超过27分钟，不追加队列
    elif len(queue_list) == MAX_QUEUE_LENS: # or current_timeConsume >= (27 * 60):
        tmp = r.lindex(queue_name, 0)
        currentUID = int(tmp.decode('utf-8'))
        return None, currentUID, MAX_QUEUE_LENS

    # 队列长度未满，且排队时间不长，则追加队列
    elif len(queue_list) > 0 and len(queue_list) < MAX_QUEUE_LENS:
        tmp = r.lindex(queue_name, 0)
        currentUID = int(tmp.decode('utf-8'))
        thisUID = get_sentence_UID()
        redi.rpush(queue_name, thisUID)
        return thisUID, currentUID, len(queue_list) + 1


def get_currentUID_fromRedis_withRankIDSave(redi, queue_name):#, uid):
    # queue_name = 'queue01'
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    assert len(queue_list) <= MAX_QUEUE_LENS
    tmp = r.lindex(queue_name, 0)
    currentUID = int(tmp.decode('utf-8'))

    # # calculate uid correspond rankID
    # queue_L = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    # rankID = 0
    # for item in queue_L:
    #     tmp = int(item.decode('utf-8'))
    #     if tmp == uid:
    #         pass
    #     else:
    #         rankID += 1
    # redi.set('rankID', rankID)

    return currentUID


def update_queue_endTime(redi, queue_name):
    # queue_name = 'queue01'
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    assert len(queue_list) <= MAX_QUEUE_LENS
    # 删除队列队首的uid
    redi.lpop(queue_name)
    # tmp = r.lindex(queue_name, 0)
    # currentUID = int(tmp.decode('utf-8'))
    # return currentUID


def get_current_queueLens_withRankUID(redi, queue_name):
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    assert len(queue_list) <= MAX_QUEUE_LENS

    thisUID = global_varible['UID'] - 1
    # calculate uid correspond rankID
    queue_L = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    rankID = 0
    for item in queue_L:
        tmp = int(item.decode('utf-8'))
        if tmp == thisUID:
            pass
        else:
            rankID += 1
    # if thisUID not in queue, reset rankID
    if rankID == MAX_QUEUE_LENS:
        rankID = ""


    return len(queue_list), rankID


def register_with_username_passwd_email(redi, username, passwd, email):
    # 判读是否有重复username
    userinfo_list = redi.lrange(username, 0, -1)
    if len(userinfo_list) == 0:
        # 添加username / passwd / email的约束设置
        redi.rpush(username, passwd, email)
        return 1
    else:
        return 2


def login_with_username_passwd(redi, username, passwd):
    # 判读username
    userinfo_list = redi.lrange(username, 0, -1)
    if len(userinfo_list) == 0:
        return False
    elif not len(userinfo_list) == 2:
        return False
    else:
        redis_passwd = redi.lindex(username, 0)
        redis_passwd = redis_passwd.decode('utf-8')
        if redis_passwd == passwd:
            return True
        else:
            return False


def register_rule_forUsername(redi, in_str):
    tmp = redi.lrange(in_str, 0, -1)
    if len(tmp) == 0:
        pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9\._]{6,12}')
        if not pattern.search(in_str) == None:
            return 1
        else:
            return 0
    else:
        return 2


def register_rule_forPasswd(in_str):
    pattern = re.compile(r'[a-zA-Z0-9\._@]{6,12}')
    if not pattern.search(in_str) == None:
        return True
    else:
        return False


def register_rule_forEmail(in_str):
    pattern = re.compile(r'[a-zA-Z0-9\._@]{10,20}')
    if not pattern.search(in_str) == None and not in_str.find('@') == -1:
        return True
    else:
        return False


def update_timeConsume_startUp(redi, time_consume, time_name):
    KEYNAME = time_name
    if redi.get(KEYNAME) == None:
        redi.set(KEYNAME, time_consume)
    else:
        queue_time = int(redi.get(KEYNAME).decode('utf-8'))
        new_timeConsume = queue_time + time_consume
        redi.set(KEYNAME, new_timeConsume)


def update_timeConsume_endUp(redi, time_consume, time_name):
    KEYNAME = time_name
    queue_time = int(redi.get(KEYNAME).decode('utf-8'))
    new_timeConsume = queue_time - time_consume
    redi.set(KEYNAME, new_timeConsume)


def queue_getCurrentTimeConsume(redi, queueLens, time_name):
    if queueLens == 0:
        return 0
    else:
        KEYNAME = time_name
        try:
            queue_time = int(redi.get(KEYNAME).decode('utf-8'))
            return queue_time
        except:
            return 0


def convert_secondTime2Mins(timeConsume):
    minutes = timeConsume // 60
    min_minutes = minutes // 3
    return min_minutes, math.ceil(minutes)


if obs_isfile(obs_client, hf_obs_file('current_uid.txt')):
    obs_download_file(obs_client, hf_obs_file('current_uid.txt'), hf_work_file('current_uid.txt'))
    with open(hf_work_file('current_uid.txt'), 'r') as fid:
        global_varible['UID'] = int(fid.read().strip())


# # 点击”提示生成“按钮后，修正web input框体中的内容
# @app.route('/prompt')
# def generate_prompt():
#     uid = global_varible["UID"]
#     top_k = flask.request.args.get('top_k')
#     top_p = flask.request.args.get('top_p')
#     result_len = flask.request.args.get('result_len')
#     userRawInput = flask.request.args.get('u')
#     task_name = flask.request.args.get('task_name')
#     print('任务类型： {}'.format(task_name))
#
#     user_in_data = {
#         'uid': uid,
#         'session_id': 1,
#         'time_stamp': 1,
#         'query': userRawInput,
#         "query_type": task_name,
#         "parameters": {}
#     }
#     user_in_data = json.dumps(user_in_data)
#     response = requests.post('http://192.168.202.129:7777/prompt_reco/', data=user_in_data)
#     if response.status_code == 200:
#         res = json.loads(response.text)
#         request_file = hf_work_file('autoPrompt', f'userInputWithPrompt_{uid}.json')
#
#         PromptSentence = res['prompt_recommendation']
#         if PromptSentence == "":
#             PromptSentence = userRawInput
#         parameters_recommendation = res['parameters_recommendation']
#
#         with open(request_file, 'w', encoding='utf-8') as fid:
#             tmp0 = {
#                 'RawSentence': userRawInput,
#                 'PromptSentence': PromptSentence,
#                 'task_name': task_name,
#                 'Pred_task_name': res['query_type_recognition'],
#                 'top_k': top_k,
#                 'top_p': top_p,
#                 'result_len': result_len,
#                 "parameters_recommendation": parameters_recommendation
#             }
#             json.dump(tmp0, fid, ensure_ascii=False)
#         try:
#             rsvp = [PromptSentence, parameters_recommendation["top_p"], parameters_recommendation["top_k"]]
#         except:
#             rsvp = [PromptSentence, top_p, top_k]
#         return flask.jsonify(ok=True, rsvp=rsvp)
#
#     else:
#         rsvp = []
#         return flask.jsonify(ok=False, rsvp=rsvp)

# 点击”用户评分“按钮后，失效该模块，不允许二次打分
@app.route('/record')
def do_record():
    post_process_output = flask.request.args.get('post_process_output')
    star = flask.request.args.get('star')
    star = int(star)

    # 目前点击”用户评分“按钮后，无法再拿到UID，所以单独保存输出和star的值到json文件
    time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    random_num = random.randrange(100000, 999999)
    thisStar_json_name = 'userStar_' + str(time_stamp) + "_" + str(random_num) + '.json'
    thisStar_file = hf_work_file('stars', thisStar_json_name)

    with open(thisStar_file, 'w', encoding='utf-8') as fid:
        tmp0 = {
            'sentence': post_process_output,
            'star': star,
        }
        json.dump(tmp0, fid, ensure_ascii=False)
    return flask.jsonify(ok=True, rsvp=['recorded!'])

# 点击”用户评分“按钮后，失效该模块，不允许二次打分
@app.route('/task')
def do_task():
    task_name = flask.request.args.get('task_name')
    if not task_name.find('盘古对对联') == -1:
        InputContents = ['上联：欢天喜地度佳节', '上联：春雨润人间，社会和谐万象新', '上联：瑞风播福泽，事业昌盛千家乐']
        OutputContents = ['下联：花好月圆庆团圆', '下联：高歌展宏图，战甲通天万仞高', '下联：骏骥踏春来，前程似锦万代兴']
        topK = 0
        topP = 0.9
        resultLens = 50

    elif not task_name.find('盘古问答') == -1:
        InputContents = ['四川的省会是？', '哪个省的地方戏是“黄梅戏”：', '中国的四大发明有哪些？', '中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？']
        OutputContents = ['成都,简称蓉,别称蓉。', '安徽的黄梅戏,是中国五大地方戏之一,也是中国五大戏曲剧种之一。', '火药、指南针、造纸术、印刷术。', '中国的首都是北京,美国的首都是华盛顿,日本的首都是东京,法国的首都是巴黎,澳大利亚的首都是堪培拉。']
        topK = 0
        topP = 0.5
        resultLens = 50

    elif not task_name.find('盘古对话生成') == -1:
        InputContents = ['书生：羌笛何须怨杨柳，春风不度玉门关。\n飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）\n书生：小兄弟，要不要一起喝一杯？\n飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。\n书生：小兄弟，看你年纪轻轻，不至于这么势利吧？\n飞云：',
                         ]
        OutputContents = ['咳，年轻人说的话要有选择性，我既然说了我先请你，你却又说一个人喝一杯，可见我是有目的的，要是小兄弟真是不肯给在下好处，也可以不请我，到时候我便跟你说我请了我一个人情.若是小兄弟答应我，日后我自然会等你。\n书生：啊，我明白了，这样就不会让你和我喝酒了。\n飞云：对对，小兄弟是有诚信的人，不如这样，你叫在下这样做，我也是这样做，我们就到此为止，日后互不相欠。',
                          ]
        topK = 0
        topP = 0.9
        resultLens = 50

    elif not task_name.find('盘古续写小说') == -1:
        InputContents = ['张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，',
                         '火神祝融取山之木制琴三把，名为凰来、鸾来、凤来。祝融对三琴爱惜有加，尤喜凤来，时常弹奏。凤来化灵，具人之形态，且能说人语。祝融大喜，请地皇女娲用牵引命魂之术将琴灵化为人身，以父子情谊相待，称之为太子长琴。',
                         ]
        OutputContents = ['却又略带疑惑。张无忌问赵敏有没有感觉到那股恐惧。赵敏说她都可以感受得到,可是自己从未有过那感觉。张无忌问赵敏可知道自己这把刀有多大吗。赵敏说它有十四寸,厚两寸五分,重二斤八两,其锋利无比。张无忌哈哈大笑,说我也不过就这一把了。',
                          '琴灵自然待祝融如同自己骨肉，祝融得琴通灵，聪颖过人，但性情刚烈，不近人情。有一次，长琴召来蛟龙偷入禁地，祝融得知此事后即将长琴囚禁。长琴不甘被囚禁，在蛟龙的帮助下逃离禁地。长琴在女娲宫修炼时，太子长琴于女娲宫中每日听琴，忽见一只白狐狸从禁地出逃']
        topK = 0
        topP = 0.9
        resultLens = 50

    elif not task_name.find('盘古专业文档撰写') == -1:
        InputContents = ['人工智能成为国际竞争的新焦点。人工智能是引领未来的战略性技术，世界主要发达国家把发展人工智能作为提升国家竞争力、维护国家安全的重大战略，加紧出台规划和政策，围绕核心技术、顶尖人才、标准规范等强化部署，力图在新一轮国际科技竞争中掌握主导权。当前，',
                         '计算处理子系统由众多高密度计算节点构成。计算节点是高性能计算机计算处理系统的基本组成和核心部件，其平衡设计、可靠性设计和能耗管理对于系统的效率、稳定性和能效比具有至关重要的影响。具体而言，']
        OutputContents = ['一是加快构建人工智能发展新体系。要加快人工智能领域开放性、协同性、产业链和价值链深度融合,以关键共性技术突破带动技术、产业和市场创新发展,加速人工智能技术突破和模式创新,形成我国高质量人工智能发展新优势。\n二是构建人工智能发展新体制。要健全适应人工智能发展特点的国家创新体系,深化科技体制改革,加快建设国家重点实验室体系、科技基础设施体系、科研院所能力提升体系、高水平人才队伍体系、技术标准体系、网信军民融合体系、知识产权保护体系、行业自律和社会监督体系等。',
                          '可以从以下几方面进行设计。\n1.计算节点的物理设计\n计算节点的物理设计需要考虑计算效率、稳定性、可用性、兼容性等因素。计算节点的计算效率主要是由于其内部硬件资源和系统资源之和的数量决定的,计算节点的稳定性主要取决于其所处的网络结构以及自身的软件结构。\n2.计算节点的可靠性设计\n为了保证计算节点可靠稳定地运行,计算节点的可靠性设计主要包括物理设计、软件设计和软件维护三部分。物理设计需要考虑的主要问题包括:计算节点数量、计算节点内部结构、计算节点各处理器的数量、计算节点的电压计算、节点的操作系统及操作系统平台的设计。'
                          ]
        topK = 0
        topP = 0.9
        resultLens = 50

    elif not task_name.find('盘古自由生成') == -1:
        InputContents = ['']
        OutputContents = ['']
        topK = 0
        topP = 0.9
        resultLens = 50

    else:
        InputContents = ['']
        OutputContents = ['']
        topK = 0
        topP = 0.9
        resultLens = 50


    randomNum = random.randint(0, len(InputContents) - 1)

    return flask.jsonify(ok=True, rsvp='{}#{}#{}#{}#{}'.format(InputContents[randomNum],
                                                               OutputContents[randomNum],
                                                               topK, topP, resultLens))


@app.route('/register')
def do_register():
    user_name = flask.request.args.get('r_user_name')
    password = flask.request.args.get('r_password')
    email = flask.request.args.get('r_email')
    if user_name == '' or password == '' or email == '':
        registerSuccessFlag = 0
    elif register_rule_forUsername(r, user_name) == 0 or \
            register_rule_forPasswd(password) == False or \
            register_rule_forEmail(email) == False:
        registerSuccessFlag = 0
    elif register_rule_forUsername(r, user_name) == 2:
        registerSuccessFlag = 2
    else:
        registerSuccessFlag = register_with_username_passwd_email(r, user_name, password, email)

    print(user_name, password, email)
    return flask.jsonify(ok=registerSuccessFlag, rsvp=['{},{},{}'.format(user_name, password, email)])


@app.route('/login')
def do_login():
    user_name = flask.request.args.get('user_name')
    password = flask.request.args.get('password')
    if user_name == '' or password == '':
        loginSuccessFlag = False
    else:
        loginSuccessFlag = login_with_username_passwd(r, user_name, password)

    print(user_name, password)
    return flask.jsonify(ok=loginSuccessFlag, rsvp=['{},{}'.format(user_name, password)])


@app.route('/queue')
def do_queue():
    queue_name = flask.request.args.get('queue_name')
    time_name = flask.request.args.get('time_name')
    time.sleep(0.5)
    queueLens, rankID = get_current_queueLens_withRankUID(queue_r, queue_name)

    # 因为这个接口延迟1s执行，所以之前可以先对redi的TimeConsume赋值，这里进行获取，得到当前排队预计费时
    time_consume = queue_getCurrentTimeConsume(queue_r, queueLens, time_name)
    _, time_consume = convert_secondTime2Mins(time_consume)

    if queueLens == 0:
        return flask.jsonify(ok=True, rsvp=['错误请求'], flag=False)
    else:
        print('{}, 预计等待{}分钟'.format(queueLens, time_consume + 1))
        # return flask.jsonify(ok=True, rsvp=['{}, 预等{}~{}分钟'.format(queueLens, min_time_consume + 1, time_consume + 1)], flag=False)
        return flask.jsonify(ok=True, rsvp=['{}, 预计等待{}分钟'.format(queueLens, time_consume + 1)],
                             flag=False)
    # return flask.jsonify(ok=True, rsvp=['{}, 上限={}'.format(queueLens, MAX_QUEUE_LENS)], flag=False)


@app.route('/query')
def do_query():

    global timeout
    input_sentence = flask.request.args.get('u')

    ## 获取 uid 进程发送的请求
    try:
        top_k = float(flask.request.args.get('top_k'))
    except Exception:
        top_k = 0
    try:
        top_p = float(flask.request.args.get('top_p'))
    except Exception:
        top_p = 0
    try:
        result_len = int(flask.request.args.get('result_len'))
    except Exception:
        print('[WARNING/client.py/do_query()/result_len]: something must be wrong')
        result_len = 50
        return flask.jsonify(ok=True, rsvp=['client.py error'], flag=True,)
    if input_sentence == None:
        return flask.jsonify(ok=True, rsvp=['请设置合理有效的文本输入(input_sentence)'], flag=True)
    if result_len == None or result_len <= 0 or result_len > 50:
        return flask.jsonify(ok=True, rsvp=['请设置合理有效的生成长度(result_len∈(0, {}]中整数，非整数api会自动向下取整)'.format(50)], flag=True,
                             )
    ## 判读topK范围是否合理，合理后，是否需要向下取整
    if top_k < 0 or top_k > 10:
        return flask.jsonify(ok=True, rsvp=['请设置合理有效的topK，范围取[0, 10]中整数，非整数api会自动向下取整'], flag=True)
    if top_p < 0 or top_p > 1:
        return flask.jsonify(ok=True, rsvp=['请设置合理有效的topP，范围取[0, 1)'], flag=True)
    if top_p == 0 and top_k == 0:
        return flask.jsonify(ok=True, rsvp=['topK或topP至少需要设置一个大于0的值！！topK∈[0, 10], topP∈[0,1)'], flag=True,
                             )
    # topK向下取整
    if int(top_k) != top_k:
        top_k = math.floor(top_k)
    # topK向下取整
    if int(result_len) != result_len:
        result_len = math.floor(result_len)

    top_k = int(top_k)
    result_len = int(result_len)
    print("start_sentence is {}".format(input_sentence))

    input_sentence, HasSensitiveWord, SensitiveWord = pre_process(flask.request.args.get('u'))
    if HasSensitiveWord:
        return flask.jsonify(ok=True, rsvp=['请勿输入敏感词: "{}"！'.format(SensitiveWord)], flag=True, sentence_uid=0)

    queue_name = QUEUE_NAME #flask.request.args.get('queue_name')
    time_name = TIMECONSUME_NAME #flask.request.args.get('time_name')


    uid, _, queue_lens = update_queue_startUp(queue_r, queue_name, time_name)
    # print queueList
    queue_list = queue_r.lrange(queue_name, 0, MAX_QUEUE_LENS)
    print('当前队列长度为: {}'.format(len(queue_list)))

    if not uid == None:


        ############################################
        # 队列时间预估，时间覆盖'time_consume'
        time_consume = math.ceil(result_len / 500 * 135)
        update_timeConsume_startUp(queue_r, time_consume, time_name)
        ############################################

        request_file = hf_work_file('request', f'request_{uid}.json')
        with open(request_file, 'w', encoding='utf-8') as fid:
            tmp0 = {
                'sentence': input_sentence,
                'top_k': top_k,
                'top_p': top_p,
                'result_len': result_len,
            }
            json.dump(tmp0, fid, ensure_ascii=False)

        # 根据队列长度设置 setting timeout
        queue_timeout = 55.5
        if result_len > 256 and result_len <= 512:
            timeout = 55 # 400 #+ queue_timeout
        elif result_len <= 256 and result_len > 128:
            timeout = 55 #240 #+ queue_timeout
        elif result_len <= 128:
            timeout = 55 #180 #+ queue_timeout
        elif result_len > 512:
            timeout = 55 #500 #+ queue_timeout

        # while True 去获取队列队首的currentUID是否是uid？是的话，则开始当前uid的推理流程
        obs_upload_file(obs_client, request_file, hf_obs_file('request', f'{uid}.json'))
        request_ready_file = hf_work_file('request', f'{uid}_ready')
        with open(request_ready_file, 'w') as fid:
            pass
        obs_upload_file(obs_client, request_ready_file, hf_obs_file('request', f'{uid}_ready'))

        reply_file = hf_work_file('reply', f'reply_{uid}.json')
        reply_file_obs = hf_obs_file('reply', f'{uid}.json')
        reply_ready_file_obs = hf_obs_file('reply', f'{uid}_ready')

        reply_postProcessFile = hf_work_file('reply', f'reply_{uid}_postProcess.json')

        # 读取OBS开始计时
        start_time = time.time()
        raw_output = ""
        while True:
            z0 = obs_listdir(obs_client, hf_obs_file('reply'))
            if f'{uid}_ready' in z0:
                obs_download_file(obs_client, reply_file_obs, reply_file)
                obs_client.deleteObject(bucket_name, obs_path_to_key(reply_file_obs))
                obs_client.deleteObject(bucket_name, obs_path_to_key(reply_ready_file_obs))
                finish_flag = True
                break
            else:
                time.sleep(SCAN_SLEEP_TIME)
            if time.time() - start_time > timeout:
                finish_flag = False
                break

        if finish_flag:
            # 队列队首的id出栈
            update_queue_endTime(queue_r, queue_name)
            update_timeConsume_endUp(queue_r, time_consume, time_name)

            # 后处理可以的同时，训练服务器可以开始继续处理下一条请求
            with open(reply_file, 'r', encoding='utf-8') as fid:
                raw_output = json.load(fid)['sentence']
                output_sentence = post_process(raw_output, input_sentence)

            tmp0 = tokenizer.spm_cut(output_sentence)
            rsvp = [''.join(tmp0[:(x + 1)]) for x in range(len(tmp0))]
            rsvp = [x for x in rsvp if x]  # remove
            if len(rsvp) == 0:
                rsvp = ['OutputEmptyWarning']
            else:
                with open(reply_postProcessFile, 'w', encoding='utf-8') as fid:
                    json.dump({'sentence': rsvp[-1]}, fid, ensure_ascii=False)

        else:
            update_queue_endTime(queue_r, queue_name)
            update_timeConsume_endUp(queue_r, time_consume, time_name)
            rsvp = ['Wating for reply TimeoutError']


        print("flag:", finish_flag, "raw_output:{}\n".format(raw_output), "output setence:", rsvp[-1])
        return flask.jsonify(ok=True, rsvp=rsvp, flag=finish_flag, sentence_uid=uid)

    else:
        rsvp = ['当前排队人数过多，请稍后再点击！']
        finish_flag = False
        return flask.jsonify(ok=False, rsvp=rsvp, flag=finish_flag, sentence_uid=uid)


@app.route('/')
def index():
    return flask.render_template("Home_hu.html")

@app.route('/index_params3_usingJiebaPrint_addLoginTimeout_hu.html')
def component():
    return flask.render_template("index_params3_usingJiebaPrint_addLoginTimeout_hu.html")

class Config():
    JOBS = [{
            'id': 'job',
            'func': 'do_query',
            'args': '',
            'trigger': {'type':'cron', 'second':'*/1'}
    }]


if __name__ == '__main__':
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    app.run(host="0.0.0.0", port=28888, debug=True, threaded=True)

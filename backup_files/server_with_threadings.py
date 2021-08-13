import math
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from process_sentence import pre_process, post_process
import threading

import os
import time
import random
import obs
import sentencepiece as spm
import jieba
import json
import redis

from urllib.parse import unquote
tokenizer = None
import threading
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(5)

################################################
OFFLOAD = '00_01'       # set obs dir name_slice
url = "http://localhost:8899/inference_api"
QUEUE_NAME = 'queue{}'.format(OFFLOAD)
TIMECONSUME_NAME = 'timeConsume{}'.format(OFFLOAD)
################################################

######### redis config #########################
MAX_QUEUE_LENS = 3

def update_queue_startUp(redi, queue_name, time_name):
    # 获取当前queue长度
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)   # redi.get(queue_name)
    print(queue_list)
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
        tmp = redi.lindex(queue_name, 0)
        currentUID = int(tmp.decode('utf-8'))
        return None, currentUID, MAX_QUEUE_LENS

    # 队列长度未满，且排队时间不长，则追加队列
    elif len(queue_list) > 0 and len(queue_list) < MAX_QUEUE_LENS:
        tmp = redi.lindex(queue_name, 0)
        currentUID = int(tmp.decode('utf-8'))
        thisUID = get_sentence_UID()
        redi.rpush(queue_name, thisUID)
        return thisUID, currentUID, len(queue_list) + 1

def update_queue_endTime(redi, queue_name):
    # queue_name = 'queue01'
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    assert len(queue_list) <= MAX_QUEUE_LENS
    # 删除队列队首的uid
    redi.lpop(queue_name)
    # tmp = r.lindex(queue_name, 0)
    # currentUID = int(tmp.decode('utf-8'))
    # return currentUID

def update_timeConsume_endUp(redi, time_consume, time_name):
    KEYNAME = time_name
    queue_time = int(redi.get(KEYNAME).decode('utf-8'))
    new_timeConsume = queue_time - time_consume
    redi.set(KEYNAME, new_timeConsume)

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

def get_current_queueLens_timeConsume(redi, queue_name, time_name):
    queue_list = redi.lrange(queue_name, 0, MAX_QUEUE_LENS)
    assert len(queue_list) <= MAX_QUEUE_LENS
    queue_len = len(queue_list)
    if not queue_len == 0:
        timeConsume = int(redi.get(time_name).decode('utf-8'))
        return queue_len, timeConsume
    else:
        return 0, 0

def update_timeConsume_startUp(redi, time_consume, time_name):
    KEYNAME = time_name
    if redi.get(KEYNAME) == None:
        redi.set(KEYNAME, time_consume)
        return time_consume
    else:
        queue_time = int(redi.get(KEYNAME).decode('utf-8'))
        new_timeConsume = queue_time + time_consume
        redi.set(KEYNAME, new_timeConsume)
        return new_timeConsume

def convert_secondTime2Mins(timeConsume):
    minutes = timeConsume // 60
    min_minutes = minutes // 3
    return min_minutes, math.ceil(minutes)
#################################################

QUEUE_NAME = 'queue{}'.format(OFFLOAD)
TIMECONSUME_NAME = 'timeConsume{}'.format(OFFLOAD)

global_varible_lock = threading.Lock()
global_varible = {
    'UID': 0,
}

r = redis.StrictRedis(host='192.168.204.68', port=6379, db=0)

time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
random_num = random.randrange(100000, 999999)
GLOBAL_DIR_NAME = str(time_stamp) + "_" + str(random_num) + '-pangu-13B-inferenceAPI'

hf_obs_dir_format = lambda x: x if x.endswith('/') else (x+'/')

SCAN_SLEEP_TIME = 0.5 #every 0.5 seconds scan
SCAN_DIRECTORY = 's3://pcl-verify/yizx/serving_logs_13B_increment_flaskPassParams_releaseV2_autoid_codesTest_offload{}'.format(OFFLOAD)
bucket_name = SCAN_DIRECTORY[5:].split('/',1)[0]

hf_obs_file = lambda *x: '/'.join([SCAN_DIRECTORY, *x])
hf_work_file = lambda *x: os.path.join(GLOBAL_DIR_NAME, *x)

access_key_id = 'T5SLNRWNNNINAQBBNFJ6'
secret_access_key = 'sVEDyI8KlNKTkVBuRkgwjVyTXlygiq21pzQG9qUu'
server = '112.95.163.82'
obs_client = obs.ObsClient(access_key_id=access_key_id, secret_access_key=secret_access_key, server=server)

########### obs config###########################
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
#################################################

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

def restart_webFront_clearRedis(redi):
    tmp_list = ['queue{}'.format(OFFLOAD), 'timeConsume{}'.format(OFFLOAD)]

    for item in tmp_list:
        redi.delete(item)

def warmup():
    global tokenizer
    restart_webFront_clearRedis(r)
    # requests.DEFAULT_RETRIES = 5
    # s = requests.session()
    # s.keep_alive = False

    if not os.path.exists(hf_work_file('request')):
        os.makedirs(hf_work_file('request'))
    if not os.path.exists(hf_work_file('reply')):
        os.makedirs(hf_work_file('reply'))
    if not os.path.exists(hf_work_file('stars')):
        os.makedirs(hf_work_file('stars'))
    if not os.path.exists(hf_work_file('autoPrompt')):
        os.makedirs(hf_work_file('autoPrompt'))

    tokenizer = JIEBATokenizer('bpe_4w_pcl/vocab.vocab', 'bpe_4w_pcl/vocab.model')
    z0 = '他是苹果公司的创始人'
    tokenizer.spm_cut(z0)

    payload = {
        'input_sentence': 'userRawInput',
        'top_k': 2,
        'top_p': 0.9,
        'result_len': 256
    }
    # response = requests.request("POST", url+'test', data=payload)
    print('Finished WarmUp!')

    if obs_isfile(obs_client, hf_obs_file('current_uid.txt')):
        obs_download_file(obs_client, hf_obs_file('current_uid.txt'), hf_work_file('current_uid.txt'))
        with open(hf_work_file('current_uid.txt'), 'r') as fid:
            global_varible['UID'] = int(fid.read().strip())
    print('UID start from: {}'.format(global_varible['UID']))

def get_sentence_UID():
    try:
        global_varible_lock.acquire()
        ret = global_varible['UID']
        global_varible['UID'] += 1
    finally:
        global_varible_lock.release()
    return ret

def input_data_constrain(in_dict, MAX_RESULT_LEN=128):
    # value2float in in_dict
    input_sentence = in_dict.get('input_sentence')
    topP = in_dict.get('topP')
    topK = in_dict.get('topK')
    result_len = in_dict.get('result_len')

    if input_sentence == None:
        return '请设置合理有效的文本输入(input_sentence)'
    if result_len == None or result_len <= 0 or result_len > MAX_RESULT_LEN:
        return '请设置合理有效的生成长度(result_len∈(0, {}]中整数，非整数api会自动向下取整)'.format(MAX_RESULT_LEN)
    ## 判读topK范围是否合理，合理后，是否需要向下取整
    if topK < 0 or topK > 10:
        return '请设置合理有效的topK，范围取[0, 10]中整数，非整数api会自动向下取整'
    if topP < 0 or topP > 1:
        return '请设置合理有效的topP，范围取[0, 1)'
    if topP == 0 and topK == 0:
        return 'topK或topP至少需要设置一个大于0的值！！topK∈[0, 10], topP∈[0,1)'
    # topK向下取整
    if int(topK) != topK:
        topK = math.floor(topK)
    # topK向下取整
    if int(result_len) != result_len:
        result_len = math.floor(result_len)

    topK = int(topK)
    result_len = int(result_len)

    return_dict = {}
    return_dict['input_sentence'] = input_sentence
    return_dict['topK'] = topK
    return_dict['topP'] = topP
    return_dict['result_len'] = result_len

    return return_dict, 'Passed!'

########### define threading to exec obs reading loop ###############
class reading_loop_for_obs_reply(threading.Thread):
    def __init__(self, mainProcess, uid, timeout):
        threading.Thread.__init__(self)
        self.mainProcess = mainProcess
        self.uid = uid
        self.timeout = timeout
        self.get_reply = False

    def run(self):
        self.finished_flag = upload_then_loop_reading_obs_reply_file(self.uid, self.timeout)

def upload_then_loop_reading_obs_reply_file(uid, timeout, input_sentence):
    request_file = hf_work_file('request', f'request_{uid}.json')
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

    ###############################################################
    # waiting util obs reply_file return, then response with POST
    # 读取OBS开始计时
    start_time = time.time()
    finish_flag = False
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
        update_queue_endTime(r, QUEUE_NAME)
        update_timeConsume_endUp(r, timeout, TIMECONSUME_NAME)
        with open(reply_file, 'r', encoding='utf-8') as fid:
            raw_output = json.load(fid)['sentence']
            output_sentence = post_process(raw_output, input_sentence)
        if len(output_sentence) == 0:
            print("output_sentence is :{}".format(output_sentence))
            #self.wfile.write("{}".format('生成句子为空！').encode('utf-8'))
        else:
            with open(reply_postProcessFile, 'w', encoding='utf-8') as fid:
                json.dump({'sentence': output_sentence}, fid, ensure_ascii=False)
            print("start_sentence is:{}".format(input_sentence))
            print("output_sentence is :{}".format(output_sentence))
    else:
        update_queue_endTime(r, QUEUE_NAME)
        update_timeConsume_endUp(r, timeout, TIMECONSUME_NAME)


#####################################################################

class S(BaseHTTPRequestHandler):
    def do_GET(self):
        paths = {
            '/inference_api': {'status': 200},
        }
        if self.path in paths:
            self.send_response(200)
            self.end_headers()
            queue_len, timeConsume = get_current_queueLens_timeConsume(r, QUEUE_NAME, TIMECONSUME_NAME)
            self.wfile.write('队列长度: {}, 预计排队时间: {}s, 当前请求uid为: {}'.format(queue_len, timeConsume,
                                                                      global_varible['UID'] - 1).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        paths = {
            '/inference_api': {'status': 200},
            '/inference_result': {'status': 200},
                }
        # self.send_header('Content-Type', 'text/plain;charset=utf-8')
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        if self.path in paths:
            self.send_response(200)
            self.end_headers()
            if self.path == '/inference_api':
                # get user inputs
                res = unquote("{}".format(post_data), encoding='utf-8')
                in_dict = {}
                for item in res[2:-1].split('&'):
                    key, value = item.split('=')
                    if not key == 'input_sentence':
                        try:
                            value = float(value)
                        except:
                            self.wfile.write("请传入正确格式的{}值".format(key).encode('utf-8'))
                            return None
                    in_dict[key] = value

                # 范围约束
                return_hint = input_data_constrain(in_dict)
                if not len(return_hint) == 2:
                    self.wfile.write("{}".format(return_hint).encode('utf-8'))
                    return None
                else:
                    res_dict, _ = return_hint
                    # self.wfile.write("{}".format(res_dict).encode('utf-8'))

                    # 输入预处理，返回是否存在敏感词
                    user_in_sent = res_dict['input_sentence']
                    input_sentence, HasSensitiveWord, SensitiveWord = pre_process(user_in_sent)
                    if HasSensitiveWord:
                        self.wfile.write("{}".format('请勿输入敏感词: "{}"！'.format(SensitiveWord)).encode('utf-8'))
                        return None
                    # update queue and get timeConsume
                    uid, _, queue_lens = update_queue_startUp(r, QUEUE_NAME, TIMECONSUME_NAME)
                    queue_list = r.lrange(QUEUE_NAME, 0, MAX_QUEUE_LENS)
                    print('当前队列长度为: {}'.format(len(queue_list)))

                    topK = res_dict['topK']
                    topP = res_dict['topP']
                    result_len = res_dict['result_len']
                    # upload in_sent to obs
                    if not uid == None:
                        ############################################
                        # 队列时间预估，时间覆盖'time_consume'
                        time_consume = math.ceil(res_dict['result_len'] / 500 * 105)  # default * 135
                        timeQueueConsume = update_timeConsume_startUp(r, time_consume, TIMECONSUME_NAME)
                        print('推理此条输入预计费时 {}s.'.format(timeQueueConsume))
                        # self.wfile.write("推理此条输入预计费时 {}s.\n".format(timeQueueConsume).encode('utf-8'))
                        ############################################
                        request_file = hf_work_file('request', f'request_{uid}.json')
                        with open(request_file, 'w', encoding='utf-8') as fid:
                            tmp0 = {
                                'sentence': input_sentence,
                                'top_k': topK,
                                'top_p': topP,
                                'result_len': result_len,
                            }
                            json.dump(tmp0, fid, ensure_ascii=False)

                        # loop_thread = reading_loop_for_obs_reply(self, uid, timeQueueConsume)
                        # #threading.Thread(target=upload_then_loop_reading_obs_reply_file, args=(self, uid, timeQueueConsume))
                        # loop_thread.start()
                        # loop_thread.join()
                        # finish_flag = loop_thread.finished_flag

                        executor.submit(upload_then_loop_reading_obs_reply_file, uid, timeQueueConsume, input_sentence)
                        self.wfile.write('已收到用户request，input_sentence is :{}。\n>>> 推理中，请稍等，预计等待{}s <<<'.format(input_sentence,
                                                                                             timeQueueConsume).encode('utf-8'))
            elif self.path == '/inference_result':
                res = unquote("{}".format(post_data), encoding='utf-8')
                request_id = res[2:-1]
                request_reply_file = hf_work_file('reply', f'reply_{request_id}.json')
                try:
                    with open(request_reply_file, 'r', encoding='utf-8') as fid:
                        raw_output = json.load(fid)['sentence']
                    # self.wfile.write('output_sentence is :{}'.format(raw_output).encode('utf-8'))
                    self.wfile.write('{}'.format(raw_output).encode('utf-8'))

                except:
                    self.wfile.write('推理过程暂未完成，请稍后再重试！'.encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=S, port=8899):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting inference_api_server...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("inference_api_server closed()")
    logging.info('Stopping inference_api_server...\n')


if __name__ == '__main__':
    warmup()
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
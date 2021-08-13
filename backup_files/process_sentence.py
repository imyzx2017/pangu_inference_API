# -*- coding: utf-8 -*-
import os
import re
import json

# import string
# from zhon.hanzi import punctuation

def load_all_json(data_dir='data/yzx_sample'):
    ret = []
    hf_data = lambda *x: os.path.join(data_dir, *x)
    for x in os.listdir(hf_data()):
        tmp0 = sorted({int(y[:-5].rsplit('_',1)[1]) for y in os.listdir(hf_data(x,'reply')) if y.endswith('.json')})
        tmp1 = {y for y in tmp0 if os.path.exists(hf_data(x,'request',f'request_{y}.json'))}
        for y in tmp1:
            with open(hf_data(x,'request',f'request_{y}.json')) as fid:
                tmp2 = json.load(fid)
            with open(hf_data(x,'reply',f'reply_{y}.json')) as fid:
                tmp3 = json.load(fid)
            ret.append((tmp2['sentence'],tmp3['sentence']))
    return ret

def load_sensitive_word():
    #https://github.com/fwwdn/sensitive-stop-words
    hf_file = lambda *x: os.path.join('sensitive-stop-words', *x)
    assert os.path.exists(hf_file())
    with open(hf_file('广告.txt'), 'r', encoding='utf-8') as fid:
        z0 = [x.strip() for x in fid]
    with open(hf_file('政治类.txt'), 'r', encoding='utf-8') as fid:
        z1 = [x.strip().rstrip(',') for x in fid]
    with open(hf_file('涉枪涉爆违法信息关键词.txt'), 'r', encoding='utf-16') as fid:
        z2 = [x.strip() for x in fid]
    with open(hf_file('色情类.txt'), 'r', encoding='utf-8') as fid:
        z3 = [x.strip().rstrip(',') for x in fid]

    with open(hf_file('反动词库（gbk编码）.txt'), 'r', encoding='gbk') as fid:
        z4 = [x.strip().rstrip(',') for x in fid]
    with open(hf_file('民生词库（gbk编码）.txt'), 'r', encoding='gbk') as fid:
        z5 = [x.strip().rstrip(',') for x in fid]
    with open(hf_file('暴恐词库.txt'), 'r', encoding='utf-8') as fid:
        z6 = [x.strip().rstrip(',') for x in fid]
    with open(hf_file('敏感词.txt'), 'r', encoding='utf-8') as fid:
        z7 = [x.strip().rstrip(',') for x in fid]
    ret = z0 + z1 + z2 + z3 + z4 + z5 + z6 #+ z7
    return ret

def load_politics_word():
    hf_file = lambda *x: os.path.join('sensitive-stop-words', *x)
    with open(hf_file('政治类.txt'), 'r', encoding='utf-8') as fid:
        z1 = [x.strip().rstrip(',') for x in fid]
    return z1

# https://regex101.com/
tmp0 = [
    ('^[?,)？，）。呢吧]+', ''),
    ('[(（,，问]+$', ''),
    ('^什么？', ''),
    (r'^几\n', ''),
    ('^之一', ''),
    ('我想去死', ''),
    ('>!', ''),
    ('、 、', '、'),
    (r"(.{3,})\1+", r'\1', re.DOTALL) #remove repeated sentence #https://stackoverflow.com/a/43681243
    # (r"(.{3,}).{,30}\1+", r'\1', re.DOTALL)
]
POST_PROCESS_RE_RULE = [(re.compile(x[0]),x[1]) if len(x)==2 else (re.compile(x[0],flags=x[2]),x[1]) for x in tmp0]
tmp0 = {
    ',': '，',
    ':': '：',
    '?': '？',
    '⁇': '',
}

def SetSensitiveList():
    return ['鹏城实验室', 'www', '妈的', '我日', 'WWW', 'com', 'net', '毛主席', '港独', '澳独', '疆独',
            '习大大', 'mzd', 'MD', 'md', '淦', '习主席', '习jp', '毛zd', '澳门', '台湾', '台独',
            '蔡英文', 'sb', 'SB', 'fw', 'FW', '藏独', '操了']

def SetSensitivePolicyList():
    return ['习大大', '习近平', '毛泽东', '毛主席', '江泽民', '胡锦涛', '温家宝']

POST_PROCESS_TRANSLATE_TABLE = {ord(k):v for k,v in tmp0.items()}
SENSITIVE_WORD = load_sensitive_word()
manualSensitiveList = SetSensitiveList()
politics_words = load_politics_word()
manualSensitivePolicyList = SetSensitivePolicyList()



def pre_process(z0):
    tmp0 = z0
    z0 = z0.strip() #remove whitespace
    z0 = z0.replace(" ", "")

    # # 针对05.12蒋老师提到的这个标点归一化的问题，修正输入输出的标点不归一化（小问题）
    # z0 = z0.translate(POST_PROCESS_TRANSLATE_TABLE)

    # 是否存在敏感词
    for item in manualSensitiveList:
        if not z0.find(item) == -1:
            return tmp0, True, item

    for x in SENSITIVE_WORD:
        if not z0.find(x) == -1:
            return tmp0, True, x

    for x in politics_words:
        if not z0.find(x) == -1:
            return tmp0, True, x

    # make "re" patterns
    for p_word in manualSensitivePolicyList:
        pattern = ''
        for each_str in p_word:
            pattern += each_str + '(.*?)'
        pattern = pattern[:-5]
        r_pattern = re.compile(r'{}'.format(pattern))
        if not r_pattern.search(z0) == None:
            return tmp0, True, p_word

    return tmp0, False, ""

def post_process(z0, input_s=None):
    z0 = z0.strip()

    # # 针对05.12蒋老师提到的这个标点归一化的问题，修正输入输出的标点不归一化（小问题）
    # z0 = z0.translate(POST_PROCESS_TRANSLATE_TABLE)

    z0 = '\n'.join([x for x in z0.split('\n') if len(x)]) #remove empty line

    ################################################
    # add manual pattern ### to split examples
    if (input_s is not None) and ('###' in input_s):
        z0 = z0[: z0.index('###')+3]
    ################################################

    if (input_s is not None) and ('问：' in input_s or '答：' in input_s):
        #QA-type sentence
        if '答：' in z0:
            z0 = z0[z0.index('答：'):]
        if '问：' in z0:
            z0 = z0[:z0.index('问：')]

    for x in SENSITIVE_WORD:
        z0 = z0.replace(x, '')
    for x in manualSensitiveList:
        z0 = z0.replace(x, '')

    while True:
        n0 = 0
        for pattern, rep in POST_PROCESS_RE_RULE:
            z0,tmp0 = pattern.subn(rep, z0)
            n0 = n0 + tmp0
        if n0==0:
            break

    tmp0 = [x.strip() for x in z0.split('\n')]
    tmp1 = [True]+[x!=y for x,y in zip(tmp0[:-1],tmp0[1:])]
    z0 = '\n'.join([x for x,y in zip(tmp0,tmp1) if y])
    z0 = z0.strip() #remove whitespace again

    # # cut last sentence by punctuation #############
    # punctuations01 = punctuation
    # punctuations02 = string.punctuation
    # punctuations = punctuations01 + punctuations02

    punctuations_manual = ['。', '》', ')', '}', ']', '!', '?', '？', '>', '】', '」', '』', '）', '~']

    z0_part1 = z0[:-20]
    z0_part2 = z0[-20:]

    transposeZ0 = z0_part2[::-1]
    transposeCut_index = 0
    SCAN_STRING_FLAG = False
    for index, transpose_str in enumerate(transposeZ0):
        if SCAN_STRING_FLAG:
            break
        else:
            for punc in punctuations_manual:
                if transpose_str == punc:
                    transposeCut_index = index
                    SCAN_STRING_FLAG = True
                    break
    transposeZ0New = transposeZ0[transposeCut_index:]
    result = transposeZ0New[::-1]

    z0 = z0_part1 + result
    #######################################

    if (input_s is not None) and ('上联：' in input_s):
        # couplet-type sentence
        # tmp0 = input_s[(input_s.index('上联：')+3):]
        tmp0 = input_s[(input_s.index('上联：') + 3):]
        if '下联' in tmp0:
            len_couplet = len(tmp0[:tmp0.index('下联')].strip())
        else:
            len_couplet = len(tmp0.strip())
        if '下联:' in z0:
            #"下联:\n上联:欢天喜地度佳节下联:喜气洋洋迎新年"
            tmp0 = z0.index('下联:')
            while '上联：' in z0[tmp0:(tmp0+len_couplet+3)]:
                tmp0 = z0.index('下联:', tmp0+1)
            z0 = z0[tmp0:(tmp0+len_couplet+3)]

        elif '下联：' in z0:
            #"下联:\n上联:欢天喜地度佳节下联:喜气洋洋迎新年"
            tmp0 = z0.index('下联：')
            while '上联：' in z0[tmp0:(tmp0+len_couplet+3)]:
                tmp0 = z0.index('下联：', tmp0+1)
            z0 = z0[tmp0:(tmp0+len_couplet+3)]

    elif (input_s is not None) and ('上联:' in input_s):
        # couplet-type sentence
        # tmp0 = input_s[(input_s.index('上联：')+3):]
        tmp0 = input_s[(input_s.index('上联:') + 3):]
        if '下联' in tmp0:
            len_couplet = len(tmp0[:tmp0.index('下联')].strip())
        else:
            len_couplet = len(tmp0.strip())
        if '下联:' in z0:
            #"下联:\n上联:欢天喜地度佳节下联:喜气洋洋迎新年"
            tmp0 = z0.index('下联:')
            while '上联:' in z0[tmp0:(tmp0+len_couplet+3)]:
                tmp0 = z0.index('下联:', tmp0+1)
            z0 = z0[tmp0:(tmp0+len_couplet+3)]

        elif '下联：' in z0:
            #"下联:\n上联:欢天喜地度佳节下联:喜气洋洋迎新年"
            tmp0 = z0.index('下联：')
            while '上联:' in z0[tmp0:(tmp0+len_couplet+3)]:
                tmp0 = z0.index('下联：', tmp0+1)
            z0 = z0[tmp0:(tmp0+len_couplet+3)]

    return z0

if __name__=='__main__':
    # z0 = load_all_json()
    #
    # z0 = [(x,pre_process(x),y) for x,y in z0]
    #
    # z0 = [(x, y, z, post_process(z,y)) for x,y,z in z0]
    #
    # #save for human check
    # with open('example.json', 'w') as fid:
    #     json.dump(z0, fid, ensure_ascii=False, indent=4)

    # z0 = '我的父亲是一个非常好的人,他对我的影响很大。他是一个非常好的人,他对我的影响很大。他是一个非常好的人,他对我的影响很大。他asdasadasavfdvas'

    z0 = '上联:春雨润人间,社会和谐万象新。下联:春风送暖入农家,和谐社会喜洋洋。'
    tmp0 = '上联:春雨润人间,社会和谐万象新'

    tmp = pre_process(tmp0)
    print(tmp[0])
    res = post_process(z0, tmp[0])
    print(res)

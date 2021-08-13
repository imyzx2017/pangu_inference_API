import re
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


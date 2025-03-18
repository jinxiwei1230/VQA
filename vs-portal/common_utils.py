import time
import pymysql
import vs_common
import redis

class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None

def  getProcessRecordById(id):
    connection = pymysql.connect(**vs_common.mysql_conf)
    with connection:
        with connection.cursor() as cursur:
            sql = "SELECT * FROM `table_name` WHERE `id`=%s"
            cursur.execute(sql, (id,))
            result = cursur.fetchone()
            return result

def updateProcessStateById(id, state):
    connection = pymysql.connect(**vs_common.mysql_conf)
    with connection:
        with connection.cursor() as cursur:
            sql = "UPDATE `table_name` SET `state` = %s, updated_time = %s WHERE `id`= %s"
            cursur.execute(sql, (state, time.time()*1000,id,))
        connection.commit()

def updateVideoInfoById(id, fps, width, height, duration, size):
    connection = pymysql.connect(**vs_common.mysql_conf)
    with connection:
        with connection.cursor() as cursur:
            sql = "UPDATE `table_name` SET video_fps = %s, \
            video_width = %s, video_height = %s, video_duration = %s, video_size = %s, \
            updated_time = %s WHERE `id`= %s"
            cursur.execute(sql, (fps, width, height, duration, size, time.time()*1000,id,))
        connection.commit()

def updateProcessRecordById(id, state, record):
    connection = pymysql.connect(**vs_common.mysql_conf)
    with connection:
        with connection.cursor() as cursur:
            sql = "UPDATE `table_name` SET `state` = %s, `record` = %s, updated_time = %s WHERE `id`= %s"
            cursur.execute(sql, (state, record, time.time()*1000, id,))
        connection.commit()

def updateVideoSituationNum(id, num):
    connection = pymysql.connect(**vs_common.mysql_conf)
    with connection:
        with connection.cursor() as cursur:
            sql = "UPDATE `table_name` SET `video_situations_num` = %s WHERE `id`= %s"
            cursur.execute(sql, (num, id,))
        connection.commit()

# def updateFaceRecogSituationNum(id):
#     connection = pymysql.connect(**vs_common.mysql_conf)
#     processed_num = getProcessRecordById(id)[14] + 1
#     with connection:
#         with connection.cursor() as cursur:
#             sql = "UPDATE `table_name` SET `face_re_process_num` = %s WHERE `id`= %s"
#             cursur.execute(sql, (processed_num, id))
#         connection.commit()


def updateFaceRecogSituationNum(id):
    r = redis.Redis(**vs_common.redis_conf)
    r.incr(id)

def getFaceRecogProcessed(id):
    r = redis.Redis(**vs_common.redis_conf)
    return int(r.get(id))

def delRedisinfo(id):
    r = redis.Redis(**vs_common.redis_conf)
    r.delete(id)
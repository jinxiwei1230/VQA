import redis

import vs_common

r = redis.Redis(**vs_common.redis_conf)
# r.delete('55')
# r.expire()
# #r.incr('test1')
# print(r.get('55'))
# print(int(r.get('55')) == 20)
#
# for i in range():
#     print(r.get(51))

print(r.get(51))

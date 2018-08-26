import hashlib
import random
import urllib3
import json
from urllib.parse import quote
import re

http=urllib3.PoolManager()
appKey = '598fa0364b74abcf'
secretKey = '8EiwlX1Fds87OYCrZ12C4CNqPHcmgkyS'

# httplib=client
# httpClient = None
baseurl = 'openapi.youdao.com/api'
fromLang = 'EN'
toLang = 'zh-CHS'
salt = random.randint(1, 65536)



def tran(names):
  vals=[]
  for q in names:
    q=str(q)
    q=re.sub("[\s+\.\!\/_,$%^*(+\"\':;\-\]\[·\\\<\>@#&(){}]",' ',q)
    sign = appKey+q+str(salt)+secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = baseurl+'?appKey='+appKey+'&q='+quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    print(myurl)
    res=http.request('GET',myurl)
    res=json.loads(res.data)
    val=''
    if (res.__contains__('web')):
      for o in res['web']:
        if o['key'] == q:
          val=o['value']
        else:
          val=res['web'][0]['value']
    else:
      val=res['translation']
    vals.append(val)
    # print(json.loads(res.data)['translation'])
  return vals
# print(tran('Labrador retriever'))
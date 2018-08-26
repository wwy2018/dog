import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, Response, jsonify
# from classify import train
from yo import train
from trans import tran
import urllib3
import json
http=urllib3.PoolManager()

app = Flask(__name__)

appid='wx3753d7b32375de85'
# @app.route('/getAccessToken')
# def getAccessToken():
#   url='https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid='+appid+'&secret='+appsecret
#   res=http.request('GET',url)
#   ACCESS_TOKEN=json.loads(res.data)['access_token']
#   return ACCESS_TOKEN
@app.route('/')
def test():
  return 'test'
@app.route('/getDogName', methods=['POST'])
def getDogName():
  file=request.files['file'].read()
  res=train(file)
  names=[]
  for r in res:
    names.append(r['class'])
  print(names)
  trans=tran(names)
  print(trans)
  return Response(json.dumps({'trans':trans,'name':names}), mimetype='application/json')
  # return jsonify({'trans':trans[0],'name':name})

if __name__ == '__main__':
  # app.run(port=5000,host='0.0.0.0',ssl_context=('./ssl/1534134180555.pem','./ssl/1534134180555.key'))
  app.run(port=5000,debug=True,host='0.0.0.0')
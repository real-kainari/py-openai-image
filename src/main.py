import os
import openai
import base64
import cv2
import numpy

from dotenv import load_dotenv

# evnファイルを読み込む
load_dotenv()

# APIキーの設定
openai.api_key = os.environ['OPENAI_API_KEY']

# プロンプトを入力
prompt = input('prompt : ')

# プロンプトを送信
response = openai.Image.create(
    prompt=prompt,
    n=1,
    size='512x512',
    response_format='b64_json'
)

# レスポンスを取得
b64_json = response.data[0].b64_json

# Base64で画像にデコード
img_binary = base64.b64decode(b64_json)
jpg = numpy.frombuffer(img_binary, dtype=numpy.uint8)
img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

# 画像表示
cv2.imshow('OpenAI Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
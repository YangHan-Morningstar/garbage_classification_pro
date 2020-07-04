import get_image_results
import pypinyin
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import json
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open('data/name_vocabulary.json') as file:
    name_vocabulary = json.load(file)
with open('data/dict_name_num.json') as file:
    dict_name_num = json.load(file)

# 文本分析
def text_classifier(str):
    model = load_model('pretrained_models/Bid_model.h5')
    labels = {1: '可回收垃圾', 2: '有害垃圾', 4: '湿垃圾', 8: '干垃圾', 16: '大型垃圾'}

    inputing = []
    for i in str:
        i_pinin = pypinyin.pinyin(i, style=pypinyin.NORMAL)[0][0]
        if i_pinin in name_vocabulary:
            inputing.append(dict_name_num[i_pinin])

    inputing = pad_sequences([inputing], maxlen=17, padding='post')
    prediction = model.predict(inputing)
    ans = np.argmax(prediction, axis=-1)[0]
    return labels[ans]

while True:
    print('输入图片路径')
    image_path = input()
    time_begin = time.time()
    object_detection_results, image_classifier_results = get_image_results.get_results(image_path)
    time_end = time.time()

    if len(object_detection_results) != 0:
        for name in object_detection_results:
            # 转中文
            print('物体种类（英文）是' + name)
    else:
        print('物体种类（中文）是' + image_classifier_results[0], '所属垃圾类别为' + text_classifier(image_classifier_results[0]))
    
    print("The duration is", time_end - time_begin)





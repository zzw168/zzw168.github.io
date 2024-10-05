# coding:utf-8

import os
import json
import numpy as np


def json2txt(path_json, path_txt):
    with open(path_json, 'r', encoding='gb18030') as path_json:
        jsonx = json.load(path_json)
        with open(path_txt, 'w+') as ftxt:
            for shape in jsonx:
                xy_list = np.array(shape['content'])
                label = str(shape["labels"]["labelName"])
                strxy = ''
                for xy in xy_list:
                    strxy += str(int(xy["x"])) + '/' + str(int(xy["y"])) + ','
                strxy += " %s" % label
                ftxt.writelines(strxy + "\n")


dir_json = 'jsons/'
dir_txt = 'txts/'
if not os.path.exists(dir_txt):
    os.makedirs(dir_txt)
list_json = os.listdir(dir_json)
for cnt, json_name in enumerate(list_json):
    print('cnt=%d,name=%s' % (cnt, json_name))
    if ".json" in json_name:
        path_json = dir_json + json_name
        path_txt = dir_txt + json_name.replace('.json', '.txt')
        # print(path_json, path_txt)
        json2txt(path_json, path_txt)

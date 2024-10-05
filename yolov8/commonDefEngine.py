import socket
import copy

import cv2
import os

import time

from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import urlparse, parse_qs

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

from skimage.metrics import structural_similarity as ssim

color = (0, 255, 0)

names = {0: 'yellow', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'green', 6: 'Brown',
         7: 'black',
         8: 'pink', 9: 'White', 10: 'xx_s_yello', 11: 'xx_s_white', 12: 'xx_s_red',
         13: 'xx_s_black'}


def reset_frame_size(integration_frame_array, target_width, target_height):
    resized_images = []

    for i, item in enumerate(integration_frame_array):
        resized_img = cv2.resize(item, (target_width, target_height))

        cv2.putText(resized_img, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255), thickness=2)

        resized_images.append(resized_img)

    return resized_images


def setSaveImg(selfs, post_params, extra_param_config):
    # 提取表单参数
    form_field1 = {}

    if post_params.get('saveImgNum', False):
        extra_param_config['saveImgNum'] = post_params.get('saveImgNum')[0]
    else:
        extra_param_config['saveImgNum'] = ''

    if post_params.get('saveImgPath', False):
        extra_param_config['saveImgPath'] = post_params.get('saveImgPath')[0]
    else:
        extra_param_config['saveImgPath'] = './testimg3'

    if post_params.get('saveImgRun', False):
        extra_param_config['saveImgRun'] = post_params.get('saveImgRun')[0]
    else:
        extra_param_config['saveImgRun'] = 0

    if post_params.get('saveBackground', False):

        extra_param_config['saveBackground'] = post_params.get('saveBackground')[0]
    else:
        extra_param_config['saveBackground'] = 0

    setResponseContent(selfs, "ok")


def z_udp(send_data, address):
    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(send_data.encode('utf-8'), address)
    except socket.error as e:
        print(f"socket 错误: {e}")
    finally:
        udp_socket.close()


# def z_udp(send_data, address):
#     # 1. 创建udp套接字
#     udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     # 2. 准备接收方的地址
#     # dest_addr = ('127.0.0.1', 8080)
#     # 4. 发送数据到指定的电脑上
#     udp_socket.sendto(send_data.encode('utf-8'), address)
#     # 5. 关闭套接字
#     udp_socket.close()


def ssimImage(img1, img2):
    # 加载两张图片
    imageA = io.imread(img1)

    imageB = img2

    imageB = resize(imageB, imageA.shape, anti_aliasing=True)

    # 如果图像是RGBA，去除Alpha通道
    if imageA.shape[-1] == 4:
        imageA = imageA[..., :3]
    if imageB.shape[-1] == 4:
        imageB = imageB[..., :3]

    # 转换为灰度图像
    imageA_gray = rgb2gray(imageA)
    imageB_gray = rgb2gray(imageB)

    # 将像素值从[0, 1]映射到[0, 255]，并转换数据类型为uint8
    imageA_gray = (imageA_gray * 255).astype(np.uint8)
    imageB_gray = (imageB_gray * 255).astype(np.uint8)

    imageA_gray = cv2.Canny(imageA_gray, 100, 200)
    imageB_resized = cv2.Canny(imageB_gray, 100, 200)

    return ssim(imageA_gray, imageB_resized, data_range=imageB_resized.max() - imageB_resized.min())


def is_camera_black_screen(frame):
    frame22 = frame.copy()
    # 将图像转换为灰度图像以便处理
    gray = cv2.cvtColor(frame22, cv2.COLOR_BGR2GRAY)
    # 计算图像的平均亮度
    mean_brightness = cv2.mean(gray)[0]
    # 设置阈值来判断图像是否几乎全是黑色
    threshold = 10

    if mean_brightness < threshold or int(mean_brightness) == 0:
        return True
    return False


# 接收http相关


def setbbb(query_params):
    # 从参数字典中获取特定参数的值
    pass


def setResponseContent(selfs, con):
    # 发送响应
    selfs.send_response(200)
    selfs.send_header('Content-type', 'text/html')
    selfs.end_headers()

    # 构造响应内容
    response_content = con
    selfs.wfile.write(response_content.encode('utf-8'))


def getRunToggle(selfs, extra_param_config):
    if extra_param_config['run_toggle']:

        num = '1'

        for i, cap in enumerate(extra_param_config['flipList']):

            if extra_param_config['allCamerasTurnedOnList'].get(str(i)) is not None:
                if extra_param_config['allCamerasTurnedOnList'][str(i)] == 0:
                    num = '2'
                    break

        extra_param_config['closeCountdown'] = int(time.time()) + 600

        setResponseContent(selfs, num)

    else:
        setResponseContent(selfs, "0")


def setRunToggle(selfs, post_params, extra_param_config):
    run_toggle2 = '1'
    if post_params.get('run_toggle', '1'):
        run_toggle2 = post_params.get('run_toggle')[0]

    if str(run_toggle2) == '1':

        extra_param_config['run_toggle'] = True
        RequestHandler.extra_param_config['closeCountdown'] = int(time.time()) + 600

    else:
        extra_param_config['run_toggle'] = False

    setResponseContent(selfs, "ok")


# 定义全局变量
class RequestHandler(BaseHTTPRequestHandler):
    extra_param_config = None

    def do_GET(self):
        # 解析URL，提取查询字符串中的参数
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        setbbb(query_params)

        # 发送响应
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        RequestHandler.extra_param_config['closeCountdown'] = int(time.time()) + 600

        # 构造响应内容，包含GET请求参数的值
        response_content = f"ok"

        self.wfile.write(response_content.encode('utf-8'))

    def do_POST(self):

        # 读取 POST 数据
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        # 解析 POST 数据
        post_params = parse_qs(post_data)

        requestType = post_params.get('requestType', ['saveImg'])[0]

        if requestType == 'saveImg':
            setSaveImg(self, post_params, RequestHandler.extra_param_config)

        elif requestType == 'set_run_toggle':
            setRunToggle(self, post_params, RequestHandler.extra_param_config)

        elif requestType == 'get_run_toggle':
            getRunToggle(self, RequestHandler.extra_param_config)

    # 上面是http处理


def run_server(host, port, config):
    try:
        server_address = (host, port)

        RequestHandler.extra_param_config = config

        server = HTTPServer(server_address, RequestHandler)
        print(f'Starting server at {host}:{port}, use <Ctrl-C> to stop')
        server.serve_forever()
    except Exception as e:
        print(f"Server crashed with exception: {e}. Restarting server...")
        run_server(host, port)


# 处理帧 操作
def processing_frames(cap_num, cap, ifsaveimg, config, image_queue):
    global names

    ret, frame = cap.read()


    print(type(frame))

    if not ret:
        print("读取帧失败")
        config['allCamerasTurnedOnList'][str(cap_num)] = 0

        return False
    else:
        config['allCamerasTurnedOnList'][str(cap_num)] = 1

    if is_camera_black_screen(frame):
        config['allCamerasTurnedOnList'][str(cap_num)] = 0
        return 'black'

    indexes_num_equals_2 = [key for key, value in config['imgSimilarList'].items() if value['num'] == cap_num]

    if indexes_num_equals_2:

        indexes_num_equals_1 = indexes_num_equals_2[0]

        if config['flipList'][str(cap_num)] != -2:
            frame = cv2.flip(frame, config['flipList'][str(cap_num)])


        if indexes_num_equals_1 == '5':
            height2, width2 = frame.shape[:2]
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame = cv2.resize(frame, (height2, width2))



        frame1 = copy.copy(frame)



        time11 = time.time()

        result = config['model'].predict(frame)
        results = config['model'].visualize(result)



        # results = config['model'].infer(frame)

        time22 = time.time()

        print(str(int((time22 - time11) * 1000)) + "---------111")



        qiu_array = []
        qiu_array1 = []
        if len(results) != 0:  # 整合球的数据

            names = {0: 'yellow', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'green', 6: 'Brown',
                     7: 'black',
                     8: 'pink', 9: 'White', 10: 'xx_s_yello', 11: 'xx_s_white', 12: 'xx_s_red',
                     13: 'xx_s_black'}

            for r in results:
                if int(r[5].item()) < 10:
                    array = [r[0], r[1], r[2], r[3], r[5], names[r[4]]]

                    array1 = [r[0], r[1], r[2], r[3], r[5], names[r[4]], int(indexes_num_equals_1)]

                    qiu_array.append(array)
                    qiu_array1.append(array1)

            if len(qiu_array) != 0:  # 整合球的数据
                for r in qiu_array:
                    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 0, 255),
                                  thickness=2)
                    cv2.putText(frame, "%s %s" % (round(r[4], 2), r[5]),
                                (r[0], r[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 0, 255), thickness=3)

        if len(qiu_array):  # 处理范围内跟排名
            z_udp(str(qiu_array1), config['server_address_data'])  # 发送数据

        # 判断是否保存截图
        if len(qiu_array) or int(config['saveBackground']) == 1:

            # 使用 in 关键字判断是否包含子字符串
            if str(indexes_num_equals_1) in config['saveImgNum'] and int(config['saveImgRun']) == 1 and ifsaveimg:

                if int(config['saveBackground']) == 1:

                    image_queue.put((f"{config['saveImgPath']}/",
                                     f"{str(indexes_num_equals_1) + '-' + str(int(time.time() * 1000))}.jpg", frame1))

                else:

                    result_dict = {sublist[-1]: sum(sublist[:2]) for sublist in qiu_array}

                    for k, v in result_dict.items():

                        if abs(config['nameLists'][str(cap_num)][k] - v) > 4:
                            config['nameLists'][str(cap_num)][k] = v

                            image_queue.put((f"{config['saveImgPath']}/",
                                             f"{str(indexes_num_equals_1) + '-' + str(int(time.time() * 1000))}.jpg",
                                             frame1))

                            break

    return frame


def processing_frames_end(res1, i, cap, cap_array, integration_frame_array, config):
    if isinstance(res1, str):
        if res1 == 'black':
            cap.release()
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            set_cap(cap, config)
            cap_array[i] = cap
            print('重连摄像头')

    else:
        if res1 is not False:
            integration_frame_array.append(res1)
        else:
            cap.release()
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            set_cap(cap, config)
            cap_array[i] = cap
            print('重连摄像头')


# 初始化摄像头
def init_camera(cap_num_list, config):
    cap_array = []

    for k, i in enumerate(cap_num_list):

        cap_num = i
        cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f'无法打开摄像头{cap_num}')
            continue

        print('设置摄像头：' + str(cap_num))
        set_cap(cap, config)
        ret, frame = cap.read()
        if not ret:
            print(f'无法读取画面{cap_num}')
        # cv2.imwrite(f"./test/{cap_num}.jpg", frame)
        cap_array.append(cap)

    return cap_array


# 设置摄像头参数
def set_cap(cap, config):  # 设置视频截图参数（不压缩图片，节省压缩过程时间）

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['w'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['h'])
    cap.set(cv2.CAP_PROP_FPS, config['fps'])
    W1 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H1 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps1 = cap.get(cv2.CAP_PROP_FPS)
    print(f"设置{W1}*{H1}  FPS={fps1}")


def reconfig(config, resConfigtime):
    if int(time.time()) - resConfigtime >= 10:
        # 读取文件内容
        with open('cameraPositionConfig.txt', 'r') as file:
            data_str = file.read()

        # 将字符串转换为字典格式
        config['imgSimilarList'] = eval(data_str)
        resConfigtime = int(time.time())

        for i, cap in config['imgSimilarList'].items():
            config['flipList'][str(cap['num'])] = cap['flip']

    return resConfigtime

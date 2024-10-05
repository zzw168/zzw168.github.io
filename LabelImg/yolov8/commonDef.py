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
from matplotlib.patches import Polygon





listside = {}
listsidepolygon = {}

for i in range(12):
    key = i+1

    with open(f"./border/{key}.txt", 'r') as file:
        # 读取文件内容
        file_content = file.read()

    # 将字符串拆分为点对
    coord_pairs = file_content.split(',')

    # 将点对转换为元组列表
    points = [tuple(map(int, pair.split('/'))) for pair in coord_pairs]
    listside[key] = points
    # 输出结果

    # 创建一个多边形对象
    listsidepolygon[key] = np.array(points, np.int32)







color = (0, 255, 0)

names = {0: 'yellow', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'green', 6: 'Brown',
         7: 'black',
         8: 'pink', 9: 'White', 10: 'xx_s_yello', 11: 'xx_s_white', 12: 'xx_s_red',
         13: 'xx_s_black'}


namesColor = {0: (227, 204, 74), 1: (38, 102, 174), 2: (182, 48, 81), 3: (80, 57, 139), 4: (255, 118, 103),
              5: (40, 141, 107), 6: (106, 48, 73), 7: (42, 46, 57), 8: (254, 87, 179), 9: (208, 219, 205)}
alpha = int(0.5 * 255)
namesColor = {k: (v[2], v[1], v[0], alpha) for k, v in namesColor.items()}

addside = 5




# 判断目标的每个顶点是否在矩形内
def is_point_in_polygon(point, polygon):
    return polygon.contains_point(point)



def reset_frame_size(integration_frame_array, target_width, target_height,config):
    resized_images = []

    for i, item in enumerate(integration_frame_array):
        resized_img = cv2.resize(item, (target_width, target_height))

        indexes_num_equals_2 = [key for key, value in config['imgSimilarList'].items() if value['num'] == i]

        indexes_num_equals_1 = indexes_num_equals_2[0]

        cv2.putText(resized_img, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255), thickness=3)

        cv2.putText(resized_img, str(indexes_num_equals_1), (125, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255), thickness=3)

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
def processing_frames(cap_num, frame, config, image_queue, iftensor=False):
    global names,namesColor,listsidepolygon



    if frame is False:
        print("读取帧失败")
        config['allCamerasTurnedOnList'][str(cap_num)] = 0
        return False
    else:
        config['allCamerasTurnedOnList'][str(cap_num)] = 1

    if isinstance(frame, str) and frame == 'black':
        config['allCamerasTurnedOnList'][str(cap_num)] = 0
        return 'black'


    indexes_num_equals_2 = [key for key, value in config['imgSimilarList'].items() if value['num'] == cap_num]

    if indexes_num_equals_2:

        indexes_num_equals_1 = indexes_num_equals_2[0]

        if config['flipList'][str(cap_num)] != -2:
            frame = cv2.flip(frame, config['flipList'][str(cap_num)])


        # if indexes_num_equals_1 == '3':
        #     height2, width2 = frame.shape[:2]
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #
        #     frame = cv2.resize(frame, (height2, width2))


        polygon = listsidepolygon[int(indexes_num_equals_1)]
        frame1 = copy.copy(frame)


        if iftensor == False:



            results = config['model'].predict(source=frame, show=False, verbose=True, conf=0.5, iou=0.45,
                                              imgsz=config['myimgsz'])



            qiu_array = []
            qiu_array1 = []
            if len(results) != 0:
                result = results[0].boxes.data

                qiu_array1.append(int(time.time() * 1000) - int(config['runtime']))

                for r in result:

                    if int(r[5].item()) < 10:
                        array = [int(r[0].item()), int(r[1].item()), int(r[2].item()), int(r[3].item()),
                                 round(r[4].item(), 2), names[int(r[5].item())]]
                        array1 = [int(r[0].item()), int(r[1].item()), int(r[2].item()), int(r[3].item()),
                                  round(r[4].item(), 2), names[int(r[5].item())], int(indexes_num_equals_1)]




                        width = int(r[2].item()) - int(r[0].item())
                        height = int(r[3].item()) - int(r[1].item())

                        if width > 8 and height > 8:


                            x = (array[0] + array[2]) / 2
                            y = (array[1] + array[3]) / 2
                            point = (x, y)


                            Result2 = cv2.pointPolygonTest(polygon, point, False)  # -1=在外部,0=在线上，1=在内部


                            if Result2 == 1 or Result2 == 0:

                                cv2.rectangle(frame, (array[0] - addside, array[1] - addside),
                                              (array[2] + addside, array[3] + addside),
                                              namesColor[int(r[5].item())],
                                              thickness=4)
                                cv2.putText(frame, "%s" % (array[5]),
                                            (array[0], array[1] - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1,
                                            color=namesColor[int(r[5].item())], thickness=3)

                                qiu_array.append(array)
                                qiu_array1.append(array1)


        else:


            result = config['model'].predict(frame)
            results = config['model'].visualize(result)


            qiu_array = []
            qiu_array1 = []
            if len(results) != 0:  # 整合球的数据

                qiu_array1.append(int(time.time() * 1000) - int(config['runtime']))

                names = {0: 'yellow', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'green', 6: 'Brown',
                         7: 'black',
                         8: 'pink', 9: 'White', 10: 'xx_s_yello', 11: 'xx_s_white', 12: 'xx_s_red',
                         13: 'xx_s_black'}

                for r in results:
                    if int(r[5].item()) < 10:
                        array = [r[0], r[1], r[2], r[3], r[5], names[r[4]]]

                        array1 = [r[0], r[1], r[2], r[3], r[5], names[r[4]], int(indexes_num_equals_1)]


                        width = int(r[2]) - int(r[0])
                        height = int(r[3]) - int(r[1])

                        if width > 8 and height > 8:


                            x = (array[0] + array[2]) / 2
                            y = (array[1] + array[3]) / 2
                            point = (x, y)


                            # Result2 = cv2.pointPolygonTest(polygon, point, False)  # -1=在外部,0=在线上，1=在内部

                            Result2 = 1

                            if Result2 == 1 or Result2 == 0:

                                cv2.rectangle(frame, (array[0] - addside, array[1] - addside),
                                              (array[2] + addside, array[3] + addside),
                                              namesColor[r[4]],
                                              thickness=4)
                                cv2.putText(frame, "%s" % (array[5]),
                                            (array[0], array[1] - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1,
                                            color=namesColor[r[4]], thickness=3)

                                qiu_array.append(array)
                                qiu_array1.append(array1)


                # if len(qiu_array) != 0:  # 整合球的数据
                #     for r in qiu_array:
                #         cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 0, 255),
                #                       thickness=2)
                #         cv2.putText(frame, "%s %s" % (round(r[4], 2), r[5]),
                #                     (r[0], r[1] - 5),
                #                     cv2.FONT_HERSHEY_SIMPLEX,
                #                     fontScale=1,
                #                     color=(0, 0, 255), thickness=3)

        if len(qiu_array):  # 处理范围内跟排名

            # print(qiu_array1)
            z_udp(str(qiu_array1), config['server_address_data'])  # 发送数据

        # 判断是否保存截图
        if len(qiu_array) > 1 or int(config['saveBackground']) == 1:

            # 使用 in 关键字判断是否包含子字符串
            if str(indexes_num_equals_1) in config['saveImgNum'] and int(config['saveImgRun']) == 1 :

                image_queue.put((f"{config['saveImgPath']}/",f"{str(indexes_num_equals_1) + '-' + str(int(time.time() * 1000))}.jpg",frame1))

    return frame



def processing_frames_end(res1, i, cap_array, integration_frame_array, config):

    if isinstance(res1, str):
        if res1 == 'black':
            cap_array[i].release()
            cap_array[i] = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            set_cap(cap_array[i], config)

            print('重连摄像头')

    else:
        if res1 is not False:
            integration_frame_array.append(res1)
        else:
            cap_array[i].release()
            cap_array[i] = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            set_cap(cap_array[i], config)

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
        # f = f"{cap_num}.jpg"
        # cv2.imwrite(f, frame)
        cap_array.append(cap)

    return cap_array


# 设置摄像头参数
def set_cap(cap, config):  # 设置视频截图参数（不压缩图片，节省压缩过程时间）

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['w'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['h'])
    cap.set(cv2.CAP_PROP_FPS, config['fps'])



    cap.set(cv2.CAP_PROP_BRIGHTNESS, 101)  # 设置亮度，范围一般在0到255之间
    cap.set(cv2.CAP_PROP_CONTRAST, 110)  # 设置对比度，范围一般在0到255之间
    cap.set(cv2.CAP_PROP_SATURATION, 91)  # 设置饱和度，范围一般在0到255之间
    cap.set(cv2.CAP_PROP_HUE, 10)  # 设置色调，范围一般在0到255之间
    cap.set(cv2.CAP_PROP_GAIN, 34)  # 设置增益，范围一般在0到255之间
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # 设置曝光时间，通常曝光时间是负值，值越大曝光越长

    # 设置白平衡 U 和 V 值
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)  # 设置 U 值
    cap.set(cv2.CAP_PROP_SHARPNESS, 22)
    # 尝试设置摄像头的焦距
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, 20)  # 焦距范围可能取决于摄像头，一般是 0 到最大值
    # 尝试设置摄像头的变焦
    zoom_supported = cap.set(cv2.CAP_PROP_ZOOM, 20)  # 变焦因子（放大倍数），取决于摄像头
    # cap.set(cv2.CAP_PROP_SETTINGS, 1)





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




def start_udp_server(host , port , config ):

    # 创建UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定套接字到指定的地址和端口
    server_address = (host, port)
    udp_socket.bind(server_address)

    print(f"UDP服务器已启动，监听地址: {host} 端口: {port}")
    config['runtime'] = int(time.time() * 1000)



    try:
        while True:
            # 接收数据
            data, client_address = udp_socket.recvfrom(1024)


            print(data)

            config['runtime'] = int(time.time() * 1000)

            print(config['runtime'])

    except KeyboardInterrupt:
        print("服务器已关闭")

    finally:
        # 关闭套接字
        udp_socket.close()




def get_frame( cap, i, config):

    ret, frame = cap.read()

    # f = f"{i}.jpg"
    # cv2.imwrite(f, frame)

    if not ret:
        print("读取帧失败")
        config['allCamerasTurnedOnList'][str(i)] = 0
        return i, False

    config['allCamerasTurnedOnList'][str(i)] = 1

    if is_camera_black_screen(frame):
        config['allCamerasTurnedOnList'][str(i)] = 0
        return i, 'black'

    return i, frame
from ultralytics import YOLO
import cv2
import threading
import time
import queue
import numpy as np


from commonDef import run_server, processing_frames, reset_frame_size, init_camera, reconfig, processing_frames_end,start_udp_server,listside,get_frame
from SaveTheImageToAShared import image_queue_worker
import concurrent.futures

import python_trt as  myTr

# pip install scikit-image numpy matplotlib
# pip install ultralytics
# pip install opencv-python
# pip install scikit-image


target_width, target_height = 960, 540  # 1920, 1000
canvas = np.zeros((2160, 1920, 3), dtype=np.uint8)

# 定义多边形的坐标
rect_coords_dict = listside

# 原始图像尺寸
original_width = 1920
original_height = 1080

# 根据目标尺寸进行比例缩放
def scale_coords(coords, target_width, target_height, original_width, original_height):
    return [(int(x * target_width / original_width), int(y * target_height / original_height)) for x, y in coords]

# 生成等比例缩放的遮罩
masks = {1:'',2:'',3:'',4:'',5:'',6:'',7:'',8:'',9:'',10:'',11:'',12:''}
for i in range(1, 12):  # Assuming you have 6 sets of coordinates
    coords = rect_coords_dict[i]
    scaled_coords = scale_coords(coords, target_width, target_height, original_width, original_height)
    mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    pts = np.array(scaled_coords, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=True, color=(0, 255, 0), thickness=6)
    masks[i] = mask




def run():
    global target_width, target_height, canvas, config, mask

    # 创建队列
    image_queue = queue.Queue()
    # 创建并启动工作线程
    image_queue_threading = threading.Thread(target=image_queue_worker, args=(image_queue,))
    image_queue_threading.start()

    # config['model'] = YOLO("best.pt")
    config['model'] = myTr.Detector(model_path=b"./best8.engine", dll_path="./trt/yolov8.dll")

    # 正式

    cap_array = []
    cv2.namedWindow("display", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("display", 1100, 800)

    cap_num_list = [0,1,2,3,4,5,6,7]
    cap_array = init_camera(cap_num_list, config)

    print('开始')

    # 获取当前时间戳
    startImgtime = 0
    resConfigtime = 0

    while True:

        if int(time.time()) > config['closeCountdown']:
            config['run_toggle'] = False

        if config['run_toggle']:

            time11 = time.time()
            
            if config['model'] == False:


                # config['model'] = YOLO("best.pt")
                config['model'] = myTr.Detector(model_path=b"./best8.engine", dll_path="./trt/yolov8.dll")

                canvas = np.zeros((2160, 1920, 3), dtype=np.uint8)

                cv2.namedWindow("display", cv2.WINDOW_GUI_EXPANDED)  # 创建一个具有扩展GUI的窗口
                cv2.resizeWindow("display", 1100, 800)  # 设置窗口大小为1100x1000像素



            resConfigtime = reconfig(config, resConfigtime)




            framearr = [False,False,False,False,False,False,False,False]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(get_frame, cap, i, config)
                    for i, cap in enumerate(cap_array)
                ]

                for future in concurrent.futures.as_completed(futures):
                    cap_num, result = future.result()

                    framearr[cap_num] = result


            time33 = time.time()
            # print(str(int((time33 - time11) * 1000)) + "---------11")


            integration_frame_array = []
            for i, frame in enumerate(framearr):

                # print(i)
                # print(frame)

                res1 = processing_frames(i, frame, config, image_queue,True)

                processing_frames_end(res1, i, cap_array, integration_frame_array,config)

            # unfinished_tasks = image_queue.unfinished_tasks
            # print(f"Unfinished tasks: {unfinished_tasks}")


            time11_1 = time.time()

            resized_images = reset_frame_size(integration_frame_array, target_width, target_height,config)





            if config['imgSimilarList'].get('7') is not None and config['imgSimilarList']['7']['num'] < len(resized_images):
                canvas[0:540, 0:960] = resized_images[config['imgSimilarList']['7']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[0:540, 0:960] = cv2.addWeighted(canvas[0:540, 0:960],1, masks[6], 1, 0)
            if config['imgSimilarList'].get('5') is not None and config['imgSimilarList']['5']['num'] < len(resized_images):
                canvas[540:1080, 0:960] = resized_images[config['imgSimilarList']['5']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[540:1080, 0:960] = cv2.addWeighted(canvas[540:1080, 0:960],1, masks[6], 1, 0)

            if config['imgSimilarList'].get('3') is not None and config['imgSimilarList']['3']['num'] < len(resized_images):
                canvas[1080:1620, 0:960] = resized_images[config['imgSimilarList']['3']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[1080:1620, 0:960] = cv2.addWeighted(canvas[1080:1620, 0:960],1, masks[6], 1, 0)

            if config['imgSimilarList'].get('1') is not None and config['imgSimilarList']['1']['num'] < len(resized_images):
                canvas[1620:2160, 0:960] = resized_images[config['imgSimilarList']['1']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[1620:2160, 0:960] = cv2.addWeighted(canvas[1620:2160, 0:960],1, masks[6], 1, 0)



            if config['imgSimilarList'].get('8') is not None and config['imgSimilarList']['8']['num'] < len(resized_images):
                canvas[0:target_height, target_width:1920] = resized_images[config['imgSimilarList']['8']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[0:target_height, target_width:1920] = cv2.addWeighted(canvas[0:target_height, target_width:1920],1, masks[6], 1, 0)

            if config['imgSimilarList'].get('6') is not None and config['imgSimilarList']['6']['num'] < len(resized_images):
                canvas[target_height:1080, target_width:1920] = resized_images[config['imgSimilarList']['6']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[target_height:1080, target_width:1920] = cv2.addWeighted(canvas[target_height:1080, target_width:1920],1, masks[6], 1, 0)

            if config['imgSimilarList'].get('4') is not None and config['imgSimilarList']['4']['num'] < len(resized_images):
                canvas[1080:1620, target_width:1920] = resized_images[config['imgSimilarList']['4']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[1080:1620, target_width:1920] = cv2.addWeighted(canvas[1080:1620, target_width:1920],1, masks[6], 1, 0)

            if config['imgSimilarList'].get('2') is not None and config['imgSimilarList']['2']['num'] < len(resized_images):
                canvas[1620:2160, target_width:1920] = resized_images[config['imgSimilarList']['2']['num']]  # 左上部
                # if config['show_edges']:
                #     canvas[1620:2160, target_width:1920] = cv2.addWeighted(canvas[1620:2160, target_width:1920],1, masks[6], 1, 0)




            time11_2 = time.time()


            # print(str(int((time11_2 - time11_1) * 1000)) + "--------33")



            time22 = time.time()
            print(str(int((time22 - time11) * 1000)) + "---------")

            cv2.imshow("display", canvas)
            cv2.waitKey(1)


        else:

            del canvas
            canvas = False

            del config['model']
            config['model'] = False

            cv2.destroyAllWindows()

            time.sleep(2)


if __name__ == "__main__":

    config = {
        'myimgsz': 1920,
        'imgSimilarList': {},
        'allCamerasTurnedOnList': {},
        'flipList': {
            '0': -2,
            '1': -2,
            '2': -2,
            '3': -2,
            '4': -2,
            '5': -2,
            '6': -2,
            '7': -2,
        },
        'model': None,
        'server_address_data': ("192.168.0.59", 19734),
        'nameLists': {},
        'saveImgRun': 0,
        'saveImgNum': '',
        'saveImgPath': './testimg3',
        'saveBackground': 0,
        'run_toggle': True,
        'closeCountdown': int(time.time()) + 600,
        'w': 1920,
        'h': 1080,
        'fps': 15,
        'runtime': 0,
        'show_edges': True,
    }


    run_server_var = threading.Thread(target=run_server, args=('', 8080, config))
    run_server_var.start()


    run_server_udp = threading.Thread(target=start_udp_server, args=('', 19735, config))
    run_server_udp.start()

    run()
import os
import cv2

def image_queue_worker(image_queue):
    while True:
        # 从队列中获取图片数据
        save_path,image_name,image_data = image_queue.get()

        if os.path.exists(save_path):

            cv2.imwrite(save_path+image_name, image_data)

        else:
            print(f"硬盘地址 {save_path} 不存在")

        image_queue.task_done()


import os
os.getcwd()

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import cv2
import glob



#json için import şeysileri
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_json(object_prediction_list):
    list = []
    for i in object_prediction_list:
        # bbox = i[0]
        # score = i[2]
        # category = i[3]
        # print("bbox = " , bbox ,"\n" , "score = ",score,"\n","category = " ,category,"\n")
        print(i)
        print(type(i))
        

#Doublev3.pt
model_path = "weights/doublev3.pt"
#image_paths = glob.glob("dataset/*.*")

image_paths = glob.glob("dataset/*.*")

#drone_car.pt





#Detection for doublev3.pt
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path=model_path,
    confidence_threshold=0.5,
    device="cpu", # or 'cuda:0'
    image_size= 640
)

detection_model.model.agnostic =  True

for i , image_path in enumerate(image_paths):
    #Resmi bölüm yapmadan tek seferde tespiti için aşağıdaki yorum satırı kullanılıyor
    result = get_prediction(image_path, detection_model)
    #result.export_visuals(export_dir="demo_data/tekno/", file_name= "yolov5_image{}".format(i))

    """result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height = 400, #0.625 ile çarptım
    slice_width = 500, #0.3125 ile çarptım 
    overlap_height_ratio = 0.2, 
    overlap_width_ratio = 0.2,
    postprocess_class_agnostic= True,

)"""

    #Tespit edilen nesnelerin özelliklerini bastırmak için
    list = result.object_prediction_list
    save_json(list)
    print(type(list))
    
    result.export_visuals(export_dir="demo_data/tespit/", file_name= "sliced_image{}".format(i))




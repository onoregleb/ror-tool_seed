import sys
sys.path.append('yolov5')

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from math import ceil
from yolov5.utils.general import non_max_suppression

def img_preproc(img):
  img = torch.from_numpy(img)/255.0
  img = img.unsqueeze(0)
  img = torch.permute(img, (0, 3, 1, 2))
  return img

def get_prediction(img, model):
  pred = model(img)
  pred = non_max_suppression(pred)
  markup = pd.DataFrame(pred[0].numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
  return markup

def markup_img(img, markup):
  for i in range(len(markup)):
    x1, y1 = markup.iloc[i]['xmin'], markup.iloc[i]['ymin']
    x2, y2 = markup.iloc[i]['xmax'], markup.iloc[i]['ymax']
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)

def draw_model_markup(model, image):
  img = img_preproc(image)
  markup = get_prediction(img, model)
  markup_img(image, markup)
  return markup

def format_change(image, target_size):
  height, width, _ = image.shape

  # Проверяем, нужно ли поворачивать изображение
  if height > width:
    # Поворачиваем изображение на 90 градусов
    image = np.rot90(image)

  rows = ceil(image.shape[0] / target_size[1])
  cols = ceil(image.shape[1] / target_size[0])
  # if image.shape[0] % target_size[0] != 0 or image.shape[1] % target_size[1] !=0:
  image = stretch_image(image, (cols * target_size[0], rows * target_size[1]))
  # rows = image.shape[0] // target_size[0]
  # cols = image.shape[1] // target_size[1]
  image_list = []
  for row in range(rows):
    for col in range(cols):
      # Вырежьте кусок изображения
      start_x = col * target_size[0]
      start_y = row * target_size[1]
      end_x = start_x + target_size[0]
      end_y = start_y + target_size[1]
      cropped_image = image[start_y:end_y, start_x:end_x]
      # Сгенерируйте имя для сохраняемого файла (например, image_0_0.jpg)
      # output_filename = f'image_{target_size[0]}x{target_size[1]}_{row}_{col}_{input_filename}'
      image_list.append(cropped_image)
  return image_list, (cols * target_size[0], rows * target_size[1])


def combine_images(image_list, target_size):
  """
  Функция для объединения нарезанных изображений в одно большое изображение.

  Аргументы:
  image_list (list): Список нарезанных изображений.
  target_size (tuple): Размер выходного изображения в формате (высота, ширина).

  Возвращает:
  numpy.ndarray: Объединенное изображение.
  """
  num_rows = target_size[1] // image_list[0].shape[0]
  num_cols = target_size[0] // image_list[0].shape[1]

  combined_image = np.zeros((num_rows * image_list[0].shape[0], num_cols * image_list[0].shape[1], 3), dtype=np.uint8)

  idx = 0
  for r in range(num_rows):
    for c in range(num_cols):
      start_row = r * image_list[0].shape[0]
      end_row = start_row + image_list[idx].shape[0]
      start_col = c * image_list[0].shape[1]
      end_col = start_col + image_list[idx].shape[1]
      combined_image[start_row:end_row, start_col:end_col] = image_list[idx]
      idx += 1

  return combined_image

def stretch_image(original_image, new_size):
  original_image = Image.fromarray(original_image)
  output_image = original_image.resize(new_size, Image.NEAREST)
  return np.array(output_image)

def get_diameter(markup):
  markup['diameter'] = (markup['xmax'] - markup['xmin'] + markup['ymax'] - markup['ymin'])/2


def pipeline(image_dir, model_dir, yolo_dir, subimg_size=(640, 640), progress_callback=None):
    nparr = np.fromstring(image_dir, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model_path = model_dir
    yolo_path = yolo_dir
    model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local', force_reload=True)
    sub_imgs, new_size = format_change(img, subimg_size)
    total_subimages = len(sub_imgs)
    markups = []
    processed_subimages = 0
    for subimg in sub_imgs:
        markup = draw_model_markup(model, subimg)
        markups.append(markup)
        processed_subimages += 1
        progress = int((processed_subimages / total_subimages) * 100)
        if progress_callback:
            progress_callback(progress)
    print(len(sub_imgs))
    reconstructed_img = combine_images(sub_imgs,new_size)
    reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
    try:
        full_markup = pd.concat(markups)
    except ValueError:
        full_markup
    get_diameter(full_markup)
    diams = full_markup['diameter']
    return reconstructed_img, diams.values



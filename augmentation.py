import os
from PIL import Image, ImageStat, ImageOps
import random
import math
import tqdm

def get_average_color(image):
    """画像の平均色を計算する"""
    stat = ImageStat.Stat(image)
    # 平均色を取得（RGB）
    r, g, b = map(int, stat.mean)
    return (r, g, b)

def rotate_image(image, angle):
    """画像を指定された角度で回転させ、平均色で余白を埋める"""
    average_color = get_average_color(image)
    rotated_image = image.rotate(angle, expand=True, fillcolor=average_color)
    return rotated_image

def crop_max_rectangle(rotated_image, orig_width, orig_height, angle=0):
    """回転した画像から最大の長方形を切り出す"""
    rotated_width, rotated_height = rotated_image.size

    # 回転により「見える」最大の長方形のサイズを計算
    angle_cos = abs(math.cos(math.radians(angle)))
    angle_sin = abs(math.sin(math.radians(angle)))
    crop_width = int(min(orig_width * angle_cos + orig_height * angle_sin, rotated_width))
    crop_height = int(min(orig_width * angle_sin + orig_height * angle_cos, rotated_height))

    # 画像の中心から長方形を切り出す
    left = (rotated_width - crop_width) // 2
    top = (rotated_height - crop_height) // 2
    cropped_rect = rotated_image.crop((left, top, left + crop_width, top + crop_height))
    return cropped_rect

def crop_square(cropped_rect_image, left, top, crop_size):
    """ランダムな正方形を切り出す"""
    return cropped_rect_image.crop((left, top, left + crop_size, top + crop_size))

def apply_random_flip(image, is_horizontal):
    """画像にランダムなフリップ（水平または垂直）を適用する"""
    if is_horizontal:
        return ImageOps.mirror(image)  # 水平フリップ
    return image

def process_images(source_path, target_path, output_path_source, output_path_target, size=(448, 448), num_copies=4):
    source_image = Image.open(source_path)
    target_image = Image.open(target_path)
    orig_source_width, orig_source_height = source_image.size
    orig_target_width, orig_target_height = target_image.size

    for i in range(num_copies):
        angle = random.uniform(-20, 20)
        rotated_source = rotate_image(source_image, angle)
        rotated_target = rotate_image(target_image, angle)

        # フリップ処理を適用
        is_horizontal = random.choice([True, False])
        flipped_source = apply_random_flip(rotated_source, is_horizontal)
        flipped_target = apply_random_flip(rotated_target, is_horizontal)

        cropped_source_rect = crop_max_rectangle(flipped_source, orig_source_width, orig_source_height, angle)
        cropped_target_rect = crop_max_rectangle(flipped_target, orig_target_width, orig_target_height, angle)
        # cropped_source_rect.save(os.path.join(output_path_source, f"{os.path.basename(source_path).split('.')[0]}_{i}_rect.jpg"))

        cropped_source_width, cropped_source_height = cropped_source_rect.size
        cropped_target_width, cropped_target_height = cropped_target_rect.size
        
        source_max_square_size = min(cropped_source_height, cropped_source_width)
        target_max_square_size = min(cropped_target_height, cropped_target_width)
        crop_ratio = random.uniform(0.8, 1.0)
        crop_source_size = int(source_max_square_size * crop_ratio)
        crop_target_size = int(target_max_square_size * crop_ratio)

        left_source = random.randint(0, cropped_source_width - crop_source_size)
        top_source = random.randint(0, cropped_source_height - crop_source_size)
        left_target = left_source * orig_target_width // orig_source_width
        top_target = top_source * orig_target_height // orig_source_height
        final_source = crop_square(cropped_source_rect, left_source, top_source, crop_source_size).resize(size)
        final_target = crop_square(cropped_target_rect, left_target, top_target, crop_target_size).resize(size)

        # フォルダが存在しない場合は作成
        os.makedirs(output_path_source, exist_ok=True)
        os.makedirs(output_path_target, exist_ok=True)
        final_source.save(os.path.join(output_path_source, f"{os.path.basename(source_path).split('.')[0]}_{i}.jpg"))
        final_target.save(os.path.join(output_path_target, f"{os.path.basename(target_path).split('.')[0]}_{i}.jpg"))


if __name__ == '__main__':
    # 各フォルダのパスを適宜設定
    source_folder = 'dataset/dataset/source' # 頭に/をつけると絶対パスになってしまうので注意
    target_folder = 'dataset/dataset/target'
    output_folder = 'dataset/dataset/output'
    output_path_source = os.path.join(output_folder, 'source')
    output_path_target = os.path.join(output_folder, 'target')

    # 画像処理
    for image_name in tqdm.tqdm(os.listdir(source_folder)):
        if image_name.endswith('.jpg'):
            source_path = os.path.join(source_folder, image_name)
            target_path = os.path.join(target_folder, image_name)
            process_images(source_path, target_path, output_path_source, output_path_target)

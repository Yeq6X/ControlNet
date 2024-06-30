import json
import cv2
import numpy as np

from torch.utils.data import Dataset

from augmentation import rotate_image, apply_random_flip, crop_max_rectangle, crop_square
from PIL import Image
import random



class MyDataset(Dataset):
    def __init__(self, dataset_name, augment=True):
        self.dataset_name = dataset_name
        self.augment = augment
        self.data = []
        with open(f'./training/{dataset_name}/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(f'./training/{self.dataset_name}/' + source_filename)
        target = cv2.imread(f'./training/{self.dataset_name}/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.augment:
            # 画像のオーグメンテーションを適用
            size=(448, 448)
            source_image = Image.fromarray(source) # NumPy配列をPILの画像に変換
            target_image = Image.fromarray(target)
            
            orig_source_width, orig_source_height = source_image.size
            orig_target_width, orig_target_height = target_image.size

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

            source = np.array(final_source) # PILの画像をNumPy配列に変換
            target = np.array(final_target)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == '__main__':
    dataset = MyDataset()
    print(len(dataset))
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)

    for i, item in enumerate(dataloader):
        # 画像を表示
        import matplotlib.pyplot as plt
        plt.imshow(item['jpg'][0])
        plt.figure()
        plt.imshow(item['hint'][0])
        plt.show()
        print(item['txt'][0])
        print(item['jpg'].shape)
        print(item['hint'].shape)
        if i == 0:
            break
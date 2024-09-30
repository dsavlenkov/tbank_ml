import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка предобученной модели Mask R-CNN
def load_maskrcnn_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Предсказание маски и класса объекта с использованием Mask R-CNN
def predict_mask(image, model, device):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    if len(outputs['masks']) == 0:
        print("Object not found")
        return None, None
    
    mask = outputs['masks'][0, 0].cpu().numpy()
    class_id = outputs['labels'][0].item()
    
    return mask, class_id

# Постобработка маски
def postprocess_mask(mask, image_size):
    mask = cv2.resize(mask, image_size)
    mask_bin = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_blurred = cv2.GaussianBlur(mask_bin.astype(np.float32), (15, 15), 0)
    mask_blurred = np.clip(mask_blurred, 0, 1)
    
    return mask_blurred

# Применение маски для удаления фона
def apply_mask(image, mask):
    image_np = np.array(image)
    mask_3c = np.stack([mask]*3, axis=-1)
    mask_resized = cv2.resize(mask_3c, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked_image = image_np * mask_resized
    return masked_image

# Замена фона
def replace_background(image, mask, background_type="solid", color=(255, 255, 255), background_image=None):
    image_np = np.array(image)
    mask_3c = np.stack([mask]*3, axis=-1)

    if background_type == "solid":
        background = np.full_like(image_np, color, dtype=np.uint8)
    elif background_type == "image" and background_image is not None:
        background = cv2.resize(background_image, (image_np.shape[1], image_np.shape[0]))
    else:
        raise ValueError("Invalid background type or background image missing")
    
    mask_inv = 1 - mask_3c
    final_image = (image_np * mask_3c + background * mask_inv).astype(np.uint8)
    return final_image

# Генерация описания товара
def generate_description(class_name, model, tokenizer):
    prompt = f"This is a {class_name} product that features"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

# Список классов COCO
COCO_CLASSES = [ ... ]  # Добавьте ваш список классов COCO здесь

def process_images(input_folder, output_folder, background_type="solid", color=(255, 255, 255), background_image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_maskrcnn_model().to(device)
    
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(500):  # Обработка изображений с 0.jpg по 499.jpg
        image_file = f"{i}.jpg"
        image_path = os.path.join(input_folder, image_file)

        if not os.path.exists(image_path):  # Проверяем существование файла
            print(f"Image {image_file} not found, skipping.")
            continue

        image = Image.open(image_path).convert('RGB')
        mask, class_id = predict_mask(image, model, device)

        if mask is None:
            print(f"Object not found in image {image_file}")
            continue
        
        mask = postprocess_mask(mask, (image.width, image.height))
        masked_image = apply_mask(image, mask)

        background_image = None
        if background_image_path:
            background_image = cv2.imread(background_image_path)
        
        final_image = replace_background(masked_image, mask, background_type, color, background_image)

        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        cv2.imwrite(output_image_path, final_image)
        print(f"Image saved: {output_image_path}")
        
        # Проверяем, что class_id в допустимом диапазоне
        if 1 <= class_id <= len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id - 1]  # Получение названия класса
            description = generate_description(class_name, gpt_model, tokenizer)
        else:
            print(f"Invalid class_id {class_id} for image {image_file}. Skipping description generation.")
            continue  # Пропускаем генерацию описания, если class_id недопустим
        
        description_file = os.path.join(output_folder, f"description_{image_file}.txt")
        with open(description_file, 'w') as f:
            f.write(description)
        print(f"Description saved: {description_file}")


# Пример использования
if __name__ == "__main__":
    input_folder = "sirius_data/sirius_data"  # Папка с исходными изображениями
    output_folder = "output_images"  # Папка для сохранения обработанных изображений
    
    process_images(input_folder, output_folder, background_type="solid", color=(220, 220, 220))

import cv2
import os

# 画像フォルダのパス
folder_path = "C:/Users/suzushiro/py/sample"
output_folder = "C:/Users/suzushiro/py/output"

# 顔検出用のカスケード分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# フォルダ内の画像を処理
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 顔の中心を計算
        center_x = x + w // 2
        center_y = y + h // 2
        # 512x512の範囲を計算
        crop_x = max(0, center_x - 256)
        crop_y = max(0, center_y - 256)
        crop_img = img[crop_y:crop_y+512, crop_x:crop_x+512]
        # 出力
        cv2.imwrite(os.path.join(output_folder, f"cropped_{filename}"), crop_img)
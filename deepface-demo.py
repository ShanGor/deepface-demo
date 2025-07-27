from deepface import DeepFace
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
custom_model_path = os.path.join(current_dir, "models")
os.environ['DEEPFACE_HOME'] = custom_model_path

# 0.4 means 60%, it is [(1-0.4) * 100 %]
threshold = 0.4

# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'  # 重置颜色


def generate_compared_images(doc_img, selfie_img, result, task_id, similarity_score):
    import cv2
    import numpy as np

    # 读取两张图片
    img1 = cv2.imread(doc_img)
    img2 = cv2.imread(selfie_img)

    # 获取人脸区域坐标
    region1 = result['facial_areas']['img1']
    region2 = result['facial_areas']['img2']

    # 在图片上绘制人脸区域矩形框
    if region1:
        x, y, w, h = region1['x'], region1['y'], region1['w'], region1['h']
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if region2:
        x, y, w, h = region2['x'], region2['y'], region2['w'], region2['h']
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 调整图片大小以便并排显示
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    # 调整图片尺寸以匹配
    img1_resized = cv2.resize(img1, (width, height))
    img2_resized = cv2.resize(img2, (width, height))

    # 水平拼接两张图片
    comparison_img = np.hstack((img1_resized, img2_resized))

    # 添加标签
    cv2.putText(comparison_img, 'Document', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison_img, 'Selfie', (width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 根据验证结果添加对勾或叉号
    verified = result['verified']
    if verified:
        # 绘制绿色对勾
        center_x = width  # 两张图片连接处
        center_y = height // 2

        # 对勾的三个点
        pt1 = (center_x - 50, center_y)
        pt2 = (center_x - 10, center_y + 40)
        pt3 = (center_x + 60, center_y - 30)

        # 绘制较粗的线条（通过绘制多次稍微偏移的线条）
        for i in range(-1, 2):
            for j in range(-1, 2):
                cv2.line(comparison_img, (pt1[0] + i, pt1[1] + j), (pt2[0] + i, pt2[1] + j), (0, 255, 0), 3)
                cv2.line(comparison_img, (pt2[0] + i, pt2[1] + j), (pt3[0] + i, pt3[1] + j), (0, 255, 0), 3)
    else:
        # 绘制红色叉号
        center_x = width  # 两张图片连接处
        center_y = height // 2

        # 叉号的两个对角线
        pt1 = (center_x - 40, center_y - 40)
        pt2 = (center_x + 40, center_y + 40)
        pt3 = (center_x - 40, center_y + 40)
        pt4 = (center_x + 40, center_y - 40)

        # 绘制较粗的线条（通过绘制多次稍微偏移的线条）
        for i in range(-1, 2):
            for j in range(-1, 2):
                cv2.line(comparison_img, (pt1[0] + i, pt1[1] + j), (pt2[0] + i, pt2[1] + j), (0, 0, 255), 3)
                cv2.line(comparison_img, (pt3[0] + i, pt3[1] + j), (pt4[0] + i, pt4[1] + j), (0, 0, 255), 3)

    # 添加验证结果文本
    result_text = "PASSED" if verified else "FAILED"
    text_color = (0, 255, 0) if verified else (0, 0, 255)  # 绿色或红色
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    text_x = (comparison_img.shape[1] - text_size[0]) // 2
    text_y = height - 30
    cv2.putText(comparison_img, result_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)
    text_y = text_y - 30 - text_size[1]
    cv2.putText(comparison_img, similarity_score, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

    # 保存对比图片
    output_path = os.path.join(current_dir, "test-cases", "result", f"{task_id}.jpg")
    cv2.imwrite(output_path, comparison_img)

    print(f"Comparison image saved to: {output_path}")


def verify_selfie(doc_img, selfie_img, task_id):
    # mark the start time
    start = time.time()
    try:
        result = DeepFace.verify(model_name="Facenet512",
                                 distance_metric="cosine",
                                 threshold=threshold,
                                 detector_backend='opencv',
                                 img1_path=doc_img,
                                 enforce_detection=True,
                                 img2_path=selfie_img)
    except Exception as e:
        print(f"Detection failed with 'opencv': {str(e)}; Retry with 'mtcnn'")
        result = DeepFace.verify(model_name="Facenet512",
                                 distance_metric="cosine",
                                 threshold=threshold,
                                 detector_backend='mtcnn',
                                 img1_path=doc_img,
                                 enforce_detection=False,
                                 img2_path=selfie_img)
    end_time = time.time()
    print(f"Time taken to verify: {end_time - start:.2f} seconds")
    similarity_score = f"{((1 - result['distance']) * 100):.2f}%"
    if result['verified']:
        print(f"{GREEN}Passed verification for similarity threshold {((1-threshold) * 100):.2f}%")
    else:
        print(f"{RED}Failed verification for similarity threshold {((1-threshold) * 100):.2f}%")
    print(f"{RESET}Similarity Score is: {similarity_score}")
    # generate a comparison image, with left-hand side being the document,
    # and right-hand side being the image to be verified. Put rectangle on the face area.
    generate_compared_images(id_doc, selfie_img, result, task_id, similarity_score)


id_doc = os.path.join(current_dir, "test-cases\\id-doc\\EEP-AamirKhan.png")

print("\n=== Verifying document VS first selfie")
selfie = os.path.join(current_dir, "test-cases\\selfie\\Aamir Khan-1.png")
verify_selfie(id_doc, selfie, 1)

print("\n=== Verifying document VS second selfie")
selfie = os.path.join(current_dir, "test-cases\\selfie\\Aamir Khan-2.png")
verify_selfie(id_doc, selfie, 2)

print("\n=== Verifying document VS third selfie")
selfie = os.path.join(current_dir, "test-cases\\selfie\\Aamir Khan-old.png")
verify_selfie(id_doc, selfie, 3)

id_doc = os.path.join(current_dir, "test-cases\\id-doc\\EEP-DanielWu.png")
print("\n=== Verifying document VS Daniel Wu selfie, take 1")
selfie = os.path.join(current_dir, "test-cases\\selfie\\yanzu.png")
verify_selfie(id_doc, selfie, 4)

id_doc = os.path.join(current_dir, "test-cases\\id-doc\\yanzu.png")
print("\n=== Verifying document VS Daniel Wu selfie, take 2")
selfie = os.path.join(current_dir, "test-cases\\selfie\\yanzu.png")
verify_selfie(id_doc, selfie, 5)

id_doc = os.path.join(current_dir, "test-cases\\id-doc\\yanzu.png")
print("\n=== Verifying document VS Daniel Wu selfie, take 3")
selfie = os.path.join(current_dir, "test-cases\\selfie\\daniel-wu.jpg")
verify_selfie(id_doc, selfie, 6)

id_doc = os.path.join(current_dir, "test-cases\\id-doc\\yanzu.png")
print("\n=== Verifying document VS Daniel Wu selfie, take 4")
selfie = os.path.join(current_dir, "test-cases\\selfie\\daniel.webp")
verify_selfie(id_doc, selfie, 7)

id_doc = os.path.join(current_dir, "test-cases\\id-doc\\jin-cheng-wu.jpg")
print("\n=== Verifying document VS JinChengWu selfie, take 1")
selfie = os.path.join(current_dir, "test-cases\\selfie\\jin-cheng-wu-1.jpg")
verify_selfie(id_doc, selfie, 8)

print("\n=== Verifying document VS JinChengWu selfie, take 2")
selfie = os.path.join(current_dir, "test-cases\\selfie\\jin-cheng-wu.jpg")
verify_selfie(id_doc, selfie, 9)

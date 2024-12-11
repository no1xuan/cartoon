import gc
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from cartoonize import Cartoonizer

app = Flask(__name__)
cartoon_model = Cartoonizer()
cartoon_model.load_model()

def decode_image_from_base64(base64_str):
    base64_str = base64_str.strip()
    if base64_str.startswith('data:image'):
        header, base64_str = base64_str.split(',', 1)
    img_data = base64.b64decode(base64_str)
    np_array = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图像解码失败")
    return image

def get_image_mime_type(image):
    if image is None:
        raise ValueError("图像为空")
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        return 'image/jpeg'
    success, buffer = cv2.imencode('.png', image)
    if success:
        return 'image/png'
    raise ValueError("无法识别图像格式")

@app.route('/cartoon', methods=['POST'])
def generate_cartoon():
    try:
        # 获取 JSON 数据
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': '没有提供图像'}), 400

        base64_image = data['image']
        image = decode_image_from_base64(base64_image)

        # 使用生成漫画图像方法
        cartoon_image = cartoon_model.inference(image)

        # 手动释放内存
        del image
        # 强制进行垃圾回收
        gc.collect()

        mime_type = get_image_mime_type(cartoon_image)

        # 将漫画图像编码为 Base64
        _, buffer = cv2.imencode(f'.{mime_type.split("/")[1]}', cartoon_image)
        cartoon_base64 = base64.b64encode(buffer).decode('utf-8')

        # 手动释放漫画图像内存
        del cartoon_image
        gc.collect()  # 强制进行垃圾回收

        base64_with_header = f"data:{mime_type};base64,{cartoon_base64}"

        return jsonify({'cartoon_image': base64_with_header}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3008)

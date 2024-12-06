from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from cartoonize import Cartoonizer

app = Flask(__name__)
cartoon_model = Cartoonizer()
cartoon_model.load_model()

def decode_image_from_base64(base64_str):
    """
    解码 Base64 字符串为图像
    """
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
    """
    根据图像格式返回 MIME 类型
    """
    if image is None:
        raise ValueError("图像为空")

    success, buffer = cv2.imencode('.jpg', image)  # 默认先用 .jpg 编码
    if success:
        return 'image/jpeg'
    
    success, buffer = cv2.imencode('.png', image)  # 尝试 PNG 格式
    if success:
        return 'image/png'
    
    raise ValueError("无法识别图像格式")

@app.route('/cartoon', methods=['POST'])
def generate_cartoon():
    """
    生成漫画图像接口
    """
    if 'image' not in request.form:
        return jsonify({'error': '没有提供图像'}), 400

    try:
        # 获取 Base64 编码的图像数据
        base64_image = request.form['image']

        # 解码图像
        image = decode_image_from_base64(base64_image)

        # 生成漫画图
        cartoon_image = cartoon_model.inference(image)

        # 释放原始图像内存
        del image

        # 获取图像的 MIME 类型
        mime_type = get_image_mime_type(cartoon_image)

        # 编码为 Base64
        _, buffer = cv2.imencode(f'.{mime_type.split("/")[1]}', cartoon_image)
        cartoon_base64 = base64.b64encode(buffer).decode('utf-8')

        # 释放漫画图像内存
        del cartoon_image

        # 添加 MIME 类型头
        base64_with_header = f"data:{mime_type};base64,{cartoon_base64}"

        return jsonify({'cartoon_image': base64_with_header}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3008)

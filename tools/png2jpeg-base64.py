import os
import io
import base64
from PIL import Image

def compress_image_to_base64(image_path: str, output_folder: str, target_size=(400, 400), quality=75) -> None:
    """
    读取图像文件，调整大小并转换为 JPEG 格式以压缩数据，然后编码为 Base64 字符串，
    并将结果保存到指定的输出文件夹中。
    使用 LANCZOS 作为高质量重采样滤波器。

    :param image_path: 输入图像的路径。
    :param output_folder: 输出文件夹的路径。
    :param target_size: 调整后的图像尺寸（默认 256x256）。
    :param quality: 保存 JPEG 图像的质量（默认 75）。
    """
    try:
        # 检查输入图像路径是否有效
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"输入图像路径无效：{image_path}")

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 打开图像文件
        with Image.open(image_path) as img:
            # 调整图像大小并使用 LANCZOS 滤波器
            img = img.resize(target_size, Image.LANCZOS)
            buffered = io.BytesIO()
            # 将图像保存为 JPEG 格式
            img.save(buffered, format="JPEG", quality=quality)
            base64_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 保存调整后的JPEG图像
            jpeg_output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_compressed.jpeg")
            img.save(jpeg_output_path, format="JPEG", quality=quality)

            # 生成输出文件路径
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_base64.txt")

            # 将 Base64 字符串保存为文本文件
            with open(output_path, "w") as f:
                f.write(f"![image](data:image/jpeg;base64,{base64_image_data})")

            print(f"图像已成功压缩并保存到 {output_path} 和 {jpeg_output_path}")

    except Exception as e:
        print(f"压缩或编码图像 {image_path} 失败：{e}")

if __name__ == "__main__":

    try:
    # 示例用法
        input_image = f"dataCollection/Test/metaworld_soccer-v2/image1.png"
        output_folder = f"test/imgProcess"

    # 检查用户输入的路径是否为空
        if not input_image or not output_folder:
            raise ValueError("图像路径或输出文件夹路径不能为空！")

        # 调用图像压缩函数
        compress_image_to_base64(input_image, output_folder)
    except Exception as e:
        print(f"程序运行失败：{e}")

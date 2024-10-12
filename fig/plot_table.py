from PIL import Image

def create_image_matrix(image_paths, layout):
    # 计算每个子图的尺寸
    num_images = len(image_paths)
    rows = layout['rows']
    cols = layout['cols']
    images_per_row = layout['images_per_row']

    # 创建一个新的空白图像，用于拼接图片
    result_width = cols * Image.open(image_paths[0]).width
    result_height = rows * Image.open(image_paths[0]).height
    result_image = Image.new("RGBA", (result_width, result_height), (255, 255, 255, 0))

    # 将图片拼接到空白图像上
    for i, image_path in enumerate(image_paths):
        for j,j_per in enumerate(images_per_row):
            if i<=sum(images_per_row[:j+1])-1:
                row=j
                break
        col=i-sum(images_per_row[:row])
        x_offset = col * Image.open(image_paths[0]).width
        y_offset = row * Image.open(image_paths[0]).height
        img = Image.open(image_path)
        result_image.paste(img, (x_offset, y_offset))

    # 展示和保存拼接后的图像
    result_image.show()
    result_image.save("output.png")

# 假设你有一个包含图片路径的列表
image_paths = ['exp.png', 'low.png', 'low2exp.png',
               'high.png', 'high2exp.png', 'low.png',
               'low2high.png', 'low2high2exp.png']
layout = {
    'rows': 4,
    'cols': 3,
    'images_per_row': [1, 2, 2, 3]
}

# 调用函数
create_image_matrix(image_paths, layout)
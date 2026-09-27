from PIL import Image
import os

# 确保在figures目录下运行
png_files = [f for f in os.listdir('.') if f.endswith('.png')]

for png in png_files:
    img = Image.open(png).convert('RGB')  # 加convert('RGB')防止报错
    pdf_name = png.replace('.png', '.pdf')
    img.save(pdf_name, 'PDF', resolution=300)
    print(f'已转换: {png} → {pdf_name}')
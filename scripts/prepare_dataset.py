"""
课堂行为检测数据集整理脚本
将SCB数据集整理为YOLOv8训练格式
"""
import os
import shutil
import random

# 源数据集路径（使用者需先下载 SCB 数据集，把下面三个路径改为你本地解压后的目录；也可设置环境变量 SCB_HANDRISE_DIR / SCB_DISCUSS_DIR / SCB_BOWTURN_DIR）
SOURCES = {
    'handrise_read_write': os.environ.get('SCB_HANDRISE_DIR', r'<请填入 SCB5-Handrise-Read-write-2024-9-17 目录绝对路径>'),
    'discuss': os.environ.get('SCB_DISCUSS_DIR', r'<请填入 SCB5-Discuss-2024-9-17 目录绝对路径>'),
    'bow_turn_head': os.environ.get('SCB_BOWTURN_DIR', r'<请填入 SCB_BowTurnHead_20250509 目录绝对路径>'),
}

# 输出目录（默认输出到项目内 datasets/dataset，可用环境变量 SCB_OUTPUT_DIR 覆盖）
OUTPUT = os.environ.get('SCB_OUTPUT_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'dataset'))

# 类别映射（合并为6类）
CLASS_MAP = {
    'hand-raising': 0,
    'read': 1,
    'write': 2,
    'discuss': 1,       # 讨论归入阅读
    'BowHead': 4,       # 低头
    'TurnHead': 4,      # 转头归入低头
}

def find_images_and_labels(src_dir):
    """查找图片和标注文件"""
    imgs, labels = [], []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                imgs.append(fp)
            elif f.endswith('.txt'):
                labels.append(fp)
    return imgs, labels

def remap_label(label_path, class_map):
    """重新映射类别编号"""
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            # 找到对应的旧类别名
            for name, new_id in class_map.items():
                # 需要根据原始数据集的类别顺序来映射
                pass
            new_lines.append(line)
    return new_lines

def main():
    os.makedirs(os.path.join(OUTPUT, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, 'labels', 'val'), exist_ok=True)
    
    all_imgs = []
    all_labels = []
    
    for name, path in SOURCES.items():
        if not os.path.exists(path):
            print(f'[跳过] {name}: 目录不存在 {path}')
            continue
        
        # 查找图片和标注
        imgs, labels = find_images_and_labels(path)
        print(f'[发现] {name}: {len(imgs)} 张图片, {len(labels)} 个标注')
        
        # 配对图片和标注
        label_dict = {}
        for l in labels:
            base = os.path.splitext(os.path.basename(l))[0]
            label_dict[base] = l
        
        for img in imgs:
            base = os.path.splitext(os.path.basename(img))[0]
            if base in label_dict:
                all_imgs.append(img)
                all_labels.append(label_dict[base])
    
    if not all_imgs:
        print('[错误] 没有找到任何图片和标注，请检查数据集路径')
        return
    
    print(f'\n[总计] {len(all_imgs)} 对图片-标注')
    
    # 打乱并分割
    pairs = list(zip(all_imgs, all_labels))
    random.seed(42)
    random.shuffle(pairs)
    
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    
    print(f'[划分] 训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}')
    
    # 复制文件
    for img, label in train_pairs:
        shutil.copy2(img, os.path.join(OUTPUT, 'images', 'train', os.path.basename(img)))
        shutil.copy2(label, os.path.join(OUTPUT, 'labels', 'train', os.path.basename(label)))
    
    for img, label in val_pairs:
        shutil.copy2(img, os.path.join(OUTPUT, 'images', 'val', os.path.basename(img)))
        shutil.copy2(label, os.path.join(OUTPUT, 'labels', 'val', os.path.basename(label)))
    
    # 生成 data.yaml
    yaml_content = f"""train: {os.path.join(OUTPUT, 'images', 'train')}
val: {os.path.join(OUTPUT, 'images', 'val')}

nc: 6
names: ['hand-raising', 'reading', 'writing', 'using phone', 'bowing head', 'leaning over table']
"""
    yaml_path = os.path.join(OUTPUT, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f'\n[完成] 数据集已整理到: {OUTPUT}')
    print(f'[配置] data.yaml: {yaml_path}')

if __name__ == '__main__':
    main()

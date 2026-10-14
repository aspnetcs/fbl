"""
YOLOv8 课堂行为检测模型训练脚本
"""
from ultralytics import YOLO
import os

def main():
    # 数据集配置文件路径
    data_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'dataset', 'data.yaml')
    
    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        print('[错误] 数据集未准备好，请先运行 prepare_dataset.py')
        return
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=300,         # 训练轮数
        batch=8,            # 批量大小（显存不够改小）
        imgsz=640,          # 图片尺寸
        device=0,           # GPU编号
        workers=4,          # 数据加载线程
        project='runs/train',
        name='classroom_behavior',
        patience=50,        # 早停耐心值
        save=True,          # 保存模型
        save_period=50,     # 每50轮保存一次
    )
    
    # 复制最佳模型到项目目录
    best_src = 'runs/train/classroom_behavior/weights/best.pt'
    best_dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'v8-4-pose-all2', 'weights', 'best.pt')
    
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f'\n[完成] 最佳模型已复制到: {best_dst}')
    else:
        print(f'\n[警告] 未找到最佳模型: {best_src}')

if __name__ == '__main__':
    main()

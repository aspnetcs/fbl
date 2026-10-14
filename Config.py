#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径 (打包后模型在 models/best.pt, 修正原 runs/... 失效路径, GUI 也能跑)
model_path = 'models/best.pt'
# 类别名与 best.pt 内嵌 names 对齐 (索引 0-5 一致, 仅显示文本)
names = {0: 'hand-raising', 1: 'reading', 2: 'writing', 3: 'using phone', 4: 'bowing head', 5: 'leaning over table'}
# # class names
# names:  [ 'hand-raising', 'reading', 'writing','using phone', 'bowing the head', 'leaning over the table']
CH_names = ['举手','阅读','写作','使用手机','低头','靠在桌子上']
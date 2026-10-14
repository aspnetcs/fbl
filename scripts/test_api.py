import requests, json, os, sys

# 测试图片: 通过命令行参数传入，或用环境变量 TEST_IMAGE 指定
TEST_IMAGE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('TEST_IMAGE', '')
if not TEST_IMAGE or not os.path.exists(TEST_IMAGE):
    print('[用法] python test_api.py <图片绝对路径>')
    print('       或设置环境变量 TEST_IMAGE=图片路径')
    sys.exit(1)

# 1. 测试工具列表
r = requests.get('http://127.0.0.1:9800/tools/list')
tools = r.json()['tools']
print(f'可用工具({len(tools)}个):')
for t in tools:
    print(f'  - {t["name"]}: {t["description"]}')

# 3. 测试图片检测
r = requests.post('http://127.0.0.1:9800/tools/call', json={
    'name': 'detect_image',
    'arguments': {'image_path': TEST_IMAGE}
})
result = r.json()['result']
if isinstance(result, str):
    result = json.loads(result)
print(f'检测结果: {result["total"]}人, 耗时{result["time"]}')
print(f'统计: {result["stats"]}')

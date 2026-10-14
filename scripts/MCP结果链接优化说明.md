# 课堂行为检测 MCP 结果链接优化说明

## 一、改了什么

之前的痛点：
- 检测完返回的 `result_image` / `result_video` 是服务端本地路径，连接端打不开。
- 没有公开下载端点，agent 只能本地重绘后再发图。
- 依赖 ngrok 免费隧道，地址一变旧链接全挂。

本次优化：
1. **服务端自动生成在线链接**
   - 图片检测返回 `result_image_url`（在线预览）和 `result_image_download_url`（下载）。
   - 视频检测返回 `result_video_url`（在线播放）和 `result_video_download_url`（下载），以及封面图链接。
2. **新增公开只读端点 `/results/<name>`**
   - 直接浏览器打开即可在线观看/预览。
   - 加 `?download=1` 触发下载。
3. **可选接入腾讯云 COS（推荐）**
   - 配置环境变量后，检测结果自动上传到对象存储，返回 COS 公开 URL。
   - 不再依赖 ngrok/cloudflared 隧道，链接长期稳定。
4. **客户端助手脚本**
   - `classroom_mcp_client.py` 自动从文本中提取 URL、判断图片/视频、调用 MCP、输出 Markdown 总结表格。

## 二、新增返回字段

| 字段 | 含义 | 出现场景 |
|---|---|---|
| `result_image_url` | 标注图在线预览地址 | 图片/视频检测 |
| `result_image_download_url` | 标注图下载地址 | 图片/视频检测 |
| `result_video_url` | 标注视频在线播放地址 | 视频检测 |
| `result_video_download_url` | 标注视频下载地址 | 视频检测 |
| `result_image_backend` / `result_video_backend` | 存储后端：`cos` 或 `local` | 全部 |

## 三、快速重启服务

在 PowerShell 中执行项目目录下的重启脚本：

```powershell
cd "<你的项目目录>/课堂行为检测系统"
.\scripts\restart_mcp_server.ps1
```

脚本会：
- 停止占用 9800 端口的旧进程。
- 用 Python 启动新的 `mcp_server.py`（`restart_mcp_server.ps1` 会自动探测 `python.exe`，也可通过环境变量 `MCP_PYTHON` 显式指定）。
- 归档旧日志到 `scripts/server.log.<时间戳>`。
- 执行本地健康检查。

重启后，记得重新拉起 ngrok / cloudflared 隧道指向 `http://127.0.0.1:9800`（若未配置 COS）。

## 四、配置腾讯云 COS（真正稳定）

不想链接跟着 ngrok 变，强烈建议配 COS。步骤：

1. 登录 [腾讯云 COS 控制台](https://console.cloud.tencent.com/cos)，创建 Bucket，开启**公有读私有写**。
2. 创建 API 密钥，拿到 `SecretId` 和 `SecretKey`。
3. 安装 Python SDK：
   ```powershell
   python -m pip install cos-python-sdk-v5 -q
   ```
4. 在启动 MCP 服务前设置环境变量（可加入 `.env` 文件或系统环境变量）：
   ```powershell
   $env:MCP_COS_SECRET_ID="你的SecretId"
   $env:MCP_COS_SECRET_KEY="你的SecretKey"
   $env:MCP_COS_REGION="ap-guangzhou"   # 你的Bucket所在地域
   $env:MCP_COS_BUCKET="你的Bucket名称-1250000000"
   $env:MCP_COS_PREFIX="mcp-results/"    # 可选, COS中的目录前缀
   # 可选: 如果用了自定义CDN/全球加速域名
   # $env:MCP_COS_DOMAIN="https://cdn.example.com"
   ```
5. 重启 MCP 服务。

配好 COS 后，`result_*_backend` 会显示 `cos`，链接形如：
`https://yourbucket.cos.ap-guangzhou.myqcloud.com/mcp-results/17882..._detect_xxx.jpg`

只要 Bucket 在，链接就稳定。

## 五、客户端脚本用法

脚本位置：`classroom_mcp_client.py`（随客户端分发，置于你的工作目录即可）

### 1. 分析公网 URL

```powershell
python classroom_mcp_client.py "https://example.com/class.jpg"
python classroom_mcp_client.py "分析这个视频 https://example.com/class.mp4"
```

### 2. 上传本地文件分析

```powershell
python classroom_mcp_client.py --local "<你的本地图片路径>"
```

### 3. 调整视频采样间隔

```powershell
python classroom_mcp_client.py "https://example.com/class.mp4" --sample-interval 15
```

## 六、输出示例

### 图片

```markdown
## 图片检测结果
- **来源**: https://example.com/class.jpg
- **耗时**: 0.234s
- **检出人数**: 16
- **行为分布**: 阅读: 8, 写作: 5, 举手: 3

| 类型 | 在线地址 | 下载地址 |
|---|---|---|
| 标注图 | [在线观看/预览](https://...) | [下载](https://...?download=1) |

> 存储后端: `cos`
```

### 视频

```markdown
## 视频检测结果
- **来源**: https://example.com/class.mp4
- **分辨率**: 1920x1080, **时长**: 120s, **FPS**: 30
- **采样间隔**: 30
- **累计检出**: 245
- **行为分布**: 阅读: 120, 写作: 80, 举手: 45

| 类型 | 在线地址 | 下载地址 |
|---|---|---|
| 标注视频 | [在线观看/预览](https://...) | [下载](https://...?download=1) |
| 视频封面/末帧标注图 | [在线观看/预览](https://...) | [下载](https://...?download=1) |

> 存储后端: `cos`
```

## 七、注意事项

- `/results` 端点公开只读，不要存放敏感原片；原片仍走受 Bearer 保护的 `/uploaded`。
- 本地结果文件默认保留 7 天（`MCP_RESULTS_TTL_SEC` 可调），过期自动清理。
- 如果未配置 COS，链接依赖本机公网隧道（ngrok/cloudflared），稳定性由隧道决定。

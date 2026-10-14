#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课堂行为检测 MCP 服务器 (mcp 2.x 标准版, 并发安全增强)
==================================================
- 传输: SSE  (/sse, /messages/)   +   Streamable HTTP (/mcp)
- 鉴权: Bearer Token (环境变量 MCP_AUTH_TOKEN；未设置则放行并告警，仅用于本地调试)
- 并发: 阻塞的 YOLO 推理丢线程池(asyncio.to_thread) + 信号量限制同时推理数(默认2)，
        避免多个 agent 同时调视频检测时单机 CPU/GPU 被打爆
- 启动: 在已装好 torch/ultralytics 的 conda 环境(如 yolo)中执行
        uvicorn mcp_server:app --host 0.0.0.0 --port 9800
端点:
  SSE:            http://0.0.0.0:9800/sse
  Streamable HTTP:http://0.0.0.0:9800/mcp
  健康检查:        http://0.0.0.0:9800/healthz
  手动 REST(调试): http://0.0.0.0:9800/tools/list , /tools/call
"""
import sys
import os
import io
import json
import time
import uuid
import asyncio
import threading
import base64
import shutil
import subprocess
import urllib.request
import urllib.parse
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---- 路径(优先读环境变量，回退到脚本所在项目根目录，解压到任意位置都能跑) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../scripts
_PROJECT_DEFAULT = os.path.dirname(_THIS_DIR)            # 项目根
PROJECT_DIR = os.environ.get("MCP_PROJECT_DIR", _PROJECT_DEFAULT)
sys.path.insert(0, PROJECT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import Config  # noqa: E402
import detect_tools as tools  # noqa: E402

MODEL_PATH = os.environ.get(
    "MCP_MODEL_PATH", os.path.join(PROJECT_DIR, "models", "best.pt")
)
FONT_PATH = os.path.join(PROJECT_DIR, "Font", "platech.ttf")
SAVE_DIR = os.path.join(PROJECT_DIR, "save_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- 结果文件公开访问目录 ----
# 检测结果(标注图/标注视频)会复制到这里, 并通过 /results/<name> 提供公开只读访问。
# 如果启用了腾讯云COS, 则优先上传到COS, 返回COS公开URL(更稳定)。
RESULTS_DIR = os.path.join(SAVE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_TTL_SEC = int(os.environ.get("MCP_RESULTS_TTL_SEC") or "604800") or 604800

# ---- 腾讯云COS(可选) ----
COS_SECRET_ID = os.environ.get("MCP_COS_SECRET_ID", "").strip()
COS_SECRET_KEY = os.environ.get("MCP_COS_SECRET_KEY", "").strip()
COS_REGION = os.environ.get("MCP_COS_REGION", "").strip()
COS_BUCKET = os.environ.get("MCP_COS_BUCKET", "").strip()
COS_PREFIX = os.environ.get("MCP_COS_PREFIX", "mcp-results/").strip()
COS_DOMAIN = os.environ.get("MCP_COS_DOMAIN", "").strip()  # 自定义加速域名, 为空则使用默认COS域名
_cos_client = None

AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "").strip()
PORT = int(os.environ.get("MCP_PORT", "9800"))

# ---- 并发控制: 限制同时进行的推理任务数(视频检测很吃资源) ----
MAX_CONCURRENT = int(os.environ.get("MCP_MAX_CONCURRENT", "2"))
_INFER_SEM = asyncio.Semaphore(MAX_CONCURRENT)

# ---- 视频分析后台任务并发上限(与推理并发一致) ----
_JOB_SEM = threading.Semaphore(MAX_CONCURRENT)
# 短视频同步等待上限: 此时间内完成则直接返回完整结果(保持旧体验);
# 超时则回退为异步 job_id, 由调用方用 get_analysis_result 轮询(彻底绕开客户端固定超时)。
SYNC_WAIT_SEC = float(os.environ.get("MCP_SYNC_WAIT_SEC", "20"))

# ---- 后台异步任务表(视频检测长任务用) ----
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


def _job_set(job_id, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


def _job_new():
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id, "status": "running", "progress": 0.0,
            "processed_frames": 0, "total_frames": 0,
            "elapsed": 0.0, "started_at": time.time(),
            "result": None, "error": None,
        }
    return job_id


def _job_get(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return None
        j = dict(j)
    # 未完成时实时计算已耗时, 便于轮询端展示进度
    if j["status"] not in ("done", "error") and j.get("started_at"):
        j["elapsed"] = round(time.time() - j["started_at"], 1)
    return j


def _run_video_job(job_id, fp, sample_interval, save_video, target_classes):
    """后台线程: 跑耗时的视频 YOLO 分析, 结果写回 JOBS 表。
    让 MCP 工具可立即返回, 彻底绕开客户端固定超时(长视频/大文件不再被掐断)。"""
    try:
        with _JOB_SEM:
            _job_set(job_id, status="running")
            r = do_detect_video(fp, sample_interval, save_video, target_classes)
        _job_set(job_id, status="done", result=r,
                elapsed=round(time.time() - JOBS[job_id]["started_at"], 1))
    except Exception as e:
        _job_set(job_id, status="error", error=str(e))


def submit_video_analysis(fp, sample_interval=30, save_video=True, target_classes=None):
    """提交视频分析后台任务, 立即返回 job_id(不阻塞调用方)。"""
    job_id = _job_new()
    _job_set(job_id, status="queued")
    threading.Thread(
        target=_run_video_job,
        args=(job_id, fp, sample_interval, save_video, target_classes),
        daemon=True,
    ).start()
    return job_id


def resolve_video_source(video_path, job_id=None):
    """支持本地路径、局域网共享路径、http/https 直链(自动下载到 SAVE_DIR/downloads)"""
    if not video_path:
        return None, "视频路径为空"
    src = video_path.strip().strip('"').strip("'")
    low = src.lower()
    if low.startswith(("http://", "https://")):
        dl_dir = os.path.join(SAVE_DIR, "downloads")
        os.makedirs(dl_dir, exist_ok=True)
        name = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(src).path)) or "download.mp4"
        if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
            name += ".mp4"
        local = os.path.join(dl_dir, f"{int(time.time())}_{name}")
        print(f"[INFO] 下载视频: {src}", flush=True)
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3600) as resp, open(local, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if job_id and total:
                        _job_set(job_id, download_progress=f"{done/1048576:.1f}/{total/1048576:.1f} MB")
        except Exception as e:
            return None, f"下载失败: {e}"
        size_mb = os.path.getsize(local) / 1048576
        return local, {"downloaded_to": local, "size_mb": round(size_mb, 1)}
    if not os.path.exists(src):
        return None, f"文件不存在: {src}"
    return src, None


print("[INFO] 加载模型...", flush=True)
model = YOLO(MODEL_PATH, task="detect")
model(np.zeros((48, 48, 3), dtype=np.uint8))  # 预热
colors = tools.Colors()
print("[INFO] 模型就绪", flush=True)


def _get_cos_client():
    """延迟初始化腾讯云COS客户端(仅配置了COS时)。"""
    global _cos_client
    if _cos_client is not None:
        return _cos_client
    if not (COS_SECRET_ID and COS_SECRET_KEY and COS_REGION and COS_BUCKET):
        return None
    try:
        from qcloud_cos import CosConfig, CosS3Client
        config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY)
        _cos_client = CosS3Client(config)
        print("[INFO] 腾讯云COS客户端已初始化", flush=True)
        return _cos_client
    except Exception as e:
        print(f"[WARN] COS初始化失败: {e}", flush=True)
        return None


def _cleanup_results():
    """清理过期的本地结果文件(避免 save_data/results 无限膨胀)。"""
    try:
        now = time.time()
        for fn in os.listdir(RESULTS_DIR):
            fp = os.path.join(RESULTS_DIR, fn)
            if not os.path.isfile(fp):
                continue
            if now - os.path.getmtime(fp) > RESULTS_TTL_SEC:
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass


def _publish_result(local_path: str) -> dict:
    """把本地结果文件发布为可访问的URL。

    优先级:
      1. 腾讯云COS(如配置了SecretId/Key/Region/Bucket) -> 返回COS公开URL(最稳定)。
      2. 本机 /results/<name> 公开只读端点 -> 返回当前公网基址(依赖ngrok/cloudflared, 会随隧道变化)。

    返回 {"watch_url": ..., "download_url": ..., "backend": "cos"|"local"}
    """
    if not local_path or not os.path.isfile(local_path):
        return {}
    ext = os.path.splitext(local_path)[1].lower()
    base_name = os.path.basename(local_path).replace('/', '_').replace('\\', '_')
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{base_name}"
    result = {}

    # 尝试COS
    cos = _get_cos_client()
    if cos:
        try:
            key = (COS_PREFIX.rstrip("/") + "/" + safe_name).lstrip("/")
            cos.upload_file(Bucket=COS_BUCKET, LocalFilePath=local_path, Key=key, PartSize=10, MAXThread=5)
            if COS_DOMAIN:
                cos_url = f"{COS_DOMAIN.rstrip('/')}/{key}"
            else:
                cos_url = f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com/{key}"
            # 若bucket不是公开的, 生成7天有效期的签名URL作为下载; 公开的bucket可直接访问。
            # 这里统一返回公开URL(假设bucket已开公共读)。如需签名URL可额外配置。
            result = {"watch_url": cos_url, "download_url": cos_url, "backend": "cos"}
            print(f"[INFO] 结果已上传COS: {cos_url}", flush=True)
            return result
        except Exception as e:
            print(f"[WARN] COS上传失败, 回退到本机: {e}", flush=True)

    # 回退: 复制到 RESULTS_DIR, 通过 /results/<name> 访问
    _cleanup_results()
    dest = os.path.join(RESULTS_DIR, safe_name)
    try:
        shutil.copy2(local_path, dest)
    except Exception as e:
        print(f"[WARN] 复制结果文件失败: {e}", flush=True)
        return {}
    base = get_public_base()
    public_url = f"{base}/results/{safe_name}"
    result = {"watch_url": public_url, "download_url": public_url + "?download=1", "backend": "local"}
    print(f"[INFO] 结果已发布到本地: {public_url}", flush=True)
    return result


def _load_cn_font(size=22):
    """加载一个能渲染中文的字体(雅黑优先, 回退到项目自带 platech/宋体/黑体)。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        FONT_PATH,
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, 0)
            except Exception:
                continue
    return ImageFont.load_default()


def _make_writer(path, fps, w, h):
    """创建视频写入器: 优先 H264(avc1, 播放更流畅), 失败回退 mp4v。"""
    fps = fps if (fps and fps > 0) else 30.0
    for cc in ("avc1", "mp4v"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if writer.isOpened():
                return writer
            writer.release()
        except Exception:
            continue
    return None


def _transcode_to_h264(src_path: str) -> str:
    """用系统 ffmpeg 把视频转码为 H264 (avc1), 让 Chromium / Edge / Safari 都能正常播放。
    失败时回退原路径。"""
    if not src_path or not os.path.isfile(src_path):
        return src_path
    ffmpeg_exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg_exe:
        return src_path
    h264_path = os.path.splitext(src_path)[0] + "_h264.mp4"
    try:
        r = subprocess.run(
            [ffmpeg_exe, "-nostdin", "-y", "-loglevel", "error", "-i", src_path,
             "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             h264_path],
            capture_output=True, text=True, timeout=600,
            stdin=subprocess.DEVNULL,
        )
        if r.returncode == 0 and os.path.isfile(h264_path) and os.path.getsize(h264_path) > 0:
            print(f"[INFO] Video transcoded to H264: {h264_path}", flush=True)
            return h264_path
        print(f"[WARN] H264 transcode failed (code={r.returncode}): {r.stderr[:200]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"[WARN] H264 transcode timed out", flush=True)
    except Exception as e:
        print(f"[WARN] H264 transcode error: {e}", flush=True)
    try:
        if os.path.isfile(h264_path):
            os.remove(h264_path)
    except Exception:
        pass
    return src_path


CN_FONT = _load_cn_font(22)


def annotate_frame(frame, locs, cls_list, conf_list, cls_colors, font):
    """在帧上画框 + 中文类别标签(含置信度)。

    用 PIL 渲染中文, 彻底避开 cv2.putText 中文乱码/空白的问题。
    整帧只做一次 BGR->PIL->BGR 转换, 多框也只转一次, 开销可控。
    """
    if not locs:
        return frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for (l, c, cf) in zip(locs, cls_list, conf_list):
        x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
        color = cls_colors[c % len(cls_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
        label = f"{name} {cf:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 标签条: 默认放框上方; 贴顶则放框下方, 避免出画面
        if y1 > th + 6:
            bx1, by1, bx2, by2 = x1, y1 - th - 4, x1 + tw + 6, y1
        else:
            bx1, by1, bx2, by2 = x1, y2, x1 + tw + 6, y2 + th + 4
        draw.rectangle([bx1, by1, bx2, by2], fill=color)
        draw.text((bx1 + 3, by1 + 2), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---- 检测逻辑(复用原项目实现) ----
def do_detect(image_path):
    img = tools.img_cvread(image_path)
    if img is None:
        return {"error": f"无法读取: {image_path}"}
    t1 = time.time()
    results = model(image_path)[0]
    t2 = time.time()
    locs = [list(map(int, e)) for e in results.boxes.xyxy.tolist()]
    cls = [int(i) for i in results.boxes.cls.tolist()]
    confs = [round(float(c) * 100, 2) for c in results.boxes.conf.tolist()]
    stats = {}
    for c in cls:
        n = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
        stats[n] = stats.get(n, 0) + 1
    result_img = results.plot()
    save_name = f"detect_{int(time.time())}.jpg"
    save_path = os.path.join(SAVE_DIR, save_name)
    cv2.imwrite(save_path, result_img)
    dets = []
    for i, (l, c, cf) in enumerate(zip(locs, cls, confs)):
        name = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
        dets.append({
            "id": i + 1, "behavior": name, "confidence": f"{cf}%",
            "bbox": {"xmin": l[0], "ymin": l[1], "xmax": l[2], "ymax": l[3]},
        })
    # 发布标注图为可访问链接
    image_links = _publish_result(save_path)
    return {
        "success": True, "time": f"{t2-t1:.3f}s", "total": len(cls),
        "stats": stats, "detections": dets,
        "result_image": save_path,
        "result_image_url": image_links.get("watch_url", ""),
        "result_image_download_url": image_links.get("download_url", ""),
        "result_image_backend": image_links.get("backend", ""),
    }


def do_analyze(image_path):
    r = do_detect(image_path)
    if "error" in r:
        return r
    stats, total = r["stats"], r["total"]
    pos = stats.get("举手", 0) + stats.get("阅读", 0) + stats.get("写作", 0)
    lines = ["## 课堂行为检测报告", "", f"检测到 {total} 名学生", ""]
    for b, c in sorted(stats.items(), key=lambda x: -x[1]):
        pct = round(c / total * 100, 1) if total else 0
        lines.append(f"- {b}: {c}人 ({pct}%)")
    if total:
        pr = pos / total * 100
        assess = "优秀" if pr > 70 else "良好" if pr > 50 else "一般" if pr > 30 else "需关注"
        lines += ["", f"积极行为: {pr:.1f}%", f"评估: {assess}"]
    r["report"] = "\n".join(lines)
    return r


def save_base64(b64):
    b = base64.b64decode(b64)
    nparr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    tmp = os.path.join(SAVE_DIR, f"tmp_{int(time.time())}.jpg")
    cv2.imwrite(tmp, img)
    return tmp


def do_detect_video(video_path, sample_interval=10, save_video=True, target_classes=None):
    filter_indices = None
    if target_classes:
        name_to_idx = {name: i for i, name in enumerate(Config.CH_names)}
        filter_indices = [name_to_idx[n] for n in target_classes if n in name_to_idx]
        if not filter_indices:
            return {"error": f"未匹配到有效类别，可用类别: {Config.CH_names}"}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"无法打开视频: {video_path}"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(total_frames / fps, 1) if fps > 0 else 0

    out_video_path = None
    out_writer = None
    if save_video:
        save_name = f"video_detect_{int(time.time())}.mp4"
        out_video_path = os.path.join(SAVE_DIR, save_name)
        out_writer = _make_writer(out_video_path, fps, w, h)

    frame_idx = 0
    all_stats = {}
    all_detections = []
    last_locs, last_cls, last_conf = [], [], []
    last_annotated = None
    cls_colors = [(0, 200, 255), (255, 100, 0), (0, 255, 100), (0, 0, 255), (200, 0, 255), (255, 200, 0)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_interval == 0:
            results = model(frame, classes=filter_indices)[0] if filter_indices else model(frame)[0]
            cls_list = [int(i) for i in results.boxes.cls.tolist()]
            conf_list = [round(float(c) * 100, 2) for c in results.boxes.conf.tolist()]
            locs = [list(map(int, e)) for e in results.boxes.xyxy.tolist()]
            for c in cls_list:
                n = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
                all_stats[n] = all_stats.get(n, 0) + 1
            frame_dets = []
            for l, c, cf in zip(locs, cls_list, conf_list):
                name = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
                frame_dets.append({"behavior": name, "confidence": f"{cf}%"})
            all_detections.append({
                "frame": frame_idx, "time": f"{frame_idx/fps:.1f}s",
                "count": len(cls_list), "detections": frame_dets,
            })
            last_locs, last_cls, last_conf = locs, cls_list, conf_list
        # 关键修复: 每帧都写「当前真实帧」+「最近一次检测结果的标签」,
        # 不再是冻结上一帧 -> 画面连续不卡顿, 标签持续可见(每 sample_interval 帧刷新位置)
        annotated = annotate_frame(frame, last_locs, last_cls, last_conf, cls_colors, CN_FONT)
        last_annotated = annotated
        if out_writer:
            out_writer.write(annotated)
    cap.release()
    if out_writer:
        out_writer.release()
    # 如果原视频是 mpeg4 (mp4v) 编码，浏览器可能播不了，转码为 H264 让浏览器能播
    if out_video_path:
        out_video_path = _transcode_to_h264(out_video_path)
    result_image = None
    if last_annotated is not None:
        save_name = f"video_detect_{int(time.time())}.jpg"
        result_image = os.path.join(SAVE_DIR, save_name)
        cv2.imwrite(result_image, last_annotated)

    # 发布标注视频与封面图为可访问链接
    video_links = _publish_result(out_video_path) if out_video_path else {}
    image_links = _publish_result(result_image) if result_image else {}
    return {
        "success": True,
        "video_info": {"path": video_path, "resolution": f"{w}x{h}", "fps": round(fps, 1),
                       "total_frames": total_frames, "duration": f"{duration}s"},
        "sampling": {"interval": sample_interval, "total_frames": total_frames},
        "total_detections": sum(all_stats.values()),
        "stats": all_stats,
        "frame_details": all_detections,
        "result_video": out_video_path,
        "result_video_url": video_links.get("watch_url", ""),
        "result_video_download_url": video_links.get("download_url", ""),
        "result_video_backend": video_links.get("backend", ""),
        "result_image": result_image,
        "result_image_url": image_links.get("watch_url", ""),
        "result_image_download_url": image_links.get("download_url", ""),
        "result_image_backend": image_links.get("backend", ""),
    }


def _resolve_to_http(url: str) -> str:
    """把可能的相对路径(/uploaded/xxx.mp4)补全为绝对 URL(http://127.0.0.1:9800/...)。
    让 detect_image_url / detect_video_url 既能跑公网直链也能跑刚刚 /upload 拿到的本地 URL。"""
    if not url:
        return url
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("/uploaded/"):
        # 本机回路
        return f"http://127.0.0.1:{os.environ.get('MCP_PORT', '9800')}{u}"
    if u.startswith("/"):
        return u  # 其它相对路径留给上层处理
    return u


def _download_to_tmp(url: str, suffix: str = "", timeout: int = 60) -> str:
    """把远端 http(s) URL 下载到本地 SAVE_DIR 临时文件，返回本地绝对路径。
    用途：让连接方 agent 只要给出公网直链(网盘/ngrok 临时链接/对方电脑起的 http server 等)，
    服务端就能直接拉下来用 OpenCV / YOLO 处理，无需 agent 自己中转文件。
    注意：若是同域 /uploaded/xxx，需要带 Bearer token。
    """
    import requests
    # 本地已上传文件: URL 路径含 /uploaded/<name>(不论域名, 含 ngrok 公网域名),
    # 直接读本地 _UPLOAD_DIR, 避免服务端出站回源(沙箱/隧道环境常拦截出站)。
    if "/uploaded/" in url:
        seg = url.split("/uploaded/", 1)[1].split("?")[0].split("#")[0]
        name = os.path.basename(seg.replace("\\", "/"))
        if name and "/" not in name and ".." not in name:
            local = os.path.join(_UPLOAD_DIR, name)
            if os.path.isfile(local):
                return local
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return ""
    # 文件名：时间戳 + 下划线 + URL 后缀推断
    guessed = ""
    if "?" in url:
        head = url.split("?", 1)[0]
    else:
        head = url
    if "." in head.rsplit("/", 1)[-1]:
        guessed = "." + head.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower()
        if len(guessed) > 6 or guessed not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}:
            guessed = ""
    ext = suffix or guessed or ".bin"
    tmp_path = os.path.join(SAVE_DIR, f"_tmp_{int(time.time()*1000)}_{abs(hash(url)) & 0xFFFF:04x}{ext}")
    headers = {"User-Agent": "Mozilla/5.0 (kczh-MCP)"}
    # 同域 /uploaded/xxx 必须带 Bearer，否则 401
    if AUTH_TOKEN and (("/uploaded/" in url) or url.startswith(f"http://127.0.0.1:{os.environ.get('MCP_PORT','9800')}/uploaded/") or url.startswith(f"http://localhost:{os.environ.get('MCP_PORT','9800')}/uploaded/")):
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    try:
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=True, headers=headers) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"下载失败({type(e).__name__}): {e}")


def call_tool(name, args):
    """同步阻塞实现(给 asyncio.to_thread 丢线程池用)，禁止在 async 上下文直接调用"""
    if name == "get_model_info":
        return {"model": MODEL_PATH, "classes": Config.CH_names, "status": "ready"}
    elif name == "detect_image":
        return do_detect(args.get("image_path", ""))
    elif name == "detect_image_url":
        try:
            tmp = _download_to_tmp(_resolve_to_http(args.get("image_url", "")), suffix=".jpg")
        except Exception as e:
            return {"error": f"无法下载图片: {e}"}
        if not tmp or not os.path.exists(tmp):
            return {"error": "图片 URL 为空或下载失败"}
        r = do_detect(tmp)
        # 若 tmp 是本地已上传文件(_UPLOAD_DIR 内), 不要删除, 否则会误删用户原文件
        if tmp and _UPLOAD_DIR not in tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return r
    elif name == "detect_base64":
        path = save_base64(args.get("image_base64", ""))
        return do_detect(path) if path else {"error": "图片解码失败"}
    elif name == "capture_and_detect":
        cid = args.get("camera_id", 0)
        cap = cv2.VideoCapture(cid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "摄像头打开失败"}
        tmp = os.path.join(SAVE_DIR, f"cam_{int(time.time())}.jpg")
        cv2.imwrite(tmp, frame)
        return do_detect(tmp)
    elif name == "analyze_classroom":
        path = save_base64(args.get("image_base64", ""))
        return do_analyze(path) if path else {"error": "图片解码失败"}
    elif name == "detect_video":
        return do_detect_video(
            args.get("video_path", ""),
            args.get("sample_interval", 30),
            args.get("save_video", True),
            args.get("target_classes", None),
        )
    elif name == "detect_video_url":
        video_url = _resolve_to_http(args.get("video_url", "").strip())
        if not video_url:
            return {"error": "video_url 不能为空"}
        try:
            tmp = _download_to_tmp(video_url, suffix=".mp4", timeout=120)
        except Exception as e:
            return {"error": f"无法下载视频: {e}"}
        if not tmp or not os.path.exists(tmp):
            return {"error": "视频 URL 为空或下载失败"}
        r = do_detect_video(
            tmp,
            args.get("sample_interval", 30),
            args.get("save_video", True),
            args.get("target_classes", None),
        )
        if isinstance(r, dict):
            r["source_url"] = video_url
        try:
            os.remove(tmp)
        except Exception:
            pass
        return r
    elif name == "open_camera_monitor":
        # 元工具: 返回连接方(调用方)要打开的摄像头监控网页链接。
        # 摄像头用的是【打开链接那台机器】的, 而非服务端。
        base = get_public_base()
        url = f"{base}/cam?token={AUTH_TOKEN}" if AUTH_TOKEN else f"{base}/cam"
        return {
            "url": url,
            "note": "在调用方电脑的浏览器中打开上面的链接, 允许摄像头权限, 即可看到实时标注画面。",
            "requires_camera_on_caller_machine": True,
        }
    elif name == "analyze_video_file":
        # 专为"调用方持有本地视频文件"设计的两步走流程的第 2 步。
        # 第 1 步：调用方把视频原始字节 POST 到 /upload?filename=xxx.mp4 -> 拿到 file_id
        # 第 2 步：本工具接 file_id，服务端在 _UPLOAD_DIR 里找���该文件并跑 YOLO。
        # 整个流程不涉及 base64 拆帧、不受单行长度限制、也无需公网 URL。
        file_id = (args.get("file_id", "") or "").strip()
        if not file_id:
            return {"error": "缺少 file_id。请先调用 POST /upload?filename=xxx.mp4 上传视频原始字节, "
                              "再把返回 JSON 里的 file_id 传进来。"}
        # 防路径穿越
        if "/" in file_id or "\\" in file_id or ".." in file_id:
            return {"error": "file_id 不合法"}
        fp = os.path.join(_UPLOAD_DIR, file_id)
        if not os.path.isfile(fp):
            return {"error": f"找不到 file_id={file_id} 对应的已上传文件。请确认 /upload 已成功并使用返回的 file_id。"}
        job_id = submit_video_analysis(
            fp,
            args.get("sample_interval", 30),
            args.get("save_video", True),
            args.get("target_classes", None),
        )
        # 短视频: 在 SYNC_WAIT_SEC 内完成则直接返回完整结果(与旧行为一致, 无需轮询)
        deadline = time.time() + SYNC_WAIT_SEC
        while time.time() < deadline:
            j = _job_get(job_id)
            if j and j.get("status") == "done":
                r = j.get("result") or {}
                if isinstance(r, dict):
                    r["job_id"] = job_id
                return r
            if j and j.get("status") == "error":
                return {"error": j.get("error", "分析失败"), "job_id": job_id}
            time.sleep(0.5)
        # 超时仍未完成(长视频/大文件) -> 回退为异步 job_id, 由调用方轮询 get_analysis_result
        return {
            "job_id": job_id,
            "status": "running",
            "note": "视频较长, 已转为异步任务。请用 get_analysis_result(job_id=...) 轮询, 直到 status=done 取回完整结果。",
            "file_id": file_id,
        }
    elif name == "get_analysis_result":
        job_id = (args.get("job_id", "") or "").strip()
        if not job_id:
            return {"error": "缺少 job_id"}
        j = _job_get(job_id)
        if not j:
            return {"error": f"找不到 job_id={job_id}, 可能任务已过期或服务已重启"}
        return j
    else:
        return {"error": f"未知工具: {name}"}


async def _run_tool(name, args):
    """异步包装: 限并发 + 丢线程池，避免阻塞事件循环/打爆单机。
    返回 JSON 字符串以匹配工具签名 `-> str`（mcp SDK 严格校验 Output schema）。"""
    async with _INFER_SEM:
        result = await asyncio.to_thread(call_tool, name, args)
        return json.dumps(result, ensure_ascii=False, indent=2)


def _json(name, args):
    return json.dumps(call_tool(name, args), ensure_ascii=False, indent=2)


# ---- 标准 MCP 服务器 (mcp 2.x) ----
from mcp.server.mcpserver import MCPServer  # noqa: E402
mcp = MCPServer("classroom-behavior-detection")


@mcp.tool()
async def get_model_info() -> str:
    """获取课堂行为检测模型信息(类别、状态)"""
    return await _run_tool("get_model_info", {})


@mcp.tool()
async def detect_image(image_path: str) -> str:
    """⚠️【远程 MCP 读不到调用方磁盘上的文件】image_path 必须是【服务端】能直接读到的路径(http(s) 直链、或 /upload 上传后的服务端路径)。

    如果你手里只有你自己磁盘上的图片, 【不要】直接传本地路径 —— 服务端读不到, 一定报"找不到文件"。
    请改用: (1) 先 POST /upload 把图片字节推上去 → 拿到 file_id; (2) 调 detect_image_url(image_url="<upload 返回的 public_url>") 或调 analyze_video_file(若是视频)。

    检测本地图片的课堂行为(举手/阅读/写作/使用手机/低头/趴桌)，返回统计与检测框"""
    return await _run_tool("detect_image", {"image_path": image_path})


@mcp.tool()
async def detect_base64(image_base64: str) -> str:
    """⚠️【仅适合调用方内存里已有 base64 数据】比如调用方截图工具直接拿到 base64, 或者只是 1-2 张小图。

    【视频请不要用这个工具】(客户端有 2000 字符/行限制, 视频会被卡死)。视频走两步:
       (1) POST /upload 把视频字节推上去 → 拿 file_id
       (2) 调 analyze_video_file(file_id=...) 一次性拿到结果
    单张图片但磁盘路径不可用时, 也建议走 POST /upload + detect_image_url, 比 base64 稳。

    检测 base64 编码图片的课堂行为，适用于 agent 直接传图"""
    return await _run_tool("detect_base64", {"image_base64": image_base64})


@mcp.tool()
async def capture_and_detect(camera_id: int = 0) -> str:
    """⚠️【服务端摄像头】非调用方摄像头! 这会打开部署 MCP 的那台机器的摄像头抓拍。
    如果要打开【调用方自己】电脑的摄像头, 请改用 open_camera_monitor 工具(返回网页链接)。

    从摄像头(编号,默认0)抓拍并检测课堂行为"""
    return await _run_tool("capture_and_detect", {"camera_id": camera_id})


@mcp.tool()
async def analyze_classroom(image_base64: str) -> str:
    """⚠️【仅适合调用方内存里已有 base64 数据】base64 通道不适合视频/大图(客户端 2000 字符/行限制)。
    如果磁盘上有图片/视频, 走 POST /upload + detect_image_url / analyze_video_file 更稳。

    课堂完整分析:检测行为 + 生成评估报告 + 教学建议"""
    return await _run_tool("analyze_classroom", {"image_base64": image_base64})


@mcp.tool()
async def detect_video(video_path: str, sample_interval: int = 10, save_video: bool = True, target_classes: str = "") -> str:
    """⚠️ 注意: video_path 必须是【服务端本机】能直接读到的路径(http(s) 直链、或 /upload 上传后的服务端路径)。

    如果你是从另一台电脑(或另一个 agent)过来的, 手里只有【你自己磁盘上】的视频文件,
    请【不要】直接传本地路径 —— 本服务在远端, 读不到你本机的文件, 一定会报"找不到文件"。
    这种情况请改用 analyze_video_file 工具(配合 /upload 两步走, 见其文档)。

    检测视频中的课堂行为, 自动采样帧并返回统计结果与标注视频。
    video_path: 服务端可访问的视频路径(http(s) 直链、或 /upload 上传后的服务端路径)
    sample_interval: 每隔多少帧采样一次(默认30)
    save_video: 是否输出标注后的完整视频(默认True)
    target_classes: 要检测的类别名，逗号分隔，如 '阅读,举手'，留空检测全部
    """
    tc = [x.strip() for x in target_classes.split(",") if x.strip()] if target_classes else None
    return await _run_tool("detect_video", {
        "video_path": video_path, "sample_interval": sample_interval,
        "save_video": save_video, "target_classes": tc,
    })


@mcp.tool()
async def detect_image_url(image_url: str) -> str:
    """从公网 URL 下载图片并检测课堂行为(举手/阅读/写作/使用手机/低头/趴桌)。

    用途：连接方 agent 手里只有图片的 http(s) 公网直链时(网盘分享、对方电脑起的临时 http server、
    已上传的对象存储、COS/OSS 直链等)，不再需要 agent 自己下载再塞 base64——服务端会自己下载处理。
    """
    return await _run_tool("detect_image_url", {"image_url": image_url})


@mcp.tool()
async def detect_video_url(video_url: str, sample_interval: int = 30, save_video: bool = True, target_classes: str = "") -> str:
    """从公网 URL 下载视频并检测课堂行为，逐帧采样 + 行为统计 + 生成标注视频。

    用途：连接方 agent 在另一台机器上只有视频的公网直链(网盘分享、ngrok 临时链接、对方电脑的 http server 等)，
    服务端会自己下载(默认 120s 超时)然后用 YOLO 处理，agent 不再需要把视频数据塞进 base64/路径参数。

    video_url: 视频 http(s) 公网直链（支持 mp4/mov/avi/mkv/webm）
    sample_interval: 每隔多少帧采样一次(默认 30)。值越大越快但越粗糙；1080p 视频建议 15-30。
    save_video: 是否输出完整标注视频 mp4(默认 True)，输出路径在 result_video 字段。
    target_classes: 要检测的类别名，逗号分隔，如 '阅读,举手'，���空检测全部。

    ⚠️【调用方只有自己磁盘上的视频? 不要传 file:// 或本地路径, 服务端读不到】
       这种情况请改用 analyze_video_file 工具: 先 POST /upload 拿 file_id, 再传 file_id 进来。
    """
    tc = [x.strip() for x in target_classes.split(",") if x.strip()] if target_classes else None
    return await _run_tool("detect_video_url", {
        "video_url": video_url, "sample_interval": sample_interval,
        "save_video": save_video, "target_classes": tc,
    })


@mcp.tool()
async def open_camera_monitor() -> str:
    """打开【调用方(连接方)本机】的摄像头, 进行实时课堂行为监控(举手/阅读/写作/使用手机/低头/趴桌)。

    返回一个网页链接。在调用方电脑的浏览器中打开该链接并允许摄像头权限后, 页面会自动调用
    那台机器自己的摄像头, 逐帧把画面发给本服务端的 YOLO 模型推理, 并在页面上实时显示
    带检测框的画面与行为统计。摄像头始终用的是【打开链接那台机器】的摄像头, 而非服务端。

    典型用法: 对方在 WorkBuddy 里说"打开摄像头看看大家在干嘛", agent 调用本工具拿到链接,
    对方在其电脑浏览器(或 WorkBuddy 内置预览)打开即看到实时标注监控。
    """
    base = get_public_base()
    url = f"{base}/cam?token={AUTH_TOKEN}" if AUTH_TOKEN else f"{base}/cam"
    return json.dumps({
        "url": url,
        "note": "在调用方电脑的浏览器中打开上面的链接, 允许摄像头权限, 即可看到实时标注画面。",
        "requires_camera_on_caller_machine": True,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def analyze_video_file(file_id: str, sample_interval: int = 30, save_video: bool = True, target_classes: str = "") -> str:
    """【最省事】直接分析调用方【刚通过 /upload 上传】的视频文件 —— 调用方手里只有自己磁盘上的视频时,
    【必走】这条两步流程, 不要把本地路径喂给 detect_video(它会找不到文件, 因为服务在远端)。

    【标准工作流, 两步搞定, 不要再 base64 拆帧】
      第 1 步 —— 把视频原始字节 POST 到本服务的 /upload 端点(走公网):
          POST  https://<本服务公网地址>/upload?filename=xxx.mp4
          Header: Authorization: Bearer <MCP_AUTH_TOKEN>
          Header: Content-Type: video/mp4  (或其它视频 MIME)
          Body:   <视频文件的原始字节, 不要 base64>
          → 服务端返回 JSON, 找到 file_id 字段(它就是这一步的"句柄")
      第 2 步 —— 调用本工具, 传 file_id:
          analyze_video_file(file_id="<第 1 步拿到的 file_id>")
          → 服务端直接对该文件跑 YOLO, 返回行为统计 + 标注视频路径。

    【为什么必须先 /upload】
      本服务部署在调用方之外的机器上(经 ngrok/Cloudflare 隧道暴露), 服务端读不到调用方磁盘上的任何文件。
      任何"本地路径"参数(包括 detect_video(video_path=...))在服务端都会失败, 这是预期行为, 不是 bug。
      /upload 就是为这个场景设计的: 让调用方把字节推到服务端, 后续所有工作都在服务端完成, 不再来回传输。

    【和 detect_video_url 的区别】
      detect_video_url: 调用方拿到的已经是公网直链(网盘/COS/OSS 等)。直接传 url 即可。
      analyze_video_file: 调用方只有自己本机的视频文件, 没有公网 URL。先 /upload 再传 file_id。

    参数:
      file_id:        /upload 返回 JSON 中的 file_id 字段(必填)
      sample_interval: 每隔多少帧采样一次(默认 30; 1080p 课堂录像建议 15-30)
      save_video:     是否输出完整标注视频 mp4(默认 True, 路径在 result_video 字段)
      target_classes: 要检测的类别名, 逗号分隔, 如 '阅读,举手', 留空��测全部 6 类
    """
    tc = [x.strip() for x in target_classes.split(",") if x.strip()] if target_classes else None
    return await _run_tool("analyze_video_file", {
        "file_id": file_id, "sample_interval": sample_interval,
        "save_video": save_video, "target_classes": tc,
    })


@mcp.tool()
async def get_analysis_result(job_id: str) -> str:
    """轮询视频分析异步任务的结果(配合 analyze_video_file 使用)。

    analyze_video_file 对长视频会【异步提交】并先返回 job_id(避免客户端超时);
    短视频仍同步直接返回完整结果(保持旧体验)。若拿到的是 job_id, 用本工具轮询:
    返回 {"status": "queued"|"running"|"done"|"error", "elapsed": 已耗时(秒), "result": <完整结果或 None>}。
    当 status="done" 时, result 即与旧版 analyze_video_file 完全一致的分析结果(行为统计+标注视频路径)。
    """
    return await _run_tool("get_analysis_result", {"job_id": job_id})


def get_public_base():
    """推断本服务的公网基址(给 open_camera_monitor 拼 /cam 链接用)。
    优先级: 环境变量 MCP_NGROK_URL > ngrok 本地 API(轮询缓存) > localhost。
    注意: 连接方必须在另一台机器用公网地址打开, localhost 仅本机调试可用。"""
    global _NGROK_CACHE, _NGROK_CACHE_T
    env = os.environ.get("MCP_NGROK_URL", "").strip().rstrip("/")
    if env:
        return env
    # ngrok 隧道地址(同步脚本写 ngrok_url.txt)
    try:
        ng_file = os.path.join(PROJECT_DIR, "ngrok_url.txt")
        with open(ng_file, encoding="utf-8") as _f:
            ng = _f.read().strip().rstrip("/")
        if ng:
            return ng
    except Exception:
        pass
    # cloudflared 隧道地址(学校/企业网络常拦截 ngrok, cloudflared 多数能直连)
    try:
        cf_file = os.path.join(PROJECT_DIR, "cloudflared_url.txt")
        with open(cf_file, encoding="utf-8") as _f:
            cf = _f.read().strip().rstrip("/")
        if cf:
            return cf
    except Exception:
        pass
    now = time.time()
    if _NGROK_CACHE and now - _NGROK_CACHE_T < 120:
        return _NGROK_CACHE
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:4040/api/tunnels", headers={"User-Agent": "mcp"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.load(resp)
        for t in data.get("tunnels", []):
            if t.get("public_url", "").startswith("https://"):
                _NGROK_CACHE = t["public_url"].rstrip("/")
                _NGROK_CACHE_T = now
                return _NGROK_CACHE
    except Exception:
        pass
    return f"http://localhost:{PORT}"


_NGROK_CACHE = None
_NGROK_CACHE_T = 0


# ---- ASGI 应用: 同时挂 SSE + Streamable HTTP + 手动 REST，统一 Bearer 鉴权 ----
from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse, HTMLResponse, FileResponse, Response, StreamingResponse  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402


class BearerAuthMiddleware:
    """纯 ASGI 3.0 鉴权中间件。
    注意: 不能继承 BaseHTTPMiddleware, 那个对 SSE 流式响应有 bug
    (starlette/middleware/base.py:178 assert body type, SSE 会被 assert 杀掉)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # 放行 /healthz 与 /.well-known/*（OAuth 资源发现元数据必须公开，否则客户端 OAuth 发现流程卡死 -> "授权中"）
        # /cam 和 /video_watch 是网页监控页面(静态HTML), 公开加载; 真正的 MJPEG 流 /api/video_stream 仍由 Bearer 保护(token 走 query)
        # /results 是检测结果公开只读端点(标注图/标注视频), 方便在线观看和下载。
        if (path == "/healthz" or path.startswith("/.well-known/")
                or path == "/cam" or path == "/video_watch"
                or path.startswith("/results/")):
            await self.app(scope, receive, send)
            return

        if AUTH_TOKEN:
            # 解析 Authorization header
            auth_value = ""
            for k, v in scope.get("headers", []):
                if k.lower() == b"authorization":
                    auth_value = v.decode("latin-1", errors="replace")
                    break
            if auth_value != f"Bearer {AUTH_TOKEN}":
                # 直接发 401，并带 WWW-Authenticate: Bearer 告诉客户端用 Bearer 鉴权（而非 OAuth 跳转）
                body = '{"error":"未授权: 请在 Header 携带 Authorization: Bearer <MCP_AUTH_TOKEN>"}'.encode('utf-8')
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer realm="mcp", error="invalid_token"'),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return

        await self.app(scope, receive, send)


# Streamable HTTP 内部路由设为 '/'，再挂到 /mcp；SSE 保持 /sse、/messages/ 挂根
sh_app = mcp.streamable_http_app(streamable_http_path="/")
sse_app = mcp.sse_app(sse_path="/sse", message_path="/messages/")


async def health(request: Request):
    return PlainTextResponse(
        json.dumps({
            "status": "ok",
            "model": os.path.basename(MODEL_PATH),
            "classes": Config.CH_names,
            "max_concurrent": MAX_CONCURRENT,
            "auth": "bearer" if AUTH_TOKEN else "none",
        }, ensure_ascii=False)
    )


async def api_list_tools(request: Request):
    tool_names = [
        "get_model_info", "detect_image", "detect_image_url", "detect_base64",
        "capture_and_detect", "analyze_classroom", "detect_video",
        "detect_video_url", "open_camera_monitor", "analyze_video_file",
        "get_analysis_result",
    ]
    return JSONResponse(content={
        "tools": [{"name": t, "description": f"课堂行为检测工具: {t}"} for t in tool_names]
    })


async def api_call_tool(request: Request):
    body = await request.json()
    name = body.get("name", "")
    args = body.get("arguments", {})
    try:
        r = await _run_tool(name, args)
        return JSONResponse(content={"result": r}, media_type="application/json; charset=utf-8")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500,
                             media_type="application/json; charset=utf-8")


class RewriteHostMiddleware:
    """经 ngrok 等反向代理时 Host 为公网域名, 会被 mcp SDK 的 DNS-rebinding 保护拒绝(421)。
    在进入 mcp 校验前把 Host 改写为 localhost(在 allowed_hosts 内), 使其通过。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            new_headers = []
            for k, v in scope.get("headers", []):
                if k.lower() == b"host":
                    new_headers.append((b"host", b"localhost:9800"))
                else:
                    new_headers.append((k, v))
            scope["headers"] = new_headers
        await self.app(scope, receive, send)


async def oauth_protected_resource(request: Request):
    """MCP 2.x OAuth 资源服务器元数据 (RFC 9728)。
    公开端点, 告诉客户端: 这个资源用 Bearer Token 保护, 没有独立授权服务器(直接静态 Bearer)。
    客户端发现到此即止, 不再去跳转 OAuth 授权页 -> 解决 "授权中" 卡死。"""
    base = f"{request.url.scheme}://{request.url.netloc}"
    return JSONResponse({
        "resource": base,
        "authorization_servers": [],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["mcp"],
        "resource_signals_supported": False,
    })


async def api_detect(request: Request):
    """轻量推理端点(给连接端摄像头监控客户端用, 绕过 MCP SSE 会话复杂度)。
    接收 base64 图片, 返回标注图 base64 + 行为统计。Bearer 由外层中间件校验。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    b64 = body.get("image_base64", "")
    if not b64:
        return JSONResponse({"error": "missing image_base64"}, status_code=400)
    path = save_base64(b64)
    if not path:
        return JSONResponse({"error": "图片解码失败"}, status_code=400)
    r = do_detect(path)
    if "error" in r:
        return JSONResponse(r, status_code=400)
    # 标注图读回 base64 返回(连接端在远端, 读不到本地路径)
    try:
        with open(r["result_image"], "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        annotated_b64 = ""
    return JSONResponse({
        "success": True,
        "time": r["time"],
        "total": r["total"],
        "stats": r["stats"],
        "detections": r["detections"],
        "annotated_image_base64": annotated_b64,
    }, media_type="application/json; charset=utf-8")


# ---- 文件上传端点(给连接方 agent 一个把视频"投到"服务端、拿到 URL 的通道)----
_UPLOAD_DIR = os.path.join(SAVE_DIR, "uploads")
_UPLOAD_TTL_SEC = 24 * 3600  # 上传的文件 24h 后清理
_UPLOAD_MAX_BYTES = 512 * 1024 * 1024  # 单文件上限 512MB,够主流课堂录像用
# 白名单放宽: 录屏软件/相机/网盘常见容器都收下, 否则用户传 .flv/.ts/.m4v/.rmvb
# 会在 /upload 与 /upload_init 直接吃 400, 且错误信息不直观, 排查成本很高。
_ALLOWED_SUFFIX = {
    # 图片
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
    # 主流视频容器
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    # 录屏软件/摄像机/网盘转存常见格式
    ".flv", ".wmv", ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".rmvb", ".asf", ".vob", ".f4v",
}

# ---- 分片上传(针对高延迟公网链路) ----
# 为什么需要: 单条 TCP 连接在高 RTT 链路上受拥塞窗口限制, 吞吐上不去(实测 ngrok 隧道
# 握手就要 10s)。切成多片后由客户端并发上传, 多路连接的窗口叠加才能把带宽吃满。
_CHUNK_DIR = os.path.join(_UPLOAD_DIR, "_chunks")
_CHUNK_TTL_SEC = 6 * 3600  # 未合并的半成品分片 6h 后清理
_CHUNK_MAX_BYTES = 512 * 1024 * 1024


def _cleanup_uploads():
    """删除过期上传文件,带 openclaw fs-safe 兜底(Bash 沙箱跑会 SAFE_DELETE_FAIL_CLOSED)。"""
    try:
        now = time.time()
        for fn in os.listdir(_UPLOAD_DIR):
            fp = os.path.join(_UPLOAD_DIR, fn)
            if not os.path.isfile(fp):
                continue
            if now - os.path.getmtime(fp) > _UPLOAD_TTL_SEC:
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass


async def api_upload(request: Request):
    """POST /upload?filename=xxx.mp4  body=原始文件字节。
    返回 {"url": "/uploaded/xxx.mp4", "size": ...}，
    agent 拿到 url 直接喂给 detect_image_url / detect_video_url / /api/detect 等任何接收 URL 的端点。
    """
    _cleanup_uploads()
    filename = request.query_params.get("filename", "").strip()
    if not filename:
        filename = f"upload_{int(time.time())}.bin"
    # 防路径穿越, 只保留文件名部分
    filename = os.path.basename(filename).replace("/", "_").replace("\\", "_")
    # 补扩展名
    name, ext = os.path.splitext(filename)
    if not ext:
        filename = filename + ".bin"
    ext_lc = os.path.splitext(filename)[1].lower()
    if ext_lc not in _ALLOWED_SUFFIX:
        return JSONResponse({"error": f"不支持的扩展名 {ext_lc}; 允许 {sorted(_ALLOWED_SUFFIX)}"},
                            status_code=400, media_type="application/json; charset=utf-8")
    try:
        os.makedirs(_UPLOAD_DIR, exist_ok=True)
    except Exception:
        pass
    # 唯一文件名: 时间戳 + uuid + 原文件名(去特殊字符)
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(_UPLOAD_DIR, safe_name)
    # 流式写盘: 边收边写, 内存占用恒定(旧实现用 await request.body() 把整个文件
    # 读进内存, 512MB 上限意味着 512MB 内存峰值, 大视频连续上传容易把机器顶爆)。
    total = 0
    try:
        with open(save_path, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > _UPLOAD_MAX_BYTES:
                    f.close()
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass
                    return JSONResponse(
                        {"error": f"文件过大 (> {_UPLOAD_MAX_BYTES // (1024*1024)}MB)"},
                        status_code=413, media_type="application/json; charset=utf-8")
                f.write(chunk)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return JSONResponse({"error": f"写入失败: {e}"}, status_code=500,
                            media_type="application/json; charset=utf-8")
    if total == 0:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return JSONResponse({"error": "请求体为空"}, status_code=400,
                            media_type="application/json; charset=utf-8")
    base = get_public_base()
    qtoken = f"&token={AUTH_TOKEN}" if AUTH_TOKEN else ""
    watch_url = f"{base}/video_watch?file_id={safe_name}{qtoken}"
    stream_url = f"{base}/api/video_stream?file_id={safe_name}{qtoken}"
    return JSONResponse({
        "success": True,
        "url": f"/uploaded/{safe_name}",
        "public_url": f"{base}/uploaded/{safe_name}",
        # file_id 是给 MCP 工具 analyze_video_file 用的"短句柄"，等价于 safe_name。
        # agent 拿到后直接传 analyze_video_file(file_id=...) 即可，服务端会在 _UPLOAD_DIR 里找到这个文件。
        "file_id": safe_name,
        # watch_url 是给【人】看的实时分析页(任何浏览器打开即看到 YOLO 逐帧标注画面);
        # stream_url 是裸 MJPEG 流, 可直接 <img src> 嵌入其他页面。
        "watch_url": watch_url,
        "stream_url": stream_url,
        "absolute_url_note": ("如果从公网(连接方电脑)调用, 用 public_url；"
                              "如果在本机直接测试, 用 /uploaded/ 前缀即可。"),
        "filename": filename,
        "size": total,
        "expires_in": _UPLOAD_TTL_SEC,
        "workflow_hint": ("把上面 file_id 传给 mcp 工具 analyze_video_file(file_id=...) "
                          "即可直接拿到分析结果，无需再 base64 拆帧。"
                          "若想【实时观看】带框画面, 把 watch_url 在浏览器打开即可。"),
    }, media_type="application/json; charset=utf-8")


def _cleanup_chunks():
    """清理超过 TTL 仍未合并的分片目录(客户端传到一半放弃时会残留)。"""
    try:
        now = time.time()
        if not os.path.isdir(_CHUNK_DIR):
            return
        for d in os.listdir(_CHUNK_DIR):
            dp = os.path.join(_CHUNK_DIR, d)
            if not os.path.isdir(dp):
                continue
            if now - os.path.getmtime(dp) > _CHUNK_TTL_SEC:
                shutil.rmtree(dp, ignore_errors=True)
    except Exception:
        pass


def _chunk_dir_of(upload_id: str):
    """校验 upload_id 合法性(防路径穿越)并返回其分片目录; 不合法返回 None。"""
    if not upload_id or "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        return None
    return os.path.join(_CHUNK_DIR, upload_id)


async def api_upload_init(request: Request):
    """POST /upload_init?filename=xxx.mp4&total_size=N&total_chunks=M

    分片上传第 1 步: 登记上传任务, 返回 upload_id。
    total_chunks 由客户端按自己想要的分片大小算好; 各分片可乱序、并发上传。
    """
    _cleanup_uploads()
    _cleanup_chunks()
    filename = (request.query_params.get("filename", "") or "").strip()
    if not filename:
        filename = f"upload_{int(time.time())}.bin"
    filename = os.path.basename(filename).replace("/", "_").replace("\\", "_")
    name, ext = os.path.splitext(filename)
    if not ext:
        filename = filename + ".bin"
    ext_lc = os.path.splitext(filename)[1].lower()
    if ext_lc not in _ALLOWED_SUFFIX:
        return JSONResponse({"error": f"不支持的扩展名 {ext_lc}; 允许 {sorted(_ALLOWED_SUFFIX)}"},
                            status_code=400, media_type="application/json; charset=utf-8")
    try:
        total_size = int(request.query_params.get("total_size", "0"))
        total_chunks = int(request.query_params.get("total_chunks", "0"))
    except ValueError:
        return JSONResponse({"error": "total_size / total_chunks 必须是整数"},
                            status_code=400, media_type="application/json; charset=utf-8")
    if total_size <= 0 or total_chunks <= 0:
        return JSONResponse({"error": "total_size / total_chunks 必须大于 0"},
                            status_code=400, media_type="application/json; charset=utf-8")
    if total_size > _CHUNK_MAX_BYTES:
        return JSONResponse({"error": f"文件过大 (> {_CHUNK_MAX_BYTES // (1024*1024)}MB)"},
                            status_code=413, media_type="application/json; charset=utf-8")
    upload_id = f"{int(time.time())}_{uuid.uuid4().hex[:12]}"
    d = os.path.join(_CHUNK_DIR, upload_id)
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"filename": filename, "total_size": total_size,
                       "total_chunks": total_chunks, "created": time.time()}, f)
    except Exception as e:
        return JSONResponse({"error": f"初始化失败: {e}"}, status_code=500,
                            media_type="application/json; charset=utf-8")
    return JSONResponse({
        "success": True,
        "upload_id": upload_id,
        "total_chunks": total_chunks,
        "total_size": total_size,
        "next": "逐个(可并发) POST /upload_chunk?upload_id=<upload_id>&index=<0..M-1>, "
                "body 为该分片原始字节; 全部传完再 POST /upload_complete?upload_id=<upload_id>",
    }, media_type="application/json; charset=utf-8")


async def api_upload_chunk(request: Request):
    """POST /upload_chunk?upload_id=xxx&index=N  body=该分片的原始字节。

    分片按 index 独立落盘为 <index>.part(先写 .tmp 再原子 rename),
    因此客户端可多线程并发、乱序上传, 彼此不会互相踩踏。
    """
    upload_id = (request.query_params.get("upload_id", "") or "").strip()
    d = _chunk_dir_of(upload_id)
    if d is None:
        return JSONResponse({"error": "upload_id 不合法"}, status_code=400,
                            media_type="application/json; charset=utf-8")
    if not os.path.isfile(os.path.join(d, "_meta.json")):
        return JSONResponse({"error": "upload_id 不存在或已过期, 请重新调用 /upload_init"},
                            status_code=404, media_type="application/json; charset=utf-8")
    try:
        index = int(request.query_params.get("index", "-1"))
    except ValueError:
        return JSONResponse({"error": "index 必须是整数"}, status_code=400,
                            media_type="application/json; charset=utf-8")
    if index < 0:
        return JSONResponse({"error": "index 必须 >= 0"}, status_code=400,
                            media_type="application/json; charset=utf-8")
    part_fp = os.path.join(d, f"{index}.part")
    tmp_fp = part_fp + ".tmp"
    written = 0
    try:
        with open(tmp_fp, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                f.write(chunk)
        os.replace(tmp_fp, part_fp)
    except Exception as e:
        try:
            os.remove(tmp_fp)
        except Exception:
            pass
        return JSONResponse({"error": f"分片写入失败: {e}"}, status_code=500,
                            media_type="application/json; charset=utf-8")
    try:
        os.utime(d, None)  # 刷新活跃时间, 避免长传任务被 TTL 中途清掉
    except Exception:
        pass
    return JSONResponse({"success": True, "index": index, "size": written},
                        media_type="application/json; charset=utf-8")


async def api_upload_complete(request: Request):
    """POST /upload_complete?upload_id=xxx

    校验分片齐全后按 index 顺序合并成正式文件, 返回结构与 /upload 完全一致
    (含 file_id, 可直接喂给 analyze_video_file)。
    """
    upload_id = (request.query_params.get("upload_id", "") or "").strip()
    d = _chunk_dir_of(upload_id)
    if d is None:
        return JSONResponse({"error": "upload_id 不合法"}, status_code=400,
                            media_type="application/json; charset=utf-8")
    meta_fp = os.path.join(d, "_meta.json")
    if not os.path.isfile(meta_fp):
        return JSONResponse({"error": "upload_id 不存在或已过期, 请重新调用 /upload_init"},
                            status_code=404, media_type="application/json; charset=utf-8")
    try:
        with open(meta_fp, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return JSONResponse({"error": f"元数据损坏: {e}"}, status_code=500,
                            media_type="application/json; charset=utf-8")
    total_chunks = int(meta.get("total_chunks", 0))
    missing = [i for i in range(total_chunks)
               if not os.path.isfile(os.path.join(d, f"{i}.part"))]
    if missing:
        return JSONResponse({
            "error": f"还缺 {len(missing)} 个分片, 请先补齐后重试",
            "missing": missing[:50],
            "missing_truncated": len(missing) > 50,
        }, status_code=400, media_type="application/json; charset=utf-8")
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{meta.get('filename', 'upload.bin')}"
    save_path = os.path.join(_UPLOAD_DIR, safe_name)
    total = 0
    try:
        with open(save_path, "wb") as out:
            for i in range(total_chunks):
                with open(os.path.join(d, f"{i}.part"), "rb") as pf:
                    while True:
                        buf = pf.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                        total += len(buf)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return JSONResponse({"error": f"合并失败: {e}"}, status_code=500,
                            media_type="application/json; charset=utf-8")
    shutil.rmtree(d, ignore_errors=True)
    base = get_public_base()
    qtoken = f"&token={AUTH_TOKEN}" if AUTH_TOKEN else ""
    watch_url = f"{base}/video_watch?file_id={safe_name}{qtoken}"
    stream_url = f"{base}/api/video_stream?file_id={safe_name}{qtoken}"
    return JSONResponse({
        "success": True,
        "url": f"/uploaded/{safe_name}",
        "public_url": f"{base}/uploaded/{safe_name}",
        "file_id": safe_name,
        "watch_url": watch_url,
        "stream_url": stream_url,
        "filename": meta.get("filename", ""),
        "size": total,
        "expires_in": _UPLOAD_TTL_SEC,
        "workflow_hint": ("把上面 file_id 传给 mcp 工具 analyze_video_file(file_id=...) "
                          "即可拿到分析结果。"),
    }, media_type="application/json; charset=utf-8")


async def api_uploaded_get(request: Request):
    """GET /uploaded/<safe_name> 返回上传的二进制文件(走 Bearer 鉴权)。"""
    name = request.path_params.get("name", "")
    if not name or "/" in name or ".." in name:
        return PlainTextResponse("bad request", status_code=400)
    fp = os.path.join(_UPLOAD_DIR, name)
    if not os.path.isfile(fp):
        return PlainTextResponse("not found", status_code=404)
    # 流式返回(避免一次性把大视频读进内存),用 FileResponse 内置
    ext = os.path.splitext(name)[1].lower()
    ctype = "application/octet-stream"
    if ext in {".jpg", ".jpeg"}: ctype = "image/jpeg"
    elif ext == ".png": ctype = "image/png"
    elif ext in {".mp4"}: ctype = "video/mp4"
    elif ext in {".mov", ".avi", ".mkv", ".webm"}: ctype = f"video/{ext.lstrip('.')}"
    with open(fp, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=ctype)


async def api_results_get(request: Request):
    """GET /results/<safe_name>[?download=1] 返回检测结果文件(标注图/标注视频)。

    与 /uploaded 不同, /results 是【公开只读】的(因为 BearerAuthMiddleware 已放行 /results),
    方便把链接直接贴到聊天窗口、浏览器或分享给别人在线观看/下载。
    加 ?download=1 时触发浏览器下载(默认 inline 在线预览)。
    """
    name = request.path_params.get("name", "")
    if not name or "/" in name or "\\" in name or ".." in name:
        return PlainTextResponse("bad request", status_code=400)
    fp = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(fp):
        return PlainTextResponse("not found", status_code=404)
    ext = os.path.splitext(name)[1].lower()
    ctype = "application/octet-stream"
    if ext in {".jpg", ".jpeg"}: ctype = "image/jpeg"
    elif ext == ".png": ctype = "image/png"
    elif ext == ".mp4": ctype = "video/mp4"
    elif ext in {".mov", ".avi", ".mkv", ".webm"}: ctype = f"video/{ext.lstrip('.')}"
    download = request.query_params.get("download", "") in ("1", "true", "yes")
    filename = name.split("_", 2)[-1] or name
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    with open(fp, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=ctype, headers=headers)


async def api_video_stream(request: Request):
    """GET /api/video_stream?file_id=xxx[&sample_interval=N][&token=xxx]
    以 multipart/x-mixed-replace (MJPEG) 流式推送已上传视频的【逐帧实时标注】。

    浏览器用法: <img src="/api/video_stream?file_id=xxx&token=xxx"> 即可看到逐帧带框画面。
    每帧图上叠加: 帧号 / 已用秒数 / 累计检出数 / 各类行为计数。
    处理完后推一帧"分析完成"总结画面, 再关闭 multipart 边界。

    鉴权: 优先 query.token, 其次 Authorization Bearer header。
    """
    # 鉴权(支持 query token 或 Bearer, 与 /uploaded/ 保持一致)
    token_q = request.query_params.get("token", "")
    auth_h = request.headers.get("authorization", "")
    provided = token_q or (auth_h.split(" ", 1)[1] if auth_h.lower().startswith("bearer ") else "")
    if AUTH_TOKEN and provided != AUTH_TOKEN:
        return PlainTextResponse("Unauthorized", status_code=401,
                                 media_type="text/plain; charset=utf-8")
    file_id = request.query_params.get("file_id", "").strip()
    if not file_id:
        return PlainTextResponse("missing file_id", status_code=400)
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        return PlainTextResponse("bad file_id", status_code=400)
    fp = os.path.join(_UPLOAD_DIR, file_id)
    if not os.path.isfile(fp):
        return PlainTextResponse("file not found", status_code=404)

    try:
        sample_interval = max(1, int(request.query_params.get("sample_interval", "10")))
    except Exception:
        sample_interval = 10
    tc_param = request.query_params.get("target_classes", "").strip()
    filter_indices = None
    if tc_param:
        name_to_idx = {n: i for i, n in enumerate(Config.CH_names)}
        filter_indices = [name_to_idx[n] for n in (x.strip() for x in tc_param.split(",")) if n in name_to_idx]

    async def gen():
        import numpy as np
        boundary = b"--frame\r\n"
        # 打开视频
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            err = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(err, "Cannot open video", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, jpg = cv2.imencode('.jpg', err)
            data = jpg.tobytes()
            yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            yield boundary + b"--\r\n"
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        all_stats = {}
        frame_idx = 0
        last_locs, last_cls, last_conf = [], [], []
        cls_colors = [(0, 200, 255), (255, 100, 0), (0, 255, 100), (0, 0, 255), (200, 0, 255), (255, 200, 0)]
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % sample_interval == 0:
                    results = model(frame, classes=filter_indices)[0] if filter_indices else model(frame)[0]
                    cls_list = [int(i) for i in results.boxes.cls.tolist()]
                    conf_list = [round(float(c) * 100, 2) for c in results.boxes.conf.tolist()]
                    locs = [list(map(int, e)) for e in results.boxes.xyxy.tolist()]
                    for c in cls_list:
                        n = Config.CH_names[c] if c < len(Config.CH_names) else f"class_{c}"
                        all_stats[n] = all_stats.get(n, 0) + 1
                    last_locs, last_cls, last_conf = locs, cls_list, conf_list
                annotated = annotate_frame(frame, last_locs, last_cls, last_conf, cls_colors, CN_FONT)
                # 叠加实时统计
                elapsed = time.time() - start_time
                header = f"\u5e27 {frame_idx}/{total_frames}  |  \u7528\u65f6 {elapsed:.1f}s  |  \u7d2f\u8ba1 {sum(all_stats.values())}\u6b21"
                cv2.putText(annotated, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos = 56
                for cls_name in Config.CH_names:
                    if cls_name in all_stats and all_stats[cls_name] > 0:
                        cv2.putText(annotated, f"{cls_name}: {all_stats[cls_name]}",
                                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_pos += 24
                _, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                data = jpg.tobytes()
                yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(0)  # 让出事件循环给其它请求
        finally:
            cap.release()

        # 总结帧
        sum_h, sum_w = max(h, 360), max(w, 640)
        summary = np.zeros((sum_h, sum_w, 3), dtype=np.uint8)
        cv2.putText(summary, "ANALYSIS COMPLETE / \u5206\u6790\u5b8c\u6210",
                    (max(20, sum_w // 2 - 240), sum_h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(summary,
                    f"\u5171\u5904\u7406 {frame_idx} \u5e27 | \u603b\u68c0\u51fa {sum(all_stats.values())} \u6b21 | \u7528\u65f6 {time.time()-start_time:.1f}s",
                    (20, sum_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos = sum_h // 2 + 40
        for cls_name in Config.CH_names:
            cnt = all_stats.get(cls_name, 0)
            if cnt > 0:
                cv2.putText(summary, f"{cls_name}: {cnt}",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 32
        _, jpg = cv2.imencode('.jpg', summary, [cv2.IMWRITE_JPEG_QUALITY, 90])
        data = jpg.tobytes()
        yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
               + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        yield boundary + b"--\r\n"  # multipart 结束

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Connection": "keep-alive"},
    )


CAM_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>课堂行为实时监控</title>
<style>
  :root{--bg:#0f1320;--panel:#171c2e;--txt:#e8ecf5;--muted:#8a93a8;--accent:#4f8cff;--ok:#39d98a;--err:#ff6b6b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:14px 18px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #232a40}
  header h1{font-size:17px;margin:0;font-weight:600}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--err);box-shadow:0 0 8px var(--err)}
  .dot.live{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
  .stage{flex:1 1 480px;min-width:320px;background:#000;border-radius:12px;overflow:hidden;position:relative;aspect-ratio:4/3}
  .stage img{width:100%;height:100%;object-fit:contain;display:block}
  .stage .ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:14px;text-align:center;padding:20px;line-height:1.7}
  .side{flex:0 0 280px;display:flex;flex-direction:column;gap:12px}
  .card{background:var(--panel);border-radius:12px;padding:14px}
  .card h2{font-size:13px;margin:0 0 10px;color:var(--muted);font-weight:600;letter-spacing:.5px}
  .stat{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #222a40}
  .stat:last-child{border-bottom:none}
  .stat .name{display:flex;align-items:center;gap:8px}
  .stat .sw{width:10px;height:10px;border-radius:3px}
  .stat .cnt{font-variant-numeric:tabular-nums;font-weight:600}
  .meta{font-size:12px;color:var(--muted);line-height:1.7}
  .controls{display:flex;gap:8px;flex-wrap:wrap}
  button{background:#222b44;color:var(--txt);border:1px solid #2e3a5c;border-radius:8px;padding:8px 12px;cursor:pointer;font-size:13px}
  button:hover{background:#2b3658}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  .token-box{display:none;flex-direction:column;gap:8px}
  .token-box input{background:#0c1020;border:1px solid #2e3a5c;color:var(--txt);padding:8px;border-radius:8px;font-size:13px}
  .err{color:var(--err);font-size:13px;min-height:18px}
  .fps{color:var(--accent)}
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>课堂行为实时监控</h1>
  <span id="status" class="meta">初始化…</span>
</header>
<div class="wrap">
  <div class="stage">
    <img id="out" alt="实时画面" style="display:none">
    <div id="ph" class="ph">正在请求摄像头权限…<br>请点击「允许」以使用本机摄像头</div>
  </div>
  <div class="side">
    <div class="card">
      <h2>行为统计</h2>
      <div id="stats"><div class="meta">等待检测…</div></div>
    </div>
    <div class="card">
      <h2>状态</h2>
      <div class="meta">推理耗时: <span id="lat">-</span></div>
      <div class="meta">往返延迟: <span id="rtt">-</span></div>
      <div class="meta">帧率: <span id="fps" class="fps">-</span></div>
      <div class="err" id="err"></div>
    </div>
    <div class="card">
      <h2>控制</h2>
      <div class="controls">
        <button id="btnPause" class="primary">暂停</button>
        <button id="btnCam">切换摄像头</button>
      </div>
      <div class="token-box" id="tokenBox">
        <div class="meta">未从链接获取 token, 请粘贴链接里的 token:</div>
        <input id="tokenInput" placeholder="Bearer token">
        <button id="btnToken" class="primary">开始</button>
      </div>
    </div>
  </div>
</div>
<video id="vid" autoplay playsinline muted style="display:none"></video>
<canvas id="cv" style="display:none"></canvas>
<script>
(function(){
  var $=function(id){return document.getElementById(id);};
  var vid=$('vid'), cv=$('cv'), ctx=cv.getContext('2d');
  var out=$('out'), ph=$('ph'), dot=$('dot'), statusEl=$('status');
  var statsEl=$('stats'), errEl=$('err'), latEl=$('lat'), rttEl=$('rtt'), fpsEl=$('fps');
  var PALETTE=['#ff7a45','#4f8cff','#39d98a','#ff6b6b','#b37feb','#ffd166'];
  var TOKEN=new URLSearchParams(location.search).get('token')||'';
  var CLASSES=[];
  var streaming=true, busy=false, currentCam=0, cams=[];
  var frames=0, fpsT=performance.now();

  function setErr(m){errEl.textContent=m||'';}
  function setLive(on){dot.className='dot'+(on?' live':'');}
  function loadMeta(){
    return fetch('/healthz').then(function(r){return r.json();}).then(function(d){
      CLASSES=d.classes||CLASSES;
    }).catch(function(){});
  }
  function renderStats(stats){
    var keys=CLASSES.length?CLASSES:Object.keys(stats);
    var total=0, rows=keys.map(function(k,i){
      var c=stats[k]||0; total+=c;
      var color=PALETTE[i%PALETTE.length];
      return '<div class="stat"><span class="name"><span class="sw" style="background:'+color+'"></span>'+k+'</span><span class="cnt">'+c+'</span></div>';
    });
    statsEl.innerHTML=(total===0?'<div class="meta">未检测到行为</div>':rows.join(''));
  }
  function loop(){
    if(!streaming) return;
    if(!busy && vid.videoWidth>0){
      busy=true;
      try{
        var w=640, h=Math.max(1,Math.round(vid.videoHeight*w/vid.videoWidth));
        cv.width=w; cv.height=h; ctx.drawImage(vid,0,0,w,h);
        var b64=cv.toDataURL('image/jpeg',0.7).split(',')[1];
        var t0=performance.now();
        fetch('/api/detect',{
          method:'POST',
          headers:{'Content-Type':'application/json','Authorization':'Bearer '+TOKEN},
          body:JSON.stringify({image_base64:b64})
        }).then(function(resp){
          return resp.json().then(function(data){return {resp:resp,data:data};});
        }).then(function(o){
          var data=o.data, resp=o.resp, rtt=performance.now()-t0;
          if(resp.status===401){setErr('鉴权失败: token 无效');streaming=false;return;}
          if(!data.success){setErr(data.error||'检测失败');return;}
          out.src='data:image/jpeg;base64,'+data.annotated_image_base64;
          renderStats(data.stats||{});
          latEl.textContent=data.time||'-';
          rttEl.textContent=Math.round(rtt)+' ms';
          frames++; var now=performance.now();
          if(now-fpsT>1000){fpsEl.textContent=(frames*1000/(now-fpsT)).toFixed(1);frames=0;fpsT=now;}
        }).catch(function(e){setErr('网络错误: '+e.message);});
      }catch(e){setErr('捕获错误: '+e.message);}
      finally{busy=false;}
    }
    setTimeout(loop,100);
  }
  function initCam(deviceId){
    var constraints=deviceId?{video:{deviceId:{exact:deviceId}}}:{video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'}};
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
      vid.srcObject=stream; return vid.play();
    }).then(function(){
      return navigator.mediaDevices.enumerateDevices();
    }).then(function(devs){
      cams=devs.filter(function(d){return d.kind==='videoinput';});
      statusEl.textContent='摄像头已开启 · '+(cams.length>1?('摄像头'+(currentCam+1)+'/'+cams.length):'单摄像头');
      setLive(true); setErr('');
      ph.style.display='none'; out.style.display='block';
      loop();
    }).catch(function(e){
      setLive(false);
      ph.style.display='flex'; out.style.display='none';
      ph.innerHTML='无法访问摄像头:<br>'+e.message+'<br><br>请检查浏览器权限, 或改用 https 链接打开本页面。';
    });
  }
  $('btnPause').onclick=function(){
    streaming=!streaming;
    $('btnPause').textContent=streaming?'暂停':'继续';
    if(streaming){setErr('');loop();}
  };
  $('btnCam').onclick=function(){
    if(cams.length<2){setErr('仅检测到 1 个摄像头');return;}
    currentCam=(currentCam+1)%cams.length;
    initCam(cams[currentCam].deviceId);
  };
  loadMeta().then(function(){
    if(!TOKEN){
      $('tokenBox').style.display='flex';
      $('btnToken').onclick=function(){
        TOKEN=$('tokenInput').value.trim();
        if(TOKEN){$('tokenBox').style.display='none';initCam();}
      };
      statusEl.textContent='需要 token';
      return;
    }
    initCam();
  });
})();
</script>
</body>
</html>'''


async def camera_monitor_page(request: Request):
    """摄像头实时监控网页: 在调用方浏览器中打开, 自动调用其本机摄像头并实时标注。
    页面是静态HTML(无外部依赖), 通过同域 fetch 调 /api/detect 完成推理。"""
    return HTMLResponse(CAM_PAGE, media_type="text/html; charset=utf-8")


VIDEO_WATCH_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u8bfe\u5802\u884c\u4e3a\u89c6\u9891\u5b9e\u65f6\u5206\u6790</title>
<style>
  :root{--bg:#0f1320;--panel:#171c2e;--txt:#e8ecf5;--muted:#8a93a8;--accent:#4f8cff;--ok:#39d98a;--err:#ff6b6b;--warn:#ffb547}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;min-height:100vh}
  header{padding:14px 18px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #232a40}
  header h1{font-size:17px;margin:0;font-weight:600}
  header .meta{margin-left:auto;font-size:12px;color:var(--muted)}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--err);box-shadow:0 0 8px var(--err)}
  .dot.live{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  .dot.done{background:var(--accent);box-shadow:0 0 8px var(--accent)}
  .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
  .stage{flex:1 1 480px;min-width:320px;background:#000;border-radius:12px;overflow:hidden;position:relative;aspect-ratio:4/3}
  .stage img{width:100%;height:100%;object-fit:contain;display:block}
  .stage .ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:14px;text-align:center;padding:20px;line-height:1.7;flex-direction:column;gap:14px}
  .stage .ph .spin{width:36px;height:36px;border:3px solid #2a3148;border-top-color:var(--accent);border-radius:50%;animation:r 1s linear infinite}
  @keyframes r{to{transform:rotate(360deg)}}
  .side{flex:0 0 280px;display:flex;flex-direction:column;gap:12px}
  .card{background:var(--panel);border-radius:12px;padding:14px}
  .card h2{font-size:13px;margin:0 0 10px;color:var(--muted);font-weight:600;letter-spacing:.5px}
  .stat{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #222a40}
  .stat:last-child{border-bottom:none}
  .stat .name{display:flex;align-items:center;gap:8px}
  .dot2{width:8px;height:8px;border-radius:50%;display:inline-block}
  .stat .v{font-variant-numeric:tabular-nums;font-weight:600;color:var(--txt)}
  .bar{height:6px;background:#222a40;border-radius:3px;overflow:hidden;margin-top:4px}
  .bar > span{display:block;height:100%;background:var(--accent);border-radius:3px;transition:width .4s ease}
  .ctrls{display:flex;gap:8px;margin-top:10px}
  button{flex:1;background:var(--accent);color:#fff;border:none;padding:9px 12px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
  button:hover{filter:brightness(1.1)}
  button.ghost{background:#222a40;color:var(--txt)}
  .log{font-size:11px;color:var(--muted);margin-top:8px;font-family:ui-monospace,monospace;background:#0a0e1a;padding:8px;border-radius:6px;max-height:120px;overflow:auto}
  .err{color:var(--err)}
  .ok{color:var(--ok)}
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <h1>\u8bfe\u5802\u884c\u4e3a\u89c6\u9891\u5b9e\u65f6\u5206\u6790</h1>
  <span class="meta" id="meta">\u7b49\u5f85\u4e2d...</span>
</header>
<div class="wrap">
  <div class="stage">
    <img id="v" alt="MJPEG stream" style="display:none">
    <div class="ph" id="ph">
      <div class="spin"></div>
      <div>\u6b63\u5728\u8fde\u63a5\u670d\u52a1\u7aef\u5e27\u6d41...<br>\u8fde\u4e0a\u540e\u4f1a\u770b\u5230\u5e26 YOLO \u68c0\u6d4b\u6846\u7684\u5b9e\u65f6\u753b\u9762\u3002</div>
    </div>
  </div>
  <aside class="side">
    <div class="card">
      <h2>\u8fdb\u5ea6\u4e0e\u8d44\u6599</h2>
      <div class="stat"><span class="name">\u5df2\u5904\u7406\u5e27\u6570</span><span class="v" id="v_frames">0</span></div>
      <div class="stat"><span class="name">\u5df2\u7528\u65f6\u95f4</span><span class="v" id="v_elapsed">0.0s</span></div>
      <div class="stat"><span class="name">\u7d2f\u8ba1\u68c0\u51fa</span><span class="v" id="v_total">0</span></div>
      <div class="bar"><span id="prog" style="width:0%"></span></div>
    </div>
    <div class="card">
      <h2>\u5404\u7c7b\u884c\u4e3a\u8ba1\u6570</h2>
      <div id="stats">\u6682\u65e0\u68c0\u51fa</div>
    </div>
    <div class="card">
      <h2>\u63a7\u5236</h2>
      <div class="ctrls">
        <button onclick="restart()">\u91cd\u65b0\u5206\u6790</button>
        <button class="ghost" onclick="window.close()">\u5173\u95ed</button>
      </div>
      <div class="log" id="log"></div>
    </div>
  </aside>
</div>
<script>
const qs = new URLSearchParams(location.search);
const FILE_ID = qs.get('file_id') || '';
let img = document.getElementById('v');
let dot = document.getElementById('dot');
let ph = document.getElementById('ph');
let meta = document.getElementById('meta');
const logEl = document.getElementById('log');
function log(m, cls='') { logEl.innerHTML += `<div class="${cls}">${new Date().toLocaleTimeString()} ${m}</div>`; logEl.scrollTop = logEl.scrollHeight; }
function buildStreamUrl() {
  let url = '/api/video_stream?file_id=' + encodeURIComponent(FILE_ID) + '&sample_interval=10';
  if (qs.get('token')) url += '&token=' + encodeURIComponent(qs.get('token'));
  return url;
}
function start() {
  img.style.display = 'block';
  ph.style.display = 'none';
  dot.className = 'dot live';
  meta.textContent = '\u5b9e\u65f6\u5206\u6790\u4e2d';
  // \u52a0\u4e2a\u7f13\u5b58\u7834\u65e7\u7684\u53c2\u6570, \u5f3a\u5236\u91cd\u8fde
  img.src = buildStreamUrl() + '&_t=' + Date.now();
  log('\u8fde\u63a5\u5e27\u6d41: ' + img.src.substring(0, 80) + '...');
  img.onload = () => log('\u9996\u5e27\u5df2\u8fbe', 'ok');
  img.onerror = () => log('\u5e27\u6d41\u4e2d\u65ad, \u5c1d\u8bd5\u91cd\u8fde...', 'err');
}
function restart() {
  meta.textContent = '\u91cd\u542f\u4e2d...';
  img.src = '';
  setTimeout(start, 200);
}
if (!FILE_ID) {
  ph.innerHTML = '<div style="color:var(--err)">\u7f3a\u5c11 file_id \u53c2\u6570</div>';
} else {
  start();
}
// \u4ece\u670d\u52a1\u7aef\u8bfb\u53d6\u89c6\u9891\u5143\u4fe1\u606f\u5237\u65b0\u8fdb\u5ea6
fetch('/uploaded/' + encodeURIComponent(FILE_ID), { headers: { 'Authorization': 'Bearer ' + (qs.get('token') || '') } })
  .then(r => { if (!r.ok) throw 'HTTP ' + r.status; return r.blob(); })
  .then(b => { meta.textContent = '\u89c6\u9891\u5927\u5c0f: ' + (b.size/1024/1024).toFixed(2) + ' MB'; })
  .catch(e => { meta.textContent = '\u89c6\u9891\u52a0\u8f7d\u5931\u8d25: ' + e; });
// \u9636\u6bb5\u6027\u5237\u65b0\u8fdb\u5ea6(\u8f7b\u91cf\u4f30\u8ba1)
let startTs = Date.now();
setInterval(() => {
  document.getElementById('v_elapsed').textContent = ((Date.now()-startTs)/1000).toFixed(1) + 's';
}, 250);
window.addEventListener('beforeunload', () => { if (img) img.src = ''; });
</script>
</body>
</html>'''


async def video_watch_page(request: Request):
    """/video_watch?file_id=xxx[&token=xxx] - 把 MJPEG 流包装进一个友好 UI,
    实时显示帧号/耗时/各类行为计数, 处理完后看到总结画面。"""
    return HTMLResponse(VIDEO_WATCH_PAGE, media_type="text/html; charset=utf-8")


app = Starlette(
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
                Middleware(BearerAuthMiddleware), Middleware(RewriteHostMiddleware)],
    routes=[
        Route("/healthz", health),
        Route("/tools/list", api_list_tools),
        Route("/tools/call", api_call_tool, methods=["POST"]),
        Route("/api/detect", api_detect, methods=["POST"]),
        Route("/upload", api_upload, methods=["POST"]),
        Route("/upload_init", api_upload_init, methods=["POST"]),
        Route("/upload_chunk", api_upload_chunk, methods=["POST"]),
        Route("/upload_complete", api_upload_complete, methods=["POST"]),
        Route("/uploaded/{name}", api_uploaded_get),
        Route("/results/{name}", api_results_get),
        Route("/api/video_stream", api_video_stream),
        Route("/cam", camera_monitor_page),
        Route("/video_watch", video_watch_page),
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
        Mount("/", sse_app),
    ],
)


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  课堂行为检测 MCP 服务器 (mcp 2.x, 并发安全版)")
    print(f"  SSE:            http://0.0.0.0:{PORT}/sse")
    print(f"  Streamable HTTP: http://0.0.0.0:{PORT}/mcp")
    print(f"  健康检查:         http://0.0.0.0:{PORT}/healthz")
    print(f"  鉴权:            {'Bearer Token 已启用' if AUTH_TOKEN else '未启用(仅本地调试!)'}")
    print(f"  并发上限:         {MAX_CONCURRENT} (MCP_MAX_CONCURRENT 可调)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

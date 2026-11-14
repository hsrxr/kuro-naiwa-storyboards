"""Serve the review page and its single writable annotation file on localhost."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import unquote, urlsplit, quote
import argparse
import json
import mimetypes
import os
import webbrowser
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PAGE = Path("视频素材/完整素材_场景与人物替换/04_待人工处理/逐张标注.html")
NAME = "33镜头_人工标注清单.json"
TARGET = ROOT / NAME
LOCK = Lock()
ROLES = {'牛来', '牢大', '小黑子', '曼波', '奶娃', '水豚嘟嘟', '熊大', 'aabc', '肥嘟嘟', '乌萨奇', '魔童', '豆包'}
IDENTITY = ("id", "frame", "seconds", "image", "width", "height")


def revision(content):
    return '"' + sha256(content).hexdigest() + '"'


def validate(data, current):
    if data["format"] != current["format"] or len(data["shots"]) != len(current["shots"]):
        raise ValueError("清单格式或镜头总数有误")
    for shot, source in zip(data["shots"], current["shots"]):
        if any(shot[k] != source[k] for k in IDENTITY):
            raise ValueError("镜头编号或底图信息不匹配")
        regions = shot["regions"]
        if not isinstance(shot["note"], str) or not isinstance(regions, list):
            raise ValueError("备注或区域格式有误")
        if shot["status"] not in {"skip", "pending", "annotated"} or bool(regions) != (shot["status"] == "annotated"):
            raise ValueError("处理状态与区域不一致")
        for region in regions:
            label = region["character"]
            if (region["action"], label) not in ({("replace", r) for r in ROLES} | {("skip_region", None)}):
                raise ValueError("未知角色或操作")
            x, y, right, bottom = region["bbox_xyxy"]
            if not all(type(n) is int for n in (x,y,right,bottom)) or not (0 <= x < right <= shot["width"] and 0 <= y < bottom <= shot["height"]):
                raise ValueError("区域超出底图范围")


def replace_file(path, content):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        temp.write_bytes(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Review(BaseHTTPRequestHandler):
    def reply(self, status, body=b"", mime="text/plain; charset=utf-8", tag=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        if tag:
            self.send_header("ETag", tag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def local_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        if not self.local_host():
            return self.reply(403)
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + quote(PAGE.as_posix()))
            self.end_headers()
            return
        if path == "/api/location":
            return self.reply(200, str(TARGET).encode())
        if path == "/api/annotations":
            data = TARGET.read_bytes()
            return self.reply(200, data, "application/json", revision(data))
        candidate = (ROOT / path.lstrip("/")).resolve()
        dirs = [ROOT / PAGE.parent / d for d in ("原截图对照", "待处理底图", "前后帧参考")]
        dirs.append(ROOT / "视频素材/人物")
        if candidate != ROOT / PAGE and not (candidate.parent in dirs and candidate.suffix.lower() in {".png", ".jpg"}):
            return self.reply(404)
        if not candidate.is_file():
            return self.reply(404)
        self.reply(200, candidate.read_bytes(), mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")

    def do_PUT(self):
        if not self.local_host() or self.path != "/api/annotations" or self.headers.get("Origin") != "http://" + self.headers["Host"]:
            return self.reply(403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 2000000:
                return self.reply(413)
            data = json.loads(self.rfile.read(length))
            with LOCK:
                old = TARGET.read_bytes()
                if self.headers.get("If-Match") != revision(old):
                    return self.reply(409)
                validate(data, json.loads(old))
                encoded = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
                replace_file(TARGET, encoded)
                mirror = ROOT / "kuro-naiwa-storyboards" / NAME
                if mirror.is_file() and mirror.read_bytes() in (old, encoded):
                    replace_file(mirror, encoded)
            self.reply(200, b'{}', "application/json", revision(encoded))
        except (KeyError, ValueError, TypeError, UnicodeError):
            self.reply(400)
        except OSError:
            self.reply(500)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--port", type=int, default=18733)
    args = parser.parse_args()
    for port in range(args.port, args.port + (12 if args.open else 1)):
        url = f"http://127.0.0.1:{port}/"
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Review)
            break
        except OSError:
            if not args.open:
                raise
            try:
                with urlopen(url + "api/location", timeout=1) as response:
                    if response.read().decode() == str(TARGET):
                        webbrowser.open(url)
                        raise SystemExit(0)
            except OSError:
                pass
    else:
        raise RuntimeError("无法启动标注页面，请指定可用端口")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()

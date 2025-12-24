# FA-Kara WebUI

基于 [FA-Kara](https://github.com/moriwx/FA-Kara) 的卡拉OK歌词对齐工具 WebUI 版本。

支持日语/英语/中文歌词的时间轴对齐，输出 ASS/LRC 格式。

## ✨ 特性

- 🎤 **自动对齐** - 基于音频和歌词文本自动生成时间轴
- 🌍 **多语言支持** - 日语、英语、中文
- 📝 **多格式输出** - ASS 字幕、LRC 歌词（标准/Ruby 注音）
- 🎨 **现代 UI** - 赛博朋克风格 Gradio 界面
- 🐳 **容器化部署** - 支持 Podman/Docker Compose 一键启动
- 🔗 **内网穿透** - 可选 frpc 支持远程访问

## 🚀 快速开始

### 前置要求

- NVIDIA GPU + CUDA 驱动
- Podman 或 Docker（带 GPU 支持）

### 启动服务

```bash
# 仅启动 FA-Kara WebUI
podman compose up -d --build

# 启动 WebUI + frpc 内网穿透
podman compose --profile frpc up -d --build
```

访问：http://localhost:27860

### 停止服务

```bash
podman compose down
# 或带 frpc
podman compose --profile frpc down
```

## 📁 项目结构

```
fa-kara-web/
├── app.py              # Gradio WebUI 入口
├── compose.yaml        # Podman/Docker Compose 配置
├── Dockerfile          # 容器镜像构建
├── requirements.txt    # Python 依赖
├── config/
│   ├── frpc.toml           # frpc 配置 (不提交到 git)
│   └── frpc.example.toml   # frpc 配置模板
└── FA-Kara/            # 原始 FA-Kara 核心代码
    ├── align.py
    ├── main.py
    └── ...
```

## ⚙️ 配置

### 内网穿透 (可选)

1. 复制配置模板：
   ```bash
   cp config/frpc.example.toml config/frpc.toml
   ```

2. 编辑 `config/frpc.toml`，填入你的 frps 服务器信息

3. 使用 `--profile frpc` 启动

### 端口修改

编辑 `compose.yaml` 中的 ports 映射：
```yaml
ports:
  - "27860:7860"  # 修改 27860 为你想要的端口
```

## 🛠️ 开发

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt
cd FA-Kara && pip install -r requirements.txt

# 启动
python app.py
```

### 重新构建镜像

```bash
podman compose build --no-cache
```

## 📝 License

本项目基于 FA-Kara，遵循其原始许可证。

## 🙏 致谢

- [FA-Kara](https://github.com/moriwx/FA-Kara) - 原始项目


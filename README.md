# 雷池智能问答助手

基于 Flask + LLM 的智能问答系统，支持图片识别和知识库检索。

## 功能特性

- 💬 智能问答 - 基于知识库的智能问答系统
- 🖼️ 图片识别 - 支持上传图片并识别其中的文字
- 🔍 OCR 提取 - 自动提取图片中的文字信息
- ✨ AI 润色 - LLM 智能润色回答内容
- 🎨 精美 UI - 现代化界面设计，支持多种动画效果

## 技术栈

- **后端**: Flask + Python
- **前端**: HTML + CSS + JavaScript
- **AI**: Qwen VL Plus (视觉模型) + Qwen 2.5 (对话模型)
- **知识库**: 本地 JSON 知识库 + Embedding 检索

## 快速开始

```bash
# 安装依赖
pip install flask flask-cors openai

# 启动服务
python app.py

# 访问
http://localhost:5000
```

## API 接口

- `GET /` - 首页
- `POST /api/chat` - 文字问答
- `POST /api/chat_with_image` - 图片问答
- `POST /api/ocr` - 图片文字识别
- `POST /api/dingtalk` - 钉钉机器人

## 项目结构

```
agent/
├── app.py              # Flask 后端
├── knowledge.json      # 知识库
├── templates/
│   └── index.html      # 前端页面
└── README.md           # 项目文档
```

##  License

MIT

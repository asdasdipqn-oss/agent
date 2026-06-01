# 雷池智能问答助手

基于 Flask + LLM 的智能问答系统，集成雷池 WAF API，支持知识库检索、图片识别和 WAF 操作。

## 功能特性

- 智能问答 - 基于知识库的智能问答系统
- 图片识别 - 支持上传图片并识别其中的文字
- OCR 提取 - 自动提取图片中的文字信息
- AI 润色 - LLM 润色回答内容
- WAF Agent - 通过自然语言操作雷池 WAF（查看攻击日志、管理规则、IP组等）
- 钉钉机器人 - 支持钉钉消息回调接口

## 技术栈

- **后端**: Flask + Python
- **前端**: HTML + CSS + JavaScript
- **AI**: Qwen VL Plus (视觉模型) + Qwen 2.5 (对话模型)
- **知识库**: 本地 JSON 知识库 + Embedding 检索
- **WAF API**: 雷池 WAF OpenAPI 接口对接

## 快速开始

```bash
# 安装依赖
pip install flask flask-cors openai

# 启动服务
python app.py

# 访问
http://localhost:5001
```

## API 接口

- `GET /` - 首页
- `POST /api/chat` - 文字问答（支持 WAF 工具调用）
- `POST /api/chat_with_image` - 图片问答
- `POST /api/ocr` - 图片文字识别
- `POST /api/dingtalk` - 钉钉机器人
- `POST /api/feedback` - 提交反馈
- `POST /api/knowledge_search` - 知识库搜索

## WAF Agent 工具列表

通过 `/api/chat` 接口传入 WAF 连接信息，可使用自然语言操控雷池 WAF：

| 工具名 | 功能 |
|--------|------|
| query_attack_logs | 查询攻击日志 |
| get_attack_log_detail | 查询攻击日志详情 |
| query_attack_events | 查询攻击事件 |
| query_qps | 查询 QPS 统计 |
| list_custom_rules | 列出自定义规则(黑白名单) |
| create_custom_rule | 创建自定义规则 |
| toggle_rule | 启用/禁用规则 |
| delete_rule | 删除规则 |
| list_ip_groups | 列出IP组 |
| create_ip_group | 创建IP组 |
| add_ip_to_group | 向IP组添加IP |
| get_ip_group_detail | 获取IP组详情 |
| update_ip_group | 更新IP组 |
| manage_api_token | 管理 API Token |
| search_knowledge | 搜索知识库 |

完整工具列表请参考 `waf_tools.py` 和 `waf_api_reference.md`。

## 项目结构

```
agent/
├── app.py                # Flask 后端（问答+Agent主入口）
├── waf_agent.py          # WAF Agent（LLM工具调用编排）
├── waf_tools.py          # WAF API客户端及工具定义
├── waf_swagger.json      # 雷池 WAF Swagger 接口定义
├── waf_api_reference.md  # WAF API 接口文档
├── knowledge.json        # 知识库数据
├── feedback_data.json    # 用户反馈数据
├── templates/
│   ├── index.html        # 前端问答页面
│   └── feedback.html     # 反馈页面
├── test_waf_functions.py # WAF 工具测试
├── test_waf_tools.sh     # WAF 工具 Shell 测试
└── README.md
```

## License

MIT
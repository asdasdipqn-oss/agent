# 雷池智能问答助手

基于 Flask + LLM 的雷池 WAF 智能问答与运维系统，集成 91 个 WAF 操作工具，支持通过自然语言操控雷池 WAF，同时提供知识库检索、图片识别、钉钉机器人等功能。

## 功能特性

- **智能问答** - 基于知识库的智能问答，支持 Embedding 语义检索 + LLM 润色回答
- **WAF Agent** - 通过自然语言操控雷池 WAF，涵盖攻击日志、规则管理、站点配置、统计查询等 91 个工具
- **图片识别** - 支持上传图片并识别其中的文字（Qwen VL Plus 视觉模型）
- **OCR 提取** - 自动提取图片中的文字信息
- **钉钉机器人** - 支持钉钉群消息回调，远程操控 WAF
- **反馈系统** - 用户可对回答进行点赞/点踩，帮助优化知识库

## 技术栈

- **后端**: Flask + Python 3
- **前端**: HTML + CSS + JavaScript（Glassmorphism 风格）
- **LLM**: OpenAI 兼容接口（默认使用长亭 AI 平台 Qwen 系列模型）
  - 对话模型: Qwen 2.5
  - 视觉模型: Qwen VL Plus
- **知识库**: 本地 JSON + Embedding 向量检索（cosine similarity）
- **WAF API**: 雷池 WAF OpenAPI 全量接口对接

## 快速开始

### 环境要求

- Python 3.8+
- 雷池 WAF 实例（用于 Agent 功能）

### 安装依赖

```bash
pip install flask flask-cors openai requests
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM API Key | 内置默认值 |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://aiapi.chaitin.net/v1` |
| `FLASK_PORT` | 服务端口 | `5001` |

WAF 连接信息通过前端界面或 API 请求参数动态传入，无需预配置。

### 启动服务

```bash
python app.py
```

访问 `http://localhost:5001` 即可使用。

## API 接口

### 页面路由

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 问答首页 |
| GET | `/feedback` | 反馈管理页面 |

### 问答与 Agent

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 文字问答（支持 WAF Agent 工具调用，流式响应） |
| POST | `/api/chat_with_image` | 图片问答 |
| POST | `/api/ocr` | 图片文字识别 |

请求参数示例（`/api/chat`）：

```json
{
  "query": "查看最近的攻击日志",
  "session_id": "user123",
  "waf_base_url": "https://your-waf:9443",
  "waf_api_token": "your-api-token"
}
```

### 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge/add` | 添加知识条目 |

### 历史与会话

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/conversation` | 获取当前会话 |
| POST | `/api/conversation/clear` | 清空当前会话 |
| GET | `/api/history` | 获取历史记录 |
| POST | `/api/history` | 保存历史记录 |
| POST | `/api/history/clear` | 清空历史记录 |

### 反馈

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/feedback` | 提交反馈（点赞/点踩） |
| GET | `/api/feedback` | 获取反馈列表 |
| PUT | `/api/feedback/<id>` | 更新反馈 |
| DELETE | `/api/feedback/<id>` | 删除反馈 |
| GET | `/api/feedback/search` | 搜索反馈 |
| POST | `/api/feedback/manual` | 手动添加反馈 |

### 钉钉集成

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/dingtalk` | 钉钉机器人回调接口 |

### 系统管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/waf/test` | 测试 WAF 连接 |
| POST | `/api/cache/clear` | 清空缓存 |

## WAF Agent 工具

WAF Agent 共提供 **91 个工具**，覆盖雷池 WAF 的全部 OpenAPI 接口。用户通过自然语言提问，LLM 自动选择合适的工具组合完成操作。

### 攻击日志（6 个）

| 工具 | 说明 |
|------|------|
| `query_attack_logs` | 查询攻击日志，支持按 IP、主机、URL、攻击类型筛选 |
| `get_attack_log_detail` | 获取单条攻击日志详情（请求头、响应头等） |
| `query_attack_events` | 查询攻击事件（聚合后的攻击记录） |
| `get_rule_attack_logs` | 查询黑白名单规则相关的攻击日志 |
| `get_auth_defense_logs` | 查询认证防护日志 |
| `export_attack_logs` | 导出攻击日志 |

### 自定义规则 / 黑白名单（6 个）

| 工具 | 说明 |
|------|------|
| `list_custom_rules` | 列出自定义规则（黑白名单） |
| `create_custom_rule` | 创建自定义规则（白名单 action=0，黑名单 action=1） |
| `toggle_rule` | 启用/禁用规则 |
| `delete_rule` | 删除规则 |
| `get_rule_detail` | 获取规则详情（匹配模式、动作等） |
| `update_rule` | 更新规则 |
| `order_rules` | 调整规则排序 |

### IP 组管理（8 个）

| 工具 | 说明 |
|------|------|
| `list_ip_groups` | 列出 IP 组 |
| `create_ip_group` | 创建 IP 组 |
| `add_ip_to_group` | 向 IP 组添加 IP |
| `get_ip_group_detail` | 获取 IP 组详情（含 IP 列表） |
| `update_ip_group` | 更新 IP 组（名称、IP 列表） |
| `delete_ip_group` | 删除 IP 组 |
| `get_crawler_group` | 获取搜索引擎蜘蛛 IP 组 |
| `update_crawler_group` | 更新搜索引擎蜘蛛配置 |
| `get_ip_group_by_link` | 通过链接获取 IP 地址 |
| `create_ip_group_by_link` | 通过链接创建 IP 组 |

### 站点管理（18 个）

| 工具 | 说明 |
|------|------|
| `list_sites` | 列出所有站点 |
| `get_site_detail` | 获取站点详细配置 |
| `create_site` | 创建站点 |
| `delete_site` | 删除站点 |
| `update_site_basic_info` | 更新站点基本信息 |
| `update_site_group` | 更新站点所属分组 |
| `get_site_proxy` | 获取站点代理/安全配置 |
| `set_site_proxy` | 设置站点代理/安全配置（HTTPS、HSTS、HTTP2 等） |
| `get_site_nginx_config` | 获取站点 Nginx 配置 |
| `set_site_nginx_config` | 更新站点 Nginx 配置 |
| `get_site_resources` | 获取站点路由资源列表 |
| `manage_site_excludes` | 管理路由采集排除配置 |
| `get_site_logs` | 获取站点访问/错误日志 |
| `manage_site_groups` | 管理站点分组（增删改查） |
| `manage_site_semantics` | 管理站点语义分析配置 |
| `manage_forwarding_rules` | 管理站点转发规则 |
| `manage_waiting_room` | 管理等候室配置和日志 |
| `manage_site_resources` | 获取站点资源 |

### 防护配置（12 个）

| 工具 | 说明 |
|------|------|
| `health_check` | 执行一次站点健康检查 |
| `set_site_health_check` | 开启/关闭健康检查功能 |
| `set_site_mode` | 设置站点模式（防护/下线/只检） |
| `set_challenge` | 设置人机验证配置 |
| `set_auth_defense` | 设置身份认证配置 |
| `set_rate_limit` | 设置频率限制（ACL） |
| `set_acl_enabled` | 开启/关闭 CC 防护 |
| `get_acl_logs` | 查询频率限制日志 |
| `release_acl_block` | 解除 IP 封禁 |
| `manage_anti_tamper` | 管理反篡改规则 |
| `manage_dynamic_defense` | 管理动态防护配置（HTML/JS/图片加密） |
| `get_blocking_message` | 获取/更新拦截页面消息 |

### 人机验证与增强规则（6 个）

| 工具 | 说明 |
|------|------|
| `get_challenge_config` | 获取全局人机验证配置 |
| `set_challenge_config` | 设置全局人机验证配置 |
| `get_enhance_rules` | 获取 Skynet 增强规则 |
| `update_enhance_rules` | 更新 Skynet 增强规则 |
| `manage_enhance_rule_switch` | 管理增强规则全局开关 |
| `get_anti_bot_logs` | 查询人机验证日志 |

### 统计与趋势（4 个）

| 工具 | 说明 |
|------|------|
| `query_qps` | 查询当前 QPS |
| `get_advance_stats` | 查询高级统计（访问/攻击/客户端/域名/地理位置/页面/状态码） |
| `get_advance_trends` | 查询高级趋势（访问趋势/拦截趋势） |
| `manage_report` | 管理报告（创建/查看/删除） |

### 证书管理（3 个）

| 工具 | 说明 |
|------|------|
| `list_certs` | 列出所有 SSL 证书 |
| `get_cert_detail` | 获取证书详情 |
| `manage_certs` | 管理证书（创建/更新/删除） |

### 认证与用户（7 个）

| 工具 | 说明 |
|------|------|
| `list_users` | 列出控制台用户 |
| `manage_users` | 管理控制台用户（创建/更新/删除/重置二次认证） |
| `manage_auth_sources` | 管理认证防护源（LDAP/本地等） |
| `list_auth_source_users` | 列出认证源用户 |
| `manage_auth_defense_users` | 管理认证防护用户 |

### 告警与通知（2 个）

| 工具 | 说明 |
|------|------|
| `get_alarm_config` | 获取告警配置（钉钉/飞书/企微/Telegram/Discord） |
| `update_alarm_config` | 更新告警配置或测试告警 |

### 系统配置（14 个）

| 工具 | 说明 |
|------|------|
| `get_system_info` | 获取系统信息（版本/架构等） |
| `get_license` | 获取授权信息 |
| `manage_license` | 管理授权（申请/重新申请/删除） |
| `manage_api_token` | 管理 API Token |
| `manage_global_proxy` | 管理全局代理设置 |
| `manage_syslog` | 管理 Syslog 配置 |
| `manage_ja4` | 管理 JA4 指纹配置 |
| `manage_detector` | 管理检测引擎性能模式 |
| `manage_log_clean` | 管理日志清理配置（留存天数） |
| `manage_security_posture` | 管理安全态势（实时/统计/趋势） |
| `manage_portal` | 管理门户配置 |
| `manage_network_proxy` | 管理系统网络代理 |
| `manage_intelligence` | 管理恶意 IP 情报共享 |
| `get_audit_log` | 查询审计日志 |

### 语义分析与云策略（4 个）

| 工具 | 说明 |
|------|------|
| `get_global_semantics` | 获取全局语义分析配置 |
| `set_global_semantics` | 更新全局语义分析配置 |
| `manage_cloud_policies` | 管理云策略（列出/订阅） |
| `manage_rate_limit` | 管理站点频率限制规则 |

### 频率限制（1 个）

| 工具 | 说明 |
|------|------|
| `search_knowledge` | 搜索知识库 |

## 工作原理

```
用户提问 → Flask API → LLM 判断是否需要工具调用
                              ↓
                    ┌──── 是 ────┐
                    ↓            ↓
              选择 WAF 工具   直接回答
                    ↓
           调用雷池 WAF API
                    ↓
           返回结果 → LLM 总结 → 流式响应
```

1. 用户通过 Web 界面或 API 提交问题
2. 系统先检索知识库（Embedding 语义匹配），若命中则直接返回
3. 若需要 WAF 操作，LLM 自动选择工具并调用对应 API
4. 支持多轮工具调用（如先查询规则列表，再查看规则详情）
5. 结果经 LLM 润色后流式返回给用户

## 项目结构

```
agent/
├── app.py                  # Flask 后端主入口（路由、问答逻辑、Agent 编排）
├── waf_agent.py            # WAF Agent（LLM 工具调用循环、工具注册）
├── waf_tools.py            # WAF API 客户端 + 91 个工具定义
├── waf_swagger.json        # 雷池 WAF OpenAPI Swagger 定义
├── waf_api_reference.md    # WAF API 接口参考文档
├── knowledge.json          # 知识库数据（373 条雷池问答）
├── feedback_data.json      # 用户反馈数据
├── templates/
│   ├── index.html          # 问答页面（流式响应 + WAF 工具展示）
│   └── feedback.html       # 反馈管理页面
├── test_waf_functions.py   # WAF 工具单元测试
├── test_waf_tools.sh       # WAF 工具 Shell 集成测试
├── .gitignore
└── README.md
```

## 相关链接

- [雷池 WAF 官网](https://waf.chaitin.cn/)
- [雷池 WAF GitHub](https://github.com/chaitin/SafeLine)

## License

MIT
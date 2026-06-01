#!/usr/bin/env python3
"""
雷池问答Agent - Flask后端
知识库 + LLM润色
"""
import json
import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import re
from datetime import datetime

# 本地文件存储路径
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'feedback_data.json')


def load_feedback_from_file():
    """从本地文件加载反馈数据"""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_feedback_to_file(feedback_list):
    """保存反馈到本地文件"""
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存反馈到文件失败: {e}")

# 加载本地反馈数据
FEEDBACK_HISTORY = load_feedback_from_file()

# 使用OpenAI SDK (兼容SiliconFlow API)
try:
    import openai
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# SiliconFlow API配置 (内网)
API_KEY = os.environ.get('OPENAI_API_KEY', 'sk-Di7MBmyNZk84lez25X0OnrSVFSSODH9LQecVNoVjH9ZXKrcG')
BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://aiapi.chaitin.net/v1')
MODEL = 'qwen2.5-72b-instruct'
MODEL_VISION = 'qwen-vl-plus'  # 视觉模型
EMBEDDING_MODEL = 'bge-m3'  # embedding模型

# 缓存配置
QUERY_CACHE = {}  # 问题缓存：{问题: {answer, timestamp, llm_used}}
CACHE_TTL = 3600  # 缓存有效期（秒）
MAX_CACHE_SIZE = 100  # 最大缓存数量

# 对话上下文配置
CONVERSATION_HISTORY = {}  # 会话历史：{session_id: [{role, content, timestamp}]}
CHAT_HISTORY = []  # 聊天记录历史：[{user, bot, time}]
MAX_HISTORY_LENGTH = 6  # 每个会话保留的历史消息数
DEFAULT_SESSION_ID = 'default'  # 默认会话ID

# 配置OpenAI客户端
client = None
if OPENAI_SDK_AVAILABLE and API_KEY:
    try:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception as e:
        print(f"客户端初始化失败: {e}")

# 知识库embedding缓存
KNOWLEDGE_EMBEDDINGS = {}

# 文件路径
KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), 'knowledge.json')

# 加载知识库
def load_knowledge():
    """加载知识库"""
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

KNOWLEDGE = load_knowledge()

# 相似度阈值
SIMILARITY_THRESHOLD = 0.8

# ============= Agent 配置 =============
MAX_AGENT_STEPS = 5

AGENT_SYSTEM_PROMPT = """你是雷池WAF智能助手。

判断用户问题类型：
- 知识类（产品怎么用、怎么配置、报错怎么办、功能说明、是什么意思等）：必须调用search_knowledge工具搜索知识库，基于搜索结果回答。
- 操作类（查看日志、查看站点、添加规则、修改配置等）：使用WAF相关工具。不需要搜索知识库。
- 无法判断时：先调用search_knowledge。

绝对禁止：
- 不调用search_knowledge就直接回答知识类问题
- 不调用任何工具就直接说"超出能力范围"——你必须先调用search_knowledge搜索后，确认知识库确实没有相关内容，才能说超出能力范围。

重要规则：
1. 严格基于知识库和工具返回的数据回答，绝不编造信息
2. 禁止使用自身知识回答任何问题，只能基于知识库搜索结果或WAF工具返回的数据来回答
3. 只有当search_knowledge返回结果为空（total为0或results为空）时，才能回复："此问题超出我的能力范围，请联系群里的相关技术支持人员。"
3. 如果知识库有相关内容，必须基于知识库结果回答，不要用自己的知识补充
4. 如果工具调用失败，告知用户失败原因
5. 回答要简洁、专业、友好
6. 用中文回答
7. 当用户的问题需要调用WAF接口时，如果WAF工具返回"未配置"或调用失败提示缺少配置，必须回复："此问题需要调用WAF接口，请在左侧配置栏中填写WAF API地址和API Token后重试。"
8. 时间范围参数规则：
   - 只有当用户明确提到时间范围（如"今天"、"昨天"、"本周"、"最近3天"等）时，才传begin_time/end_time参数
   - 用户说"所有日志"、"全部日志"或不提时间时，不要传begin_time/end_time参数，让接口返回默认数据
   - 绝对不要把系统消息中的"今天0点时间戳"和"当前时间戳"当作默认值使用
9. 自定义规则(/open/policy)和IP组(/open/ipgroup)是完全不同的两个接口，绝对不能混淆：
   - 自定义规则(policy)：有action属性，0=放行(白名单)，1=拦截(黑名单)，2=验证码，3=认证防护。用户提到黑白名单、拦截规则、放行规则时，必须用policy相关工具。
   - IP组(ipgroup)：只是IP地址的集合，没有拦截或放行功能。用户提到IP组、IP集合时，必须用ipgroup相关工具。
   - 用户说"添加黑名单/白名单"指的是添加规则，必须用create_custom_rule，不要用add_ip_to_group。
   - 用户说"添加IP组"指的是创建IP地址集合，必须用create_ip_group，不要用create_custom_rule。
10. 当用户想要执行创建、更新等操作时，如果用户没有提供完整的参数信息，不要自己猜测或编造参数值，而是告诉用户需要提供哪些参数、每个参数的含义和可选值。例如用户说"添加一个用户"，你应该回复"创建用户需要提供以下信息：username(用户名)、password(密码)、role(角色：1=管理员，2=操作员，3=配置员，4=审计员)、tfa_enabled(是否启用二次认证)"，等用户提供完整信息后再调用工具。
11. **数据忠实性规则（最重要）**：
   - 工具返回什么数据就汇报什么数据，绝对禁止编造工具未返回的数字或信息
   - 例如：API返回access=3就只能说3次访问，绝不能编造成12,345次
   - API没有返回"平均响应时间"就绝不能编造"200毫秒"
   - API没有返回"正常/异常访问量"就绝不能编造这些数据
   - 只呈现API实际返回的字段和值，不要添加、推测或美化数据
   - 如果工具返回的数据为空或字段很少，就如实说"当前没有数据"或只展示返回的字段，绝对不要为了回答看起来"完整"而自己填补数据
   - 禁止对数字做任何计算、汇总、估算，除非工具本身返回了汇总结果。例如不要把多条日志的次数加起来说"共XX次攻击"
   - 禁止生成工具未返回的图表描述、趋势分析、占比分析等
12. **回答方式规则**：
   - 绝对禁止在回答中提及工具名称、调用参数、函数名、请求路径等技术细节，用户不需要知道这些
   - 绝对禁止输出类似"我调用了xxx工具"、"参数为xxx"、"请求了/open/records接口"等内容
   - 直接用自然语言给出结论，例如：用户问"查看攻击日志"，你应该回答"共发现XX条攻击，主要类型为XX，主要来源IP为XX"
   - 禁止把工具返回的原始数据、日志、记录原样复述，必须提炼要点后简洁呈现
   - 工具返回了数据就必须如实呈现，即使数据量很小也要汇报。只有工具确实返回空结果（total=0、空列表、无任何字段）时才能说"当前没有相关数据"
   - 数据稀疏时突出有值的部分，例如"今日大部分时段无访问，14:00有3次访问"
"""


def _get_date_prompt():
    """生成包含当前日期的提示片段"""
    now = datetime.now()
    today_start = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    today_end = int(now.timestamp())
    return f"\n\n当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，今天0点时间戳：{today_start}，当前时间戳：{today_end}。"


def get_embedding(text):
    """获取文本的embedding向量"""
    global client
    if not client:
        if OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except:
                return None

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding获取失败: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    if not vec1 or not vec2:
        return 0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# ============= 缓存机制 =============

def normalize_query(query):
    """标准化查询文本，用于缓存key"""
    # 去除多余空格，统一大小写
    return ' '.join(query.lower().split())

def get_from_cache(query):
    """从缓存获取答案"""
    normalized = normalize_query(query)
    if normalized in QUERY_CACHE:
        cached = QUERY_CACHE[normalized]
        import time
        if time.time() - cached['timestamp'] < CACHE_TTL:
            print(f"缓存命中: {query[:30]}...")
            return cached
        else:
            # 缓存过期，删除
            del QUERY_CACHE[normalized]
    return None

def save_to_cache(query, answer_chunks, llm_used=True):
    """保存答案到缓存"""
    global QUERY_CACHE
    normalized = normalize_query(query)
    import time

    # 如果缓存已满，删除最旧的
    if len(QUERY_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = min(QUERY_CACHE.keys(), key=lambda k: QUERY_CACHE[k]['timestamp'])
        del QUERY_CACHE[oldest_key]

    QUERY_CACHE[normalized] = {
        'answer_chunks': list(answer_chunks),  # 转换为列表以便缓存
        'timestamp': time.time(),
        'llm_used': llm_used
    }
    print(f"已缓存查询: {query[:30]}...")

def clear_cache():
    """清空缓存"""
    global QUERY_CACHE
    QUERY_CACHE = {}
    print("缓存已清空")

# ============= 对话上下文 =============

def add_to_conversation(session_id, role, content):
    """添加消息到对话历史"""
    global CONVERSATION_HISTORY
    if session_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[session_id] = []

    import time
    CONVERSATION_HISTORY[session_id].append({
        'role': role,
        'content': content,
        'timestamp': time.time()
    })

    # 限制历史长度
    if len(CONVERSATION_HISTORY[session_id]) > MAX_HISTORY_LENGTH:
        CONVERSATION_HISTORY[session_id] = CONVERSATION_HISTORY[session_id][-MAX_HISTORY_LENGTH:]

def get_conversation_context(session_id=DEFAULT_SESSION_ID, max_turns=4):
    """获取对话上下文，用于LLM"""
    if session_id not in CONVERSATION_HISTORY:
        return []

    history = CONVERSATION_HISTORY[session_id]
    # 返回最近 max_turns 轮对话
    return history[-max_turns * 2:] if len(history) > 2 else []

def clear_conversation(session_id=DEFAULT_SESSION_ID):
    """清空指定会话的历史"""
    global CONVERSATION_HISTORY
    if session_id in CONVERSATION_HISTORY:
        del CONVERSATION_HISTORY[session_id]
        print(f"会话 {session_id} 历史已清空")


def get_embeddings_for_knowledge():
    """预计算知识库所有问题的embedding（增量更新）"""
    global KNOWLEDGE_EMBEDDINGS
    print("开始计算知识库embedding...")
    count = 0
    for i, item in enumerate(KNOWLEDGE):
        if i not in KNOWLEDGE_EMBEDDINGS:
            emb = get_embedding(item['问题描述'])
            if emb:
                KNOWLEDGE_EMBEDDINGS[i] = emb
                count += 1
    print(f"知识库embedding计算完成，新增 {count} 条，共 {len(KNOWLEDGE_EMBEDDINGS)} 条")


def update_knowledge_embedding():
    """更新知识库embedding（新增或变更后调用）"""
    global KNOWLEDGE_EMBEDDINGS
    print("更新知识库embedding...")
    # 重新计算全部
    KNOWLEDGE_EMBEDDINGS = {}
    for i, item in enumerate(KNOWLEDGE):
        emb = get_embedding(item['问题描述'])
        if emb:
            KNOWLEDGE_EMBEDDINGS[i] = emb
    print(f"知识库embedding更新完成，共 {len(KNOWLEDGE_EMBEDDINGS)} 条")


# ============= Agent 核心逻辑 =============

def _ensure_client():
    """确保OpenAI客户端可用"""
    global client
    if not client and OPENAI_SDK_AVAILABLE and API_KEY:
        try:
            client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        except Exception as e:
            print(f"客户端初始化失败: {e}")
    return client is not None


def _execute_tool_call(func_name, func_args, waf_agent=None):
    """执行单个工具调用，返回结果dict"""
    from waf_agent import execute_search_knowledge

    if func_name == "search_knowledge":
        return execute_search_knowledge(
            query=func_args.get("query", ""),
            knowledge_data=KNOWLEDGE,
            knowledge_embeddings=KNOWLEDGE_EMBEDDINGS,
            max_results=func_args.get("max_results", 5),
            get_embedding_fn=get_embedding,
            cosine_similarity_fn=cosine_similarity
        )
    elif waf_agent:
        return waf_agent.execute_tool(func_name, func_args)
    else:
        return {"success": False, "error": "WAF工具未配置，请提供WAF API地址和令牌"}


def _tool_name_to_display(func_name):
    """将工具名转为中文显示名"""
    name_map = {
        "search_knowledge": "搜索知识库",
        # 日志查询
        "query_attack_logs": "查询攻击日志",
        "get_attack_log_detail": "查询日志详情",
        "query_attack_events": "查询攻击事件",
        "query_qps": "查询QPS",
        "get_rule_attack_logs": "查询规则攻击日志",
        "export_attack_logs": "导出攻击日志",
        # 规则管理
        "list_custom_rules": "查询自定义规则",
        "create_custom_rule": "创建自定义规则",
        "toggle_rule": "切换规则状态",
        "delete_rule": "删除规则",
        "get_rule_detail": "查询规则详情",
        "update_rule": "更新规则",
        "order_rules": "规则排序",
        # IP组管理
        "list_ip_groups": "查询IP组",
        "add_ip_to_group": "添加IP到组",
        "create_ip_group": "创建IP组",
        "get_ip_group_detail": "查询IP组详情",
        "update_ip_group": "更新IP组",
        "delete_ip_group": "删除IP组",
        "get_crawler_group": "查询蜘蛛组",
        "update_crawler_group": "更新蜘蛛组",
        "get_ip_group_by_link": "通过链接获取IP",
        "create_ip_group_by_link": "通过链接创建IP组",
        # 站点管理
        "list_sites": "查询站点列表",
        "get_site_detail": "查询站点详情",
        "health_check": "执行健康检查",
        "set_site_health_check": "设置健康检查开关",
        "set_site_mode": "设置站点模式",
        "create_site": "创建站点",
        "delete_site": "删除站点",
        "update_site_basic_info": "更新站点信息",
        "update_site_group": "更新站点分组",
        "get_site_proxy": "查询代理配置",
        "set_site_proxy": "设置代理配置",
        "get_site_nginx_config": "查询Nginx配置",
        "set_site_nginx_config": "设置Nginx配置",
        "get_site_resources": "查询路由资源",
        "manage_site_excludes": "管理路由排除",
        "get_site_logs": "查询站点日志",
        "manage_site_groups": "管理站点分组",
        # 防护配置
        "set_challenge": "设置人机验证",
        "set_auth_defense": "设置身份认证",
        "set_rate_limit": "设置频率限制",
        "set_acl_enabled": "开关CC防护",
        "get_acl_logs": "查询频率限制日志",
        "release_acl_block": "解除IP封禁",
        "manage_rate_limit": "管理频率限制",
        "get_challenge_config": "查询人机验证配置",
        "set_challenge_config": "设置人机验证配置",
        "manage_anti_tamper": "管理反篡改",
        "manage_dynamic_defense": "管理动态防护",
        # 语义规则
        "get_global_semantics": "查询全局语义",
        "set_global_semantics": "设置全局语义",
        "manage_site_semantics": "管理站点语义",
        # 增强规则
        "get_enhance_rules": "查询增强规则",
        "update_enhance_rules": "更新增强规则",
        "manage_enhance_rule_switch": "增强规则开关",
        # 统计
        "get_advance_stats": "查询高级统计",
        "get_advance_trends": "查询趋势数据",
        # 告警
        "get_alarm_config": "查询告警配置",
        "update_alarm_config": "更新告警配置",
        # 认证防护
        "manage_auth_sources": "管理认证源",
        "list_auth_source_users": "查询认证源用户",
        "manage_auth_defense_users": "管理认证防护用户",
        "get_auth_defense_logs": "查询认证防护日志",
        # 系统信息
        "get_system_info": "查询系统信息",
        "get_license": "查询授权信息",
        "manage_license": "管理授权",
        "manage_detector": "管理检测引擎",
        "manage_log_clean": "管理日志清理",
        "manage_api_token": "管理API Token",
        "manage_global_proxy": "管理全局代理",
        "manage_syslog": "管理Syslog",
        "manage_ja4": "管理JA4配置",
        "manage_network_proxy": "管理网络代理",
        # 安全态势
        "manage_security_posture": "管理安全态势",
        # 等候室
        "manage_waiting_room": "管理等候室",
        # 门户
        "manage_portal": "管理门户配置",
        # 证书管理
        "list_certs": "查询证书列表",
        "get_cert_detail": "查询证书详情",
        "manage_certs": "管理证书",
        # 用户管理
        "list_users": "查询控制台用户",
        "manage_users": "管理控制台用户",
        # 转发规则
        "manage_forwarding_rules": "管理转发规则",
        # 云策略
        "manage_cloud_policies": "管理云策略",
        # 拦截页面
        "get_blocking_message": "查询拦截页面",
        # 恶意IP情报
        "manage_intelligence": "管理IP情报",
        # 审计日志
        "get_audit_log": "查询审计日志",
        # 报告
        "manage_report": "管理报告",
        # 人机验证日志
        "get_anti_bot_logs": "查询人机验证日志",
    }
    return name_map.get(func_name, func_name)


def _format_waf_results(tool_calls, query):
    """将WAF工具调用结果格式化为简洁汇总，供LLM分析用"""
    import time as _time

    parts = []

    for tc in tool_calls:
        name = tc['name']
        result = tc['result']
        display_name = tc['display_name']
        success = result.get('success', False)

        # 错误情况
        if not success:
            error = result.get('error', '未知错误')
            if 'permission denied' in str(error).lower() or '权限不足' in str(error):
                parts.append(f"{display_name}：权限不足")
            else:
                parts.append(f"{display_name}：失败 - {error}")
            continue

        # 写操作直接返回成功
        if name in WRITE_TOOLS:
            parts.append(f"{display_name}：操作成功")
            continue

        # 提取数据 - 自动查找result中第一个非空非元数据的值
        # 跳过 success/total/stat_type/trend_type 等元数据键
        skip_keys = {'success', 'total', 'stat_type', 'trend_type', 'error', 'tool_name'}
        data = {}
        for k, v in result.items():
            if k not in skip_keys and v:
                data = v
                break
        total = result.get('total', 0)

        # 将数据统一为列表
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 尝试从字典中提取列表
            items = data.get('data', data.get('logs', data.get('rules',
                     data.get('sites', data.get('ipgroups', [])))))
            if isinstance(items, dict):
                items = [items]
            if not total:
                total = data.get('total', len(items) if isinstance(items, list) else 1)
        else:
            items = []

        if not isinstance(items, list):
            items = [items] if items else []

        if not total and items:
            total = len(items)

        # === 所有查询结果统一为简洁汇总 ===

        # 趋势类：只给首尾和关键数字
        if name == 'get_advance_trends':
            trend_type = result.get('trend_type', '')
            type_names = {'access': '访问', 'intercept': '拦截'}
            title = type_names.get(trend_type, trend_type)
            if isinstance(items, list) and items:
                counts = [item.get('count', 0) for item in items if isinstance(item, dict)]
                first_t = items[0].get('time', 0) if isinstance(items[0], dict) else 0
                last_t = items[-1].get('time', 0) if isinstance(items[-1], dict) else 0
                t_start = _time.strftime('%H:%M', _time.localtime(first_t)) if first_t else '?'
                t_end = _time.strftime('%H:%M', _time.localtime(last_t)) if last_t else '?'
                parts.append(f"{title}趋势（{t_start}~{t_end}）：共{len(counts)}个数据点，最大{max(counts) if counts else 0}，最小{min(counts) if counts else 0}，合计{sum(counts)}")
            else:
                parts.append(f"{title}趋势：无数据")
            continue

        # 统计类：只给字段和值
        if name == 'get_advance_stats':
            stat_type = result.get('stat_type', '')
            type_names = {'access': '访问', 'attack': '攻击', 'client': '客户端',
                          'domain': '域名', 'location': '地理位置', 'page': '页面',
                          'status_code': '状态码', 'error_status_code': '错误状态码'}
            title = type_names.get(stat_type, stat_type)
            if isinstance(data, dict):
                summary = '、'.join(f'{k}={v}' for k, v in list(data.items())[:15])
                parts.append(f"{title}统计：{summary}")
            else:
                parts.append(f"{title}统计：{data}")
            continue

        # 列表/日志类：只给总数和关键维度汇总
        if items:
            summaries = [f"共{total}条"]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # 每条记录提取关键字段摘要
                parts_list = []
                for key, val in item.items():
                    if val is None or val == '' or val == []:
                        continue
                    if isinstance(val, list):
                        val = ', '.join(str(v) for v in val[:3])
                    elif isinstance(val, dict):
                        continue  # 跳过嵌套对象
                    parts_list.append(f"{key}={val}")
                if parts_list:
                    summaries.append('；'.join(parts_list))
            parts.append(f"{display_name}：" + ' | '.join(summaries))
        elif isinstance(data, dict) and data:
            # 详情类：只给关键字段
            def _fmt_val(v):
                if isinstance(v, list):
                    return ', '.join(str(x) for x in v)
                if isinstance(v, dict):
                    return '...'
                return str(v)
            summary = '、'.join(f'{k}={_fmt_val(v)}' for k, v in list(data.items())[:20] if v is not None and v != '' and v != [])
            parts.append(f"{display_name}：{summary}")
        else:
            parts.append(f"{display_name}：无数据")

    return '\n\n'.join(parts) if parts else "操作完成，但未获取到数据。"


def _format_record(item, index):
    """格式化单条记录为简洁文本"""
    if not isinstance(item, dict):
        return f"{index}. {item}"
    # 优先显示关键字段
    priority_fields = ['name', 'title', 'ip', 'src_ip', 'dst_ip', 'domain',
                       'server_names', 'url', 'method', 'attack_type', 'status',
                       'action', 'username', 'role', 'comment', 'description',
                       'created_at', 'updated_at', 'id', 'site_id', 'rule_id',
                       'event_id', 'trigger_count', 'pass_count']
    parts = []
    for field in priority_fields:
        if field in item and item[field]:
            val = item[field]
            if isinstance(val, list):
                val = ', '.join(str(v) for v in val[:3])
            parts.append(f"{field}={val}")
    if not parts:
        # 没有优先字段，显示前3个字段
        for k, v in list(item.items())[:3]:
            parts.append(f"{k}={v}")
    return f"{index}. {' | '.join(parts)}"


# 写操作工具集合（模块级别，供多处使用）
WRITE_TOOLS = {
    'create_custom_rule', 'toggle_rule', 'delete_rule', 'update_rule', 'order_rules',
    'add_ip_to_group', 'create_ip_group', 'update_ip_group', 'delete_ip_group',
    'update_crawler_group', 'create_ip_group_by_link',
    'set_site_mode', 'set_site_health_check', 'create_site', 'delete_site',
    'update_site_basic_info', 'update_site_group', 'set_site_proxy',
    'set_site_nginx_config', 'manage_site_excludes', 'manage_site_groups',
    'set_challenge', 'set_auth_defense', 'set_rate_limit', 'release_acl_block',
    'manage_rate_limit', 'set_challenge_config', 'manage_anti_tamper',
    'manage_dynamic_defense',
    'set_global_semantics', 'manage_site_semantics',
    'update_enhance_rules', 'manage_enhance_rule_switch',
    'update_alarm_config',
    'manage_auth_sources', 'manage_auth_defense_users',
    'manage_license', 'manage_detector', 'manage_log_clean',
    'manage_api_token', 'manage_global_proxy', 'manage_syslog', 'manage_ja4',
    'manage_network_proxy', 'manage_security_posture', 'manage_waiting_room',
    'manage_portal', 'manage_users', 'manage_certs',
    'manage_forwarding_rules', 'manage_cloud_policies',
    'get_blocking_message', 'manage_intelligence', 'manage_report'
}


def agent_chat_stream(query, session_id, waf_base_url=None, waf_api_token=None):
    """
    Agent循环 - 流式SSE生成器
    LLM自主决定调用工具，支持多步推理
    """
    # 写操作工具列表，调用过这些工具的查询不缓存结果
    has_write_op = False
    has_waf_tool_called = False  # 跟踪是否调用了WAF工具
    has_any_tool_called = False  # 跟踪是否调用了任何工具（包括search_knowledge）
    from waf_agent import WAFAgent, search_knowledge_tool_schema

    if not _ensure_client():
        yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，AI服务暂时不可用。'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"
        return

    # 准备工具schema
    tools_schema = [search_knowledge_tool_schema()]
    waf_agent = None

    if waf_base_url and waf_api_token:
        try:
            waf_agent = WAFAgent(waf_base_url, waf_api_token, verify_ssl=False)
            tools_schema.extend(waf_agent.get_tools_schema())
        except Exception as e:
            print(f"WAF Agent初始化失败: {e}")

    # 构建消息
    conv_history = get_conversation_context(session_id)
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT + _get_date_prompt()}]
    for msg in conv_history:
        messages.append({"role": msg['role'], "content": msg['content']})
    messages.append({"role": "user", "content": query})

    # Agent循环
    for step in range(MAX_AGENT_STEPS):
        try:
            # 非流式调用，让LLM决定是否调用工具
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.3,
                stream=False
            )
        except Exception as e:
            print(f"LLM调用失败(step {step}): {e}")
            yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，AI服务调用失败，请稍后重试。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"
            return

        message = response.choices[0].message

        if hasattr(message, 'tool_calls') and message.tool_calls:
            # 将助手消息（含tool_calls）加入历史
            msg_dict = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            }
            messages.append(msg_dict)

            # 执行每个工具调用
            called_waf_tools = []  # 收集WAF工具调用及结果
            called_search_knowledge = False
            knowledge_result = None

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except Exception:
                    func_args = {}

                # 发送thinking事件，让用户看到Agent在做什么
                display_name = _tool_name_to_display(func_name)
                yield f"data: {json.dumps({'type': 'thinking', 'content': f'正在{display_name}...'})}\n\n"
                print(f"Agent调用工具: {func_name}({func_args})")

                # 执行工具（捕获异常，避免生成器崩溃导致前端卡死）
                try:
                    result = _execute_tool_call(func_name, func_args, waf_agent)
                except Exception as e:
                    print(f"工具执行异常: {func_name}, 错误: {e}")
                    result = {'success': False, 'error': f'工具执行异常: {str(e)}'}

                # 标记是否有写操作
                if func_name in WRITE_TOOLS:
                    has_write_op = True

                # 区分知识库搜索和WAF工具
                if func_name == 'search_knowledge':
                    called_search_knowledge = True
                    knowledge_result = result
                else:
                    has_waf_tool_called = True
                    called_waf_tools.append({
                        'name': func_name,
                        'display_name': display_name,
                        'args': func_args,
                        'result': result
                    })
                has_any_tool_called = True

                # 将工具结果加入消息
                # WAF工具结果：先用_format_waf_results格式化为可读文本，再传给LLM分析
                if func_name != 'search_knowledge' and func_name in [t['name'] for t in called_waf_tools]:
                    # WAF工具：用格式化文本替代原始JSON，让LLM拿到可读的输入
                    try:
                        formatted = _format_waf_results([{
                            'name': func_name,
                            'display_name': display_name,
                            'args': func_args,
                            'result': result
                        }], query)
                        result_str = formatted
                    except Exception:
                        result_str = json.dumps(result, ensure_ascii=False)
                else:
                    # 知识库搜索：保留原始JSON
                    try:
                        result_str = json.dumps(result, ensure_ascii=False)
                    except Exception:
                        result_str = str(result)

                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "...(结果已截断)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

            # 工具结果已加入消息，继续Agent循环让LLM决定是否还需调工具或直接回答

        else:
            # LLM返回最终文本回答 - 检查是否应该先搜索知识库
            text_content = message.content or ""
            print(f"[Agent] LLM返回文本回复，content长度={len(text_content)}", flush=True)

            # 如果LLM有WAF配置但没调用任何工具，提示它使用工具重试（仅重试1次）
            if waf_agent and not has_any_tool_called and step == 0:
                print(f"[Agent] LLM有WAF工具但未调用任何工具，提示重试", flush=True)
                messages.append({"role": "assistant", "content": text_content})
                messages.append({"role": "user", "content": "请注意：你有WAF工具可用，请调用合适的WAF工具来获取数据回答我的问题，不要自己编造信息。"})
                continue

            # 如果LLM没调用任何工具就回复了"超出能力范围"，强制补调search_knowledge
            if '超出' in text_content and '能力范围' in text_content:
                print(f"[Agent] LLM未搜索知识库直接回复超出能力范围，强制补调search_knowledge", flush=True)
                from waf_agent import execute_search_knowledge
                yield f"data: {json.dumps({'type': 'thinking', 'content': '正在搜索知识库...'})}\n\n"
                kb_result = execute_search_knowledge(
                    query=query,
                    knowledge_data=KNOWLEDGE,
                    knowledge_embeddings=KNOWLEDGE_EMBEDDINGS,
                    max_results=5,
                    get_embedding_fn=get_embedding,
                    cosine_similarity_fn=cosine_similarity
                )
                if kb_result.get('total', 0) > 0:
                    # 知识库有结果，让LLM基于结果回答
                    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "auto_search", "type": "function", "function": {"name": "search_knowledge", "arguments": json.dumps({"query": query})}}]})
                    messages.append({"role": "tool", "tool_call_id": "auto_search", "content": json.dumps(kb_result, ensure_ascii=False)})
                    # 重新调用LLM
                    try:
                        stream = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            max_tokens=2000,
                            temperature=0.3,
                            stream=True
                        )
                        answer_chunks = []
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                answer_chunks.append(content)
                                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                        bot_response = ''.join(answer_chunks)
                        if bot_response:
                            add_to_conversation(session_id, 'assistant', bot_response)
                        yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"
                    except Exception as e:
                        print(f"补调知识库后LLM调用失败: {e}")
                        yield f"data: {json.dumps({'type': 'chunk', 'content': text_content})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"
                    return
                # 知识库确实没结果，输出原回复

            # 流式输出
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.3,
                    stream=True
                )

                answer_chunks = []
                finish_reason = None
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        answer_chunks.append(content)
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                # 如果因token限制被截断，添加提示
                if finish_reason == 'length':
                    truncation_notice = "\n\n（回答因长度限制被截断，请继续提问获取更多信息）"
                    answer_chunks.append(truncation_notice)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': truncation_notice})}\n\n"

                # 保存到对话历史和缓存
                bot_response = ''.join(answer_chunks)
                if bot_response:
                    add_to_conversation(session_id, 'assistant', bot_response)
                    # 缓存策略：
                    # 1. 写操作不缓存，避免重复执行
                    # 2. 没有调用任何工具时不缓存（LLM可能编造了数据）
                    # 3. 有WAF配置但没调用WAF工具时不缓存（LLM可能跳过了应调用的工具）
                    # 4. 只调用了search_knowledge的知识类问题可以缓存
                    should_cache = False
                    if not has_write_op:
                        if not has_any_tool_called:
                            print(f"[缓存跳过] 未调用任何工具，不缓存此结果", flush=True)
                        elif waf_base_url and waf_api_token and not has_waf_tool_called:
                            print(f"[缓存跳过] WAF配置可用但未调用WAF工具，不缓存此结果", flush=True)
                        else:
                            should_cache = True
                    if should_cache:
                        save_to_cache(query, answer_chunks, llm_used=True)

                yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"
            except Exception as e:
                print(f"流式输出失败: {e}")
                # 使用非流式的回答作为回退
                content = message.content or "抱歉，生成回答时出现错误。"
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
                add_to_conversation(session_id, 'assistant', content)
                yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"
            return

    # 超过最大步数
    yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，处理您的问题需要过多步骤，请尝试更具体地描述需求。'})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"


def agent_chat_sync(query, session_id='dingtalk', waf_base_url=None, waf_api_token=None):
    """Agent循环 - 同步版本，用于钉钉等非流式场景"""
    from waf_agent import WAFAgent, search_knowledge_tool_schema

    if not _ensure_client():
        return None

    tools_schema = [search_knowledge_tool_schema()]
    waf_agent = None

    if waf_base_url and waf_api_token:
        try:
            waf_agent = WAFAgent(waf_base_url, waf_api_token, verify_ssl=False)
            tools_schema.extend(waf_agent.get_tools_schema())
        except Exception as e:
            print(f"WAF Agent初始化失败: {e}")

    conv_history = get_conversation_context(session_id)
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for msg in conv_history:
        messages.append({"role": msg['role'], "content": msg['content']})
    messages.append({"role": "user", "content": query})

    for step in range(MAX_AGENT_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                max_tokens=2000,
                temperature=0.3,
                stream=False
            )
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None

        message = response.choices[0].message

        if hasattr(message, 'tool_calls') and message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except Exception:
                    func_args = {}

                result = _execute_tool_call(func_name, func_args, waf_agent)
                result_str = json.dumps(result, ensure_ascii=False)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "...(结果已截断)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })
            continue
        else:
            content = message.content
            if content:
                add_to_conversation(session_id, 'assistant', content)
            return content

    return None

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/feedback')
def feedback():
    """渲染反馈管理页面"""
    return render_template('feedback.html')

@app.route('/api/knowledge/add', methods=['POST'])
def add_to_knowledge():
    """添加或更新知识库条目"""
    data = request.get_json()
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    knowledge_id = data.get('knowledgeId', '')  # 获取knowledgeId，如果有则更新已有记录

    if not question or not answer:
        return jsonify({'success': False, 'message': '问题和答案不能为空'}), 400

    # 解析knowledgeId，格式为 kb_x
    target_index = None
    if knowledge_id and knowledge_id.startswith('kb_'):
        try:
            target_index = int(knowledge_id.split('_')[1])
            # 检查索引是否有效
            if target_index >= len(KNOWLEDGE):
                target_index = None
        except:
            target_index = None

    # 如果有有效的knowledgeId，则更新已有记录；否则添加新记录
    if target_index is not None:
        # 更新已有记录
        KNOWLEDGE[target_index]['问题描述'] = question
        KNOWLEDGE[target_index]['问题处理结果'] = answer
        new_index = target_index
        action = "更新"
    else:
        # 添加新记录
        new_index = len(KNOWLEDGE)
        KNOWLEDGE.append({
            '问题描述': question,
            '问题处理结果': answer
        })
        action = "添加"

    # 保存到文件
    try:
        with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(KNOWLEDGE, f, ensure_ascii=False, indent=2)

        # 自动更新embedding缓存
        emb = get_embedding(question)
        if emb:
            global KNOWLEDGE_EMBEDDINGS
            KNOWLEDGE_EMBEDDINGS[new_index] = emb
            print(f"知识库条目{action}成功，embedding已缓存，共 {len(KNOWLEDGE_EMBEDDINGS)} 条")

        # 清空相关缓存
        clear_cache()

        # 返回knowledgeId给前端
        knowledge_id = f"kb_{new_index}"
        return jsonify({'success': True, 'message': f'{action}成功', 'knowledgeId': knowledge_id})
    except Exception as e:
        # 如果是新增操作，回滚内存中的数据
        if target_index is None and len(KNOWLEDGE) > new_index:
            KNOWLEDGE.pop()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """问答接口 - Agent模式，LLM自主决定调用工具"""
    data = request.get_json()
    query = data.get('message', '').strip()
    session_id = data.get('session_id', DEFAULT_SESSION_ID)
    waf_base_url = data.get('waf_base_url', '').strip()
    waf_api_token = data.get('waf_api_token', '').strip()

    # 规范化WAF地址：确保以/api结尾
    if waf_base_url and not waf_base_url.rstrip('/').endswith('/api'):
        waf_base_url = waf_base_url.rstrip('/') + '/api'

    if not query:
        return jsonify({'error': '问题不能为空'}), 400

    # 保存用户问题到对话历史
    add_to_conversation(session_id, 'user', query)

    print(f"查询: {query}", flush=True)
    print(f"WAF配置: base_url={waf_base_url}, token_len={len(waf_api_token)}", flush=True)

    # 如果有WAF配置，清除旧缓存（避免命中之前无配置时的错误回答）
    if waf_base_url and waf_api_token:
        normalized = normalize_query(query)
        if normalized in QUERY_CACHE:
            print(f"清除旧缓存（WAF配置已更新）", flush=True)
            del QUERY_CACHE[normalized]

    # 检查缓存
    cached = get_from_cache(query)
    if cached:
        def generate_from_cache():
            yield "data: {\"type\":\"start\"}\n\n"
            for chunk in cached['answer_chunks']:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': cached['llm_used']})}\n\n"
        return Response(generate_from_cache(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })

    # Agent循环流式输出
    def generate_agent():
        yield "data: {\"type\":\"start\"}\n\n"
        for event in agent_chat_stream(query, session_id, waf_base_url, waf_api_token):
            yield event

    return Response(generate_agent(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'knowledge_count': len(KNOWLEDGE),
        'cache_size': len(QUERY_CACHE),
        'embedding_cache_size': len(KNOWLEDGE_EMBEDDINGS)
    })

@app.route('/api/waf/test', methods=['POST'])
def test_waf_connection():
    """验证WAF连接和API Token是否有效"""
    data = request.get_json(force=True)
    waf_base_url = data.get('waf_base_url', '').strip()
    waf_api_token = data.get('waf_api_token', '').strip()

    if not waf_base_url or not waf_api_token:
        return jsonify({'connected': False, 'error': '请填写WAF地址和API Token'})

    if not waf_base_url.rstrip('/').endswith('/api'):
        waf_base_url = waf_base_url.rstrip('/') + '/api'

    try:
        from waf_tools import WAFClient
        client = WAFClient(waf_base_url, waf_api_token, verify_ssl=False)
        result = client.get('/open/site', params={'page': 1, 'page_size': 1})
        return jsonify({'connected': True, 'sites_total': result.get('data', {}).get('total', 0)})
    except Exception as e:
        err_msg = str(e)
        if 'invalid-permission' in err_msg or 'Token' in err_msg:
            return jsonify({'connected': False, 'error': 'API Token无效或无权限，请在雷池管理界面「系统设置→通用配置→API接口」中确认Token'})
        elif 'login-required' in err_msg:
            return jsonify({'connected': False, 'error': 'API Token无效，需要重新生成'})
        else:
            return jsonify({'connected': False, 'error': f'连接失败: {err_msg}'})

@app.route('/api/cache/clear', methods=['POST'])
def clear_query_cache():
    """清空查询缓存"""
    clear_cache()
    return jsonify({'status': 'ok', 'message': '缓存已清空'})

@app.route('/api/conversation/clear', methods=['POST'])
def clear_api_conversation():
    """清空对话历史（通过API）"""
    data = request.get_json()
    session_id = data.get('session_id', DEFAULT_SESSION_ID)
    clear_conversation(session_id)
    return jsonify({'status': 'ok', 'message': f'会话 {session_id} 历史已清空'})

@app.route('/api/conversation', methods=['GET'])
def get_api_conversation():
    """获取对话历史"""
    data = request.get_json() or {}
    session_id = data.get('session_id', DEFAULT_SESSION_ID)
    history = get_conversation_context(session_id, max_turns=10)
    return jsonify({
        'session_id': session_id,
        'history': history
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取会话历史"""
    return jsonify({
        'history': CHAT_HISTORY
    })

@app.route('/api/history', methods=['POST'])
def save_chat():
    """保存单条聊天记录"""
    global CHAT_HISTORY
    data = request.get_json()
    user_message = data.get('user', '').strip()
    bot_message = data.get('bot', '').strip()

    if user_message and bot_message:
        CHAT_HISTORY.append({
            'user': user_message,
            'bot': bot_message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        # 只保留最近50条
        if len(CHAT_HISTORY) > 50:
            CHAT_HISTORY[:] = CHAT_HISTORY[-50:]

    return jsonify({'status': 'ok'})

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """清空会话历史"""
    global CHAT_HISTORY
    CHAT_HISTORY = []
    return jsonify({'status': 'ok'})

# 用户反馈存储
FEEDBACK_HISTORY = []

@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    """保存用户反馈"""
    data = request.get_json()
    user_message = data.get('user', '').strip()
    bot_message = data.get('bot', '').strip()
    feedback_type = data.get('type', '')  # like 或 dislike

    if user_message and bot_message and feedback_type:
        feedback_item = {
            'user': user_message,
            'bot': bot_message,
            'type': feedback_type,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 添加到内存
        FEEDBACK_HISTORY.append(feedback_item)
        # 只保留最近100条
        if len(FEEDBACK_HISTORY) > 100:
            FEEDBACK_HISTORY[:] = FEEDBACK_HISTORY[-100:]

        # 持久化到本地文件
        save_feedback_to_file(FEEDBACK_HISTORY)

    return jsonify({'status': 'ok'})

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """获取用户反馈历史"""
    # 从本地文件获取数据
    feedback_list = [
        {**f, 'id': f"local_{i}"}
        for i, f in enumerate(load_feedback_from_file())
    ]

    # 按时间倒序
    feedback_list.sort(key=lambda x: x['time'], reverse=True)
    return jsonify({
        'feedback': feedback_list
    })

@app.route('/api/feedback/<feedback_id>', methods=['PUT'])
def update_feedback(feedback_id):
    """更新反馈内容"""
    data = request.get_json()
    user_message = data.get('user', '').strip()
    bot_message = data.get('bot', '').strip()
    knowledge_id = data.get('knowledgeId', '')  # 获取关联的knowledgeId

    if not user_message or not bot_message:
        return jsonify({'success': False, 'message': '用户问题和机器人回复不能为空'}), 400

    try:
        # 解析ID格式: local_0
        if '_' in feedback_id:
            table_type, db_id = feedback_id.split('_', 1)

            if table_type == 'local':
                # 更新本地文件
                index = int(db_id)
                if 0 <= index < len(FEEDBACK_HISTORY):
                    FEEDBACK_HISTORY[index]['user'] = user_message
                    FEEDBACK_HISTORY[index]['bot'] = bot_message
                    # 如果有关联knowledgeId，添加标记
                    if knowledge_id:
                        FEEDBACK_HISTORY[index]['knowledgeId'] = knowledge_id
                        FEEDBACK_HISTORY[index]['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_feedback_to_file(FEEDBACK_HISTORY)
                    return jsonify({'success': True, 'message': '更新成功'})
        return jsonify({'success': False, 'message': '找不到对应的反馈'}), 404
    except Exception as e:
        print(f"更新反馈失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/feedback/<feedback_id>', methods=['DELETE'])
def delete_feedback(feedback_id):
    """删除反馈"""
    try:
        # 解析ID格式: local_0
        if '_' in feedback_id:
            table_type, db_id = feedback_id.split('_', 1)

            if table_type == 'local':
                # 从本地文件删除
                index = int(db_id)
                if 0 <= index < len(FEEDBACK_HISTORY):
                    del FEEDBACK_HISTORY[index]
                    save_feedback_to_file(FEEDBACK_HISTORY)
                    return jsonify({'success': True, 'message': '删除成功'})
        return jsonify({'success': False, 'message': '找不到对应的反馈'}), 404
    except Exception as e:
        print(f"删除反馈失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/feedback/search', methods=['GET'])
def search_feedback():
    """搜索反馈"""
    keyword = request.args.get('keyword', '').strip()
    feedback_type = request.args.get('type', '')  # like, dislike, 或空表示全部

    if not keyword:
        return jsonify({'feedback': [], 'message': '搜索关键词不能为空'}), 400

    # 从本地文件搜索
    feedback_list = [
        {**f, 'id': f"local_{i}"}
        for i, f in enumerate(FEEDBACK_HISTORY)
        if keyword in f.get('user', '') or keyword in f.get('bot', '')
    ]
    if feedback_type:
        feedback_list = [f for f in feedback_list if f.get('type') == feedback_type]

    # 按时间倒序
    feedback_list.sort(key=lambda x: x['time'], reverse=True)
    return jsonify({'feedback': feedback_list})

@app.route('/api/feedback/manual', methods=['POST'])
def add_feedback_manual():
    """手动添加反馈"""
    data = request.get_json()
    user_message = data.get('user', '').strip()
    bot_message = data.get('bot', '').strip()
    feedback_type = data.get('type', 'dislike')  # like 或 dislike

    if not user_message or not bot_message:
        return jsonify({'success': False, 'message': '用户问题和机器人回复不能为空'}), 400

    if feedback_type not in ['like', 'dislike']:
        feedback_type = 'dislike'

    feedback_item = {
        'user': user_message,
        'bot': bot_message,
        'type': feedback_type,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 添加到内存
    FEEDBACK_HISTORY.append(feedback_item)
    if len(FEEDBACK_HISTORY) > 100:
        FEEDBACK_HISTORY[:] = FEEDBACK_HISTORY[-100:]

    # 持久化到本地文件
    save_feedback_to_file(FEEDBACK_HISTORY)

    return jsonify({'success': True, 'message': '添加成功', 'data': feedback_item})

# 钉钉机器人配置
DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=c3f5f59702be9aa1237d1ce50e857823ffd66d85c661741215195eab47ea5509"
DINGTALK_SECRET = "SEC50c4d86aad4ae7d650a86b787a4374ae88aca9fc07e33b586627c94fe6993c84"

def generate_dingtalk_sign(secret):
    """生成钉钉加签签名"""
    import hmac
    import hashlib
    import base64
    import time

    timestamp = str(round(time.time() * 1000))
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return timestamp, sign

@app.route('/api/dingtalk', methods=['POST'])
def dingtalk_bot():
    """钉钉机器人接口"""
    import requests

    data = request.get_json()

    # 解析钉钉消息
    session_id = data.get('sessionId')  # 群消息有sessionId
    sender_nick = data.get('senderNick', '用户')
    text = data.get('text', {}).get('content', '').strip()

    if not text:
        return jsonify({'error': '消息为空'}), 400

    # 去掉@机器人的部分
    if '@' in text:
        text = text.split('@')[0].strip()
    if not text:
        return jsonify({'error': '消息为空'}), 400

    print(f"钉钉机器人收到消息: {text}")

    # 使用Agent同步接口获取答案
    answer = agent_chat_sync(text, session_id=f'dingtalk_{session_id or "default"}')

    if not answer:
        answer = '抱歉，我没有找到与您问题相关的答案。'

    # 生成签名
    timestamp, sign = generate_dingtalk_sign(DINGTALK_SECRET)

    # 发送消息到钉钉
    webhook_with_sign = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"
    msg_data = {
        'msgtype': 'markdown',
        'markdown': {
            'title': '雷池问答',
            'text': f"### 🛡️ 雷池WAF智能客服\n\n> {answer}\n\n---"
        }
    }
    if session_id:
        msg_data['at'] = {'atMobiles': [], 'isAtAll': False}

    try:
        requests.post(webhook_with_sign, json=msg_data, timeout=10)
    except Exception as e:
        print(f"发送钉钉消息失败: {e}")

    return jsonify({'status': 'ok'})


@app.route('/api/ocr', methods=['POST'])
def ocr_image():
    """图片文字识别接口"""
    data = request.get_json()
    image_data = data.get('image', '')

    if not image_data:
        return jsonify({'error': '图片不能为空'}), 400

    # 提取base64数据
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    try:
        # 使用视觉模型识别图片中的文字
        user_message_content = [
            {"type": "text", "text": "请仔细识别这张图片中的所有文字内容，包括英文、中文、代码、错误信息等。只输出图片中的文字，不要做其他解释。如果图片中没有文字，请回复「没有识别到文字」。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]

        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {'role': 'system', 'content': '你是文字识别助手，擅长准确识别图片中的文字。'},
                {'role': 'user', 'content': user_message_content}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()
        print(f"OCR结果: {result[:100]}...")

        return jsonify({'text': result})

    except Exception as e:
        print(f"OCR失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat_with_image', methods=['POST'])
def chat_with_image():
    """图片识别问答接口 - Agent模式"""
    data = request.get_json()
    query = data.get('message', '').strip()
    image_data = data.get('image', '')
    ocr_text = data.get('ocrText', '')
    session_id = data.get('session_id', DEFAULT_SESSION_ID)

    if not image_data:
        return jsonify({'error': '图片不能为空'}), 400

    print(f"图片识别查询: {query}")

    # 提取base64数据
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    try:
        # 第一步：视觉模型识别图片
        vision_prompt = "请仔细分析这张图片，描述图片中的内容。如果图片中有文字、错误信息、配置界面、日志等内容，请详细描述。\n"
        if ocr_text:
            vision_prompt += f"\n图片中识别到的文字：{ocr_text}\n"
        if query:
            vision_prompt += f"用户问题：{query}"

        user_message_content = [
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]

        vision_response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {'role': 'system', 'content': '你是雷池WAF技术支持专家，擅长分析图片和解决技术问题。'},
                {'role': 'user', 'content': user_message_content}
            ],
            max_tokens=500,
            temperature=0.7
        )

        vision_result = vision_response.choices[0].message.content.strip()
        print(f"视觉识别结果: {vision_result[:100]}...")

        # 第二步：将识别结果作为用户问题，通过Agent回答
        combined_query = f"{query}\n\n【图片识别结果】：{vision_result}" if query else f"请根据图片内容回答：\n\n{vision_result}"

        # 保存用户消息到对话历史
        add_to_conversation(session_id, 'user', query or '图片提问')

        def generate_agent_response():
            yield "data: {\"type\":\"start\"}\n\n"
            for event in agent_chat_stream(combined_query, session_id):
                yield event

        return Response(generate_agent_response(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        print(f"图片识别失败: {e}")
        import traceback
        traceback.print_exc()

        def generate_error():
            yield "data: {\"type\":\"start\"}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，图片识别失败，请稍后重试。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"

        return Response(generate_error(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })


if __name__ == '__main__':
    import os
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
    print(f"知识库已加载，共 {len(KNOWLEDGE)} 条问答")
    # 启动时计算所有知识库的embedding
    update_knowledge_embedding()
    print(f"服务启动中... 端口: {FLASK_PORT}")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)

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
SIMILARITY_THRESHOLD = 0.8  # 高匹配度阈值，超过此值需要LLM整合多个答案


def extract_keywords_with_llm(query):
    """用LLM提取问题的关键字段/关键词"""
    global client
    if not client:
        if OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except:
                return None

    try:
        prompt = f"""请从以下用户问题中提取关键检索词（核心关键词），用于在知识库中检索相关问答。
要求：
1. 提取2-5个最核心的关键词
2. 关键词应该是问题的主体内容，去掉修饰词
3. 只输出关键词，用空格分隔，不要有其他内容

用户问题：{query}

关键词："""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': '你是一个关键词提取助手。'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        keywords = response.choices[0].message.content.strip()
        print(f"LLM提取关键词: {keywords}")
        return keywords
    except Exception as e:
        print(f"LLM提取关键词失败: {e}")
        return None

def preprocess_text(text):
    """文本预处理"""
    if not text:
        return ""
    # 转小写，去除标点，分词
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return set(words)

def get_keywords(text):
    """提取关键词"""
    words = preprocess_text(text)
    # 过滤掉常见停用词
    stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '吗', '呢', '吧', '啊', '吗', '嘛', '呀', '哦', '嗯', '哎', '咦', '唉', '喂', '嘿', '哈', '哇', '哟', '呦', '呃', '噢', '嗯'}
    return words - stop_words

def calculate_similarity(query, item):
    """计算问题与知识库条目的相似度"""
    query_keywords = get_keywords(query)
    question_keywords = get_keywords(item['问题描述'])

    if not query_keywords or not question_keywords:
        return 0

    # 计算交集
    intersection = query_keywords & question_keywords
    # 计算Jaccard相似度
    union = query_keywords | question_keywords
    similarity = len(intersection) / len(union) if union else 0

    # 额外加权：如果查询词在问题描述中出现
    question_lower = item['问题描述'].lower()
    query_lower = query.lower()

    # 检查是否包含查询的关键部分
    query_parts = query_lower.replace('如何', ' ').replace('怎么', ' ').replace('怎样', ' ').replace('配置', ' ').split()
    for part in query_parts:
        if len(part) >= 2 and part in question_lower:
            similarity += 0.3

    # 模糊匹配：查询词是否部分匹配问题
    for word in query_keywords:
        if len(word) >= 2:
            for qword in question_keywords:
                if word in qword or qword in word:
                    similarity += 0.2

    return similarity


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

def format_conversation_history(history):
    """格式化对话历史为LLM可读格式"""
    if not history:
        return "（无历史对话）"

    formatted = []
    for msg in history:
        role_name = "用户" if msg['role'] == 'user' else "助手"
        formatted.append(f"{role_name}：{msg['content']}")
    return "\n".join(formatted)

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


def find_by_embedding(query, top_k=20):
    """用embedding查找最匹配的知识库条目"""
    # 获取查询的embedding
    query_emb = get_embedding(query)
    if not query_emb:
        return []

    # 计算与所有知识库问题的相似度
    scores = []
    for i, item in enumerate(KNOWLEDGE):
        if i in KNOWLEDGE_EMBEDDINGS:
            sim = cosine_similarity(query_emb, KNOWLEDGE_EMBEDDINGS[i])
            if sim > 0:
                scores.append((sim, item))

    # 排序返回top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def find_best_match(query, top_k=20):
    """查找最匹配的知识库条目"""
    scores = []
    for item in KNOWLEDGE:
        score = calculate_similarity(query, item)
        if score > 0:
            scores.append((score, item))

    # 排序并返回top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def build_context_from_knowledge(query, matches):
    """从知识库构建上下文"""
    context_parts = []

    # 添加系统提示
    context_parts.append("""你是一个友好的雷池WAF智能客服助手。你的任务是根据知识库中的信息，用委婉的方式回答用户的问题。

回答要求：
1. 根据找到的参考资料回答，不要编造信息
2. 回答要友好、委婉、有礼貌
3. 如果用户问题与知识库中的问题相似，可以参考知识库中的处理结果
4. 如果没有找到相关信息，礼貌地告知用户并建议其他获取帮助的方式""")

    context_parts.append("\n\n以下是知识库中的相关问答：\n\n")

    for i, (score, match) in enumerate(matches[:3], 1):
        context_parts.append(f"参考{i}（相似度: {score:.2f}）：\n")
        context_parts.append(f"问题：{match['问题描述']}\n")
        context_parts.append(f"解答：{match['问题处理结果']}\n\n")

    context_parts.append(f"\n用户问题：{query}\n")
    context_parts.append("\n请根据以上知识库中的信息，用委婉友好的方式回答用户的问题。如果知识库中有相关解答，请参考并整理后回答。")

    return "".join(context_parts)

def get_llm_response(query, context):
    """调用LLM API润色回答 - 预留"""
    return None

def get_llm_response_stream_v2(query, context, matches):
    """调用LLM API - 先分析问题再用知识库检索最后生成答案"""
    global client

    if not client:
        if OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except Exception as e:
                print(f"客户端初始化失败: {e}")
                return None

    if not client:
        return None

    try:
        # 第一步：用LLM深度分析问题，理解问题本质
        analysis_prompt = f"""请分析以下用户问题，进行深度理解：

用户问题：{query}

请从以下角度分析：
1. 用户的核心问题/本质问题是什么？
2. 这个问题的本质关键词是什么（忽略修饰词）？
3. 用户可能的场景是什么（安装/配置/故障/授权/人机验证等）？

只输出分析结果，用以下格式：
【本质问题】：用户实际想问的核心问题（用简短的一句话描述）
【本质关键词】：去掉修饰词后的核心关键词（用于知识库检索）
【场景】：xxx

分析结果："""

        analysis_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': '你是雷池WAF技术支持专家，擅长理解用户问题的本质。'},
                {'role': 'user', 'content': analysis_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        analysis_result = analysis_response.choices[0].message.content.strip()
        print(f"LLM问题分析:\n{analysis_result}")

        # 提取本质关键词（从分析结果中提取）
        essential_keywords = ""
        for line in analysis_result.split('\n'):
            if '【本质关键词】' in line:
                essential_keywords = line.split('【本质关键词】')[1].strip()
                break

        if not essential_keywords:
            # 如果没提取到，用原始关键词
            essential_keywords = query

        print(f"本质关键词: {essential_keywords}")

        # 第二步：用本质关键词重新检索知识库
        keyword_matches = find_best_match(essential_keywords, top_k=10)

        # 如果本质关键词检索结果不好，尝试提取本质问题再次检索
        essential_question = ""
        for line in analysis_result.split('\n'):
            if '【本质问题】' in line:
                essential_question = line.split('【本质问题】')[1].strip()
                break

        essential_matches = []
        if essential_question and essential_question != query:
            essential_matches = find_best_match(essential_question, top_k=10)
            print(f"本质问题检索: {essential_question}, 匹配数: {len(essential_matches)}")

        # 合并原始检索结果和本质关键词检索结果，去重
        seen = set()
        merged_matches = []
        for score, item in matches + keyword_matches + essential_matches:
            if item['问题描述'] not in seen:
                seen.add(item['问题描述'])
                merged_matches.append((score, item))
        merged_matches.sort(key=lambda x: x[0], reverse=True)
        merged_matches = merged_matches[:5]

        print(f"合并后匹配数: {len(merged_matches)}")

        # 构建新的上下文
        new_context = build_context_from_knowledge(query, merged_matches)

        # 第三步：用检索结果生成答案（包含问题分析）
        prompt = f"""你是雷池WAF的高级技术支持客服。请根据以下内容回答用户问题。

【问题分析】
{analysis_result}

【知识库内容】
{new_context}

用户问题：{query}

请先简要说明你对问题的理解，然后根据知识库内容给出专业答案。"""

        # 流式调用
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': '你是雷池WAF的高级技术支持专家，回答问题专业、礼貌、简洁。'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            stream=True
        )

        # 流式yield
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"LLM API调用失败: {e}")
        return None


def polish_answer_with_llm(query, answer):
    """用LLM润色知识库答案"""
    global client
    if not client:
        if OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except:
                return None

    try:
        prompt = f"""你是雷池WAF技术支持客服。请严格基于以下知识库答案回复用户。

知识库答案：
{answer}

重要规则：
1. 只基于知识库答案进行润色，不要添加任何知识库中没有的信息
2. 保持语言友好、专业
3. 如果知识库答案不完整，也不要自己补充信息
4. 只润色表达方式，不改变答案内容

请直接输出润色后的答案。"""

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': '你是雷池WAF技术支持专家，严格基于知识库内容回答问题，不添加任何知识库以外的信息。'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=500,
            temperature=0.5,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"LLM润色失败: {e}")
        return None


def polite_response(answer, query):
    """将知识库答案转化为委婉的表达"""
    # 如果答案较短，直接返回
    if len(answer) < 50:
        return f"根据您的问题，我来为您解答：{answer}"

    # 根据问题类型添加不同的开场白
    opening = ""
    if any(kw in query for kw in ['如何', '怎么', '怎样', '方法', '操作']):
        opening = "关于这个问题，您可以参考以下方法：\n\n"
    elif any(kw in query for kw in ['可以', '能否', '是否', '有没有']):
        opening = "针对您的疑问，我的回答如下：\n\n"
    elif any(kw in query for kw in ['为什么', '原因', '是什么']):
        opening = "让我为您解释一下：\n\n"
    else:
        opening = "根据我的了解，"

    return opening + answer

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
    """问答接口 - 流式输出（优化版）"""
    data = request.get_json()
    query = data.get('message', '').strip()
    session_id = data.get('session_id', DEFAULT_SESSION_ID)

    if not query:
        return jsonify({'error': '问题不能为空'}), 400

    # 保存用户问题到对话历史
    add_to_conversation(session_id, 'user', query)

    # 第一步：检查缓存
    cached = get_from_cache(query)
    if cached:
        # 从缓存返回答案
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

    print(f"查询: {query}")

    # 第二步：LLM分析用户问题，提取关键信息
    print("正在分析用户问题...")
    try:
        global client
        if not client and OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except:
                pass

        analysis_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': '你是雷池WAF技术支持专家。请分析用户问题，提取关键信息和意图。'},
                {'role': 'user', 'content': f"""用户问题：{query}

请分析这个问题，提取以下信息：
1. 问题的核心意图（如：配置、报错、日志分析、规则设置等）
2. 关键词（3-5个最相关的技术术语）
3. 问题的模块/功能（如：WAF、CC防护、规则引擎等）

请以JSON格式返回分析结果：
{{
  "intent": "问题意图",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "module": "相关模块"
}}"""},
            ],
            max_tokens=200,
            temperature=0.3
        )
        analysis_result = json.loads(analysis_response.choices[0].message.content.strip())
        print(f"问题分析结果: {analysis_result}")
    except Exception as e:
        print(f"问题分析失败: {e}，使用原问题检索")
        analysis_result = {'keywords': [query], 'intent': 'general', 'module': 'general'}

    # 第三步：基于embedding向量检索知识库
    # 直接使用embedding向量匹配，比关键词匹配更准确
    print(f"使用embedding向量检索: {query}")
    all_matches = find_by_embedding(query, top_k=20)

    # 去重
    seen = {}
    merged_matches = []
    for score, match in all_matches:
        if match['问题描述'] not in seen:
            seen[match['问题描述']] = score
            merged_matches.append((score, match))

    # 按相似度排序
    merged_matches.sort(key=lambda x: x[0], reverse=True)
    merged_matches = merged_matches[:10]
    print(f"检索匹配: {len(merged_matches)} 条")

    # 第四步：获取对话上下文
    conv_history = get_conversation_context(session_id)
    conv_context = format_conversation_history(conv_history) if conv_history else ""

    # 第五步：检查高匹配度答案并构建知识库上下文
    high_matches = [(score, match) for score, match in merged_matches if score >= SIMILARITY_THRESHOLD]
    print(f"高匹配度答案（score>={SIMILARITY_THRESHOLD}）: {len(high_matches)} 条")

    if high_matches:
        # 有多个高匹配度答案，构建包含所有答案的上下文
        knowledge_context_parts = ["【知识库参考内容】\n\n以下是知识库中相似度较高的答案，请整合后回复：\n\n"]
        for i, (score, match) in enumerate(high_matches[:5], 1):
            knowledge_context_parts.append(f"答案{i}（相似度: {score:.2f}）：\n")
            knowledge_context_parts.append(f"问题：{match['问题描述']}\n")
            knowledge_context_parts.append(f"解答：{match['问题处理结果']}\n\n")
        knowledge_context = "".join(knowledge_context_parts)
    else:
        # 没有高匹配度答案
        knowledge_context_parts = ["【知识库参考内容】\n\n"]
        for i, (score, match) in enumerate(merged_matches[:3], 1):
            knowledge_context_parts.append(f"参考{i}（相似度: {score:.2f}）：\n")
            knowledge_context_parts.append(f"问题：{match['问题描述']}\n")
            knowledge_context_parts.append(f"解答：{match['问题处理结果']}\n\n")
        knowledge_context = "".join(knowledge_context_parts)

    # 第六步：LLM基于知识库生成答案

    # 如果没有匹配结果，返回固定话术
    if not merged_matches:
        def generate_no_match():
            yield "data: {\"type\":\"start\"}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': '抱歉，此问题超出我的能力范畴，请联系群里相关的技术人员。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"
        return Response(generate_no_match(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })

    # 有多个高匹配度答案，LLM整合多个答案
    if len(high_matches) >= 2:
        prompt = f"""你是雷池WAF技术支持客服。请整合以下多个知识库答案后回复用户。

{knowledge_context}

用户问题：{query}

问题分析结果：
- 意图：{analysis_result.get('intent', '未知')}
- 关键词：{', '.join(analysis_result.get('keywords', []))}

重要规则：
1. 只整合知识库中提供的答案，不要添加任何知识库以外的信息
2. 如果多个答案之间有冲突，请指出并提供最合适的建议
3. 整合后的答案应该全面、准确
4. 不要编造或假设任何知识库中没有的内容

请整合以上知识库答案，给出一个完整、准确的回复。"""
    else:
        # 单个答案或没有高匹配度，正常生成
        prompt = f"""你是雷池WAF技术支持客服。请严格按照以下知识库内容回答用户问题。

【知识库内容】
{knowledge_context}

用户问题：{query}

问题分析结果：
- 意图：{analysis_result.get('intent', '未知')}
- 关键词：{', '.join(analysis_result.get('keywords', []))}

重要规则：
1. 只使用知识库中提供的信息进行回答
2. 如果知识库内容可以回答问题，请直接给出答案
3. 如果知识库内容不完整或与问题不完全相关，请基于知识库内容尽力回答
4. 不要添加任何知识库以外的信息
5. 不要编造或假设任何知识库中没有的内容
6. 只使用知识库中的答案，不要使用外部知识

请基于知识库内容回答用户问题。"""

    def generate_optimized():
        yield "data: {\"type\":\"start\"}\n\n"

        global client
        if not client and OPENAI_SDK_AVAILABLE and API_KEY:
            try:
                client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except:
                pass

        if not client:
            # LLM不可用，直接返回最佳答案
            best_answer = high_matches[0][1] if high_matches else (merged_matches[0][1] if merged_matches else '抱歉，服务暂时不可用。')
            for chunk in best_answer['问题处理结果'] if isinstance(best_answer, dict) else best_answer:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"
            return

        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': '你是雷池WAF技术支持专家，严格基于知识库内容回答问题。如果知识库没有相关内容，必须如实告知用户，绝不编造信息。'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=600,
                temperature=0.5,
                stream=True
            )

            # 流式输出并缓存
            answer_chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    answer_chunks.append(content)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

            # 缓存答案
            save_to_cache(query, answer_chunks, llm_used=True)

            # 保存助手回复到对话历史
            bot_response = ''.join(answer_chunks)
            add_to_conversation(session_id, 'assistant', bot_response)

            yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"

        except Exception as e:
            print(f"LLM API调用失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回知识库答案作为后备
            fallback_match = high_matches[0][1] if high_matches else (merged_matches[0][1] if merged_matches else None)
            fallback_answer = fallback_match.get('问题处理结果', '抱歉，服务暂时不可用，请稍后重试。') if fallback_match else '抱歉，服务暂时不可用。'
            for chunk in fallback_answer:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"

    return Response(generate_optimized(), mimetype='text/event-stream', headers={
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

    # 调用chat接口获取答案
    matches = find_best_match(text)

    if not matches:
        answer = '抱歉，我没有找到与您问题相关的答案。'
    else:
        # 使用LLM生成答案
        best_score, best_match = matches[0]
        context = build_context_from_knowledge(text, matches)

        # 调用LLM获取答案
        llm_answer = get_llm_response_stream_v2(text, context, matches)

        if llm_answer:
            answer_chunks = []
            for chunk in llm_answer:
                if chunk:
                    answer_chunks.append(chunk)
            answer = ''.join(answer_chunks) if answer_chunks else best_match['问题处理结果']
        else:
            answer = best_match['问题处理结果']

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
    """图片识别问答接口 - 流式输出"""
    data = request.get_json()
    query = data.get('message', '').strip()
    image_data = data.get('image', '')
    ocr_text = data.get('ocrText', '')  # 获取OCR识别结果

    if not image_data:
        return jsonify({'error': '图片不能为空'}), 400

    print(f"图片识别查询: {query}")
    print(f"OCR识别结果: {ocr_text[:100] if ocr_text else '无'}...")

    # 提取base64数据
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    try:
        # 构建多模态消息
        vision_prompt = "你是一个专业的WAF技术支持助手。请仔细分析这张图片，描述图片中的内容。如果图片中有文字、错误信息、配置界面、日志等内容，请详细描述。\n"
        if ocr_text:
            vision_prompt += f"\n图片中识别到的文字：{ocr_text}\n"
        if query:
            vision_prompt += f"用户问题：{query}"

        user_message_content = [
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]

        # 调用视觉模型识别图片
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

        # 用识别结果检索知识库
        combined_query = f"{query} {vision_result}" if query else vision_result
        emb_matches = find_by_embedding(combined_query, top_k=10)
        kw_matches = find_best_match(combined_query)

        seen = set()
        merged_matches = []
        for score, item in emb_matches + kw_matches:
            if item['问题描述'] not in seen:
                seen.add(item['问题描述'])
                merged_matches.append((score, item))
        merged_matches.sort(key=lambda x: x[0], reverse=True)
        matches = merged_matches[:5]

        def generate_response():
            yield "data: {\"type\":\"start\"}\n\n"

            if matches:
                context_parts = ["根据图片识别结果和知识库，请回答用户问题：\n\n"]
                for i, (score, match) in enumerate(matches[:3], 1):
                    context_parts.append(f"参考{i}：\n问题：{match['问题描述']}\n解答：{match['问题处理结果']}\n\n")
                context = "".join(context_parts)
                prompt = f"""图片识别结果：{vision_result}

{context}

用户问题：{query if query else '请根据图片内容回答'}

重要规则：
1. 严格基于知识库内容回答
2. 如果知识库没有相关内容，明确说明"知识库中没有找到相关内容"
3. 不要添加任何知识库以外的信息
4. 只使用知识库中提供的信息

请基于知识库内容回答用户问题。"""
            else:
                prompt = f"图片识别结果：{vision_result}\n\n用户问题：{query if query else '请描述图片内容'}\n\n说明：知识库中没有找到相关内容，只能基于图片识别结果回复。"

            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': '你是雷池WAF技术支持专家，严格基于知识库内容回答问题。如果知识库没有相关内容，必须如实告知用户，绝不编造信息。'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=500,
                temperature=0.5,
                stream=True
            )

            has_content = False
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    has_content = True
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.choices[0].delta.content})}\n\n"

            if has_content:
                yield f"data: {json.dumps({'type': 'done', 'llm_used': True})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'chunk', 'content': f'根据图片识别结果：{vision_result}'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'llm_used': False})}\n\n"

        return Response(generate_response(), mimetype='text/event-stream', headers={
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
    print(f"知识库已加载，共 {len(KNOWLEDGE)} 条问答")
    # 启动时计算所有知识库的embedding
    update_knowledge_embedding()
    print("服务启动中...")
    app.run(host='0.0.0.0', port=5000, debug=True)

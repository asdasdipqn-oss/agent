#!/usr/bin/env python3
# 添加"已修改"标签到反馈列表
import re

with open('/Users/wangqizhi/agent2/templates/feedback.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 在时间后面添加"已修改"标签
pattern = r'(<span class="feedback-time">\$\{item\.time\}</span>)'
replacement = r'\1\n                        const isUpdated = item.knowledgeId ? \'<span class="badge badge-updated">已修改</span>\' : \'\';\n                        ${isUpdated}'
content = re.sub(pattern, replacement, content)

with open('/Users/wangqizhi/agent2/templates/feedback.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("已添加'已修改'标签")

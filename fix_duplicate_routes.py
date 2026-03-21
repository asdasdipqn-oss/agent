#!/usr/bin/env python3
"""删除错误的POST /api/feedback路由"""

import re

with open('/Users/wangqizhi/agent2/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找所有@app.route('/api/feedback'定义（包括错误的POST）
routes_to_remove = []
for match in re.finditer(r"@app\.route\('/api/feedback[^']*\)", content):
    routes_to_remove.append(match.group(0))

# 删除错误的POST路由
if routes_to_remove:
    print(f"发现 {len(routes_to_remove)} 个重复的'/api/feedback'路由，正在删除...")

    # 对于每个路由，删除该行到下一个路由
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检查是否是POST路由定义
        if "@app.route('/api/feedback'" in line and "methods=['POST']" in line:
            # 删除这行
            lines[i] = ''  # 标记为已删除
            print(f"  第 {i+1} 行: POST /api/feedback")
            i += 1  # 跳过已删除的行
        else:
            break

    # 写回修改后的内容
    with open('/Users/wangqizhi/agent2/app.py', 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    print(f"已删除 {len([ln for ln in lines if ln])} 行重复的'/api/feedback'路由")
else:
    print("没有找到重复的'/api/feedback'路由")

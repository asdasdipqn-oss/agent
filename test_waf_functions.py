#!/usr/bin/env python3
"""
测试WAF工具调用功能
"""
import os
import sys

# 设置环境变量（测试用，请替换为实际值）
os.environ['WAF_BASE_URL'] = os.environ.get('WAF_BASE_URL', 'https://wqzc-star3.pl.in.chaitin.net')
os.environ['WAF_API_TOKEN'] = os.environ.get('WAF_API_TOKEN', 'your-api-token-here')

# 导入函数进行测试
from app import (
    should_use_waf_tools,
    select_waf_tool_with_llm,
    execute_waf_tool_and_format_result,
    format_waf_tool_result
)

print("=" * 50)
print("  WAF工具调用功能单元测试")
print("=" * 50)

# 测试1: should_use_waf_tools
print("\n【测试1】should_use_waf_tools(query)")
print("-" * 40)

test_queries = [
    "查看当前QPS",
    "查询攻击日志",
    "查看站点列表",
    "你好"
]

for query in test_queries:
    print(f"\n查询: {query}")
    use_tool, reason = should_use_waf_tools(query)
    print(f"  use_tool={use_tool}")
    print(f"  reason={reason}")

# 测试2: select_waf_tool_with_llm（需要client）
print("\n【测试2】select_waf_tool_with_llm(query, tools_schema)")
print("-" * 40)

try:
    from waf_agent import WAFAgent
    agent = WAFAgent(os.environ['WAF_BASE_URL'], os.environ['WAF_API_TOKEN'])
    tools_schema = agent.get_tools_schema()

    query = "查看当前QPS"
    print(f"查询: {query}")
    tool_name, tool_params = select_waf_tool_with_llm(query, tools_schema)
    print(f"  tool_name={tool_name}")
    print(f"  tool_params={tool_params}")
except Exception as e:
    print(f"  错误: {e}")
    import traceback
    traceback.print_exc()

# 测试3: format_waf_tool_result
print("\n【测试3】format_waf_tool_result()")
print("-" * 40)

test_results = [
    {
        "tool_name": "query_qps",
        "result": {"success": True, "total_qps": 123}
    },
    {
        "tool_name": "query_attack_logs",
        "result": {
            "success": True,
            "total": 5,
            "logs": [
                {"timestamp": "2024-01-01 10:00", "src_ip": "1.2.3.4", "url_path": "/api/test", "attack_type": "SQL注入", "action": "拦截"}
            ]
        }
    },
    {
        "tool_name": "list_sites",
        "result": {
            "success": True,
            "total": 2,
            "sites": [
                {"id": 1, "title": "example.com", "server_names": ["www.example.com"], "is_enabled": True}
            ]
        }
    }
]

for test in test_results:
    print(f"\n工具: {test['tool_name']}")
    formatted = format_waf_tool_result(test['tool_name'], test['result'])
    print(f"  格式化结果:\n{formatted}")

print("\n" + "=" * 50)
print("  测试完成")
print("=" * 50)

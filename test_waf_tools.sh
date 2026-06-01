#!/bin/bash
# WAF工具调用功能测试脚本

echo "========================================="
echo "  WAF工具调用功能测试"
echo "========================================="
echo ""

# 测试API健康检查
echo "1. 检查服务状态..."
HEALTH=$(curl -s http://localhost:5000/api/health)
echo "$HEALTH" | python3 -m json.tool
echo ""

# 测试工具调用 - QPS查询
echo "2. 测试QPS查询（应触发WAF工具调用）..."
echo "----------------------------------------"
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查看当前QPS", "session_id": "test_qps"}' | \
  python3 -c "
import sys, json
for line in sys.stdin:
    if 'data:' in line:
        try:
            data = json.loads(line.split('data: ', 1)[1])
            if data.get('type') == 'tool_call':
                print('✅ 工具调用成功触发!')
                print(f\"   工具消息: {data.get('message', '')}\")
                if data.get('tool_result'):
                    print(f\"   工具结果: {data.get('tool_result', '')[:100]}...\")
            elif data.get('type') == 'done':
                print('✅ 回答完成')
                break
        except:
            pass
"
echo ""

# 测试工具调用 - 攻击日志查询
echo "3. 测试攻击日志查询..."
echo "----------------------------------------"
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查询最近5条攻击日志", "session_id": "test_logs"}' | \
  python3 -c "
import sys, json
for line in sys.stdin:
    if 'data:' in line:
        try:
            data = json.loads(line.split('data: ', 1)[1])
            if data.get('type') == 'tool_call':
                print('✅ 工具调用成功触发!')
                print(f\"   工具消息: {data.get('message', '')}\")
                if data.get('tool_result'):
                    print(f\"   工具结果: {data.get('tool_result', '')[:100]}...\")
            elif data.get('type') == 'done':
                print('✅ 回答完成')
                break
        except:
            pass
"
echo ""

# 测试工具调用 - 站点列表
echo "4. 测试站点列表查询..."
echo "----------------------------------------"
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查看所有站点", "session_id": "test_sites"}' | \
  python3 -c "
import sys, json
for line in sys.stdin:
    if 'data:' in line:
        try:
            data = json.loads(line.split('data: ', 1)[1])
            if data.get('type') == 'tool_call':
                print('✅ 工具调用成功触发!')
                print(f\"   工具消息: {data.get('message', '')}\")
                if data.get('tool_result'):
                    print(f\"   工具结果: {data.get('tool_result', '')[:100]}...\")
            elif data.get('type') == 'done':
                print('✅ 回答完成')
                break
        except:
            pass
"
echo ""

echo "========================================="
echo "  测试完成"
echo "========================================="
echo ""
echo "说明："
echo "- 如果看到'✅ 工具调用成功触发!'，说明WAF工具集成正常"
echo "- 如果没有看到工具调用，请检查WAF_API_TOKEN和WAF_BASE_URL配置"

#!/usr/bin/env python3
# 修改addToKnowledge函数，添加成功后更新反馈列表

with open('/Users/wangqizhi/agent2/templates/feedback.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到addToKnowledge函数并修改
old_add_to_knowledge = '''        // 添加到知识库
        async function addToKnowledge() {
            const question = document.getElementById('kb-question').value.trim();
            const answer = document.getElementById('kb-answer').value.trim();

            if (!question || !answer) {
                showToast('问题和答案不能为空', 'error');
                return;
            }

            try {
                const response = await fetch('/api/knowledge/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, answer })
                });
                const data = await response.json();

                if (data.success) {
                    showToast('添加成功！', 'success');
                    closeKnowledgeModal();
                } else {
                    showToast(data.message || '添加失败', 'error');
                }
            } catch (err) {
                console.error('添加失败:', err);
                showToast('添加失败，请重试', 'error');
            }
        }'''

new_add_to_knowledge = '''        // 添加到知识库
        async function addToKnowledge() {
            const question = document.getElementById('kb-question').value.trim();
            const answer = document.getElementById('kb-answer').value.trim();

            if (!question || !answer) {
                showToast('问题和答案不能为空', 'error');
                return;
            }

            try {
                const kbResponse = await fetch('/api/knowledge/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, answer })
                });
                const kbData = await kbResponse.json();

                if (kbData.success) {
                    showToast('添加成功！', 'success');
                    closeKnowledgeModal();

                    // 获取反馈ID并更新，添加knowledgeId标记
                    const feedbackId = document.getElementById('kb-feedback-id').value;
                    if (feedbackId) {
                        const updateResponse = await fetch(`/api/feedback/${feedbackId}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user: document.getElementById('kb-question').value,
                                bot: document.getElementById('kb-answer').value,
                                knowledgeId: kbData.knowledgeId
                            })
                        });
                        const updateData = await updateResponse.json();
                        if (updateData.success) {
                            // 重新加载反馈列表以显示"已修改"标签
                            loadFeedback();
                        }
                    }
                } else {
                    showToast(kbData.message || '添加失败', 'error');
                }
            } catch (err) {
                console.error('添加失败:', err);
                showToast('添加失败，请重试', 'error');
            }
        }'''

if old_add_to_knowledge in content:
    content = content.replace(old_add_to_knowledge, new_add_to_knowledge)
    with open('/Users/wangqizhi/agent2/templates/feedback.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("已更新addToKnowledge函数")
else:
    print("未找到目标代码")

        // 添加到知识库
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
        }

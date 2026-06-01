#!/usr/bin/env python3
"""
雷池WAF Agent调度层
基于LLM function calling实现工具调用，支持多步推理
"""
import json
import re
from typing import List, Dict, Any, Optional
from waf_tools import WAFClient, WAFTool


class WAFAgent:
    """雷池WAF Agent - 工具执行器"""

    def __init__(self, base_url: str, api_token: str, verify_ssl: bool = False):
        self.client = WAFClient(base_url, api_token, verify_ssl)
        self.tools: List[WAFTool] = []
        self._register_tools()

    def _register_tools(self):
        """注册所有WAF工具"""
        from waf_tools import (
            QueryAttackLogsTool, GetAttackLogDetailTool, QueryAttackEventsTool,
            QueryQpsTool, GetRuleAttackLogsTool, ExportAttackLogsTool,
            ListRulesTool, CreateRuleTool, ToggleRuleTool, DeleteRuleTool,
            GetRuleDetailTool, UpdateRuleTool, OrderRulesTool,
            ListIPGroupsTool, AddIPToGroupTool, CreateIPGroupTool,
            GetIPGroupDetailTool, UpdateIPGroupTool, DeleteIPGroupTool,
            GetCrawlerGroupTool, UpdateCrawlerGroupTool,
            GetIPGroupByLinkTool, CreateIPGroupByLinkTool,
            ListSitesTool, GetSiteDetailTool, HealthCheckTool,
            SetSiteHealthCheckTool, SetSiteModeTool,
            CreateSiteTool, DeleteSiteTool, UpdateSiteBasicInfoTool,
            UpdateSiteGroupTool, GetSiteProxyTool, SetSiteProxyTool,
            GetSiteNginxConfigTool, SetSiteNginxConfigTool,
            GetSiteResourcesTool, ManageSiteExcludesTool,
            GetSiteLogsTool, ManageSiteGroupsTool,
            SetChallengeTool, SetAuthDefenseTool, SetACLTool,
            SetACLEnableTool, GetACLLogsTool, ReleaseACLBlockTool,
            ManageRateLimitTool, GetChallengeConfigTool,
            SetChallengeConfigTool, ManageAntiTamperTool,
            ManageDynamicDefenseTool,
            GetGlobalSemanticsTool, SetGlobalSemanticsTool,
            ManageSiteSemanticsTool,
            GetEnhanceRulesTool, UpdateEnhanceRulesTool,
            ManageEnhanceRuleSwitchTool,
            GetAdvanceStatsTool, GetAdvanceTrendsTool,
            GetAlarmConfigTool, UpdateAlarmConfigTool,
            ManageAuthSourcesTool, ListAuthSourceUsersTool,
            ManageAuthDefenseUsersTool, GetAuthDefenseLogsTool,
            GetSystemInfoTool, GetLicenseTool, ManageLicenseTool,
            ManageDetectorTool, ManageLogCleanTool, ManageApiTokenTool,
            ManageGlobalProxyTool, ManageSyslogTool, ManageJA4Tool,
            ManageSecurityPostureTool, ManageWaitingRoomTool,
            ManagePortalTool, ManageUsersTool, ManageCertsTool,
            ManageForwardingRulesTool, ManageCloudPoliciesTool,
            GetBlockingMessageTool, ManageIntelligenceTool,
            GetAuditLogTool, ManageReportTool, GetAntiBotLogsTool,
            ManageNetworkProxyTool,
            ListCertsTool, GetCertDetailTool, ListUsersTool
        )

        self.tools = [
            # 日志查询
            QueryAttackLogsTool(self.client),
            GetAttackLogDetailTool(self.client),
            QueryAttackEventsTool(self.client),
            QueryQpsTool(self.client),
            GetRuleAttackLogsTool(self.client),
            ExportAttackLogsTool(self.client),
            # 规则管理
            ListRulesTool(self.client),
            CreateRuleTool(self.client),
            ToggleRuleTool(self.client),
            DeleteRuleTool(self.client),
            GetRuleDetailTool(self.client),
            UpdateRuleTool(self.client),
            OrderRulesTool(self.client),
            # IP组管理
            ListIPGroupsTool(self.client),
            AddIPToGroupTool(self.client),
            CreateIPGroupTool(self.client),
            GetIPGroupDetailTool(self.client),
            UpdateIPGroupTool(self.client),
            DeleteIPGroupTool(self.client),
            GetCrawlerGroupTool(self.client),
            UpdateCrawlerGroupTool(self.client),
            GetIPGroupByLinkTool(self.client),
            CreateIPGroupByLinkTool(self.client),
            # 站点管理
            ListSitesTool(self.client),
            GetSiteDetailTool(self.client),
            HealthCheckTool(self.client),
            SetSiteHealthCheckTool(self.client),
            SetSiteModeTool(self.client),
            CreateSiteTool(self.client),
            DeleteSiteTool(self.client),
            UpdateSiteBasicInfoTool(self.client),
            UpdateSiteGroupTool(self.client),
            GetSiteProxyTool(self.client),
            SetSiteProxyTool(self.client),
            GetSiteNginxConfigTool(self.client),
            SetSiteNginxConfigTool(self.client),
            GetSiteResourcesTool(self.client),
            ManageSiteExcludesTool(self.client),
            GetSiteLogsTool(self.client),
            ManageSiteGroupsTool(self.client),
            # 防护配置
            SetChallengeTool(self.client),
            SetAuthDefenseTool(self.client),
            SetACLTool(self.client),
            SetACLEnableTool(self.client),
            GetACLLogsTool(self.client),
            ReleaseACLBlockTool(self.client),
            ManageRateLimitTool(self.client),
            GetChallengeConfigTool(self.client),
            SetChallengeConfigTool(self.client),
            ManageAntiTamperTool(self.client),
            ManageDynamicDefenseTool(self.client),
            # 语义规则
            GetGlobalSemanticsTool(self.client),
            SetGlobalSemanticsTool(self.client),
            ManageSiteSemanticsTool(self.client),
            # 增强规则
            GetEnhanceRulesTool(self.client),
            UpdateEnhanceRulesTool(self.client),
            ManageEnhanceRuleSwitchTool(self.client),
            # 统计
            GetAdvanceStatsTool(self.client),
            GetAdvanceTrendsTool(self.client),
            # 告警
            GetAlarmConfigTool(self.client),
            UpdateAlarmConfigTool(self.client),
            # 认证防护
            ManageAuthSourcesTool(self.client),
            ListAuthSourceUsersTool(self.client),
            ManageAuthDefenseUsersTool(self.client),
            GetAuthDefenseLogsTool(self.client),
            # 系统信息
            GetSystemInfoTool(self.client),
            GetLicenseTool(self.client),
            ManageLicenseTool(self.client),
            ManageDetectorTool(self.client),
            ManageLogCleanTool(self.client),
            ManageApiTokenTool(self.client),
            ManageGlobalProxyTool(self.client),
            ManageSyslogTool(self.client),
            ManageJA4Tool(self.client),
            ManageNetworkProxyTool(self.client),
            # 安全态势
            ManageSecurityPostureTool(self.client),
            # 等候室
            ManageWaitingRoomTool(self.client),
            # 门户
            ManagePortalTool(self.client),
            # 证书管理
            ListCertsTool(self.client),
            GetCertDetailTool(self.client),
            ManageCertsTool(self.client),
            # 用户管理
            ListUsersTool(self.client),
            ManageUsersTool(self.client),
            # 转发规则
            ManageForwardingRulesTool(self.client),
            # 云策略
            ManageCloudPoliciesTool(self.client),
            # 拦截页面
            GetBlockingMessageTool(self.client),
            # 恶意IP情报
            ManageIntelligenceTool(self.client),
            # 审计日志
            GetAuditLogTool(self.client),
            # 报告
            ManageReportTool(self.client),
            # 反爬虫日志
            GetAntiBotLogsTool(self.client),
        ]

    def get_tools_schema(self) -> List[Dict]:
        """获取OpenAI function calling格式的工具Schema"""
        return [tool.to_dict() for tool in self.tools]

    def get_tool(self, tool_name: str) -> Optional[WAFTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def execute_tool(self, tool_name: str, arguments: dict) -> Dict[str, Any]:
        """执行指定工具，LLM通过function calling提供参数"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {'success': False, 'error': f'工具 {tool_name} 不存在'}

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return {'success': False, 'error': str(e), 'tool_name': tool_name}


def search_knowledge_tool_schema() -> Dict:
    """知识库搜索工具的OpenAI function calling schema"""
    return {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索雷池WAF知识库，获取产品使用、配置指导、故障排查、授权等文档信息。当用户提出关于WAF的配置、使用、故障等问题时应优先使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题描述"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回的最大结果数量，默认5"
                    }
                },
                "required": ["query"]
            }
        }
    }


def execute_search_knowledge(query: str, knowledge_data: list, knowledge_embeddings: dict,
                             max_results: int = 5,
                             get_embedding_fn=None, cosine_similarity_fn=None) -> Dict[str, Any]:
    """
    执行知识库搜索：关键词匹配 + embedding语义检索，合并去重

    Args:
        get_embedding_fn: 获取embedding向量的函数，避免循环导入
        cosine_similarity_fn: 计算余弦相似度的函数，避免循环导入
    """
    query_lower = query.lower()

    # === 关键词匹配 ===
    chinese_words = re.findall(r'[\u4e00-\u9fff]{1,}', query_lower)
    english_words = re.findall(r'[a-zA-Z]{2,}', query_lower)
    numbers = re.findall(r'\d+', query_lower)
    keywords = set(chinese_words + english_words + numbers)
    if not keywords:
        keywords = {query_lower}

    keyword_results = []
    for item in knowledge_data:
        problem_desc = item.get('问题描述', '').lower()
        score = sum(1 for kw in keywords if kw in problem_desc)
        if score > 0:
            keyword_results.append({'item': item, 'score': score})

    keyword_results.sort(key=lambda x: x['score'], reverse=True)

    # === Embedding语义检索 ===
    embedding_results = []
    if knowledge_embeddings and get_embedding_fn and cosine_similarity_fn:
        try:
            query_embedding = get_embedding_fn(query)
            if query_embedding:
                similarities = []
                for i, item in enumerate(knowledge_data):
                    if i in knowledge_embeddings:
                        sim = cosine_similarity_fn(query_embedding, knowledge_embeddings[i])
                        similarities.append({'item': item, 'score': sim})
                similarities.sort(key=lambda x: x['score'], reverse=True)
                # 取相似度 > 0.3 的结果
                embedding_results = [s for s in similarities if s['score'] > 0.3][:max_results]
        except Exception as e:
            print(f"Embedding搜索失败: {e}")

    # === 合并去重 ===
    seen_questions = set()
    merged = []

    # 先放关键词匹配结果（权重高）
    for r in keyword_results[:max_results]:
        q = r['item'].get('问题描述', '')
        if q not in seen_questions:
            seen_questions.add(q)
            merged.append(r['item'])

    # 再补充embedding结果
    for r in embedding_results:
        q = r['item'].get('问题描述', '')
        if q not in seen_questions:
            seen_questions.add(q)
            merged.append(r['item'])

    return {
        'success': True,
        'total': len(merged),
        'results': merged[:max_results]
    }


if __name__ == '__main__':
    agent = WAFAgent('http://localhost:5000/api', 'test-token')

    print("已注册WAF工具:")
    for tool in agent.get_tools_schema():
        func = tool['function']
        print(f"  - {func['name']}: {func['description']}")

    print("\n知识库搜索工具:")
    ktool = search_knowledge_tool_schema()
    print(f"  - {ktool['function']['name']}: {ktool['function']['description']}")

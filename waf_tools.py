#!/usr/bin/env python3
"""
雷池WAF Agent工具集
"""
import json
import requests
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
from datetime import datetime


class WAFClient:
    """雷池WAF API客户端"""

    def __init__(self, base_url: str, api_token: str, verify_ssl: bool = False):
        """
        初始化WAF客户端

        Args:
            base_url: WAF API 基础地址
            api_token: WAF API 访问令牌
            verify_ssl: 是否验证 SSL 证书，默认 False（内网环境通常使用自签名证书）
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'X-SLCE-API-Token': api_token,
            'Content-Type': 'application/json'
        })
        # 禁用 SSL 证书验证（适用于内网环境）
        self.session.verify = verify_ssl
        # 增加请求超时时间
        self.session.timeout = 30
        # 禁用代理，避免内网请求走本地代理导致SSL失败
        self.session.trust_env = False

    def _check_api_error(self, result: Dict):
        """统一检查WAF API返回的错误"""
        err_msg = result.get('err') or result.get('msg') or ''
        if err_msg and err_msg not in ('success',):
            if err_msg == 'invalid-permission':
                raise Exception('API Token无效或无权限，请在雷池管理界面「系统设置→通用配置→API接口」中生成正确的API Token')
            elif err_msg == 'login-required':
                raise Exception('API需要登录认证，请检查API Token是否正确')
            elif err_msg == 'permission denied':
                raise Exception('当前API Token权限不足，无法执行此操作')
            else:
                raise Exception(f'WAF API错误: {err_msg}')

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            print(f"[WAF GET] {url} params={params}", flush=True)
            print(f"[WAF GET] token_header: X-SLCE-API-Token={self.api_token[:8]}...", flush=True)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            print(f"[WAF GET RESP] {endpoint} => {json.dumps(result, ensure_ascii=False)[:500]}", flush=True)
            self._check_api_error(result)
            return result
        except requests.exceptions.HTTPError as e:
            # 尝试从响应体中获取详细错误信息
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                    elif isinstance(error_data, str):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                except:
                    pass
            print(f"GET请求失败: {error_detail}")
            raise Exception(error_detail)

    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """POST请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            print(f"[WAF POST] {url} data={json.dumps(data, ensure_ascii=False)[:300] if data else '{}'}", flush=True)
            print(f"[WAF POST] token_header: X-SLCE-API-Token={self.api_token[:8]}...", flush=True)
            response = self.session.post(url, json=data)
            print(f"[WAF POST RESP] {endpoint} => {json.dumps(response.json(), ensure_ascii=False)[:500]}", flush=True)
            response.raise_for_status()
            result = response.json()
            self._check_api_error(result)
            return result
        except requests.exceptions.HTTPError as e:
            # 尝试从响应体中获取详细错误信息
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                    elif isinstance(error_data, str):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                except:
                    pass
            print(f"POST请求失败: {error_detail}")
            raise Exception(error_detail)

    def put(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """PUT请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            print(f"[WAF PUT] {url} data={json.dumps(data, ensure_ascii=False)[:300] if data else '{}'}", flush=True)
            print(f"[WAF PUT] token_header: X-SLCE-API-Token={self.api_token[:8]}...", flush=True)
            response = self.session.put(url, json=data)
            print(f"[WAF PUT RESP] {endpoint} => {json.dumps(response.json(), ensure_ascii=False)[:500]}", flush=True)
            response.raise_for_status()
            result = response.json()
            self._check_api_error(result)
            return result
        except requests.exceptions.HTTPError as e:
            # 尝试从响应体中获取详细错误信息
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                    elif isinstance(error_data, str):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                except:
                    pass
            print(f"PUT请求失败: {error_detail}")
            raise Exception(error_detail)

    def delete(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """DELETE请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.delete(url, json=data)
            response.raise_for_status()
            result = response.json()
            self._check_api_error(result)
            return result
        except requests.exceptions.HTTPError as e:
            # 尝试从响应体中获取详细错误信息
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                    elif isinstance(error_data, str):
                        error_detail = f"HTTP {e.response.status_code}: {error_data}"
                except:
                    pass
            print(f"DELETE请求失败: {error_detail}")
            raise Exception(error_detail)


class WAFTool(ABC):
    """WAF工具基类"""

    def __init__(self, client: WAFClient):
        self.client = client

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Dict]:
        """工具参数定义"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        pass

    def to_dict(self) -> Dict:
        """转换为OpenAI function calling格式的工具定义"""
        # 将参数列表转换为JSON Schema格式
        param_list = self.parameters
        properties = {}
        required = []

        for param in param_list:
            prop = {"type": param["type"], "description": param["description"]}
            # 数组类型补充items
            if param["type"] == "array":
                prop["items"] = param.get("items", {"type": "string"})
            # 枚举类型补充enum
            if "enum" in param:
                prop["enum"] = param["enum"]
            properties[param["name"]] = prop
            if param.get("required", False):
                required.append(param["name"])

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# ==================== 日志查询工具 ====================

class QueryAttackLogsTool(WAFTool):
    """查询攻击日志工具"""

    @property
    def name(self) -> str:
        return "query_attack_logs"

    @property
    def description(self) -> str:
        return "查询WAF攻击日志，支持按IP、主机、URL、攻击类型等条件筛选"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "ip",
                "type": "string",
                "description": "源IP地址",
                "required": False
            },
            {
                "name": "host",
                "type": "string",
                "description": "主机名",
                "required": False
            },
            {
                "name": "url",
                "type": "string",
                "description": "URL路径",
                "required": False
            },
            {
                "name": "attack_type",
                "type": "string",
                "description": "攻击类型",
                "required": False
            },
            {
                "name": "action",
                "type": "string",
                "description": "操作类型",
                "enum": ["拦截", "放行"],
                "required": False
            },
            {
                "name": "start",
                "type": "integer",
                "description": "开始时间戳",
                "required": False
            },
            {
                "name": "end",
                "type": "integer",
                "description": "结束时间戳",
                "required": False
            },
            {
                "name": "page",
                "type": "integer",
                "description": "页码，默认1",
                "required": False
            },
            {
                "name": "page_size",
                "type": "integer",
                "description": "每页数量，最大100",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行查询攻击日志"""
        params = {}
        for key in ['ip', 'host', 'url', 'attack_type', 'action', 'start', 'page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        # 注意：WAF API的end参数有bug，传入end会导致返回空数据，因此不传end

        print(f"[query_attack_logs] 请求参数: {params}", flush=True)
        result = self.client.get('/open/records', params=params)

        # 格式化返回结果
        data = result.get('data', {})
        logs = data.get('data', [])
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                'id': log.get('id'),
                'timestamp': log.get('timestamp'),
                'src_ip': log.get('src_ip'),
                'host': log.get('host'),
                'url_path': log.get('url_path'),
                'method': log.get('method'),
                'attack_type': log.get('attack_type'),
                'action': '拦截' if log.get('action') == 1 else '放行',
                'rule_id': log.get('rule_id'),
                'risk_level': log.get('risk_level')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'logs': formatted_logs
        }


class GetAttackLogDetailTool(WAFTool):
    """获取攻击日志详情工具"""

    @property
    def name(self) -> str:
        return "get_attack_log_detail"

    @property
    def description(self) -> str:
        return "获取单条攻击日志的详细信息，包括请求头、响应头等"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "log_id",
                "type": "string",
                "description": "日志ID",
                "required": True
            }
        ]

    def execute(self, log_id: str, **kwargs) -> Dict[str, Any]:
        """执行获取攻击日志详情"""
        result = self.client.get(f'/open/record/{log_id}')

        log = result.get('data', {})
        return {
            'success': True,
            'log': {
                'id': log.get('id'),
                'timestamp': log.get('timestamp'),
                'src_ip': log.get('src_ip'),
                'dst_ip': log.get('dst_ip'),
                'host': log.get('host'),
                'method': log.get('method'),
                'url_path': log.get('url_path'),
                'query_string': log.get('query_string'),
                'status_code': log.get('status_code'),
                'req_header': log.get('req_header'),
                'req_body': log.get('req_body'),
                'rsp_header': log.get('rsp_header'),
                'rsp_body': log.get('rsp_body'),
                'reason': log.get('reason'),
                'attack_type': log.get('attack_type'),
                'risk_level': log.get('risk_level')
            }
        }


class QueryAttackEventsTool(WAFTool):
    """查询攻击事件工具"""

    @property
    def name(self) -> str:
        return "query_attack_events"

    @property
    def description(self) -> str:
        return "查询攻击事件（聚合后的攻击记录）"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "ip",
                "type": "string",
                "description": "源IP地址",
                "required": False
            },
            {
                "name": "host",
                "type": "string",
                "description": "主机名",
                "required": False
            },
            {
                "name": "port",
                "type": "string",
                "description": "端口",
                "required": False
            },
            {
                "name": "start",
                "type": "integer",
                "description": "开始时间戳",
                "required": False
            },
            {
                "name": "end",
                "type": "integer",
                "description": "结束时间戳",
                "required": False
            },
            {
                "name": "page",
                "type": "integer",
                "description": "页码",
                "required": False
            },
            {
                "name": "page_size",
                "type": "integer",
                "description": "每页数量",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行查询攻击事件"""
        params = {}
        for key in ['ip', 'host', 'port', 'start', 'page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        # 注意：WAF API的end参数有bug，传入end会导致返回空数据，因此不传end

        result = self.client.get('/open/events', params=params)

        events = result.get('data', {}).get('nodes', [])
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event.get('id'),
                'start_at': event.get('start_at'),
                'end_at': event.get('end_at'),
                'ip': event.get('ip'),
                'host': event.get('host'),
                'port': event.get('port'),
                'deny_count': event.get('deny_count'),
                'pass_count': event.get('pass_count')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'events': formatted_events
        }


class QueryQpsTool(WAFTool):
    """查询QPS工具"""

    @property
    def name(self) -> str:
        return "query_qps"

    @property
    def description(self) -> str:
        return "查询当前QPS（每秒请求数）统计信息"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行查询QPS"""
        result = self.client.get('/stat/qps')

        nodes = result.get('data', {}).get('nodes', [])
        total_qps = sum(node.get('qps', 0) for node in nodes)

        return {
            'success': True,
            'total_qps': total_qps,
            'nodes': nodes
        }


# ==================== 规则管理工具 ====================

class ListRulesTool(WAFTool):
    """列出自定义规则工具"""

    @property
    def name(self) -> str:
        return "list_custom_rules"

    @property
    def description(self) -> str:
        return "查询自定义规则(/open/policy)。黑白名单是规则的一种：白名单=action0(放行)，黑名单=action1(拦截)。用户提到黑白名单、拦截规则、放行规则时用此工具。IP组是另一个接口，不要用此工具查IP组。返回结果包含规则ID和名称，如需查看规则详情，请先用此工具获取规则ID，再用get_rule_detail查询。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "action",
                "type": "integer",
                "description": "操作类型筛选：0=放行(白名单)，1=拦截(黑名单)，2=验证码，3=认证防护",
                "enum": [0, 1, 2, 3],
                "required": False
            },
            {
                "name": "page",
                "type": "integer",
                "description": "页码",
                "required": False
            },
            {
                "name": "page_size",
                "type": "integer",
                "description": "每页数量",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出规则"""
        params = {}
        for key in ['action', 'page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        result = self.client.get('/open/policy', params=params)

        rules = result.get('data', {}).get('data', [])
        formatted_rules = []
        for rule in rules:
            formatted_rules.append({
                'id': rule.get('id'),
                'name': rule.get('name'),
                'action': rule.get('action'),
                'level': rule.get('level'),
                'is_enabled': rule.get('is_enabled'),
                'created_at': rule.get('created_at')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'rules': formatted_rules
        }


class CreateRuleTool(WAFTool):
    """创建自定义规则工具"""

    @property
    def name(self) -> str:
        return "create_custom_rule"

    @property
    def description(self) -> str:
        return "创建自定义规则(/open/policy)。黑白名单是规则的一种：白名单=action0(放行)，黑名单=action1(拦截)。用户要求添加黑白名单、拦截规则、放行规则时用此工具。IP组是另一个接口，创建IP组请用create_ip_group。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "name",
                "type": "string",
                "description": "规则名称",
                "required": True
            },
            {
                "name": "action",
                "type": "integer",
                "description": "操作类型：0=放行(白名单)，1=拦截(黑名单)，2=验证码，3=认证防护",
                "enum": [0, 1, 2, 3],
                "required": True
            },
            {
                "name": "level",
                "type": "integer",
                "description": "风险级别",
                "required": False
            },
            {
                "name": "pattern",
                "type": "array",
                "description": "规则匹配模式，每组是AND关系，组间是OR关系。每条模式包含k(字段名如src_ip/host/url等)、op(操作符如eq/not_eq/has/prefix/re)、v(值数组)。示例：[[{\"k\":\"src_ip\",\"op\":\"eq\",\"v\":[\"1.1.1.1\"]}]] 表示源IP等于1.1.1.1时匹配",
                "items": {"type": "array", "items": {"type": "object"}},
                "required": False
            },
            {
                "name": "is_enabled",
                "type": "boolean",
                "description": "是否启用",
                "required": False
            }
        ]

    def execute(self, name: str, action: int, **kwargs) -> Dict[str, Any]:
        """执行创建规则"""
        data = {
            "name": name,
            "action": action,
            "is_enabled": kwargs.get('is_enabled', True),
            "level": kwargs.get('level', 1),
            "log": kwargs.get('log', True),
            "pattern": kwargs.get('pattern', []),
            "insert_position": kwargs.get('insert_position', 'first')
        }

        result = self.client.post('/open/policy', data=data)

        action_names = {0: '放行(白名单)', 1: '拦截(黑名单)', 2: '验证码', 3: '认证防护'}
        return {
            'success': True,
            'rule_id': result.get('data'),
            'action_name': action_names.get(action, '未知'),
            'message': f'规则"{name}"已创建，操作：{action_names.get(action, "未知")}'
        }


class ToggleRuleTool(WAFTool):
    """启用/禁用规则工具"""

    @property
    def name(self) -> str:
        return "toggle_rule"

    @property
    def description(self) -> str:
        return "启用或禁用指定的自定义规则"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "rule_id",
                "type": "integer",
                "description": "规则ID",
                "required": True
            },
            {
                "name": "is_enabled",
                "type": "boolean",
                "description": "是否启用",
                "required": True
            }
        ]

    def execute(self, rule_id: int, is_enabled: bool, **kwargs) -> Dict[str, Any]:
        """执行切换规则状态"""
        data = {
            "id": rule_id,
            "is_enabled": is_enabled
        }

        self.client.put('/open/policy/switch', data=data)

        return {
            'success': True,
            'rule_id': rule_id,
            'is_enabled': is_enabled
        }


class DeleteRuleTool(WAFTool):
    """删除规则工具"""

    @property
    def name(self) -> str:
        return "delete_rule"

    @property
    def description(self) -> str:
        return "删除指定的自定义规则"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "rule_id",
                "type": "integer",
                "description": "规则ID",
                "required": True
            }
        ]

    def execute(self, rule_id: int, **kwargs) -> Dict[str, Any]:
        """执行删除规则"""
        data = {"id": rule_id}

        self.client.delete('/open/policy', data=data)

        return {
            'success': True,
            'rule_id': rule_id,
            'message': '规则已删除'
        }


class ListIPGroupsTool(WAFTool):
    """列出IP组工具"""

    @property
    def name(self) -> str:
        return "list_ip_groups"

    @property
    def description(self) -> str:
        return "查询IP组列表(/open/ipgroup)。IP组是IP地址的集合，只是存储IP，本身没有拦截或放行功能。用户提到IP组、IP集合、IP名单时用此工具。黑白名单是自定义规则，查黑白名单请用list_custom_rules。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "top",
                "type": "integer",
                "description": "返回数量限制",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出IP组"""
        params = {}
        if 'top' in kwargs and kwargs['top'] is not None:
            params['top'] = kwargs['top']

        result = self.client.get('/open/ipgroup', params=params)

        groups = result.get('data', {}).get('nodes', [])
        formatted_groups = []
        for group in groups:
            formatted_groups.append({
                'id': group.get('id'),
                'comment': group.get('comment'),
                'total': group.get('total'),
                'builtin': group.get('builtin'),
                'updated_at': group.get('updated_at')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'groups': formatted_groups
        }


class AddIPToGroupTool(WAFTool):
    """添加IP到IP组工具"""

    @property
    def name(self) -> str:
        return "add_ip_to_group"

    @property
    def description(self) -> str:
        return "向已有IP组添加IP地址(/open/ipgroup/append)。只是往IP组里加IP，不会产生拦截或放行效果。添加黑白名单规则请用create_custom_rule。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "ip_group_id",
                "type": "integer",
                "description": "IP组ID",
                "required": True
            },
            {
                "name": "ips",
                "type": "array",
                "description": "要添加的IP地址列表",
                "items": {"type": "string"},
                "required": True
            }
        ]

    def execute(self, ip_group_id: int, ips: List[str], **kwargs) -> Dict[str, Any]:
        """执行添加IP到IP组"""
        data = {
            "ip_group_ids": [ip_group_id],
            "ips": ips
        }

        self.client.post('/open/ipgroup/append', data=data)

        return {
            'success': True,
            'ip_group_id': ip_group_id,
            'ips': ips,
            'message': f'已添加 {len(ips)} 个IP到IP组 {ip_group_id}'
        }


class CreateIPGroupTool(WAFTool):
    """创建IP组工具"""

    @property
    def name(self) -> str:
        return "create_ip_group"

    @property
    def description(self) -> str:
        return "创建IP组(/open/ipgroup)。IP组只是存储IP地址的集合，本身没有拦截或放行功能，需要配合自定义规则使用。添加黑白名单规则请用create_custom_rule。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "comment",
                "type": "string",
                "description": "IP组名称/备注",
                "required": True
            },
            {
                "name": "ips",
                "type": "array",
                "description": "IP地址列表",
                "items": {"type": "string"},
                "required": True
            }
        ]

    def execute(self, comment: str, ips: List[str], **kwargs) -> Dict[str, Any]:
        """执行创建IP组"""
        data = {
            "comment": comment,
            "ips": ips
        }

        result = self.client.post('/open/ipgroup', data=data)

        return {
            'success': True,
            'group_id': result.get('data'),
            'comment': comment,
            'ips_count': len(ips),
            'message': f'IP组"{comment}"已创建，包含{len(ips)}个IP地址'
        }


class GetIPGroupDetailTool(WAFTool):
    """获取IP组详情工具"""

    @property
    def name(self) -> str:
        return "get_ip_group_detail"

    @property
    def description(self) -> str:
        return "获取IP组详情(/open/ipgroup/detail)，包括IP地址列表。IP组只是IP集合，不是规则。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "group_id",
                "type": "integer",
                "description": "IP组ID",
                "required": True
            }
        ]

    def execute(self, group_id: int, **kwargs) -> Dict[str, Any]:
        """执行获取IP组详情"""
        result = self.client.get('/open/ipgroup/detail', params={'id': group_id})

        data = result.get('data', {}).get('data', {})
        return {
            'success': True,
            'group': {
                'id': data.get('id'),
                'comment': data.get('comment'),
                'ips': data.get('ips', []),
                'total': data.get('total'),
                'builtin': data.get('builtin'),
                'reference': data.get('reference'),
                'updated_at': data.get('updated_at')
            }
        }


# ==================== 站点管理工具 ====================

class ListSitesTool(WAFTool):
    """列出站点工具"""

    @property
    def name(self) -> str:
        return "list_sites"

    @property
    def description(self) -> str:
        return "列出所有WAF保护的站点"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site",
                "type": "string",
                "description": "站点名称过滤",
                "required": False
            },
            {
                "name": "group_id",
                "type": "integer",
                "description": "分组ID",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出站点"""
        params = {}
        for key in ['site', 'group_id']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        result = self.client.get('/open/site', params=params)

        sites = result.get('data', {}).get('data', [])
        formatted_sites = []
        for site in sites:
            formatted_sites.append({
                'id': site.get('id'),
                'title': site.get('title'),
                'server_names': site.get('server_names'),
                'ports': site.get('ports'),
                'is_enabled': site.get('is_enabled'),
                'mode': site.get('mode'),
                'group_id': site.get('group_id'),
                'health_state': site.get('health_state')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'sites': formatted_sites
        }


class GetSiteDetailTool(WAFTool):
    """获取站点详情工具"""

    @property
    def name(self) -> str:
        return "get_site_detail"

    @property
    def description(self) -> str:
        return "获取指定站点的详细配置信息"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_id",
                "type": "integer",
                "description": "站点ID",
                "required": True
            }
        ]

    def execute(self, site_id: int, **kwargs) -> Dict[str, Any]:
        """执行获取站点详情"""
        result = self.client.get(f'/open/site/{site_id}')

        site = result.get('data', {})
        return {
            'success': True,
            'site': {
                'id': site.get('id'),
                'title': site.get('title'),
                'server_names': site.get('server_names'),
                'ports': site.get('ports'),
                'upstreams': site.get('upstreams'),
                'is_enabled': site.get('is_enabled'),
                'mode': site.get('mode'),
                'group_id': site.get('group_id'),
                'acl_enabled': site.get('acl_enabled'),
                'waiting_room_enabled': site.get('chaos_is_enabled'),
                'health_state': site.get('health_state'),
                'access_log_limit': site.get('access_log_limit'),
                'error_log_limit': site.get('error_log_limit')
            }
        }


class HealthCheckTool(WAFTool):
    """站点健康检查工具"""

    @property
    def name(self) -> str:
        return "health_check"

    @property
    def description(self) -> str:
        return "对指定站点执行一次健康检查，检测upstream源站是否可达。注意：这只是执行检查，不是开关。要开启/关闭健康检查功能请用set_site_health_check工具。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "hosts",
                "type": "array",
                "description": "主机名列表",
                "items": {"type": "string"},
                "required": False
            },
            {
                "name": "upstreams",
                "type": "array",
                "description": "upstream地址列表",
                "items": {"type": "string"},
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行健康检查"""
        data = {}
        if 'hosts' in kwargs:
            data['hosts'] = kwargs['hosts']
        if 'upstreams' in kwargs:
            data['upstreams'] = kwargs['upstreams']

        result = self.client.post('/open/site/healthcheck', data=data)

        return {
            'success': True,
            'health_results': result.get('data', {})
        }


class SetSiteHealthCheckTool(WAFTool):
    """设置站点健康检查开关工具"""

    @property
    def name(self) -> str:
        return "set_site_health_check"

    @property
    def description(self) -> str:
        return "开启或关闭站点的健康检查功能。开启后WAF会定期检测源站是否可达。关闭后不再检测。注意：这不是执行检查，是开关配置。执行检查请用health_check工具。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_id",
                "type": "integer",
                "description": "站点ID",
                "required": True
            },
            {
                "name": "enabled",
                "type": "boolean",
                "description": "True=开启健康检查，False=关闭健康检查",
                "required": True
            }
        ]

    def execute(self, site_id: int, enabled: bool, **kwargs) -> Dict[str, Any]:
        """执行设置健康检查开关"""
        # 先获取站点完整配置
        result = self.client.get(f'/open/site/{site_id}')
        site = result.get('data', {})

        # 修改health_check字段
        site['health_check'] = enabled

        # 更新站点
        self.client.put('/open/site', data=site)

        return {
            'success': True,
            'site_id': site_id,
            'health_check': enabled,
            'message': f'站点{site_id}的健康检查已{"开启" if enabled else "关闭"}'
        }


class SetSiteModeTool(WAFTool):
    """设置站点模式工具"""

    @property
    def name(self) -> str:
        return "set_site_mode"

    @property
    def description(self) -> str:
        return "设置站点运行模式（0:防护模式, 1:下线模式, 2:只检模式）"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_ids",
                "type": "array",
                "description": "站点ID列表",
                "items": {"type": "integer"},
                "required": True
            },
            {
                "name": "mode",
                "type": "integer",
                "description": "运行模式（0:防护模式, 1:下线模式, 2:只检模式）",
                "enum": [0, 1, 2],
                "required": True
            }
        ]

    def execute(self, site_ids: List[int], mode: int, **kwargs) -> Dict[str, Any]:
        """执行设置站点模式"""
        data = {
            'ids': site_ids,
            'mode': mode
        }

        self.client.put('/open/site/mode', data=data)

        mode_names = {0: '防护模式', 1: '下线模式', 2: '只检模式'}
        return {
            'success': True,
            'site_ids': site_ids,
            'mode': mode,
            'mode_name': mode_names.get(mode, '未知模式')
        }


# ==================== 防护配置工具 ====================

class SetChallengeTool(WAFTool):
    """人机验证工具"""

    @property
    def name(self) -> str:
        return "set_challenge"

    @property
    def description(self) -> str:
        return "设置站点的人机验证配置(PUT /open/site/challenge)。人机验证用于识别访问者是否为真实人类，防止自动化攻击。enable为开关，level为防护级别，expire为验证通过后的有效期（秒）。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_id",
                "type": "integer",
                "description": "站点ID",
                "required": True
            },
            {
                "name": "enable",
                "type": "boolean",
                "description": "是否启用",
                "required": True
            },
            {
                "name": "level",
                "type": "integer",
                "description": "防护级别",
                "required": False
            },
            {
                "name": "expire",
                "type": "integer",
                "description": "过期时间（秒）",
                "required": False
            }
        ]

    def execute(self, site_id: int, enable: bool, **kwargs) -> Dict[str, Any]:
        """执行设置人机验证"""
        data = {
            'id': site_id,
            'enable': enable
        }
        if 'level' in kwargs:
            data['level'] = kwargs['level']
        if 'expire' in kwargs:
            data['expire'] = kwargs['expire']

        self.client.put('/open/site/challenge', data=data)

        return {
            'success': True,
            'site_id': site_id,
            'enable': enable,
            'message': f'站点{site_id}的人机验证已{"开启" if enable else "关闭"}'
        }


class SetAuthDefenseTool(WAFTool):
    """身份认证工具"""

    @property
    def name(self) -> str:
        return "set_auth_defense"

    @property
    def description(self) -> str:
        return "设置站点的身份认证配置(PUT /open/site/defense)。身份认证要求访问者登录后才能访问站点。enable为开关，auth_source_ids为认证源ID列表，negate为是否取反匹配，pattern为匹配规则。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_id",
                "type": "integer",
                "description": "站点ID",
                "required": True
            },
            {
                "name": "enable",
                "type": "boolean",
                "description": "是否启用",
                "required": True
            },
            {
                "name": "pattern",
                "type": "array",
                "description": "匹配模式",
                "items": {"type": "object"},
                "required": False
            },
            {
                "name": "auth_source_ids",
                "type": "array",
                "description": "认证源ID列表",
                "items": {"type": "integer"},
                "required": False
            },
            {
                "name": "negate",
                "type": "boolean",
                "description": "是否取反",
                "required": False
            }
        ]

    def execute(self, site_id: int, enable: bool, **kwargs) -> Dict[str, Any]:
        """执行设置认证防护"""
        data = {
            'id': site_id,
            'enable': enable
        }
        if 'pattern' in kwargs:
            data['pattern'] = kwargs['pattern']
        if 'auth_source_ids' in kwargs:
            data['auth_source_ids'] = kwargs['auth_source_ids']
        if 'negate' in kwargs:
            data['negate'] = kwargs['negate']

        self.client.put('/open/site/defense', data=data)

        return {
            'success': True,
            'site_id': site_id,
            'enable': enable,
            'message': f'站点{site_id}的身份认证已{"开启" if enable else "关闭"}'
        }


class SetACLTool(WAFTool):
    """设置频率限制工具"""

    @property
    def name(self) -> str:
        return "set_rate_limit"

    @property
    def description(self) -> str:
        return "设置站点的频率限制（ACL）配置"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "site_id",
                "type": "integer",
                "description": "站点ID",
                "required": True
            },
            {
                "name": "rules",
                "type": "array",
                "description": "频率限制规则列表",
                "items": {"type": "object"},
                "required": True
            },
            {
                "name": "use_global",
                "type": "boolean",
                "description": "是否使用全局配置",
                "required": False
            }
        ]

    def execute(self, site_id: int, rules: List[Dict], **kwargs) -> Dict[str, Any]:
        """执行设置频率限制"""
        data = {
            'rules': rules
        }
        if 'use_global' in kwargs:
            data['use_global'] = kwargs['use_global']

        self.client.put(f'/open/site/{site_id}/acl', data=data)

        return {
            'success': True,
            'site_id': site_id,
            'rules_count': len(rules),
            'message': f'已设置 {len(rules)} 条频率限制规则'
        }


class SetACLEnableTool(WAFTool):
    """CC防护开关工具"""

    @property
    def name(self) -> str:
        return "set_acl_enabled"

    @property
    def description(self) -> str:
        return "开启或关闭站点的CC防护功能(PUT /open/site/challenge)。enable为CC防护开关。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "enable", "type": "boolean", "description": "CC防护开关，true=开启，false=关闭", "required": True}
        ]

    def execute(self, site_id: int, enable: bool, **kwargs) -> Dict[str, Any]:
        data = {'id': site_id, 'enable': enable}
        self.client.put('/open/site/challenge', data=data)
        status = "开启" if enable else "关闭"
        return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}的CC防护已{status}'}


class GetACLLogsTool(WAFTool):
    """获取频率限制日志工具"""

    @property
    def name(self) -> str:
        return "get_acl_logs"

    @property
    def description(self) -> str:
        return "查询频率限制日志"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "page",
                "type": "integer",
                "description": "页码",
                "required": False
            },
            {
                "name": "page_size",
                "type": "integer",
                "description": "每页数量",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行查询频率限制日志"""
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        result = self.client.get('/open/records/acl', params=params)

        logs = result.get('data', {}).get('data', [])
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                'id': log.get('id'),
                'created_at': log.get('created_at'),
                'ip': log.get('ip'),
                'action': log.get('action'),
                'count': log.get('count'),
                'period': log.get('period'),
                'site_title': log.get('site_title'),
                'site_server_names': log.get('site_server_names')
            })

        return {
            'success': True,
            'total': result.get('data', {}).get('total', 0),
            'logs': formatted_logs
        }


class ReleaseACLBlockTool(WAFTool):
    """解除频率限制封禁工具"""

    @property
    def name(self) -> str:
        return "release_acl_block"

    @property
    def description(self) -> str:
        return "解除频率限制的IP封禁"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "rule_ids",
                "type": "array",
                "description": "规则ID列表",
                "items": {"type": "integer"},
                "required": True
            }
        ]

    def execute(self, rule_ids: List[int], **kwargs) -> Dict[str, Any]:
        """执行解除频率限制封禁"""
        data = {'id': rule_ids}

        self.client.put('/open/acl/relieve', data=data)

        return {
            'success': True,
            'rule_ids': rule_ids,
            'count': len(rule_ids),
            'message': f'已解除 {len(rule_ids)} 个频率限制封禁'
        }


# ==================== 证书管理工具 ====================

class ListCertsTool(WAFTool):
    """列出证书工具"""

    @property
    def name(self) -> str:
        return "list_certs"

    @property
    def description(self) -> str:
        return "列出所有SSL证书信息，包括证书名称、域名、有效期等"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "page",
                "type": "integer",
                "description": "页码",
                "required": False
            },
            {
                "name": "page_size",
                "type": "integer",
                "description": "每页数量",
                "required": False
            }
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出证书"""
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        result = self.client.get('/open/cert', params=params)

        data = result.get('data', {})
        certs = data.get('nodes', []) or data.get('data', [])
        formatted_certs = []
        for cert in certs:
            formatted_certs.append({
                'id': cert.get('id'),
                'name': cert.get('name'),
                'type': cert.get('type'),
                'domains': cert.get('domains'),
                'issuer': cert.get('issuer'),
                'expired': cert.get('expired'),
                'self_signature': cert.get('self_signature'),
                'trusted': cert.get('trusted'),
                'valid_before': cert.get('valid_before'),
                'related_sites': cert.get('related_sites')
            })

        return {
            'success': True,
            'total': data.get('total', 0),
            'certs': formatted_certs
        }


class GetCertDetailTool(WAFTool):
    """获取证书详情工具"""

    @property
    def name(self) -> str:
        return "get_cert_detail"

    @property
    def description(self) -> str:
        return "获取指定SSL证书的详细信息"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {
                "name": "cert_id",
                "type": "integer",
                "description": "证书ID",
                "required": True
            }
        ]

    def execute(self, cert_id: int, **kwargs) -> Dict[str, Any]:
        """执行获取证书详情"""
        result = self.client.get(f'/open/cert/{cert_id}')

        cert = result.get('data', {})
        return {
            'success': True,
            'cert': {
                'id': cert.get('id'),
                'name': cert.get('name'),
                'type': cert.get('type'),
                'domains': cert.get('domains'),
                'not_before': cert.get('not_before'),
                'not_after': cert.get('not_after'),
                'issuer': cert.get('issuer'),
                'is_enabled': cert.get('is_enabled'),
                'auto_ssl': cert.get('auto_ssl'),
                'created_at': cert.get('created_at'),
                'updated_at': cert.get('updated_at')
            }
        }


# ==================== 用户管理工具 ====================

class ListUsersTool(WAFTool):
    """列出控制台用户工具"""

    @property
    def name(self) -> str:
        return "list_users"

    @property
    def description(self) -> str:
        return "列出WAF控制台的所有用户信息，包括用户名、角色、登录状态、二次认证状态等"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出用户"""
        result = self.client.get('/open/users')

        users = result.get('data', {}).get('data', [])
        formatted_users = []
        for user in users:
            formatted_users.append({
                'id': user.get('id'),
                'username': user.get('username'),
                'role': user.get('role'),
                'password_enabled': user.get('password_enabled'),
                'tfa_enabled': user.get('tfa_enabled'),
                'tfa_binded': user.get('tfa_binded'),
                'last_login': user.get('last_login'),
                'pwd_updated_at': user.get('pwd_updated_at'),
                'lock_until': user.get('lock_until')
            })

        return {
            'success': True,
            'total': len(formatted_users),
            'users': formatted_users
        }


# ==================== 规则管理（补充） ====================

class GetRuleDetailTool(WAFTool):
    """获取规则详情工具"""

    @property
    def name(self) -> str:
        return "get_rule_detail"

    @property
    def description(self) -> str:
        return "获取自定义规则的详细信息(/open/policy/detail)，包括匹配模式、动作等。重要：必须先通过list_custom_rules获取规则ID，不要猜测rule_id。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "rule_id", "type": "integer", "description": "规则ID", "required": True}
        ]

    def execute(self, rule_id: int, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/policy/detail', params={'id': rule_id})
        rule = result.get('data', {})
        return {
            'success': True,
            'rule': {
                'id': rule.get('id'),
                'name': rule.get('name'),
                'action': rule.get('action'),
                'is_enabled': rule.get('is_enabled'),
                'level': rule.get('level'),
                'log': rule.get('log'),
                'pattern': rule.get('pattern'),
                'builtin': rule.get('builtin'),
                'expire': rule.get('expire'),
                'created_at': rule.get('created_at'),
                'updated_at': rule.get('updated_at')
            }
        }


class UpdateRuleTool(WAFTool):
    """更新规则工具"""

    @property
    def name(self) -> str:
        return "update_rule"

    @property
    def description(self) -> str:
        return "更新自定义规则(/open/policy)。可修改名称、动作、匹配模式、启停等。需要传入规则ID和要更新的字段。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "rule_id", "type": "integer", "description": "规则ID", "required": True},
            {"name": "name", "type": "string", "description": "规则名称", "required": False},
            {"name": "action", "type": "integer", "description": "操作类型：0=放行，1=拦截，2=验证码，3=认证防护", "required": False},
            {"name": "is_enabled", "type": "boolean", "description": "是否启用", "required": False},
            {"name": "pattern", "type": "array", "description": "匹配模式", "items": {"type": "array", "items": {"type": "object"}}, "required": False},
            {"name": "level", "type": "integer", "description": "风险级别", "required": False},
            {"name": "log", "type": "boolean", "description": "是否记录日志", "required": False},
            {"name": "expire", "type": "integer", "description": "过期时间戳，0表示永不过期", "required": False}
        ]

    def execute(self, rule_id: int, **kwargs) -> Dict[str, Any]:
        data = {'id': rule_id}
        for key in ['name', 'action', 'is_enabled', 'pattern', 'level', 'log', 'expire']:
            if key in kwargs and kwargs[key] is not None:
                data[key] = kwargs[key]
        result = self.client.put('/open/policy', data=data)
        return {'success': True, 'rule_id': rule_id, 'message': f'规则{rule_id}已更新'}


class OrderRulesTool(WAFTool):
    """规则排序工具"""

    @property
    def name(self) -> str:
        return "order_rules"

    @property
    def description(self) -> str:
        return "调整自定义规则的排序(/open/policy/order)。可按ID排序或按值排序。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "order_type", "type": "string", "description": "排序类型：order=按ID排序，value=按值排序，batch=批量排序", "required": True},
            {"name": "order_data", "type": "object", "description": "排序数据，具体结构取决于排序类型", "required": True}
        ]

    def execute(self, order_type: str, order_data: dict, **kwargs) -> Dict[str, Any]:
        if order_type == 'value':
            result = self.client.put('/open/policy/order/value', data=order_data)
        elif order_type == 'batch':
            result = self.client.put('/open/policy/orders', data=order_data)
        else:
            result = self.client.put('/open/policy/order', data=order_data)
        return {'success': True, 'message': '规则排序已更新'}


# ==================== IP组管理（补充） ====================

class UpdateIPGroupTool(WAFTool):
    """更新IP组工具"""

    @property
    def name(self) -> str:
        return "update_ip_group"

    @property
    def description(self) -> str:
        return "更新IP组信息(/open/ipgroup)，可修改名称和IP列表"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "group_id", "type": "integer", "description": "IP组ID", "required": True},
            {"name": "comment", "type": "string", "description": "IP组名称/备注", "required": False},
            {"name": "ips", "type": "array", "description": "IP地址列表", "items": {"type": "string"}, "required": False}
        ]

    def execute(self, group_id: int, **kwargs) -> Dict[str, Any]:
        data = {'id': group_id}
        if 'comment' in kwargs:
            data['comment'] = kwargs['comment']
        if 'ips' in kwargs:
            data['ips'] = kwargs['ips']
        self.client.put('/open/ipgroup', data=data)
        return {'success': True, 'group_id': group_id, 'message': f'IP组{group_id}已更新'}


class DeleteIPGroupTool(WAFTool):
    """删除IP组工具"""

    @property
    def name(self) -> str:
        return "delete_ip_group"

    @property
    def description(self) -> str:
        return "删除指定的IP组(/open/ipgroup)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "group_id", "type": "integer", "description": "IP组ID", "required": True}
        ]

    def execute(self, group_id: int, **kwargs) -> Dict[str, Any]:
        self.client.delete('/open/ipgroup', data={'id': group_id})
        return {'success': True, 'group_id': group_id, 'message': f'IP组{group_id}已删除'}


class GetCrawlerGroupTool(WAFTool):
    """获取搜索引擎蜘蛛组工具"""

    @property
    def name(self) -> str:
        return "get_crawler_group"

    @property
    def description(self) -> str:
        return "获取搜索引擎蜘蛛IP组信息(/open/ipgroup/crawler)"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/ipgroup/crawler')
        return {'success': True, 'data': result.get('data', {})}


class UpdateCrawlerGroupTool(WAFTool):
    """更新搜索引擎蜘蛛组工具"""

    @property
    def name(self) -> str:
        return "update_crawler_group"

    @property
    def description(self) -> str:
        return "更新搜索引擎蜘蛛IP组配置(/open/ipgroup/crawler)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "ip_group_id", "type": "integer", "description": "蜘蛛IP组ID", "required": True}
        ]

    def execute(self, ip_group_id: int, **kwargs) -> Dict[str, Any]:
        self.client.post('/open/ipgroup/crawler', data={'ip_group_id': ip_group_id})
        return {'success': True, 'message': '搜索引擎蜘蛛组已更新'}


class GetIPGroupByLinkTool(WAFTool):
    """通过链接获取IP组工具"""

    @property
    def name(self) -> str:
        return "get_ip_group_by_link"

    @property
    def description(self) -> str:
        return "通过链接获取IP地址(/open/ipgroup/link)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "url", "type": "string", "description": "链接地址", "required": True}
        ]

    def execute(self, url: str, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/ipgroup/link', params={'url': url})
        return {'success': True, 'data': result.get('data', {})}


class CreateIPGroupByLinkTool(WAFTool):
    """通过链接创建IP组工具"""

    @property
    def name(self) -> str:
        return "create_ip_group_by_link"

    @property
    def description(self) -> str:
        return "通过链接创建IP组(/open/ipgroup/link)，自动从链接获取IP列表"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "comment", "type": "string", "description": "IP组名称/备注", "required": True},
            {"name": "url", "type": "string", "description": "链接地址", "required": True}
        ]

    def execute(self, comment: str, url: str, **kwargs) -> Dict[str, Any]:
        result = self.client.post('/open/ipgroup/link', data={'comment': comment, 'url': url})
        return {'success': True, 'group_id': result.get('data'), 'message': f'IP组"{comment}"已通过链接创建'}


# ==================== 站点管理（补充） ====================

class CreateSiteTool(WAFTool):
    """创建站点工具"""

    @property
    def name(self) -> str:
        return "create_site"

    @property
    def description(self) -> str:
        return "创建新的WAF保护站点(/open/site)。需要提供域名、端口、upstream源站地址等。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "server_names", "type": "array", "description": "域名列表", "items": {"type": "string"}, "required": True},
            {"name": "ports", "type": "array", "description": "监听端口列表", "items": {"type": "string"}, "required": True},
            {"name": "upstreams", "type": "array", "description": "源站地址列表", "items": {"type": "string"}, "required": True},
            {"name": "comment", "type": "string", "description": "站点备注", "required": False},
            {"name": "group_id", "type": "integer", "description": "分组ID", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        data = {}
        for key in ['server_names', 'ports', 'upstreams', 'comment', 'group_id']:
            if key in kwargs and kwargs[key] is not None:
                data[key] = kwargs[key]
        result = self.client.post('/open/site', data=data)
        return {'success': True, 'site_id': result.get('data'), 'message': '站点已创建'}


class DeleteSiteTool(WAFTool):
    """删除站点工具"""

    @property
    def name(self) -> str:
        return "delete_site"

    @property
    def description(self) -> str:
        return "删除指定的WAF保护站点(/open/site)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_ids", "type": "array", "description": "站点ID列表", "items": {"type": "integer"}, "required": True}
        ]

    def execute(self, site_ids: List[int], **kwargs) -> Dict[str, Any]:
        self.client.delete('/open/site', data={'ids': site_ids})
        return {'success': True, 'site_ids': site_ids, 'message': f'已删除{len(site_ids)}个站点'}


class UpdateSiteBasicInfoTool(WAFTool):
    """更新站点基本信息工具"""

    @property
    def name(self) -> str:
        return "update_site_basic_info"

    @property
    def description(self) -> str:
        return "更新站点基本信息(/open/site/{id}/basic_info)，如备注、健康检查开关等"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "update_data", "type": "object", "description": "要更新的字段，如comment、health_check等", "required": True}
        ]

    def execute(self, site_id: int, update_data: dict, **kwargs) -> Dict[str, Any]:
        self.client.put(f'/open/site/{site_id}/basic_info', data=update_data)
        return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}基本信息已更新'}


class UpdateSiteGroupTool(WAFTool):
    """更新站点分组工具"""

    @property
    def name(self) -> str:
        return "update_site_group"

    @property
    def description(self) -> str:
        return "更新站点所属分组(/open/site/{id}/group)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "group_id", "type": "integer", "description": "分组ID", "required": True}
        ]

    def execute(self, site_id: int, group_id: int, **kwargs) -> Dict[str, Any]:
        self.client.put(f'/open/site/{site_id}/group', data={'group_id': group_id})
        return {'success': True, 'site_id': site_id, 'message': f'站点已移至分组{group_id}'}


class GetSiteProxyTool(WAFTool):
    """获取站点代理配置工具"""

    @property
    def name(self) -> str:
        return "get_site_proxy"

    @property
    def description(self) -> str:
        return "获取站点的代理/安全配置(/open/site/{id}/proxy)，包括HTTPS、HSTS、HTTP2等"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True}
        ]

    def execute(self, site_id: int, **kwargs) -> Dict[str, Any]:
        result = self.client.get(f'/open/site/{site_id}/proxy')
        return {'success': True, 'site_id': site_id, 'config': result.get('data', {})}


class SetSiteProxyTool(WAFTool):
    """设置站点代理配置工具"""

    @property
    def name(self) -> str:
        return "set_site_proxy"

    @property
    def description(self) -> str:
        return "设置站点的代理/安全配置(/open/site/{id}/proxy)，如HTTPS强制跳转、HSTS、HTTP2等。config为完整配置对象。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "config", "type": "object", "description": "代理配置对象", "required": True}
        ]

    def execute(self, site_id: int, config: dict, **kwargs) -> Dict[str, Any]:
        self.client.put(f'/open/site/{site_id}/proxy', data=config)
        return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}代理配置已更新'}


class GetSiteNginxConfigTool(WAFTool):
    """获取站点Nginx配置工具"""

    @property
    def name(self) -> str:
        return "get_site_nginx_config"

    @property
    def description(self) -> str:
        return "获取站点的Nginx配置(/open/site/{id}/nginx_config)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True}
        ]

    def execute(self, site_id: int, **kwargs) -> Dict[str, Any]:
        result = self.client.get(f'/open/site/{site_id}/nginx_config')
        return {'success': True, 'site_id': site_id, 'config': result.get('data', {})}


class SetSiteNginxConfigTool(WAFTool):
    """设置站点Nginx配置工具"""

    @property
    def name(self) -> str:
        return "set_site_nginx_config"

    @property
    def description(self) -> str:
        return "更新站点的Nginx配置(/open/site/{id}/nginx_config)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "config", "type": "object", "description": "Nginx配置对象", "required": True}
        ]

    def execute(self, site_id: int, config: dict, **kwargs) -> Dict[str, Any]:
        self.client.put(f'/open/site/{site_id}/nginx_config', data=config)
        return {'success': True, 'site_id': site_id, 'message': f'站点{site_id} Nginx配置已更新'}


class GetSiteResourcesTool(WAFTool):
    """获取站点路由资源工具"""

    @property
    def name(self) -> str:
        return "get_site_resources"

    @property
    def description(self) -> str:
        return "获取站点的路由资源列表(/open/site/{id}/resources)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True}
        ]

    def execute(self, site_id: int, **kwargs) -> Dict[str, Any]:
        result = self.client.get(f'/open/site/{site_id}/resources')
        return {'success': True, 'site_id': site_id, 'resources': result.get('data', {})}


class ManageSiteExcludesTool(WAFTool):
    """站点路由采集排除工具"""

    @property
    def name(self) -> str:
        return "manage_site_excludes"

    @property
    def description(self) -> str:
        return "管理站点路由采集排除配置(GET/POST /open/site/{id}/excludes)。action=get获取，action=set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置", "required": True},
            {"name": "excludes", "type": "object", "description": "排除配置，action=set时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get(f'/open/site/{site_id}/excludes')
            return {'success': True, 'site_id': site_id, 'excludes': result.get('data', {})}
        else:
            self.client.post(f'/open/site/{site_id}/excludes', data=kwargs.get('excludes', {}))
            return {'success': True, 'site_id': site_id, 'message': '路由采集排除配置已更新'}


class GetSiteLogsTool(WAFTool):
    """获取站点日志工具"""

    @property
    def name(self) -> str:
        return "get_site_logs"

    @property
    def description(self) -> str:
        return "获取站点的访问/错误日志(/open/site/{id}/log)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "type", "type": "string", "description": "日志类型：access=访问日志，error=错误日志", "required": False},
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, site_id: int, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['type', 'page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get(f'/open/site/{site_id}/log', params=params)
        return {'success': True, 'site_id': site_id, 'data': result.get('data', {})}


class ManageSiteGroupsTool(WAFTool):
    """站点分组管理工具"""

    @property
    def name(self) -> str:
        return "manage_site_groups"

    @property
    def description(self) -> str:
        return "管理站点分组(GET/POST/PUT/DELETE /open/site/group)。action=list列出，create创建，update更新，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "list=列出分组，create=创建分组，update=更新分组，delete=删除分组", "required": True},
            {"name": "group_id", "type": "integer", "description": "分组ID，update/delete时必传", "required": False},
            {"name": "name", "type": "string", "description": "分组名称，create/update时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get('/open/site/group')
            return {'success': True, 'groups': result.get('data', {})}
        elif action == 'create':
            result = self.client.post('/open/site/group', data={'name': kwargs.get('name', '')})
            return {'success': True, 'group_id': result.get('data'), 'message': f'分组"{kwargs.get("name")}"已创建'}
        elif action == 'update':
            group_id = kwargs.get('group_id')
            self.client.put(f'/open/site/group/{group_id}', data={'name': kwargs.get('name', '')})
            return {'success': True, 'message': f'分组{group_id}已更新'}
        elif action == 'delete':
            group_id = kwargs.get('group_id')
            self.client.delete(f'/open/site/group/{group_id}')
            return {'success': True, 'message': f'分组{group_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


# ==================== 防护配置（补充） ====================

class ManageRateLimitTool(WAFTool):
    """频率限制CRUD工具"""

    @property
    def name(self) -> str:
        return "manage_rate_limit"

    @property
    def description(self) -> str:
        return "管理站点频率限制规则(GET/POST/PUT/DELETE /open/site/{id}/acl)。action=list列出，add添加，update更新，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "list=列出规则，add=添加规则，update=更新规则，delete=删除规则", "required": True},
            {"name": "rule_id", "type": "integer", "description": "规则ID，update/delete时必传", "required": False},
            {"name": "rule_data", "type": "object", "description": "规则数据，add/update时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get(f'/open/site/{site_id}/acl')
            return {'success': True, 'site_id': site_id, 'rules': result.get('data', {})}
        elif action == 'add':
            result = self.client.post(f'/open/site/{site_id}/acl', data=kwargs.get('rule_data', {}))
            return {'success': True, 'rule_id': result.get('data'), 'message': '频率限制规则已添加'}
        elif action == 'update':
            rule_id = kwargs.get('rule_id')
            self.client.put(f'/open/site/{site_id}/acl/{rule_id}', data=kwargs.get('rule_data', {}))
            return {'success': True, 'message': f'频率限制规则{rule_id}已更新'}
        elif action == 'delete':
            rule_id = kwargs.get('rule_id')
            self.client.delete(f'/open/site/{site_id}/acl/{rule_id}')
            return {'success': True, 'message': f'频率限制规则{rule_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class GetChallengeConfigTool(WAFTool):
    """获取人机验证全局配置工具"""

    @property
    def name(self) -> str:
        return "get_challenge_config"

    @property
    def description(self) -> str:
        return "获取全局人机验证配置(/open/challenge/config)"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/challenge/config')
        return {'success': True, 'config': result.get('data', {})}


class SetChallengeConfigTool(WAFTool):
    """设置人机验证全局配置工具"""

    @property
    def name(self) -> str:
        return "set_challenge_config"

    @property
    def description(self) -> str:
        return "设置全局人机验证配置(/open/challenge/config)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "config", "type": "object", "description": "人机验证配置对象", "required": True}
        ]

    def execute(self, config: dict, **kwargs) -> Dict[str, Any]:
        self.client.post('/open/challenge/config', data=config)
        return {'success': True, 'message': '全局人机验证配置已更新'}


class ManageAntiTamperTool(WAFTool):
    """反篡改管理工具"""

    @property
    def name(self) -> str:
        return "manage_anti_tamper"

    @property
    def description(self) -> str:
        return "管理反篡改规则(GET/POST/PUT/DELETE /business/site/{site_id}/anti_tamper)。action=list列出，create创建，update更新，delete删除，refresh刷新。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "list/create/update/delete/refresh", "required": True},
            {"name": "rule_id", "type": "integer", "description": "规则ID，update/delete时必传", "required": False},
            {"name": "rule_data", "type": "object", "description": "规则数据，create/update时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get(f'/business/site/{site_id}/anti_tamper')
            return {'success': True, 'site_id': site_id, 'rules': result.get('data', {})}
        elif action == 'create':
            result = self.client.post(f'/business/site/{site_id}/anti_tamper', data=kwargs.get('rule_data', {}))
            return {'success': True, 'rule_id': result.get('data'), 'message': '反篡改规则已创建'}
        elif action == 'update':
            rule_id = kwargs.get('rule_id')
            self.client.put(f'/business/anti_tamper/{rule_id}', data=kwargs.get('rule_data', {}))
            return {'success': True, 'message': f'反篡改规则{rule_id}已更新'}
        elif action == 'delete':
            rule_id = kwargs.get('rule_id')
            self.client.delete(f'/business/anti_tamper/{rule_id}')
            return {'success': True, 'message': f'反篡改规则{rule_id}已删除'}
        elif action == 'refresh':
            self.client.put(f'/business/site/{site_id}/anti_tamper')
            return {'success': True, 'message': f'站点{site_id}反篡改已刷新'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageDynamicDefenseTool(WAFTool):
    """动态防护管理工具"""

    @property
    def name(self) -> str:
        return "manage_dynamic_defense"

    @property
    def description(self) -> str:
        return "管理站点动态防护配置(GET/POST /open/site/{id}/chaos)。action=get获取，set设置。动态防护包括HTML加密、JS加密、图片加密等。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置", "required": True},
            {"name": "config", "type": "object", "description": "动态防护配置对象，action=set时必传。包含is_enabled、html_encryption、js_encryption、img_encryption等字段", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get(f'/open/site/{site_id}/chaos')
            return {'success': True, 'site_id': site_id, 'config': result.get('data', {})}
        else:
            self.client.post(f'/open/site/{site_id}/chaos', data=kwargs.get('config', {}))
            return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}动态防护配置已更新'}


# ==================== 语义规则 ====================

class GetGlobalSemanticsTool(WAFTool):
    """获取全局语义配置工具"""

    @property
    def name(self) -> str:
        return "get_global_semantics"

    @property
    def description(self) -> str:
        return "获取全局语义分析配置(/open/global/mode)"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/global/mode')
        return {'success': True, 'config': result.get('data', {})}


class SetGlobalSemanticsTool(WAFTool):
    """设置全局语义配置工具"""

    @property
    def name(self) -> str:
        return "set_global_semantics"

    @property
    def description(self) -> str:
        return "更新全局语义分析配置(/open/global/mode)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "config", "type": "object", "description": "语义配置对象", "required": True}
        ]

    def execute(self, config: dict, **kwargs) -> Dict[str, Any]:
        self.client.put('/open/global/mode', data=config)
        return {'success': True, 'message': '全局语义配置已更新'}


class ManageSiteSemanticsTool(WAFTool):
    """站点语义配置工具"""

    @property
    def name(self) -> str:
        return "manage_site_semantics"

    @property
    def description(self) -> str:
        return "管理站点语义分析配置(GET/PUT /open/site/{id}/semantics)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置", "required": True},
            {"name": "config", "type": "object", "description": "语义配置对象，action=set时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get(f'/open/site/{site_id}/semantics')
            return {'success': True, 'site_id': site_id, 'config': result.get('data', {})}
        else:
            self.client.put(f'/open/site/{site_id}/semantics', data=kwargs.get('config', {}))
            return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}语义配置已更新'}


# ==================== 增强规则 ====================

class GetEnhanceRulesTool(WAFTool):
    """获取增强规则工具"""

    @property
    def name(self) -> str:
        return "get_enhance_rules"

    @property
    def description(self) -> str:
        return "获取Skynet增强规则列表(/open/skynet/rule)"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/skynet/rule')
        return {'success': True, 'rules': result.get('data', {})}


class UpdateEnhanceRulesTool(WAFTool):
    """更新增强规则工具"""

    @property
    def name(self) -> str:
        return "update_enhance_rules"

    @property
    def description(self) -> str:
        return "更新Skynet增强规则(/open/skynet/rule)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "rule_data", "type": "object", "description": "增强规则数据", "required": True}
        ]

    def execute(self, rule_data: dict, **kwargs) -> Dict[str, Any]:
        self.client.put('/open/skynet/rule', data=rule_data)
        return {'success': True, 'message': '增强规则已更新'}


class ManageEnhanceRuleSwitchTool(WAFTool):
    """增强规则开关工具"""

    @property
    def name(self) -> str:
        return "manage_enhance_rule_switch"

    @property
    def description(self) -> str:
        return "管理增强规则全局开关(GET/PUT /open/skynet/rule/switch)。action=get获取状态，set设置状态。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取状态，set=设置状态", "required": True},
            {"name": "enabled", "type": "boolean", "description": "是否启用，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/skynet/rule/switch')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.put('/open/skynet/rule/switch', data={'is_enabled': kwargs.get('enabled', True)})
            return {'success': True, 'message': f'增强规则全局开关已{"开启" if kwargs.get("enabled") else "关闭"}'}


# ==================== 统计 ====================

class GetAdvanceStatsTool(WAFTool):
    """高级统计查询工具"""

    @property
    def name(self) -> str:
        return "get_advance_stats"

    @property
    def description(self) -> str:
        return "查询高级统计数据(/stat/advance/*)。stat_type可选：access(访问), attack(攻击), client(客户端), domain(域名), location(地理位置), page(页面), status_code(状态码), error_status_code(错误状态码)。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "stat_type", "type": "string", "description": "统计类型：access/attack/client/domain/location/page/status_code/error_status_code", "required": True},
            {"name": "site_id", "type": "integer", "description": "站点ID，不传则查全部", "required": False},
            {"name": "begin_time", "type": "integer", "description": "开始时间戳", "required": False},
            {"name": "end_time", "type": "integer", "description": "结束时间戳", "required": False}
        ]

    def execute(self, stat_type: str, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['site_id', 'begin_time', 'end_time']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get(f'/stat/advance/{stat_type}', params=params)
        return {'success': True, 'stat_type': stat_type, 'data': result.get('data', {})}


class GetAdvanceTrendsTool(WAFTool):
    """高级趋势查询工具"""

    @property
    def name(self) -> str:
        return "get_advance_trends"

    @property
    def description(self) -> str:
        return "查询高级趋势数据(/stat/advance/trend/*)。trend_type可选：access(访问趋势), intercept(拦截趋势)。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "trend_type", "type": "string", "description": "趋势类型：access=访问趋势，intercept=拦截趋势", "required": True},
            {"name": "site_id", "type": "integer", "description": "站点ID，不传则查全部", "required": False},
            {"name": "begin_time", "type": "integer", "description": "开始时间戳", "required": False},
            {"name": "end_time", "type": "integer", "description": "结束时间戳", "required": False}
        ]

    def execute(self, trend_type: str, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['site_id', 'begin_time', 'end_time']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get(f'/stat/advance/trend/{trend_type}', params=params)
        return {'success': True, 'trend_type': trend_type, 'data': result.get('data', {})}


# ==================== 告警 ====================

class GetAlarmConfigTool(WAFTool):
    """获取告警配置工具"""

    @property
    def name(self) -> str:
        return "get_alarm_config"

    @property
    def description(self) -> str:
        return "获取告警配置(/alarm)，包括钉钉、飞书、企微、Telegram、Discord等告警渠道"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/alarm')
        return {'success': True, 'config': result.get('data', {})}


class UpdateAlarmConfigTool(WAFTool):
    """更新告警配置工具"""

    @property
    def name(self) -> str:
        return "update_alarm_config"

    @property
    def description(self) -> str:
        return "更新告警配置(/alarm)或测试告警(/alarm/test)。action=update更新配置，action=test测试告警。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "update=更新配置，test=测试告警", "required": True},
            {"name": "config", "type": "object", "description": "告警配置对象", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'test':
            result = self.client.post('/alarm/test', data=kwargs.get('config', {}))
            return {'success': True, 'data': result.get('data', {}), 'message': '告警测试已发送'}
        else:
            self.client.put('/alarm', data=kwargs.get('config', {}))
            return {'success': True, 'message': '告警配置已更新'}


# ==================== 认证防护 ====================

class ManageAuthSourcesTool(WAFTool):
    """认证源管理工具"""

    @property
    def name(self) -> str:
        return "manage_auth_sources"

    @property
    def description(self) -> str:
        return "管理认证防护源(GET/POST/PUT/DELETE /open/auth_defense/source)。action=list列出，create创建，update更新，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "list/create/update/delete", "required": True},
            {"name": "source_id", "type": "integer", "description": "认证源ID，update/delete时必传", "required": False},
            {"name": "source_data", "type": "object", "description": "认证源数据，create/update时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get('/open/auth_defense/source')
            return {'success': True, 'sources': result.get('data', {})}
        elif action == 'create':
            result = self.client.post('/open/auth_defense/source', data=kwargs.get('source_data', {}))
            return {'success': True, 'source_id': result.get('data'), 'message': '认证源已创建'}
        elif action == 'update':
            source_id = kwargs.get('source_id')
            self.client.put(f'/open/auth_defense/source/{source_id}', data=kwargs.get('source_data', {}))
            return {'success': True, 'message': f'认证源{source_id}已更新'}
        elif action == 'delete':
            source_id = kwargs.get('source_id')
            self.client.delete(f'/open/auth_defense/source/{source_id}')
            return {'success': True, 'message': f'认证源{source_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ListAuthSourceUsersTool(WAFTool):
    """列出认证源用户工具"""

    @property
    def name(self) -> str:
        return "list_auth_source_users"

    @property
    def description(self) -> str:
        return "列出指定认证源的用户列表(/open/auth_defense/source/{id}/user)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "source_id", "type": "integer", "description": "认证源ID", "required": True},
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, source_id: int, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get(f'/open/auth_defense/source/{source_id}/user', params=params)
        return {'success': True, 'source_id': source_id, 'users': result.get('data', {})}


class ManageAuthDefenseUsersTool(WAFTool):
    """认证防护用户管理工具"""

    @property
    def name(self) -> str:
        return "manage_auth_defense_users"

    @property
    def description(self) -> str:
        return "管理认证防护用户(GET/POST/PUT/DELETE /open/auth_defense/user)。action=list列出，create创建，update更新，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "list/create/update/delete", "required": True},
            {"name": "user_id", "type": "integer", "description": "用户ID，update/delete时必传", "required": False},
            {"name": "user_data", "type": "object", "description": "用户数据，create/update时必传", "required": False},
            {"name": "page", "type": "integer", "description": "页码，list时可用", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量，list时可用", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            params = {}
            for key in ['page', 'page_size']:
                if key in kwargs and kwargs[key] is not None:
                    params[key] = kwargs[key]
            result = self.client.get('/open/auth_defense/user', params=params)
            return {'success': True, 'users': result.get('data', {})}
        elif action == 'create':
            result = self.client.post('/open/auth_defense/user', data=kwargs.get('user_data', {}))
            return {'success': True, 'user_id': result.get('data'), 'message': '认证防护用户已创建'}
        elif action == 'update':
            user_id = kwargs.get('user_id')
            self.client.put(f'/open/auth_defense/user/{user_id}', data=kwargs.get('user_data', {}))
            return {'success': True, 'message': f'认证防护用户{user_id}已更新'}
        elif action == 'delete':
            user_id = kwargs.get('user_id')
            self.client.delete(f'/open/auth_defense/user/{user_id}')
            return {'success': True, 'message': f'认证防护用户{user_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class GetAuthDefenseLogsTool(WAFTool):
    """认证防护日志工具"""

    @property
    def name(self) -> str:
        return "get_auth_defense_logs"

    @property
    def description(self) -> str:
        return "查询认证防护日志(/open/records/auth_defense)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get('/open/records/auth_defense', params=params)
        return {'success': True, 'data': result.get('data', {})}


# ==================== 系统信息 ====================

class GetSystemInfoTool(WAFTool):
    """系统信息工具"""

    @property
    def name(self) -> str:
        return "get_system_info"

    @property
    def description(self) -> str:
        return "获取系统信息，包括版本、架构、版本类型、系统key等"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "info_type", "type": "string", "description": "info=基本信息，edition=版本类型，arch=系统架构，key=系统key", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        info_type = kwargs.get('info_type', 'info')
        type_map = {
            'info': '/open/system',
            'edition': '/open/system/edition',
            'arch': '/open/system/arch',
            'key': '/open/system/key'
        }
        endpoint = type_map.get(info_type, '/open/system')
        result = self.client.get(endpoint)
        return {'success': True, 'info_type': info_type, 'data': result.get('data', {})}


class GetLicenseTool(WAFTool):
    """获取授权信息工具"""

    @property
    def name(self) -> str:
        return "get_license"

    @property
    def description(self) -> str:
        return "获取授权信息(/open/system/authorize)，包括组织名、到期时间、版本等"

    @property
    def parameters(self) -> List[Dict]:
        return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        result = self.client.get('/open/system/authorize')
        return {'success': True, 'license': result.get('data', {})}


class ManageLicenseTool(WAFTool):
    """授权管理工具"""

    @property
    def name(self) -> str:
        return "manage_license"

    @property
    def description(self) -> str:
        return "管理授权(POST/PUT/DELETE /open/system/authorize)。action=apply申请授权，reapply重新申请，delete删除授权。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "apply=申请授权，reapply=重新申请，delete=删除授权", "required": True},
            {"name": "license_data", "type": "object", "description": "授权数据，apply时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'apply':
            result = self.client.post('/open/system/authorize', data=kwargs.get('license_data', {}))
            return {'success': True, 'data': result.get('data'), 'message': '授权申请已提交'}
        elif action == 'reapply':
            result = self.client.put('/open/system/authorize')
            return {'success': True, 'data': result.get('data'), 'message': '授权重新申请已提交'}
        elif action == 'delete':
            self.client.delete('/open/system/authorize')
            return {'success': True, 'message': '授权已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageDetectorTool(WAFTool):
    """检测引擎工具"""

    @property
    def name(self) -> str:
        return "manage_detector"

    @property
    def description(self) -> str:
        return "管理检测引擎性能模式(GET/POST /open/detector)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取模式，set=设置模式", "required": True},
            {"name": "mode", "type": "string", "description": "性能模式，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/detector')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.post('/open/detector', data={'mode': kwargs.get('mode', '')})
            return {'success': True, 'message': f'检测引擎模式已设置为{kwargs.get("mode")}'}


class ManageLogCleanTool(WAFTool):
    """日志清理配置工具"""

    @property
    def name(self) -> str:
        return "manage_log_clean"

    @property
    def description(self) -> str:
        return "管理日志清理配置(GET/POST /open/global/log_clean)。action=get获取配置，set设置配置。包含max_day(防护日志留存天数)、max_report_day(统计数据留存天数)、max_stat_day(防护报告留存天数)。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置", "required": True},
            {"name": "max_day", "type": "integer", "description": "防护日志留存天数", "required": False},
            {"name": "max_report_day", "type": "integer", "description": "统计数据留存天数", "required": False},
            {"name": "max_stat_day", "type": "integer", "description": "防护报告留存天数", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/global/log_clean')
            return {'success': True, 'data': result.get('data', {})}
        else:
            config = {}
            for key in ['max_day', 'max_report_day', 'max_stat_day']:
                if key in kwargs and kwargs[key] is not None:
                    config[key] = kwargs[key]
            self.client.post('/open/global/log_clean', data=config)
            return {'success': True, 'message': f'日志清理配置已更新'}


class ManageApiTokenTool(WAFTool):
    """API Token管理工具"""

    @property
    def name(self) -> str:
        return "manage_api_token"

    @property
    def description(self) -> str:
        return "管理API Token(GET/PUT/DELETE /open/auth/token)。action=get获取，update更新，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取Token，update=更新Token，delete=删除Token", "required": True}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/auth/token')
            return {'success': True, 'data': result.get('data', {})}
        elif action == 'update':
            result = self.client.put('/open/auth/token')
            return {'success': True, 'data': result.get('data', {}), 'message': 'API Token已更新'}
        elif action == 'delete':
            self.client.delete('/open/auth/token')
            return {'success': True, 'message': 'API Token已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageGlobalProxyTool(WAFTool):
    """全局代理设置工具"""

    @property
    def name(self) -> str:
        return "manage_global_proxy"

    @property
    def description(self) -> str:
        return "管理全局代理设置(GET/PUT /open/global/proxy)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取代理设置，set=设置代理", "required": True},
            {"name": "proxy_data", "type": "object", "description": "代理配置对象，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/global/proxy')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.put('/open/global/proxy', data=kwargs.get('proxy_data', {}))
            return {'success': True, 'message': '全局代理设置已更新'}


class ManageSyslogTool(WAFTool):
    """Syslog配置工具"""

    @property
    def name(self) -> str:
        return "manage_syslog"

    @property
    def description(self) -> str:
        return "管理Syslog配置(GET/PUT /commercial/syslog)。action=get获取，set设置，test测试连接。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置，test=测试连接", "required": True},
            {"name": "config", "type": "object", "description": "Syslog配置对象，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/commercial/syslog')
            return {'success': True, 'data': result.get('data', {})}
        elif action == 'set':
            self.client.put('/commercial/syslog', data=kwargs.get('config', {}))
            return {'success': True, 'message': 'Syslog配置已更新'}
        elif action == 'test':
            result = self.client.post('/commercial/syslog/test', data=kwargs.get('config', {}))
            return {'success': True, 'data': result.get('data', {}), 'message': 'Syslog测试已发送'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageJA4Tool(WAFTool):
    """JA4指纹配置工具"""

    @property
    def name(self) -> str:
        return "manage_ja4"

    @property
    def description(self) -> str:
        return "管理JA4指纹配置(GET/PUT /open/ja4)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置", "required": True},
            {"name": "config", "type": "object", "description": "JA4配置对象，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/ja4')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.put('/open/ja4', data=kwargs.get('config', {}))
            return {'success': True, 'message': 'JA4配置已更新'}


class ManageSecurityPostureTool(WAFTool):
    """安全态势工具"""

    @property
    def name(self) -> str:
        return "manage_security_posture"

    @property
    def description(self) -> str:
        return "管理安全态势(GET/POST /open/security_posture/*)。action=realtime实时数据，statistics统计数据，trends趋势，set_site设置站点态势。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "realtime/statistics/trends/set_site", "required": True},
            {"name": "site_data", "type": "object", "description": "站点态势配置，action=set_site时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'set_site':
            self.client.post('/open/security_posture/site', data=kwargs.get('site_data', {}))
            return {'success': True, 'message': '站点安全态势配置已更新'}
        else:
            result = self.client.get(f'/open/security_posture/{action}')
            return {'success': True, 'action': action, 'data': result.get('data', {})}


class ManageWaitingRoomTool(WAFTool):
    """等候室工具"""

    @property
    def name(self) -> str:
        return "manage_waiting_room"

    @property
    def description(self) -> str:
        return "管理等候室配置和日志(GET/POST /open/site/{id}/waiting)。action=get获取配置，set设置配置，logs查看日志。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置，logs=查看日志", "required": True},
            {"name": "config", "type": "object", "description": "等候室配置，action=set时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'logs':
            result = self.client.get('/open/records/waiting')
            return {'success': True, 'data': result.get('data', {})}
        elif action == 'get':
            result = self.client.get(f'/open/site/{site_id}/waiting')
            return {'success': True, 'site_id': site_id, 'config': result.get('data', {})}
        else:
            self.client.post(f'/open/site/{site_id}/waiting', data=kwargs.get('config', {}))
            return {'success': True, 'site_id': site_id, 'message': f'站点{site_id}等候室配置已更新'}


class ManagePortalTool(WAFTool):
    """门户配置工具"""

    @property
    def name(self) -> str:
        return "manage_portal"

    @property
    def description(self) -> str:
        return "管理门户配置(GET/PUT /open/portal等)。action=get获取配置，set设置配置，get_proxy获取代理配置，set_proxy设置代理配置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get/set/get_proxy/set_proxy/get_style/set_style", "required": True},
            {"name": "config", "type": "object", "description": "配置对象，set类操作时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/portal')
            return {'success': True, 'config': result.get('data', {})}
        elif action == 'set':
            self.client.put('/open/portal', data=kwargs.get('config', {}))
            return {'success': True, 'message': '门户配置已更新'}
        elif action == 'get_proxy':
            result = self.client.get('/open/portal/proxy_config')
            return {'success': True, 'config': result.get('data', {})}
        elif action == 'set_proxy':
            self.client.put('/open/portal/proxy_config', data=kwargs.get('config', {}))
            return {'success': True, 'message': '门户代理配置已更新'}
        elif action == 'get_style':
            result = self.client.get('/open/portal/style')
            return {'success': True, 'style': result.get('data', {})}
        elif action == 'set_style':
            self.client.put('/commercial/portal/style', data=kwargs.get('config', {}))
            return {'success': True, 'message': '门户样式已更新'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageUsersTool(WAFTool):
    """用户增删改工具"""

    @property
    def name(self) -> str:
        return "manage_users"

    @property
    def description(self) -> str:
        return "管理控制台用户(POST/PUT/DELETE /open/users)。action=create创建，update更新，delete删除，reset_totp重置二次认证。创建用户时user_data需包含：username(用户名)、password(密码)、role(角色：1=管理员，2=操作员，3=配置员，4=审计员)、tfa_enabled(是否启用二次认证，布尔值)。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "create/update/delete/reset_totp", "required": True},
            {"name": "user_id", "type": "integer", "description": "用户ID，update/delete/reset_totp时必传", "required": False},
            {"name": "user_data", "type": "object", "description": "用户数据。创建时需含：username(字符串)、password(字符串)、role(整数：1=管理员,2=操作员,3=配置员,4=审计员)、tfa_enabled(布尔值)。更新时需含id和要修改的字段。", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'create':
            user_data = kwargs.get('user_data', {})
            try:
                result = self.client.post('/open/users', data=user_data)
                return {'success': True, 'user_id': result.get('data'), 'message': f'用户"{user_data.get("username")}"已创建'}
            except Exception as e:
                return {'success': False, 'error': str(e), 'message': f'用户创建失败：{e}'}
        elif action == 'update':
            try:
                self.client.put('/open/users', data=kwargs.get('user_data', {}))
                return {'success': True, 'message': '用户已更新'}
            except Exception as e:
                return {'success': False, 'error': str(e), 'message': f'用户更新失败：{e}'}
        elif action == 'delete':
            try:
                self.client.delete('/open/users', data=kwargs.get('user_data', {}))
                return {'success': True, 'message': '用户已删除'}
            except Exception as e:
                return {'success': False, 'error': str(e), 'message': f'用户删除失败：{e}'}
        elif action == 'reset_totp':
            user_id = kwargs.get('user_id')
            try:
                self.client.post(f'/open/users/{user_id}/totp')
                return {'success': True, 'message': f'用户{user_id}二次认证已重置'}
            except Exception as e:
                return {'success': False, 'error': str(e), 'message': f'重置二次认证失败：{e}'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageCertsTool(WAFTool):
    """证书增删工具"""

    @property
    def name(self) -> str:
        return "manage_certs"

    @property
    def description(self) -> str:
        return "管理SSL证书(POST /open/cert创建/更新，DELETE /open/cert/{id}删除)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "upsert=创建或更新证书，delete=删除证书", "required": True},
            {"name": "cert_id", "type": "integer", "description": "证书ID，delete时必传", "required": False},
            {"name": "cert_data", "type": "object", "description": "证书数据，upsert时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'upsert':
            result = self.client.post('/open/cert', data=kwargs.get('cert_data', {}))
            return {'success': True, 'cert_id': result.get('data'), 'message': '证书已创建/更新'}
        elif action == 'delete':
            cert_id = kwargs.get('cert_id')
            self.client.delete(f'/open/cert/{cert_id}')
            return {'success': True, 'message': f'证书{cert_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageForwardingRulesTool(WAFTool):
    """转发规则管理工具"""

    @property
    def name(self) -> str:
        return "manage_forwarding_rules"

    @property
    def description(self) -> str:
        return "管理站点转发规则(GET/POST/PUT/DELETE /open/site/{id}/forwarding_rules)。action=list/create/update/delete/switch。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "site_id", "type": "integer", "description": "站点ID", "required": True},
            {"name": "action", "type": "string", "description": "list/create/update/delete/switch", "required": True},
            {"name": "rule_id", "type": "integer", "description": "规则ID，update/delete/switch时必传", "required": False},
            {"name": "rule_data", "type": "object", "description": "规则数据，create/update时必传", "required": False}
        ]

    def execute(self, site_id: int, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get(f'/open/site/{site_id}/forwarding_rules')
            return {'success': True, 'rules': result.get('data', {})}
        elif action == 'create':
            result = self.client.post(f'/open/site/{site_id}/forwarding_rules', data=kwargs.get('rule_data', {}))
            return {'success': True, 'rule_id': result.get('data'), 'message': '转发规则已创建'}
        elif action == 'update':
            rule_id = kwargs.get('rule_id')
            self.client.put(f'/open/site/{site_id}/forwarding_rules/{rule_id}', data=kwargs.get('rule_data', {}))
            return {'success': True, 'message': f'转发规则{rule_id}已更新'}
        elif action == 'delete':
            rule_id = kwargs.get('rule_id')
            self.client.delete(f'/open/site/{site_id}/forwarding_rules/{rule_id}')
            return {'success': True, 'message': f'转发规则{rule_id}已删除'}
        elif action == 'switch':
            rule_id = kwargs.get('rule_id')
            self.client.put(f'/open/site/{site_id}/forwarding_rules/{rule_id}/switch', data=kwargs.get('rule_data', {}))
            return {'success': True, 'message': f'转发规则{rule_id}状态已切换'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class ManageCloudPoliciesTool(WAFTool):
    """云策略工具"""

    @property
    def name(self) -> str:
        return "manage_cloud_policies"

    @property
    def description(self) -> str:
        return "管理云策略(GET /open/cloud/policies列出，POST /open/cloud/policies/subscribe订阅)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "list=列出云策略，subscribe=订阅云策略", "required": True},
            {"name": "subscribe_data", "type": "object", "description": "订阅数据，subscribe时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get('/open/cloud/policies')
            return {'success': True, 'policies': result.get('data', {})}
        else:
            self.client.post('/open/cloud/policies/subscribe', data=kwargs.get('subscribe_data', {}))
            return {'success': True, 'message': '云策略已订阅'}


class GetBlockingMessageTool(WAFTool):
    """拦截页面消息工具"""

    @property
    def name(self) -> str:
        return "get_blocking_message"

    @property
    def description(self) -> str:
        return "获取或更新拦截页面消息(GET/PUT /ManagerInfo)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取消息，set=设置消息", "required": True},
            {"name": "message_data", "type": "object", "description": "消息数据，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/ManagerInfo')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.put('/ManagerInfo', data=kwargs.get('message_data', {}))
            return {'success': True, 'message': '拦截页面消息已更新'}


class ManageIntelligenceTool(WAFTool):
    """恶意IP情报工具"""

    @property
    def name(self) -> str:
        return "manage_intelligence"

    @property
    def description(self) -> str:
        return "管理恶意IP情报共享(GET/POST /open/intelligence)。action=get获取配置，set设置配置，update_ip_lib更新IP库。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取配置，set=设置配置，update_ip_lib=更新IP库", "required": True},
            {"name": "config", "type": "object", "description": "配置数据，set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/intelligence')
            return {'success': True, 'data': result.get('data', {})}
        elif action == 'set':
            self.client.post('/open/intelligence', data=kwargs.get('config', {}))
            return {'success': True, 'message': '恶意IP情报配置已更新'}
        elif action == 'update_ip_lib':
            self.client.post('/open/intelligence/ip_lib', data=kwargs.get('config', {}))
            return {'success': True, 'message': '恶意IP库已更新'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class GetAuditLogTool(WAFTool):
    """审计日志工具"""

    @property
    def name(self) -> str:
        return "get_audit_log"

    @property
    def description(self) -> str:
        return "查询审计日志(GET /business/audit_log)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get('/open/audit_log', params=params)
        return {'success': True, 'data': result.get('data', {})}


class ManageReportTool(WAFTool):
    """报告管理工具"""

    @property
    def name(self) -> str:
        return "manage_report"

    @property
    def description(self) -> str:
        return "管理报告(GET/POST/DELETE /business/report)。action=list列出，create创建，detail详情，delete删除。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "list/create/detail/delete", "required": True},
            {"name": "report_id", "type": "integer", "description": "报告ID，detail/delete时必传", "required": False},
            {"name": "report_data", "type": "object", "description": "报告数据，create时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'list':
            result = self.client.get('/business/report')
            return {'success': True, 'reports': result.get('data', {})}
        elif action == 'create':
            result = self.client.post('/business/report', data=kwargs.get('report_data', {}))
            return {'success': True, 'report_id': result.get('data'), 'message': '报告已创建'}
        elif action == 'detail':
            report_id = kwargs.get('report_id')
            result = self.client.get(f'/business/report/{report_id}')
            return {'success': True, 'report': result.get('data', {})}
        elif action == 'delete':
            report_id = kwargs.get('report_id')
            self.client.delete(f'/business/report/{report_id}')
            return {'success': True, 'message': f'报告{report_id}已删除'}
        else:
            return {'success': False, 'error': f'未知操作: {action}'}


class GetAntiBotLogsTool(WAFTool):
    """人机验证日志工具"""

    @property
    def name(self) -> str:
        return "get_anti_bot_logs"

    @property
    def description(self) -> str:
        return "查询人机验证日志(GET /open/records/challenge)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get('/open/records/challenge', params=params)
        return {'success': True, 'data': result.get('data', {})}


class ManageNetworkProxyTool(WAFTool):
    """系统网络代理工具"""

    @property
    def name(self) -> str:
        return "manage_network_proxy"

    @property
    def description(self) -> str:
        return "管理系统网络代理(GET/PUT /open/system/network_proxy)。action=get获取，set设置。"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "action", "type": "string", "description": "get=获取代理配置，set=设置代理配置", "required": True},
            {"name": "config", "type": "object", "description": "代理配置对象，action=set时必传", "required": False}
        ]

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        if action == 'get':
            result = self.client.get('/open/system/network_proxy')
            return {'success': True, 'data': result.get('data', {})}
        else:
            self.client.put('/open/system/network_proxy', data=kwargs.get('config', {}))
            return {'success': True, 'message': '系统网络代理已更新'}


class GetRuleAttackLogsTool(WAFTool):
    """规则攻击日志工具"""

    @property
    def name(self) -> str:
        return "get_rule_attack_logs"

    @property
    def description(self) -> str:
        return "查询黑白名单规则相关的攻击日志(GET /open/records/rule)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "page", "type": "integer", "description": "页码", "required": False},
            {"name": "page_size", "type": "integer", "description": "每页数量", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['page', 'page_size']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get('/open/records/rule', params=params)
        return {'success': True, 'data': result.get('data', {})}


class ExportAttackLogsTool(WAFTool):
    """导出攻击日志工具"""

    @property
    def name(self) -> str:
        return "export_attack_logs"

    @property
    def description(self) -> str:
        return "导出攻击日志(GET /commercial/record/export)"

    @property
    def parameters(self) -> List[Dict]:
        return [
            {"name": "start", "type": "integer", "description": "开始时间戳", "required": False},
            {"name": "end", "type": "integer", "description": "结束时间戳", "required": False}
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        params = {}
        for key in ['start', 'end']:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        result = self.client.get('/commercial/record/export', params=params)
        return {'success': True, 'data': result.get('data', {})}

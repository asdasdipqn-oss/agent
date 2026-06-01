已学习完成。Swagger API JSON文档已保存到本地：waf_swagger.json

以下是我对API文档的理解总结：

## 已实现的工具（21个）
- 日志查询：query_attack_logs, get_attack_log_detail, query_attack_events, query_qps
- 规则管理：list_custom_rules, create_custom_rule, toggle_rule, delete_rule
- IP组：list_ip_groups, add_ip_to_group
- 站点管理：list_sites, get_site_detail, health_check, set_site_mode
- 防护配置：set_challenge, set_auth_defense, set_rate_limit, get_acl_logs, release_acl_block
- 证书管理：list_certs, get_cert_detail

## 待实现的API（按功能分类）

### 系统与授权
- GET /open/system - 系统信息（版本、机器ID等）
- GET /open/system/authorize - 授权信息（组织名、到期时间、版本等）
- POST /open/system/authorize - 申请授权
- GET /open/system/edition - 系统版本（社区版/专业版/企业版）
- GET /open/system/arch - 系统架构
- GET /open/system/key - 系统key
- GET /open/detector - 检测引擎性能模式
- POST /open/detector - 更新检测引擎性能模式
- GET /open/global/log_clean - 日志清理间隔
- POST /open/global/log_clean - 更新日志清理间隔

### 告警
- GET /alarm - 获取告警配置（钉钉/飞书/企微/Telegram/Discord）
- PUT /alarm - 更新告警配置
- POST /alarm/test - 测试告警

### 统计
- GET /stat/advance/access - 访问统计（PV/UV/IP/Session）
- GET /stat/advance/attack - 攻击统计（拦截数/攻击IP等）
- GET /stat/advance/client - 客户端统计（浏览器/OS）
- GET /stat/advance/domain - 域名统计
- GET /stat/advance/location - 地理位置统计
- GET /stat/advance/page - 页面统计
- GET /stat/advance/status_code - 状态码统计
- GET /stat/advance/error_status_code - 错误状态码统计
- GET /stat/advance/trend/access - 访问趋势
- GET /stat/advance/trend/intercept - 拦截趋势

### 规则管理（扩展）
- GET /open/policy/detail?id=X - 规则详情
- PUT /open/policy - 更新规则
- PUT /open/policy/order - 规则排序
- PUT /open/policy/order/value - 按值排序
- PUT /open/policy/orders - 批量排序

### IP组管理（扩展）
- POST /open/ipgroup - 创建IP组
- PUT /open/ipgroup - 更新IP组
- DELETE /open/ipgroup - 删除IP组
- GET /open/ipgroup/detail?id=X - IP组详情（含IP列表）
- GET /open/ipgroup/crawler - 搜索引擎蜘蛛组
- POST /open/ipgroup/crawler - 更新搜索引擎蜘蛛
- GET /open/ipgroup/link - 通过链接获取IP
- POST /open/ipgroup/link - 通过链接创建IP组

### 站点管理（扩展）
- POST /open/site - 创建站点
- PUT /open/site - 更新站点
- DELETE /open/site - 删除站点
- PUT /open/site/{id}/basic_info - 更新站点基本信息
- PUT /open/site/{id}/group - 更新站点分组
- GET /open/site/{id}/proxy - 获取站点代理配置
- PUT /open/site/{id}/proxy - 设置站点代理配置
- GET /open/site/{id}/nginx_config - 获取Nginx配置
- PUT /open/site/{id}/nginx_config - 更新Nginx配置
- GET /open/site/{id}/resources - 获取站点路由资源
- GET/POST /open/site/{id}/excludes - 路由采集排除配置
- GET/POST/PUT /open/site/{id}/log - 访问/错误日志
- GET /open/site/group - 站点分组列表
- POST /open/site/group - 创建分组
- PUT /open/site/group/{id} - 更新分组
- DELETE /open/site/group/{id} - 删除分组
- GET /open/site/group/switch - 获取分组开关
- PUT /commercial/site/group/switch - 设置分组开关

### 证书管理（扩展）
- POST /open/cert - 创建/更新证书
- DELETE /open/cert/{id} - 删除证书

### 认证防护（扩展）
- GET /open/auth_defense/source - 认证源列表
- POST /open/auth_defense/source - 创建认证源
- GET /open/auth_defense/source/{id} - 认证源详情
- PUT /open/auth_defense/source/{id} - 更新认证源
- DELETE /open/auth_defense/source/{id} - 删除认证源
- GET /open/auth_defense/source/{id}/user - 认证源用户列表
- GET /open/auth_defense/user - 认证防护用户列表
- POST /open/auth_defense/user - 创建认证防护用户
- PUT /open/auth_defense/user/{user_id} - 更新用户
- DELETE /open/auth_defense/user/{user_id} - 删除用户
- GET /open/records/auth_defense - 认证防护日志
- GET /open/v2/records/auth_defense - 认证防护日志V2

### 反爬虫/Challenge（扩展）
- GET /open/challenge/config - 全局反爬虫配置
- POST /open/challenge/config - 设置全局反爬虫配置
- GET /open/records/challenge - 反爬虫日志

### 频率限制（扩展）
- GET /open/site/{id}/acl - 获取站点频率限制配置
- POST /open/site/{id}/acl - 添加频率限制规则
- PUT /open/site/{id}/acl/{rule_id} - 更新频率限制规则
- DELETE /open/site/{id}/acl/{rule_id} - 删除频率限制规则

### 反篡改
- GET /business/site/{site_id}/anti_tamper - 反篡改规则列表
- POST /business/site/{site_id}/anti_tamper - 创建反篡改规则
- GET /business/anti_tamper/{id} - 反篡改规则详情
- PUT /business/anti_tamper/{id} - 更新反篡改规则
- DELETE /business/anti_tamper/{id} - 删除反篡改规则
- PUT /business/site/{site_id}/anti_tamper - 刷新反篡改

### 动态防护
- GET /open/site/{id}/chaos - 获取动态防护配置
- POST /open/site/{id}/chaos - 创建/设置动态防护

### Skynet语义规则
- GET /open/global/mode - 全局语义配置
- PUT /open/global/mode - 更新全局语义配置
- GET /open/site/{id}/semantics - 站点语义配置
- PUT /open/site/{id}/semantics - 设置站点语义配置
- GET /open/skynet/rule - 增强规则列表
- PUT /open/skynet/rule - 更新增强规则
- GET /open/skynet/rule/switch - 增强规则全局开关
- PUT /open/skynet/rule/switch - 更新增强规则全局开关

### JA4指纹
- GET /open/ja4 - JA4指纹配置
- PUT /open/ja4 - 设置JA4配置

### 等候室
- GET /open/site/{id}/waiting - 获取等候室配置
- POST /open/site/{id}/waiting - 设置等候室配置
- GET /open/records/waiting - 等候室日志

### 安全态势
- GET /open/security_posture/realtime - 实时安全态势
- GET /open/security_posture/statistics - 安全态势统计
- GET /open/security_posture/trends - 安全态势趋势
- POST /open/security_posture/site - 设置站点安全态势

### 转发规则
- GET /open/site/{id}/forwarding_rules - 转发规则列表
- POST /open/site/{id}/forwarding_rules - 创建转发规则
- PUT /open/site/{id}/forwarding_rules/{rule_id} - 编辑转发规则
- DELETE /open/site/{id}/forwarding_rules/{rule_id} - 删除转发规则
- PUT /open/site/{id}/forwarding_rules/{rule_id}/switch - 启停转发规则

### 门户
- GET /open/portal - 门户配置
- PUT /open/portal - 更新门户配置
- GET /open/portal/style - 门户样式
- PUT /commercial/portal/style - 更新门户样式

### 用户管理
- GET /open/users - 用户列表
- PUT /open/users - 更新用户
- POST /open/users - 创建用户
- DELETE /open/users - 删除用户
- POST /open/users/{id}/totp - 重置用户TOTP

### 全局设置
- GET /open/global/proxy - 全局代理设置
- PUT /open/global/proxy - 更新全局代理设置
- GET /open/intelligence - 恶意IP情报共享
- POST /open/intelligence - 更新恶意IP情报共享
- GET /ManagerInfo - 拦截页面消息
- PUT /ManagerInfo - 更新拦截页面消息

### 日志与报告
- GET /business/audit_log - 审计日志
- GET /business/report - 报告列表
- POST /business/report - 创建报告
- GET /business/report/{id} - 报告详情
- DELETE /business/report/{id} - 删除报告
- GET /commercial/record/export - 导出攻击日志

### 云策略
- GET /open/cloud/policies - 云策略列表
- POST /open/cloud/policies/subscribe - 订阅云策略

### MCP
- GET /mcp - 获取MCP配置
- POST /mcp - 设置MCP配置

### Syslog
- GET /commercial/syslog - Syslog配置
- PUT /commercial/syslog - 更新Syslog配置
- POST /commercial/syslog/test - 测试Syslog

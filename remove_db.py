#!/usr/bin/env python3
"""
Remove all database-related code from app.py
"""
import re

# Read the file
with open('/Users/wangqizhi/agent2/app.py', 'r') as f:
    lines = f.readlines()

output_lines = []
skip_until = None
skip_level = 0
in_function = False
function_to_remove = ['init_db', 'get_db_connection', 'sync_db_to_file']
current_function = None
indent_stack = []

for i, line in enumerate(lines):
    stripped = line.lstrip()

    # Skip DB_CONFIG block
    if 'DB_CONFIG = {' in line:
        continue
    if stripped.startswith('FEEDBACK_FILE'):
        output_lines.append(line)
        continue
    if stripped.startswith(('host', 'port', 'user', 'password', 'database', 'charset')):
        continue
    if stripped.startswith('}'):
        continue

    # Skip MYSQL_AVAILABLE
    if 'MYSQL_AVAILABLE' in line and '=' in line:
        continue

    # Skip init_db, get_db_connection, sync_db_to_file functions
    if any(f'def {func}' in stripped for func in function_to_remove):
        in_function = True
        current_function = stripped.split('(')[0].replace('def ', '').strip()
        indent_stack.append(len(line) - len(stripped))
        continue

    if in_function:
        current_indent = len(line) - len(stripped)
        # If we're back to or below the original function indent, we're done
        if current_indent <= min(indent_stack) if indent_stack else 0:
            in_function = False
            current_function = None
            indent_stack = []
            output_lines.append(line)
        else:
            continue

    # Skip init_db() call in the main block
    if 'init_db()' in line:
        continue
    if 'print(f"数据库已连接' in line:
        continue
    if 'print(f"数据库初始化失败' in line:
        continue

    # Skip sync_db_to_file() calls
    if 'sync_db_to_file()' in line:
        continue

    # Skip get_db_connection() and related database code
    if 'get_db_connection()' in line:
        continue
    if 'pymysql.cursors.DictCursor' in line:
        continue
    if 'conn = get_db_connection()' in line:
        continue
    if 'with conn.cursor(pymysql.cursors.DictCursor)' in line:
        continue
    if stripped.startswith('# MySQL') or stripped.startswith('# 本地文件'):
        continue

    output_lines.append(line)

# Write back
with open('/Users/wangqizhi/agent2/app.py', 'w') as f:
    f.writelines(output_lines)

print(f"Removed database code. Original: {len(lines)} lines, New: {len(output_lines)} lines")

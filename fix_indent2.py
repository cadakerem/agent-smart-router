# -*- coding: utf-8 -*-
cli_path = r'C:\Users\kbarb\Documents\GitHub\llm-proxy-cli\llm_proxy_cli.py'
with open(cli_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'sys.stderr.flush()' in line and i > 400 and i < 450:
        if line.startswith('                                sys.stderr.flush()'):
            lines[i] = '                            sys.stderr.flush()\n'

with open(cli_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
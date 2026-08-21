import re
import os

def remove_comments(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    in_multiline = False
    
    for line in lines:
        if in_multiline:
            if '"""' in line or "'''" in line:
                in_multiline = False
            continue
        
        if '"""' in line or "'''" in line:
            count = line.count('"""') + line.count("'''")
            if count % 2 != 0:
                in_multiline = True
            
            line = re.sub(r'"""[\s\S]*?"""', '', line)
            line = re.sub(r"'''[\s\S]*?'''", '', line)
        if not in_multiline:
            line = re.sub(r'#.*$', '', line)
        new_lines.append(line)
    new_content = '\n'.join(new_lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
files = [
    'd:/main/CMKA/main.py',
    'd:/main/CMKA/core/config.py',
    'd:/main/CMKA/core/detector.py',
    'd:/main/CMKA/core/stats.py',
    'd:/main/CMKA/core/tracker.py',
    'd:/main/CMKA/core/utils.py',
    'd:/main/CMKA/ui/dialogs.py',
    'd:/main/CMKA/ui/main_window.py',
]

for filepath in files:
    if os.path.exists(filepath):
        remove_comments(filepath)
        print(f"Processed: {filepath}")
#!/usr/bin/env python3
# list_directory.py

import os

def list_files(startpath='.'):
    lines = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        lines.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            lines.append(f'{subindent}{f}')
    return lines

def main():
    start = os.getcwd()
    print(f'Gerando inventário de: {start}\n')
    lines = list_files(start)
    for line in lines:
        print(line)
    # salva também em arquivo
    with open('file_list.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(lines))
    print('\nInventário salvo em file_list.txt')

if __name__ == '__main__':
    main()

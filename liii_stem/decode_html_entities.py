#!/usr/bin/env python3
"""
将 HTML 文件中的 &#xXXXX; (十六进制) 和 &#DDDD; (十进制) 数字字符引用
转换为可读的 Unicode 中文字符。

用法:
    python3 decode_html_entities.py <输入文件> [输出文件]

    - 不指定输出文件时，自动备份原文件为 .bak 并原地覆盖
    - 支持批量: python3 decode_html_entities.py *.html
"""

import re
import sys
import os
import shutil


def decode_entities(text: str) -> str:
    """将 &#xXXXX; 和 &#DDDD; 转成实际 Unicode 字符"""

    def hex_replace(m: re.Match) -> str:
        return chr(int(m.group(1), 16))

    def dec_replace(m: re.Match) -> str:
        return chr(int(m.group(1)))

    # 先处理十六进制 &#x...;
    text = re.sub(r"&#x([0-9a-fA-F]+);", hex_replace, text)
    # 再处理十进制 &#...;
    text = re.sub(r"&#(\d+);", dec_replace, text)
    return text


def process_file(input_path: str, output_path: str | None = None):
    with open(input_path, "r", encoding="utf-8") as f:
        original = f.read()

    decoded = decode_entities(original)

    if decoded == original:
        print(f"  [跳过] {input_path} — 无需转换")
        return

    if output_path is None:
        # 备份原文件
        bak = input_path + ".bak"
        shutil.copy2(input_path, bak)
        print(f"  备份 → {bak}")
        output_path = input_path

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded)

    # 统计转换数量
    hex_count = len(re.findall(r"&#x[0-9a-fA-F]+;", original))
    dec_count = len(re.findall(r"&#\d+;", original))
    print(f"  [完成] {input_path} → {output_path}  (转换 {hex_count + dec_count} 处: {hex_count} hex + {dec_count} dec)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]
    # 如果只有两个参数且第二个不是 glob 匹配到的文件，视为 [输入, 输出]
    if len(args) == 2 and not os.path.isfile(args[0]):
        print(f"错误: 文件不存在 — {args[0]}")
        sys.exit(1)

    if len(args) == 2 and not os.path.isdir(args[1]):
        process_file(args[0], args[1])
    else:
        for path in args:
            if os.path.isfile(path):
                process_file(path)
            else:
                print(f"  [忽略] {path} — 不是文件")


if __name__ == "__main__":
    main()

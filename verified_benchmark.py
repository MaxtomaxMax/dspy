import json
import subprocess
import re
import sys
import os
from pathlib import Path

# ================= 配置区 =================
JSON_FILE_PATH = "DSPy_Multihop_QA.json"  # 你生成的 Q&A JSON 文件路径
REPO_PATH = "./dspy"                    # 锁定的 DSPy 源码库路径
# ==========================================

class BenchmarkVerifier:
    def __init__(self, json_path: str, repo_path: str):
        self.json_path = Path(json_path)
        self.repo_path = Path(repo_path)
        self._check_environment()

    def _check_environment(self):
        """检查所需的文件和工具是否存在"""
        if not self.json_path.exists():
            print(f"❌ 找不到 JSON 文件: {self.json_path}")
            sys.exit(1)
        if not self.repo_path.exists():
            print(f"❌ 找不到源码仓库: {self.repo_path}")
            sys.exit(1)
            
        try:
            # 检查 ripgrep 是否安装
            subprocess.run(["rg", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ 未检测到 ripgrep (rg) 命令，请先安装。")
            sys.exit(1)

    def extract_file_paths(self, text: str) -> list:
        """从 retrieval_path 字符串中正则提取 .py 文件路径"""
        # 匹配类似于 dspy/teleprompt/bootstrap.py 的路径
        pattern = r"([a-zA-Z0-9_/\-]+\.py)"
        return re.findall(pattern, text)

    def run_rg_search(self, keyword: str, target_file: str = ""):
        """调用系统底层的 ripgrep 进行精确匹配"""
        # rg 参数：-n 显示行号, -C 2 显示上下2行上下文, --color always 保持高亮
        cmd = ["rg", "-n", "-C", "2", "--color", "always", "-F", keyword]
        
        if target_file:
            full_target = self.repo_path / target_file
            if full_target.exists():
                cmd.append(str(full_target))
            else:
                print(f"\n⚠️ 警告: 文件 {target_file} 在仓库中不存在，将在全库中搜索。")
                cmd.append(str(self.repo_path))
        else:
            cmd.append(str(self.repo_path))

        try:
            # 执行命令并实时输出
            print(f"\n> 正在执行: {' '.join(cmd)}")
            print("-" * 50)
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.stdout:
                print(result.stdout)
            else:
                print(f"📭 在目标范围内未找到包含 '{keyword}' 的匹配项。")
            print("-" * 50)
        except Exception as e:
            print(f"❌ 执行 rg 时发生错误: {e}")

    def start_interactive_session(self):
        """启动交互式验证流程"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n🚀 开始验证，共加载 {len(data)} 道测试题。")
        
        for index, item in enumerate(data):
            print(f"\n{'='*20} 题目 {index + 1}/{len(data)} {'='*20}")
            print(f"❓ 问题: {item.get('question')}")
            print(f"💡 答案: {item.get('answer')}")
            print("\n🗺️  检索路径 (Retrieval Paths):")
            
            paths = item.get('retrieval_path', [])
            for i, path in enumerate(paths):
                print(f"  [{i+1}] {path}")

            # 提取当前题目涉及的所有可能的文件路径
            extracted_files = []
            for path_str in paths:
                extracted_files.extend(self.extract_file_paths(path_str))
            extracted_files = list(set(extracted_files)) # 去重

            while True:
                print("\n操作选项:")
                print("  输入关键词 -> 在提取的文件中进行 rg 精确匹配 (如: _compiled)")
                print("  输入 'n'   -> 跳过，验证下一题")
                print("  输入 'q'   -> 退出脚本")
                
                user_input = input("\n请输入指令或关键词: ").strip()
                
                if user_input.lower() == 'q':
                    print("退出验证。")
                    return
                elif user_input.lower() == 'n':
                    break
                elif user_input:
                    # 如果有明确的文件路径，则在第一个文件中搜索（简化操作），否则全库搜索
                    target = extracted_files[0] if extracted_files else ""
                    if len(extracted_files) > 1:
                        print(f"⚠️ 提示: 检测到多个文件路径，默认在 {target} 中搜索。")
                    self.run_rg_search(keyword=user_input, target_file=target)

if __name__ == "__main__":
    verifier = BenchmarkVerifier(JSON_FILE_PATH, REPO_PATH)
    verifier.start_interactive_session()
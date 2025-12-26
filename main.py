#!/usr/bin/env python3
"""
CookMoney - AI-Powered Investment Research Analysis Tool
投资研报智能分析工具

使用方法:
    python main.py analyze <file1> <file2> ... [--output <name>]
    python main.py analyze reports/uploaded/*.pdf
"""

import sys
import argparse
from pathlib import Path

from src.analyzer import AIAnalyzer
from src.config import UPLOADED_DIR, ANALYSIS_DIR


def analyze_command(args):
    """Execute analysis command"""
    # Get file paths
    file_paths = []
    for file_pattern in args.files:
        path = Path(file_pattern)
        if path.is_file():
            file_paths.append(path)
        elif "*" in str(file_pattern):
            # Handle glob patterns
            parent = path.parent if path.parent.exists() else Path(".")
            pattern = path.name
            file_paths.extend(parent.glob(pattern))
        else:
            print(f"警告: 文件不存在或无法访问: {file_pattern}")
    
    if not file_paths:
        print("错误: 没有找到要分析的文件")
        return 1
    
    print(f"找到 {len(file_paths)} 个文件待分析:")
    for fp in file_paths:
        print(f"  - {fp}")
    
    # Analyze
    print("\n正在分析研报...")
    analyzer = AIAnalyzer()
    result = analyzer.analyze_reports(file_paths)
    
    # Save result
    output_path = analyzer.save_analysis(result, args.output)
    
    print(f"\n✓ 分析完成!")
    print(f"结果已保存到:")
    print(f"  - JSON: {output_path}")
    print(f"  - Markdown: {output_path.with_suffix('.md')}")
    
    # Display analysis
    if "error" in result:
        print(f"\n错误: {result['error']}")
        return 1
    else:
        print(f"\n{'='*60}")
        print("投资建议摘要:")
        print('='*60)
        print(result['analysis'])
        print('='*60)
    
    return 0


def list_command(args):
    """List uploaded files and analyses"""
    print("上传的研报文件:")
    print("-" * 60)
    uploaded_files = list(UPLOADED_DIR.glob("*"))
    uploaded_files = [f for f in uploaded_files if f.is_file() and f.name != ".gitkeep"]
    if uploaded_files:
        for f in sorted(uploaded_files):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
    else:
        print("  (无文件)")
    
    print("\n已生成的分析报告:")
    print("-" * 60)
    analysis_files = list(ANALYSIS_DIR.glob("*.md"))
    if analysis_files:
        for f in sorted(analysis_files, reverse=True):
            print(f"  {f.name}")
    else:
        print("  (无报告)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CookMoney - AI投资研报分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文件
  python main.py analyze reports/uploaded/report.pdf
  
  # 分析多个文件
  python main.py analyze reports/uploaded/report1.pdf reports/uploaded/chart.png
  
  # 使用通配符分析所有PDF
  python main.py analyze "reports/uploaded/*.pdf"
  
  # 指定输出文件名
  python main.py analyze reports/uploaded/report.pdf --output my_analysis
  
  # 列出所有文件和报告
  python main.py list
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="分析研报文件")
    analyze_parser.add_argument("files", nargs="+", help="要分析的文件路径（支持通配符）")
    analyze_parser.add_argument("--output", "-o", help="输出文件名（不含扩展名）")
    
    # List command
    list_parser = subparsers.add_parser("list", help="列出所有文件和分析报告")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "analyze":
        return analyze_command(args)
    elif args.command == "list":
        return list_command(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

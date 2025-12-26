#!/usr/bin/env python3
"""
Demo script showing how to use CookMoney without actual API calls
演示脚本：展示如何使用CookMoney（不调用实际API）
"""

from pathlib import Path
from src.file_processor import FileProcessor
from src.config import UPLOADED_DIR, ANALYSIS_DIR

def demo_file_processing():
    """Demonstrate file processing capabilities"""
    print("=" * 70)
    print("CookMoney - AI投资研报分析系统演示")
    print("=" * 70)
    print()
    
    # Initialize file processor
    processor = FileProcessor()
    
    # Process example file
    example_file = Path("reports/example_report.md")
    
    if not example_file.exists():
        print(f"错误: 示例文件不存在: {example_file}")
        return
    
    print(f"📄 处理文件: {example_file.name}")
    print("-" * 70)
    
    # Process the file
    result = processor.process_file(example_file)
    
    print(f"文件类型: {result['type']}")
    print(f"文件格式: {result['format']}")
    print(f"内容长度: {len(result['content'])} 字符")
    print()
    
    print("📝 内容预览 (前500字符):")
    print("-" * 70)
    print(result['content'][:500])
    print("...")
    print()
    
    # Show what would be sent to AI
    print("🤖 将要发送给AI的分析请求:")
    print("-" * 70)
    print("""
系统提示：
你是一位资深的投资分析师，专门分析各类投资研报、市场分析报告等文件。
你的任务是：
1. 仔细阅读并理解所有提供的研报内容
2. 提取关键信息：投资标的、行业趋势、风险因素、机会点等
3. 综合分析多份报告的观点，找出共识和分歧
4. 基于分析给出清晰、可执行的投资建议
5. 评估投资风险等级（低、中、高）

用户请求：
请基于以上研报内容，提供综合投资分析和建议。

研报内容：
[文件内容会在这里...]
""")
    print()
    
    # Show expected output format
    print("📊 AI分析结果示例:")
    print("-" * 70)
    print("""
基于对XX科技2023年第三季度财报的详细分析，以下是我的投资建议：

【投资标的】XX科技股份有限公司
【当前股价】95元
【目标价位】120元（上涨空间26%）

【核心投资逻辑】

1. **业绩表现优异**
   - 营收150亿元，同比增长45%，超市场预期
   - 净利润42亿元，增长50%，盈利能力持续提升
   - 净利润率达28%，显示出色的成本控制能力

2. **业务增长强劲**
   - 云服务营收增长60%，是核心增长引擎
   - AI平台营收增长55%，抓住了AI浪潮机遇
   - 企业客户突破100万家，用户基础稳固

3. **技术投入充足**
   - 研发投入占营收15%，保证技术领先优势
   - 大模型产品获市场认可，形成竞争壁垒
   - 生态系统完善，合作伙伴超5000家

4. **市场地位稳固**
   - 国内市场份额前三
   - 国际化进展顺利，海外收入占比25%
   - 品牌影响力持续提升

【风险评估】风险等级：中等

主要风险点：
- 市场竞争加剧，可能影响利润率
- 技术迭代快速，需持续高投入
- 监管政策变化，特别是数据安全方面
- 宏观经济波动，影响企业IT支出

【投资建议】

**评级：买入**

**操作策略：**
1. 建议配置：建议将XX科技纳入核心持仓，配置比例5-10%
2. 买入时机：回调至90-92元区间可分批建仓
3. 目标价位：120元（26%上涨空间）
4. 止损位：85元（下跌约10%）
5. 持有期限：建议中长期持有（6-12个月）

**适合投资者类型：**
- 看好科技行业长期发展的投资者
- 能承受中等风险的成长型投资者
- 关注AI和云计算领域的价值投资者

**后续关注点：**
1. 下季度业绩增长是否能够持续
2. AI业务的商业化进展
3. 市场竞争格局的变化
4. 监管政策的动向

【结论】

XX科技处于云计算和AI行业的高速增长期，公司业绩优异、技术领先、
市场地位稳固。尽管面临一定竞争和监管风险，但整体风险可控。
基于基本面分析和行业前景，建议买入并中长期持有。

---
*以上分析仅供参考，不构成投资建议。投资有风险，决策需谨慎。*
""")
    print()
    
    print("=" * 70)
    print("演示完成！")
    print()
    print("💡 提示:")
    print("  1. 配置 .env 文件中的 OPENAI_API_KEY 后可使用真实的AI分析")
    print("  2. 将研报文件放入 reports/uploaded/ 目录")
    print("  3. 运行: python main.py analyze reports/uploaded/your_file.pdf")
    print("  4. 查看分析结果: reports/analysis/*.md")
    print("=" * 70)


if __name__ == "__main__":
    demo_file_processing()

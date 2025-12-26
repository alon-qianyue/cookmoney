# CookMoney - AI投资研报智能分析系统

🤖 基于AI的投资研报综合分析工具，支持多种文件格式（图片、PDF、网页、文本），自动生成投资建议。

## 功能特性

- ✅ **多格式支持**: 支持图片（JPG, PNG, GIF, WebP）、PDF、HTML、文本等多种格式
- 🧠 **AI智能分析**: 使用OpenAI GPT-4o模型进行深度分析
- 📊 **综合投资建议**: 综合多份研报，生成专业的投资建议
- 💾 **结果保存**: 自动保存分析结果为JSON和Markdown格式
- 🔒 **安全可靠**: 本地处理文件，保护隐私

## 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/alon-qianyue/cookmoney.git
cd cookmoney

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 2. 准备研报文件

将你的研报文件放到 `reports/uploaded/` 目录下：

```bash
# 示例：复制研报文件
cp ~/Downloads/research_report.pdf reports/uploaded/
cp ~/Downloads/market_chart.png reports/uploaded/
```

### 3. 开始分析

```bash
# 分析单个文件
python main.py analyze reports/uploaded/research_report.pdf

# 分析多个文件
python main.py analyze reports/uploaded/report1.pdf reports/uploaded/chart.png

# 分析所有PDF文件
python main.py analyze "reports/uploaded/*.pdf"

# 查看所有文件和报告
python main.py list
```

## 使用示例

### 分析单份研报

```bash
python main.py analyze reports/uploaded/quarterly_report.pdf
```

输出示例：
```
找到 1 个文件待分析:
  - reports/uploaded/quarterly_report.pdf

正在分析研报...

✓ 分析完成!
结果已保存到:
  - JSON: reports/analysis/analysis_20231226_143022.json
  - Markdown: reports/analysis/analysis_20231226_143022.md

============================================================
投资建议摘要:
============================================================
基于该季度报告的分析，以下是投资建议：

【投资标的】: XX科技公司
【行业趋势】: 人工智能和云计算持续高速增长
【关键观点】:
1. 营收同比增长45%，超出市场预期
2. 净利润率提升至28%，显示良好的成本控制
3. 研发投入占比达15%，技术护城河加深

【投资建议】: 建议增持
【风险等级】: 中等
【主要风险】:
- 市场竞争加剧
- 监管政策变化
- 国际贸易环境不确定性

【操作建议】:
- 建议在回调至支撑位时分批建仓
- 目标价位: XX元
- 止损价位: XX元
============================================================
```

### 综合分析多份研报

```bash
# 分析同一公司的多份报告
python main.py analyze \
  reports/uploaded/company_annual_report.pdf \
  reports/uploaded/analyst_coverage.pdf \
  reports/uploaded/price_chart.png \
  --output company_comprehensive_analysis
```

### 分析图表

```bash
# 分析技术分析图表
python main.py analyze reports/uploaded/stock_chart.png
```

## 项目结构

```
cookmoney/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖
├── .env.example             # 环境变量示例
├── .gitignore               # Git忽略文件
├── main.py                  # 主程序入口
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── file_processor.py   # 文件处理器
│   └── analyzer.py         # AI分析器
└── reports/                 # 研报目录
    ├── uploaded/           # 上传的研报文件
    └── analysis/           # 生成的分析报告
```

## 支持的文件格式

### 图片格式
- JPG/JPEG
- PNG
- GIF
- WebP

### 文本格式
- PDF
- TXT
- Markdown (MD)
- HTML/HTM

## 配置说明

在 `.env` 文件中配置以下参数：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here  # 必填
OPENAI_MODEL=gpt-4o                       # 可选，默认gpt-4o

# 应用设置
MAX_FILE_SIZE_MB=10                       # 最大文件大小（MB）
SUPPORTED_IMAGE_FORMATS=jpg,jpeg,png,gif,webp
SUPPORTED_TEXT_FORMATS=txt,md,pdf,html
```

## 技术架构

- **语言**: Python 3.8+
- **AI模型**: OpenAI GPT-4o
- **图片处理**: Pillow
- **PDF处理**: PyPDF2
- **网页解析**: BeautifulSoup4
- **环境配置**: python-dotenv

## 常见问题

### 1. 如何获取OpenAI API Key？

访问 [OpenAI Platform](https://platform.openai.com/) 注册账号并创建API Key。

### 2. 文件大小限制？

默认限制为10MB，可在 `.env` 文件中修改 `MAX_FILE_SIZE_MB` 参数。

### 3. 支持哪些语言？

系统主要支持中文分析，也可以处理英文内容。

### 4. 分析结果保存在哪里？

分析结果保存在 `reports/analysis/` 目录下，包含JSON和Markdown两种格式。

### 5. 如何分析网页？

将网页保存为HTML文件后放入 `reports/uploaded/` 目录即可分析。

## 开发路线图

- [x] 基础文件处理功能
- [x] OpenAI API集成
- [x] 多格式支持
- [x] 命令行工具
- [ ] Web界面
- [ ] 批量处理优化
- [ ] 自定义分析模板
- [ ] 历史分析对比
- [ ] 投资组合建议

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

- GitHub: [alon-qianyue/cookmoney](https://github.com/alon-qianyue/cookmoney)

---

⚠️ **风险提示**: 本工具提供的投资建议仅供参考，不构成投资建议。投资有风险，决策需谨慎。
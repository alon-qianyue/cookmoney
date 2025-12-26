# CookMoney 使用指南

## 目录
1. [安装配置](#安装配置)
2. [基础使用](#基础使用)
3. [高级功能](#高级功能)
4. [最佳实践](#最佳实践)
5. [故障排除](#故障排除)

## 安装配置

### 1. 系统要求
- Python 3.8 或更高版本
- pip 包管理器
- OpenAI API 账号

### 2. 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/alon-qianyue/cookmoney.git
cd cookmoney

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
```

### 3. 配置OpenAI API

编辑 `.env` 文件：
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
```

获取API Key: https://platform.openai.com/api-keys

## 基础使用

### 1. 准备研报文件

将要分析的文件放入 `reports/uploaded/` 目录：

```bash
# 支持的文件类型：
# - 图片: .jpg, .jpeg, .png, .gif, .webp
# - 文档: .pdf, .txt, .md, .html
```

### 2. 分析单个文件

```bash
python main.py analyze reports/uploaded/report.pdf
```

### 3. 分析多个文件

```bash
# 方式1: 指定多个文件
python main.py analyze file1.pdf file2.png file3.txt

# 方式2: 使用通配符
python main.py analyze "reports/uploaded/*.pdf"

# 方式3: 混合不同格式
python main.py analyze reports/uploaded/*.pdf reports/uploaded/*.png
```

### 4. 查看结果

分析结果保存在 `reports/analysis/` 目录下：
- `.json` 文件：完整的分析数据
- `.md` 文件：易读的Markdown格式报告

```bash
# 查看所有分析报告
python main.py list

# 直接查看Markdown报告
cat reports/analysis/analysis_20231226_143022.md
```

## 高级功能

### 1. 自定义输出文件名

```bash
python main.py analyze reports/uploaded/report.pdf --output my_analysis
# 输出: reports/analysis/my_analysis.json 和 my_analysis.md
```

### 2. 批量分析

创建批处理脚本 `batch_analyze.sh`:

```bash
#!/bin/bash
# 分析所有PDF
python main.py analyze "reports/uploaded/*.pdf" --output pdf_analysis

# 分析所有图片
python main.py analyze "reports/uploaded/*.png" --output chart_analysis

# 综合分析
python main.py analyze "reports/uploaded/*" --output comprehensive_analysis
```

### 3. 分类分析

按照不同主题组织文件：

```bash
reports/uploaded/
├── tech_sector/
│   ├── company_a.pdf
│   └── company_b.pdf
├── finance_sector/
│   ├── bank_report.pdf
│   └── insurance_report.pdf
└── charts/
    ├── tech_chart.png
    └── finance_chart.png
```

分别分析：
```bash
python main.py analyze "reports/uploaded/tech_sector/*.pdf" --output tech_analysis
python main.py analyze "reports/uploaded/finance_sector/*.pdf" --output finance_analysis
```

## 最佳实践

### 1. 文件命名规范

建议使用有意义的文件名：
```
好的命名：
- 阿里巴巴_2023Q3_财报.pdf
- 腾讯_技术分析图_20231215.png
- AI行业_市场分析_2023.md

避免：
- report.pdf
- 图片1.png
- 文件.txt
```

### 2. 分析技巧

**单一主题分析**
```bash
# 分析特定公司的多份报告
python main.py analyze \
  reports/uploaded/company_q1.pdf \
  reports/uploaded/company_q2.pdf \
  reports/uploaded/company_q3.pdf \
  --output company_yearly_trend
```

**跨领域对比**
```bash
# 对比不同行业
python main.py analyze \
  reports/uploaded/tech_report.pdf \
  reports/uploaded/finance_report.pdf \
  --output sector_comparison
```

**图文结合分析**
```bash
# 结合报告和图表
python main.py analyze \
  reports/uploaded/annual_report.pdf \
  reports/uploaded/revenue_chart.png \
  reports/uploaded/profit_chart.png \
  --output comprehensive_analysis
```

### 3. 结果管理

定期整理分析结果：
```bash
# 创建日期目录
mkdir -p reports/analysis/2023/12
mv reports/analysis/analysis_20231*.* reports/analysis/2023/12/

# 或按主题整理
mkdir -p reports/analysis/tech_sector
mv reports/analysis/tech_*.* reports/analysis/tech_sector/
```

## 故障排除

### 问题1: API Key错误

**错误信息**: `ValueError: OPENAI_API_KEY not set in environment variables`

**解决方案**:
```bash
# 检查 .env 文件是否存在
ls -la .env

# 确保API Key已设置
cat .env | grep OPENAI_API_KEY

# 重新设置
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### 问题2: 文件过大

**错误信息**: `ValueError: File size exceeds maximum allowed size`

**解决方案**:
```bash
# 方法1: 调整配置
echo "MAX_FILE_SIZE_MB=20" >> .env

# 方法2: 压缩文件
# 对于图片，可以降低分辨率
# 对于PDF，可以压缩或拆分
```

### 问题3: 依赖安装失败

**解决方案**:
```bash
# 升级pip
pip install --upgrade pip

# 单独安装问题依赖
pip install openai
pip install Pillow
pip install PyPDF2

# 或使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4: 编码错误

**错误信息**: `UnicodeDecodeError`

**解决方案**:
```bash
# 确保文件使用UTF-8编码
# 对于文本文件，可以转换编码：
iconv -f GBK -t UTF-8 input.txt > output.txt
```

### 问题5: OpenAI API限流

**错误信息**: `Rate limit exceeded`

**解决方案**:
- 等待一段时间后重试
- 升级OpenAI API套餐
- 减少单次分析的文件数量

## 示例工作流

### 日常使用流程

```bash
# 1. 收集研报
# 将下载的研报文件复制到 reports/uploaded/

# 2. 查看待分析文件
python main.py list

# 3. 执行分析
python main.py analyze "reports/uploaded/*.pdf" --output daily_analysis

# 4. 查看结果
cat reports/analysis/daily_analysis.md

# 5. 整理归档
mv reports/uploaded/* reports/uploaded/archive/
mv reports/analysis/daily_analysis.* reports/analysis/archive/
```

### 深度研究流程

```bash
# 1. 准备材料
mkdir -p reports/uploaded/project_alpha
# 收集相关公司、行业的所有资料

# 2. 分阶段分析
python main.py analyze "reports/uploaded/project_alpha/company_*.pdf" --output alpha_company
python main.py analyze "reports/uploaded/project_alpha/industry_*.pdf" --output alpha_industry
python main.py analyze "reports/uploaded/project_alpha/chart_*.png" --output alpha_charts

# 3. 综合分析
python main.py analyze "reports/uploaded/project_alpha/*" --output alpha_comprehensive

# 4. 生成最终报告
# 人工整合各个分析结果，形成投资决策
```

## 技巧与窍门

1. **使用虚拟环境**: 避免依赖冲突
2. **定期备份**: 备份重要的分析结果
3. **版本控制**: 使用Git管理分析结果（可选）
4. **批量处理**: 编写脚本自动化重复任务
5. **模型选择**: 根据需求选择合适的GPT模型（gpt-4o更智能但更贵）

## 获取帮助

```bash
# 查看帮助
python main.py --help
python main.py analyze --help

# 查看版本
python -c "from src import __version__; print(__version__)"
```

## 更新日志

查看项目更新: https://github.com/alon-qianyue/cookmoney/releases

---

如有问题，欢迎提交Issue: https://github.com/alon-qianyue/cookmoney/issues

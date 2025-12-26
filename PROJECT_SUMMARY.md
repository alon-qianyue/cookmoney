# CookMoney 项目总结

## 项目概述

CookMoney 是一个基于AI的投资研报智能分析系统，能够自动分析各种格式的研报文件（图片、PDF、文本、网页），并生成专业的投资建议。

## 实现的功能

### 1. 多格式文件处理
- ✅ **图片格式**: JPG, JPEG, PNG, GIF, WebP
- ✅ **文档格式**: PDF, TXT, MD, HTML
- ✅ **自动编码检测**: 支持多种字符编码，智能处理中文内容
- ✅ **文件验证**: 大小限制、格式验证

### 2. AI分析引擎
- ✅ **OpenAI集成**: 使用GPT-4o模型进行深度分析
- ✅ **智能提示词**: 专业的投资分析师角色设定
- ✅ **多文件综合**: 支持同时分析多份研报
- ✅ **混合内容**: 同时处理文本和图片内容

### 3. 投资建议生成
- ✅ **全面分析**: 提取关键信息、行业趋势、风险因素
- ✅ **投资建议**: 生成具体可执行的投资策略
- ✅ **风险评估**: 评估投资风险等级（低、中、高）
- ✅ **中文输出**: 专业的中文投资分析报告

### 4. 命令行工具
- ✅ **简单易用**: 友好的CLI界面
- ✅ **批量处理**: 支持通配符和多文件分析
- ✅ **结果管理**: 自动保存为JSON和Markdown格式
- ✅ **文件列表**: 查看上传文件和分析报告

### 5. 配置管理
- ✅ **环境变量**: 使用.env文件管理配置
- ✅ **可定制**: API密钥、模型、文件大小限制等可配置
- ✅ **默认值**: 合理的默认配置，开箱即用

## 技术栈

- **语言**: Python 3.8+
- **AI模型**: OpenAI GPT-4o
- **依赖库**:
  - `openai`: OpenAI API客户端
  - `pypdf`: PDF文件解析（现代化的替代品）
  - `Pillow`: 图片处理
  - `beautifulsoup4`: HTML解析
  - `chardet`: 字符编码检测
  - `python-dotenv`: 环境变量管理

## 项目结构

```
cookmoney/
├── README.md                    # 项目文档
├── QUICKSTART.md               # 快速开始指南
├── USAGE_GUIDE.md              # 详细使用指南
├── requirements.txt            # Python依赖
├── .env.example               # 环境变量模板
├── .gitignore                 # Git忽略文件
├── main.py                    # 主程序入口
├── demo.py                    # 演示脚本
├── test_basic.py              # 基础测试
├── src/                       # 源代码
│   ├── __init__.py
│   ├── config.py             # 配置管理
│   ├── file_processor.py     # 文件处理器
│   └── analyzer.py           # AI分析器
└── reports/                   # 研报目录
    ├── uploaded/             # 上传的研报文件
    ├── analysis/             # 生成的分析报告
    ├── example_report.md     # 示例公司财报
    └── example_market_strategy.md  # 示例市场策略
```

## 代码质量

### 通过的检查
- ✅ **单元测试**: 7个测试全部通过
- ✅ **代码审查**: 所有审查意见已解决
- ✅ **安全扫描**: CodeQL无安全漏洞
- ✅ **依赖更新**: 使用最新稳定版本

### 改进点
1. **PyPDF2 → pypdf**: 升级到维护良好的PDF库
2. **编码检测**: 添加chardet处理不同编码
3. **可配置限制**: 内容字符数限制可通过环境变量配置
4. **错误处理**: 改进文件读取的编码错误处理

## 使用示例

### 基础使用
```bash
# 安装依赖
pip install -r requirements.txt

# 配置API Key
cp .env.example .env
# 编辑 .env 设置 OPENAI_API_KEY

# 分析研报
python main.py analyze reports/example_report.md
```

### 高级使用
```bash
# 分析多个文件
python main.py analyze reports/*.md

# 综合分析（文本+图片）
python main.py analyze report.pdf chart.png data.txt

# 自定义输出
python main.py analyze report.pdf --output my_analysis
```

## 输出示例

分析报告包含：
- 📊 投资标的和当前评估
- 📈 核心投资逻辑（业绩、增长、技术、市场地位）
- ⚠️ 风险评估和主要风险点
- 💡 投资建议（评级、操作策略、适合投资者类型）
- 🎯 后续关注点

## 文档

1. **README.md**: 完整的项目介绍和使用说明
2. **QUICKSTART.md**: 5分钟快速上手指南
3. **USAGE_GUIDE.md**: 详细使用指南和故障排除
4. **demo.py**: 交互式演示脚本

## 安全性

- ✅ 本地处理文件，保护隐私
- ✅ API Key通过环境变量管理
- ✅ 文件大小验证，防止滥用
- ✅ 无已知安全漏洞（CodeQL扫描通过）

## 可扩展性

系统设计支持未来扩展：
- 📱 Web界面
- 🔄 批量处理优化
- 📝 自定义分析模板
- 📊 历史分析对比
- 💼 投资组合建议
- 🌐 多语言支持
- 📈 实时数据集成

## 测试覆盖

- ✅ 文件处理测试
- ✅ 配置管理测试
- ✅ CLI功能测试
- ✅ 目录结构测试
- ✅ 模块导入测试

## 部署建议

### 开发环境
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 生产环境
- 使用虚拟环境隔离依赖
- 配置适当的API限流
- 定期备份分析结果
- 监控API使用量和成本

## 许可和免责声明

- **许可证**: MIT License
- **免责声明**: 本工具提供的投资建议仅供参考，不构成投资建议。投资有风险，决策需谨慎。

## 贡献者

- 项目开发: GitHub Copilot
- 仓库所有者: alon-qianyue

## 联系方式

- GitHub: https://github.com/alon-qianyue/cookmoney
- Issues: https://github.com/alon-qianyue/cookmoney/issues

---

**最后更新**: 2023年12月26日

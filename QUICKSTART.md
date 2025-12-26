# 快速开始指南 (Quick Start Guide)

## 5分钟快速上手

### 步骤1: 克隆项目
```bash
git clone https://github.com/alon-qianyue/cookmoney.git
cd cookmoney
```

### 步骤2: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤3: 配置API Key
```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，添加你的OpenAI API Key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 步骤4: 查看演示
```bash
# 运行演示程序，了解系统如何工作
python demo.py
```

### 步骤5: 分析你的第一份研报
```bash
# 1. 将研报文件放到 reports/uploaded/ 目录
cp ~/Downloads/your_report.pdf reports/uploaded/

# 2. 运行分析
python main.py analyze reports/uploaded/your_report.pdf

# 3. 查看结果
cat reports/analysis/analysis_*.md
```

## 示例工作流

```bash
# 查看所有文件
python main.py list

# 分析单个PDF
python main.py analyze reports/uploaded/report.pdf

# 分析多个文件
python main.py analyze reports/uploaded/*.pdf reports/uploaded/*.png

# 指定输出文件名
python main.py analyze reports/uploaded/report.pdf --output my_analysis
```

## 常用命令

```bash
# 查看帮助
python main.py --help

# 查看分析命令帮助
python main.py analyze --help

# 列出所有文件和报告
python main.py list
```

## 获取OpenAI API Key

1. 访问 https://platform.openai.com/
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的API Key
5. 复制Key到 .env 文件

## 需要帮助？

- 查看完整文档: [README.md](README.md)
- 详细使用指南: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- 提交问题: https://github.com/alon-qianyue/cookmoney/issues

## 文件格式支持

✅ 图片: JPG, PNG, GIF, WebP
✅ 文档: PDF, TXT, MD, HTML

---

**投资有风险，使用需谨慎！本工具仅供参考，不构成投资建议。**

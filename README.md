# cookmoney
## 结构说明
### data目录下是研报和信息源，目录结构是年/月/周
- 比如 2025/12/week3, 意思就是2025年 12月 第3周的数据

### cookrulers 目录是分析需要遵循的规则
- `cookrulers/analysis.md`：研究员的分析SOP（把研报变成可交易结论）
- `cookrulers/expect.md`：交易员的执行SOP（仓位、止损、止盈、回撤与复盘规则）

## 运行逻辑
- 通过cookrulers/analysis.md 的分析逻辑分析data目录下指定目录的信息
- 遵循cookrulers/expect.md 的准则，给出分析结果和推荐的投资标的，将结果写入对应目录下，新增一个文件，文件名需要带分析模型名字，比如用gpt5.2

## pip 源加速（安装依赖更稳定）

- **临时使用镜像（推荐）**：

```bash
pip install -r cookmoney/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --default-timeout 1000 --retries 10
```

- **全局配置**：
  - 仓库提供了示例：`cookmoney/pip.conf.example`
  - macOS 常见路径：`~/.config/pip/pip.conf`

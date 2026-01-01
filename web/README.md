### CookMoney Web 系统（本地）

功能：
- 设置时间段、资产池与策略参数
- 生成“下一周买/卖/持有信号”（含综合指标 `risk_score`）

> 说明：当前版本先上线“信号面板”；回测（期初金额/复利/年度追加的完整收益表）在下一步补齐到页面里。

---

### 安装

在项目根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r cookmoney/requirements.txt
```

#### pip 加速（推荐）

- **一次性命令**（不改全局配置）：

```bash
pip install -r cookmoney/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --default-timeout 1000 --retries 10
```

- **全局配置**：把 `cookmoney/pip.conf.example` 复制到你的 pip 配置路径即可（macOS 常见为 `~/.config/pip/pip.conf`）。

---

### 运行

```bash
source .venv/bin/activate
streamlit run cookmoney/web/app.py
```



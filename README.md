# BTC 动量–反转仪表盘 (JLST 2025)

基于 **Jegadeesh, Luo, Subrahmanyam & Titman (2025, RFS)**《Short-Term Reversals and Longer-Term Momentum around the World》的 BTC 特化反转–动量复合模型，使用 **TradingView Lightweight Charts** 构建的交互式仪表盘。

## 功能

- **三段式信号结构**：短期反转（ρ₁<0）× 中长期动量（ρ₃…ρ₁₂>0），中间设过渡死区（ρ₂≈0）
- **BTC 特化**：对数收益、24/7 日历、减半周期调制、信息冲击衰减、瀑布（连环爆仓）检测
- **动态权重（预测 c）**：噪声代理 = ATR% + 量比 + 波动机制 + **资金费率**，噪声↑ → 反转权重↑
- **机制信心（预测 b）**：corr(Rev, Mom) 共振时自动降权
- **三图联动**：价格 K 线（含信号标记）+ 复合信号 + 资金费率，十字光标与时间轴同步
- **日 / 周 / 月** 三个周期，参数经网格回测优化（`data/params.json`）
- GitHub Actions 每日自动更新数据

## 在线查看

启用 GitHub Pages 后访问：`https://<your-username>.github.io/btc-momentum-reversal/`

## 数据来源

| 数据 | 来源 | 备注 |
|------|------|------|
| BTCUSDT 现货 K 线 | Binance（`data-api.binance.vision` / `api.binance.com`） | 2017-08 至今 |
| 资金费率 | Binance USDⓈ-M `fapi/fundingRate` | 不可达时回退至 `data.binance.vision` 溢价指数推算 |
| 备用行情 | CryptoQuant `price-ohlcv` | 需环境变量 `CRYPTOQUANT_API_KEY`（免费档近 365 天） |

## 本地运行

```bash
pip install -r requirements.txt   # 仅回测需要 pandas/numpy；数据管线纯标准库
python scripts/update_data.py     # 抓取数据并计算指标 -> data/*.json
python -m http.server 8000        # 打开 http://localhost:8000
```

## 参数回测

```bash
python scripts/backtest.py          # 全网格（600 组合）
python scripts/backtest.py --quick  # 快速验证
```

训练期（2017-08 ~ 2023-12）按 `(命中率-0.5) × Sharpe`（7d/14d/30d 前向收益均值）选优，
输出样本外（2024 至今）表现至 `data/params.json`，管线下次运行自动采用。

## 目录结构

```
├── index.html                        # 仪表盘（Lightweight Charts，本地 vendor）
├── vendor/                           # lightweight-charts v5 独立构建
├── scripts/
│   ├── update_data.py                # 数据管线（行情 + 资金费率 + 指标计算）
│   └── backtest.py                   # 参数网格回测
├── data/                             # 生成的 JSON（每日由 Actions 更新）
└── .github/workflows/update.yml      # 每日定时更新
```

## 论文 → 指标映射

| 论文结论 | 实现 |
|----------|------|
| Table 2 ρ₁<0（短期反转） | `Rev = -log(C/C[revLen])`，波动调整 + z-score |
| Table 2 ρ₂≈0（过渡区） | `|comp| < 0.25` 死区不发信号 |
| Table 2 ρ₃…ρ₁₂>0（动量） | `Mom = log(C[skip]/C[skip+momLen])` |
| Proposition 2 | `momSkip` 跳过最近一个月 |
| 预测 a（信息冲击） | 放量大波动后 10 根 K 线 Rev ×0.4 |
| 预测 b（机制） | corr(Rev,Mom)>0.2 → 复合 ×0.6 |
| 预测 c（噪声） | 噪声 z 越高，wRev 越大、wMom 越小 |

## 免责声明

仅供量化研究与教育用途，不构成投资建议。论文结论基于横截面十分位对冲组合，本项目为单资产时序近似。

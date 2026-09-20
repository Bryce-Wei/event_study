# 真实案例的数据来源与处理记录

本目录提供 TSLA 分析师评级变动的**小样本、日频事件研究案例**。行情是真实历史数据；评级记录由公开报道中的事实字段人工整理，不是供应商导出的全量原始评级数据库。没有复制新闻全文或付费研报。

## 1. 原始行情

- 标的：[TSLA 历史行情（Yahoo Finance）](https://finance.yahoo.com/quote/TSLA/history/)。
- 基准：[SPY 历史行情（Yahoo Finance）](https://finance.yahoo.com/quote/SPY/history/)。SPY 为本示例的市场代理，不能等同于 TSLA 的完整风险模型。
- 请求区间：2023-01-01 至 2025-01-01，不含终点；实际交易日期、抓取时间、原始接口、行数、SHA-256 均记录在 `source_manifest.json`。
- `raw/*.yahoo.json`：保存下载时的原始接口响应。
- `raw/*_prices.csv`：由响应解析的 OHLCV 与调整后收盘价，便于检查。
- `returns.csv`：TSLA 和 SPY 调整后收盘价的日简单收益，分别为 `ret`、`mkt`；数值 0.01 表示 1%。
- 不对缺失价格做前向填充；不把无报价日当作 0 收益；第一行没有前一交易日价格，因此收益为空。
- Yahoo 的调整后价格可能随公司行动和数据修订变化。本项目保存的是本次获取的历史快照，不是当时可见的 point-in-time 数据库。离线复现使用随附快照；显式重新下载可能得到不同数值。

## 2. 原始评级事实记录

`raw/ratings_source_records.csv` 保留机构、评级前后值、原始/报道日期、出处、可验证的时间及说明。`events.csv` 是提供给分析代码的标准化输入。除第 3 条外，原始研报精确发布时间未核实，不将新闻页面的时间冒充研报发布时刻。

| ID | 原始或报道日期 | 对齐交易日 | 机构 | 原评级 → 新评级 | 来源 |
|---|---|---|---|---|---|
| 1 | 2023-06-21 | 2023-06-21 | Barclays | Overweight → Equalweight | [Investing.com 报道](https://www.investing.com/news/stock-market-news/4-big-analyst-cuts-tesla-slashed-at-barclays-after-too-sharp-rally-3109875) |
| 2 | 2023-06-22 | 2023-06-22 | Morgan Stanley | Overweight → Equal Weight | [Investing.com 报道](https://www.investing.com/news/stock-market-news/tesla-loses-one-of-its-biggest-street-bulls-as-morgan-stanley-cuts-rating-432SI-3110961) |
| 3 | 2023-09-10（周日） | 2023-09-11 | Morgan Stanley | Equal-weight → Overweight | [Reuters，经 Investing.com 发布](https://m.investing.com/news/stock-market-news/tesla-supercomputer-could-boost-ev-makers-market-cap-by-600-billion-morgan-stanley-3171337?ampMode=1) |
| 4 | 2024-03-13 | 2024-03-13 | Wells Fargo | Equal Weight → Underweight | [Investing.com 报道](https://www.investing.com/news/stock-market-news/tesla-stock-drops-as-wells-fargo-cuts-rating-to-sell-sees-30-downside-risk-3335448) |
| 5 | 2024-04-24 | 2024-04-24 | Bank of America | Neutral → Buy | [Investing.com 报道](https://www.investing.com/news/stock-market-news/tesla-stock-extends-gains-to-12-as-bank-of-america-upgrades-to-buy-3392978) |
| 6 | 2024-07-12 | 2024-07-12 | UBS | Neutral → Sell | [Investing.com 报道](https://www.investing.com/news/stock-market-news/tesla-stock-extends-selloff-as-ubs-downgrades-to-sell-on-valuation-concerns-3516888) |

上述评级名称保留来源拼写，不把不同机构的评级分档视作统一分值。只根据已核实的上调/下调动作分类，未把目标价变化或重申评级当作评级调整。

### 补充核验

- Morgan Stanley 的 [2023-09-10 原始报告第三方镜像](https://ussnsfile.moomoo.com/709297661694503734/Morgan%20Stanley%EF%BC%9AUnlocking%20Tesla%E2%80%99s%20AI%20Mojo%E2%80%A6%20Enter%20the%20Dojo%20Upgrade%20to%20OW%2C%20PT%20%24400%2C%20Top%20Pick%20.pdf) 首页标注 `September 10, 2023 11:28 PM GMT`；记录为 `2023-09-10T23:28:00Z`，美国市场下一交易日为 09-11。本仓库仅链接，不再分发报告。
- Barclays 变动亦获 [Reuters 当日报道](https://www.investing.com/news/economy/futures-muted-ahead-of-powells-congressional-testimony-3109967)支持。
- Wells Fargo 变动的盘前发生时段获 [Reuters 同日报道（法文转载）](https://www.boursorama.com/bourse/actualites/tesla-chute-apres-que-wells-fargo-a-revu-a-la-baisse-sa-notation-et-reduit-ses-previsions-de-ventes-5df698489bcdd0450f09759144f83282)支持。

## 3. 样本选择和解释边界

这 6 条是为了让项目可核验、可运行而整理的方便样本，未覆盖该区间全部评级事件，也未按收益方向筛选。不能用于估计评级调整的一般效果。

2023-06-21 和 06-22 的窗口重叠；同一只股票的多次事件也不当然独立。2024-04-24 同时出现财报与业务展望消息；2024-07-12 紧邻 Robotaxi 延期报道。观测到的异常收益包含其他消息和风险暴露，不能全部归因于分析师评级变化。

这里只做日频展示。实际研究需用交易所日历和准确的发布时间区分盘前、盘中、盘后；盘后事件通常应对齐下一交易日。新增分析程序按输入日期映射到不早于该日期的首个交易日，不自动判断盘后信息。

原始新闻网页的显示时区未能全部确认，`source_timezone_verified=false` 表示这一限制；其时间字符串仅作出处记录，不参与收益计算。

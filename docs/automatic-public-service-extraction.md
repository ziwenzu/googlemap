# 全自动公共服务部门抽取（当前主流程）

按研究者2026-09-21的最新要求，本流程不使用人工识别，不安排复核，
不使用学生编码表、不做官方名录匹配、不调用语言模型。
之前的广义公共部门识别和人工验证文件仅作历史保留，不构成本流程前置条件。

## 研究范围

只按八类服务部门名单提取：DMV／驾照车辆登记、警局／治安官、USPS邮局、
社会保障、福利／就业与失业救济、税务、法院、市政办事／登记许可。
学校、大学、医院、公园、图书馆、消防与监狱不在本轮名单内。
本数据集不试图穷尽所有公共部门，也不把学校排除理解为学校属于私营。

## 自动规则

1. 地点名称命中部门关键词，或Google类别精确命中类别白名单，即成为命中项。
2. 排除名单覆盖私人快递店、驾校、律师和报税商户、商店餐饮、
   博物馆纪念设施及非服务窗口邮政设施；排除规则优先于纳入。
3. 地址、描述不参与匹配，不按评分或评论数量筛选。
4. 按稳定的gmap_id去重。跨州重复采用确定顺序的第一条合格元数据，
   汇总来源分区及命中服务类型；同址不同gmap_id仍保留。
5. 50州与DC组成主表。Other中与主表重复的ID合并来源；
   仅在Other出现的ID独立输出，不自动推断它们位于美国。
6. 输出属于规则定义的机构集合，不报告未经测量的准确率或召回率。

所有关键词与类别名单在config/public-service-keywords-v1.json中。
类别名单不是官方所有权认证；规则可能误收或漏收，但不会转成人工任务。
model_probability与adjudicated_label留空，表示未运行概率模型或人工裁决，
不是待填写表格。candidate及exclusion_reason保留自动筛选来源信息。

## 运行

全国元数据已下载后：

```sh
make service-keywords-national
```

仅运行Vermont测试：

```sh
make service-keywords-pilot
```

输出为data/processed/public-service-keywords-v1/：

- places.csv：主表，gmap_id唯一，可直接供评论抽取脚本使用。
- by-service/*.csv：按主要服务类型分成八份，互不重叠。
- places-other-only.csv：只在Other分区出现的合格地点。
- matched-exclusions.csv：命中纳入词但被自动排除的原始行及原因。
- summary.json：完成状态、分区计数、规则与代码指纹、输出校验值。

每条保留命中关键词/类别、规则版本及来源行号。
数据始终在移动硬盘；逐行读取gzip，不展开完整州文件。
新输出目录必须不存在；重跑时更换SERVICE_OUTPUT，避免覆盖旧结果。
缺失分区、损坏JSON或输入中途变化会使执行失败；只有summary.json
标记complete的目录才视为完成。

现有scripts/02_extract_reviews.py可直接读取places.csv，
使用--place-mode all-candidates，无须任何人工标签。
本次目标是地点抽取，不自动启动全国评论重抽取。

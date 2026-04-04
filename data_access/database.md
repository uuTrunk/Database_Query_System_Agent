# world.sql 库表设计说明

本文档说明 agent 项目中 world 数据库的表结构设计、关系建模思路与索引布局。

对应 SQL 文件：world.sql。

## 1. 数据库定位

world 库采用经典的国家-城市-语言三层模型，目标是支撑以下类型的数据问题：

1. 国家维度统计（人口、洲别、区域）。
2. 城市维度分析（国家内城市分布、人口排行）。
3. 语言维度分析（官方语言、语言覆盖范围、使用比例）。
4. 多表关联分析（国家与首都、国家与语言、国家与城市）。

整体设计倾向于：

1. 规范化建模，减少冗余。
2. 显式主外键，保证一致性。
3. 面向检索路径设计索引，优先优化筛选与排序。

## 2. 表设计总览

world.sql 主要包含 3 张业务核心表：

1. city：城市信息。
2. country：国家信息。
3. countrylanguage：国家与语言关系。

关系拓扑如下：

1. country(Code) 1 -> N city(CountryCode)
2. country(Code) 1 -> N countrylanguage(CountryCode)
3. country(Capital) N -> 1 city(ID)

说明：

1. 一个国家有多个城市。
2. 一个国家有多种语言记录。
3. 国家表通过 Capital 字段指向城市表主键 ID，表示该国首都城市。

## 3. city 表设计

### 3.1 字段语义

1. ID：城市主键，自增整数。
2. Name：城市名称。
3. CountryCode：所属国家代码，关联 country.Code。
4. District：行政区或地区。
5. Population：城市人口。

### 3.2 主外键与约束

1. 主键：ID。
2. 外键：CountryCode -> country.Code。
3. 外键策略：ON DELETE RESTRICT、ON UPDATE RESTRICT。

设计含义：

1. 禁止删除或修改被城市记录引用的国家编码，确保引用完整性。
2. 防止脏数据（孤儿城市记录）出现。

### 3.3 索引设计

1. 主键索引：ID。
2. 单列索引：CountryCode。
3. 复合索引：idx_city_country_population(CountryCode, Population)。
4. 单列索引：idx_city_name(Name)。

优化目标：

1. 按国家查城市（CountryCode）。
2. 国家内按人口筛选或排序（CountryCode + Population）。
3. 按城市名检索（Name）。

## 4. country 表设计

### 4.1 字段语义

核心字段包括：

1. Code：国家三字码，主键。
2. Name：国家名。
3. Continent：洲枚举。
4. Region：区域名。
5. Population：国家人口。
6. Capital：首都城市 ID，逻辑上对应 city.ID。
7. 其余指标字段：SurfaceArea、GNP、LifeExpectancy 等。

### 4.2 主键与关联

1. 主键：Code。
2. 被引用：city.CountryCode、countrylanguage.CountryCode。
3. 关联字段：Capital 用于连接 city.ID。

设计含义：

1. 使用稳定编码作为业务实体标识，便于跨表连接。
2. 首都关系单独建在国家表中，支持国家维度直接拉取首都信息。

### 4.3 索引设计

1. 主键索引：Code。
2. 复合索引：idx_country_continent_population(Continent, Population)。
3. 复合索引：idx_country_region_population(Region, Population)。
4. 单列索引：idx_country_name(Name)。
5. 单列索引：idx_country_capital(Capital)。

优化目标：

1. 洲内人口分析（Continent + Population）。
2. 区域人口分析（Region + Population）。
3. 国家名称检索（Name）。
4. 国家与首都城市关联（Capital）。

## 5. countrylanguage 表设计

### 5.1 字段语义

1. CountryCode：国家代码。
2. Language：语言名称。
3. IsOfficial：是否官方语言，取值 T 或 F。
4. Percentage：该语言在国家中的使用比例。

### 5.2 主外键与约束

1. 复合主键：(CountryCode, Language)。
2. 外键：CountryCode -> country.Code。
3. 外键策略：ON DELETE RESTRICT、ON UPDATE RESTRICT。

设计含义：

1. 同一国家下同一语言只允许一条记录。
2. 语言信息必须隶属有效国家。

### 5.3 索引设计

1. 主键索引：(CountryCode, Language)。
2. 单列索引：CountryCode。
3. 单列索引：idx_countrylanguage_language(Language)。
4. 复合索引：idx_countrylanguage_official_percentage(IsOfficial, Percentage)。

优化目标：

1. 按国家取语言列表（CountryCode）。
2. 按语言反查国家集合（Language）。
3. 按官方语言与占比筛选（IsOfficial + Percentage）。

## 6. 设计特点与工程取舍

1. 高规范化：国家、城市、语言分层，减少重复存储。
2. 关系清晰：主外键显式声明，避免隐式关联。
3. 查询导向索引：围绕 CountryCode、Population、Language 等高频维度优化。
4. 字段类型稳健：国家码固定长度字符，人口用整型，比例用定点数。
5. 约束偏保守：RESTRICT 策略优先保护一致性，避免级联误删。

## 7. 典型查询路径

1. 某国家城市人口 TopN：city 按 CountryCode 过滤并按 Population 排序。
2. 洲内国家人口排行：country 按 Continent 过滤并按 Population 排序。
3. 按语言找国家：countrylanguage 按 Language 过滤并关联 country。
4. 国家与首都信息联查：country.Capital 关联 city.ID。

## 8. 与当前 agent 执行方式的关系

当前 agent 的数据读取逻辑会将整表加载到 pandas 再做后续推理与处理。

因此：

1. 无过滤的全量读取阶段，索引收益有限。
2. 若将筛选、分组、排序等计算更多下推到数据库端执行，上述索引收益会显著提升。

## 9. 可继续优化方向

1. 基于真实查询日志评估冗余索引并精简。
2. 对高频 SQL 使用 EXPLAIN 持续验证索引命中。
3. 对大结果集查询增加分页与列裁剪，降低 IO 和网络成本。
4. 对确定的热点聚合场景引入更贴近查询模式的复合索引。

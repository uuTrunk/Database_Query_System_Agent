# 索引优化说明

本文档说明本次 `world.sql` 的索引优化内容、目的与预期收益。

## 1. 优化目标

本次优化从以下两个角度提升查询速度：

1. 连接性能：减少多表关联时的扫描成本。
2. 过滤与排序性能：提升按国家/洲/语种/人口等条件检索时的响应速度。

## 2. 新增索引清单

### 2.1 city 表

- `idx_city_country_population (CountryCode, Population)`
  - 用于国家维度下的人口筛选、排序（如“某国人口最多的城市”）。
- `idx_city_name (Name)`
  - 用于按城市名等值查询或前缀检索。

### 2.2 country 表

- `idx_country_continent_population (Continent, Population)`
  - 用于洲内国家人口统计、TopN 查询。
- `idx_country_region_population (Region, Population)`
  - 用于区域维度人口筛选与排序。
- `idx_country_name (Name)`
  - 用于国家名检索。
- `idx_country_capital (Capital)`
  - 用于按首都关联 `city.ID` 的场景。

### 2.3 countrylanguage 表

- `idx_countrylanguage_language (Language)`
  - 用于按语种筛选国家。
- `idx_countrylanguage_official_percentage (IsOfficial, Percentage)`
  - 用于“官方语言 + 占比区间/排序”类查询。

## 3. 典型受益查询

1. 国家与城市人口排行：
   - `country.Code = city.CountryCode` 关联后按人口排序。
2. 按洲/区域筛选国家并统计：
   - `WHERE Continent = ?` 或 `WHERE Region = ?`。
3. 按语种与官方属性筛选：
   - `WHERE Language = ?` 或 `WHERE IsOfficial = 'T' AND Percentage > ?`。

## 4. 使用建议

1. 让查询条件尽量命中复合索引的最左前缀。
2. 在 SQL 中先过滤再关联，避免无效大表扫描。
3. 只选择必要列，减少回表与网络传输开销。

## 5. 验证方式

可通过 `EXPLAIN` 检查执行计划是否命中新索引，例如：

```sql
EXPLAIN SELECT Name, Population
FROM city
WHERE CountryCode = 'CHN'
ORDER BY Population DESC
LIMIT 10;
```

若 `key` 字段显示为新增索引名，说明优化已生效。

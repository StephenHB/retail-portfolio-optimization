# Data layout

## Demo dataset (Kaggle)

**Name:** [Retail Insights: A Comprehensive Sales Dataset](https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset)  
**Author:** `rajneesh231`  
**License:** Community Data License Agreement – Sharing – Version 1.0  

The publisher bundle includes **`data.csv`** and **`data.xlsx`**. This repo’s adapter reads **`data.csv`** only.

### What the file contains

- **Grain:** one row per **order line** (transaction), not one row per store-day aggregate.
- **Time span:** depends on the file version you download; expect multi-year daily/monthly order dates.
- **Product taxonomy:** each row has a **`Product Category`** (e.g. Office Supplies, Technology, Furniture) used as the Brzęczek-style **category** dimension.
- **Money:** amounts such as **`Order Total`** are stored as strings with currency symbols and commas (e.g. `$4,757.22`).
- **Other columns** (customer, address, ship mode, discounts, etc.) are present in the full export; the retail adapter currently uses only **`Order Date`**, **`Product Category`**, and **`Order Total`** for the monthly category panel.

### Local path

Place the extracted CSV here:

```text
data/raw/data.csv
```

The `data/raw/` directory is **gitignored** (except `.gitkeep`) so downloads are not committed.

### Code entry point

- Loader and panel logic: [`src/retail_portfolio/data/kaggle_retail_insights.py`](../src/retail_portfolio/data/kaggle_retail_insights.py)  
- Demo figure (defaults to this CSV): `PYTHONPATH=src python scripts/visualize_brzezcek_demo.py`  
- Numeric write-up for the latest run: [`figures/brzezcek_demo_summary.md`](../figures/brzezcek_demo_summary.md)

### Download

- [Kaggle dataset page](https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset) (manual), or  
- [Kaggle API](https://www.kaggle.com/docs/api):  
  `kaggle datasets download -d rajneesh231/retail-insights-a-comprehensive-sales-dataset`  
  then unzip and copy `data.csv` into `data/raw/`.

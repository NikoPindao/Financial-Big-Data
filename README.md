# Financial-Big-Data
Financial Big Data Project: clustering cryptocurrencies and identifying market regimes (stable bull, stable bear, volatile bull, volatile bear, consolidation)

<div align="center">
<h1>Cryptocurrency Market Regime Analysis: LSTM and Clustering Approach</h1>
<div>
    Anders Hominal - 297073, Niko Pindao - 314958 & Adam Ezzaim - 325748
</div>
</div>

---

</div>

Overview of our market regime detection system. We combine LSTM neural networks with clustering techniques to identify and predict market regimes in cryptocurrency markets. The system processes price and volume data through feature engineering, applies clustering for regime detection, and uses LSTM for prediction.

## Requirements
Please refer to the requirements.txt file for the required packages.

## Installation
#### Clone the repository (ssh)
```bash
git clone git@github.com:NikoPindao/Financial-Big-Data.git
```

#### Create a virtual environment
```bash
conda create -n <your_env_name> python=3.10
```

#### Activate the virtual environment
```bash
conda activate <your_env_name>
```

#### Install requirements
```bash
pip install -r requirements.txt
```

## Usage
### 1. Data Collection
#### Option 1: Download from Binance API
Download historical data from the Binance API with customizable parameters, you should always follow the same format for the date and time (YYYY-MM-DD HH:MM:SS).
```bash
python main.py --fetch_data True --start_date "2020-01-01 00:00:00" --end_date "2024-12-31 23:59:59" --top_n 100
```
Parameters:
- `start_date`: Start date for data collection (default: "2020-01-01")
- `end_date`: End date for data collection (default: "2024-12-31")
- `top_n`: Number of top cryptocurrencies by volume to analyze (default: 100)

#### Option 2: Use Pre-downloaded Data
You can download the data from [here](https://drive.google.com/drive/folders/1_ZTwo38ZC8DvMJ61xCt6zYiWw192YLgo?usp=sharing), place it in the `data/raw` directory. The data is in parquet format, and the 'data/raw' directory should contain 101 files. You can safely replace the `top_simbols.py`. The pre-downloaded data is for 100 assets for the full period (2020-01-01 to 2024-12-31), this is the same data that we used for the market analysis, the LSTM analysis and the report. 

### 2. Market Analysis
Run clustering market analysis for specific periods:
```bash
python main.py --market_analysis True --period "2023"
```
Available periods:
- `"2023"`: Analysis for year 2023 (2023-01-01 to 2023-12-31)
- `"2024"`: Analysis for year 2024 (2024-01-01 to 2024-12-31)
- `"full_period"`: Complete analysis (2020-01-01 to 2024-12-31)
- `None`: Run analysis for all periods (default)

### 3. LSTM Analysis
Run LSTM analysis and generate prediction metrics, by default it will train on 6 epochs and batch size of 32:
Note that it runs by defaults on BNB, ETH and Bitcoin. Modify directly in the `lstm_analysis.py` file if you want to change the assets.
```bash
python main.py --lstm_analysis True
```

### 4. Complete Pipeline
Run the entire analysis pipeline with custom parameters:
```bash
python main.py --fetch_data True --start_date "2020-01-01 00:00:00" --end_date "2024-12-31 23:59:59" --top_n 100 --market_analysis True --period "full_period" --lstm_analysis True
```

Example combinations:
We ran for the full period with the following command with 100 assets
```bash
# Only fetch recent data
python main.py --fetch_data True --start_date "2020-01-01 00:00:00" --end_date "2024-12-31 23:59:59"
python main.py --market_analysis True --lstm_analysis True

```

## Output Structure
```
crypto-regime-analysis/
├── data/
│   ├── raw/                  # Raw price data
│   └── plots/               # Visualizations
│       ├── market_analysis/  # Clustering and regime plots
│       └── lstm_analysis/    # LSTM prediction plots
├── src/
│   ├── clustering/          # Clustering algorithms
│   ├── data_processing/     # Data preparation
│   ├── models/              # LSTM implementation
├── main.py                  # Main file to run the project
├── market_analysis.py       # Market analysis
├── lstm_analysis.py         # LSTM analysis
├── report.pdf               # Report
```

## Generated Plots
The analysis will generate various plots in the `data/plots` directory:
Note: all plots are in html and needs to be opened in a browser to be displayed.
- Market Analysis (`data/plots/market_analysis/`):
  - Regime distribution plots
  - Clustering visualizations
  - Transition matrices
  - Volatility analysis
- LSTM Analysis (`data/plots/lstm_analysis/`):
  - Performance metrics
  - Training history

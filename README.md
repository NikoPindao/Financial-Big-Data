# Financial-Big-Data
Financial Big Data Project: clustering cryptocurrencies and identifying market regimes (stable bull, stable bear, volatile bull, volatile bear, consolidation)

<div align="center">

<h1>Cryptocurrency Market Regime Analysis: LSTM and Clustering Approach</h1>

<div>
    Anders Hominal - 297073, Niko Pindao - xxxx, Adam Ezzaim - xxx
</div>
mgs/system_overview.jpg"  width="95%" height="100%">
</div>

---

</div>

Overview of our market regime detection system. We combine LSTM neural networks with clustering techniques to identify and predict market regimes in cryptocurrency markets. The system processes price and volume data through feature engineering, applies clustering for regime detection, and uses LSTM for prediction.

## Requirements
Please refer to the requirements.txt file for the required packages.
```

## Installation
```bash
# Clone the repository (ssh)
git clone git@github.com:NikoPindao/Financial-Big-Data.git

# Create a virtual environment
conda create -n <your_env_name> python=3.10

# Activate the virtual environment
conda activate <your_env_name>

# Install requirements
pip install -r requirements.txt
```

## Data Collection
### Download historical data from the binance API from 2020-01-01 to 2024-12-31. It used the top 100 cryptocurrencies traded by volume. We download hourly data.
```bash
python src/data_processing/data_fetching.py
```

### Run clustering market analysis and outplut plots in data/plot/market_analysis directory
```bash
python market_analysis.py
```

### Run lstm analysis and predictions metrics
```bash
python lstm_analysis.py
```

## Directory Structure
```
crypto-regime-analysis/
├── data/
│   ├── raw/                  # Raw price data
│   └── plots/               # Visualizations
├── src/
│   ├── clustering/          # Clustering algorithms
│   ├── data_processing/     # Data preparation
│   ├── models/              # LSTM implementation
```

```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List
import numpy as np

load_dotenv()

DEEPSEEK_API = os.getenv("DEEPSEEK_API")

class FinancialDataPipeline:
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical market data with built-in validation and educational logging.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period to fetch (e.g., '1y' for one year)
            interval: Data interval (e.g., '1d' for daily)
            
        Returns:
            DataFrame containing the historical data, or None if fetch fails
        """
        try:
            self.logger.info(f"Fetching historical data for {symbol}")
            
            # Fetch data using yfinance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
            # Educational: Log the shape and columns of retrieved data
            self.logger.info(f"Retrieved {len(df)} records with columns: {df.columns.tolist()}")
            
            # Validate data quality
            if self._validate_data(df):
                return self._process_raw_data(df)
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the quality of retrieved data.
        
        Educational: Demonstrates important financial data validation checks.
        """
        if df.empty:
            self.logger.warning("Retrieved empty dataset")
            return False
            
        # Check for essential columns
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_columns.issubset(df.columns):
            self.logger.warning(f"Missing required columns. Found: {df.columns}")
            return False
            
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Found missing values: {missing_values[missing_values > 0]}")
            return False
            
        return True
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw financial data and calculate basic technical indicators.
        
        Educational: Shows common financial data transformations.
        """
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate trading volume metrics
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        return df
    
    def get_latest_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch latest fundamental data for educational analysis.
        
        Returns:
            Dictionary containing key fundamental metrics
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key metrics for educational purposes
            fundamentals = {
                'Market_Cap': info.get('marketCap'),
                'PE_Ratio': info.get('trailingPE'),
                'Book_Value': info.get('bookValue'),
                'Dividend_Yield': info.get('dividendYield'),
                'Profit_Margins': info.get('profitMargins')
            }
            
            self.logger.info(f"Retrieved fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return {}

class FundamentalAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def generate_analysis_prompt(self, fundamentals: dict, historical_metrics: dict) -> str:
        prompt = f"""
        Based on these current fundamental metrics:
        - P/E Ratio: {fundamentals.get('PE_Ratio')} (5yr avg: {historical_metrics.get('PE_Ratio_avg')})
        - Book Value: {fundamentals.get('Book_Value')} (5yr growth: {historical_metrics.get('Book_Value_growth')}%)
        - Dividend Yield: {fundamentals.get('Dividend_Yield')} (5yr avg: {historical_metrics.get('Dividend_Yield_avg')})
        - Profit Margins: {fundamentals.get('Profit_Margins')} (5yr trend: {historical_metrics.get('Margin_trend')})
        - Market Cap: {fundamentals.get('Market_Cap')} (5yr CAGR: {historical_metrics.get('MarketCap_CAGR')}%)

        Please analyze this company's valuation and explain:
        1. How current metrics compare to historical averages and what this suggests
        2. Whether recent trends indicate improving or deteriorating fundamentals
        3. Key changes in the company's financial health over the past 5 years
        4. Whether the stock appears overvalued or undervalued given this historical context
        
        Structure your response to be educational, explaining your reasoning for each point.
        """
        return prompt
        
    async def get_analysis(self, fundamentals: dict) -> dict:
        prompt = self.generate_analysis_prompt(fundamentals)
        
        try:
            response = await self.llm.analyze(prompt)
            return {
                'analysis': response,
                'metrics_used': list(fundamentals.keys()),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Error getting LLM analysis: {str(e)}")
            return None


if __name__ == "__main__":
    pipeline = FinancialDataPipeline()

    #print(pipeline.fetch_historical_data("MSFT"))
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import os
import sys
from tabulate import tabulate
from datetime import datetime

# --- WARNA TERMINAL ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class UltimateScreener:
    def __init__(self, ticker_file='tickers.txt'):
        self.ticker_file = ticker_file
        self.ensure_ticker_file_exists()

    def ensure_ticker_file_exists(self):
        """Buat dummy file jika tidak ada"""
        if not os.path.exists(self.ticker_file):
            print(f"{Colors.WARNING}[!] File {self.ticker_file} tidak ditemukan. Membuat default LQ45...{Colors.ENDC}")
            default_tickers = [
                "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP",
                "GOTO", "ADRO", "PGAS", "PTBA", "ANTM", "INCO", "MDKA", "BRIS",
                "KLBF", "INDF", "INKP", "TKIM", "CPIN", "JPFA", "SMGR", "INTP",
                "AMRT", "MAPI", "MEDC", "AKRA", "EXCL", "ISAT", "ACES", "BRPT"
            ]
            with open(self.ticker_file, 'w') as f:
                for t in default_tickers:
                    f.write(f"{t}.JK\n")

    def load_tickers(self):
        with open(self.ticker_file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
            clean_tickers = []
            for t in tickers:
                t = t.replace(" ", "")
                if not t.endswith(".JK"):
                    t = f"{t}.JK"
                clean_tickers.append(t)
        return list(set(clean_tickers))

    def calculate_indicators(self, df):
        # Butuh data minimal 200 candle untuk MA200
        if len(df) < 200: return None
        data = df.copy()

        # --- UPDATE: Menambahkan MA9, MA20, MA50 ---
        data['MA9'] = data['Close'].rolling(window=9).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # RSI (14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        k = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        d = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        data['MACD'] = k - d
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

        # Stochastic (14, 3, 3)
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        denom = high_max - low_min
        denom = denom.replace(0, 0.0001)
        data['Stoch_K'] = 100 * ((data['Close'] - low_min) / denom)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

        # OBV
        data['OBV_Change'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 
                             np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
        data['OBV'] = data['OBV_Change'].cumsum()
        
        return data

    def analyze_behavior(self, df, ticker):
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 0 
        
        # 1. MA Trend (Existing)
        trend_status = "SIDEWAYS"
        if curr['Close'] > curr['MA200']:
            score += 1 
            trend_status = "UP > MA200"
        else:
            trend_status = "DOWN < MA200"
            score -= 1

        # --- UPDATE: MA CROSSING LOGIC ---
        
        # A. MA9 Crossing MA20 (Short Term)
        if prev['MA9'] < prev['MA20'] and curr['MA9'] > curr['MA20']:
            signals.append("MA9 CROSS MA20 (BULL)")
            score += 1.5
        elif prev['MA9'] > prev['MA20'] and curr['MA9'] < curr['MA20']:
            signals.append("MA9 CROSS MA20 (BEAR)")
            score -= 1.5

        # B. MA20 Crossing MA50 (Medium Term)
        if prev['MA20'] < prev['MA50'] and curr['MA20'] > curr['MA50']:
            signals.append("MA20 CROSS MA50 (BULL)")
            score += 2.0
        elif prev['MA20'] > prev['MA50'] and curr['MA20'] < curr['MA50']:
            signals.append("MA20 CROSS MA50 (BEAR)")
            score -= 2.0

        # 2. MACD
        macd_cross = "-"
        if prev['MACD'] < prev['MACD_Signal'] and curr['MACD'] > curr['MACD_Signal']:
            macd_cross = "GOLDEN CROSS"
            signals.append("MACD BULL")
            score += 2
        elif prev['MACD'] > prev['MACD_Signal'] and curr['MACD'] < curr['MACD_Signal']:
            macd_cross = "DEATH CROSS"
            signals.append("MACD BEAR")
            score -= 2

        # 3. RSI
        rsi_val = curr['RSI']
        rsi_desc = f"{rsi_val:.0f}"
        if rsi_val < 30:
            rsi_desc += " (OS)"
            if curr['RSI'] > prev['RSI']:
                signals.append("RSI REV UP")
                score += 1.5
        elif rsi_val > 70:
            rsi_desc += " (OB)"
            if curr['RSI'] < prev['RSI']:
                signals.append("RSI REV DOWN")
                score -= 1.5

        # 4. Stochastic
        stoch_desc = "-"
        if curr['Stoch_K'] < 20 and curr['Stoch_K'] > curr['Stoch_D'] and prev['Stoch_K'] < prev['Stoch_D']:
            stoch_desc = "CROSS UP"
            signals.append("STOCH BULL")
            score += 1
        elif curr['Stoch_K'] > 80 and curr['Stoch_K'] < curr['Stoch_D'] and prev['Stoch_K'] > prev['Stoch_D']:
            stoch_desc = "CROSS DOWN"
            signals.append("STOCH BEAR")
            score -= 1

        # 5. OBV
        obv_trend = "-"
        if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]:
            obv_trend = "UP"
            score += 0.5
        else:
            obv_trend = "DOWN"
            score -= 0.5

        # Action Color
        action = "WAIT"
        color = Colors.ENDC
        if score >= 3.5:
            action = "STRONG BUY"
            color = Colors.GREEN
        elif score >= 1.5:
            action = "BUY"
            color = Colors.CYAN
        elif score <= -3.5:
            action = "STRONG SELL"
            color = Colors.FAIL
        elif score <= -1.5:
            action = "SELL"
            color = Colors.WARNING
        
        def format_vol(n):
            if n > 1_000_000: return f"{n/1_000_000:.1f}M"
            if n > 1_000: return f"{n/1_000:.1f}K"
            return str(n)

        return {
            "Stock": ticker.replace(".JK", ""),
            "Price": f"{curr['Close']:,.0f}",
            "Vol": format_vol(curr['Volume']),
            "Trend": trend_status,
            "OBV": obv_trend,
            "MACD": macd_cross,
            "RSI": rsi_desc,
            "Stoch": stoch_desc,
            "Action": f"{color}{action}{Colors.ENDC}",
            "Signal": ", ".join(signals) if signals else "-",
            "_score": score
        }

    def run(self):
        tickers = self.load_tickers()
        print(f"{Colors.HEADER}=== ULTIMATE IHSG SCREENER ==={Colors.ENDC}")
        print(f"[*] Target: {len(tickers)} saham")
        print(f"[*] Mode: Sungkem (Delay aktif)\n")
        
        results = []
        
        try:
            for i, ticker in enumerate(tickers):
                try:
                    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
                    if df.empty: continue
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        try: df.columns = df.columns.get_level_values(0)
                        except: pass
                    
                    processed_df = self.calculate_indicators(df)
                    if processed_df is not None:
                        res = self.analyze_behavior(processed_df, ticker)
                        results.append(res)
                    
                    # Progress Bar CLI Only
                    sys.stdout.write(f"\rProcessing: {i+1}/{len(tickers)} ({ticker})")
                    sys.stdout.flush()
                    time.sleep(random.uniform(0.2, 0.8))

                except Exception: continue
        except KeyboardInterrupt:
            print("\n[!] Stopped.")
        
        print(f"\n\n{Colors.GREEN}[*] Done!{Colors.ENDC}")
        
        if not results:
            print("No data.")
            return

        # Sorting & Display
        results.sort(key=lambda x: x['_score'], reverse=True)
        top_results = results[:50]
        
        display_data = []
        for r in top_results:
            row = r.copy()
            del row['_score']
            display_data.append(row)

        print("\n" + tabulate(display_data, headers="keys", tablefmt="simple_grid"))

if __name__ == "__main__":
    app = UltimateScreener()
    app.run()
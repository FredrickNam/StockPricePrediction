"""
=============================================================================
 S&P 500 Transformer 모델 평가 및 시각화 (6단계)
 파일명: evaluator.py
 목적  : 테스트 데이터 평가(MSE, MAE, Hit Ratio), 대시보드 시각화,
         최신 데이터를 활용한 동적 추론 기능을 제공합니다.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import Tuple

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_engineering.build_features import add_features
from dataset.dataset_builder import make_sliding_windows, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ModelEvaluator")

class ModelEvaluator:
    """
    학습된 Transformer 모델의 성능 평가, 대시보드 시각화 및 추론을 수행하는 클래스.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        feat_scaler,
        target_scaler,
        device: torch.device = None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.feat_scaler = feat_scaler
        self.target_scaler = target_scaler

    def evaluate(self) -> Tuple[float, float, float]:
        """
        Test DataLoader 를 순회하며 모델의 예측값과 실제 타겟값을 비교합니다.
        역변환(Inverse Transform)을 통해 실제 로그 수익률 스케일에서 오차를 계산합니다.
        
        Returns:
            mse, mae, hit_ratio
        """
        self.model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                
                all_preds.append(y_pred.cpu().numpy())
                all_trues.append(y_batch.numpy())
        
        preds_arr = np.concatenate(all_preds).reshape(-1, 1)
        trues_arr = np.concatenate(all_trues).reshape(-1, 1)

        # 타겟 스케일러를 사용해 원래 로그 수익률로 복원
        preds_inv = self.target_scaler.inverse_transform(preds_arr)
        trues_inv = self.target_scaler.inverse_transform(trues_arr)

        mse = mean_squared_error(trues_inv, preds_inv)
        mae = mean_absolute_error(trues_inv, preds_inv)

        # 적중률 (Directional Accuracy): 부호가 일치하는 비율
        pred_signs = np.sign(preds_inv)
        true_signs = np.sign(trues_inv)
        hit_ratio = np.mean(pred_signs == true_signs) * 100

        logger.info("--- 테스트 데이터 평가 결과 ---")
        logger.info(f"    MSE (Log Return)   : {mse:.6f}")
        logger.info(f"    MAE (Log Return)   : {mae:.6f}")
        logger.info(f"    Hit Ratio (방향성) : {hit_ratio:.2f}%")
        
        # 회귀-분류 브릿지 (F1 Score 및 Confusion Matrix)
        self._compute_classification_bridge(trues_inv, preds_inv)

        return mse, mae, hit_ratio

    def _compute_classification_bridge(self, trues_inv: np.ndarray, preds_inv: np.ndarray):
        """회귀 예측값(로그 수익률)을 이진 시그널로 변환하여 분류 지표 산출"""
        # 실전 트레이딩 마찰 비용(수수료+슬리피지)을 고려한 임계값 설정 (예: 0.1%)
        THRESHOLD = 0.001
        
        # 수익률이 임계값 초과면 1(상승/매수), 이하이면 0(하락/매도/관망)
        true_classes = (trues_inv > THRESHOLD).astype(int)
        pred_classes = (preds_inv > THRESHOLD).astype(int)
        
        f1_macro = f1_score(true_classes, pred_classes, average="macro", zero_division=0)
        cm = confusion_matrix(true_classes, pred_classes)
        
        logger.info("--- 분류 지표 (Classification Bridge) ---")
        logger.info(f"    F1 Score (Macro) : {f1_macro:.4f}")
        logger.info(f"    Confusion Matrix (TN, FP, FN, TP) :\n{cm}")
        
        return f1_macro, cm

    def _compute_sharpe_ratio(self, strategy_returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        모델의 일일 전략 수익률(Strategy Returns) 배열을 입력받아 연율화된 샤프 비율(Sharpe Ratio)을 산출합니다.
        * 1년 거래일을 통상적인 252일로 가정합니다.
        """
        if len(strategy_returns) == 0:
            return 0.0
            
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        
        # 변동성이 0인 경우 (무조건 현금 관망 등) 방어 로직
        if std_return == 0:
            return 0.0
            
        # 일일 샤프 비율 계산
        daily_sharpe = (mean_return - risk_free_rate) / std_return
        
        # 연율화 (Annualization)
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        logger.info(f"--- 퀀트 핵심 지표 (Quant Metrics) ---")
        logger.info(f"    Annualized Sharpe Ratio : {annualized_sharpe:.4f}")
        
        return annualized_sharpe

    def plot_dashboard(self, ticker: str, test_df: pd.DataFrame, seq_len: int = 20):
        """
        특정 종목의 Test 기간 내 수익률, 주가/이평선, RSI 지표를 시각화합니다.
        
        Args:
            ticker  : 시각화할 주식 티커
            test_df : 시간순 분할 후 스케일링이 적용된 테스트 데이터프레임
            seq_len : 슬라이딩 윈도우 길이
        """
        # 1. 대상 종목 필터링
        ticker_df_scaled = test_df[test_df["Ticker"] == ticker].copy()
        if len(ticker_df_scaled) <= seq_len:
            logger.error(f"[{ticker}] 테스트 데이터가 충분하지 않습니다.")
            return

        # 2. 원본 특성 복원 (스케일링 역변환)
        unscaled_feats = self.feat_scaler.inverse_transform(ticker_df_scaled[FEATURE_COLS])
        ticker_df = pd.DataFrame(unscaled_feats, columns=FEATURE_COLS, index=ticker_df_scaled.index)
        ticker_df["Date"] = pd.to_datetime(ticker_df_scaled["Date"])

        # 3. 모델 입력을 위한 시퀀스 윈도우 생성
        X_arr, y_arr = make_sliding_windows(
            ticker_df_scaled, feature_cols=FEATURE_COLS, target_col="Target_Next_Return", seq_len=seq_len
        )
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor)
        
        preds_inv = self.target_scaler.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).flatten()
        trues_inv = self.target_scaler.inverse_transform(y_arr.reshape(-1, 1)).flatten()

        # x축에 사용할 날짜 배열 (윈도우 크기 제외)
        dates = ticker_df["Date"].iloc[seq_len - 1 : len(ticker_df)].values
        
        # 4. 누적 수익률 계산 (실전 거래 비용 고려)
        THRESHOLD = 0.001  # 최소 마찰 비용 (0.1%)
        # 예측값이 임계값보다 커야 매수(실제 수익률 - 비용), 그렇지 않으면 관망(0%)
        strategy_returns = np.where(preds_inv > THRESHOLD, trues_inv - THRESHOLD, 0.0)
        
        # 샤프 비율 계산 및 로깅
        self._compute_sharpe_ratio(strategy_returns)
        
        cum_real = np.exp(np.cumsum(trues_inv))
        cum_strategy = np.exp(np.cumsum(strategy_returns))

        # 5. 다중 패널 시각화
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f"[{ticker}] Model Evaluation Dashboard (Test Period)", fontsize=16, fontweight="bold")

        # --- 패널 1: 누적 수익률 ---
        ax1 = axes[0]
        ax1.plot(dates, cum_real, label="Actual Cumulative Return", color="blue", linewidth=2)
        ax1.plot(dates, cum_strategy, label="Model Strategy Return", color="orange", linewidth=2)
        ax1.set_title("Cumulative Returns Comparison")
        ax1.set_ylabel("Return Multiplier")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- 패널 2: 주가 및 추세 (이동평균) ---
        ax2 = axes[1]
        plot_close = ticker_df["Close"].iloc[seq_len - 1 : len(ticker_df)].values
        plot_sma5  = ticker_df["SMA_5"].iloc[seq_len - 1 : len(ticker_df)].values
        plot_sma20 = ticker_df["SMA_20"].iloc[seq_len - 1 : len(ticker_df)].values

        ax2.plot(dates, plot_close, label="Close Price", color="black", alpha=0.8)
        ax2.plot(dates, plot_sma5, label="SMA 5", color="magenta", alpha=0.6)
        ax2.plot(dates, plot_sma20, label="SMA 20", color="green", alpha=0.6)
        ax2.set_title("Price & Moving Averages")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- 패널 3: 보조 지표 (RSI) ---
        ax3 = axes[2]
        plot_rsi = ticker_df["RSI_14"].iloc[seq_len - 1 : len(ticker_df)].values
        ax3.plot(dates, plot_rsi, label="RSI (14)", color="purple")
        ax3.axhline(70, color="red", linestyle="--", alpha=0.6, label="Overbought (70)")
        ax3.axhline(30, color="blue", linestyle="--", alpha=0.6, label="Oversold (30)")
        ax3.set_title("RSI Indicator")
        ax3.set_ylabel("RSI")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(PROJECT_ROOT, "models", f"{ticker}_dashboard.png")
        plt.savefig(save_path)
        logger.info(f"--- 대시보드 저장 완료: {save_path} ---")

    def predict_tomorrow(
        self,
        ticker: str,
        raw_dir: str = os.path.join(PROJECT_ROOT, "data", "raw"),
        seq_len: int = 20
    ) -> float:
        """
        가장 최근의 원시 데이터를 읽어 전처리를 수행한 후,
        학습된 모델을 통해 다음 날의 수익률을 동적으로 추론합니다.
        
        Args:
            ticker  : 추론할 주식 티커
            raw_dir : 원시 CSV 파일들이 위치한 디렉토리
            seq_len : 모델이 요구하는 시퀀스 길이
            
        Returns:
            내일의 예상 수익률 (%)
        """
        raw_path = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(raw_path):
            logger.error(f"원시 데이터 파일을 찾을 수 없습니다: {raw_path}")
            return 0.0

        df = pd.read_csv(raw_path)
        # 지표(SMA_20 등) 계산을 위해 여유롭게 최근 60일 데이터 추출
        df = df.tail(60).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # 최신 데이터를 기준으로 기술적 지표 생성 (미래 데이터 누수 없음)
        df = add_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 타겟(Target_Next_Return)이 없는 가장 최신일(오늘) 데이터가 삭제되는 것을 방지하기 위해,
        # 오직 모델 입력 피처(FEATURE_COLS)에 결측치가 있는 행만 제거합니다.
        df.dropna(subset=FEATURE_COLS, inplace=True)

        if len(df) < seq_len:
            logger.error(f"전처리 후 데이터가 부족합니다. (필요: {seq_len}, 현재: {len(df)})")
            return 0.0

        # 모델 입력 크기에 맞게 최근 seq_len 일 데이터 슬라이싱
        latest_data = df.tail(seq_len).copy()
        
        # 특성 추출 및 스케일링
        features = latest_data[FEATURE_COLS].to_numpy(dtype=np.float32)
        features_scaled = self.feat_scaler.transform(features)

        # 3D 텐서 변환: [Batch=1, SeqLen=20, Features=11]
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor)
        
        # 모델 출력 역변환
        pred_inv = self.target_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))[0][0]
        
        # 로그 수익률을 일반 백분율(%)로 변환
        expected_return_pct = (np.exp(pred_inv) - 1.0) * 100.0
        
        last_date = latest_data["Date"].iloc[-1].strftime("%Y-%m-%d")
        
        logger.info("--- 동적 추론 (Dynamic Inference) ---")
        logger.info(f"    대상 종목 : {ticker}")
        logger.info(f"    기준 일자 : {last_date} (최근 {seq_len}일 기준)")
        logger.info(f"    내일의 예상 수익률 : {expected_return_pct:+.2f}%")
        
        return expected_return_pct

    def plot_portfolio_backtest(self, test_df: pd.DataFrame, seq_len: int = 20, top_n: int = 5):
        """
        테스트 기간 전체에 대해 매일 상위 N개 종목을 매수(Long)하는 포트폴리오 전략을 백테스트하고 
        S&P 500 동일 비중 수익률(벤치마크)과 비교 시각화합니다.
        """
        import matplotlib.ticker as plticker
        logger.info("전체 종목 대상 포트폴리오 백테스트 준비 중...")
        
        # 1. 원본 데이터프레임에서 날짜와 티커 메타데이터 추출 (make_sliding_windows 와 동일한 순서)
        meta_list = []
        for ticker, group in test_df.groupby("Ticker", sort=False):
            group = group.sort_values("Date")
            n_rows = len(group)
            if n_rows <= seq_len:
                continue
            dates = group["Date"].iloc[seq_len - 1 : n_rows].values
            for d in dates:
                meta_list.append((ticker, d))
                
        # 2. 모델 추론 (test_loader를 통해 전체 예측)
        self.model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                all_preds.append(self.model(X_batch).cpu().numpy())
                all_trues.append(y_batch.numpy())
                
        preds_inv = self.target_scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
        trues_inv = self.target_scaler.inverse_transform(np.concatenate(all_trues).reshape(-1, 1)).flatten()
        
        # 3. 데이터프레임으로 매핑
        results_df = pd.DataFrame(meta_list, columns=["Ticker", "Date"])
        results_df["Pred"] = preds_inv
        results_df["Actual"] = trues_inv
        
        # [수정] 2025년 7월 1일 이후(최근 1년) 데이터만 시각화 대상으로 필터링
        results_df["Date"] = pd.to_datetime(results_df["Date"])
        results_df = results_df[results_df["Date"] >= "2025-07-01"].copy()
        
        # 로그 수익률을 산술 수익률로 변환 (직관적인 백테스트를 위해)
        results_df["Pred_Pct"] = (np.exp(results_df["Pred"]) - 1.0) * 100.0
        results_df["Actual_Ret"] = np.exp(results_df["Actual"]) - 1.0
        
        # 4. 일일 포트폴리오 수익률 계산 로직
        THRESHOLD = 0.001 # 거래 비용 (수수료 및 슬리피지 0.1%)
        
        def calc_daily_long(daily_df):
            longs = daily_df[daily_df["Pred_Pct"] >= 1.0].nlargest(top_n, "Pred_Pct")
            return longs["Actual_Ret"].mean() - THRESHOLD if len(longs) > 0 else 0.0
            
        def calc_daily_short(daily_df):
            shorts = daily_df[daily_df["Pred_Pct"] <= -1.0].nsmallest(top_n, "Pred_Pct")
            return (-shorts["Actual_Ret"]).mean() - THRESHOLD if len(shorts) > 0 else 0.0
            
        def calc_daily_benchmark(daily_df):
            return daily_df["Actual_Ret"].mean()
            
        # 날짜별 일일 수익률 계산
        daily_long = results_df.groupby("Date").apply(calc_daily_long)
        daily_short = results_df.groupby("Date").apply(calc_daily_short)
        daily_benchmark = results_df.groupby("Date").apply(calc_daily_benchmark)
        
        # 롱-숏 혼합 포트폴리오 (자본의 100% 롱, 100% 숏 = 200% Gross Exposure 가정)
        daily_combined = daily_long + daily_short
        
        daily_combined.sort_index(inplace=True)
        daily_long.sort_index(inplace=True)
        daily_short.sort_index(inplace=True)
        daily_benchmark.sort_index(inplace=True)
        dates = daily_combined.index.values
        
        # 누적 수익률(배수) 계산
        # 산술 수익률의 누적은 cumprod(1 + r) 사용
        cum_combined = (1.0 + daily_combined.values).cumprod()
        cum_long = (1.0 + daily_long.values).cumprod()
        cum_short = (1.0 + daily_short.values).cumprod()
        cum_benchmark = (1.0 + daily_benchmark.values).cumprod()
        
        # 5. 시각화 (3단 패널)
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        
        # [상단 패널] 통합 롱숏 포트폴리오 vs 벤치마크
        ax1 = axes[0]
        ax1.plot(dates, cum_combined, label=f"AI Long-Short Combined Portfolio", color="purple", linewidth=2.5)
        ax1.plot(dates, cum_benchmark, label="S&P 500 Equal Weight (Benchmark)", color="gray", linestyle="--", linewidth=2)
        ax1.set_title(f"Cross-Sectional Portfolio Backtest (Test Period)\nLong Top-{top_n} (Pred >= 1.0%) & Short Bottom-{top_n} (Pred <= -1.0%)", fontsize=16, fontweight="bold")
        ax1.set_ylabel("Cumulative Return Multiplier")
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.4)
        
        # [중단 패널] 롱 단독 vs 숏 단독 기여도
        ax2 = axes[1]
        ax2.plot(dates, cum_long, label=f"Long Only (Top-{top_n})", color="red", linewidth=2)
        ax2.plot(dates, cum_short, label=f"Short Only (Bottom-{top_n})", color="blue", linewidth=2)
        ax2.plot(dates, cum_benchmark, label="Benchmark", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.set_title("Long vs Short Portfolio Contribution", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Cumulative Return Multiplier")
        ax2.legend(fontsize=12, loc='upper left')
        ax2.grid(True, alpha=0.4)
        
        # [하단 패널] 30일 롤링 상관계수 (Rolling Correlation)
        ax3 = axes[2]
        roll_corr_long = daily_long.rolling(window=30).corr(daily_benchmark)
        roll_corr_short = daily_short.rolling(window=30).corr(daily_benchmark)
        
        ax3.plot(dates, roll_corr_long, label="Long vs Benchmark (30d)", color="red", linewidth=2)
        ax3.plot(dates, roll_corr_short, label="Short vs Benchmark (30d)", color="blue", linewidth=2)
        ax3.axhline(0, color="black", linestyle="--", alpha=0.8)
        ax3.set_title("30-Day Rolling Correlation with Benchmark", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Pearson Correlation")
        
        ax3.xaxis.set_major_locator(plticker.MaxNLocator(10))
        plt.xticks(rotation=45)
        
        # 전체 기간에 대한 정적 상관계수 계산 및 범례 추가
        corr_long = daily_long.corr(daily_benchmark)
        corr_short = daily_short.corr(daily_benchmark)
        text_str = f"Overall Correlation:\nLong: {corr_long:.2f}\nShort: {corr_short:.2f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax3.text(0.02, 0.95, text_str, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
                 
        ax3.legend(fontsize=12, loc='upper right')
        ax3.grid(True, alpha=0.4)
        
        plt.tight_layout()
        
        save_path = os.path.join(PROJECT_ROOT, "models", "portfolio_backtest_dashboard.png")
        plt.savefig(save_path)
        logger.info(f"--- 전체 포트폴리오 백테스트 대시보드 저장 완료: {save_path} ---")
        
        final_combined = cum_combined[-1] if len(cum_combined) > 0 else 1.0
        final_long = cum_long[-1] if len(cum_long) > 0 else 1.0
        final_short = cum_short[-1] if len(cum_short) > 0 else 1.0
        final_benchmark = cum_benchmark[-1] if len(cum_benchmark) > 0 else 1.0
        
        # Sharpe Ratio 계산 (연환산 252 거래일 가정)
        def calc_sharpe(ret_series):
            if ret_series.std() == 0: return 0.0
            return (ret_series.mean() / ret_series.std()) * np.sqrt(252)
            
        sharpe_combined = calc_sharpe(daily_combined)
        sharpe_long = calc_sharpe(daily_long)
        sharpe_short = calc_sharpe(daily_short)
        sharpe_bench = calc_sharpe(daily_benchmark)
        
        logger.info(f"    AI [통합 롱숏] 최종 수익 배수 : {final_combined:.4f}x (Sharpe: {sharpe_combined:.2f})")
        logger.info(f"    AI [롱 단독]   최종 수익 배수 : {final_long:.4f}x (Sharpe: {sharpe_long:.2f})")
        logger.info(f"    AI [숏 단독]   최종 수익 배수 : {final_short:.4f}x (Sharpe: {sharpe_short:.2f})")
        logger.info(f"    시장(동일비중) 최종 수익 배수 : {final_benchmark:.4f}x (Sharpe: {sharpe_bench:.2f})")

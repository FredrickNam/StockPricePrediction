"""
=============================================================================
 S&P 500 Transformer 독립 평가 스크립트
 파일명: run_evaluation.py
 실행  : .\venv\Scripts\python.exe run_evaluation.py

 목적:
   학습이 완료되어 저장된 모델 가중치를 로드하고,
   평가 지표 산출, 대시보드 시각화, 다음 날 예측(동적 추론)을 독립적으로 실행합니다.
=============================================================================
"""

import sys
import os
import logging
import torch

# 프로젝트 루트를 import 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.dataset_builder  import build_dataloaders
from models.transformer_model import TimeSeriesTransformer
from evaluation.evaluator     import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvalRunner")

# =============================================================================
# ▶ 설정값 (main.py 와 동일하게 맞춰야 모델 로드가 정상 작동합니다)
# =============================================================================
D_MODEL        = 64
NHEAD          = 4
NUM_LAYERS     = 2
DIM_FEEDFORWARD = 256
DROPOUT        = 0.1

SEQ_LEN        = 20
BATCH_SIZE     = 64
TARGET_TICKER  = "SWKS"  # 분석 및 예측을 시각화할 타겟 종목

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_transformer.pth")

from typing import List, Tuple

def extract_cross_sectional_portfolio(
    evaluator, 
    tickers: List[str], 
    top_n: int = 10, 
    seq_len: int = 20
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    모든 대상 종목의 내일 예상 수익률을 추론하고 정렬하여,
    가장 기대 수익이 높은 Top N(Long)과 낮은 Bottom N(Short)을 반환합니다.
    """
    predictions = []
    
    # 임시로 로거 레벨을 올려 개별 종목 추론 로그 도배 방지
    original_level = logging.getLogger("ModelEvaluator").level
    logging.getLogger("ModelEvaluator").setLevel(logging.WARNING)
    
    for ticker in tickers:
        try:
            expected_ret = evaluator.predict_tomorrow(ticker=ticker, seq_len=seq_len)
            predictions.append((ticker, expected_ret))
        except Exception:
            continue
            
    logging.getLogger("ModelEvaluator").setLevel(original_level)
            
    # 수익률 기준 내림차순 정렬
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    if len(predictions) < top_n * 2:
        top_n = max(1, len(predictions) // 2)
        
    # 매수(Long) 조건: 예상 수익률이 1.0% 이상인 종목만 선별
    long_candidates = [p for p in predictions if p[1] >= 1.0]
    long_portfolio = long_candidates[:top_n] if long_candidates else []
    
    # 공매도(Short) 조건: 예상 수익률이 -1.0% 이하(즉 1% 이상 하락)인 종목만 선별
    short_candidates = [p for p in predictions if p[1] <= -1.0]
    short_portfolio = short_candidates[-top_n:] if short_candidates else []
    
    logger_cs = logging.getLogger("CrossSectionalRanker")
    logger_cs.info(f"=== 횡단면 포트폴리오 랭킹 (Top/Bottom {top_n}) ===")
    logger_cs.info("--- 매수 (Long) 포트폴리오 ---")
    if long_portfolio:
        for t, ret in long_portfolio:
            logger_cs.info(f"    {t:<5} : {ret:+.2f}%")
    else:
        logger_cs.info("    [조건 미달] 1.0% 이상 상승할 것으로 확신하는 종목이 없습니다.")
        
    logger_cs.info("--- 공매도 (Short) 포트폴리오 ---")
    if short_portfolio:
        for t, ret in short_portfolio:
            logger_cs.info(f"    {t:<5} : {ret:+.2f}%")
    else:
        logger_cs.info("    [조건 미달] -1.0% 이상 하락할 것으로 확신하는 종목이 없습니다.")
        
    return long_portfolio, short_portfolio

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(" S&P 500 Transformer 독립 평가 시작")
    logger.info("=" * 60)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"저장된 모델을 찾을 수 없습니다: {MODEL_PATH}")
        logger.error("main.py 를 먼저 실행하여 모델을 학습시켜주세요.")
        sys.exit(1)

    # 1. 데이터 로더 준비 (평가를 위해 스케일러와 Test Data 필요)
    logger.info("데이터 로더 및 스케일러 준비 중...")
    result = build_dataloaders(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    test_loader   = result["test_loader"]
    feat_scaler   = result["feat_scaler"]
    target_scaler = result["target_scaler"]
    n_features    = result["n_features"]
    test_df       = result["test_df"]

    # 2. 모델 구조 초기화 및 가중치 로드
    logger.info("모델 가중치 로드 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        n_features=n_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )
    
    # 학습 완료된 가중치 및 스케일러 적용 (체크포인트 무결성)
    import pickle
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("feat_scaler"):
            feat_scaler = pickle.loads(checkpoint["feat_scaler"])
        if checkpoint.get("target_scaler"):
            target_scaler = pickle.loads(checkpoint["target_scaler"])
    else:
        model.load_state_dict(checkpoint)
    
    # 3. ModelEvaluator 초기화
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        feat_scaler=feat_scaler,
        target_scaler=target_scaler,
        device=device
    )
    
    logger.info("-" * 60)
    # [A] 테스트 데이터 전반 평가
    evaluator.evaluate()
    
    logger.info("-" * 60)
    # [B] 단일 종목 시각화 대시보드 저장
    evaluator.plot_dashboard(ticker=TARGET_TICKER, test_df=test_df, seq_len=SEQ_LEN)
    
    logger.info("-" * 60)
    # [C] 최신 데이터를 이용한 내일 수익률 추론
    evaluator.predict_tomorrow(ticker=TARGET_TICKER, seq_len=SEQ_LEN)

    logger.info("-" * 60)
    # [D] 횡단면 랭킹 (Cross-sectional Ranking)
    # 테스트를 위해 test_df에 존재하는 고유 티커 목록 사용
    unique_tickers = test_df["Ticker"].unique().tolist()
    logger.info(f"전체 {len(unique_tickers)}개 종목에 대한 횡단면 랭킹 분석 중...")
    extract_cross_sectional_portfolio(evaluator, unique_tickers, top_n=5, seq_len=SEQ_LEN)
    
    logger.info("-" * 60)
    # [E] 전체 종목 포트폴리오 백테스트 대시보드 저장 및 성과 분석 (Top 5 전략)
    logger.info("과거 백테스트 포트폴리오 성과 및 대시보드 분석 중...")
    evaluator.plot_portfolio_backtest(test_df=test_df, seq_len=SEQ_LEN, top_n=5)

    logger.info("=" * 60)
    logger.info(" 독립 평가 프로세스 완료.")
    logger.info("=" * 60)

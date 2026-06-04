"""
=============================================================================
 S&P 500 Transformer 점진적 학습 (Online/Incremental Learning) 파이프라인
 파일명: online_learning.py
 실행  : .\venv\Scripts\python.exe online_learning.py

 목적:
   매일 새로운 종가 데이터가 수집되었을 때, 전체 데이터를 처음부터 학습하지 않고
   최신 데이터(예: 최근 n일치)만으로 기존 모델 가중치를 파인튜닝(미세조정)합니다.
=============================================================================
"""

import sys
import os
import logging
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.dataset_builder  import build_dataloaders
from models.transformer_model import TimeSeriesTransformer
from training.trainer         import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OnlineLearning")

# =============================================================================
# ▶ 설정값 (기존 하이퍼파라미터 및 경로 유지)
# =============================================================================
D_MODEL        = 64
NHEAD          = 4
NUM_LAYERS     = 2
DIM_FEEDFORWARD = 256
DROPOUT        = 0.1

SEQ_LEN        = 20
BATCH_SIZE     = 64
FINE_TUNE_EPOCHS = 2  # 점진적 업데이트 시 적은 에포크 수 사용

NEW_DATA_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "sp500_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_transformer.pth")

def run_incremental_learning():
    logger.info("=" * 60)
    logger.info(" S&P 500 Transformer 일일 점진적 학습 시작")
    logger.info("=" * 60)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"저장된 모델을 찾을 수 없습니다: {MODEL_PATH}")
        logger.error("초기 학습(main.py)이 선행되어야 합니다.")
        sys.exit(1)

    # 1. 신규 데이터가 포함된 CSV 로드 및 DataLoader 생성
    # 참고: 점진적 학습 시에도 스케일러 및 데이터 구조 유지를 위해 build_dataloaders 재사용
    # (실제 환경에서는 최근 며칠 치 데이터만 잘라내어 사용하는 필터링 로직이 추가될 수 있음)
    logger.info("신규 데이터 로더 준비 중...")
    result = build_dataloaders(
        processed_csv=NEW_DATA_CSV, 
        seq_len=SEQ_LEN, 
        batch_size=BATCH_SIZE,
        train_ratio=0.90, # 최신 데이터 대부분을 훈련에 사용
        val_ratio=0.10
    )

    # 2. 모델 초기화
    logger.info("모델 아키텍처 초기화 중...")
    model = TimeSeriesTransformer(
        n_features=result["n_features"],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )

    # 3. Trainer 생성 (새로운 데이터 로더 연결)
    trainer = ModelTrainer(
        model=model,
        train_loader=result["train_loader"],
        val_loader=result["val_loader"],
        save_path=MODEL_PATH,
        feat_scaler=result["feat_scaler"],
        target_scaler=result["target_scaler"]
    )

    # 4. 무결성 체크포인트 복원
    logger.info("기존 체크포인트(가중치/옵티마이저/스케일러) 복원 중...")
    try:
        start_epoch = trainer.load_checkpoint_for_incremental(MODEL_PATH)
        logger.info(f"기존 학습 상태 복원 완료. (이전 마지막 Epoch: {start_epoch - 1})")
    except Exception as e:
        logger.error(f"체크포인트 복원 실패: {e}")
        sys.exit(1)

    # 5. 미세 조정 (Fine-tuning) 진행
    trainer.epochs = start_epoch + FINE_TUNE_EPOCHS - 1
    logger.info(f"신규 데이터 파인튜닝 시작 (추가 {FINE_TUNE_EPOCHS} Epochs)...")

    for epoch in range(start_epoch, trainer.epochs + 1):
        train_loss = trainer._train_one_epoch(epoch)
        val_loss = trainer._validate_one_epoch()
        stop = trainer._check_early_stopping(val_loss, epoch)
        if stop:
            logger.info("조기 종료 조건 만족으로 파인튜닝 조기 중단.")
            break

    logger.info("=" * 60)
    logger.info(" 점진적 학습 파이프라인(Online Learning) 완료 및 모델 갱신 성공.")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_incremental_learning()

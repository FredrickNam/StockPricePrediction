"""
=============================================================================
 S&P 500 Transformer 학습 파이프라인 통합 진입점
 파일명: main.py
 실행  : .\venv\Scripts\python.exe main.py
         (또는 VS Code 에서 이 파일을 열고 실행 버튼 클릭)

 전체 실행 순서:
   1. 전처리 데이터 로드 (캐시 재활용)
   2. 스케일링 + 슬라이딩 윈도우 + DataLoader 생성
   3. Transformer 모델 초기화
   4. 학습 + 조기 종료 + 최적 모델 저장
=============================================================================
"""

import sys
import os
import logging

# 프로젝트 루트를 import 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dataset.dataset_builder  import build_dataloaders
from models.transformer_model import TimeSeriesTransformer
from training.trainer         import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Main")


# =============================================================================
# ▶ 하이퍼파라미터 설정 (여기서 한 번에 조정)
# =============================================================================
# -- 모델 구조 --
D_MODEL        = 64     # Transformer 내부 임베딩 차원
NHEAD          = 4      # Multi-Head Attention 헤드 수 (D_MODEL 의 약수)
NUM_LAYERS     = 2      # Encoder 레이어 수
DIM_FEEDFORWARD = 256   # FFN 은닉 차원
DROPOUT        = 0.1    # Dropout 비율

# -- 학습 설정 --
LR           = 1e-3     # AdamW 초기 학습률
WEIGHT_DECAY = 1e-4     # L2 규제 계수
L1_ALPHA     = 1e-5     # 커스텀 L1 페널티 계수
EPOCHS       = 100      # 최대 학습 에포크
PATIENCE     = 10       # 조기 종료 인내 횟수
BATCH_SIZE   = 64       # 미니배치 크기
SEQ_LEN      = 20       # 슬라이딩 윈도우 길이


# =============================================================================
# ▶ 메인 실행
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(" S&P 500 Transformer 학습 파이프라인 시작")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1+2: 데이터 로드 + 스케일링 + DataLoader
    # ------------------------------------------------------------------
    logger.info("[Step 1/3] 데이터 로드 및 DataLoader 생성...")
    result = build_dataloaders(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    train_loader  = result["train_loader"]
    val_loader    = result["val_loader"]
    test_loader   = result["test_loader"]
    target_scaler = result["target_scaler"]
    n_features    = result["n_features"]

    # ------------------------------------------------------------------
    # Step 2: 모델 초기화
    # ------------------------------------------------------------------
    logger.info("[Step 2/3] Transformer 모델 초기화...")
    model = TimeSeriesTransformer(
        n_features=n_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )
    logger.info(f"    학습 파라미터 수: {model.count_parameters():,}")

    # ------------------------------------------------------------------
    # Step 3: 학습
    # ------------------------------------------------------------------
    logger.info("[Step 3/3] 학습 시작...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        l1_alpha=L1_ALPHA,
        epochs=EPOCHS,
        patience=PATIENCE,
        feat_scaler=result["feat_scaler"],
        target_scaler=result["target_scaler"]
    )
    history = trainer.fit()

    # ------------------------------------------------------------------
    # 완료: 최적 모델 로드 확인
    # ------------------------------------------------------------------
    best_model = trainer.load_best_model()
    logger.info("=" * 60)
    logger.info(" 파이프라인 완료. 최적 모델: models/best_transformer.pth")
    logger.info("=" * 60)

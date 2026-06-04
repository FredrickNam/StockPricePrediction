"""
=============================================================================
 S&P 500 Transformer 학습용 데이터셋 빌더 (3단계: 스케일링 및 텐서 변환)
 파일명: dataset_builder.py
 목적  : 전처리 완료된 sp500_features.csv 를 읽어
         시간순 분할 -> 스케일링 -> 슬라이딩 윈도우 -> PyTorch DataLoader
         순서로 변환하는 파이프라인을 제공합니다.
 실행  : python dataset/dataset_builder.py          (동작 테스트)
 임포트: from dataset.dataset_builder import build_dataloaders
 의존성: torch, numpy, pandas, scikit-learn
=============================================================================
"""

import os
import sys
import logging
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

# =============================================================================
# ▶ 경로 상수
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "sp500_features.csv")
SCALER_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "scalers")

# =============================================================================
# ▶ 하이퍼파라미터 상수
# =============================================================================
SEQ_LEN    = 20      # 슬라이딩 윈도우 길이 (약 한 달치 거래일)
BATCH_SIZE = 64      # 미니배치 크기

TRAIN_RATIO = 0.70   # 학습 데이터 비율
VAL_RATIO   = 0.15   # 검증 데이터 비율
# TEST_RATIO  = 0.15 (나머지 전체)

# 모델 입력 특성 컬럼 목록 (스케일링 대상)
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Trading_Value", "Log_Return",
    "RSI_14", "SMA_5", "SMA_20", "Volatility_20",
]

# 타겟 변수 (내일의 로그 수익률 - 별도 스케일러로 관리)
TARGET_COL = "Target_Next_Return"

# =============================================================================
# ▶ 로거 설정
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetBuilder")


# =============================================================================
# ▶ Step 1. 시간순 데이터 분할 (Chronological Split)
# =============================================================================
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 DataFrame 을 고유 날짜 기준으로 Train / Validation / Test 로 분할합니다.

    --- 설계 원칙 ---
    * 시계열 데이터는 미래 정보가 과거 학습에 유입되는 Leakage 를 반드시 차단해야
      합니다. 따라서 random shuffle 이 포함된 분할 방식(예: train_test_split)을
      절대 사용하지 않습니다.
    * 날짜(Date) 고유값을 오름차순 정렬한 뒤 지정 비율로 인덱스를 잘라
      날짜 집합을 구성하고, 해당 날짜에 속한 행을 필터링합니다.
    * 행 수가 아닌 날짜 수 기준으로 분할하는 이유:
      같은 날짜에 여러 종목(Ticker) 이 존재하므로, 행 수 기준 분할은 특정 날짜의
      일부 종목이 Train 에, 나머지가 Val 에 들어가는 불일치를 야기합니다.

    Args:
        df         : 전체 전처리 DataFrame (Date 컬럼 포함)
        train_ratio: 학습 비율 (기본 0.70)
        val_ratio  : 검증 비율 (기본 0.15)

    Returns:
        (train_df, val_df, test_df) 튜플
    """
    # 고유 날짜를 오름차순으로 정렬
    unique_dates: np.ndarray = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)

    # 각 분할의 마지막 날짜 인덱스를 계산
    train_end = int(n_dates * train_ratio)
    val_end   = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_end])
    val_dates   = set(unique_dates[train_end:val_end])
    test_dates  = set(unique_dates[val_end:])

    train_df = df[df["Date"].isin(train_dates)].copy()
    val_df   = df[df["Date"].isin(val_dates)].copy()
    test_df  = df[df["Date"].isin(test_dates)].copy()

    logger.info(
        "--- 시간순 분할 완료 ---\n"
        f"    Train : {train_df['Date'].min()} ~ {train_df['Date'].max()}"
        f"  ({len(train_df):>8,} 행, {len(train_dates)} 거래일)\n"
        f"    Val   : {val_df['Date'].min()} ~ {val_df['Date'].max()}"
        f"  ({len(val_df):>8,} 행, {len(val_dates)} 거래일)\n"
        f"    Test  : {test_df['Date'].min()} ~ {test_df['Date'].max()}"
        f"  ({len(test_df):>8,} 행, {len(test_dates)} 거래일)"
    )

    return train_df, val_df, test_df


# =============================================================================
# ▶ Step 2. 스케일링 (Data Leakage 완벽 차단)
# =============================================================================
def fit_and_scale(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    target_col:   str  = TARGET_COL,
    save_dir:     str  = SCALER_SAVE_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler, RobustScaler]:
    """
    Train 데이터에만 fit 하여 스케일러를 학습하고,
    Validation 과 Test 에는 transform 만 적용합니다.

    --- Leakage 차단 원칙 ---
    * fit_transform 은 Train 에만 적용합니다.
      Val / Test 에 fit 을 적용하면 미래 데이터의 통계치(평균, 표준편차)가
      스케일러에 유입되어 모델 평가가 낙관적으로 왜곡됩니다.
    * 타겟 변수(Target_Next_Return) 는 모델 출력값 복원(Inverse Transform)을
      위해 별도의 스케일러(target_scaler) 로 관리합니다.
      이를 통해 예측 결과를 원래 수익률 단위(%)로 해석할 수 있습니다.

    Args:
        train_df    : 학습용 DataFrame
        val_df      : 검증용 DataFrame
        test_df     : 테스트용 DataFrame
        feature_cols: 스케일링 대상 특성 컬럼 목록
        target_col  : 타겟 컬럼명
        save_dir    : 스케일러 직렬화 저장 디렉토리

    Returns:
        (scaled_train, scaled_val, scaled_test, feat_scaler, target_scaler)
    """
    feat_scaler   = RobustScaler()
    target_scaler = RobustScaler()

    # --- Train: fit + transform ---
    train_df = train_df.copy()
    train_df[feature_cols] = feat_scaler.fit_transform(train_df[feature_cols])
    train_df[[target_col]] = target_scaler.fit_transform(train_df[[target_col]])

    # --- Val / Test: transform only (스케일러 재학습 금지) ---
    val_df = val_df.copy()
    val_df[feature_cols]  = feat_scaler.transform(val_df[feature_cols])
    val_df[[target_col]]  = target_scaler.transform(val_df[[target_col]])

    test_df = test_df.copy()
    test_df[feature_cols]  = feat_scaler.transform(test_df[feature_cols])
    test_df[[target_col]]  = target_scaler.transform(test_df[[target_col]])

    # --- 스케일러 저장 (학습 완료 후 역변환 및 서빙에 재사용) ---
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "feat_scaler.pkl"), "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(os.path.join(save_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    logger.info(
        f"--- 스케일링 완료 (StandardScaler) ---\n"
        f"    특성 컬럼 수 : {len(feature_cols)}개\n"
        f"    스케일러 저장 : {save_dir}"
    )

    return train_df, val_df, test_df, feat_scaler, target_scaler


# =============================================================================
# ▶ Step 3. 슬라이딩 윈도우 시퀀스 생성
# =============================================================================
def make_sliding_windows(
    df:           pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    target_col:   str  = TARGET_COL,
    seq_len:      int  = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    종목(Ticker) 별로 슬라이딩 윈도우를 생성하여 X, y 배열을 반환합니다.

    --- 윈도우 구조 ---
    * 입력(X): t-19 일 ~ t 일 까지의 특성 행렬  -> 형태: (seq_len, n_features)
    * 타겟(y): t 일의 Target_Next_Return        -> 형태: 스칼라
      (Target_Next_Return = t+1 일의 Log_Return, 즉 '내일' 수익률)

    --- 종목 간 오염 방지 ---
    * Ticker 별로 그룹화하여 독립적으로 윈도우를 생성합니다.
    * 종목 A 의 마지막 행과 종목 B 의 첫 번째 행이 하나의 윈도우에 섞이면
      전혀 다른 종목의 과거 패턴이 입력으로 사용되는 오류가 발생합니다.

    Args:
        df          : 스케일링된 단일 분할(Train / Val / Test) DataFrame
        feature_cols: 입력 특성 컬럼 목록
        target_col  : 타겟 컬럼명
        seq_len     : 슬라이딩 윈도우 길이 (기본 20일)

    Returns:
        X: shape (N, seq_len, n_features) 의 numpy 배열
        y: shape (N,) 의 numpy 배열
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    # 종목 단위로 반복 (날짜 순서 유지)
    for ticker, group in df.groupby("Ticker", sort=False):
        group = group.sort_values("Date")

        feat_arr   = group[feature_cols].to_numpy(dtype=np.float32)  # (T, F)
        target_arr = group[target_col].to_numpy(dtype=np.float32)    # (T,)

        n_rows = len(feat_arr)

        # 시퀀스 길이보다 데이터가 부족한 종목은 건너뜀
        if n_rows <= seq_len:
            logger.debug(f"[{ticker}] 데이터 부족 ({n_rows}행) -> 윈도우 건너뜀")
            continue

        # 슬라이딩 윈도우 생성
        # [수정] range(seq_len - 1, n_rows - 1) -> range(seq_len - 1, n_rows)
        # 마지막 행(최신 거래일)까지 완벽히 추출합니다.
        for i in range(seq_len - 1, n_rows):
            X_list.append(feat_arr[i - seq_len + 1 : i + 1])
            y_list.append(target_arr[i])

    if not X_list:
        raise ValueError(
            "슬라이딩 윈도우가 하나도 생성되지 않았습니다. "
            "seq_len 또는 데이터 행 수를 확인하십시오."
        )

    X = np.stack(X_list, axis=0)  # (N, seq_len, F)
    y = np.array(y_list)          # (N,)

    return X, y


# =============================================================================
# ▶ PyTorch 커스텀 Dataset 클래스
# =============================================================================
class StockSequenceDataset(Dataset):
    """
    슬라이딩 윈도우 기반 주식 시계열 PyTorch Dataset.

    --- 구조 ---
    * __len__     : 전체 윈도우 개수를 반환합니다.
    * __getitem__ : 인덱스에 해당하는 (X_window, y_target) 쌍을 반환합니다.
                    X_window 형태: (seq_len, n_features) float32 Tensor
                    y_target 형태: 스칼라 float32 Tensor

    Args:
        X (np.ndarray): 입력 배열, shape (N, seq_len, n_features)
        y (np.ndarray): 타겟 배열, shape (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # numpy 배열을 float32 PyTorch Tensor 로 변환하여 메모리에 보관
        self.X: Tensor = torch.from_numpy(X.astype(np.float32))  # (N, seq_len, F)
        self.y: Tensor = torch.from_numpy(y.astype(np.float32))  # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


# =============================================================================
# ▶ 메인 빌더 함수 (외부 임포트 진입점)
# =============================================================================
def build_dataloaders(
    processed_csv: str = PROCESSED_CSV,
    seq_len:       int = SEQ_LEN,
    batch_size:    int = BATCH_SIZE,
    train_ratio:   float = TRAIN_RATIO,
    val_ratio:     float = VAL_RATIO,
    feature_cols:  list = FEATURE_COLS,
    target_col:    str  = TARGET_COL,
    num_workers:   int  = 0,
) -> Dict:
    """
    전처리 CSV -> DataLoader 까지의 전체 파이프라인을 실행합니다.

    호출 예시:
        from dataset.dataset_builder import build_dataloaders
        result = build_dataloaders()
        train_loader = result["train_loader"]
        val_loader   = result["val_loader"]
        test_loader  = result["test_loader"]
        target_scaler = result["target_scaler"]

    Args:
        processed_csv : 전처리 완료된 CSV 파일 경로
        seq_len       : 슬라이딩 윈도우 길이 (기본 20)
        batch_size    : DataLoader 배치 크기 (기본 64)
        train_ratio   : 학습 비율 (기본 0.70)
        val_ratio     : 검증 비율 (기본 0.15)
        feature_cols  : 입력 특성 컬럼 목록
        target_col    : 타겟 컬럼명
        num_workers   : DataLoader worker 수 (Windows 는 0 권장)

    Returns:
        Dict {
            "train_loader"  : Train DataLoader  (shuffle=True),
            "val_loader"    : Val DataLoader    (shuffle=False),
            "test_loader"   : Test DataLoader   (shuffle=False),
            "feat_scaler"   : 특성 StandardScaler,
            "target_scaler" : 타겟 StandardScaler (역변환에 사용),
            "n_features"    : 특성 차원 수 (모델 입력 크기),
        }
    """
    # ------------------------------------------------------------------
    # 1. CSV 로드
    # ------------------------------------------------------------------
    logger.info(f"CSV 로드 중: {processed_csv}")
    df = pd.read_csv(processed_csv, parse_dates=["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    logger.info(f"전체 데이터: {len(df):,} 행 x {len(df.columns)} 열")

    # ------------------------------------------------------------------
    # 2. 시간순 분할 (Chronological Split)
    # ------------------------------------------------------------------
    train_df, val_df, test_df = chronological_split(df, train_ratio, val_ratio)

    # ------------------------------------------------------------------
    # 3. 스케일링 (Train fit -> Val/Test transform only)
    # ------------------------------------------------------------------
    train_df, val_df, test_df, feat_scaler, target_scaler = fit_and_scale(
        train_df, val_df, test_df, feature_cols, target_col
    )

    # ------------------------------------------------------------------
    # 4. 종목별 슬라이딩 윈도우 생성
    # ------------------------------------------------------------------
    logger.info("슬라이딩 윈도우 생성 중 (seq_len=%d) ...", seq_len)

    X_train, y_train = make_sliding_windows(train_df, feature_cols, target_col, seq_len)
    X_val,   y_val   = make_sliding_windows(val_df,   feature_cols, target_col, seq_len)
    X_test,  y_test  = make_sliding_windows(test_df,  feature_cols, target_col, seq_len)

    logger.info(
        "--- 윈도우 생성 완료 ---\n"
        f"    Train  X: {X_train.shape}  y: {y_train.shape}\n"
        f"    Val    X: {X_val.shape}    y: {y_val.shape}\n"
        f"    Test   X: {X_test.shape}   y: {y_test.shape}"
    )

    # ------------------------------------------------------------------
    # 5. PyTorch Dataset 생성
    # ------------------------------------------------------------------
    train_dataset = StockSequenceDataset(X_train, y_train)
    val_dataset   = StockSequenceDataset(X_val,   y_val)
    test_dataset  = StockSequenceDataset(X_test,  y_test)

    # ------------------------------------------------------------------
    # 6. DataLoader 생성
    #    - Train : shuffle=True  -> 미니배치 학습 다양성 확보 (종목 순서 무작위)
    #    - Val   : shuffle=False -> 평가 시 시계열 순서 유지 (재현성 보장)
    #    - Test  : shuffle=False -> 동일 이유
    #    * drop_last=True  (Train): 마지막 불완전 배치가 Batch Norm 등에 영향을
    #      주지 않도록 제거합니다.
    #    * pin_memory=True: GPU 전송 속도 개선 (CUDA 사용 시 자동 활성화)
    # ------------------------------------------------------------------
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    n_features = X_train.shape[2]

    logger.info(
        "--- DataLoader 생성 완료 ---\n"
        f"    배치 크기  : {batch_size}\n"
        f"    특성 차원  : {n_features}\n"
        f"    Train 배치 : {len(train_loader)} 개\n"
        f"    Val   배치 : {len(val_loader)} 개\n"
        f"    Test  배치 : {len(test_loader)} 개\n"
        f"    CUDA       : {torch.cuda.is_available()}"
    )

    return {
        "train_loader"  : train_loader,
        "val_loader"    : val_loader,
        "test_loader"   : test_loader,
        "feat_scaler"   : feat_scaler,
        "target_scaler" : target_scaler,
        "n_features"    : n_features,
        "test_df"       : test_df,
    }


# =============================================================================
# ▶ 동작 검증 (직접 실행 시)
# =============================================================================
if __name__ == "__main__":
    result = build_dataloaders()

    train_loader   = result["train_loader"]
    target_scaler  = result["target_scaler"]

    # 첫 번째 배치 형태 확인
    X_batch, y_batch = next(iter(train_loader))
    logger.info(
        "--- 배치 샘플 확인 ---\n"
        f"    X 형태 : {tuple(X_batch.shape)}  (Batch, SeqLen, Features)\n"
        f"    y 형태 : {tuple(y_batch.shape)}  (Batch,)\n"
        f"    X dtype: {X_batch.dtype}\n"
        f"    y dtype: {y_batch.dtype}"
    )

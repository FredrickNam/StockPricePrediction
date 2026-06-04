"""
=============================================================================
 S&P 500 딥러닝(Transformer) 학습용 특성 공학(Feature Engineering) 파이프라인
 파일명: build_features.py
 목적  : ./data/raw/ 내 개별 종목 CSV를 읽어 기술적 지표·수익률·타겟 변수를
         생성하고, ./data/processed/sp500_features.csv 로 저장합니다.
 실행  : python feature_engineering/build_features.py          # 캐시 재활용
         python feature_engineering/build_features.py --force  # 강제 재처리
 의존성: pandas, numpy, pandas_ta
=============================================================================
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore  # 타입 스텁 없음 (정상 동작)

# =============================================================================
# ▶ 경로 상수 정의
# =============================================================================
# 스크립트 위치를 기준으로 프로젝트 루트를 자동 탐색
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "sp500_features.csv")

# =============================================================================
# ▶ 로거 설정
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FeatureEngineering")


# =============================================================================
# ▶ 단일 종목 CSV 로드 함수
# =============================================================================
def load_ticker_csv(filepath: str) -> pd.DataFrame | None:
    """
    단일 종목 CSV 파일을 읽어 날짜 정렬된 DataFrame으로 반환합니다.

    Args:
        filepath (str): CSV 파일 절대 경로

    Returns:
        pd.DataFrame | None: 로드 성공 시 DataFrame, 실패 시 None
    """
    ticker = os.path.splitext(os.path.basename(filepath))[0]

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"[{ticker}] CSV 읽기 실패: {e}")
        return None

    # 파일이 헤더만 있거나 비어있는 경우 건너뜀 (예: LANC.csv = 81 bytes)
    if df.empty or len(df) < 30:
        logger.warning(f"[{ticker}] 데이터 행 수 부족 ({len(df)}행) → 건너뜀")
        return None

    # 'Date' 컬럼 파싱 및 정렬
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 티커 컬럼 추가
    df["Ticker"] = ticker

    return df


# =============================================================================
# ▶ 파생 특성(Feature) 생성 함수
# =============================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    단일 종목 DataFrame에 기술적 지표·수익률·타겟 변수를 추가합니다.
    (호출부에서 단일 종목씩 넘겨주므로 groupby 오버헤드 없이 고속 연산 수행)
    """
    df = df.copy()

    # 1. 거래대금
    df["Trading_Value"] = df["Close"] * df["Volume"]

    # 2. 일일 로그 수익률
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 3. 기술적 지표 (pandas-ta)
    df.ta.rsi(close="Close", length=14, append=True)
    df.ta.sma(close="Close", length=5, append=True)
    df.ta.sma(close="Close", length=20, append=True)
    df["Volatility_20"] = df["Log_Return"].rolling(window=20).std()

    df.rename(columns={"RSI_14": "RSI_14", "SMA_5": "SMA_5", "SMA_20": "SMA_20"}, inplace=True)

    # 4. 타겟 변수 (내일의 로그 수익률)
    df["Target_Next_Return"] = np.log(df["Close"].shift(-1) / df["Close"])

    return df


# =============================================================================
# ▶ 전체 파이프라인 실행 함수
# =============================================================================
def load_cached(output_file: str) -> pd.DataFrame | None:
    """
    이미 전처리된 sp500_features.csv가 존재하면 로드하여 반환합니다.
    단, data/raw/ 내의 최신 파일이 전처리 결과 파일보다 새로우면 캐시를 무시합니다.
    --force 플래그가 있으면 무조건 None을 반환하여 강제 재처리를 유도합니다.

    Args:
        output_file (str): 전처리 결과 CSV 경로

    Returns:
        pd.DataFrame | None: 캐시 히트 시 DataFrame, 미스 시 None
    """
    force = "--force" in sys.argv

    if force:
        logger.info("--force 플래그 감지 → 기존 캐시 무시, 전체 재처리합니다.")
        return None

    if not os.path.exists(output_file):
        return None

    # 자동 갱신 로직: raw 데이터의 수정 시간이 output_file 보다 최신인지 확인
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(output_file)), "raw")
    if os.path.exists(raw_dir):
        raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if raw_files:
            latest_raw_time = max(os.path.getmtime(f) for f in raw_files)
            output_time = os.path.getmtime(output_file)
            
            if latest_raw_time > output_time:
                logger.info("신규 원본 데이터(raw) 업데이트가 감지되었습니다. 자동으로 기존 캐시를 무시하고 재처리합니다.")
                return None

    file_size_mb = os.path.getsize(output_file) / (1024 ** 2)
    logger.info(
        f"✅ 최신 전처리 캐시 발견: {output_file}  ({file_size_mb:.1f} MB)"
    )
    logger.info("   → 재처리 없이 기존 파일을 로드합니다. 강제 재처리: --force")
    df = pd.read_csv(output_file, parse_dates=["Date"])
    logger.info(f"   로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
    return df


def run_pipeline() -> None:
    """
    ./data/raw/ 의 모든 CSV를 순회하며 특성을 생성한 뒤
    ./data/processed/sp500_features.csv 로 저장합니다.
    캐시(sp500_features.csv)가 있으면 재처리 없이 바로 반환합니다.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── 캐시 재활용 확인 (--force 없으면 기존 파일 로드 후 조기 종료)
    cached = load_cached(OUTPUT_FILE)
    if cached is not None:
        return  # 이미 처리된 데이터 존재 → 파이프라인 건너뜀

    csv_files = sorted(
        [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    )
    logger.info(f"총 {len(csv_files)}개 CSV 파일 발견 → 처리 시작")

    all_dfs: list[pd.DataFrame] = []
    skipped = 0

    for idx, fname in enumerate(csv_files, start=1):
        ticker   = os.path.splitext(fname)[0]
        filepath = os.path.join(RAW_DIR, fname)

        # ── 1. CSV 로드
        df = load_ticker_csv(filepath)
        if df is None:
            skipped += 1
            continue

        rows_before = len(df)

        # ── 2. 특성 생성 (leakage 없이 단일 종목 단위로 수행)
        df = add_features(df)

        # ── 3. 무한대(Inf) 값 → NaN 치환
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # ── 4. NaN 행 제거
        df.dropna(inplace=True)

        rows_after_drop = len(df)

        logger.info(
            f"[{idx:>3}/{len(csv_files)}] {ticker:<8} "
            f"| 원본: {rows_before:>4}행 "
            f"→ NaN 제거 후: {rows_after_drop:>4}행 "
            f"(제거: {rows_before - rows_after_drop}행)"
        )

        if df.empty:
            logger.warning(f"[{ticker}] NaN 제거 후 데이터 없음 → 건너뜀")
            skipped += 1
            continue

        all_dfs.append(df)

    # ── 5. 전체 종목 병합 및 날짜 기준 오름차순 정렬
    logger.info(f"\n{'='*60}")
    logger.info(f"유효 종목 수: {len(all_dfs)}개 (건너뜀: {skipped}개)")

    if not all_dfs:
        logger.error("처리된 데이터가 없습니다. 파이프라인 종료.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 병합 전후 행 수 출력
    total_rows = len(merged_df)
    logger.info(f"병합 후 전체 행 수: {total_rows:,}행")

    # 날짜 오름차순 정렬 (같은 날짜 내에서는 티커 알파벳 순)
    merged_df = merged_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ── 6. 최종 저장
    merged_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"저장 완료 → {OUTPUT_FILE}")
    logger.info(f"최종 데이터 크기: {merged_df.shape[0]:,}행 × {merged_df.shape[1]}열")
    logger.info(f"컬럼 목록: {list(merged_df.columns)}")
    logger.info("="*60)


# =============================================================================
# ▶ 엔트리포인트
# =============================================================================
if __name__ == "__main__":
    run_pipeline()

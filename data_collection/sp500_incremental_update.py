"""
=============================================================================
 S&P 500 과거 주가 데이터 증분 업데이트(Incremental Update) 파이프라인
 파일명: sp500_incremental_update.py
 목적  : 이미 수집된 CSV 파일이 있으면 마지막 날짜 이후 신규 데이터만
         다운로드하여 병합·최신화합니다. 없으면 10년 치 전체를 수집합니다.
 작성일: 2026-06-02
=============================================================================
"""

import os
import time
import datetime
import logging

import pandas as pd
import yfinance as yf


# =============================================================================
# ▶ 로깅 설정
# =============================================================================
def setup_logging(log_dir: str) -> logging.Logger:
    """
    콘솔 + 파일 이중 로거를 설정하고 반환합니다.

    Args:
        log_dir (str): 로그 파일 저장 디렉토리

    Returns:
        logging.Logger: 설정 완료된 로거 객체
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path    = os.path.join(log_dir, f"incremental_{timestamp}.log")

    logger = logging.getLogger("SP500Incremental")
    logger.setLevel(logging.DEBUG)

    # 콘솔 출력 (INFO 이상)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    # 파일 출력 (DEBUG 이상, 상세 기록)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# =============================================================================
# ▶ S&P 500 시가총액 상위 100개 티커 (하드코딩, 웹 크롤링 금지)
#   - BRK.B → BRK-B 등 yfinance 호환 포맷으로 변환
# =============================================================================
SP500_TICKERS = [
    # ── 메가캡 기술 / AI ──────────────────────────────────────────────────
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet Class A
    "GOOG",   # Alphabet Class C
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "AVGO",   # Broadcom
    "ORCL",   # Oracle
    # ── 반도체 ───────────────────────────────────────────────────────────
    "AMD",    # Advanced Micro Devices
    "QCOM",   # Qualcomm
    "TXN",    # Texas Instruments
    "INTC",   # Intel
    "MU",     # Micron Technology
    "AMAT",   # Applied Materials
    "LRCX",   # Lam Research
    "KLAC",   # KLA Corporation
    "MRVL",   # Marvell Technology
    "ADI",    # Analog Devices
    # ── 소프트웨어 / 클라우드 ──────────────────────────────────────────
    "CRM",    # Salesforce
    "ADBE",   # Adobe
    "NOW",    # ServiceNow
    "INTU",   # Intuit
    "PANW",   # Palo Alto Networks
    "CRWD",   # CrowdStrike
    "SNPS",   # Synopsys
    "CDNS",   # Cadence Design
    "DDOG",   # Datadog
    "WDAY",   # Workday
    # ── 핀테크 / 결제 ────────────────────────────────────────────────────
    "V",      # Visa
    "MA",     # Mastercard
    "PYPL",   # PayPal
    "FI",     # Fiserv
    "GPN",    # Global Payments
    # ── 금융 - 은행 ──────────────────────────────────────────────────────
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "C",      # Citigroup
    "USB",    # U.S. Bancorp
    "TFC",    # Truist Financial
    # ── 금융 - 보험 / 투자 ───────────────────────────────────────────────
    "BRK-B",  # Berkshire Hathaway B (BRK.B → BRK-B)
    "BLK",    # BlackRock
    "SPGI",   # S&P Global
    "MCO",    # Moody's
    "CB",     # Chubb
    "PGR",    # Progressive
    "ALL",    # Allstate
    # ── 헬스케어 - 제약 / 바이오 ─────────────────────────────────────────
    "LLY",    # Eli Lilly
    "JNJ",    # Johnson & Johnson
    "ABBV",   # AbbVie
    "MRK",    # Merck
    "PFE",    # Pfizer
    "BMY",    # Bristol-Myers Squibb
    "AMGN",   # Amgen
    "GILD",   # Gilead Sciences
    "REGN",   # Regeneron
    "VRTX",   # Vertex Pharmaceuticals
    # ── 헬스케어 - 의료기기 / 서비스 ────────────────────────────────────
    "UNH",    # UnitedHealth Group
    "CVS",    # CVS Health
    "CI",     # Cigna
    "ELV",    # Elevance Health
    "ISRG",   # Intuitive Surgical
    "MDT",    # Medtronic
    "ABT",    # Abbott Laboratories
    "SYK",    # Stryker
    "TMO",    # Thermo Fisher
    "DHR",    # Danaher
    # ── 에너지 ────────────────────────────────────────────────────────────
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "EOG",    # EOG Resources
    "SLB",    # SLB (Schlumberger)
    "MPC",    # Marathon Petroleum
    "PSX",    # Phillips 66
    "OXY",    # Occidental Petroleum
    # ── 소비재 - 필수소비재 ───────────────────────────────────────────────
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    "COST",   # Costco
    "WMT",    # Walmart
    "PM",     # Philip Morris
    "MO",     # Altria
    # ── 소비재 - 임의소비재 ───────────────────────────────────────────────
    "HD",     # Home Depot
    "LOW",    # Lowe's
    "MCD",    # McDonald's
    "SBUX",   # Starbucks
    "NKE",    # Nike
    "TGT",    # Target
    "ABNB",   # Airbnb
    "UBER",   # Uber
    # ── 산업재 ────────────────────────────────────────────────────────────
    "CAT",    # Caterpillar
    "HON",    # Honeywell
    "BA",     # Boeing
    "LMT",    # Lockheed Martin
    "RTX",    # RTX (Raytheon)
    "GE",     # GE Aerospace
    "UPS",    # UPS
    "UNP",    # Union Pacific
    # ── 통신 / 미디어 ─────────────────────────────────────────────────────
    "GOOGL",  # (중복 허용 - set으로 제거)
    "NFLX",   # Netflix
    "DIS",    # Walt Disney
    "CMCSA",  # Comcast
    "T",      # AT&T
    "VZ",     # Verizon
    "TMUS",   # T-Mobile
]

# 중복 제거 후 알파벳 순 정렬
SP500_TICKERS = sorted(list(set(SP500_TICKERS)))


# =============================================================================
# ▶ 날짜 전처리 유틸 함수
#   핵심 문제: yfinance 반환 데이터의 DatetimeIndex는 타임존(UTC 등)이
#   포함될 수 있고, 기존 CSV에서 읽은 날짜는 타임존이 없는 문자열입니다.
#   이를 통일하지 않으면 concat 시 TypeError가 발생합니다.
# =============================================================================
def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 'Date' 컬럼을 'YYYY-MM-DD' 문자열로 통일합니다.

    처리 순서:
      1. DatetimeIndex인 경우 reset_index()로 'Date' 컬럼으로 변환
      2. pd.to_datetime()으로 파싱 (타임존 있으면 UTC로 변환 후 제거)
      3. 최종적으로 'YYYY-MM-DD' 문자열 포맷으로 저장

    Args:
        df (pd.DataFrame): 원본 데이터프레임

    Returns:
        pd.DataFrame: 'Date' 컬럼이 정규화된 데이터프레임
    """
    # Step 1: DatetimeIndex → 일반 컬럼 변환
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Step 2: 'Date' 컬럼이 없는 경우 방어 처리
    if "Date" not in df.columns:
        # 첫 번째 컬럼을 'Date'로 가정 (yfinance 구버전 대응)
        df = df.rename(columns={df.columns[0]: "Date"})

    # Step 3: 타임존 제거 후 날짜 문자열로 통일
    # - utc=True 옵션으로 timezone-aware 컬럼을 UTC 기준으로 파싱 후
    #   .dt.tz_localize(None)으로 타임존 정보 제거
    try:
        df["Date"] = (
            pd.to_datetime(df["Date"], utc=True)   # timezone-aware → UTC 통일
            .dt.tz_localize(None)                   # timezone 정보 제거
            .dt.strftime("%Y-%m-%d")                # 'YYYY-MM-DD' 문자열로 변환
        )
    except TypeError:
        # 이미 naive datetime인 경우 (CSV에서 읽은 날짜 등)
        df["Date"] = (
            pd.to_datetime(df["Date"])
            .dt.strftime("%Y-%m-%d")
        )

    return df


# =============================================================================
# ▶ SP500IncrementalCollector 클래스
# =============================================================================
class SP500IncrementalCollector:
    """
    S&P 500 주가 데이터를 증분 업데이트 방식으로 수집·관리하는 클래스.

    증분 업데이트 전략:
      - CSV 미존재 → 10년 전체 다운로드 [신규 생성]
      - CSV 존재하고 최신 → 건너뜀              [최신 상태 유지]
      - CSV 존재하고 구버전 → 신규 날짜만 추가   [n일치 업데이트]

    Attributes:
        tickers    (list[str]): 수집 대상 티커 리스트
        save_dir   (str)      : CSV 저장 디렉토리
        full_period(str)      : 전체 수집 시 기간 (기본 '10y')
        retry_cnt  (int)      : 실패 시 재시도 횟수
        retry_wait (float)    : 재시도 대기 시간(초)
        logger                : 로거 객체
    """

    # 상태 레이블 상수
    STATUS_NEW    = "[신규 생성]    "
    STATUS_UPDATE = "[업데이트]     "
    STATUS_LATEST = "[최신 상태 유지]"
    STATUS_FAIL   = "[실패]         "

    def __init__(
        self,
        tickers:     list,
        save_dir:    str   = "./data/raw",
        full_period: str   = "10y",
        retry_cnt:   int   = 3,
        retry_wait:  float = 5.0,
        logger:      logging.Logger = None,
    ):
        self.tickers     = tickers
        self.save_dir    = save_dir
        self.full_period = full_period
        self.retry_cnt   = retry_cnt
        self.retry_wait  = retry_wait
        self.logger      = logger or logging.getLogger("SP500Incremental")

        os.makedirs(self.save_dir, exist_ok=True)

        # 결과 집계
        self.result_new:    list[str] = []   # 신규 생성
        self.result_update: list[str] = []   # 증분 업데이트
        self.result_latest: list[str] = []   # 이미 최신
        self.result_fail:   list[str] = []   # 실패

    # -------------------------------------------------------------------------
    # ▷ 내부 유틸: 기존 CSV 읽기
    # -------------------------------------------------------------------------
    def _load_existing(self, csv_path: str) -> pd.DataFrame | None:
        """
        기존 CSV 파일을 읽어 날짜 컬럼을 정규화한 뒤 반환합니다.

        Args:
            csv_path (str): CSV 파일 경로

        Returns:
            pd.DataFrame | None: 성공 시 정규화된 DataFrame, 실패 시 None
        """
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df = normalize_date_column(df)
            return df
        except Exception as exc:
            self.logger.warning(f"  기존 CSV 로드 실패 ({csv_path}): {exc}")
            return None

    # -------------------------------------------------------------------------
    # ▷ 내부 유틸: yfinance 다운로드 (재시도 포함)
    # -------------------------------------------------------------------------
    def _download(
        self,
        ticker:    str,
        start_dt:  str | None = None,
        end_dt:    str | None = None,
        period:    str | None = None,
    ) -> pd.DataFrame | None:
        """
        yfinance로 단일 티커 데이터를 다운로드합니다.
        실패 시 retry_cnt 만큼 재시도합니다.

        Args:
            ticker   (str): 티커 심볼
            start_dt (str): 시작 날짜 'YYYY-MM-DD' (period와 택일)
            end_dt   (str): 종료 날짜 'YYYY-MM-DD'
            period   (str): 기간 문자열 '10y' 등 (start_dt와 택일)

        Returns:
            pd.DataFrame | None: 성공 시 정규화된 DataFrame, 실패 시 None
        """
        for attempt in range(1, self.retry_cnt + 1):
            try:
                self.logger.debug(
                    f"  [{ticker}] 다운로드 시도 {attempt}/{self.retry_cnt} "
                    f"(period={period}, start={start_dt}, end={end_dt})"
                )

                if period:
                    # 전체 기간 수집 모드
                    raw = yf.download(
                        ticker,
                        period=period,
                        auto_adjust=True,
                        progress=False,
                    )
                else:
                    # 증분 수집 모드: start ~ end 범위 지정
                    raw = yf.download(
                        ticker,
                        start=start_dt,
                        end=end_dt,
                        auto_adjust=True,
                        progress=False,
                    )

                # 빈 결과 처리 (상장폐지 또는 해당 기간 데이터 없음)
                if raw is None or raw.empty:
                    self.logger.debug(f"  [{ticker}] 빈 응답 반환")
                    return None

                # 멀티인덱스 컬럼 평탄화 (yfinance 버전에 따라 발생)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # 날짜 컬럼 정규화 (타임존 제거 + YYYY-MM-DD 통일)
                raw = normalize_date_column(raw)

                self.logger.debug(f"  [{ticker}] 다운로드 성공 ({len(raw)}행)")
                return raw

            except Exception as exc:
                self.logger.warning(
                    f"  [{ticker}] 시도 {attempt}/{self.retry_cnt} 예외: {exc}"
                )
                if attempt < self.retry_cnt:
                    self.logger.debug(
                        f"  [{ticker}] {self.retry_wait}초 후 재시도..."
                    )
                    time.sleep(self.retry_wait)
                else:
                    self.logger.error(
                        f"  [{ticker}] 최대 재시도 초과 → 수집 불가"
                    )
                    return None

    # -------------------------------------------------------------------------
    # ▷ 내부 유틸: CSV 저장
    # -------------------------------------------------------------------------
    def _save(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        DataFrame을 CSV 파일로 저장(덮어쓰기)합니다.

        Args:
            ticker (str)         : 티커 심볼 (파일명에 사용)
            df     (pd.DataFrame): 저장할 데이터프레임

        Returns:
            bool: 저장 성공 여부
        """
        try:
            path = os.path.join(self.save_dir, f"{ticker}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.logger.debug(f"  [{ticker}] 저장 완료 → {path}")
            return True
        except OSError as exc:
            self.logger.error(f"  [{ticker}] CSV 저장 오류: {exc}")
            return False

    # -------------------------------------------------------------------------
    # ▷ 핵심 로직: 단일 티커 증분 업데이트
    # -------------------------------------------------------------------------
    def _process_ticker(self, ticker: str) -> str:
        """
        단일 티커에 대한 증분 업데이트 전체 흐름을 처리합니다.

        ┌─────────────────────────────────────────────────────────────┐
        │  증분 업데이트 의사결정 트리                                  │
        │                                                              │
        │  CSV 존재? ──NO──► 10년 전체 다운로드 ──► 저장 [신규 생성]  │
        │      │                                                       │
        │     YES                                                      │
        │      │                                                       │
        │      ▼                                                       │
        │  last_date 파악                                              │
        │      │                                                       │
        │  last_date == 오늘? ──YES──► 아무것도 안 함 [최신 상태 유지] │
        │      │                                                       │
        │     NO                                                       │
        │      │                                                       │
        │      ▼                                                       │
        │  (last_date + 1일) ~ 오늘 범위로 신규 데이터 다운로드        │
        │      │                                                       │
        │   신규 없음? ──YES──► [최신 상태 유지]                        │
        │      │                                                       │
        │     NO                                                       │
        │      │                                                       │
        │      ▼                                                       │
        │  concat(기존, 신규) → drop_duplicates(Date, keep='last')    │
        │      │                                                       │
        │      ▼                                                       │
        │  덮어쓰기 저장 [n일치 업데이트]                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            ticker (str): 처리할 티커 심볼

        Returns:
            str: 처리 결과 상태 레이블
        """
        csv_path = os.path.join(self.save_dir, f"{ticker}.csv")
        today    = datetime.date.today().strftime("%Y-%m-%d")

        # ── 분기 A: 기존 파일 없음 → 전체 데이터 신규 수집 ──────────────
        if not os.path.exists(csv_path):
            self.logger.debug(f"  [{ticker}] CSV 없음 → 전체({self.full_period}) 수집 시작")

            new_df = self._download(ticker, period=self.full_period)

            if new_df is None or new_df.empty:
                return self.STATUS_FAIL

            if not self._save(ticker, new_df):
                return self.STATUS_FAIL

            return self.STATUS_NEW

        # ── 분기 B: 기존 파일 있음 → 증분 업데이트 ──────────────────────
        existing_df = self._load_existing(csv_path)

        if existing_df is None or existing_df.empty:
            # 파일이 손상되었거나 읽기 실패 → 전체 재수집으로 복구
            self.logger.warning(
                f"  [{ticker}] 기존 CSV 손상/비어있음 → 전체 재수집으로 복구"
            )
            new_df = self._download(ticker, period=self.full_period)
            if new_df is None:
                return self.STATUS_FAIL
            self._save(ticker, new_df)
            return self.STATUS_NEW

        # ── 마지막 날짜 파악 ──────────────────────────────────────────────
        # Date 컬럼을 기준으로 가장 최근 날짜를 문자열로 추출
        last_date_str = existing_df["Date"].max()     # 'YYYY-MM-DD' 문자열
        last_date     = datetime.date.fromisoformat(last_date_str)

        self.logger.debug(
            f"  [{ticker}] 기존 마지막 날짜: {last_date_str}, 오늘: {today}"
        )

        # ── 이미 최신인지 확인 ────────────────────────────────────────────
        # 마지막 날짜가 오늘이거나 오늘 이후라면(주말·공휴일 엣지케이스 포함)
        # 업데이트가 필요 없으므로 즉시 반환
        if last_date >= datetime.date.today():
            return self.STATUS_LATEST

        # ── 신규 데이터 다운로드 범위 계산 ───────────────────────────────
        # yfinance start 파라미터는 해당 날짜를 '포함'하므로
        # 마지막 날짜 다음 날부터 요청해야 중복을 피할 수 있습니다.
        # 단, 중복 제거(drop_duplicates)로 안전장치를 추가로 둡니다.
        start_dt = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        end_dt   = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        # end_dt에 +1일: yfinance end 파라미터는 exclusive(미포함)이므로 오늘 데이터 포함

        self.logger.debug(
            f"  [{ticker}] 증분 수집 범위: {start_dt} ~ {end_dt}"
        )

        incremental_df = self._download(
            ticker, start_dt=start_dt, end_dt=end_dt
        )

        # ── 신규 데이터가 없으면 최신 상태로 간주 ────────────────────────
        if incremental_df is None or incremental_df.empty:
            self.logger.debug(
                f"  [{ticker}] 신규 데이터 없음 (장 미개장 또는 상장폐지)"
            )
            return self.STATUS_LATEST

        new_rows = len(incremental_df)

        # ── 기존 데이터 + 신규 데이터 병합 ───────────────────────────────
        # 1) 수직 연결 (행 추가)
        merged_df = pd.concat(
            [existing_df, incremental_df],
            ignore_index=True
        )

        # 2) Date 기준 중복 제거
        #    keep='last': 동일 날짜가 있을 때 신규 데이터(마지막 행)를 우선 유지
        #    이렇게 하면 기존 데이터의 수정(예: 배당 재계산)도 반영됩니다.
        merged_df = merged_df.drop_duplicates(subset=["Date"], keep="last")

        # 3) 날짜 기준 오름차순 정렬
        merged_df = merged_df.sort_values("Date").reset_index(drop=True)

        # 4) 덮어쓰기 저장
        if not self._save(ticker, merged_df):
            return self.STATUS_FAIL

        self.logger.debug(
            f"  [{ticker}] 병합 완료: 기존 {len(existing_df)}행 + "
            f"신규 {new_rows}행 → 최종 {len(merged_df)}행"
        )

        # 상태 레이블에 업데이트된 신규 행 수를 포함해 반환
        return f"[{new_rows}일치 업데이트]"

    # -------------------------------------------------------------------------
    # ▷ 공개 메서드: 전체 루프 실행
    # -------------------------------------------------------------------------
    def run(self) -> dict:
        """
        전체 티커 리스트에 대해 증분 업데이트 루프를 실행합니다.

        Returns:
            dict: {
                "new"   : [신규 생성 티커 리스트],
                "update": [업데이트된 티커 리스트],
                "latest": [이미 최신인 티커 리스트],
                "fail"  : [실패 티커 리스트],
            }
        """
        total      = len(self.tickers)
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info(f"  S&P 500 증분 업데이트 시작 | 대상: {total}개 종목")
        self.logger.info(f"  저장 경로: {os.path.abspath(self.save_dir)}")
        self.logger.info("=" * 70)

        for idx, ticker in enumerate(self.tickers, start=1):
            self.logger.info(f"[{idx:>3}/{total}] {ticker:<8} 처리 중...")

            try:
                status = self._process_ticker(ticker)

            except Exception as exc:
                # _process_ticker 내부에서 처리되지 않은 예외까지 캐치
                self.logger.error(
                    f"  [{ticker}] 예상치 못한 오류 → 건너뜀: {exc}"
                )
                status = self.STATUS_FAIL

            # 결과 집계
            if status == self.STATUS_NEW:
                self.result_new.append(ticker)
            elif "업데이트" in status:
                self.result_update.append(ticker)
            elif status == self.STATUS_LATEST:
                self.result_latest.append(ticker)
            else:  # FAIL 또는 기타
                self.result_fail.append(ticker)

            self.logger.info(f"  └─ 결과: {status}")

            # API 요청 간 짧은 딜레이 (서버 부하 방지)
            time.sleep(0.3)

        elapsed = time.time() - start_time
        self._print_summary(total, elapsed)

        return {
            "new"   : self.result_new,
            "update": self.result_update,
            "latest": self.result_latest,
            "fail"  : self.result_fail,
        }

    # -------------------------------------------------------------------------
    def _print_summary(self, total: int, elapsed: float) -> None:
        """
        전체 실행 완료 후 종합 요약 리포트를 출력합니다.
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("  ▶ 증분 업데이트 최종 요약 리포트")
        self.logger.info("=" * 70)
        self.logger.info(f"  전체 대상   : {total:>4}개")
        self.logger.info(f"  신규 생성   : {len(self.result_new):>4}개  → {self.result_new}")
        self.logger.info(f"  데이터 업데이트: {len(self.result_update):>4}개  → {self.result_update}")
        self.logger.info(f"  최신 상태   : {len(self.result_latest):>4}개")
        self.logger.info(f"  실패        : {len(self.result_fail):>4}개")
        self.logger.info(f"  총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        self.logger.info("-" * 70)

        if self.result_fail:
            self.logger.info("  ⚠ 실패 종목:")
            for i, t in enumerate(self.result_fail, 1):
                self.logger.info(f"    {i:>3}. {t}")
        else:
            self.logger.info("  🎉 모든 종목 처리 완료 (실패 없음)")

        self.logger.info("=" * 70)


# =============================================================================
# ▶ 메인 실행 블록
# =============================================================================
if __name__ == "__main__":
    LOG_DIR  = "./data/logs"
    DATA_DIR = "./data/raw"

    # 1) 로거 초기화
    logger = setup_logging(LOG_DIR)

    # 2) 수집기 초기화
    collector = SP500IncrementalCollector(
        tickers     = SP500_TICKERS,
        save_dir    = DATA_DIR,
        full_period = "10y",      # 최초 수집 시 10년 치 다운로드
        retry_cnt   = 3,          # API 오류 시 최대 3회 재시도
        retry_wait  = 5.0,        # 재시도 전 5초 대기
        logger      = logger,
    )

    # 3) 증분 업데이트 실행
    result = collector.run()

    # 4) 실패 티커를 별도 파일로 저장 (재실행 편의용)
    if result["fail"]:
        fail_path = os.path.join(LOG_DIR, "failed_tickers.txt")
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result["fail"]))
            logger.info(f"  실패 티커 저장 → {os.path.abspath(fail_path)}")
        except OSError as e:
            logger.error(f"  실패 파일 저장 오류: {e}")

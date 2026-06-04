"""
=============================================================================
 S&P 500 과거 주가 데이터 수집 파이프라인 - 1단계
 파일명: sp500_data_collector.py
 목적  : yfinance를 이용해 S&P 500 구성 종목 500개의 최근 10년 치
         OHLCV 데이터를 개별 CSV 파일로 ./data/raw/ 에 저장합니다.
 작성일: 2026-06-02
=============================================================================
"""

import os
import time
import datetime
import logging
import yfinance as yf
import pandas as pd

# =============================================================================
# ▶ 로깅 설정
#   - 콘솔(StreamHandler)과 파일(FileHandler) 양쪽으로 로그를 출력합니다.
#   - 파일 로그는 data/raw/ 디렉토리 생성 후 저장됩니다.
# =============================================================================
def setup_logging(log_dir: str) -> logging.Logger:
    """
    로거를 설정하고 반환합니다.

    Args:
        log_dir (str): 로그 파일이 저장될 디렉토리 경로

    Returns:
        logging.Logger: 설정이 완료된 로거 객체
    """
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir,
        f"collection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("SP500Collector")
    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러 - INFO 레벨 이상만 출력
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    # 파일 핸들러 - DEBUG 레벨 이상 모두 기록
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# ▶ S&P 500 구성 종목 티커 리스트 (하드코딩)
#   - 2025년 기준 시가총액 상위 500개 종목
#   - yfinance에서 인식 가능한 포맷으로 변환:
#       BRK.B → BRK-B  /  BF.B → BF-B
#   - 웹 크롤링 없이 코드 내 직접 정의합니다.
# =============================================================================
SP500_TICKERS = [
    # ── 메가캡 기술주 ──
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "ORCL",
    # ── 반도체 ──
    "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC",
    "MCHP", "NXPI", "ON", "MPWR", "SWKS", "MRVL", "ADI",
    # ── 소프트웨어 / SaaS ──
    "CRM", "ADBE", "NOW", "INTU", "PANW", "CRWD", "FTNT", "SNPS",
    "CDNS", "ANSS", "PTC", "MSCI", "PAYC", "HUBS", "DDOG",
    "ZS", "OKTA", "VEEV", "WDAY", "MDB", "TEAM", "SNOW", "NET",
    "CFLT", "GTLB",
    # ── 하드웨어 / 장비 ──
    "CSCO", "HPQ", "HPE", "DELL", "NTAP", "WDC", "STX",
    # ── 핀테크 / 결제 ──
    "V", "MA", "PYPL", "SQ", "FIS", "FI", "GPN", "AFRM",
    # ── 금융 - 은행 ──
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "TFC",
    "PNC", "FITB", "HBAN", "RF", "CFG", "KEY", "MTB",
    # ── 금융 - 보험 ──
    "BRK-B", "AIG", "MET", "PRU", "AFL", "ALL", "PGR", "TRV",
    "CB", "HIG", "UNM", "GL", "RGA", "LNC",
    # ── 금융 - 자산운용 / 증권 ──
    "BLK", "SCHW", "SPGI", "MCO", "ICE", "CME", "NDAQ", "CBOE",
    "BEN", "IVZ",
    # ── 헬스케어 - 제약 / 바이오 ──
    "LLY", "JNJ", "ABBV", "MRK", "PFE", "BMY", "AMGN", "GILD",
    "BIIB", "REGN", "VRTX", "MRNA", "ALXN", "INCY", "SGEN",
    "EXAS", "IONS", "NBIX", "ACAD", "BMRN",
    # ── 헬스케어 - 의료기기 / 서비스 ──
    "UNH", "CVS", "CI", "HUM", "CNC", "ELV", "MOH",
    "ABT", "MDT", "SYK", "BSX", "ZBH", "EW", "ISRG",
    "BDX", "RMD", "HOLX", "TFX", "PODD", "DXCM",
    # ── 에너지 ──
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
    "PXD", "HAL", "DVN", "FANG", "APA", "HES", "OXY",
    "BKR", "NOV", "FTI", "CTRA", "MRO",
    # ── 소재 ──
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "STLD",
    "RS", "ALB", "MOS", "CF", "EMN", "CE", "LYB",
    "DD", "DOW", "PPG", "IFF", "RPM",
    # ── 산업재 - 항공우주 / 방산 ──
    "BA", "LMT", "RTX", "NOC", "GD", "HII", "L3H", "TDG",
    "HWM", "AXON",
    # ── 산업재 - 기계 / 기타 ──
    "HON", "GE", "MMM", "EMR", "ROK", "PH", "ETN", "IR",
    "DOV", "AME", "XYL", "OTIS", "CARR", "TT", "JCI",
    "PCAR", "CAT", "DE", "CMI", "TEX",
    # ── 산업재 - 운송 / 물류 ──
    "UPS", "FDX", "JBHT", "CHRW", "XPO", "SAIA", "ODFL",
    "NSC", "UNP", "CSX", "CNI", "CP",
    # ── 소비재 - 필수소비재 ──
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ",
    "CL", "KHC", "GIS", "CPB", "CAG", "SJM", "HRL",
    "CHD", "CLX", "EL", "KVUE",
    # ── 소비재 - 임의소비재 ──
    "HD", "LOW", "TGT", "DG", "DLTR", "ROSS", "TJX",
    "MCD", "SBUX", "YUM", "CMG", "DPZ",
    "NKE", "RL", "PVH", "HBI", "VFC",
    "GM", "F", "TM", "RIVN", "LCID",
    "APTV", "BWA", "LEA", "LKQ",
    # ── 통신 서비스 ──
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "DISH",
    "NFLX", "DIS", "WBD", "PARA", "FOX", "FOXA",
    "EA", "TTWO", "MTCH", "SNAP", "PINS", "RDDT",
    # ── 유틸리티 ──
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "WEC",
    "ES", "PPL", "ETR", "FE", "CMS", "NI", "OGE",
    "AES", "PCG", "CNP", "EVRG", "SRE",
    # ── 부동산 (REIT) ──
    "AMT", "PLD", "CCI", "EQIX", "PSA", "EQR", "AVB",
    "VTR", "WELL", "PEAK", "SPG", "O", "WPC", "NNN",
    "KIM", "REG", "BXP", "ARE", "VICI", "MGM",
    # ── 이커머스 / 인터넷 플랫폼 ──
    "EBAY", "ETSY", "W", "CHWY", "CPNG",
    # ── 클라우드 / 인프라 ──
    "AKAM", "CDW", "GDDY", "VRT", "SMCI",
    # ── 기타 대형주 ──
    "BF-B", "WELL", "RCL", "CCL", "NCLH", "MAR", "HLT",
    "H", "IHG", "LVS", "WYNN", "MGM",
    "DHI", "LEN", "PHM", "TOL", "NVR", "MDC",
    "UDR", "CPT", "ESSab", "AIV",
    "ZM", "DOCU", "TWLO", "ROKU", "U",
    "UBER", "LYFT", "ABNB", "DASH",
    "COIN", "HOOD", "SOFI",
    "TSM", "ASML", "SAP", "NVO", "SHOP",
    "ADSK", "ANGI", "NLOK", "GEN",
    "BIO", "A", "WAT", "MTD", "TMO", "DHR", "IDXX",
    "IQV", "IQVIA", "LH", "DGX", "PKI",
    "ZTS", "ELAN", "PRGO",
    "AWK", "WTR", "SWX", "SJW",
    "KMB", "CHD", "SPB",
    "STZ", "BUD", "TAP", "SAM",
    "MKC", "LANC",
    "FMC", "CTVA",
    "IEX", "ROP", "FAST", "GWW", "MSC", "WSO",
    "TDY", "LDOS", "SAIC", "BAH", "CACI",
    "VRSK", "BR", "FDS", "TRI",
    "CBRE", "JLL", "CWK",
    "URI", "RSG", "WM", "CWST",
    "TRMB", "FWRD", "GATX",
    "MTZ", "PWR", "EME", "ACM",
    "AAON", "AOS", "GNRC",
    "POOL", "SWK", "SNA", "PNR",
    "AVY", "IP", "PKG", "SEE",
    "OI", "BERY", "SLVM",
    "AFL", "CNO",
    "WRB", "RNR", "MKL", "ERIE",
]

# 중복 제거 및 최종 리스트 확정
SP500_TICKERS = sorted(list(set(SP500_TICKERS)))


# =============================================================================
# ▶ SP500DataCollector 클래스
#   데이터 수집의 핵심 로직을 담당하는 클래스입니다.
# =============================================================================
class SP500DataCollector:
    """
    S&P 500 구성 종목의 과거 주가 데이터를 수집하고 저장하는 클래스.

    Attributes:
        tickers   (list): 수집 대상 티커 리스트
        save_dir  (str) : 개별 CSV 파일이 저장될 디렉토리
        period    (str) : yfinance에 전달할 기간 문자열
        retry_cnt (int) : API 오류 시 최대 재시도 횟수
        retry_wait(float): 재시도 전 대기 시간(초)
        logger         : 로거 객체
    """

    def __init__(
        self,
        tickers: list,
        save_dir: str = "./data/raw",
        period: str = "10y",
        retry_cnt: int = 3,
        retry_wait: float = 5.0,
        logger: logging.Logger = None,
    ):
        self.tickers    = tickers
        self.save_dir   = save_dir
        self.period     = period
        self.retry_cnt  = retry_cnt
        self.retry_wait = retry_wait
        self.logger     = logger or logging.getLogger("SP500Collector")

        # 저장 디렉토리가 없으면 자동 생성
        os.makedirs(self.save_dir, exist_ok=True)

        # 수집 결과 추적용 변수
        self.success_list: list[str] = []
        self.fail_list:    list[str] = []

    # -------------------------------------------------------------------------
    def _download_single(self, ticker: str) -> pd.DataFrame | None:
        """
        단일 티커의 OHLCV 데이터를 yfinance로 다운로드합니다.
        API 일시적 오류에 대비해 retry_cnt 횟수만큼 재시도합니다.

        Args:
            ticker (str): 주식 티커 심볼

        Returns:
            pd.DataFrame | None: 성공 시 DataFrame, 실패 시 None
        """
        for attempt in range(1, self.retry_cnt + 1):
            try:
                self.logger.debug(f"[{ticker}] 다운로드 시도 {attempt}/{self.retry_cnt}")

                # yfinance로 데이터 다운로드
                # auto_adjust=True: 배당·분할 조정 종가(Adj Close) 자동 반영
                # progress=False  : tqdm 진행 바 비활성화(로그 가독성 확보)
                df = yf.download(
                    ticker,
                    period=self.period,
                    auto_adjust=True,
                    progress=False,
                )

                # 빈 데이터프레임 처리 (상장폐지·심볼 변경 등)
                if df is None or df.empty:
                    self.logger.warning(f"[{ticker}] 빈 데이터 반환 → 건너뜀")
                    return None

                # 멀티인덱스 컬럼이 생성된 경우 단일 레벨로 평탄화
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 인덱스(Date)를 일반 컬럼으로 이동
                df = df.reset_index()

                # Date 컬럼을 문자열로 변환 (CSV 저장 시 가독성 향상)
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                self.logger.debug(f"[{ticker}] 다운로드 완료 - {len(df)}행")
                return df

            except Exception as exc:
                self.logger.warning(
                    f"[{ticker}] 시도 {attempt}/{self.retry_cnt} 실패: {exc}"
                )
                if attempt < self.retry_cnt:
                    self.logger.info(f"[{ticker}] {self.retry_wait}초 후 재시도...")
                    time.sleep(self.retry_wait)
                else:
                    self.logger.error(f"[{ticker}] 최대 재시도 횟수 초과 → 수집 실패")
                    return None

    # -------------------------------------------------------------------------
    def _save_csv(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        DataFrame을 CSV 파일로 저장합니다.

        Args:
            ticker (str)         : 주식 티커 심볼 (파일명에 사용)
            df     (pd.DataFrame): 저장할 데이터프레임

        Returns:
            bool: 저장 성공 여부
        """
        try:
            filepath = os.path.join(self.save_dir, f"{ticker}.csv")
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            self.logger.debug(f"[{ticker}] CSV 저장 완료 → {filepath}")
            return True
        except OSError as exc:
            self.logger.error(f"[{ticker}] CSV 저장 실패: {exc}")
            return False

    # -------------------------------------------------------------------------
    def run(self) -> dict:
        """
        전체 티커 리스트에 대해 데이터 수집·저장 루프를 실행합니다.
        개별 종목 오류가 발생해도 전체 루프는 계속 진행됩니다.

        Returns:
            dict: {"success": [...], "fail": [...]} 형태의 결과 딕셔너리
        """
        total = len(self.tickers)
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info(f"  S&P 500 데이터 수집 시작 | 대상 종목 수: {total}개")
        self.logger.info(f"  수집 기간: {self.period} | 저장 경로: {self.save_dir}")
        self.logger.info("=" * 70)

        for idx, ticker in enumerate(self.tickers, start=1):
            self.logger.info(f"[{idx:>3}/{total}] {ticker} 처리 중...")

            # 1) 데이터 다운로드
            df = self._download_single(ticker)

            if df is None:
                # 다운로드 실패 → 실패 목록에 추가 후 다음 종목으로
                self.fail_list.append(ticker)
                continue

            # 2) CSV 저장
            saved = self._save_csv(ticker, df)

            if saved:
                self.success_list.append(ticker)
                self.logger.info(
                    f"  └─ ✅ 성공 | 행 수: {len(df):,}행"
                )
            else:
                self.fail_list.append(ticker)
                self.logger.info(f"  └─ ❌ 저장 실패")

            # yfinance API 요청 간 짧은 딜레이 (서버 부하 방지)
            time.sleep(0.3)

        elapsed = time.time() - start_time
        self._print_summary(total, elapsed)

        return {"success": self.success_list, "fail": self.fail_list}

    # -------------------------------------------------------------------------
    def _print_summary(self, total: int, elapsed: float) -> None:
        """
        전체 수집이 완료된 후 요약 리포트를 콘솔 및 로그에 출력합니다.

        Args:
            total   (int)  : 전체 대상 종목 수
            elapsed (float): 총 소요 시간(초)
        """
        success_cnt = len(self.success_list)
        fail_cnt    = len(self.fail_list)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("  ▶ 최종 수집 요약 리포트")
        self.logger.info("=" * 70)
        self.logger.info(f"  전체 대상  : {total:>4}개")
        self.logger.info(f"  수집 성공  : {success_cnt:>4}개")
        self.logger.info(f"  수집 실패  : {fail_cnt:>4}개")
        self.logger.info(f"  총 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        self.logger.info("-" * 70)

        if self.fail_list:
            self.logger.info("  ⚠ 실패 종목 목록:")
            for i, t in enumerate(self.fail_list, 1):
                self.logger.info(f"    {i:>3}. {t}")
        else:
            self.logger.info("  🎉 모든 종목 수집 성공!")

        self.logger.info("=" * 70)
        self.logger.info(f"  저장 위치: {os.path.abspath(self.save_dir)}")
        self.logger.info("=" * 70)


# =============================================================================
# ▶ 메인 실행 블록
# =============================================================================
if __name__ == "__main__":
    # 1) 로깅 초기화
    LOG_DIR  = "./data/logs"
    DATA_DIR = "./data/raw"

    logger = setup_logging(LOG_DIR)

    # 2) 컬렉터 인스턴스 생성
    collector = SP500DataCollector(
        tickers    = SP500_TICKERS,   # 하드코딩된 티커 리스트
        save_dir   = DATA_DIR,        # 개별 CSV 저장 디렉토리
        period     = "10y",           # 최근 10년 치 데이터
        retry_cnt  = 3,               # 실패 시 최대 3회 재시도
        retry_wait = 5.0,             # 재시도 전 5초 대기
        logger     = logger,
    )

    # 3) 수집 실행
    result = collector.run()

    # 4) 실패 목록을 별도 TXT 파일로도 저장 (후처리 편의용)
    if result["fail"]:
        fail_path = os.path.join(LOG_DIR, "failed_tickers.txt")
        try:
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result["fail"]))
            logger.info(f"  실패 티커 목록 저장 완료 → {os.path.abspath(fail_path)}")
        except OSError as e:
            logger.error(f"  실패 티커 목록 파일 저장 오류: {e}")

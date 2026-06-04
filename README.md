# S&P 500 Transformer Stock Price Prediction

이 프로젝트는 S&P 500 주식 데이터를 기반으로 하는 **딥러닝(Transformer) 기반 주가 예측 파이프라인**입니다. 데이터 수집부터 피처 엔지니어링, 모델 학습, 평가 및 횡단면 포트폴리오 백테스트까지 엔드투엔드(End-to-End)로 구성되어 있으며, 매일 발생하는 신규 데이터에 대응하기 위한 점진적 학습(Online Learning) 기능을 제공합니다.

## 🚀 주요 기능 (Features)

- **Time-Series Transformer 모델**: 시계열 데이터의 장기 의존성(Long-term dependencies)을 포착하기 위해 최적화된 Transformer 구조를 사용합니다.
- **통합 학습 파이프라인 (`main.py`)**: 스케일링, 슬라이딩 윈도우(Sliding Window), DataLoader 생성, 조기 종료(Early Stopping)를 포함한 전체 모델 학습 과정을 자동화합니다.
- **점진적/온라인 학습 (`online_learning.py`)**: 매일 새로운 주가 데이터가 수집될 때 전체 데이터를 다시 학습하지 않고, 기존 모델 가중치를 기반으로 파인튜닝(Fine-tuning)하여 학습 비용을 최소화합니다.
- **독립 평가 및 백테스트 (`run_evaluation.py`)**: 
  - 개별 종목의 단일 예측 및 대시보드 시각화.
  - 내일의 예상 수익률 추론.
  - 전체 종목 대상 횡단면 랭킹(Cross-sectional Ranking)을 통한 Long/Short 포트폴리오 추출 및 백테스팅.

## 📂 프로젝트 구조 (Project Structure)

```text
StockPricePrediction/
├── data_collection/      # yfinance 등 API를 이용한 데이터 수집 모듈
├── dataset/              # 시계열 데이터 전처리 및 PyTorch DataLoader 빌더
├── feature_engineering/  # 기술적 지표 계산 및 데이터 피처 엔지니어링
├── models/               # TimeSeriesTransformer 모델 아키텍처 및 저장된 모델(가중치)
├── training/             # 모델 학습 루프 (Trainer 클래스) 및 조기 종료 로직
├── evaluation/           # 모델 평가, 지표 산출, 대시보드 시각화 모듈
├── main.py               # 초기 전체 학습을 수행하는 메인 스크립트
├── online_learning.py    # 최신 데이터만을 사용해 모델을 미세조정하는 점진적 학습 스크립트
├── run_evaluation.py     # 학습된 모델을 불러와 성능을 평가하고 포트폴리오 성과를 분석하는 스크립트
└── requirements.txt      # 프로젝트 실행을 위한 의존성 패키지 목록
```
*(참고: `StudyRoom` 폴더는 본 파이프라인과 직접적인 연관이 없는 스터디 목적의 공간입니다.)*

## 📈 데이터 명세 (Data Specification)

이 프로젝트에서 수집하고 가공하는 금융 데이터의 세부 항목은 다음과 같습니다.

### 1. 원시 시장 데이터 (Raw Market Data)
*수집 출처: yfinance (1단계 데이터 수집)*
- **Open (시가)**: 장 시작 가격
- **High (고가)**: 장중 최고 가격
- **Low (저가)**: 장중 최저 가격
- **Close (종가)**: 장 마감 가격 (우리의 경우 배당 및 분할이 반영된 수정종가 Adj Close 사용)
- **Volume (거래량)**: 하루 동안 거래된 주식의 총 수량

### 2. 파생 및 기술적 데이터 (Derived & Technical Data)
*가공 출처: pandas 및 pandas-ta (2단계 특성 공학)*
- **Trading Value (거래대금)**: 종가와 거래량을 곱한 값으로, 시장에 유입된 실제 자금 규모를 나타냅니다.
- **RSI (상대강도지수)**: 14일간의 가격 상승폭과 하락폭을 바탕으로 산출한 과매수/과매도 지표입니다.
- **SMA_5 (단기 이동평균)**: 최근 5일(약 1주일)간 종가의 단순 평균값입니다.
- **SMA_20 (중기 이동평균)**: 최근 20일(약 1개월)간 종가의 단순 평균값입니다.
- **Volatility (변동성)**: 최근 20일간 일일 수익률의 표준편차로, 해당 주식의 위험도(Risk)를 나타냅니다.
- **Log Return (일일 로그 수익률)**: 연속 복리 효과와 통계적 안정성을 반영한 주가 변동률입니다. 수식은 $\ln(\text{Close}_{t} / \text{Close}_{t-1})$ 입니다.

### 3. 타겟 데이터 (Target Data)
- **Target_Next_Return (익일 로그 수익률)**: AI 모델이 최종적으로 맞춰야 하는 정답지(Label)입니다. 2번에서 구한 일일 로그 수익률을 하루 위로 끌어올려(shift), '오늘' 시점의 데이터 행에 '내일'의 수익률이 매핑되도록 만든 데이터입니다.

## 🛠 설치 및 환경 설정 (Installation)

Python 3.8 이상의 환경을 권장합니다.

1. **가상 환경 생성 및 활성화 (선택 사항)**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **의존성 라이브러리 설치**:
   ```bash
   pip install -r requirements.txt
   ```
   *PyTorch의 경우, 사용 중인 GPU 환경(CUDA 버전 등)에 맞게 설치 명령어를 조정해야 할 수 있습니다.*

## 🏃‍♂️ 실행 가이드 (How to Run)

프로젝트를 **처음 실행할 때(초기 학습)**와 **매일 장 마감 후(일일 점진적 학습)** 실행하는 방법이 다릅니다. 아래 가이드에 따라 순서대로 실행해주세요.

### 🌟 1. 처음 사용할 때 (초기 전체 학습)

처음 환경을 세팅하고 10년 치 데이터를 모아 초기 모델을 학습시키는 과정입니다. 처음 1회만 수행하면 됩니다.

**1-1. 전체 과거 데이터 수집**  
S&P 500 종목의 10년 치 주가 데이터를 다운로드합니다.
```bash
python data_collection/sp500_data_collector.py
```
> 완료 시 `data/raw/` 폴더에 개별 종목의 과거 데이터 CSV 파일이 저장됩니다.

**1-2. 데이터 전처리 및 피처 엔지니어링**  
수집한 주가 데이터를 바탕으로 기술적 지표(RSI, SMA 등) 및 타겟 변수를 계산하여 통합 데이터셋을 생성합니다.
```bash
python feature_engineering/build_features.py
```
> 완료 시 `data/processed/sp500_features.csv` 파일로 저장됩니다.

**1-3. 전체 모델 초기 학습**  
통합 데이터셋을 활용해 Transformer 모델을 초기화하고 전체 학습을 진행합니다.
```bash
python main.py
```
> 완료 시 `models/best_transformer.pth` 위치에 최적의 모델 가중치가 저장됩니다.

**1-4. 모델 평가 및 포트폴리오 분석**  
학습된 모델 가중치를 로드하여 테스트 성능을 시각화하고 랭킹을 추출합니다.
```bash
python run_evaluation.py
```
> 단일 종목 시각화뿐만 아니라, 예상 수익률 기반 Top N(Long), Bottom N(Short) 포트폴리오 랭킹을 출력합니다.

---

### 🔄 2. 일일 사용 시 (장 마감 후 온라인 학습)

매일 장 마감 후 새로운 데이터를 받아 기존 모델을 미세조정(Fine-tuning)하는 과정입니다. 매일 1회 실행합니다.

**2-1. 최신 데이터 증분 업데이트**  
기존에 수집된 마지막 날짜 이후의 새로운 주가 데이터만 다운로드하여 기존 파일에 병합합니다.
```bash
python data_collection/sp500_incremental_update.py
```

**2-2. 전처리 및 피처 엔지니어링 갱신**  
새로 추가된 데이터가 포함된 전체 피처를 갱신합니다. 스크립트가 새로 다운로드된 데이터의 갱신 시간을 자동으로 감지하여 전처리를 다시 수행하므로, 별도의 옵션 없이 실행 버튼만 누르시면 됩니다.
```bash
python feature_engineering/build_features.py
```

**2-3. 일일 점진적 학습 (Online Learning)**  
매일 새로 들어온 데이터를 포함하여 최신 패턴을 기존 모델에 파인튜닝합니다.
```bash
python online_learning.py
```
> 이전 체크포인트 상태를 복원하고 적은 Epoch 수로 모델을 업데이트하여 빠르게 학습을 완료할 수 있습니다.

**2-4. 모델 평가 및 최신 포트폴리오 확인**  
방금 업데이트된 모델로 내일 장을 대비하기 위해 평가를 수행합니다.
```bash
python run_evaluation.py
```

## 📊 사용 기술 스택 (Tech Stack)

- **언어**: Python 3.x
- **딥러닝 프레임워크**: PyTorch
- **데이터 처리**: Pandas, NumPy, scikit-learn
- **시각화**: Matplotlib
- **데이터 소스**: yfinance

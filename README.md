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

**1. 전체 모델 초기 학습**  
초기 데이터 전처리, 모델 초기화 및 학습을 진행합니다.
```bash
python main.py
```
> 완료 시 `models/best_transformer.pth` 위치에 최적의 모델 가중치가 저장됩니다.

**2. 모델 평가 및 포트폴리오 분석**  
학습된 모델 가중치를 로드하여 테스트 세트 성능을 측정하고 시각화합니다.
```bash
python run_evaluation.py
```
> 단일 종목 시각화뿐만 아니라, 예상 수익률 기반 Top N(Long), Bottom N(Short) 포트폴리오 랭킹을 출력합니다.

**3. 일일 점진적 학습 (Online Learning)**  
매일 장 마감 후 새로운 주가 데이터가 수집되었을 때, 최신 데이터로 기존 모델을 파인튜닝합니다.
```bash
python online_learning.py
```
> 이전 체크포인트 상태를 복원하고 적은 Epoch 수로 모델을 업데이트하여 효율적인 유지가 가능합니다.

## 📊 사용 기술 스택 (Tech Stack)

- **언어**: Python 3.x
- **딥러닝 프레임워크**: PyTorch
- **데이터 처리**: Pandas, NumPy, scikit-learn
- **시각화**: Matplotlib
- **데이터 소스**: yfinance

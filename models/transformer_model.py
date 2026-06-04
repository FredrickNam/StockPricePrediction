"""
=============================================================================
 S&P 500 주가 수익률 예측용 시계열 Transformer 모델
 파일명: transformer_model.py
 논문  : "Attention is All You Need" (Vaswani et al., 2017)
 목적  : [Batch, SeqLen, Features] 형태의 주식 시계열 텐서를 입력받아
         다음 날의 로그 수익률(스칼라)을 예측하는 회귀 모델을 정의합니다.
 임포트: from models.transformer_model import TimeSeriesTransformer
 의존성: torch
=============================================================================
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# ▶ 위치 인코딩 (Positional Encoding)
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Sine / Cosine 기반 고정 위치 인코딩 레이어.

    --- 역할 ---
    Transformer 는 Self-Attention 구조 상 토큰(시점) 간의 순서를 구분하지 못합니다.
    위치 인코딩은 각 시점의 순서 정보를 임베딩 벡터에 주입하여 시간 흐름을
    모델이 인식할 수 있도록 합니다.

    --- 수식 (Vaswani et al., 2017) ---
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    pos  : 시퀀스 내 위치 (0 ~ seq_len-1)
    i    : 임베딩 차원 인덱스 (0 ~ d_model/2 - 1)

    Args:
        d_model  : Transformer 내부 임베딩 차원 (입력 투영 후 차원과 동일)
        max_len  : 지원하는 최대 시퀀스 길이 (기본 512, 실사용 seq_len 보다 크면 됨)
        dropout  : 위치 인코딩 적용 후 Dropout 비율
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # --- PE 행렬 사전 계산 (학습 파라미터 아님, 고정값) ---
        # pe: (max_len, d_model) 초기화
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1) -- 각 행이 시퀀스 위치 인덱스
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,) -- 주파수 스케일 팩터
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원: cos

        # pe 를 버퍼로 등록: 모델 저장/로드 시 포함되지만 역전파 대상 아님
        # unsqueeze(0) -> (1, max_len, d_model) : 배치 차원 브로드캐스팅 대비
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [Batch, SeqLen, d_model]
        Returns:
            x: [Batch, SeqLen, d_model]  (위치 인코딩 더하기 후 Dropout)
        """
        # self.pe[:, :x.size(1), :] -> [1, SeqLen, d_model] 슬라이싱 후 브로드캐스트
        x = x + self.pe[:, : x.size(1), :]  # 위치 정보 주입 (덧셈, 학습 없음)
        return self.dropout(x)


# =============================================================================
# ▶ 시계열 Transformer 메인 모델
# =============================================================================
class TimeSeriesTransformer(nn.Module):
    """
    주식 수익률 예측용 시계열 Transformer (인코더 전용 구조).

    --- 전체 데이터 흐름 ---
    입력
      -> [1] 입력 투영 (Linear)           : [B, S, F] -> [B, S, d_model]
      -> [2] 위치 인코딩 + Dropout        : [B, S, d_model] (차원 유지)
      -> [3] Transformer Encoder (x N)    : [B, S, d_model] (차원 유지)
      -> [4] 마지막 시점 슬라이싱         : [B, d_model]
      -> [5] 출력 헤드 (Linear)           : [B, 1]
      -> [6] squeeze(-1)                  : [B]
    출력: 배치 내 각 샘플의 예측 수익률 스칼라

    Args:
        n_features    : 입력 특성 수 (데이터셋 빌더의 n_features, 기본 11)
        d_model       : Transformer 내부 임베딩 차원 (기본 64)
        nhead         : Multi-Head Attention 헤드 수 (d_model 의 약수, 기본 4)
        num_layers    : TransformerEncoderLayer 반복 수 (기본 2)
        dim_feedforward: FFN(Position-wise Feed-Forward) 은닉 차원 (기본 256)
        dropout       : Dropout 비율 (기본 0.1)
        max_seq_len   : 위치 인코딩 최대 시퀀스 길이 (기본 512)

    * nhead 는 반드시 d_model 의 약수여야 합니다.
      예) d_model=64, nhead=4  ->  head 당 차원 = 64/4 = 16  (정수, 유효)
          d_model=64, nhead=5  ->  64 % 5 != 0  (오류 발생)
    """

    def __init__(
        self,
        n_features:     int   = 11,
        d_model:        int   = 64,
        nhead:          int   = 4,
        num_layers:     int   = 2,
        dim_feedforward: int  = 256,
        dropout:        float = 0.1,
        max_seq_len:    int   = 512,
    ) -> None:
        super().__init__()

        # 파라미터 검증
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model({d_model}) 은 nhead({nhead}) 의 배수여야 합니다. "
                f"(d_model % nhead = {d_model % nhead})"
            )

        # ------------------------------------------------------------------
        # [1] 입력 투영 레이어 (Input Projection)
        # ------------------------------------------------------------------
        # 원시 특성 차원(n_features=11) 을 Transformer 내부 차원(d_model) 으로 변환.
        # 선형 변환이므로 학습 가능한 가중치(W, b)가 존재합니다.
        self.input_projection = nn.Linear(n_features, d_model)

        # ------------------------------------------------------------------
        # [2] 위치 인코딩 (Positional Encoding)
        # ------------------------------------------------------------------
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # ------------------------------------------------------------------
        # [3] Transformer 인코더
        # ------------------------------------------------------------------
        # TransformerEncoderLayer: 단일 블록 (Self-Attention + FFN + LayerNorm)
        #   batch_first=True : 입력 텐서 차원 순서를 [B, S, d_model] 로 통일
        #                       (False 이면 [S, B, d_model] 로 혼선 가능)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # [B, S, d_model] 입력 보장
            norm_first=True,    # Pre-LayerNorm (학습 안정성 향상, GPT-3 방식)
        )

        # TransformerEncoder: 위 레이어를 num_layers 회 반복 적층
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # 가변 길이 패킹 비활성 (고정 길이 사용)
        )

        # ------------------------------------------------------------------
        # [4] 출력 헤드 (Prediction Head)
        # ------------------------------------------------------------------
        # 마지막 시점의 은닉 상태(d_model) -> 수익률 스칼라(1) 회귀
        self.output_head = nn.Linear(d_model, 1)

        # 모델 가중치 초기화
        self._init_weights()

    # ------------------------------------------------------------------
    # 가중치 초기화 (Xavier Uniform)
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        선형 레이어 가중치를 Xavier Uniform 으로 초기화합니다.
        Xavier 초기화는 깊은 네트워크에서 기울기 소실/폭발을 억제합니다.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # forward (순전파)
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 연산. 각 단계별 텐서 형태 변화를 주석으로 명시합니다.

        Args:
            x: 입력 텐서  [Batch, SeqLen, n_features]
               예) [64, 20, 11]

        Returns:
            out: 예측 수익률  [Batch]
                 예) [64]
        """
        # ------ [1] 입력 투영 -----------------------------------------------
        # [B, S, n_features]  ->  [B, S, d_model]
        # 예) [64, 20, 11]    ->  [64, 20, 64]
        x = self.input_projection(x)

        # ------ [2] 위치 인코딩 + Dropout ------------------------------------
        # [B, S, d_model]  ->  [B, S, d_model]  (차원 변화 없음)
        # 예) [64, 20, 64]  ->  [64, 20, 64]
        x = self.pos_encoding(x)

        # ------ [3] Transformer 인코더 (Multi-Head Self-Attention) -----------
        # [B, S, d_model]  ->  [B, S, d_model]  (차원 변화 없음)
        # 예) [64, 20, 64]  ->  [64, 20, 64]
        # 내부적으로 Q, K, V 를 생성하고 Self-Attention + FFN 을 num_layers 회 반복
        x = self.transformer_encoder(x)

        # ------ [4] 마지막 시점 슬라이싱 ------------------------------------
        # [B, S, d_model]  ->  [B, d_model]
        # 예) [64, 20, 64]  ->  [64, 64]
        # t 일까지의 정보를 모두 집약한 마지막 시점(t)의 은닉 벡터만 추출합니다.
        # 이 벡터가 '내일(t+1) 수익률 예측'의 기반이 됩니다.
        x = x[:, -1, :]

        # ------ [5] 출력 헤드 (회귀) ----------------------------------------
        # [B, d_model]  ->  [B, 1]
        # 예) [64, 64]   ->  [64, 1]
        x = self.output_head(x)

        # ------ [6] 불필요한 차원 제거 (squeeze) ----------------------------
        # [B, 1]  ->  [B]
        # 예) [64, 1]  ->  [64]
        # Loss 계산(MSELoss 등)에서 target [B] 와 차원을 맞추기 위해 제거합니다.
        out = x.squeeze(-1)

        return out

    # ------------------------------------------------------------------
    # 유틸리티: 파라미터 수 출력
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """
        학습 가능한 전체 파라미터 수를 반환합니다.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# ▶ 동작 검증 (직접 실행 시)
# =============================================================================
if __name__ == "__main__":
    import sys

    # --- 기본 설정값 (데이터셋 빌더와 동일하게 맞춤) ---
    BATCH_SIZE  = 64
    SEQ_LEN     = 20
    N_FEATURES  = 11

    # --- 모델 인스턴스 생성 ---
    model = TimeSeriesTransformer(
        n_features=N_FEATURES,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    model.eval()

    print("=" * 60)
    print(" TimeSeriesTransformer 구조 확인")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"학습 가능 파라미터 수 : {model.count_parameters():,}")
    print("-" * 60)

    # --- 더미 입력으로 forward pass 확인 ---
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
    print(f"입력 텐서 형태  : {tuple(dummy_input.shape)}")

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"출력 텐서 형태  : {tuple(dummy_output.shape)}")
    print(f"출력 값 범위    : [{dummy_output.min().item():.4f}, {dummy_output.max().item():.4f}]")
    print(f"출력 dtype      : {dummy_output.dtype}")
    print("=" * 60)
    print("forward pass 정상 완료.")

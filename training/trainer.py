"""
=============================================================================
 S&P 500 Transformer 모델 학습 파이프라인 (5단계: 학습 및 최적화)
 파일명: trainer.py
 목적  : TimeSeriesTransformer 모델을 학습·검증하고,
         조기 종료(Early Stopping) 및 최적 모델 저장을 관리합니다.
 실행  : python training/trainer.py
 임포트: from training.trainer import ModelTrainer
 의존성: torch, numpy
=============================================================================
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

# =============================================================================
# ▶ 경로 상수
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

# =============================================================================
# ▶ 로거 설정
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModelTrainer")


# =============================================================================
# ▶ ModelTrainer 클래스
# =============================================================================
class ModelTrainer:
    """
    TimeSeriesTransformer 의 학습, 검증, 조기 종료, 체크포인트 저장을
    통합 관리하는 Trainer 클래스.

    --- 학습 파이프라인 순서 ---
    (1) 손실 함수  : MSELoss  (회귀 기본)
    (2) 규제       : AdamW 의 L2(Weight Decay) + 커스텀 L1 Penalty (Elastic Net)
    (3) 학습 루프  : 매 Epoch -> Train -> Validation -> 기록
    (4) 조기 종료  : patience 만큼 Val Loss 미개선 시 학습 종료
    (5) 체크포인트 : Val Loss 최저 갱신 시 최적 가중치 저장

    Args:
        model          : 학습할 TimeSeriesTransformer 모델 인스턴스
        train_loader   : 학습용 DataLoader (shuffle=True)
        val_loader     : 검증용 DataLoader (shuffle=False)
        lr             : AdamW 초기 학습률 (기본 1e-3)
        weight_decay   : AdamW L2 규제 계수 (기본 1e-4)
        l1_alpha       : 커스텀 L1 Penalty 계수 (기본 1e-5)
        epochs         : 최대 학습 Epoch 수 (기본 100)
        patience       : 조기 종료 인내 횟수 (기본 10)
        save_path      : 최적 모델 가중치 저장 경로
        device         : 학습 장치 (None 이면 CUDA 자동 감지)
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        l1_alpha:     float = 1e-5,
        epochs:       int   = 100,
        patience:     int   = 10,
        save_path:    str   = os.path.join(MODEL_SAVE_DIR, "best_transformer.pth"),
        device:       torch.device = None,
        feat_scaler=None,
        target_scaler=None,
    ) -> None:
        # ------------------------------------------------------------------
        # 장치 설정: CUDA 가 있으면 GPU, 없으면 CPU
        # ------------------------------------------------------------------
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.epochs       = epochs
        self.patience     = patience
        self.l1_alpha     = l1_alpha
        self.save_path    = save_path

        # ------------------------------------------------------------------
        # 스케일러 보관 (체크포인트 무결성용)
        # ------------------------------------------------------------------
        self.feat_scaler   = feat_scaler
        self.target_scaler = target_scaler

        # ------------------------------------------------------------------
        # (1) 손실 함수: MSELoss
        # ------------------------------------------------------------------
        # MSE(Mean Squared Error) = 예측값과 실제값 차이의 제곱 평균
        # 회귀 문제의 표준 손실 함수로, 큰 오차에 더 강하게 패널티를 부여합니다.
        # reduction="mean": 배치 내 모든 샘플의 평균 손실을 반환합니다.
        self.criterion = nn.MSELoss(reduction="mean")

        # ------------------------------------------------------------------
        # (2) 옵티마이저: AdamW (L2 규제 내장)
        # ------------------------------------------------------------------
        # AdamW 는 Adam 의 개선 버전으로 가중치 감쇠(Weight Decay)를
        # Gradient 업데이트와 분리하여 적용합니다.
        # weight_decay 는 L2 규제 계수로, 가중치 절댓값이 커질수록
        # 페널티를 부여하여 과적합을 억제합니다.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # ------------------------------------------------------------------
        # 학습 이력 저장 (에포크별 Train / Val Loss)
        # ------------------------------------------------------------------
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss":   [],
        }

        # 조기 종료 상태 변수
        self._best_val_loss:   float = float("inf")  # 역대 최저 Val Loss
        self._patience_counter: int  = 0              # 미개선 연속 횟수
        self._stopped_epoch:    int  = 0              # 조기 종료된 에포크

        # 저장 디렉토리 생성
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        logger.info(
            "--- ModelTrainer 초기화 완료 ---\n"
            f"    장치           : {self.device}\n"
            f"    최대 Epoch     : {self.epochs}\n"
            f"    Early Stopping : patience={self.patience}\n"
            f"    L1 alpha       : {self.l1_alpha}\n"
            f"    AdamW lr       : {lr},  weight_decay={weight_decay}\n"
            f"    저장 경로      : {self.save_path}"
        )

    # ------------------------------------------------------------------
    # L1 페널티 계산 (커스텀 Elastic Net 구현)
    # ------------------------------------------------------------------
    def _compute_l1_penalty(self) -> Tensor:
        """
        모델의 모든 학습 가능 가중치(weight)에 대한 L1 Norm 합을 계산합니다.

        --- 의도 ---
        AdamW 는 L2 규제만 기본 제공합니다.
        L1 규제를 추가하면 Elastic Net 효과를 얻을 수 있습니다.
        L1 은 가중치를 정확히 0으로 수렴시키는 희소성(Sparsity) 효과가 있어
        불필요한 특성의 영향을 자동으로 제거합니다.

        * bias 파라미터는 과적합에 덜 기여하므로 L1 대상에서 제외합니다.
        * 실제 최종 손실: MSE + l1_alpha * sum(|w_i|)

        Returns:
            l1_penalty: 스칼라 Tensor (역전파 가능)
        """
        l1_penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            # bias 제외, weight 만 L1 페널티 대상
            if param.requires_grad and "bias" not in name:
                l1_penalty = l1_penalty + param.abs().sum()
        return l1_penalty

    # ------------------------------------------------------------------
    # 단일 에포크 학습 루프
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        """
        Train DataLoader 를 전체 순회하며 한 에포크를 학습합니다.

        --- 루프 흐름 ---
        (a) model.train()    : Dropout, BatchNorm 등 학습 모드 활성화
        (b) 순전파(Forward)  : 모델이 예측값 생성
        (c) 손실 계산        : MSELoss + L1 Penalty (Elastic Net)
        (d) 역전파(Backward) : 기울기 계산 (gradient 누적 방지 주의)
        (e) 기울기 클리핑    : gradient exploding 방지 (max_norm=1.0)
        (f) 파라미터 업데이트: optimizer.step()

        Returns:
            epoch 평균 Train Loss (float)
        """
        self.model.train()  # 학습 모드 전환 (Dropout 활성화 등)
        total_loss = 0.0
        n_batches  = 0
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)

        for i, (X_batch, y_batch) in enumerate(self.train_loader, start=1):
            # --- 텐서를 학습 장치(CPU/GPU)로 이동 ---
            # X_batch: [Batch, SeqLen, Features]  예) [64, 20, 11]
            # y_batch: [Batch]                     예) [64]
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # (b) 순전파: 모델 예측값 생성
            # y_pred: [Batch]  예) [64]
            y_pred = self.model(X_batch)

            # (c-1) MSE 손실 계산
            # mse_loss: 스칼라 Tensor
            mse_loss = self.criterion(y_pred, y_batch)

            # (c-2) 커스텀 L1 페널티 계산
            # l1_penalty: 스칼라 Tensor (역전파 가능)
            l1_penalty = self._compute_l1_penalty()

            # (c-3) 최종 손실: Elastic Net = MSE + alpha * L1
            # 두 항을 합산하여 단일 역전파로 통합 최적화
            loss = mse_loss + self.l1_alpha * l1_penalty

            # (d) 역전파 전 기울기 초기화
            # 초기화하지 않으면 이전 배치의 gradient 가 누적됩니다.
            self.optimizer.zero_grad()

            # 역전파: loss 에 대한 모든 파라미터의 편미분 계산
            loss.backward()

            # (e) 기울기 클리핑: gradient 의 L2 norm 이 max_norm 을 초과하면 스케일 조정
            # Transformer 처럼 깊은 모델에서 기울기 폭발(Exploding Gradient) 방지
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # (f) 파라미터 업데이트 (AdamW 스텝)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if i % log_interval == 0 or i == total_batches:
                progress_pct = (i / total_batches) * 100
                logger.info(
                    f"    [Epoch {epoch}] Train 진행률: {progress_pct:>5.1f}% "
                    f"({i:>5}/{total_batches}) | Batch Loss: {loss.item():.6f}"
                )

        return total_loss / n_batches

    # ------------------------------------------------------------------
    # 단일 에포크 검증 루프
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate_one_epoch(self) -> float:
        """
        Validation DataLoader 를 전체 순회하며 검증 손실을 계산합니다.

        --- 설계 원칙 ---
        * model.eval()        : Dropout 비활성화, BatchNorm 고정 모드 전환
        * torch.no_grad()     : 기울기 계산 비활성화 (메모리 절약, 속도 향상)
          데코레이터(@torch.no_grad()) 로 메서드 전체에 적용합니다.
        * 검증에서는 L1 페널티를 포함하지 않고 순수 MSELoss 만 측정합니다.
          모델의 실제 예측 성능만 분리하여 평가하기 위함입니다.

        Returns:
            epoch 평균 Validation Loss (float)
        """
        self.model.eval()  # 평가 모드 전환 (Dropout 비활성화)
        total_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 순전파만 수행 (역전파 없음)
            y_pred = self.model(X_batch)

            # 순수 MSE Loss (L1 페널티 제외 - 검증 단계는 성능 측정만)
            loss = self.criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / n_batches

    # ------------------------------------------------------------------
    # 조기 종료 및 체크포인트 판단
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """모델 파라미터, 옵티마이저 상태, 에포크를 모두 포함한 무결성 저장"""
        import pickle
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self._best_val_loss,
            "feat_scaler": pickle.dumps(self.feat_scaler) if self.feat_scaler else None,
            "target_scaler": pickle.dumps(self.target_scaler) if self.target_scaler else None
        }
        
        # 항상 최신 상태를 저장 (재학습 대비)
        last_path = self.save_path.replace("best_", "last_")
        torch.save(checkpoint, last_path)
        
        if is_best:
            torch.save(checkpoint, self.save_path)

    def load_checkpoint_for_incremental(self, checkpoint_path: str) -> int:
        """점진적 학습(Incremental Learning)을 위한 완전한 상태 복원"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"체크포인트 로드 완료: {checkpoint_path} (시작 Epoch: {start_epoch})")
        return start_epoch

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """
        Val Loss 기준으로 조기 종료 여부를 판단하고 최적 모델을 저장합니다.

        --- 동작 ---
        * val_loss 가 역대 최저(best_val_loss) 를 갱신하면:
          - patience_counter 리셋
          - 최적 가중치를 파일로 저장 (state_dict 방식)
        * 갱신하지 못하면:
          - patience_counter 1 증가
          - patience_counter >= patience 이면 True 반환 (학습 중단 신호)

        state_dict 저장 방식을 사용하는 이유:
          모델 전체 객체가 아닌 가중치 딕셔너리만 저장하므로
          파일 크기가 작고, 다른 코드에서 모델 구조 재정의 후 로드 가능합니다.

        Args:
            val_loss: 현재 에포크의 검증 손실
            epoch   : 현재 에포크 번호 (1-indexed)

        Returns:
            True: 조기 종료 조건 충족 (학습 중단)
            False: 계속 학습
        """
        is_best = False
        if val_loss < self._best_val_loss:
            # --- 최저 Val Loss 갱신 ---
            improvement = self._best_val_loss - val_loss
            self._best_val_loss    = val_loss
            self._patience_counter = 0  # 카운터 리셋
            is_best = True

            logger.info(
                f"    [*] Val Loss 최저 갱신: {val_loss:.6f} "
                f"(개선폭: {improvement:.6f}) -> 모델 저장 완료"
            )
        else:
            # --- 미개선 ---
            self._patience_counter += 1
            logger.debug(
                f"    Val Loss 미개선 "
                f"({self._patience_counter}/{self.patience})"
            )
            
        # 무결성 체크포인트 저장 (항상 last 저장, 최적 갱신 시 best 덮어쓰기)
        self.save_checkpoint(epoch, val_loss, is_best=is_best)

        if self._patience_counter >= self.patience:
            self._stopped_epoch = epoch
            return True  # 조기 종료

        return False

    # ------------------------------------------------------------------
    # 전체 학습 루프 (메인 메서드)
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, List[float]]:
        """
        지정된 epochs 동안 학습과 검증을 반복합니다.
        조기 종료 조건 충족 시 즉시 종료합니다.

        --- 전체 루프 구조 ---
        for epoch in range(1, epochs+1):
            train_loss = _train_one_epoch()
            val_loss   = _validate_one_epoch()
            기록 및 출력
            _check_early_stopping() -> True 이면 break

        Returns:
            history: {"train_loss": [...], "val_loss": [...]}
                     각 원소는 에포크별 평균 손실값
        """
        logger.info(f"{'='*60}")
        logger.info(" 학습 시작")
        logger.info(f"{'='*60}")
        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # --- 학습 ---
            train_loss = self._train_one_epoch(epoch)

            # --- 검증 ---
            val_loss = self._validate_one_epoch()

            # --- 이력 기록 ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            elapsed = time.time() - epoch_start

            # --- 콘솔 출력 ---
            logger.info(
                f"Epoch [{epoch:>3}/{self.epochs}] "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f} "
                f"| {elapsed:.1f}s"
            )

            # --- 조기 종료 및 체크포인트 판단 ---
            stop = self._check_early_stopping(val_loss, epoch)
            if stop:
                logger.info(
                    f"\n--- 조기 종료 (Early Stopping) ---\n"
                    f"    {self.patience} 에포크 동안 Val Loss 미개선\n"
                    f"    최종 종료 에포크 : {epoch}\n"
                    f"    최저 Val Loss   : {self._best_val_loss:.6f}"
                )
                break

        total_elapsed = time.time() - total_start
        logger.info(f"{'='*60}")
        logger.info(
            f" 학습 완료\n"
            f"    총 소요 시간  : {total_elapsed:.1f}s\n"
            f"    최저 Val Loss : {self._best_val_loss:.6f}\n"
            f"    최적 모델     : {self.save_path}"
        )
        logger.info(f"{'='*60}")

        return self.history

    # ------------------------------------------------------------------
    # 최적 모델 가중치 로드 (학습 완료 후 복원)
    # ------------------------------------------------------------------
    def load_best_model(self) -> nn.Module:
        """
        저장된 최적 가중치를 모델에 로드하고 반환합니다.

        학습 완료 후 테스트 평가 또는 추론(Inference) 전에 호출하십시오.
        마지막 에포크의 가중치가 아닌 Val Loss 가 가장 낮았던 시점의
        가중치를 복원하여 일반화 성능을 보장합니다.

        Returns:
            최적 가중치가 로드된 모델 (eval 모드)
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"저장된 모델 파일을 찾을 수 없습니다: {self.save_path}\n"
                f"fit() 을 먼저 실행하십시오."
            )
        # map_location: GPU 저장 모델을 CPU 에서 로드할 때도 안전하게 처리
        checkpoint = torch.load(self.save_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)  # 하위 호환성 (이전 state_dict 전용 방식)
        self.model.eval()
        logger.info(f"최적 모델 로드 완료: {self.save_path}")
        return self.model


# =============================================================================
# ▶ 동작 검증 (직접 실행 시)
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    from models.transformer_model import TimeSeriesTransformer
    from dataset.dataset_builder import build_dataloaders

    # --- 데이터 로더 빌드 ---
    result = build_dataloaders()
    train_loader   = result["train_loader"]
    val_loader     = result["val_loader"]
    n_features     = result["n_features"]

    # --- 모델 생성 ---
    model = TimeSeriesTransformer(
        n_features=n_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    # --- Trainer 생성 및 학습 ---
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        weight_decay=1e-4,
        l1_alpha=1e-5,
        epochs=50,          # 검증용 축소 실행
        patience=10,
    )

    history = trainer.fit()

    # --- 최적 모델 복원 ---
    best_model = trainer.load_best_model()

    # --- 학습 이력 요약 ---
    logger.info(
        f"--- 학습 이력 요약 ---\n"
        f"    기록된 에포크 수 : {len(history['train_loss'])}\n"
        f"    최종 Train Loss  : {history['train_loss'][-1]:.6f}\n"
        f"    최종 Val Loss    : {history['val_loss'][-1]:.6f}"
    )

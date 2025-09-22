# 🧠 실행 기능 예측 AI 모델: 내 몸의 상태로 머리를 예측하다

> **An AI Model for Predicting Executive Function: Predicting Cognitive State from Biometric Data**

## 📋 Overview

본 연구는 바이오메디컬 AI(Biomedical AI) 수업의 팀 프로젝트로, 웨어러블 기기에서 수집된 **생체 및 행동 데이터(Biometric and Behavioral Data)**를 기반으로 개인의 **실행 기능(Executive Function) 수준**을 예측하는 AI 모델을 개발한 연구입니다.

실행 기능은 목표 지향적 행동에 필수적인 고차원적 인지 능력으로, 학업 성취도와 일상 기능의 핵심입니다. 하지만 기존의 실행 기능 측정 방식은 일회성 검사에 의존하여 연속적인 모니터링이 어렵다는 한계가 있었습니다. 본 연구는 **웨어러블 기기**를 통해 일상에서 손쉽게 수집할 수 있는 심박수, 수면, 스트레스 등의 데이터를 활용하여, 사용자가 자신의 인지 상태를 **편리하고 즉각적으로 확인**할 수 있는 AI 모델을 제시합니다.

### 🎯 Research Objectives

1.  **실행 기능 영향 요인 분석**: 심박수, 스트레스, 수면, 카페인 섭취 등 생체 및 행동 데이터가 실행 기능에 미치는 영향 분석
2.  **AI 기반 예측 모델 개발**: 시계열 데이터(심박수, 스트레스)와 정형 데이터를 통합적으로 학습하여 실행 기능의 5가지 레벨(매우 낮음, 낮음, 보통, 높음, 매우 높음)을 분류하는 모델 개발
3.  **개인 맞춤형 인지 상태 모니터링**: 개인화된 데이터를 통해 사용자가 자신의 실행 기능 변화를 추적하고, 최상의 퍼포먼스를 위한 라이프스타일 조절 가이드라인 제시

-----

## 🏗️ Architecture & Models

### Base Models & Concepts

  - **데이터 소스**: Xiaomi Mi Band 9를 통해 24시간 수집된 **심박수(Heart Rate), 수면(Sleep), 스트레스(Stress)** 데이터 및 사용자가 직접 기록한 **카페인 섭취량(Caffeine Intake)**
  - **실행 기능 평가**: 인지 억제 및 주의 전환 능력을 측정하는 표준화된 심리학 테스트인 \*\*스트룹 테스트(Stroop Color-Word Test)\*\*를 통해 실행 기능 점수 측정
  - **AI 모델**:
      - **PyCaret (AutoML)**: 최적의 클래식 머신러닝 모델 탐색
      - **CNN-LSTM (Deep Learning)**: 심박수, 스트레스와 같은 시계열 데이터의 시간적 패턴(국소적, 장기적 특징)을 효과적으로 추출하기 위해 직접 설계한 딥러닝 모델

### Experimental Framework

| 데이터 종류 | 처리 방식 | 모델 입력 | 특징 |
| --- | --- | --- | --- |
| **시계열 데이터** (심박수, 스트레스) | 테스트 직전 30분 데이터 추출 (평균, 최소/최대, 표준편차 등) | **CNN-LSTM** 모델의 Conv1D-LSTM 레이어 | 시간의 흐름에 따른 동적 변화 및 패턴 학습에 유리 |
| **정형 데이터** (수면 시간, 카페인 섭취 여부/시간, 개인 신체 정보) | 테이블 형태 데이터 | **CNN-LSTM** 모델의 Fully Connected Layer | 특정 시점의 상태 정보 반영 |
| **타겟 변수** (실행 기능) | 스트룹 테스트 결과(정답 수, 반응 시간)를 종합하여 **Stroop Score** 산출 후, Z-score 기반 5개 레벨로 범주화 | 모델의 최종 출력 (분류) | 객관적이고 정량화된 인지 능력 지표 |

-----

## 📁 Project Structure

```
Biomedical_AI/
├── BMAI.ipynb                  # 데이터 전처리, 모델 학습 및 평가 코드
├── best_model.h5               # 최종 선택된 CNN-LSTM 모델 가중치
├── best_model_fold_1.h5        # 교차 검증 폴드 1 모델
├── best_model_fold_2.h5        # 교차 검증 폴드 2 모델
├── best_model_fold_3.h5        # 교차 검증 폴드 3 모델
├── best_model_fold_4.h5        # 교차 검증 폴드 4 모델
├── best_model_fold_5.h5        # 교차 검증 폴드 5 모델
├── *.png                       # 분석 과정 시각화 이미지 (히트맵, 예측 결과 등)
└── 바메3조최종발표.pdf           # 프로젝트 최종 발표 자료
```

-----

## 🔬 Methodology

### 1\. 데이터 수집 (Data Collection)

  - **웨어러블 데이터**: 샤오미 미밴드9의 상시 측정 옵션을 통해 24시간 심박수, 수면, 스트레스 데이터 자동 수집
  - **행동 데이터**: 카페인 섭취 시, 양과 시간을 수동으로 기록
  - **실행 기능 데이터**: 매일 기상 후부터 취침 전까지 3시간 간격, 총 5회 스트룹 테스트를 수행하여 반응 속도, 정답 수 등 결과 기록

### 2\. 데이터 전처리 및 Feature Engineering

  - **개인별 표준화**: 모든 데이터는 개인차를 극복하기 위해 Z-score 표준화 진행
  - **Stroop Score 이상치 제거**: Box Plot을 사용하여 테스트 ID별 이상치 데이터 제거
  - **실행 기능 점수화 및 레벨 분류**: 정답 수(R)와 반응 시간(T)을 통합하여 최종 `StroopScore`를 계산하고, 이를 정규분포 기반 5개 구간(매우 낮음 \~ 매우 높음)으로 나누어 라벨링
  - **시계열 Feature 추출**: 각 스트룹 테스트 시점 기준, 직전 30분의 심박수 및 스트레스 시계열 데이터에서 **평균, 최소값, 최대값, 표준편차** 등의 통계적 피처 추출
  - **카페인 Feature**: 테스트 전 카페인 섭취 유무(Flag) 및 섭취 후 경과 시간 계산

### 3\. 모델 선택 및 평가

  - **통계적 유의성 검정**: ANOVA, LMM의 정규성 가정이 만족되지 않아 비모수 검정인 **Friedman Test**를 통해 피처 유의성 확인 → 단순 선형 관계가 없어 딥러닝 모델이 적합할 것으로 판단
  - **모델 비교**: AutoML(PyCaret)을 통한 클래식 ML 모델과 직접 설계한 CNN-LSTM 모델의 성능 비교
  - **교차 검증**: 데이터 수가 적은 점을 감안하여 **Stratified K-Fold (k=5)** 교차 검증을 통해 모델의 일반화 성능을 안정적으로 평가

-----

## 📊 Experimental Results

### 모델 성능 비교 (5-Fold Cross-Validation)

| 성능 지표 | PyCaret (Best Model) | **CNN-LSTM (최종 모델)** |
| :--- | :--- | :--- |
| **정확도 (Accuracy)** | 0.384 ± 0.091 | **0.8187 ± 0.0511** |
| **정밀도 (Precision)** | 0.385 ± 0.088 | **0.8428 ± 0.0627** |
| **재현율 (Recall)** | 0.384 ± 0.094 | **0.8164 ± 0.0478** |
| **F1 점수 (F1-score)** | 0.377 ± 0.084 | **0.8094 ± 0.0475** |

*PyCaret 모델 중에서는 LDA(Linear Discriminant Analysis)가 0.4462의 정확도로 가장 높았으나, 이는 CNN-LSTM 모델의 성능에 크게 미치지 못했습니다.*

### Confusion Matrix (평균 혼동 행렬)

**PyCaret 모델**: 예측이 대각선에 집중되지 않고 전반적으로 퍼져 있어 분류 성능이 매우 낮음을 보여줍니다.

**CNN-LSTM 모델**: 예측 결과가 실제 레이블(True Label)과 일치하는 대각선 행렬에 명확하게 집중되어 있어, 높은 분류 정확도를 시각적으로 확인할 수 있습니다.

-----

## 🔍 Key Findings

### 1\. **CNN-LSTM 모델의 압도적인 성능**

  - 시계열 데이터의 복합적인 패턴을 학습하는 **CNN-LSTM 모델**이 약 \*\*81.9%\*\*의 정확도를 달성하며, 정형 데이터 기반의 클래식 ML 모델(최고 약 44.6%)보다 월등히 뛰어난 성능을 보였습니다. 이는 실행 기능 예측에 있어 심박수와 스트레스의 **시간적 동특성**이 매우 중요한 정보임을 시사합니다.

### 2\. **비선형 관계의 중요성**

  - 개별 Feature들은 실행 기능 점수와 뚜렷한 **1:1 선형 상관관계**를 보이지 않았습니다. 하지만 딥러닝 모델은 이러한 변수들 간의 **비선형적이고 복잡한 상호작용**을 학습하여 높은 예측 성능을 달성할 수 있었습니다.

### 3\. **데이터의 개인화 및 맥락의 중요성**

  - 모든 데이터를 개인별로 표준화하는 과정이 모델 성능에 중요하게 작용했습니다. 이는 실행 기능에 영향을 미치는 생체 신호의 기준점이 개인마다 다름을 의미합니다.

-----

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### Running Experiments

1.  **Jupyter Notebook 실행**
    `BMAI.ipynb` 파일을 열어 데이터 로드, 전처리, 모델 학습 및 평가의 전체 과정을 순차적으로 실행할 수 있습니다.

2.  **학습된 모델 로드 및 예측**

    ```python
    from tensorflow.keras.models import load_model

    # 저장된 최종 모델 로드
    model = load_model('best_model.h5')

    # 새로운 데이터에 대한 예측 수행
    # new_data는 전처리 과정을 거친 입력 데이터 형태여야 함
    # prediction = model.predict(new_data)
    ```

-----

## 📋 Practical Guidelines & 활용 방안

본 연구 결과를 바탕으로 한 실용적인 기대 효과는 다음과 같습니다.

### 🎯 **개인 맞춤형 퍼포먼스 관리**

  - **전략**: 자신의 일별/시간별 실행 기능 예측 그래프를 확인하고, 인지 능력이 높은 시간에 중요한 학업이나 업무를 배치합니다.
  - **적용**: 학생, 연구원, 프로그래머 등 고도의 집중력이 필요한 사용자
  - **효과**: "언제 공부하는가"가 아닌 \*\*"어떤 실행 기능 상태에서 공부하는가"\*\*에 집중하여 학습 및 업무 효율 극대화

### ⚡ **실시간 인지 상태 모니터링**

  - **전략**: 웨어러블 기기를 착용하는 것만으로 자신의 현재 실행 기능 상태를 "매우 향상", "보통" 등으로 간편하게 확인하고 컨디션을 조절합니다.
  - **적용**: 운전자, 관제사 등 실시간 판단이 중요한 직업군
  - **효과**: 비용과 시간 소모 없이 뇌 상태를 간접적으로 확인하여 사고 위험 감소 및 수행 능력 향상

### 🔮 **장기적 라이프스타일 개선**

  - **전략**: 누적된 데이터를 통해 어떤 요인(수면 부족, 카페인 등)이 자신의 실행 기능에 부정적인 영향을 주는지 파악하고 생활 습관을 개선합니다.
  - **적용**: 만성 피로, 스트레스 등으로 인지 저하를 겪는 현대인
  - **효과**: 데이터 기반의 건강 관리를 통해 장기적인 삶의 질 향상

-----

## 📚 References

본 연구는 다음 주요 개념 및 연구들을 기반으로 합니다:

  - **Executive Function**: A set of cognitive processes that are necessary for the cognitive control of behavior.
  - **Stroop Effect**: A demonstration of cognitive interference where a delay in the reaction time of a task occurs due to a mismatch in stimuli.
  - **Time-Series Analysis**: Methods for analyzing time-series data in order to extract meaningful statistics and other characteristics of the data.
  - **CNN-LSTM**: A hybrid deep learning architecture that combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal feature extraction.

-----

## 👥 Contributors

  - **윤재하**

-----

**Keywords**: Executive Function, Cognitive Science, Biomedical AI, Wearable Sensors, Time-Series Analysis, Deep Learning, Stroop Test, Heart Rate, Stress, Sleep, Predictive Modeling

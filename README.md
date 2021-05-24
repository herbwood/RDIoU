# RDIoU Loss : Repulsion Distance IoU Loss for Reducing False Positive Rate in Crowd Scene Detection 

다수의 사람 혹은 군중을 탐지하는 과정에서 더 많은 bounding box를 생성함에 따라 거짓 양성 예측 비율이 높아지는 문제가 발생한다. 본 논문에서는 객체 탐지 모델의 거짓 양성 예측 비율을 감소시킬 수 있는 손실 함수를 제안한다. 제안한 손실 함수는 bounding box가 GT(Ground-Truth) box와 가까워지도록 유도함과 동시에 GT box와 겹쳐진 인접 객체 및 다른 bounding box와의 중심점 거리가 멀어지도록 유도한다. 또한 BFP(Balanced Feature Pyramid)를 사용하여 객체에 대한 위치 추정의 정확도를 향상시켰다. 실험 결과 제안한 방법을 사용할 경우 $MR^{-2}$ 비율은 1.89% 감소하였으며 AP, JI 비율은 기존 방법 대비 각각 0.5%, 0.58% 상승하여  거짓 양성 비율이 효과적으로 감소하는 것을 확인할 수 있었다. 

## RDIoU Loss function 



## Overall Architecture





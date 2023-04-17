### [Research-Project-I] 
# Audio-driven 3D facial animation with BlendShape
## 1. 연구 목적 (Problem Statement)

![Figure 1: Voice Operated Character Animation Network Architecture](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78ba516a-f950-4eae-9ed3-5ad4787d241a/voca_3d_character_animation2.png)

Figure 1: Voice Operated Character Animation Network Architecture

 본 연구는 실제 인간과 같은 성능을 가진 Audio-driven 3D Facial Animation을 BlendShape 3D Dataset으로 제작하는 데에 있다. 최근 NVIDIA의 NVAIL 프로그램에 참여하고 있는 연구원이 VOCA(Voice Operated Character Animation) 관련 딥러닝 알고리즘을 개발하였다. 이는 사람이 말하는 음성 사운드와 4D Scan한 사람 얼굴의 움직임 Dataset을 학습하여 표현한 것이다. 하지만 이는 4D Face Scan을 통해 만든 자체 Dataset인 VOCASET을 바탕으로 모델링한 것으로, 현재 3D Avatar제작에 보편적으로 사용되는 BlendShape를 사용하는 애니메이션 툴과는 달라 상용화가 힘들다.

 따라서 이러한 상용화와 관련된 문제점을 해결하기 위해서 BlendShape 3D DataSet을 이용한 학습을 통해, 현재 많이 사용되는 3D애니메이션 툴에서 쉽게 사용할 수 있는 Audio-driven 3D Facial Animation을 제작하고자 한다. 또한 BlendShape을 이용할 경우, 4D Face Scan을 통해 얻은 Data의 양보다 훨씬 방대한 양의 Dataset을 수집할 수 있기에, 기존에 개발된 것보다 더 자연스러운 인간의 형태를 띤 3D Animation을 제작할 수 있을 것으로 예상된다.
 따라서 본 연구를 통해 BlendShape의 Data를 통한 3D Facial Animation 추론이 가능해 진다면, 기존이 이미 BlendShape를 이용하여 제작된 3D Avatar에서 쉽게 이 기술을 사용할 수 있을 것이며, 앞으로 게임과 가상아바타 제작과 같은 다양한 분야에서 사용될 수 있을 것이다.

## 2. 연구 배경 (Motivation and Background)

 BlendShape 3D dataset을 사용하려는 이유는 현재 3D 얼굴 애니메이션을 생성하기 위해 가장 널리 사용되는 방법이기 때문이다. 기존에 이와 관련하여 완성도 높게 제작된 기술이 존재하였지만, 이를 학습하기 위해 사용된 DataSet이 보편적으로 사용되지 않는 값을 사용하는 경우가 많았다. Dataset이 보편적으로 사용되지 않는 기술을 사용할 경우, 정확도는 높을지라도 그 추출된 Mesh 데이터로 다시 여러가지 아바타를 생성하기 어려워진다. 하지만, BlendShape Data를 사용할 경우 Blender나 Maya와 같은 상용3D 애니메이션 툴을 이용하여 오디오기반 3D 아바타를 좀 더 쉽게 제작할 수 있을 것이다. 따라서 본 연구를 통해 좀 더 상용화된 Dataset을 이용한 학습을 통해, 이 기술이 적용되는 시점에서 좀 더 보편적으로 사용되기 쉽게 제작하고자 한다.

## 3. 연구 방법 (Design and Implementation)

 본 연구를 위해 처음에는 해당 분야에 가장 완성도 높은 모델인 NVIDIA의 VOCA(Voice Operated Character Animation)를 Baseline model로 하면서, 해당 모델을 적절히 변형하여 BlendShape Dataset을 이용하여 학습할 수 있도록 제작하는 방향으로 계획하였다. 하지만 VOCA model의 구조를 보니, 자체의 dataset인 VOCASET을 사용하는 경우에만 초점을 맞추어 제작되었으며, 이를 다른 형태의 dataset을 사용하도록 변형하기에는 다소 복잡하게 구현되어있었다. 따라서 이를 변형하여 model을 만드는 방안은 어려울 것이라 판단하였고, VOCA를 변형하는 방향이 아닌, 바이두에서 제공하는 DeepSpeech만을 사용하여 그 이외의 부분은 자체적으로 구현하여 모델을 제작하기로 계획하였다.

![Figure 2: MNIST Colab Example](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59777b5c-663a-4d4d-9289-11b22faa4241/MNIST_Colab_Example.png)

Figure 2: MNIST Colab Example

 먼저, Pytorch Lightning의 구조와 사용법을 익히고, 위와 같이 Google Colab을 통해 MNIST Dataset을 이용한간단한 예제를 돌려보며 model을 제작하기 위해 필요한 기초 지식을 학습하였다.

![Figure 3: LiveLinkFace and Blender with BlendShape Data](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32a976fb-92ec-40df-a842-ba162ba38ffe/Blender_Character_Animation.png)

Figure 3: LiveLinkFace and Blender with BlendShape Data

 또한, 앞선 계획서에 작성하였듯 Apple의 ARKit를 이용한 3D depth 카메라를 통해 BlendShape 3D Dataset을 수집하였다. 이때 dataset을 제작하기 위해 필요한 문장은 harvard sentence를 사용하였다. Epic Games의 Unreal Engine에서 Real Time Facial Capture를 위해 제작한 Live Link Face이라는 앱을 통해 실시간으로 얼굴의 움직임에 대한 BlendShape Dataset과 그에 따른 음성 Data를 수집할 수 있었다. 이렇게 해서 얻은 csv 파일을 3d Avatar에 적용되었을때 Animation을 확인할 수 있는 툴인 Blender의 사용법을 익히고, dataset들을 예시로 Animation이 잘 작동하는지 확인하였다.

이렇게 model을 제작하기에 앞서 필요한 배경지식을 습득한 이후에, 본격적으로 구현을 시작하였다.

- Data Preprocessing
먼저, 위에서 서술한 data수집 과정을 통해 얻은 dataset에서 BlendShape의 data중 말하는 것과 관련된 얼굴 하관외의 불필요한 값을 제거하고, 이에 해당하는 음성 파일이 이후에 있을 Encoder에 돌아갈 수 있도록 이를 Spectrogram으로 변환하였다.
    
    ![Figure 4: Data Preprocessing](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32feea8d-a04c-4e7c-8fd6-6edc4784c6a1/Data_Preprocessing.png)
    
    Figure 4: Data Preprocessing
    
- Encoding
 이후 DeepSpeech의 SimpleEncoder로 preprocessing한 audio data를 Encoding하였다. DeepSpeech의 Simple Encoder를 사용할 경우, 학습된 데이터를 바탕으로 더 좋은 결과물을 낼 수 있다. 하지만 구현과정에서 Simple Encoder에서 사용한 audio의 frame은 50이지만, 수집한 Audio Data는 60라는 점을 깨달았다. Simple Encoder를 사용하기 위해서는 simple encoder의 frame을 50에서 60으로 의도적으로 변형하는 과정이 필요하였다. interpolatefeatures()를 이용하여 변형하여 모델을 제작할 수는 있었지만, 그 과정으로 인해 학습 결과의 완성도에 조금의
영향은 생길 것이라 판단하였다.
    
     이에 DeepSpeech의 Simple Encoder를 사용하지 않고, 구현하는 방법도 고민해보았으나 위와 같은 단점을 감수하더라도 DeepSpeech를 사용하는 것이 더 좋은 결과를 낼 수 있을 것이라 생각하여 그대로 사용하기로 결정하였다.
    

![Figure 5: Encoding](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7ee8ba9a-846d-414d-8f25-adb0bc726d1a/simple_encoder_code.png)

Figure 5: Encoding

- Decoding
Decoding단계에서는 Encoding한 audio data를 바탕으로 BlendShape값을 inference한다. Decoder의 경우 직접 구현하는 방법을 선택하였다. 처음에는 가장 간단하게 Linear 하나만 있는 Fully Connected Layer를 사용하였지만, 더 좋은 결과를 도출하기 위해 여러 방법을 시도해보았다. 처음에 torch에서 제공하는 model을 사용하는 방법을 생각하였다. 그런 모델들은 이미 좋은 image dataset을 사용하여 pretrained된 model이기 때문에 잘 사용하면 model의 완성도가 높아질 것이라 생각하였다. 그 중 Resnet18을 사용하여 구현을 시도하였는데, 이 model의 dataset
과 dimension이 맞지 않아 data를 억지로 model에 끼워맞춰야하는 상황이었다. Resnet이 아닌 다른 model들도 시도해보았지만 모두 비슷한 상황이었고, 그렇게 data를 억지로 끼워맞추어 결과를 낸다면 의미있는 결과가 나올 수 없다고 판단하여, 다른 방법을 고려할 수 밖에 없었다. 그렇게 생각한 다른 방법은 직접 layer들을 쌓아 의미있는 모델을 만드는 것이었다. 이에 AdaptiveAvgPool, Linear, ReLU6등의 다양한 layer들을 추가하여 model을 제작하였다.
    
    ![Figure 6: Decoding](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5cd68c1a-bbcc-498f-9e1a-8be413ee02e4/Untitled.png)
    
    Figure 6: Decoding
    
     위와 같은 과정으로 model을 구현한 후, 나온 output에 noise가 많아 입술을 잘게 떠는 모습이 관찰되었다. 이러한 noise를 해결하기위해 exporter에 smoothing하는 과정을 넣었더니 잘게 떠는 문제는 해결되었지만, 입술의 움직임이 전체적으로 작게 움직여서 입모양이 명확하게 보이지 않는다는 새로운 문제가 생겼다. 하지만, noise가 있을 경우 음성과 입모양이 맞는지 판단하기조차 힘들어, smoothing과정을 넣는 것이 옳다고 판단하였고, 이에 window size를 적절히 조절해가며, model을 제작하였다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f548e775-2ad9-4527-8d81-c7d489629b9e/Untitled.png)
    
    ![Figure 7: Exporter Smoothing](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/43ddfd0a-d7eb-4c95-a02b-6015d3409698/exporter_smoothing_code.png)
    
    Figure 7: Exporter Smoothing
    
     또한, MSELoss나 CrossEntropyLoss와 같은 다양한 종류의 criterion이나, Adam,SGD 등의 optimizer를 적용해보고, learning rate등의 parameter값도 변경해보며 model을 수정하는 과정을 통해 가장 의미있는 결과를 보이는 model을 제작할 수 있었다.
    

## 4 연구 결과 및 평가 (Methodology and Evaluation)

 위의 과정을 통해 BlendShape를 통한 Audio-Driven 3D Facial Animation model을 구현하였다. 이번 과제연구에서 진행한 구현프로젝트의 결과는 특정 Audio에 따른 Animation이기 때문에, 제작된 model에 대한 성능의 평가 기준이 정말 모호하다. 명확한 평가기준이 있으면 더 좋은 모델을 만드는 데에 참고하여 사용하였겠지만 그럴만한 기준이 존재하지 않기에, 결론적으로는 제작된 Animation을 보는 ’사람’의 눈에 음성과 입모양 사이의 연결성이 어색하지 않게 보이는 모델이 완성도 높은 모델일 것이라 생각하였다. 구현 도중 어떤 방법이 더 좋을 지 판단해야하는 상황에서, 선택지에 대한 csv output을 통해 나온 Blender의 animation을 비교해보면서 더 자연스럽다고 판단되는 모델을 선택하여 제작하였다.

![Figure 8: model output CSV Example](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/172bada5-e378-4742-beff-68553c711be6/csv_example.png)

Figure 8: model output CSV Example

 아래는 audio와 model이 생성한 animation의 연결성에 대한 스스로의 평가다. 솔직히 제작한 animation의 입모양이 소리와 어느 정도 맞긴하지만, 여전히 명확하게 입모양이 보이진 않았다. 입술 떨림에 의한 noise를 줄이는 과정에서 전체적으로 입모양이 작아져서 그런것이라 예상하는데 window size를 조절하더라도 제대로 나오진 못했던 것 같다. 시간관계상 해당 구현을 이렇게 마무리할 수 밖에 없었지만, 더 명확한 모델을 제작하기 위해서는 더 다양한 시도를 해봐야 할 것 같다.

## 5 토론 및 전망 (Discussion and Future Work)

 Audio-Driven 3D Facial Animation with BlendShape라는 주제로 구현프로젝트 연구를 진행해보았다. 관련된 연구 중 가장 잘 알려진 VOCA model의 코드를 분석하고 이해하였다. BlendShape와의 연결을 위해, VOCA를 변형하는 방안보다는, DeepSpeech만을 이용하여 새로운 모델을 구현하는 방안을 택하였다. model 구현에 앞서 학습할 data를 Live Link Face App을 사용하여 수집하였고, Data에 필요한 문장은 Harvard Sentence라는 영어문장을 사용하였다. 또한, 모델 구현에 필요한 기초지식은 Pytorch와 Mnist Dataset을 이용한 예제를 통해 전반적인 구조를 학습하였고, Blender를 통해 csv파일을 이용하여 facial animation을 확인하는 방법을 익힌 후 model을 제작하였다. Model은 크게 세 단계로 나누어 구현하였다. Data Preprocessing 단계에서는 필요없는 Blendshape data를 제거하고 audio data를 Spectrogram으로 변형시켜주었고, Encoding단계에서는 DeepSpeech의 Simple Encoder를 사용하여 구현하였으며, Decoding단계에서는 다양한 layer를 추가한 형태의 Decoder를 직접 제작하였다. 이과제연구의 결과물이 애니메이션이다보니, 그 결과를 이 보고서에 첨부할 수는 없지만 그래도 그럴싸한 모델을 구현하였다. 하지만, 이것을 상용화하기에는 그 결과물이 많이 부족해보이는 것은 사실이다. 

 구현하는 과정에서 보완되어야할 점에 대해 생각해보았다. 먼저 Dataset의 Quality의 문제다. Dataset을 수집하는 과정에서, 방음이 되지않는 공간에서 smartphone을 이용하여 녹음하고, LiveLink Face App만을 이용해 BlendShape 수치를 측정하다보니 수집한 dataset 자체에 noise가 많을 수 밖에 없다고 느껴졌다. 최대한 그러한 요소를 차단하고 dataset을 제작하였다고 생각했지만, 구현을 진행하는 과정에서 더욱 크게 느껴졌고, 다시 Dataset을 제작할까 고민도 하였지만, 시간적으로 여유도 없고 무리라고 판단하여 그대로 진행하였다. 이것이 model의 완성도에 큰 영향을 미쳤을 것이라 생각한다. 더 전문적인 장비와 방음이 되는 공간에서 더 체계적으로 Dataset을 수집하여 학습시킨다면 더 완성도 있는 model이 만들어졌을 것이라 생각된다. 두 번째는 모델 구현 방법의 문제다. 더 높은 완성도의 모델을 만들기 위해서 더 다양한 방법으로 시도하여 수집한 dataset의 형태에 맞는 모델을 제작하면 좋았겠지만, 시간상의 문제로 이러한 방법으로 밖에 해결할 수 없었다는 점이 아쉽게 느껴졌다. 이 연구가 더 체계적으로 진행되고 보완된다면, Character Face Animation이 사용되는 상당히 다양한 분야에서 유용하게 사용될 기술이라고 생각한다. 이번 과제연구의 결과는 크게 유의미한 결과는 얻지 못하고 종료되었지만, 기회가 된다면 이 model을 더욱 발전시켜 제작해보고싶다. 특히 이번에는 data 수집을 영문만 하였는데, 영어뿐만아니라 한국어나 다양한 언어로 확장시켜 제작하면 기술이 더 유용하게 사용될 수 있을 것이라 생각하였다.

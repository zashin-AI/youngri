# gpt-2/src/interactive_conditional_samples.py

# Unconditional sample generation은 랜덤하게 문장을 생성하고
# 반대로 Conditional sample generation은 인터랙티브하게 동작하는데,
# 파일을 실행하면 'Model prompt'라고 뜨는데,
# 여기서 원하는 문장을 입력하면 그 문장에 이어지는 샘플을 출력해서 보여준다.

# GPT를 실행할 때 사용하는 함수로 텍스트를 받아서 모든 모델이 돌아가게 한 듯



#!/usr/bin/env python3

import fire
# fire 패키지는 Python에서의 모든 객체를 command line interface로 만들어 준다.
    # 명령 줄 인터페이스(CLI, Command line interface) 또는 명령어 인터페이스는 텍스트 터미널을 통해 사용자와 컴퓨터가 상호 작용하는 방식을 뜻한다.
    # 즉, 작업 명령은 사용자가 컴퓨터 키보드 등을 통해 문자열의 형태로 입력하며, 컴퓨터로부터의 출력 역시 문자열의 형태로 주어진다.
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder


def interact_model(
    model_name='124M'
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use / 어떤 모델을 쓸지 스트링으로 입력해라
    :seed=None : Integer seed for random number generators, fix seed to reproduce / 난수를 생성할 정수시드를 설정, 다시 만들려면 시드를 고쳐라
     results    / 결과 부분
    :nsamples=1 : Number of samples to return total / 반환되는 샘플의 수
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    / 배치값 숫자(오직 속도와 메모리에만 영향), nsamples에서 나눈 값이어야 한다
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters   
     / 생성되는 문장의 토큰 수 디폴트일 경우 모델 하이퍼파라미터에 의해 결정된다.
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
     / float 형태로 입력하는 매개 변수.
     볼츠만 머신에 들어가는 듯(볼츠만 머신: 대규모 병렬처리를 이용하는 강력한 계산 장치)
     낮으면 랜덤한 정도가 낮아지고 0에 가까워지면 모델이 경직되고 반복적이게 된다. 온도가 높을수록 더 많은 무작위가 발생한다.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     / 다양성을 제어하는 값. 1일 경우에는 각 단계에서 1개의 단어만 고려가 된다.(확률이 높은 순서대로)
      if top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
     40이라면 40개의 단어가 고려가 된다. 0일때는 제한이 없다는 특수 설정이다.
     일반적으로 40이 좋은 값이다.
     top_p = 1 / 일정 확률값 이상인 단어에 대해 필터링하는 매개 변수. 퍼센트를 뜻한다. 0일때는 필터링하지 않는다. 
     if top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
     / 사용할 하위폴더 모델을 포함하는 상위 폴더 경로, 즉 <model_name> 폴더 포함
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    # 입력받은 경로안의 "~"를 현재 사용자 디렉토리의 절대경로로 대체한다.
    
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    # assert 문은 프로그램의 내부 점검이다. 표현식이 참이 아니면 AssertionError 예외가 발생한다.
    # 배치사이즈가 정해지지 않았을 때 디폴트 값은 1으로 정한다.
    # 그런데 nsamples % batch_size의 값이 0이 아니라면 AssertionError를 반환한다.
    # 즉 nsamples의 값은 정수여야 한다.

    enc = encoder.get_encoder(model_name, models_dir)
    # encoder.py 파일에 저장된 Encoder 함수를 반환함

    hparams = model.default_hparams()
    # def default_hparams():
    #     return HParams(
    #     n_vocab=0,    # 단어 수
    #     n_ctx=1024,   # 문맥?
    #     n_embd=768,   # 임베딩 수
    #     n_head=12,    # 헤드 수?
    #     n_layer=12,   # 레이어 수
    # )
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    # 레이어 / 헤드 수 등과 같은 몇 가지 하이퍼 매개 변수가 포함된
    # hparams.json(하이퍼 파라미터 파일)을 열어 파싱한 뒤 하이퍼파라미터 값을 재정의한다.

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    # length: 생성되는 문장의 토큰 수 디폴트일 경우 모델 하이퍼파라미터에 의해 결정된다.
    # 정해져있지 않다면 hparams.n_ctx // 2 값으로 지정이 되는 것이 디폴트
    # 정해졌는데 hparams.n_ctx 보다 길다면 윈도우 사이즈보다 긴 샘플을 얻을 수 없음을 안내

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)
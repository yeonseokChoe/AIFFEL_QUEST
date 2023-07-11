#버전       V_0.01 
#작성자     최연석
#분류       subQ1
#목표       MSE 손실함수값 3000 이하를 달성
#수정사항   1st edit write(2023.07.10)


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#8) 하이퍼 파라미터인 학습률 설정하기
LEARNING_RATE = 0.1


def load_data():    
    #(1) 데이터 가져오기
    diabetes=load_diabetes()
    X = diabetes.data
    y = diabetes.target
    print("df_X shape : " + str(X.shape))
    print("df_y shape : " + str(y.shape))

    #(2) 모델에 입력할 데이터 X 준비하기
    print("df_X type : " + str(type(X)))
    #(3) 모델에 예측할 데이터 y 준비하기
    print("df_y type : " + str(type(y)))
    
    #타입 확인
    if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)) :
        raise

    return X, y


#(5) 모델 준비하기
def model(X, W, b):
    predictions = 0
    for i in range(X.shape[1]):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions


#(6) 손실함수 loss 정의하기
def mse(a, b):
    mse_ = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse_
    

def loss(X, W, b, y):
    predictions = model(X, W, b)
    return mse(predictions, y)


#(7) 기울기를 구하는 gradient 함수 구현하기
def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db


#모델학습
def fit_model(X_train, y_train, losses, W, b, l_count = 1, step_iter = 1000):
    if l_count <= 0 : 
        l_count = 1

    for i in range(1, l_count):
        dW, db = gradient(X_train, W, b, y_train)
        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db
        L = loss(X_train, W, b, y_train)
        losses.append(L)

        if i % step_iter == 0:
            print('Iteration %d : Loss %0.4f' % (i, L))

    return W, b


#시각화
def show_plt(X_test, y_test, prediction):
    plt.scatter(X_test[:, 0], y_test)
    plt.scatter(X_test[:, 0], prediction)
    plt.show()


def main():
    try:
        #데이터 load       
        df_X, df_y = load_data()

    except:
        print("Error : failed to load diabetes data")
        return 1
    
    #(4) train 데이터와 test 데이터로 분리하기
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=50)

    ## init losses, W(랜덤값), b(랜덤값)
    losses = []
    W = np.random.rand(X_train.shape[1])
    b = np.random.rand()

    #(9) 모델 학습하기
    W, b = fit_model(X_train, y_train, losses, W, b, 20001)

    #(10) test 데이터에 대한 성능 확인하기
    prediction = model(X_test, W, b)
    mse_ = loss(X_test, W, b, y_test)
    print("test mse : "+str(mse_))

    #(11) 정답 데이터와 예측한 데이터 시각화하기
    show_plt(X_test, y_test, prediction)

    print("program exit")
    return 0


if __name__ == "__main__":
	main()

    

                                  

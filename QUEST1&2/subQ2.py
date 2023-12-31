#버전       V_0.01 
#작성자     최연석
#분류       subQ2
#목표       RMSE 값 150 이하를 달성
#수정사항   1st edit write(2023.07.10)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#FILE_PATH = "./" #local
FILE_PATH = "~/data/data/bike-sharing-demand/" #lms
FILE_NAME = "train.csv"


#(1) 데이터 가져오기
def read_file(file_path, file_name):
    return pd.read_csv(file_path + file_name)


#(2) datetime 컬럼을 datetime 자료형으로 변환하고 연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성하기
def extend_datetime(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    print("datetime 타입 확인 : "+str(df.dtypes["datetime"]))

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second

    return df


#(3) year, month, day, hour, minute, second 데이터 개수 시각화하기
def show_sns_countplot(df):
    sns.set(rc={'figure.figsize':(20,8)})
    __, axes = plt.subplots(2, 3)
    sns.countplot(data=df, x=df["year"], ax=axes[0,0])
    sns.countplot(data=df, x=df["month"], ax=axes[0,1])
    sns.countplot(data=df, x=df["day"], ax=axes[0,2])
    sns.countplot(data=df, x=df["hour"], ax=axes[1,0])
    sns.countplot(data=df, x=df["minute"], ax=axes[1,1])
    sns.countplot(data=df, x=df["second"], ax=axes[1,2])
    plt.show()


def show_sns_heatmap(df):
    sns.set(rc={'figure.figsize':(20,8)})
    sns.heatmap(data = df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
    plt.show()

#(7) x축은 temp 또는 humidity로, y축은 count로 예측 결과 시각화하기
def show_plt_scatter(X_test, y_test, predictions):
    plt.scatter(X_test["temp"], y_test, label="true")
    plt.scatter(X_test["temp"], predictions, label="pred")
    plt.legend()
    plt.show()


def main():
    try:
        #데이터 load       
        df = read_file(FILE_PATH, FILE_NAME)

        #데이터 컬럼 확인
        chk_colums = set(["datetime", "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"])
        if not set(df.columns).issubset(chk_colums):
             raise
             
    except:
        print("Error : failed to read train data")
        return 1
    
    #datetime 연, 월, 일, 시, 분, 초 확장
    df = extend_datetime(df)
    
    #count 시각화(연, 월, 일, 시, 분, 초)
    show_sns_countplot(df)

    #히트맵 출력
    show_sns_heatmap(df) #registered casual hour temp atemp / humidity 음의상관관계

    #(4) X, y 컬럼 선택 및 train/test 데이터 분리
    #df_X = df[["registered", "casual", "hour", "temp", "atemp", "humidity"]] # mse -e26
    #df_X = df[["registered", "casual", "hour", "temp", "atemp"]] # mse -e27
    df_X = df[["hour", "temp", "atemp", "humidity"]] # remove registered, casual
    df_y = df["count"]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=50)
    
    #(5) LinearRegression 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(model.score(X_train,y_train))

    #(6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
    predictions = model.predict(X_test)
    print(predictions)

    mse = mean_squared_error(y_test, predictions, squared=True)
    print("mse : " + str(mse))

    rmse = mean_squared_error(y_test, predictions, squared=False)
    print("rmse : " + str(rmse))

    #temp / count 산점도 예측 결과 시각화
    show_plt_scatter(X_test, y_test, predictions)

    print("program exit")
    return 0


if __name__ == "__main__":
	main()

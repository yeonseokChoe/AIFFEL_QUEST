#버전       V_0.02 
#작성자     최연석
#분류       subQ3
#목표1      작성한 노트북을 캐글에 제출했다.
#목표2      처리, 학습과정 및 결과에 대한 설명이 시각화를 포함하여 전처리, 학습, 최적화 진행 과정이 체계적
#목표3      피처 엔지니어링과 하이퍼 파라미터 튜닝 등의 최적화 기법을 통해 Private score 기준 110000 이하의 점수      
#수정사항   write 1st edit(2023.07.11)
#          change model - xgboost(2023.07.12)


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import explained_variance_score

#FILE_PATH = "./" #local
FILE_PATH = "~/data/data/" #lms
FILE_NAME_TRAIN = "train.csv"
FILE_NAME_TEST = "test.csv"


#데이터 가져오기
def read_file(file_path, file_name_train, file_name_test):
    return pd.read_csv(file_path + file_name_train), pd.read_csv(file_path + file_name_test)


#데이터 저장
def save_file(file_path, file_name, df):
    print(df)
    df.to_csv(file_path+file_name, columns = ["id", "price"], index = False)


#분포 확인
def show_sns_distribution(df):
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(df['price'])
    plt.show()


#price log변환
def convert_log_data(df):
    df['price'] = np.log1p(df['price'])
    return df


#price exp변환(log1p 역함수)
def convert_exp_data(arr):
    arr = np.exp(arr) - 1
    return arr


#히트맵 출력
def show_sns_heatmap(df):
    sns.set(rc={'figure.figsize':(20,8)})
    sns.heatmap(data = df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
    plt.show()


#temp / count 산점도 예측 결과 시각화
def show_plt_scatter(X_test, y_test, predictions):
    plt.scatter(X_test["grade"], y_test, label="true")
    plt.scatter(X_test["grade"], predictions, label="pred")
    plt.legend()
    plt.show()


def main():
    try:
        #데이터 load       
        df_train, df_test = read_file(FILE_PATH, FILE_NAME_TRAIN, FILE_NAME_TEST)

        #데이터 컬럼 확인
        chk_colums = set(['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                            'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                            'sqft_living15', 'sqft_lot15'])

        if (not set(chk_colums).issubset(df_test.columns)):
                raise
        
        chk_colums.add('price')

        if (not set(chk_colums).issubset(df_train.columns)):
                raise
             
    except:
        print("Error : failed to read data")
        return 1

    #데이터 정보 확인
    print(df_train.info())
    print(df_test.info())
    print(df_train.head(5))

    #정규화
    #분포 확인
    show_sns_distribution(df_train)

    #price log변환
    df_train = convert_log_data(df_train)

    show_sns_distribution(df_train)

    #date 타입 변환 object -> datetime
    df_train["date"] = pd.to_datetime(df_train["date"])
    print(df_train.info())

    #히트맵 출력
    show_sns_heatmap(df_train) 

    #불필요한 데이터 제거
    del df_train["date"]
    del df_train["id"]
    print(df_train.info())

    #이상치 제거

    #train X, y 컬럼 선택 및 train 데이터 분리
    print(df_train.columns)
    df_X = df_train[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'grade', 'sqft_above',
                    'sqft_basement', 'lat', 'sqft_living15']]
    df_y = df_train["price"]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=50)
    
    # 단일 회귀 모델 테스트용 
    # #LinearRegression 모델 학습
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # print("accuracy : " + str(model.score(X_train,y_train)))

    # xgb 부스팅 회귀 모델 생성
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

    model.fit(X_train,y_train)

    xgb.plot_importance(model)

    #학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
    predictions = model.predict(X_test)
    print(predictions)

    # 단일 회귀 모델 테스트용
    # mse = mean_squared_error(y_test, predictions, squared=True)
    # print("mse : " + str(mse))
    # rmse = mean_squared_error(y_test, predictions, squared=False)
    # print("rmse : " + str(rmse))

    r_sq = model.score(X_train, y_train)
    print(r_sq)
    print(explained_variance_score(predictions,y_test))

    #temp / count 산점도 예측 결과 시각화
    show_plt_scatter(X_test, y_test, predictions)

    #test 전처리
    df_test_X = df_test[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'grade', 'sqft_above',
                    'sqft_basement', 'lat', 'sqft_living15']]
    
    print(df_test.info())

    #test predictions
    test_predictions = model.predict(df_test_X)
    test_predictions = convert_exp_data(test_predictions)
    print(test_predictions)

    #test predictions 저장
    df_test["price"] = test_predictions
    print(df_test)

    #파일 저장
    try:
        save_file(FILE_PATH, "sunmission.csv", pd.DataFrame(df_test))
    except:
        print("Error : failed to write data")
        return 1
    
    print("program exit")
    return 0


if __name__ == "__main__":
	main()

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def load_data(path_csv):
    try:
        data = pd.read_csv(path_csv)
        return data
    except Exception as e:
        print(e)
def display_data(data):
    #thông tin về dữ liệu
    print(data.info())
    print(data.describe())

    #Trực quan bản đồ giá nhà dựa trên kinh độ,vĩ độ
    plt.figure(figsize=(10,7))
    x = data["longitude"].values
    y = data["latitude"].values
    scatter_plt = plt.scatter(x,y,s=data["population"]/100
                ,c=data["median_house_value"],cmap="jet",alpha=0.2,)
    plt.colorbar(scatter_plt,label ="Giá nhà trung bình (USD)")
    plt.gca().set_facecolor('black')
    plt.xlabel("kinh độ")
    plt.ylabel("vĩ độ")
    plt.title("Bản đồ giá nhà Califonia")
    plt.show()

    #Ma trận correlation để thấy sự tương quan giữa các đặc trưng ảnh hưởng đến giá nhà
    corr_matrix = data.corr(numeric_only=True).sort_values(by="median_house_value",ascending=False)
    print(corr_matrix)
def split_data(data):
    #tạo income_category để phục vụ cho stratify sampling
    data["income_category"] = pd.cut(data["median_income"].values,bins=[0,1.5,3,4.5,6,np.inf],
                                     labels=["Nghèo","Trung bình","Khá","Giàu","Siêu giàu"])
    data["income_category"].hist()
    # plt.show()
    #tạo một đối tượng split để chia thành 2 tập train và test với tỉ lệ 80% và 20%
    #tỉ lệ này được giữ đúng theo từng nhóm trong income_category
    split_data = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    strat_test_set = pd.DataFrame([])
    strat_train_set = pd.DataFrame([])
    for train_set,test_set in split_data.split(data,data["income_category"]):
        strat_train_set = data.loc[train_set]
        strat_test_set = data.loc[test_set]
    #xoá đi income_category khi hoàn thành việc phân chia
    strat_test_set.drop("income_category",axis=1,inplace=True)
    strat_train_set.drop("income_category",axis=1,inplace=True)
    data.drop("income_category",axis=1,inplace=True)
    return strat_train_set,strat_test_set
def preprocessing_data(train_set):
    #tách dữ liệu gốc
    X_train = train_set.drop("median_house_value",axis=1)
    y_train = train_set["median_house_value"].copy()

    #tạo pipeline
    num_attr = list(X_train.select_dtypes(include="number"))
    category_attr = list(X_train.select_dtypes(include=["object","str","category"]))

    num_pipeline = Pipeline([("imputer",SimpleImputer(strategy="median"))
                                ,("std_scaler",StandardScaler())])
    category_pipeline = Pipeline([("imputer",SimpleImputer(strategy="most_frequent"))
                                  ,("onehot",OneHotEncoder())])
    full_pipeline = ColumnTransformer([("num_pipeline",num_pipeline,num_attr),
                                       ("category_pipeline",category_pipeline,category_attr)])
    #áp dụng pipeline vào tập train
    X_train_prepared = full_pipeline.fit_transform(X_train)
    return X_train_prepared,y_train,full_pipeline

def train_and_tune_model(X_train_prepared, y_train):
    #tạo 3 ứng viên model
    linear_model = LinearRegression()
    tree_reg = DecisionTreeRegressor(random_state=42)
    forest_reg = RandomForestRegressor(random_state=42)
    #chọn model tốt nhất
    list_model = [linear_model,tree_reg,forest_reg]
    for model in list_model:
        scores = cross_val_score(model,X_train_prepared,y_train,cv=10
                                 ,scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(-scores)

        print(f"Model: {model.__class__.__name__}")
        print(f"sai số trung bình: {rmse_scores.mean():.0f} USD")
    param_grid = [{"n_estimators":[3,10,30],"max_features":[2,4,6,8]}]
    grid_search = GridSearchCV(estimator=forest_reg,param_grid=param_grid,cv=5,n_jobs=-1,
                               scoring="neg_mean_squared_error",return_train_score=True)
    grid_search.fit(X_train_prepared,y_train)
    return grid_search.best_estimator_
def evaluate_and_save_model(best_model,test_set,full_pipeline):
    X_test = test_set.drop("median_house_value",axis=1)
    y_test = test_set["median_house_value"].copy()
    #tiến hành dự đoán trên tập test
    X_test_prepared = full_pipeline.transform(X_test)
    predictions = best_model.predict(X_test_prepared)

    #sai số của tập label sau khi được dự đoán so với tập giá trị chính xác
    mse = mean_squared_error(y_test,predictions)
    rmse = np.sqrt(mse)
    print(f"sai số RMSE: {rmse:.0f} USD")
    #lưu lại model
    joblib.dump(best_model,"california_housing_model.pkl")


if __name__ == "__main__":
    data = load_data(r"../datasets/housing.csv")
    # Tạm thời comment hàm hiển thị biểu đồ lại để code chạy một mạch không bị khựng lại
    display_data(data)
    # 1. Chia dữ liệu
    print("Đang chia dữ liệu...")
    train_set, test_set = split_data(data)
    # 2. Tiền xử lý
    print("Đang tiền xử lý dữ liệu...")
    X_train_prepared, y_train, full_pipeline = preprocessing_data(train_set)
    # 3. Thi đấu và Ép xung (Bước này sẽ chạy hơi lâu, mất khoảng 1-2 phút)
    print("Đang huấn luyện và tìm kiếm mô hình tốt nhất (Vui lòng đợi vài phút)...")
    best_model = train_and_tune_model(X_train_prepared, y_train)
    # 4. Đánh giá cuối cùng và Xuất xưởng
    print("Đang chấm điểm trên tập Test...")
    evaluate_and_save_model(best_model, test_set, full_pipeline)
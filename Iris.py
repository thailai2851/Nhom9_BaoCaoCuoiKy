# Import thư viện
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load dữ liệu
iris = load_iris()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Tạo mô hình Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Đánh giá hiệu suất của mô hình bằng cách sử dụng tập kiểm tra
accuracy = gnb.score(X_test, y_test)
print("Độ chính xác của mô hình là:", accuracy)


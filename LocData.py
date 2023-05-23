import nltk
from collections import defaultdict
import string

#Natural Language Toolkit (nltk)

data = [("Xin chào, tôi làm việc cho công ty ABC. Trân trọng!", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của mình không?. Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng lớn", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm", "Spam"),
        ("Xin chào, tôi là thành viên của công ty. Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen của bạn. Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 50%", "Spam"),
        ("Giảm giá khủng mừng sinh nhật", "Spam")
        ,("Xin chào, tôi là đồng nghiệp của bạn. Trân trọng!", "Không spam"),
        ("Xin chào, tôi làm việc cho công ty ABCD. Trân trọng!.", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của bản thân không?. Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng lớn.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt ngày hôm nay.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng lớn rồi", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt trong ngày", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm duy nhất hôm nay", "Spam"),
        ("Xin chào, tôi là thành viên của công ty của bạn . Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen mà bạn của bạn biết. Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi rồi đây", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 70%", "Spam"),
        ("Giảm giá khủng mừng sinh nhật lazada", "Spam")
        ,("Xin chào, tôi là đồng nghiệp của bạn nè. Trân trọng!", "Không spam"),
        ("Xin chào, tôi làm việc cho công ty ABCDEF.. Trân trọng!", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của mình không? bời vì nó tốt. Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng cây kẹo.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt nhất hôm nay.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng cực lớn", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt nhất của lazada", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm của chúng tôi hôm nay", "Spam"),
        ("Xin chào, tôi là thành viên của công ty mà bạn đang làm . Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen của bạn tại cty ACB. Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi hãy truy cập vào link acb để nhận", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 80%", "Spam"),
        ("Giảm giá khủng mừng sinh nhật shoppe", "Spam")
        ,("Xin chào, tôi là đồng nghiệp của bạn ở công ty F", "Không spam"),
        ("Xin chào, tôi làm việc cho công ty ABC tôi nghĩ bạn quen tôi.", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của mình không? Nó sẽ hữu ích cho cuộc sống . Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng chiếc xe AB.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt của cty FA chúng tôi.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng lớn rồi đi nhận thôi nào", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt truy cập vào website để nhận", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm quần áo nam", "Spam"),
        ("Xin chào, tôi là thành viên của công ty FJD. Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen của bạn tại công ty DFH. Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi rồi này truy cập vào website để nhận", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 60%", "Spam"),
        ("Giảm giá khủng mừng sinh nhật 2 tuổi lazada", "Spam")
        ,("Xin chào, tôi là đồng nghiệp của bạn tại công ty FHKG. Trân trọng!", "Không spam"),
        ("Xin chào, tôi làm việc cho công ty ABCDF là cộng sự của bạn. Trân trọng!.", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của mình không? Tôi nghĩ nó thích hợp với bạn. Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng lớn với số tiền lên đến 1000000.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt của chúng tôi với số tiền 200000.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng lớn hãy truy cập vào website để nhận thưởng", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt của chúng tôi với chiếc xe SH", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm sản phẩm xe AB phiên bản abs", "Spam"),
        ("Xin chào, tôi là thành viên của công ty này nè AFG. Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen của bạn tại khu vực Bình Định. Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi rồi hãy đi nhận nó", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 40%", "Spam"),
        ("Giảm giá khủng mừng sinh nhật 3 tuổi của shopee", "Spam")
        ,("Xin chào, tôi là đồng nghiệp của bạn ở Bình Định", "Không spam"),
        ("Xin chào, tôi làm việc cho công ty ABCFHG. Trân trọng!.", "Không spam"),
        ("Xin chào, bạn có muốn tăng cường kỹ năng lập trình của mình không?có tôi muốn học nó . Trân trọng!", "Không spam"), 
        ("Chúc mừng! Bạn đã trúng thưởng lớn nhất từ trước đến nay.", "Spam"),    
        ("Đăng ký ngay để nhận ưu đãi đặc biệt của tuần lễ vàng.", "Spam"),
        ("Chúc mừng bạn đã trúng thưởng lớn của tuần lễ vàng", "Spam"),
        ("Đăng ký ngay để nhận giảm giá và ưu đãi đặc biệt của tuần lễ vàng", "Spam"),
        ("Bạn đã trúng thưởng và giảm giá sản phẩm với số tiền lên đến 200000", "Spam"),
        ("Xin chào, tôi là thành viên của công ty HFIE. Trân trọng!","Không spam"),
        ("Trân trọng xin chào bạn, tôi là người quen của bạn ở Quy nhơn nè . Trân trọng!", "Không spam"),
        ("Bạn đã nhận được giảm giá bên tôi với số tiền cực khủng", "Spam"),
        ("Giảm giá đây chúc mừng bạn nhận được giảm giá 50% với tất cả sản phẩm", "Spam"),
        ("Giảm giá khủng mừng sinh nhật 6 tuổi của lazada", "Spam")]

def train(data):
    # Khởi tạo bộ đếm cho từng từ trong từng nhãn
    count = defaultdict(lambda: defaultdict(int))
    # Khởi tạo bộ đếm cho mỗi nhãn
    label_count = defaultdict(int)
    # Tách các từ trong từng email và đếm số lần xuất hiện của từng từ trong từng nhãn
    for email, label in data:
        for word in nltk.word_tokenize(email.translate(str.maketrans('', '', string.punctuation))):
            count[word.lower()][label.lower()] += 1
        label_count[label.lower()] += 1
    # Tính toán xác suất của từng từ trong từng nhãn
    probabilities = defaultdict(lambda: defaultdict(float))
    for word, label_count in count.items():
        total = sum(label_count.values())
        for label, count in label_count.items():
            probabilities[word][label] = count / total
    # Tính toán xác suất của mỗi nhãn
    total = sum(label_count.values())
    label_probabilities = {label: count / total for label, count in label_count.items()}
    return probabilities, label_probabilities

def predict(email, probabilities, label_probabilities):
    email_probabilities = defaultdict(float)
    # Tính toán xác suất của email cho từng nhãn
    for label in label_probabilities:
        email_probabilities[label] = label_probabilities[label]
        for word in nltk.word_tokenize(email.translate(str.maketrans('', '', string.punctuation))):
            email_probabilities[label] *= probabilities[word.lower()][label]
    # Chuẩn hóa xác suất để tổng bằng 1
    total = sum(email_probabilities.values())
    if total == 0:
        num_labels = len(label_probabilities)
        for label in label_probabilities:
            email_probabilities[label] = 1/num_labels
    else:
        for label in email_probabilities:
            email_probabilities[label] /= total
    return email_probabilities

def classify(email, probabilities, label_probabilities):
    # Tính toán xác suất của email cho từng nhãn
    email_probabilities = predict(email, probabilities, label_probabilities)
    # Chọn nhãn có xác suất cao nhất
    return max(email_probabilities, key=email_probabilities.get)

# Huấn luyện mô hình
probabilities, label_probabilities = train(data)

# Dự đoán nhãn cho email mới
new_email = "Trân trọng xin chào bạn, tôi là người quen của bạn tại khu vực Bình Định. Trân trọng!"
predicted_label = classify(new_email, probabilities, label_probabilities)

print(f"Nhãn của email '{new_email}' được dự đoán là: {predicted_label}")
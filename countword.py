import random
import matplotlib.pyplot as plt
import sqlite3
import string
import operator
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from underthesea.transformer.tfidf import TfidfVectorizer

conn = sqlite3.connect("VnExpress.db")
col = conn.execute("SELECT * FROM Vnexpress").fetchall()


classify = conn.execute("SELECT Categories, COUNT(*) from %s group by Categories" % "Vnexpress")
categories = []
number_of_paper = []
for row in classify:
    categories.append(row[0])
    number_of_paper.append(row[1])


def xulytudung(comment):
    stop_word = [] #tạo list để chứa stopword
    #mở file stopword
    with open("vietnamese-stopwords-dash.txt", encoding="utf-8") as f:
        text = f.read()
        for word in text.split():
            stop_word.append(word) #thêm từng stopword vào list
        f.close()
    remove = string.punctuation #string.punctuation chứa các ký tự !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    remove = remove.replace("|", "")  # không loại bỏ ký tự |, kí tự dùng để phân biệt tiêu đề, trích dẫn, nội dung
    punc = list(remove)
    stop_word = stop_word + punc #list top word lúc này có chứa các ký tự đặc biệt
    str = '' #tạo một chuổi để lưu lại text đã loại stopword
    for word in comment.split(" "): #tách câu thành từng từ dựa vào khoảng trắng
        if (word not in stop_word):
            if ("_" in word) or ("|" in word) or (word.isalpha() == True) or (word.isdigit() == True): #nếu là dấu gạch dưới, dấu gạch đứng, chữ cái hay số thì sẽ cộng lại thành chuổi
                #không bỏ dấu gạch dưới vì dấu này đang dùng để liên kết cụm từ với nhau
                #dấu gạch đứng để phân biệt nội dung trong câu
                str = str + word + " "
    return str
#hàm tìm kiếm từ xuất hiện nhiều nhất trong từng thể loại
common_istrue = True

def common_word():
    print("Từ xuất hiện nhiều nhất trong từng thể loại là: ")
    print("Thể loại".ljust(18) + "|".ljust(5) + "Từ khóa".ljust(15) + "|".ljust(5) + "Số lần xuất hiện".ljust(18) + "|".ljust(5) + "Tỉ lệ % trên tổng số từ".ljust(26) + "|".ljust(5) + "Tổng số từ")
    print("------------------|-------------------|----------------------|------------------------------|----------------------")
    count = 1
    count_all_word = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        index = 1
        all_text = []
        count_word = 0
        for row in analysis: #mở từng file txt đã xử lý stopword trong từng thể loại
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text = text.replace("\n", " ")
            count_word = count_word + len(text.split(" "))
            all_text.append(str(text)) #loại bỏ xuống dòng và append tất cả các bài báo theo thể vào cùng một list, mỗi thể loại báo sẽ có một list để phục vụ chi việc đếm
            index += 1
            f.close()
        count_all_word.append(count_word)

        # CountVectorizer transform text thành vector. Cụ thể CountVectorizer đã sử dụng như mặc định “phân tích” được gọi WordNGramAnalyzer , có trách nhiệm để chuyển đổi văn bản thành chữ thường, Giọng di chuyển, khai thác dấu hiệu, bộ lọc từ dừng lại, vv ...
        # Cách transform : chúng ta có một mảng các string từ list all_text,
        # bây giờ sẽ tiến hành transform mảng này sao mỗi string sẽ chuyển đổi thành 1 vector có độ dài d (số từ xuất hiện ít nhất 1 lần trong string),
        # giá trị của thành phần thứ i trong vector chính là số lần từ đó xuất hiện trong string.
        # cv.get_feature_names() đã trả lại kết quả là các từ xuất hiện ít nhất 1 lần trong tất cả các string từ list all_text.
        # Còn cv_fit,toarray sẽ trả về số lần xuất xuất hiện của một từ dưới dạng mảng.
        cv = CountVectorizer()
        cv_fit = cv.fit_transform(all_text)
        word_list = cv.get_feature_names();
        count_list = cv_fit.toarray().sum(axis=0) #tính tổng theo cột (asix = 0)
        tops = dict(zip(word_list, count_list)) #tạo một dictionary với key là tên của từ xuất hiện nhiều nhất và value và số lần đếm được
        maxvalue = max(tops.items(), key=operator.itemgetter(1))[0] #lấy ra từ có số lần xuất hiện cao nhất từ values ở cột thứ 1
        listvalue = sorted(tops, key=tops.get, reverse=True)[:10]
        print(str(count).ljust(2) + "." + str(category).ljust(15) + "|".ljust(5) + str(maxvalue).replace("_", " ").ljust(15) + "|".ljust(10) + str(
            max(count_list)).ljust(5) +  " ".ljust(8) + "!".ljust(10) + str(round((max(count_list)/count_all_word[count-1])*100,2)).ljust(4)+"%".ljust(17) + "|".ljust(5) + str(count_all_word[count-1]))

        count = count + 1




def classify():
    phanloai = conn.execute("SELECT Categories, COUNT(*) from %s group by Categories" % "Vnexpress")
    print("Số lượng bài báo theo thể loại: ")
    print("Thể loại".ljust(15) + "|".ljust(5) + "Số lượng bài")
    print("---------------|-----------------")
    categories = []
    number_of_paper = []
    for row in phanloai:
        print(str(row[0]).ljust(15) + "|".ljust(5) + str(row[1]))
        categories.append(row[0])
        number_of_paper.append(row[1])
    total_word = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        all_text = []
        len_text = []
        index = 1
        for row in analysis:
            f = open(f"REMOVED/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text.replace("\n", " ")

            all_text =all_text + text.split(" ")
            index += 1
            f.close()
        len_text.append(int(len(all_text)))
        total_word = total_word + len_text
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
    ax1.bar(categories, number_of_paper, color="g")
    ax2.bar(categories, total_word, color="b")

    ax1.set(xlabel="Số bài báo", ylabel = "Thể loại")
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax2.set(xlabel="Số từ", ylabel="Thể loại")
    ax1.set_title('SỐ LƯỢNG BÀI BÁO THEO TỪNG THỂ LOẠI', loc='center')
    ax2.set_title('SỐ LƯỢNG TỪ THEO TỪNG THỂ LOẠI', loc='center')
    plt.show()

def seach():
    tex = input("Nhập nội dung cần tìm kiếm: ")
    all_news = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        searchtext = []
        index = 1
        for row in analysis:
            f = open(f"REMOVED/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text.replace("\n", " ")
            searchtext.append(text + "\n" + str(category))
            index += 1
            f.close()
        all_news = all_news + searchtext
    vector = TfidfVectorizer() #tự động chuyển đổi một bộ dữ liệu thô sang dang ma trận và sửa dụng được các tính năng của tf-idf
    X = vector.fit_transform(all_news) #chuyển đổi dữ liệu sang ma trận để chuẩn bị tính toán
    # print(X)
    te = str(tex) #gán nội dung tìm kiếm
    te = te.lower() #chuyển nội dung tìm kiếm về dạng chữ viết thường
    te = word_tokenize(te, format='text') #kết nối các cụm từ có nghĩa nếu có xuất hiện trong từ khóa tìm kiếm
    te=xulytudung(te) #loại bỏ các stopword nếu có xuất hiện
    print(te)
    te = vector.transform([te]) #tính toán trọng số xuất hiện của từ nếu từ đó có xuất hiện trong toàn bộ text, nếu không có sẽ ko in gì cả
    print(te)
    m = str(te) # xử lí các kết quả trọng số
    # print(len(m))

    print("STT".ljust(5) + "|" + "Thể loại".ljust(20) + "|" + "Tiêu đề".ljust(80) + "|" + "Trích dẫn".ljust(120) + "|" + "Nội dung")
    print(
        "-----|--------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------")
    if len(m) > 1: #nếu chiều dài của str m  > 1 hay nói cách khác là có tồn tại ít nhất một trọng số (một từ khóa được tìm kiếm)
        re = cosine_similarity(X, te) #tính cosine giữa X và te là một ma trận có 2 cột và X dòng
        Li = []
        for i in range(len(re)):
            Li.append(re[i][0]) #thêm các giá trị của re vào list
        index = 1
        for i in np.argsort(Li)[-50:][::-1]: #sắp xếp các giá trị re trong e[elist Li theo thứ tự giảm dần về giá trị và lấy ra 50 giá trị cao nhất để biểu diễn kết quả tìm kiếm
            print(Li[i])
            r = all_news[i].split("\n")
            print(str(index).ljust(5) + "|" + str(r[3]).replace("_", " ").ljust(20) + "|" + str(r[0]).replace("_", " ").ljust(80) + "|" + str(r[1]).replace("_", " ").ljust(120) + "|" + str(r[2]).replace("_", " "))
            print("-----|--------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------")
            index = index + 1
    else:
        print('Không tìm thấy bài báo cho từ khóa :', tex)

def author():
    print("\n---------Thông tin tác giả---------")
    print('Tên sinh viên: Nguyễn Minh Thông - Nguyễn Thị Bích Liên')
    print('MSSV: 1824801040107 - 1824801040112')
    print('Lớp: D18HT02')
    print('Số điện thoại: 0907944628')
    print('Donate tại số tk BIDV: 65010002790040')

def count_max_len_word():
    print("Bài báo dài nhất có độ dài:")
    print("Thể loại".ljust(18) + "|".ljust(5) + "Số từ".ljust(15))
    print("------------------|-------------------")
    count = 1
    count_all_word = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        index = 1
        all_text = []
        count_word = 0
        for row in analysis: #mở từng file txt đã xử lý stopword trong từng thể loại
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text = text.replace("\n", " ")
            lenword = len(text.split(" "))
            if (lenword > count_word):
                count_word = lenword
             #loại bỏ xuống dòng và append tất cả các bài báo theo thể vào cùng một list, mỗi thể loại báo sẽ có một list để phục vụ chi việc đếm
            index += 1
            f.close()


        print(str(count).ljust(2) + "." + str(category).ljust(15) + "|".ljust(5) + str(count_word).replace("_", " ").ljust(15))
        count = count + 1

def count_min_len_word():
    print("Bài báo ngắn nhất có độ dài:")
    print("Thể loại".ljust(18) + "|".ljust(5) + "Số từ".ljust(15))
    print("------------------|-------------------")
    count = 1
    count_all_word = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        index = 1
        all_text = []
        count_word = 10000
        for row in analysis: #mở từng file txt đã xử lý stopword trong từng thể loại
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text = text.replace("\n", " ")
            lenword = len(text.split(" "))
            if (lenword < count_word):
                count_word = lenword
             #loại bỏ xuống dòng và append tất cả các bài báo theo thể vào cùng một list, mỗi thể loại báo sẽ có một list để phục vụ chi việc đếm
            index += 1
            f.close()


        print(str(count).ljust(2) + "." + str(category).ljust(15) + "|".ljust(5) + str(count_word).replace("_", " ").ljust(15))
        count = count + 1


def getfeature():
    print("Những từ có tần xuất hiện cao nhất theo thể loại: ")
    count = 1
    count_all_word = []
    for category in categories:
        analysis = conn.execute("SELECT * from Vnexpress where categories = '%s'" % str(category))
        index = 1
        all_text = []
        count_word = 0
        for row in analysis: #mở từng file txt đã xử lý stopword trong từng thể loại
            f = open(f"CATEGORIES/{str(category)}/{str(index)}.txt", 'r', encoding="utf8")
            text = f.read()
            text = text.replace("\n", " ")
            count_word = count_word + len(text.split(" "))
            all_text.append(str(text)) #loại bỏ xuống dòng và append tất cả các bài báo theo thể vào cùng một list, mỗi thể loại báo sẽ có một list để phục vụ chi việc đếm
            index += 1
            f.close()
        count_all_word.append(count_word)

        # CountVectorizer transform text thành vector. Cụ thể CountVectorizer đã sử dụng như mặc định “phân tích” được gọi WordNGramAnalyzer , có trách nhiệm để chuyển đổi văn bản thành chữ thường, Giọng di chuyển, khai thác dấu hiệu, bộ lọc từ dừng lại, vv ...
        # Cách transform : chúng ta có một mảng các string từ list all_text,
        # bây giờ sẽ tiến hành transform mảng này sao mỗi string sẽ chuyển đổi thành 1 vector có độ dài d (số từ xuất hiện ít nhất 1 lần trong string),
        # giá trị của thành phần thứ i trong vector chính là số lần từ đó xuất hiện trong string.
        # cv.get_feature_names() đã trả lại kết quả là các từ xuất hiện ít nhất 1 lần trong tất cả các string từ list all_text.
        # Còn cv_fit,toarray sẽ trả về số lần xuất xuất hiện của một từ dưới dạng mảng.
        cv = CountVectorizer()
        cv_fit = cv.fit_transform(all_text)
        word_list = cv.get_feature_names();
        count_list = cv_fit.toarray().sum(axis=0) #tính tổng theo cột (asix = 0)
        tops = dict(zip(word_list, count_list)) #tạo một dictionary với key là tên của từ xuất hiện nhiều nhất và value và số lần đếm được
        maxvalue = max(tops.items(), key=operator.itemgetter(1))[0] #lấy ra từ có số lần xuất hiện cao nhất từ values ở cột thứ 1
        listkey = sorted(tops, key=tops.get, reverse=True)[:10]
        print(str(count).ljust(2) + "." + str(category).ljust(15), end=": ")
        print(listkey)
        count = count + 1


count_min_len_word()
print("\n")
count_max_len_word()
print("\n")
common_word()
print("\n")
getfeature()
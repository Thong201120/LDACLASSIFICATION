import gensim
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv', encoding='utf-8')
# Kiểm tra nhãn
print(df.label.unique())
print(df.head())

# trực quan số lượng bài đọc theo nhãn
df[['label', 'data']].groupby(['label']).count().plot(kind='bar')
plt.show()
data = df['data'].values.tolist()
new_data = []


for x in range(len(data)):
    data[x] = data[x].split()
    new_data.append(data[x])


import gensim.corpora as corpora
# Khởi tạo Dictionary
id2word = corpora.Dictionary(new_data)
# Tạo kho văn bản Corpus
texts = new_data
# Tuần suất xuất hiện của tập tài liệu
corpus = [id2word.doc2bow(text) for text in texts]
# Kiểm tra
print(corpus[:1])
data_copus = [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
print(data_copus[1])

"""
Giải thích các tham số trong mô hình
Khởi tạo mô hình lda
Với các thông số:
1. corpus là tập document
2. numtopic: số lượng các chủ đề tiềm ẩn được yêu cầu được trích xuất từ Corpus đào tạo: 10
3. id2word: convert ngược lại từ index (chỉ số) sang từ vựng ta sử dụng dictionary id2word
(Ánh xạ từ ID từ thành từ. Nó được sử dụng để xác định kích thước từ vựng, cũng như để gỡ lỗi và in chủ đề.)
4. Random state: trạng thái ngẫu nhiên 100
5. Update_every: Số lượng tài liệu được lặp lại cho mỗi lần cập nhật. 
   Với: 0 cho học theo lô
        > 1 cho học lặp lại trực tuyến.
6. chunksize: Số lượng tài liệu được sử dụng trong mỗi chunk đào tạo là 200
7. passes: Số lần đi qua kho ngữ liệu corpus trong quá trình đào tạo là 6
8. alpha: 
    - Độ tin cậy ưu tiên về phân phối chủ đề tài liệu, nó có thể là:
        + vô hướng cho một đối xứng trước khi phân phối chủ đề tài liệu,
        + Mảng 1D có độ dài bằng num_topics để biểu thị một người dùng không đối xứng được xác định trước cho mỗi chủ đề.
    - Ngoài ra, các chiến lược lựa chọn trước mặc định có thể được sử dụng bằng cách cung cấp một chuỗi:
        + 'symmetric - Đối xứng': (mặc định) Sử dụng đối xứng cố định trước 1.0 / num_topics,
        + 'Asymmetric - Bất đối xứng': Sử dụng không đối xứng chuẩn hóa cố định trước 1.0 / (topic_index + sqrt (num_topics)),
        + 'Auto': Tìm hiểu trước từ bất đối xứng từ kho ngữ liệu (không khả dụng nếu được phân phối == True).
9. per_word_topics: Nếu True (Đúng), mô hình cũng tính toán một danh sách các chủ đề, được sắp xếp theo thứ tự 
    giảm dần các chủ đề có khả năng xảy ra nhất cho mỗi từ, cùng với các giá trị phi của chúng nhân với độ dài
    của đối tượng
"""
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=200,
                                           passes=6,
                                           alpha='auto',
                                           per_word_topics=True)
lda_model.save("model_lda_100.model")

# Sau khi lưu huấn luyện và lưu model xong, tiếp theo chúng ta sẽ mở lại và sử dụng
from gensim.models.ldamodel import LdaModel
lda = LdaModel.load("model_lda_100.model")
from pprint import pprint

# In các từ khóa ở trong 10 chủ đề
pprint(lda.print_topics())
doc_lda = lda[corpus]

from gensim.models.coherencemodel import CoherenceModel

# Tính toán phức tạp
def format_topics_sentences(ldamodel=lda, corpus=corpus, texts=data):
    # Tạo đầu ra (dataframe)
    sent_topics_df = pd.DataFrame()

    # Nhận chủ đề chính trong mỗi tài liệu
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        # Nhận chủ đề Thống trị, Phần trăm Đóng góp và Từ khóa cho mỗi tài liệu
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic - chủ đề thống trị
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4),
                                                                  topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # sent_topics_df.columns = ['Chủ đề thống trị', 'Phần trăm đóng góp', 'Từ khóa tài liệu']
    # Thêm văn bản gốc vào cuối đầu ra
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda, corpus=corpus, texts=data)

# Định dạng
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# df_dominant_topic.columns = ['STT', 'Chủ đề thống trị', 'Phần trăm đóng góp', 'Từ khóa', 'Văn bản gốc']
df_dominant_topic.to_csv('result.csv.csv', encoding='utf-8')
# Hiển thị
print(df_dominant_topic.head(10))

from gensim.models.coherencemodel import CoherenceModel

# Compute Perplexity
print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda, texts=new_data, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

import pyLDAvis
import matplotlib.pyplot as plt


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
plt.show()
from flask import Flask, render_template, request
import pandas as pd
from flask_paginate import Pagination, get_page_args
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# chỗ này là dữ liệu nè
all_data = pd.read_csv('./data/result.csv', encoding='utf-8', header=1, on_bad_lines='skip', sep=';')
all_data = list(all_data.values)
# chỗ này là bài viết nè
all_post = pd.read_csv('./data/datafinal.csv', encoding='utf-8', on_bad_lines='skip', sep=';')
all_post = list(all_post.values)


def get_data(offset=0, per_page=20):
    return all_data[offset: offset + per_page]


@app.route('/')
def home():
    return render_template('TrangChu.html')


@app.route('/visualize')
def visual():
    return render_template('lda_result.html')


# Thêm trang dữ liệu để tìm kiếm chỗ này
@app.route('/search')
def data():
    return render_template('search.html')


@app.route('/search-result')
def search_result():
    search_text = request.args.get('search-text')
    tokenized_data = [doc[4].replace(",", " ").replace("_", " ").split() for doc in all_post]
    bm25 = BM25Okapi(tokenized_data)
    tokenized_query = search_text.lower().split()
    doc_score = bm25.get_scores(tokenized_query)
    count = sum(x > 0 for x in doc_score)

    result = bm25.get_top_n(tokenized_query, all_post, n=count)

    def get_result(offset=0, per_page=10):
        return result[offset: offset + per_page]

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = count
    pagination_result = get_result(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template("search-result.html", result=pagination_result, search_text=request.args.get('search-text'),
                           pagination=pagination, page=page, per_page=per_page)

@app.route("/detail/<int:news_id>", methods = ["GET"])
def get_news_by_id(news_id):

    row = all_post[news_id-1]

    print(row)
    data = {
        "id": row[0],
        "theloai": row[5],
        "title": row[2],
        "sumary": row[3],
        "content": row[4]
    }
    return render_template("detail.html", data=data)

def search_result():
    search_text = request.args.get('search-text')
    tokenized_data = [doc[5].replace(",", " ").replace("_", " ").split() for doc in all_post]
    bm25 = BM25Okapi(tokenized_data)
    tokenized_query = search_text.lower().split()
    doc_score = bm25.get_scores(tokenized_query)
    count = sum(x > 0 for x in doc_score)

    result = bm25.get_top_n(tokenized_query, all_post, n=count)

    def get_result(offset=0, per_page=10):
        return result[offset: offset + per_page]

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = count
    pagination_result = get_result(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template("search-result.html", result=pagination_result, search_text=request.args.get('search-text'),
                           pagination=pagination, page=page, per_page=per_page)

@app.route('/result')
def table():
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(all_data)
    pagination_data = get_data(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    return render_template('index.html', all_data=pagination_data, pagination=pagination)


@app.route('/post')
def get_post():
    try:
        post_id = request.args.get('id', type=int)
        for post in all_post:
            if post[0] == post_id:
                result = " ".join(post[1].split()).replace('_', ' ')
                return result
        return 'Không tìm thấy bài viết'
    except Exception as ex:
        return 'Đã xảy ra lỗi ' + str(ex)


if __name__ == '__main__':
    app.run(host='0.0.0.0')

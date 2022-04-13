from gensim.models.ldamodel import LdaModel
lda = LdaModel.load("model_lda_100.model")
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

df = pd.read_csv('data.csv', encoding='utf-8')
data = df['data'].values.tolist()
new_data = []
for x in range(len(data)):
    data[x] = data[x].split()
    new_data.append(data[x])

id2word = corpora.Dictionary(new_data)
texts = new_data
corpus = [id2word.doc2bow(text) for text in texts]




if __name__ == '__main__':
    # Compute Perplexity
    # print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    #
    # coherence_model_lda = CoherenceModel(model=lda, texts=new_data, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)
    from pyLDAvis import gensim_models
    import pyLDAvis
    import matplotlib.pyplot as plt
    # Visualize the topics
    vis = gensim_models.prepare(lda, corpus, id2word)
    pyLDAvis.save_html(vis, 'lda_result.html')

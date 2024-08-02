import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word not in stop_words]
        processed_sentences.append(' '.join(words))
    return processed_sentences

def get_most_relevant_sentence(query, sentences):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences + [query])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    most_relevant_index = cosine_similarities.argmax()
    return sentences[most_relevant_index]

def chatbot(user_query, sentences):
    processed_query = preprocess(user_query)[0]
    response = get_most_relevant_sentence(processed_query, sentences)
    return response

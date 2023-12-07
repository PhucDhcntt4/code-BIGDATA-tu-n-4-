from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dữ liệu mẫu - danh sách các văn bản
documents = [
  
"Đây là tài liệu đầu tiên.",
    "Tài liệu này là tài liệu thứ hai.",
    "Và đây là cái thứ ba.",
    "Đây có phải là tài liệu đầu tiên không?"
]

# Tạo vector đặc trưng TF-IDF cho các văn bản
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Tính độ tương tự cosine giữa các văn bản
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# In ma trận độ tương tự cosine
print("Ma trận Cosine :")
print(cosine_similarities)

# In các cặp văn bản có độ tương tự cao nhất
num_documents = len(documents)
for i in range(num_documents):
    for j in range(i + 1, num_documents):
        similarity = cosine_similarities[i, j]
        print(f"Sự tương đồng tài liệu {i + 1} với tài liệu {j + 1}: {similarity:.4f}")

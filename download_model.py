from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-l6-v2")
model.save("local_model")

print("Model was saved locally")

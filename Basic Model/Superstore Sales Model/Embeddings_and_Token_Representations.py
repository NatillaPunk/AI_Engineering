import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# %matplotlib inline

import pickle
with open('datasets/vocab_pickle.pkl', 'rb') as file:
    vocab = pickle.load(file)

print("Vocabulary Size:", len(vocab))
print("Vocabulary Preview:", dict(list(vocab.items())[:20]))




## YOUR SOLUTION HERE ##
class SimpleNNWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SimpleNNWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
vocab_size = len(vocab)
embed_size = 50
hidden_size = 100
output_size = 2

text_classifier_model = SimpleNNWithEmbedding(vocab_size,embed_size,hidden_size,output_size)
state_dict = torch.load("models/text_classifier_nn.pth")
text_classifier_model.load_state_dict(state_dict)
text_classifier_model.eval()

embedding_layer = text_classifier_model.embedding.weight.detach().cpu().numpy()

# Show output
print(f"Embedding shape:", embedding_layer.shape)



## YOUR SOLUTION HERE ##
def token_similarity(token1, token2, vocab, embedding_layer):
    i, j = vocab[token1],vocab[token2]
    sim = cosine_similarity([embedding_layer[i]],[embedding_layer[j]])[0,0]

    return sim

sim1 = token_similarity("school","student",vocab,embedding_layer)
sim2 = token_similarity("school","work",vocab,embedding_layer)
sim3 = token_similarity("school","phone",vocab,embedding_layer)

# Show output - Cosine similarity scores for token pairs
print(f"Cosine Similarity (school, student): {sim1:.5}")
print(f"Cosine Similarity (school, work): {sim2:.5}")
print(f"Cosine Similarity (school, phone): {sim3:.5}")








def plot_embeddings(tokens, vocab, embeddings):
    idxs = [vocab[w] for w in tokens if w in vocab]
    pts = embeddings[idxs]
    plt.figure(figsize=(7,6))
    plt.scatter(pts[:,0], pts[:,1], s=24)
    for (x,y), w in zip(pts, tokens):
        if w in vocab:
            plt.text(x+0.01, y+0.01, w, fontsize=9)
    plt.title("Word Embeddings (t-SNE, 2D)")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.tight_layout()
    plt.show()

## YOUR SOLUTION HERE ##
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embedding_layer)
tokens_to_plot = ["school", "students", "work", "phone", "people", "life", "positive", "negative", "company"]











# Introduction to Neural Network Architectures
# Embeddings and Token Representations
# 27 min
# So far, we have applied RNNs to process sequential data, such as time series. In this exercise, we’ll shift our focus to natural language and explore how we can process words as text sequences into tokens which are then mapped into representations called embeddings.

# For this exercise, we’ll examine a trained language model and break down how it processes text sequences into useful representations that allow it to learn semantic patterns to solve language tasks.

# Tokenization
# Tokenization is the process of breaking down a text into singular units called tokens.

# Popular LLMs used widely today, like ChatGPT and Claude, use a subword-based tokenizer called Byte Pair Encoding (BPE).

# For example, BPE breaks down the sentence "They felt happiness" into the following list of tokens: ["They", " felt", " happi", "ness"]
# During 
# tokenization
# Preview: Docs Loading link description
# , we’ll also create the vocabulary containing the set of unique tokens the model can recognize and use. Specifically, the vocabulary indexes each unique token to a unique token ID.

# Embeddings
# Each token ID is just an 
# index
# Preview: Docs Loading link description
#  number (e.g., "happi" -> 8120) that the language model references (because it can’t directly understand words!). The key component that allows the model to understand language is embeddings.

# The token IDs are passed to an embedding layer, which maps each token into a numerical representation of continuous values called an embedding vector:

# import torch.nn as nn
# embedding = nn.Embedding(vocab_size, embed_size)

# Copy to Clipboard

# vocab_size: creates an embedding for each token in the vocabulary
# embed_size: number of values in each embedding vector
# # Example Token Embeddings
# Token "They"   → ID 1273 → [0.12, -0.54, ..., 0.03]
# Token " felt"  → ID 632  → [-0.34, 0.91, ..., -0.20]
# Token "happi"  → ID 8120 → [0.08, 0.65,  ..., 0.44]
# Token "ness"   → ID 421  → [0.59, -0.18, ..., 0.72]

# Copy to Clipboard

# During inference:

# Tokenizer splits the new text into tokens.
# Vocabulary maps the tokens to their token IDs.
# Token IDs are fed into the embedding layer to obtain their embedding vectors.
# Embedding vectors are passed through the layers of the language model.
# The final layer outputs task-dependent predictions (e.g., labels for classification, or next tokens for text generation).
# Token Similarity
# After training the language model on large amounts of text, the learned embeddings capture each token’s semantic information, where tokens that appear in similar contexts will also share similar embedding values.

# First, we’ll extract a model’s embedding layer to extract token embeddings using:

# E_layer = model.embedding.weight.detach().cpu().numpy()

# Copy to Clipboard

# embedding is the name of the model’s embedding layer
# Then, we’ll use the cosine similarity to measure how similar two token embedding vectors are based on the cosine angle between them:

# from sklearn.metrics.pairwise import cosine_similarity
# i, j = vocabulary["felt"], vocabulary["happi"]
# similarity = cosine_similarity([E_layer[i]], [E_layer[j]])[0,0]

# Copy to Clipboard

# Cosine similarity ranges from -1 (completely opposite) to 1 (identical direction), with 0 indicating orthogonal vectors.

# Lastly, we can visualize embeddings and see if similar words are nearby.

# Because embeddings are high-dimensional (often with hundreds of dimensions), we’ll need dimensionality reduction techniques to project them into 2-D visualizations.

# We’ll use a popular technique called t-SNE to reduce the embedding dimensions to 2-D:

# tsne_E2 = TSNE(n_components=2, random_state=42).fit_transform(E_layer)

# Copy to Clipboard

# fit_transform: inputs the model’s embedding layer
# n_components: number of output dimensions
# random_state: controls randomness during initialization and optimization (results can change slightly each run)
# # 2-D t-SNE Token Embeddings
# Token "They"   → ID 1273 → [-17.87,   4.14]
# Token " felt"  → ID 632  → [  0.54,   8.56]
# Token "happi"  → ID 8120 → [ 16.81, -13.62]
# Token "ness"   → ID 421  → [ -1.10,  -4.21]

# Copy to Clipboard

# In short, t-SNE is a non-linear dimensionality reduction technique that maps high-dimensional data points close together in a lower-dimensional space by converting their pairwise distances into probabilities and minimizing the divergence between them.

# Now we can plot the reduced 2-D embeddings using matplotlib:

# import matplotlib.pyplot as plt
# def plot_words(words, coords_2d):
#     idxs = [vocab[w] for w in words if w in vocab]
#     pts = coords_2d[idxs]
#     plt.figure(figsize=(7,6))
#     plt.scatter(pts[:,0], pts[:,1], s=24)
#     for (x,y), w in zip(pts, words):
#         if w in vocab:
#             plt.text(x+0.01, y+0.01, w, fontsize=9)
#     plt.title("Word Embeddings (t-SNE, 2D)")
#     plt.xlabel("dim-1"); plt.ylabel("dim-2")
#     plt.tight_layout(); plt.show()
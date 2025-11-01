import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
with open("data/training_data.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# Convert text to bag-of-words features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Define simple neural net
class ChannelNet(nn.Module):
    def __init__(self, input_size):
        super(ChannelNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = ChannelNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# Test accuracy
with torch.no_grad():
    preds = (model(X_test) > 0.5).float()
    acc = (preds.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"Accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
torch.save(model.state_dict(), "models/model.pt")
import pickle
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")

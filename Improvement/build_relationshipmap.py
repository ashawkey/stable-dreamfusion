import torch
import torch.nn as nn
import torch.optim as optim


labels = ["left", "right", "front", "back", "up", "down"]



# 3. 定义模型结构
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.gru(x)
        h = h[:, -1, :]
        out = self.fc(h)
        return out

# 4. 训练模型
batch_size = 2
embedding_dim = 16
hidden_dim = 32
lr = 0.01
n_epochs = 50

model = TextClassifier(len(word_to_idx), embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    running_loss = 0.0
    for i in range(0, len(X), batch_size):
        inputs = X[i:i+batch_size]
        labels = Y[i:i+batch_size]
        inputs_len = [len(seq) for seq in inputs]
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        inputs = inputs.cuda()
        labels = torch.tensor(labels).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / (len(X)/batch_size)
    print('Epoch %d loss: %.4f' % (epoch+1, epoch_loss))

# 5. 应用模型进行预测
test_texts = ["向左走", "向右走", "向前走", "向后走", "向上升", "向下降"]
test_X = [text_to_tensor(text) for text in test_texts]
test_X_len = [len(seq) for seq in test_X]
test_X = nn.utils.rnn.pad_sequence(test_X, batch_first=True).cuda()
with torch.no_grad():
    test_outputs = model(test_X)
test_pred = test_outputs.argmax(dim=1).cpu().numpy()
test_labels = [labels[i] for i in test_pred]
print("Test predicted labels:", test_labels)

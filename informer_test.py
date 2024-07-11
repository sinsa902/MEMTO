import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 데이터셋 생성 (가상의 예시)
def generate_dataset(batch_size, sequence_length, input_size, output_size):
    x = torch.randn(batch_size, sequence_length, input_size)
    y = torch.randn(batch_size, sequence_length, output_size)
    return x, y


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)  # 수정 필요
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        enc_output = self.transformer_encoder(x)
        enc_output = enc_output.permute(1, 0, 2)
        return enc_output


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)  # 수정 필요
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        dec_output = self.transformer_decoder(x)
        dec_output = dec_output.permute(1, 0, 2)
        return dec_output


class Informer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        encoder_layers,
        decoder_layers,
        output_size,
    ):
        super(Informer, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_heads, encoder_layers)
        self.decoder = Decoder(input_size, hidden_size, num_heads, decoder_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output)
        output = self.fc(dec_output)
        return output


# 하이퍼파라미터 설정
batch_size = 16
sequence_length = 10
input_size = 5
output_size = 1
hidden_size = 64
num_heads = 4
encoder_layers = 2
decoder_layers = 2
learning_rate = 0.001
num_epochs = 10

# 모델 초기화
model = Informer(
    input_size, hidden_size, num_heads, encoder_layers, decoder_layers, output_size
)

# 손실 함수와 최적화 기법 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 데이터셋 생성
train_x, train_y = generate_dataset(
    batch_size, sequence_length, input_size, output_size
)
val_x, val_y = generate_dataset(batch_size, sequence_length, input_size, output_size)

# 학습과 평가
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_x)

    # 손실 계산
    loss = criterion(outputs, train_y)

    # Backward pass 및 경사도 업데이트
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_x)
        val_loss = criterion(val_outputs, val_y)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
    )

# 학습 완료 후 모델을 사용하여 예측
test_x, test_y = generate_dataset(batch_size, sequence_length, input_size, output_size)
with torch.no_grad():
    model.eval()
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y)
    print(f"Test Loss: {test_loss.item():.4f}")

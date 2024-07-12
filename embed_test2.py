import torch
import torch.nn as nn
import torch.optim as optim

# 하이퍼파라미터 설정
num_time_steps = 100  # 시계열 데이터의 시간 스텝 수
num_features = 10  # 시계열 데이터의 피처 수
embedding_dims = 32  # 모든 범주형 변수의 임베딩 차원
num_conv_filters = 512  # Conv1D의 필터 수
conv_kernel_size = 3  # Conv1D의 커널 크기
num_categories_list = [10, 15, 20, 25, 30]  # 각 범주형 변수의 클래스 수
total_categories = sum(num_categories_list)  # 모든 범주형 변수의 클래스 수 합


# 모델 정의
class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        num_time_steps,
        num_features,
        embedding_dims,
        num_conv_filters,
        conv_kernel_size,
        total_categories,
    ):
        super(TimeSeriesModel, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_conv_filters,
            kernel_size=conv_kernel_size,
        )
        self.embedding = nn.Embedding(total_categories, embedding_dims)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            num_conv_filters * (num_time_steps - conv_kernel_size + 1)
            + embedding_dims * len(num_categories_list),
            1,
        )

    def forward(self, time_series, categorical_inputs):
        x = self.conv1d(time_series)
        x = torch.relu(x)
        x = self.flatten(x)

        embeddings = []
        offset = 0
        for i, categorical_input in enumerate(categorical_inputs):
            num_categories = num_categories_list[i]
            embedded_cat = self.embedding(categorical_input + offset)
            embedded_cat = self.flatten(embedded_cat)
            embeddings.append(embedded_cat)
            offset += num_categories

        combined = torch.cat([x] + embeddings, dim=1)
        output = torch.sigmoid(self.fc(combined))
        return output


# 모델 초기화
model = TimeSeriesModel(
    num_time_steps,
    num_features,
    embedding_dims,
    num_conv_filters,
    conv_kernel_size,
    total_categories,
)

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 예시 데이터 생성
num_samples = 1000
time_series_data = torch.rand((num_samples, num_features, num_time_steps))
categorical_data = [
    torch.randint(0, num_categories_list[i], (num_samples, 1))
    for i in range(len(num_categories_list))
]
labels = torch.randint(0, 2, (num_samples, 1)).float()

# 데이터로더 생성 (배치 단위로 학습하기 위함)
batch_size = 32
train_data = torch.utils.data.TensorDataset(time_series_data, *categorical_data, labels)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        time_series_batch = batch[0]
        categorical_batches = [batch[i] for i in range(1, 6)]
        labels_batch = batch[6]

        optimizer.zero_grad()
        outputs = model(time_series_batch, categorical_batches)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 모델 요약 출력
print(model)

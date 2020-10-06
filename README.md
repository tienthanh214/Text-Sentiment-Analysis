# Text  Sentiment Analysis
Ứng dụng Machine Learning trong việc phân tích cảm xúc (tích cực hoặc tiêu cực) của một bình luận (text)
# Mô tả

## Dataset
IMDB dataset (highly-polar movie reviews with 50k reviews).

Bao gồm 50000 records, mỗi record được xác định bởi hai giá trị "review" và "sentiment".

Trong đó: "review" dưới dạng text là bình luận đánh giá
          "sentiment" có giá trị là positive hoặc negative biểu thị cảm xúc của đánh giá
          
Link Kaggle: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Model (Neural Network)
* 1 tầng Embedding với ý nghĩa đưa input dạng text và nhận output là biểu diễn vector 128 chiều trong không gian
* 1 tầng Dropout với tỉ lệ 20%
* 1 tầng LSTM 128 hidden unit
* 1 tầng Dropout với tỉ lệ 20%
* 1 tầng output Dense với 2 unit biểu thị cho nhãn "positive" và "negative" sử dụng activation function là softmax
```
 model = Sequential([
    Embedding(WORD_SIZE + 1, 128, input_length = X_train.shape[1]),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(2, activation = tf.nn.softmax)              
])
```

Training
```
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_val, y_val)) 
```

# Accuracy
Test trên tập 10000 review từ dataset đạt được tỉ lệ là 87.29%.

Link final model: https://drive.google.com/file/d/1-0IODU8w1313wMhBsEFan20a8P9neWr7/view?usp=sharing


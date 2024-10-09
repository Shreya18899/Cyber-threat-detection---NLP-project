import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, AdamW
import torch.nn as nn
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

st = time.time()

def preprocess_text(text):
    """
    Preprocesses the input text by applying a series of natural language processing steps:
    
    1. Tokenization: Splits the input text into individual words (tokens) using NLTK's `word_tokenize`.
    2. Lowercasing: Converts all tokens to lowercase to ensure uniformity and filters out any punctuation.
    3. Lemmatization Reduces each token to its base or dictionary form using NLTK's `WordNetLemmatizer`.
    4. Stop Words Removal: Removes commonly used words (like "the", "is", "and") that provide little value for analysis.
    5. Token Rejoining: Joins the remaining tokens back into a single string for further processing or analysis.

    """
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in punctuations]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('cyber-threat-intelligence_all.csv')
print()
# Remove url and hash
df = df[~df["label"].isin(["url", "hash"])].reset_index(drop=True)
# If no label was identified consider it benign / no threat detected
df['label'].fillna('benign', inplace=True)
df = df.fillna(0)


print(df["label"].value_counts(normalize=True))

# train on first 10k observations, validate on the next 5 observations and test on the remaining ~4k observations
train_df = df[:10000].copy()
valid_df = df[10000:15000].copy()
test_df = df[15000:].copy()

train_df = train_df[['text', 'label']]
valid_df = valid_df[['text', 'label']]
test_df = test_df[['text', 'label']]

train_df['text_new'] = train_df['text'].apply(preprocess_text)
valid_df['text_new'] = valid_df['text'].apply(preprocess_text)
test_df['text_new'] = test_df['text'].apply(preprocess_text)

# drop "text" since "text_new" was created
train_df = train_df.drop(['text'], axis=1)
valid_df = valid_df.drop(['text'], axis=1)
test_df = test_df.drop(['text'], axis=1)

# The code is removing all punctuation, special characters, and symbols from the text in valid_df['text_new'],
# leaving only letters, digits, and spaces.
train_df['text_new'] = train_df['text_new'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
valid_df['text_new'] = valid_df['text_new'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
test_df['text_new'] = test_df['text_new'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

label_encoder = LabelEncoder()
label_encoder.fit(train_df['label'])

train_labels_encoded = label_encoder.transform(train_df['label'])
valid_labels_encoded = label_encoder.transform(valid_df['label'])
test_labels_encoded = label_encoder.transform(test_df['label'])

train_binary_labels = (train_df['label'] != 'benign').astype(int).values
valid_binary_labels = (valid_df['label'] != 'benign').astype(int).values
test_binary_labels = (test_df['label'] != 'benign').astype(int).values

# Load the distilbert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(dataframe):
    return tokenizer(dataframe['text_new'].tolist(), max_length=128, padding='max_length', truncation=True, return_tensors='pt')

train_encodings = tokenize_data(train_df)
valid_encodings = tokenize_data(valid_df)
test_encodings = tokenize_data(test_df)

# print(train_encodings)

class HierarchicalDistilBertModel(nn.Module):
    def __init__(self, num_threat_classes):
        super().__init__()
        # call the distilbert pretrained model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # add a linear layer of dim 768*1 which serves as a binary classifier
        self.binary_classifier = nn.Linear(self.distilbert.config.dim, 1)
        # add a linear layer of dim 768*number of threat classes which serves as a muliclass classifier
        self.multiclass_classifier = nn.Linear(self.distilbert.config.dim, num_threat_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        # threat_prob -  A probability score from the binary classifier, indicating the likelihood that the input contains a threat (binary classification).
        threat_prob = torch.sigmoid(self.binary_classifier(hidden_state))
        # class_probs: A vector of probabilities across the num_threat_classes, indicating which specific threat class the input belongs to (multiclass classification).
        class_probs = torch.softmax(self.multiclass_classifier(hidden_state), dim=1)
        return threat_prob, class_probs

class ThreatDataset(Dataset):
    """
    Pytroch dataloader to load the dataset efficiently from disc into memory
    """
    def __init__(self, encodings, binary_labels, multiclass_labels):
        self.encodings = encodings
        self.binary_labels = binary_labels
        self.multiclass_labels = multiclass_labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['binary_labels'] = torch.tensor(self.binary_labels[idx], dtype=torch.float)
        item['multiclass_labels'] = torch.tensor(self.multiclass_labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.binary_labels)

def hierarchical_loss(outputs, binary_labels, multiclass_labels):
    # computing the hierarchical loss by calculating the binary cross-entropy loss 
    # and multiclass cross entropy loss and back propagating it.
    binary_output, class_output = outputs
    binary_loss_function = nn.BCEWithLogitsLoss()
    multiclass_loss_function = nn.CrossEntropyLoss()
    binary_loss = binary_loss_function(binary_output.squeeze(), binary_labels.float())
    multiclass_loss = multiclass_loss_function(class_output, multiclass_labels.long())
    total_loss = binary_loss + multiclass_loss
    return total_loss

train_dataset = ThreatDataset(train_encodings, train_binary_labels, train_labels_encoded)
valid_dataset = ThreatDataset(valid_encodings, valid_binary_labels, valid_labels_encoded)
test_dataset = ThreatDataset(test_encodings, test_binary_labels, test_labels_encoded)
print("Datasets created")

# increasing the batch size to 64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Dataloaders called")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device present is : ", device)

model = HierarchicalDistilBertModel(num_threat_classes=len(label_encoder.classes_))
model.to(device)
print("model shifted to device")

optimizer = AdamW(model.parameters(), lr=5e-5)

def calculate_accuracy(threat_prob, binary_labels, class_probs, multiclass_labels):
    binary_pred = (threat_prob > 0.5).float()
    print(binary_pred)
    print(binary_pred.shape)
    binary_correct = (binary_pred == binary_labels).float().sum()
    print(binary_correct)
    binary_accuracy = binary_correct / binary_labels.size(0)

    _, multiclass_pred = torch.max(class_probs, dim=1)
    multiclass_correct = (multiclass_pred == multiclass_labels).float().sum()
    multiclass_accuracy = multiclass_correct / multiclass_labels.size(0)

    return binary_accuracy.item(), multiclass_accuracy.item()

best_accuracy = 0.0

for epoch in range(10):
    print("Model training started")
    model.train()
    for inputs in train_loader:
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        binary_labels = inputs['binary_labels'].to(device)
        multiclass_labels = inputs['multiclass_labels'].to(device)

        optimizer.zero_grad()
        threat_prob, class_probs = model(input_ids, attention_mask)

        loss = hierarchical_loss((threat_prob, class_probs), binary_labels, multiclass_labels)
        print("hierarchical loss computed")

        loss.backward()
        #optimizer.zero_grad() is used to reset the gradients of all model parameters before backpropagation
        optimizer.step()
        print("Model parameters updated")

    model.eval()
    total_binary_accuracy, total_multiclass_accuracy, batch_count = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs in valid_loader:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            binary_labels = inputs['binary_labels'].to(device)
            multiclass_labels = inputs['multiclass_labels'].to(device)

            threat_prob, class_probs = model(input_ids, attention_mask)

            print(f"Threat Probabilities: {threat_prob.cpu().numpy()}")
            print(f"Binary Labels: {binary_labels.cpu().numpy()}")
            print(f"Class Probabilities: {class_probs.cpu().numpy()}")
            print(f"Multiclass Labels: {multiclass_labels.cpu().numpy()}")

            binary_accuracy, multiclass_accuracy = calculate_accuracy(threat_prob, binary_labels, class_probs, multiclass_labels)
            total_binary_accuracy += binary_accuracy
            total_multiclass_accuracy += multiclass_accuracy
            batch_count += 1

    avg_binary_accuracy = total_binary_accuracy / batch_count
    avg_multiclass_accuracy = total_multiclass_accuracy / batch_count
    print(f"Epoch {epoch}: Validation Binary Accuracy: {avg_binary_accuracy}, Multiclass Accuracy: {avg_multiclass_accuracy}")

    current_accuracy = (avg_binary_accuracy + avg_multiclass_accuracy) / 2
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), "best_model_distilbert.pth")
        print("Saved best model")

print("Time taken ", time.time() - st)

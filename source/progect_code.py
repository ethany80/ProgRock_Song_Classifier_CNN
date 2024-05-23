'''
CAP6610 Spring 2024 Progect

Group: Closure/Continuation
Members:
Ethan Yanez
Austin Kreulach
'''

import os
import copy
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv

class SongSnippetWithImages:
  def __init__(self, sr, name, label, spect, mfcc, chrom, bos, band, centr):
    self.sr = sr
    self.name = name
    self.label = label
    self.spect = spect
    self.mfcc = mfcc
    self.chrom = chrom
    self.bos = bos
    self.band = band
    self.centr = centr

class CNN_1D_2Conv(nn.Module):
  def __init__(self, dropout=False):
    self.dropout = dropout

    super (CNN_1D_2Conv, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv1d(in_channels=45, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.dropout1 = nn.Dropout(0.25)
    self.layer2 = nn.Sequential(
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.dropout2 = nn.Dropout(0.25)
    self.fc1 = nn.Sequential(
        nn.Linear(in_features=64*53, out_features=600),
        nn.ReLU()
    )
    self.fc2 = nn.Sequential(
        nn.Linear(in_features=600, out_features=10),
        nn.ReLU()
    )
    self.fc3 = nn.Linear(in_features=10, out_features=2)
  def forward(self, x):
    out = self.layer1(x)
    if self.dropout:
      out = self.dropout1(out)
    out = self.layer2(out)
    if self.dropout:
      out = self.dropout2(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out
  
class CNN_1D_4Conv(nn.Module):
  def __init__(self):
    super (CNN_1D_4Conv, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv1d(in_channels=45, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.layer3 = nn.Sequential(
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.layer4 = nn.Sequential(
        nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.fc1 = nn.Sequential(
        nn.Linear(in_features=512*11, out_features=600),
        nn.ReLU()
    )
    self.fc2 = nn.Sequential(
        nn.Linear(in_features=600, out_features=100),
        nn.ReLU()
    )
    self.fc3 = nn.Sequential(
        nn.Linear(in_features=100, out_features=10),
        nn.ReLU()
    )
    self.fc4 = nn.Linear(in_features=10, out_features=2)
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)
    return out
  
class CNN_2D_2Conv(nn.Module):
  def __init__(self):
    super (CNN_2D_2Conv, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3)
    )
    self.fc1 = nn.Sequential(
        nn.Linear(in_features=32*210, out_features=600),
        nn.ReLU()
    )
    self.fc2 = nn.Sequential(
        nn.Linear(in_features=600, out_features=100),
        nn.ReLU()
    )
    self.fc3 = nn.Sequential(
        nn.Linear(in_features=100, out_features=10),
        nn.ReLU()
    )
    self.fc4 = nn.Linear(in_features=10, out_features=2)
  def forward(self, x):
    out = x.unsqueeze(1)
    out = self.layer1(out)
    out = self.layer2(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)
    return out
  
class CNN_2D_4Conv(nn.Module):
  def __init__(self):
    super (CNN_2D_4Conv, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc1 = nn.Sequential(
        nn.Linear(in_features=128*12, out_features=600),
        nn.ReLU()
    )
    self.fc2 = nn.Sequential(
        nn.Linear(in_features=600, out_features=100),
        nn.ReLU()
    )
    self.fc3 = nn.Sequential(
        nn.Linear(in_features=100, out_features=10),
        nn.ReLU()
    )
    self.fc4 = nn.Linear(in_features=10, out_features=2)
  def forward(self, x):
    out = x.unsqueeze(1)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)
    return out
  
class Classifier():
  def __init__(self, model):
    self.model = model
    self.name = model.__name__
    self.train_performance = Performance()
    self.test_performance = Performance()
  
  def display_performances(self, test=False):
    self.train_performance.display_performance(self.name + ' Training')
    if test:
      self.test_performance.display_performance(self.name + ' Test')

class Performance():
  def __init__(self):
    self.snippet_accuracies = []
    self.coin_flip_song_accuracies = []
    self.sum_outputs_song_accuracies = []
    self.combo_song_accuracies = []

  def add_accuracies(self, snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy):
    self.snippet_accuracies.append(snippet_accuracy)
    self.coin_flip_song_accuracies.append(coin_flip_song_accuracy)
    self.sum_outputs_song_accuracies.append(sum_outputs_song_accuracy)
    self.combo_song_accuracies.append(combo_song_accuracy)
  
  def get_mean_accuracies(self):
    total = len(self.snippet_accuracies)
    mean_snippet_accuracy = sum(self.snippet_accuracies) / total
    mean_coin_flip_song_accuracy = sum(self.coin_flip_song_accuracies) / total
    mean_sum_outputs_song_accuracy = sum(self.sum_outputs_song_accuracies) / total
    mean_combo_song_accuracy = sum(self.combo_song_accuracies) / total

    return mean_snippet_accuracy, mean_coin_flip_song_accuracy, mean_sum_outputs_song_accuracy, mean_combo_song_accuracy
  
  def display_performance(self, model_name):
    mean_snippet_accuracy, mean_coin_flip_song_accuracy, mean_sum_outputs_song_accuracy, mean_combo_song_accuracy = self.get_mean_accuracies()
    print(f'\n{model_name} Mean Performance:')
    print(f'Snippet Accuracy: {mean_snippet_accuracy}')
    print(f'Coin-Flip Song Accuracy: {mean_coin_flip_song_accuracy}')
    print(f'Sum-Outputs Song Accuracy: {mean_sum_outputs_song_accuracy}')
    print(f'Combo Song Accuracy: {mean_combo_song_accuracy}')

class SongData():
  def __init__(self, x, y, song_names):
    self.x = x
    self.y = y
    self.song_names = song_names

class EvalSongLabels():
  def __init__(self, true_label):
    self.true_label = true_label
    self.coin_flip_pred_label = None
    self.sum_outputs_pred_label = None
    self.combo_pred_label = None
    self.running_snippet_labels = 0
    self.running_outputs = [0, 0]

def generate_data(songs, dir, label, target_sr=11025, min_db=20.0, snippet_length=10):
  song_names = [f for f in os.listdir(dir) if f.endswith('.mp3')]
  songCount = 1
  failCount = 0

  for song_name in song_names:
    print(f'{songCount}: {song_name}')
    songCount += 1

    try:
      y, sr = librosa.load(os.path.join(dir, song_name), sr=target_sr)
    except Exception as e:
      failCount += 1
      continue

    y, index = librosa.effects.trim(y, top_db=min_db)

    snippet_samples = int(snippet_length * sr)

    remainder = int((len(y) / sr) % snippet_length)

    if remainder > 0:
        for _ in range (remainder):
          random_index = random.randint(0, len(y) - sr)

          y = np.delete(y, range(random_index, random_index + sr))

    snippets = [y[i:i+snippet_samples] for i in range(0, len(y), snippet_samples)]

    for snippet in snippets:

      spect = librosa.util.normalize(librosa.feature.melspectrogram(y=snippet, sr=sr, n_mels=10), norm=2.0, axis=1)

      mfcc = librosa.util.normalize(librosa.feature.mfcc(y=snippet, sr=sr, n_mfcc=20), norm=2.0, axis=1)

      chrom = librosa.util.normalize(librosa.feature.chroma_stft(y=snippet, sr=sr), norm=2.0, axis=1)

      bos = librosa.util.normalize(librosa.onset.onset_strength(y=snippet, sr=sr).reshape(1,-1), norm=2.0, axis=1)

      band = librosa.util.normalize(librosa.feature.spectral_bandwidth(y=snippet, sr=sr), norm=2.0, axis=1)

      centr = librosa.util.normalize(librosa.feature.spectral_centroid(y=snippet, sr=sr), norm=2.0, axis=1)

      song = SongSnippetWithImages(sr, song_name, label, spect, mfcc, chrom, bos, band, centr)

      songs.append(song)

  print(failCount)

  return

def get_songs(randomize=True):
  train_songs = []
  test_songs = []
  other_songs = []

  if not os.path.exists('processed_train_data.joblib'):
    generate_data(train_songs, 'Train_Data/Not_Progressive_Rock/Top_Of_The_Pops', False)
    generate_data(train_songs, 'Train_Data/Not_Progressive_Rock/Other_Songs', False)
    generate_data(train_songs, 'Train_Data/Progressive_Rock_Songs', True)

    joblib.dump(train_songs, 'processed_train_data.joblib')
  else:
    train_songs = joblib.load('processed_train_data.joblib')

  if not os.path.exists('processed_test_data.joblib'):
    generate_data(test_songs, 'Test_Data/Not_Progressive_Rock', False)
    generate_data(test_songs, 'Test_Data/Progressive_Rock_Songs', True)

    joblib.dump(test_songs, 'processed_test_data.joblib')
  else:
    test_songs = joblib.load('processed_test_data.joblib')

  if not os.path.exists('processed_other_data.joblib'):
    generate_data(other_songs, 'Test_Data/Other', False)

    joblib.dump(other_songs, 'processed_other_data.joblib')
  else:
    other_songs = joblib.load('processed_other_data.joblib')
  
  if randomize:
    train_songs = copy.deepcopy(train_songs)
    random.shuffle(train_songs)
    test_songs = copy.deepcopy(test_songs)
    random.shuffle(test_songs)
    other_songs = copy.deepcopy(other_songs)
    random.shuffle(other_songs)

  return train_songs, test_songs, other_songs

def display_images(song_name, dir):
  y, sr = librosa.load(os.path.join(dir, song_name))

  spect = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

  plt.figure(figsize=(15, 4))
  librosa.display.specshow(spect, sr=sr, x_axis='time', y_axis = 'hz', cmap='coolwarm')
  plt.title('Spectrogram')
  plt.show()

  melspect = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10), ref=np.max)

  plt.figure(figsize=(15, 4))
  librosa.display.specshow(melspect, sr=sr, x_axis='time', y_axis = 'mel', cmap='coolwarm')
  plt.title('Mel-Frequency Spectrogram (n_mels=10)')
  plt.show()

  chrom = librosa.feature.chroma_stft(y=y, sr=sr)

  plt.figure(figsize=(15, 4))
  librosa.display.specshow(chrom, sr=sr, x_axis='time', y_axis = 'chroma', cmap='inferno')
  plt.title('Chromogram')
  plt.show()

  bos = librosa.onset.onset_strength(y=y, sr=sr)

  norm_bos = librosa.util.normalize(bos)

  plt.figure(figsize=(15, 4))
  plt.plot(librosa.times_like(bos)/60, norm_bos, label='Onset Strength')
  plt.ylim(np.min(norm_bos), 1)
  plt.xlabel('Time (minutes)')
  plt.ylabel('Normalized Onset Strength')
  plt.legend()
  plt.title('Beat Onset Envelope')
  plt.show()

  frame_size = 1024
  hop_length = 512

  centr = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]

  t = librosa.frames_to_time(range(len(centr)))

  plt.figure(figsize=(15, 4))
  plt.plot(t, centr, label='Spectral Centroid', color='b')
  plt.ylabel('Frequency')
  plt.title('Spectral Centroid')
  plt.show()

  band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]

  plt.figure(figsize=(15, 4))
  plt.plot(t, band, label='Spectral Bandwidth', color='r')
  plt.ylabel('Frequency')
  plt.title('Spectral Bandwidth')
  plt.show()

def feature_importance(songs):
  sample_count = 10000

  batch_songs = random.sample(songs, sample_count)

  x = []
  y = []

  for song in batch_songs:
    features = np.concatenate([
        song.spect,
        song.mfcc,
        song.chrom,
        song.bos,
        song.band,
        song.centr
    ], axis=0).T

    if features.shape != (216, 45):
      continue
    x.extend(features)
    y.extend([1 if song.label == True else 0] * len(features))

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

  clf = RandomForestClassifier(verbose=True)
  clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)
  print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred)}')

  feature_importances = clf.feature_importances_

  yticks = []

  for i in range(1, 11):
    yticks.append(f'Mel{i}')
  for i in range(1, 21):
    yticks.append(f'MFCC{i}')
  for i in range(1, 13):
    yticks.append(f'Chrom{i}')
  yticks.append('Beat Onset Envelope')
  yticks.append('Spectral Bandwidth')
  yticks.append('Spectral Centroid')

  sorted_indices = sorted(range(len(feature_importances)), key=lambda i: feature_importances[i])
  yticks = [yticks[i] for i in sorted_indices]
  feature_importances = [feature_importances[i] for i in sorted_indices]

  plt.figure(figsize=(15, 8))
  plt.barh(range(len(feature_importances)), feature_importances, align='center')
  plt.yticks(range(len(feature_importances)), yticks)
  plt.xlabel('Importance')
  plt.ylabel('Features')
  plt.title('Feature Importance')
  plt.show()

def display_confusion_matrix(confusion_matrix):
  plt.figure(figsize=(6,6))
  plt.imshow(confusion_matrix, interpolation='nearest', cmap=LinearSegmentedColormap.from_list("custom_colormap", plt.cm.Blues(np.linspace(0.3, 1.0, 10))))
  plt.title('Confusion Matrix', fontsize=20)
  plt.colorbar()
  plt.xlabel('Predicted Label', fontsize=18)
  plt.ylabel('True Label', fontsize=18)
  plt.gca().xaxis.labelpad = 10
  plt.gca().yaxis.labelpad = 10
  classes = ['Prog', 'Non-Prog']
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, fontsize=12)
  plt.yticks(tick_marks, classes, rotation=90, fontsize=12)
  for i in range(confusion_matrix.shape[0]):
      for j in range(confusion_matrix.shape[1]):
          plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                  ha='center', va='center', color='white', fontsize=12)
  plt.show()

def get_train_test_sets(test_size, x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

  x_train_tens = torch.tensor(x_train, dtype=torch.float32)
  y_train_tens = torch.tensor(y_train, dtype=torch.long)
  x_test_tens = torch.tensor(x_test, dtype=torch.float32)
  y_test_tens = torch.tensor(y_test, dtype=torch.long)

  train_dataset = TensorDataset(x_train_tens, y_train_tens)
  test_dataset = TensorDataset(x_test_tens, y_test_tens)

  return train_dataset, test_dataset

def get_features_labels_names(songs):
    x = []
    y = []
    song_names = []

    for song in songs:
      features = np.concatenate([
          song.spect,
          song.mfcc,
          song.chrom,
          song.bos,
          song.band,
          song.centr
      ], axis=0)

      if features.shape != (45, 216):
        continue
      x.append(features)
      y.append(1 if song.label == True else 0)
      song_names.append(song.name)

    song_data = SongData(x, y, song_names)

    return song_data

def train_cnn(train, device, model, batch, num_epochs, learning_rate, error, show_graphs):
  train_loader = DataLoader(train, batch_size=batch)

  model.to(device)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  reduced_lr_total_iters = 1

  scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=reduced_lr_total_iters),
    optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    ], milestones= [reduced_lr_total_iters])

  losses = []

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = error(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          running_loss += loss.data
          optimizer.step()
      scheduler.step()
      epoch_loss = (running_loss/len(train_loader)).item()
      if (epoch_loss >= 0.68):
        print('RESTART')
        return train_cnn(train, device, type(model)(), batch, num_epochs, learning_rate, error, show_graphs)
      losses.append(epoch_loss)
      print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

  if show_graphs:
    plt.plot(range(1, num_epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

  return model

def evaluate(test_dataset, model, device, song_names, batch, show_graphs, type):
  test_loader = DataLoader(test_dataset, batch_size=batch)

  model.eval()
  correct = 0
  total = 0
  all_outputs = []
  pred_labels = []

  with torch.no_grad():
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          all_outputs.extend(outputs.tolist())
          predicted = torch.max(outputs, 1)[1]
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          pred_labels.extend(predicted.tolist())

  snippet_accuracy = correct / total
  print(f'\nSnippet Accuracy: {snippet_accuracy}')

  labeled_songs = {}

  for pred_label, song_name, y_label, outputs in zip(pred_labels, song_names[-len(pred_labels):], test_dataset.tensors[1].tolist(), all_outputs):
    if song_name not in labeled_songs:
      labeled_songs[song_name] = EvalSongLabels(y_label)
    if pred_label == 1:
      labeled_songs[song_name].running_snippet_labels += 1
    else:
      labeled_songs[song_name].running_snippet_labels -= 1
    labeled_songs[song_name].running_outputs[0] += outputs[0]
    labeled_songs[song_name].running_outputs[1] += outputs[1]

  for key, song_labels in labeled_songs.items():
    if song_labels.running_snippet_labels <= 0:
      song_labels.coin_flip_pred_label = 0
    else:
      song_labels.coin_flip_pred_label = 1

    song_labels.sum_outputs_pred_label = song_labels.running_outputs.index(max(song_labels.running_outputs))

    if song_labels.running_snippet_labels == 0:
      song_labels.combo_pred_label = song_labels.sum_outputs_pred_label
    else:
      song_labels.combo_pred_label = song_labels.coin_flip_pred_label

  labeled_songs_values = list(labeled_songs.values())

  coin_flip_song_accuracy = len([song for song in labeled_songs_values if song.coin_flip_pred_label == song.true_label]) / len(labeled_songs_values)
  sum_outputs_song_accuracy = len([song for song in labeled_songs_values if song.sum_outputs_pred_label == song.true_label]) / len(labeled_songs_values)
  combo_song_accuracy = len([song for song in labeled_songs_values if song.combo_pred_label == song.true_label]) / len(labeled_songs_values)

  print(f'Coin-Flip Song Accuracy: {coin_flip_song_accuracy}')
  print(f'Sum-Outputs Song Accuracy: {sum_outputs_song_accuracy}')
  print(f'Combo Song Accuracy: {combo_song_accuracy}')

  prog_correct = len([song for song in labeled_songs_values if song.coin_flip_pred_label == song.true_label == 1])
  prog_total = len([song for song in labeled_songs_values if song.true_label == 1])
  nonprog_correct = len([song for song in labeled_songs_values if song.coin_flip_pred_label == song.true_label == 0])
  nonprog_total = len([song for song in labeled_songs_values if song.true_label == 0])

  if show_graphs:
    display_confusion_matrix(np.array([[prog_correct, prog_total - prog_correct], [nonprog_total - nonprog_correct, nonprog_correct]]))

    song_probabilities = []

    for key, value in labeled_songs.items():
      sf_probabilities = np.exp(value.running_outputs) / np.sum(np.exp(value.running_outputs))

      song_probabilities.append([key, sf_probabilities[0], sf_probabilities[1], value.true_label])
    
    write_to_csv(song_probabilities, f'Outputs/{type}_Song_Probabilities.csv')

    '''
    songs_to_save = []

    for key, value in labeled_songs.items():
      songs_to_save.append([key, value.coin_flip_pred_label])

    write_to_csv(songs_to_save, f'{type}_Other_Songs_Labeled.csv')
    '''

  return snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy, labeled_songs

def write_song_misclassifications(labeled_songs_over_runs):
  song_misclassifications = {}

  for labeled_songs in labeled_songs_over_runs:
    for key, value in labeled_songs.items():
      if key not in song_misclassifications:
        song_misclassifications[key] = [0, 0]

      if value.combo_pred_label != value.true_label:
        song_misclassifications[key][0] += 1
      song_misclassifications[key][1] += 1

  song_misclassification_rates = []

  for key, value in song_misclassifications.items():
    song_misclassification_rates.append([key, value[0]/value[1]])

  write_to_csv(song_misclassification_rates, 'Outputs/Song_Misclassification.csv')

def write_performances(performance, loc):
  mean_sn, mean_cf, mean_so, mean_co = performance.get_mean_accuracies()

  combined_lists = list(zip(performance.snippet_accuracies + [mean_sn], 
                            performance.coin_flip_song_accuracies + [mean_cf], 
                            performance.sum_outputs_song_accuracies + [mean_so], 
                            performance.combo_song_accuracies + [mean_co]))

  result = [list(row) for row in combined_lists]

  write_to_csv(result, loc)

def write_to_csv(data, loc):
  os.makedirs(os.path.dirname(loc), exist_ok=True)

  with open(loc, "w", newline="", encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)

def run_classifier(classifier, train_song_data, test_song_data, run, test_size=0.2, train_batch=32, eval_batch = 32, num_epochs=100, learning_rate=0.001, error=nn.CrossEntropyLoss(), show_graphs=True):
  print(f'\nRun {run}\n')
  
  model = classifier.model()

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('Device', device)

  train_train_dataset, train_test_dataset = get_train_test_sets(test_size, train_song_data.x, train_song_data.y)

  model = train_cnn(train_train_dataset, device, model, train_batch, num_epochs, learning_rate, error, show_graphs)
  snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy, labeled_songs = evaluate(train_test_dataset, model, device, train_song_data.song_names, eval_batch, show_graphs, 'Train')

  classifier.train_performance.add_accuracies(snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy)

  if test_song_data:
    test_test_dataset = TensorDataset(torch.tensor(test_song_data.x, dtype=torch.float32), torch.tensor(test_song_data.y, dtype=torch.long))
    snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy, _ = evaluate(test_test_dataset, model, device, test_song_data.song_names, eval_batch, show_graphs, 'Test')
    classifier.test_performance.add_accuracies(snippet_accuracy, coin_flip_song_accuracy, sum_outputs_song_accuracy, combo_song_accuracy)
  
  return labeled_songs

def compare_classifiers(classifiers, train_song_data, runs, test_song_data=None):
  labeled_songs_over_runs = []
  
  for classifier in classifiers:
    for run in range(runs):
      print('------------------------------------')
      print(f'{classifier.name} Runs')
      print(f'------------------------------------')
      labeled_songs_over_runs.append(run_classifier(classifier, train_song_data, test_song_data, run))

    classifier.display_performances(test=True)
  
  for classifier in classifiers:
    write_performances(classifier.train_performance, f'Outputs/{classifier.name}/train.csv')
    write_performances(classifier.test_performance, f'Outputs/{classifier.name}/test.csv')
  
  write_song_misclassifications(labeled_songs_over_runs)

def main():
  ## Read songs into memory, create shuffled copy
  train_songs, test_songs, other_songs = get_songs()

  ## Transform song objects into train/test sets
  train_song_data = get_features_labels_names(train_songs)
  test_song_data = get_features_labels_names(test_songs)

  classifiers = [
    Classifier(type(CNN_1D_4Conv()))
  ]

  compare_classifiers(classifiers, train_song_data, 1, test_song_data=test_song_data)

main()

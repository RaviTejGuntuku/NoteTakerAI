import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math

f = open("tryst_transcription.txt", "r")
text = ""

for i in f:
    text += i

model = SentenceTransformer('all-mpnet-base-v2')
sentences = text.split('.')
embeddings = model.encode(sentences)
print(embeddings.shape)

similarities = cosine_similarity(embeddings)
sns.heatmap(similarities, annot=True).set_title('Cosine similarities matrix')


def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5*x)))


def activate_similarities(similarities: np.array, p_size=10) -> np.array:

    x = np.linspace(-10, 10, p_size)

    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x), (0, similarities.shape[0]-p_size))
    # 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each)
                 for each in range(0, similarities.shape[0])]
    # 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0, similarities.shape[0]-len(each)))
                 for each in diagonals]
    # 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    # 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    # 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


# Lets apply our function. For long sentences i reccomend to use 10 or more sentences
activated_similarities = activate_similarities(similarities, p_size=5)

# lets create empty fig for our plor
fig, ax = plt.subplots()
# 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
# order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
minmimas = argrelextrema(activated_similarities, np.less, order=2)
# plot the flow of our text with activated similarities
sns.lineplot(y=activated_similarities, x=range(
    len(activated_similarities)), ax=ax).set_title('Relative minimas')
# Now lets plot vertical lines in order to see where we created the split
plt.vlines(x=minmimas, ymin=min(activated_similarities), ymax=max(
    activated_similarities), colors='purple', ls='--', lw=1, label='vline_multiple - full height')

# Get the length of each sentence
sentece_length = [len(each) for each in sentences]
# Determine longest outlier
long = np.mean(sentece_length) + np.std(sentece_length) * 2
# Determine shortest outlier
short = np.mean(sentece_length) - np.std(sentece_length) * 2
# Shorten long sentences
text = ''
for each in sentences:
    if len(each) > long:
        comma_splitted = each.replace(',', '.')
    else:
        text += f'{each}. '
sentences = text.split('. ')
text = ''
for each in sentences:
    if len(each) < short:
        text += f'{each} '
    else:
        text += f'{each}. '

split_points = [each for each in minmimas[0]]
text = ''
for num, each in enumerate(sentences):
    if num in split_points:
        text += f'\n\n {each}. '
    else:
        text += f'{each}. '

f.close()

f1 = open("tryst_transcription.txt", "w")

f1.write(text)

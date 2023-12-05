#%%
"""
    Machine Learning Project
    Implementation for Metric Learning using Siamese Network (Triplet Loss)
"""
# Importing Libraries
import matplotlib, os, PIL, sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from tqdm import tqdm
from pathlib import Path

#%%
# Checking Versions of Libraries
print("TensorFlow: {}".format(tf.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("Numpy: {}".format(np.version.version))
print("Built With Cuda: {}".format(tf.test.is_built_with_cuda()))
print("TensorFlow GPU: {}".format(tf.config.list_physical_devices('GPU')))
print("PIL: {}".format(PIL.__version__))
print("sklearn: {}".format(sklearn.__version__))

#%%
# Configure the Image shape and Batch Size
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 16

#%%
# Set Up Paths for Data Directories
# NOTE: DO CHANGE THE PATH TO YOUR OWN DIRECTORY
main_dir = "C:/Users/liang/Downloads/aml-face-attendance"
anchor_dir = Path(os.path.join(main_dir, "Dataset/siamese_dir/anchor"))
positive_dir = Path(os.path.join(main_dir, "Dataset/siamese_dir/positive"))

#%%
# Preprocessing Image Data
def preprocess_image(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image

# Preprocessing Triplet Data
def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

#%%
# Sort the images accordingly to make sure that
# both anchor and positive are in the same order
anchor_images = sorted(
    [str(anchor_dir / f) for f in os.listdir(anchor_dir)]
)

positive_images = sorted(
    [str(positive_dir / f) for f in os.listdir(positive_dir)]
)

# Total number of images
image_count = len(anchor_images)

# Create dataset of anchors and positives
anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# Random shuffle the anchor images and positive images
rnd = np.random.RandomState(seed=42)
rnd.shuffle(anchor_images)
rnd.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Split the dataset into train and validation datasets
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

#%%
# Visualize the anchor, positive, and negative images
def visualize(anchor, positive, negative):
    def show(ax, image, label):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(label)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    labels = ['Anchor', 'Positive', 'Negative']
    for i in range(3):
        show(axs[i, 0], anchor[i], labels[0])
        show(axs[i, 1], positive[i], labels[1])
        show(axs[i, 2], negative[i], labels[2])

# Visualize the 3 anchor, positive, and negative images
visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

#%%
# Load the base model (ResNet50)
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", 
    input_shape=IMAGE_SHAPE + (3,), 
    include_top=False
)

#%%
# Display the base model architecture
base_model.summary()

#%%
# Build the model
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.Dense(256)(x)

# Create embedding model
embedding = tf.keras.Model(base_model.input, output, name="Embedding")

# Set all base model layers to be non-trainable
trainable = False
for layer in base_model.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

#%%
# Display the embedding model architecture
embedding.summary()

#%%
# Create the layer to compute the distance between anchor and positive embedding VS anchor and negative embedding
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

# Create inputs for anchor, positive and negative 
anchor_input = tf.keras.layers.Input(name="anchor", shape=IMAGE_SHAPE + (3,))
positive_input = tf.keras.layers.Input(name="positive", shape=IMAGE_SHAPE + (3,))
negative_input = tf.keras.layers.Input(name="negative", shape=IMAGE_SHAPE + (3,))

# Outputs containing the distances from the anchor to the positive and negative images
distances = DistanceLayer()(
    embedding(tf.keras.applications.resnet50.preprocess_input(anchor_input)),
    embedding(tf.keras.applications.resnet50.preprocess_input(positive_input)),
    embedding(tf.keras.applications.resnet50.preprocess_input(negative_input)),
)

# Define the siamese network
siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

#%%
# Create the Siamese Network model
# Computes the triplet loss using the three embeddings produced by the Siamese Network
class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    
#%%
# Define the model
siamese_model = SiameseModel(siamese_network)

#%%
# Compile and fit the model function
def compilefit():
    # Training history
    history_callback = tf.keras.callbacks.History()
    # Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    # Compile the model
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), weighted_metrics=[""])
    # Train the model
    siamese_model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[history_callback, early_stopping])

    return siamese_model

# NOTE: UNCOMMENT THE CODE below to compile and fit the model
# siamese_model = compilefit( )
# history = siamese_model.history.history
#%%
# Plot the training loss graph
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.legend()
# plt.show()
#%%
# Save the weights of the siamese model
def save(SiameseModel, checkpoint_name):
    SiameseModel.save_weights(Path(os.path.join("../Anti_Spoofing/checkpoints/", checkpoint_name)))

# NOTE: UNCOMMENT THE CODE below to save the model
# checkpoint_name = input("Checkpoint Name: ")
# save(siamese_model, checkpoint_name)

#%%
# Load the weights of the siamese model
def load(SiameseModel, checkpoint_name):
    SiameseModel.load_weights(Path(os.path.join('../Anti_Spoofing/checkpoints/', checkpoint_name)))

# Load the model
checkpoint_name = input("Checkpoint Name: ")
load(siamese_model, checkpoint_name)
#%%
# Save the embedding model from the siamese model into h5 format
def save_embedding(embedding_name):
    embedding.save(Path(os.path.join('../Anti_Spoofing/embeddings/', embedding_name)))

# NOTE: UNCOMMENT THE CODE below to save the embedding model
embedding_name = input("Embedding Name: ")
save_embedding(embedding_name + ".h5")
#%%
# Take a sample from the dataset to check similarity
sample = next(iter(train_dataset))
visualize(*sample)

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(tf.keras.applications.resnet50.preprocess_input(anchor)),
    embedding(tf.keras.applications.resnet50.preprocess_input(positive)),
    embedding(tf.keras.applications.resnet50.preprocess_input(negative)),
)

# Evaluation Metrics - Cosine Similarity
# Positive Similarity 
positive_similarity = tf.keras.metrics.CosineSimilarity()(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

# Negative Similarity 
negative_similarity = tf.keras.metrics.CosineSimilarity()(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())

# Evaluation Metrics - Euclidean Distance
# Positive Euclidean Distance
positive_euclidean_distance = np.linalg.norm(anchor_embedding - positive_embedding)
print("Positive euclidean distance:", positive_euclidean_distance)

# Negative Euclidean Distance
negative_euclidean_distance = np.linalg.norm(anchor_embedding - negative_embedding)
print("Negative euclidean distance:", negative_euclidean_distance)

#%%
# Calculate the similarity between two images for verification validation
verification_data_file = os.path.join(main_dir, "Dataset/verification_pairs_val.txt")

# Calculation for image similarity
def calculate_similarity(img1, img2):
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    img1_embedding = embedding(tf.keras.applications.resnet50.preprocess_input(img1[None, ...]))
    img2_embedding = embedding(tf.keras.applications.resnet50.preprocess_input(img2[None, ...]))

    # Calculate cosine similarity
    similarity = tf.keras.metrics.CosineSimilarity()(img1_embedding, img2_embedding).numpy()

    # Calculate euclidean distance (inverse)
    # euclidean_distance = np.linalg.norm(img1_embedding - img2_embedding)
    # similarity = 1 / euclidean_distance + 1

    return similarity

# Initialize the verification data, similarities and labels
verification_data = similarities = labels = []

# Import the verification data
with open(verification_data_file) as f:
    for line in f:
        trial1, trial2, result = os.path.join(main_dir, "Dataset/" + line.split()[0]), \
                                os.path.join(main_dir, "Dataset/" + line.split()[1]), \
                                int(line.split()[2])
        verification_data.append((trial1, trial2, result))

# Calculate the similarities and append to the similarities list
# Append the labels to the labels list
for data in tqdm(verification_data, desc="Calculating Similarities"):
    similarity = calculate_similarity(data[0], data[1])
    similarities.append(similarity)
    labels.append(data[2])

#%%
# Uncomment the code below to save the similarities, labels
# npz_file = input("NPZ File Name: ")
# np.savez(npz_file + ".npz", similarities=similarities, labels=labels)

#%%
# Load the similarities and labels
npz_file = input("NPZ File Name: ")
npz = np.load(os.path.join(main_dir, "Dataset/" + npz_file + ".npz"))
similarities = npz['similarities']
labels = npz['labels']
# %%
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

# Calculate F1 scores for each threshold
f1_scores = [f1_score(labels, similarities > threshold) for threshold in thresholds]

# Find the threshold that maximizes the F1 score
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1_score = f1_scores[best_threshold_index]
# Other Evaluation Metrics
accuracy = sklearn.metrics.accuracy_score(labels, similarities > best_threshold)
precision = sklearn.metrics.precision_score(labels, similarities > best_threshold)
recall = sklearn.metrics.recall_score(labels, similarities > best_threshold)

print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1_score}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Plot ROC curve with the best threshold
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], marker='o', color='red', label=f'Best Threshold (F1={best_f1_score:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Best Threshold')
plt.legend(loc="lower right")
plt.show()

#%%
# Use the first line of verification data as an example for cosine similarity and euclidean distance
trial1, trial2, result = os.path.join(main_dir, "Dataset/" + verification_data[0][0]), \
                        os.path.join(main_dir, "Dataset/" + verification_data[0][1]), \
                        int(verification_data[0][2])

trial1 = preprocess_image(trial1)
trial2 = preprocess_image(trial2)

trial1 = embedding(tf.keras.applications.resnet50.preprocess_input(trial1[None, ...]))
trial2 = embedding(tf.keras.applications.resnet50.preprocess_input(trial2[None, ...]))

# Calculate the cosine similarity
similarity = tf.keras.metrics.CosineSimilarity()(trial1, trial2).numpy()
print("Cosine similarity:", similarity)

# Calculate the euclidean distance
euclidean_distance = np.linalg.norm(trial1 - trial2)
print("Euclidean distance:", euclidean_distance)

# Result
print("Result:", result)
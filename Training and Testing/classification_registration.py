#%%
from tensorflow.keras.models import load_model, Model

# Load the trained model
model_path = r'C:\Users\102774145\Desktop\face_classification_model.h5'
trained_model = load_model(model_path)

# Get the model up to the layer before softmax
embedding_model = Model(inputs=trained_model.input, outputs=trained_model.layers[-2].output)

# Save the embedding model
embedding_model.save(r'C:\Users\102774145\Desktop\face_embedding_model.h5')

#%%
import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image

# Assuming your test dataset directory structure is like: test_data/class1, test_data/class2, ...
register_data_dir = r'C:\Users\102774145\Desktop\new_staff'

#%%
# Function to preprocess an image before passing it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

#%%
# Function to generate embeddings for a given image
def generate_embedding(img_path):
    img_array = preprocess_image(img_path)
    embedding = embedding_model.predict(img_array)
    return embedding

#%%
# Function to create an embedding database
def create_embedding_database(data_dir, embedding_model):
    embedding_database = {}

    # Iterate through each class directory
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # Get the list of image files in the class directory
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Take the embedding of the first image in each class
        if image_files:
            first_image_path = os.path.join(class_path, image_files[0])
            embedding = generate_embedding(first_image_path)

            # Save the embedding in the database
            embedding_database[class_name] = embedding.tolist()

    return embedding_database

#%%
# Create the embedding database
embedding_database = create_embedding_database(register_data_dir, embedding_model)

# Save the embedding database as a JSON file
embedding_database_path = r'C:\Users\102774145\Desktop\new_staff_embedding_database.json'
with open(embedding_database_path, 'w') as json_file:
    json.dump(embedding_database, json_file)

print("Embedding database created and saved at:", embedding_database_path)

#%%
# Load the embedding database
embedding_database_path = r'C:\Users\102774145\Desktop\new_staff_embedding_database.json'
with open(embedding_database_path, 'r') as json_file:
    embedding_database = json.load(json_file)

# Assuming your register data directory structure is like: register_data/class1, register_data/class2, ...
register_data_dir = r'C:\Users\102774145\Desktop\new_staff'

#%% Metric find best
"""
def find_most_similar_class(test_embedding, embedding_database):
    similarities = {}

    # Create the cosine similarity metric
    cosine_similarity_metric = tf.keras.metrics.CosineSimilarity(axis=-1)

    for class_name, stored_embedding in embedding_database.items():
        stored_embedding = np.array(stored_embedding).flatten()
        test_embedding_flat = test_embedding.flatten()

        # Update the metric with the embeddings
        cosine_similarity_metric.update_state([test_embedding_flat], [stored_embedding])

        # Get the result from the metric
        similarity = cosine_similarity_metric.result().numpy()

        # Reset the metric for the next pair
        cosine_similarity_metric.reset_states()

        similarities[class_name] = similarity

    most_similar_class = max(similarities, key=similarities.get)
    return most_similar_class, similarities[most_similar_class]
"""

#%% Sklearn find best and Unknown
from sklearn.metrics.pairwise import cosine_similarity

# Function to find the most similar class based on embeddings
def find_most_similar_class(test_embedding, embedding_database, threshold=0.6):
    similarities = {}

    for class_name, stored_embedding in embedding_database.items():
        stored_embedding = np.array(stored_embedding).flatten()
        test_embedding_flat = test_embedding.flatten()

        similarity = cosine_similarity([test_embedding_flat], [stored_embedding])[0][0]
        similarities[class_name] = similarity

    most_similar_class = max(similarities, key=similarities.get)

    # Check if similarity is below the threshold
    if similarities[most_similar_class] < threshold:
        return "Unknown User", similarities[most_similar_class]
    else:
        return most_similar_class, similarities[most_similar_class]

#%%
# Manually specify the path to the test image
test_image_path = r'C:\Users\102774145\Desktop\new_staff\donald\donald2.jpg'

# Generate the embedding for the test image
test_embedding = generate_embedding(test_image_path)

# Find the most similar class based on embeddings
most_similar_class, similarity = find_most_similar_class(test_embedding.flatten(), embedding_database)

print(f"Test Image: {test_image_path}")
print(f"Predicted Class: {most_similar_class}, Similarity: {similarity:.4f}")

#%%
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc, f1_score
from tqdm import tqdm

#%%
verification_data_dir = r'C:/Users/102774145/Desktop/verification_data'
verification_pair_val = r'C:/Users/102774145/Desktop/verification_pairs_val.txt'

#%%
def calculate_similarity(img1, img2):
    embedding1 = generate_embedding(img1).flatten()
    embedding2 = generate_embedding(img2).flatten()

    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
    return similarity

#%%
verification_data = []
with open(verification_pair_val, 'r') as file:
    for line in file:
        parts = line.strip().split()
        trial1, trial2, result = os.path.join(verification_data_dir, parts[0]), \
                                os.path.join(verification_data_dir, parts[1]), \
                                int(parts[2])
        verification_data.append((trial1, trial2, result))

#%%
similarities = []
labels = []
for data in tqdm(verification_data, desc="Calculating Similarities"):
    similarity = calculate_similarity(data[0], data[1])
    similarities.append(similarity)
    labels.append(data[2])
    
#%%
npz = np.load(r'C:/Users/102774145/Desktop/classification.npz')
similarities = npz['similarities']
labels = npz['labels']

#%%
# Iterate through test images and calculate true positive rate (TPR) and false positive rate (FPR)
true_labels = []
predicted_probabilities = []

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
print(f'AUC: {roc_auc:.2f}')

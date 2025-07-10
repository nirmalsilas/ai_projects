import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ipaddress


def ip_to_int(ip):
    return int(ipaddress.IPv4Address(ip))



# Load and preprocess the dataset
# Assuming the dataset is in a CSV file with the features described in readMe.txt
data = pd.read_csv("telecom_backhaul_packet.csv")

# Map string labels to integers
#label_mapping = {"not_valid": 0, "uplane": 1, "cplane": 2}
#data["label"] = data["label"].map(label_mapping)


# Convert IP addresses if they exist in the dataset
if "src_ip" in data.columns:
    data["src_ip"] = data["src_ip"].apply(ip_to_int)

if "dst_ip" in data.columns:
    data["dst_ip"] = data["dst_ip"].apply(ip_to_int)

# Convert protocol and flags (categorical) into numeric labels if needed
if "protocol" in data.columns:
    data["protocol"] = data["protocol"].astype("category").cat.codes

if "flags" in data.columns:
    data["flags"] = data["flags"].astype("category").cat.codes


# Extract features and labels
features = data.drop(columns=["label"])  # Drop the label column to get features
labels = data["label"]  # Labels: 0 = not_valid, 1 = uplane, 2 = cplane

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(features.shape[1],), activation="relu"),  # Input layer
    tf.keras.layers.Dense(32, activation="relu"),  # Hidden layer
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer
    tf.keras.layers.Dense(3, activation="softmax")  # Output layer (3 classes)
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model
model.save("packet_classifier_model.h5")
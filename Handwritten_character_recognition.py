import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class HandwrittenRecognitionSystem:
    def __init__(self):
        self.model = None
        self.history = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.class_names = [str(i) for i in range(10)] # Digits 0-9

    def load_and_preprocess(self):
        """Loads MNIST dataset and normalizes image data."""
        print("--- Loading MNIST Dataset ---")
        mnist = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # Normalize pixel values to 0-1 range
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        # Reshape for CNN (Samples, Height, Width, Channels)
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)
        
        print(f"Dataset Loaded: {len(self.X_train)} training images, {len(self.X_test)} test images.")

    def build_cnn_architecture(self):
        """Defines the Convolutional Neural Network layers."""
        print("\n--- Building CNN Architecture ---")
        self.model = models.Sequential([
            # Feature Extraction Layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Classification Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2), 
            layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def train_model(self, epochs=5):
        """Trains the model with a clean text output format."""
        print(f"\n--- Starting Training for {epochs} Epochs ---")
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=epochs, 
            validation_data=(self.X_test, self.y_test),
            verbose=2  # <-- Add this to fix the messy terminal codes!
        )

    def evaluate_and_visualize(self):
        """Generates performance plots and accuracy reports."""
        print("\n--- Evaluating Performance ---")
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # 1. Print Results Table
        stats = {"Metric": ["Test Accuracy", "Test Loss"], "Value": [f"{test_acc:.4f}", f"{test_loss:.4f}"]}
        print(pd.DataFrame(stats).to_markdown(index=False))

        # 2. Plot Training History
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_performance.png')
        print("Performance plots saved to 'training_performance.png'")

    def predict_random_sample(self):
        """Picks a random test image and demonstrates a visual prediction."""
        idx = np.random.randint(0, len(self.X_test))
        img = self.X_test[idx]
        
        prediction = self.model.predict(img.reshape(1, 28, 28, 1), verbose=0)
        predicted_label = np.argmax(prediction)
        
        plt.figure(figsize=(4,4))
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.title(f"Target: {self.y_test[idx]} | Prediction: {predicted_label}")
        plt.axis('off')
        plt.savefig('sample_prediction.png')
        print(f"Sample Prediction saved. AI predicted this digit is: {predicted_label}")

if __name__ == "__main__":
    # Initialize the system
    system = HandwrittenRecognitionSystem()
    
    # Run pipeline
    system.load_and_preprocess()
    system.build_cnn_architecture()
    system.train_model(epochs=5)
    system.evaluate_and_visualize()
    system.predict_random_sample()
   

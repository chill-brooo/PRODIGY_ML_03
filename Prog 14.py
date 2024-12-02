import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
from zipfile import ZipFile

# Step 1: Extract the dataset
data_path = r'C:\Users\Ved Thombre\Downloads\archive.zip'  # Update ZIP file path

with ZipFile(data_path, 'r') as zip:
    zip.extractall(r'C:\Users\Ved Thombre\Downloads')  # Extract to a specific directory
    print('The data set has been extracted.')

# Step 2: Check the extracted directory structure
extracted_dir = r'C:\Users\Ved Thombre\Downloads\dog-vs-cat'
print("Contents of the extracted directory:")
for root, dirs, files in os.walk(extracted_dir):
    level = root.replace(extracted_dir, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}    {f}")

# Step 3: Define paths and categories
data_dir = extracted_dir  # Update with the correct extracted folder path
categories = ['cats', 'dogs']  # Adjusted to match the folder names

# Step 4: Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(data_dir, category)  # Create path for each category
        if not os.path.exists(category_path):
            print(f"Warning: The path {category_path} does not exist.")
            continue
        
        for img in os.listdir(category_path):
            if img.endswith('.jpg') or img.endswith('.png'):  # Adjust based on  image formats
                img_array = cv2.imread(os.path.join(category_path, img))  # Read the image
                img_array = cv2.resize(img_array, (64, 64))  # Resize to 64x64
                images.append(img_array)
                # Assign label based on category
                labels.append(categories.index(category))  # 0 for cats, 1 for dogs
    
    return np.array(images), np.array(labels)

# Load the data
images, labels = load_data(data_dir)

# Check if images were loaded
if len(images) == 0:
    print("No images were loaded. Please check the directory structure.")
else:
    # Flatten images
    images = images.reshape(len(images), -1)  # Flatten to 1D

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Step 6: Create and train the SVM model
    model = svm.SVC(kernel='linear')  # can also try 'rbf', 'poly', etc.
    model.fit(X_train, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test)

    # Step 8: Print classification report
    print(classification_report(y_test, y_pred, target_names=categories))

    # Step 9: Print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Step 10: Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories)
    plt.yticks(tick_marks, categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

  

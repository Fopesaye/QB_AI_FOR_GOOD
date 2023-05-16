import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import pandas as pd

app = Flask(__name__)
# Make enumerate function available in Jinja2 environment
app.jinja_env.globals['enumerate'] = enumerate
# Define the path to the saved model
model_path = 'CNN.pt'

class ImageDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        image_path = self.metadata.iloc[index]['path']

        image = np.array(Image.open(image_path).convert('L')) / 255.0
        image = torch.from_numpy(np.array(image)).unsqueeze(0).float()
        if self.transform:
            image = self.transform(image)

        return image

class Net(nn.Module):
    def __init__(self, batch_size=32, hidden_dimension = (64, 128), last_layer_size = 128):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(batch_size, hidden_dimension[0], 3)
        self.conv3 = nn.Conv2d(hidden_dimension[0], hidden_dimension[1], 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dimension[1] * 6 * 6, last_layer_size)
        self.fc2 = nn.Linear(last_layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
# Define the path to the uploads folder
UPLOAD_FOLDER = 'static/uploads/'

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Empty the uploads folder
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        os.remove(file_path)

    # Get the uploaded images from the request
    uploaded_files = request.files.getlist('images')
    images_path = []
    # Save the uploaded files to the uploads folder
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        images_path.append(os.path.join(UPLOAD_FOLDER, filename))

    new_df = pd.DataFrame({'path': images_path})

    test_dataset = ImageDataset(metadata=new_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_outputs_epoch = []
    with torch.no_grad():
      for images in test_loader:
          images = images.to(device)
          outputs = model(images)
          total_outputs_epoch = total_outputs_epoch + list(outputs)

    predictions = torch.tensor(total_outputs_epoch).detach().cpu().numpy()
    formatted_predictions = ["{:f}".format(p) for p in predictions]
    image_paths_final = [path.replace('static/uploads/','image/') for path in images_path]
    final_df = new_df = pd.DataFrame({'File Name': image_paths_final, 'Prediction Score': formatted_predictions})
    final_df['Prediction Score'] = final_df['Prediction Score'].astype(float)
    final_df = final_df.sort_values(by='Prediction Score', ascending=False)
    # Convert final_df to an HTML table
    final_df.to_csv('final.csv', index=False)
    max_value = 0.5
    filtered_df = final_df[final_df['Prediction Score'] >= max_value]
    filtered_df = filtered_df.reset_index()
    filtered_df = filtered_df.drop(filtered_df.columns[0], axis=1)
    table_html = filtered_df.to_html(header=False)
    #return render_template('results.html', image_paths=images_path, results=predictions, table_html=table_html)
    return render_template('results.html', table_html=table_html, max_value=max_value)

@app.route('/filter_result', methods=['POST'])
def filter_result():
    # Get the maximum value from the form submission
    max_value = float(request.form.get('max_value'))

    final_df = pd.read_csv('final_df.csv', index_col=0)
    # Filter the WA_df based on the maximum value
    filtered_df = final_df[final_df['Prediction Score'] >= max_value]
    filtered_df = filtered_df.reset_index(drop=True)
    # Convert the filtered dataframe to HTML table
    table_html = filtered_df.to_html(header=False, index = False)

    return render_template('result.html', table_html=table_html, max_value=max_value)
if __name__ == '__main__':
    app.run(debug=True)

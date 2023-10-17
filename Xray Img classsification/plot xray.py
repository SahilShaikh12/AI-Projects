import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\user1\PycharmProjects\pythonProject\venv\runs\classify\train8\results.csv')
# Remove leading/trailing whitespaces and special characters
#df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check the available columns again
print("Available columns:", df.columns)

# Assuming your CSV has columns 'epoch' and 'accuracy', you can plot them
plt.plot(df['epoch'], df['accuracy1'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()

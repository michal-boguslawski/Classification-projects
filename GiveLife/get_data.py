import os
import requests

# Create the folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# URL of the raw data
url = "https://raw.githubusercontent.com/raspiani/Give-Life-Predict-Blood-Donations/master/datasets/transfusion.data"
response = requests.get(url)

# Save the file in the 'data' folder
with open("data/transfusion.data", "wb") as file:
    file.write(response.content)

print("Download complete! File saved to 'data/transfusion.data'")
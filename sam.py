import nltk
import os
import zipfile

# Set the correct download directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'backend', 'app', 'utils', 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

print(f"Downloading NLTK data to: {nltk_data_dir}")

# Download wordnet and omw-1.4
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

print("\nExtracting ZIP files...")

# Extract any zip files in corpora directory
corpora_dir = os.path.join(nltk_data_dir, 'corpora')
if os.path.exists(corpora_dir):
    for file in os.listdir(corpora_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(corpora_dir, file)
            print(f"Extracting {file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(corpora_dir)
            print(f"  ✓ Extracted {file}")
            # Optionally remove the zip file
            os.remove(zip_path)
            print(f"  ✓ Removed {file}")

print("\nVerifying extraction...")
# List the contents
for root, dirs, files in os.walk(nltk_data_dir):
    level = root.replace(nltk_data_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')

print("\nTesting lemmatizer...")
nltk.data.path.insert(0, nltk_data_dir)
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    test = lemmatizer.lemmatize("running")
    print(f"✓ Lemmatizer works! Test: 'running' -> '{test}'")
except Exception as e:
    print(f"✗ Lemmatizer failed: {e}")
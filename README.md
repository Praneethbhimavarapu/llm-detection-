 #LLM Memorization Detection System

This project helps you check if any text, image, audio, or document has been memorized by a Large Language Model (LLM).  
It compares your files with training data and tells you if something is copied or already seen before.

---

## What It Can Do

- Check text for memorization  
- Detect similar images  
- Compare PDFs and Word documents  
- Analyze audio files  
- Works fast using caching  
- Easy to use in Python or from command line

---

## How to Install

1. Clone the project
   ```bash
   git clone https://github.com/Praneethbhimavarapu/llm-detection.git
   cd llm-detection
   ```

2. Set up your Python environment
   ```bash
   python -m venv venv
   venv\Scripts\activate     # (Windows)
   # or
   source venv/bin/activate  # (Mac/Linux)
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use

### In Python
```python
from detector_system import EnhancedLLMMemorizationSystem

# Step 1: Create the system
detector = EnhancedLLMMemorizationSystem()

# Step 2: Load your training data folder
detector.load_training_data("data/training_data")

# Step 3: Test if something is memorized
result = detector.detect_memorization("The quick brown fox jumps over the lazy dog")

# Step 4: Show result
print(result.to_dict())
```

---

### From Command Line (Optional)
If you create a CLI file (like cli.py), you can do things like:

```bash
python cli.py "Check this text"
python cli.py --file document.pdf
```

---

## Folder Structure (Simplified)

```
llm-detection/
├── detector_system.py     # Main detection code
├── data/                  # Your training data files
├── training_cache.json    # Auto-created cache for speed
├── requirements.txt       # Required libraries
└── README.md              # This guide
```

---

## How It Works (Simple)

1. You give the system a folder of training data (text, PDFs, images, etc.).  
2. It learns what’s inside them and remembers it.  
3. When you test a new file or text, it checks if it matches or looks similar.  
4. It tells you if the content is new or memorized.

---

## License

This project is open source under the MIT License — you can use and modify it freely.
#

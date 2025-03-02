# 📄 RAG Document Q&A with Amazon Bedrock & Streamlit  

This is a **Retrieval-Augmented Generation (RAG) application** that allows users to ask **questions** about their **PDF documents** using **Amazon Bedrock** for LLM-powered responses.  
It retrieves **relevant document sections** and generates accurate answers.  

🚀 **Powered by:**  
- **Amazon Bedrock** (LLM Inference & Embeddings)  
- **FAISS** (Vector Store for efficient retrieval)  
- **Streamlit** (Frontend for easy interaction)  

---

## ✨ Features  
✅ **Ask questions about documents** (PDF files)  
✅ **Retrieves relevant document sections**  
✅ **Uses Amazon Bedrock LLM for accurate answers**  
---

## 🛠️ Setup Instructions  

### **1️⃣ Install AWS CLI**  
🔗 **[Download AWS CLI](https://aws.amazon.com/cli/)** and install it for your OS.  

### **2️⃣ Set Up AWS IAM Credentials**  
1. Go to the **AWS Console** → **IAM** → **Users**  
2. **Create a new IAM user** with programmatic access  
3. **Attach the following permissions**:
   - `AmazonBedrockFullAccess`
   - `AmazonS3FullAccess` (if using S3 for document storage)  
4. **Download** the **CSV file** containing **Access Key & Secret Key**  

### **3️⃣ Configure AWS Credentials**  
Run the following command in your terminal and enter the credentials from the CSV file:  
```sh
aws configure
```
 - **AWS Access Key ID**: ```<Enter from CSV>```
 - **AWS Secret Access Key**: ```<Enter from CSV>```
 - **Default Region Name**: ```us-east-1``` (or your AWS region)
 - **Default output format**: Just use default by pressing ```Enter``` on keyboard.


## 🔧 Project Installation

### **1️⃣ Create a Virtual Environment**
Run the following command using Conda:
```sh
conda create -p venv python==3.10 -y
```
Then activate the environment
```sh
conda activate venv/
```
### **2️⃣ Install Dependencies**
Inside the project folder, run:
```sh
pip install -r requirements.txt
```

## 📂 Prepare Your PDF Data
1. Create a ```data/``` folder in the root directory
2. Place all your PDF files inside this folder

## 🚀 Running the Application

### **1️⃣ Start the Streamlit App**
```sh
streamlit run main.py
```
This will open the streamlit web app in your browser at ```localhost:8501```

### **2️⃣ Generate Vector Store (Index PDFs)**
Once the app is running:
 - Click the **"Store Vectors"** button (Left sidebar)
 - Wait for **"Done"** message in a green box

### **3️⃣ Ask Questions!**
 - Enter a question in the **text input area**
 - Click **"Send Query"**
 - The model will **retrieve document information** and **generate an answer**
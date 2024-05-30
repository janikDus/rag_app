# rag_app
Simple API base on FAST API, which process given document line by line via Chroma, prepare prompt for ChatGPT and collect the response from model

Used python packages are defined in requirements.txt
install cmd:
pip install -r requirements.txt

How to run app default option:
uvicorn api_chroma_llm:app
- API will be avaiable on http://127.0.0.1:8000/process_task/, for testing can be used http://127.0.0.1:8000/docs/
For more options see https://www.uvicorn.org/

API parameters:
path_to_user_file - file to process is expected on same machine where the app is running (default - leave empty string than process file no_sale_countries.md)
chroma_query - define question for filter relevant lines from processed file (default - leave empty string than guestion is "What is the reason?")
gpt_query - define question for LLM model ChatGPT (default - leave empty string than guestion is "Waht is the common reason?")

Configuration options are in configs.py file:
ChatGPT - API key (need to be fill) and Model name (default: gpt-3.5-turbo)
Chroma - count of most relevant documents

# rag_app
Simple API base on FAST API, which process given document line by line via Chroma, prepare prompt for ChatGPT and collect the response from model

Used python packages are defined in requirements.txt
install cmd:
pip install -r requirements.txt

How to run app default option:
uvicorn api_chroma_llm:app
- API will be avaiable on http://127.0.0.1:8000/process_task/, for testing can be used http://127.0.0.1:8000/docs/
For more options see https://www.uvicorn.org/

'''
MS - Artificial Intelligence and Machine Learning
Course: CSC525 - Principles of Machine Learning
Module 8: Portfolio Project
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
May - June, 2026

Problem statement: 
Problem statement: Option #2: NLP Chatbot Project Final Version
Please submit your final NLP chatbot project in the form of an executable file or link allowing the instructor to chat with your chatbot. 
chatbot must meet the following conditions:
o	Use NLP learning methods to respond to user inputs.
o	Chatbot responses should not be nonsense.
o	Responses should be clearly in response to the input text.
Please include with your submission at least a page stating: 
  whether the chatbot is open-domain or closed, 
  what tools and libraries used, 
  and what type of NLP model is used in the chatbot, as well as 
  any instructions for running or accessing the chatbot. 

'''

import os
import html
import requests
import pandas as pd

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
#from myPersistChromaDb import ChromaDbPersist  # local class
from persistChromaDb_Utils import ChromaDbPersist




# Helper function.
# Clears the terminal
def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return

# Helper function.
# Checks is ollama running
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# reads input data file, cleanup data, create dataframe and return the dataframe.
def loadAndCleanupFileData(dataFile: str):
    if dataFile is None or len(dataFile.strip()) < 1:
        return    

    # Load your CSV file
    try:        
        df = pd.read_csv(dataFile)
    except Exception as e:
        print("data file load Error", e)
        exit(1)

    # Sample data (Salary_Data.csv) downloaded from the link
    """
    tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
    119237,105834,True,Wed Oct 11 06:55:44 +0000 2017,@AppleSupport causing the reply to be disregarded and the tapped notification under the keyboard is opened😡😡😡,119236,
    119238,ChaseSupport,False,Wed Oct 11 13:25:49 +0000 2017,"@105835 Your business means a lot to us. Please DM your name, zip code and additional details about your concern. ^RR https://t.co/znUu1VJn9r",,119239

    tweet_id,  author_id,  inbound,       created_at,                                  text,                                                                                                                                                                                   response_tweet_id,   in_response_to_tweet_id
    119237,   105834,       True,      Wed Oct 11 06:55:44 +0000 2017,   @AppleSupport causing the reply to be disregarded and the tapped notification under the keyboard is opened😡😡😡, 119236,
    """
    #print(df.head()) # ok

    # columns considered: tweet_id, created_at, text
    usedColumns = ['tweet_id', 'created_at', 'text']
    df2 = df[df.columns.intersection(usedColumns)]
    df2['created_at'] = pd.to_datetime(df2['created_at'], utc=True)
    df2.sort_values(by='created_at', ascending=True, inplace=True)
    # print(df2.head())  #ok
    df2['text'] = df2['text'].str.replace(r'^\S+\s*', '', regex=True)
    # print(df2.head())  #ok
    df2['text'] = df2['text'].str.replace(r'[\U00010000-\U0010ffff]', ' ', regex=True)  # remove imojis
    # print(df2.head())  #ok
    # First decode HTML entities like &amp; → &
    df2['text'] = df2['text'].apply(html.unescape)
    df2.reset_index(drop=True, inplace=True)
    #print(df2.head()) #ok
    # if needed, apply more data cleanup logic
    return df2

# This function creates the text embedding database and collection
# inputs:
#   dfInput -- dataframe, created from the input customer supprt sample.csv file
#   dbFile -- vector database name
#   collectionName -- vector data collection name
def createEmbeddingInVectorDb(dfInput: pd.DataFrame, dbFile: str, collectionName: str):
    if not isinstance(dfInput, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required_cols = {"tweet_id", "created_at", "text"}
    missing = required_cols - set(dfInput.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if dbFile is None or len(dbFile.strip()) < 1:
        print("Invalid Db file name")
        return

    if collectionName is None or len(collectionName.strip()) < 1:
        print("Invalid Collection name")
        return
    
    dbFile = dbFile.strip()
    collectionName = collectionName.lower().strip()
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")    

    vector_store = Chroma(
        collection_name=collectionName,
        persist_directory=dbFile,
        embedding_function=embeddings
    )

    # This helper function implemented in persistChromaDb_Utils.py
    # check if db present, collection has data , etc
    # from persistChromaDb_Utils import ChromaDbPersist # def __init__(self, dbFile: str):
    # def checkDbAndCollection(self, colltionName: str) -> int:
    checkChroma = ChromaDbPersist(dbFile)
    ret = checkChroma.checkDbAndCollection(collectionName)
    # inputs:
    #   collectionName: str = collection name
    # returns:
    #   0 = db, collection : both not exists
    #   1 = db exists, collectgion does not exist
    #   2 = db , collection : both exist, collection has no data
    #   3 = db , collection : both exist, collection has data
    #   99 = some error / exception

    # check if database and collection already populated
    add_documents = ret != 3  # don't override existing collection with data
    # add_documents = True

    if add_documents:   # Not to reload, if data already exists
        iids = []
        ddocs = []
        mmetadata = []
        for i, row in dfInput.iterrows():
            adoc = Document(page_content=row["text"])
            ddocs.append(adoc)
            iids.append(str(i))
            metadata={"date": str(row["created_at"])}
            mmetadata.append(metadata)
        if len(ddocs) > 0:
            vector_store.add_documents(documents=ddocs, ids=iids, metadatas=mmetadata)            
    return


# Application execution main entry point.
# it calls all above functions to execute the needed job for this project.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC525 - Principles of Machine Learning")
    print("Module 8: Portfolio Project")
    print("    Option #2: NLP ChatBot App using Ollama")
    print("\nMake sure ollama services: running. http://localhost:11434/api/tags")
    
    if is_ollama_running() == False:
        print("\nOllama services are not running. Run Ollama, then try again...")
        print("Running the Ollama services from the App itself is not implemented \n     because it may need different user account login.")
        exit(1)

    # input data source: https://csuglobal.instructure.com/courses/119239/assignments/2196864?module_item_id=6353788
    #  Customer Support Tweets on Twitter: https://www.kaggle.com/thoughtvector/customer-support-on-twitter
    #  locally downloaded and saved as file: sample.csv
    #  The data file: sample.csv has data for Apple iPhone customer supports
    # Please update the path or filename for your environment.    
    trainingDataFile: str = "./sample.csv"
    if os.path.exists(trainingDataFile) == False:
        print("Data file not found.")
        print("Please check and update Data file path.")
        exit(1)  # I should not proceed without raw data file.

    db_file: str = "./csc525_Portfolio_Project_chroma_db" # vector db
    collectionName: str = "customer_support"   # vector db collection

    # This helper function implemented in persistChromaDb_Utils.py
    # check if db present, collection has data , etc
    #   from persistChromaDb_Utils import ChromaDbPersist # def __init__(self, dbFile: str):
    #   def checkDbAndCollection(self, colltionName: str) -> int:
    checkChroma = ChromaDbPersist(db_file)
    ret = checkChroma.checkDbAndCollection(collectionName)
    # inputs:
    #   collectionName: str = collection name
    # returns:
    #   0 = db, collection : both not exists
    #   1 = db exists, collectgion does not exist
    #   2 = db , collection : both exist, collection has no data
    #   3 = db , collection : both exist, collection has data
    #   99 = some error / exception

    #exists = os.path.exists(db_file)
    exists: bool = ret != 3  # don't override existing collection with data
    # exists = False
    # print(ret, exists)
    if exists == False:
        # load raw data file(s) for the 'customer support' logs.
        # data source:  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
        #    The data file: sample.csv has data for Apple iPhone supports
        #    I've downloaded this file within my project scope.        
        dfRet = loadAndCleanupFileData(trainingDataFile)
        createEmbeddingInVectorDb(dfInput=dfRet, dbFile=db_file, collectionName=collectionName)
    
    # retriever = Chroma(collection_name=collectionName, persist_directory=db_file).as_retriever(search_kwargs={"k": 5}) # ok
    retriever = Chroma(collection_name=collectionName, persist_directory=db_file).as_retriever( search_type="similarity", search_kwargs={"k": 5} ) # ok
    # retriever = Chroma(collection_name=collectionName, persist_directory=db_file).as_retriever( search_kwargs={"k": 5, "fetch_k": 5} ) # Error
    # retriever = Chroma(collection_name=collectionName, persist_directory=db_file).as_retriever( search_type="mmr", search_kwargs={"k": 5} ) # Error
    if retriever is None:
        print("retver is None. Please check the vector database.")
        exit(1)
            
    model = OllamaLLM(model="llama3.2")
    template = """
    You are an expert in answering questions about Customer Support

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """
    
    # Following two lines of code: 
    # Converts our raw string template into a structured prompt object.
    # It knows how to fill in {reviews} and {question} when you later call the chain.
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    # Create Q & A loop
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question == "q":
            break
        
        # Sends the user’s question to our Chroma‑based retriever.
        reviews = retriever.invoke(question) 
        # The retriever then: Embeds the question, Searches the vector database
        # and returns the top‑k most similar documents (in our case, k=5).
        # reviews : now contains those retrieved documents—usually customer reviews, 
        #                               support logs, or other domain‑specific text.

        # The final step in the RAG pipeline: it runs the full prompt + model chain and produces the chatbot’s answer.
        # It performs two actions in sequence:
        # 1. Fill the prompt template: {reviews} is replaced with the retrieved documents, {question} is replaced with the user’s question.
        # 2. Send the completed prompt to the LLM: The Ollama model (llama3.2) generates an answer using both the question and the retrieved context.
        result = chain.invoke({"reviews": reviews, "question": question})
        # now, this 'result' variable contains the LLM’s final response — the chatbot’s answers.

        print(result)

# 
# (customer support logs) input data file: sample.csv
# used data columns: tweet_id, created_at, text
# dataset available at" https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
#
# Environment, Installs
# 1. Install Ollama services/runtimes: download and install from: https://ollama.com/
# 2. after installation, make sure ollama service running
#   (ping API endpoint)  http://localhost:11434/api/version
#     -- it should not fail or show error or 404 or localhost refused to connect, etc.
# C:\Windows\System32>ollama list 
#     -- above command will start the ollama services/runtimes, if it was not running.
#     -- show locally installed models.
#    (This may need the same user account that performed the installation or administrator privileges.)
#
# 3. Install Ollama models: 
#  3a. C:\Windows\System32>ollama pull llama3.2
#  3b. C:\Windows\System32>ollama pull mxbai-embed-large
# to check installed models, run: C:\Users\User1>ollama list
#   or
# (ping API endpoint) http://localhost:11434/api/tags
#
# --- needed pip installs ---
# (csc525) C:\Projs\Python\csc525>python.exe -m pip install --upgrade pip
# (csc525) C:\Projs\Python\csc525>pip install pandas requests
# (csc525) C:\Projs\Python\csc525>pip install langchain-core langchain langchain-ollama langchain-chroma
# (csc525) C:\Projs\Python\csc525>pip install --upgrade chromadb  # needed in: persistChromaDb_Utils.py
#
# to run  a model: C:\Windows\System32>ollama run llama3.2
# 
# If the model is updating/upgrading, it may show Error: upgrade in progress...
# 
# to execute the ChatBot App
# first run: ollama runtime/services:  C:\Users\User1>ollama list
# (csc525) C:\Projs\Python\csc525>python csc525_module8_portfolio_project.py
#
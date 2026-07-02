'''
MS - Artificial Intelligence and Machine Learning
Course: CSC525 - Principles of Machine Learning
Module 8: Portfolio Project
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
May - June, 2026

Problem statement: Option #2: NLP Chatbot Project Final Version

This is a helper class.

'''


import os
import chromadb

# Helper class for vector database
class ChromaDbPersist:
    def __init__(self, dbFile: str):
        self.dbFileName: str = ""
        if dbFile is not None:
            self.dbFileName = dbFile.strip()
        self.client: chromadb.api.client.PersistentClient = None
        self.embeddingFn: chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction = None                
        self.checkDbAndCollection("")

    # inputs:
    #   self.dbFileName: str # vector db file name
    #   collectionName: str = collection name
    # returns:
    #   0 = db, collection : both not exists
    #   1 = db exists, collectgion does not exist
    #   2 = db , collection : both exist, collection has no data
    #   3 = db , collection : both exist, collection has data
    #   99 = some error / exception
    def checkDbAndCollection(self, colltionName: str) -> int:
        if self.dbFileName is None or len(self.dbFileName) < 1:
            return 0        

        ret: int = 0
        try:            
            if os.path.exists(self.dbFileName) == False:
                return 0
            
            client = chromadb.PersistentClient(path=self.dbFileName)
            if client is not None:
                self.client = client
            
            ret = 1
            if colltionName is None or len(colltionName.strip()) < 1:
                return ret
            colltionName = colltionName.strip().lower()
            if colltionName in [c.name.strip().lower() for c in client.list_collections()]:
                ret = 2
            currentCollectionData = client.get_collection(colltionName)
            result = currentCollectionData.get(limit=1)
            if len(result["ids"]) > 0:
                ret = 3            
        except Exception as e:
            ret = 99
            print(f"Perist ChromaDb - error: {self.dbFileName}: {e}")
        
        return ret
    
#
# installs
# (csc525) C:\Projs\Python\csc525>pip install --upgrade chromadb
#
pip install openai==1.27
pip install langchain==0.1.19
pip install langchain-openai==0.1.6
pip install yt_dlp==2024.4.9
pip install tiktoken==0.6.0
pip install docarray==0.40.0

import os 
import glob
import openai 
import yt_dlp as youtube_dl
from yt_dlp import DownloadError 
import docarray 

client = openai.OpenAI(
    api_key="")



youtube_url = ""


output_dir = "files/audio/"


ydl_config = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
    "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
    "verbose": True
}



if not os.path.exists(output_dir): 
    os.makedirs(output_dir)


print(f"Downloading video from {youtube_url}")


try: 
    with youtube_dl.YoutubeDL(ydl_config) as ydl: 
        ydl.download([youtube_url])
except DownloadError: 
    with youtube_dl.YoutubeDL(ydl_config) as ydl: 
        ydl.download([youtube_url])




audio_files = glob.glob(os.path.join(output_dir, "*.mp3"))

audio_filename = audio_files[0]

print(audio_filename)



audio_file = audio_filename
output_file = "files/transcripts/transcript.txt"
model = "whisper-1"


print("converting audio to text...")


client = openai.OpenAI(api_key="")


with open(audio_file, "rb") as audio:
    
    response = client.audio.transcriptions.create(file=audio, model=model)


transcript = response.text


print(transcript)

os.makedirs(os.path.dirname(output_file), exist_ok=True)


with open(output_file, "w") as file:
    file.write(transcript)


from langchain.document_loaders import TextLoader


loader = TextLoader("./files/text")

docs = loader.load()

docs[0]


import tiktoken
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch


db = DocArrayInMemorySearch.from_documents(
    docs, 
    OpenAIEmbeddings()
)

retriever = db.as_retriever()


llm = ChatOpenAI(temperature = 0.0)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,            
    chain_type="stuff", 
    retriever=retriever,
    verbose=True        
)


query = "What is this tutorial about?"
response = qa_stuff.invoke(query)
response


query = "What is the difference between a training set and test set?"
response = qa_stuff.invoke(query)
response


query = "Who should watch this lesson?"
response = qa_stuff.invoke(query)
response 


query = "Who is the greatest football team on earth?"
response = qa_stuff.invoke(query)
response



query = "How long is the circumference of the earth?"
response = qa_stuff.invoke(query)
response 

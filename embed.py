from spacy.lang.en import English
import re
import pandas as pd
import requests

model = "all-minilm"

# nlp = English()
# nlp.add_pipe("sentencizer")

f = open('data.txt', 'r', encoding="utf8")
rawText = f.read()
f.close()

textDataArray = rawText.split("#-#-#")

# sentenceList = list(nlp(rawText).sents)
# textDataArray = [str(sentence) for sentence in sentenceList]

num_sentence_chunk_size = 1

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

sentenceChunks = split_list(input_list=textDataArray, slice_size=num_sentence_chunk_size)

joinedChunks = []
for sentence_chunk in sentenceChunks:
    joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
    joinedChunks.append(joined_sentence_chunk)

print("Total chunks "+str(len(joinedChunks)))

embeddings = []
i = 1
for sentence in joinedChunks:
    embedingData = {}
    res = requests.post("http://localhost:11434/api/embeddings", json={
        'model': model,
        'prompt': sentence
    })
    embedingData["embedding"] = (res.json())["embedding"]
    embedingData["sentence"] = sentence
    embeddings.append(embedingData)
    print("Completed chunks: "+str(i))
    i += 1

savePath = model+"_embeddings.csv"
pd.DataFrame(embeddings).to_csv(savePath, index=False)


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
# classifier=pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# r=classifier("they all hate us, let us fight them")


# generator=pipeline("text-generation",model="distilgpt2")
# r=generator("in this course, we will teach you how to",
#             max_length=50,
#             num_return_sequences=2,
#             )
classifier=pipeline("zero-shot-classification")
r=classifier("why we don't opens new websites, who knows if we got many followers, becuase of we are republicans",
             candidate_labels=["education","politics","business"],
             
             )
print(r)
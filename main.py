import os
import re
import json
import time
import emoji
import pandas as pd
from datetime import datetime
from openai import OpenAI
from underthesea import word_tokenize, pos_tag, ner

BATCH_SIZE = 15
SLEEP_TIME = 2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEENCODE = {"cf":"cà phê","lun":"luôn","nhg":"nhưng","mk":"mình","bn":"bạn",
"ko":"không","k":"không","dc":"được","vs":"với","cx":"cũng",
"ntn":"như thế nào","trc":"trước","bt":"bình thường","nt":"nhắn tin",
"ms":"mới","r":"rồi"}

ABBREVIATIONS = {"q1":"Quận 1","hn":"Hà Nội","hcm":"TP.HCM","tp":"thành phố"}

STOPWORDS = set(["là","của","và","thì","mà","ở","với","cho","đã","rồi"])
KEEP_WORDS = set(["rất","quá","lắm","cực","siêu","vô cùng","không","chẳng","chưa"])

PROPER_NOUNS = ["HIEUTHUHAI","Dương Lâm","Lamoon","Kiều Minh Tuấn"]

def clean_text(text):
    if pd.isna(text):
        return "", ""
    emojis = emoji.distinct_emoji_list(text)
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = re.sub(r'(.)\1+', r'\1', text)
    for k,v in TEENCODE.items():
        text = re.sub(rf'\b{k}\b', v, text)
    for k,v in ABBREVIATIONS.items():
        text = re.sub(rf'\b{k}\b', v, text)
    text = re.sub(r'!{2,}','!',text)
    text = re.sub(r'\?{2,}','?',text)
    text = re.sub(r'\.{4,}','...',text)
    text = re.sub(r'^(omg|lol|haha)\s*','',text)
    text = re.sub(r'\s*(omg|lol|haha)$','',text)
    text = re.sub(r'(\w)-(\w)',r'\1\2',text)
    text = " ".join(text.split())
    return text, ",".join(emojis)

def tokenize(text):
    return word_tokenize(text).split()

def remove_stopwords(tokens):
    return [t for t in tokens if (t not in STOPWORDS or t in KEEP_WORDS)]

def pos_tagging(text):
    return str(pos_tag(text))

def ner_tagging(text):
    return str(ner(text))

def wsd_rules(text):
    notes=[]
    if "đỉnh" in text: notes.append("đỉnh=positive")
    if "đắt" in text:
        if "giá" in text or "tiền" in text:
            notes.append("đắt=expensive")
        else:
            notes.append("đắt=selling well")
    if "hài" in text: notes.append("hài=funny")
    if "chết" in text: notes.append("chết=exclamation")
    return "; ".join(notes)

def analyze_batch(batch_texts,start_index):
    prompt="Analyze Vietnamese YouTube comments. Return ONLY JSON."
    content="\n".join(f"{i+start_index}: {c}" for i,c in enumerate(batch_texts))
    try:
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"ONLY JSON"},
                      {"role":"user","content":prompt+content}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        fallback=[]
        for i in range(len(batch_texts)):
            fallback.append({"index":i+start_index,"is_spam":False,
            "topic":[],"keywords":[],"overall_sentiment":"Neutral",
            "sentiment_score":0.0,"praised_topics":[],
            "criticized_topics":[],"ai_error":str(e)})
        return fallback

def main():
    df=pd.read_excel("youtube_comments.xlsx")
    df=df[(df["content"].str.len()>=50)&(df["content"].str.len()<=500)]
    df[["content_cleaned","emojis_detected"]]=df["content"].apply(lambda x:pd.Series(clean_text(x)))
    df["tokens"]=df["content_cleaned"].apply(tokenize)
    df["tokens_no_stopwords"]=df["tokens"].apply(remove_stopwords)
    df["tokens_lemmatized"]=df["tokens_no_stopwords"]
    df["pos_tags"]=df["content_cleaned"].apply(pos_tagging)
    df["named_entities"]=df["content_cleaned"].apply(ner_tagging)
    df["wsd_notes"]=df["content_cleaned"].apply(wsd_rules)

    results=[]
    texts=df["content_cleaned"].tolist()
    for i in range(0,len(texts),BATCH_SIZE):
        batch=texts[i:i+BATCH_SIZE]
        res=analyze_batch(batch,i)
        results.extend(res)
        time.sleep(SLEEP_TIME)

    ai_df=pd.DataFrame(results)
    df=pd.concat([df.reset_index(drop=True),ai_df],axis=1)

    pos=sum(df["overall_sentiment"]=="Positive")
    neg=sum(df["overall_sentiment"]=="Negative")
    con=sum(df["overall_sentiment"]=="Constructive")
    total=len(df)
    nss=(pos+con-neg)/total
    avg=df["sentiment_score"].mean()

    summary=pd.DataFrame({"Metric":["Positive","Negative","Constructive","Total","NSS","AvgScore"],
                          "Value":[pos,neg,con,total,nss,avg]})

    ts=datetime.now().strftime("%Y%m%d%H%M")
    with pd.ExcelWriter(f"analyzed_comments_{ts}.xlsx") as writer:
        df.to_excel(writer,"Analyzed Comments",index=False)
        summary.to_excel(writer,"NSS Summary",index=False)

if __name__=="__main__":
    main()

# Spacy-NER-skills-recognition
When you call `nlp` on a text, spaCy first tokenizes the text to produce a Doc object. The Doc is then processed in several steps– also referred to as the **processing pipeline**. The trained pipelines typically include a tagger, a lemmatizer, a parser, and an entity recognizer. Each pipeline component returns the processed Doc, which is then passed on to the next component.
![pipeline](https://github.com/Galal-pic/GD-Project/assets/70837846/a45ad63a-1176-4d45-b39b-df947d083f44)

## Building a Custom NER Model with SpaCy
* Named Entity Recognition (NER) is a subtask of information extraction that aims to identify and classify named entities such as names, organizations, locations, dates, and more. Although SpaCy provides a robust pre-trained NER model, there are situations where building a custom NER model from scratch becomes difficult, so in this project, we tuned the model to extract skills from job descriptions rather than names of people and organizations, This is done through some steps
  1. Create data suitable for training the model
  2. Dealing with a  pre-trained model to understand the English language, such as `en_core_web_sm`
  3. Retrain the model on this data
  4. Model Evaluation 

## 1. Create Data 
* We trained the model on about 300 examples, but the difficult task was how to annotate the data. Because we have the job description as income, but we do not have the skills that are within the description, this site helped us with this [NER Text Annotator](https://tecoholic.github.io/ner-annotator/)

## 2. pre-trained model `en_core_web_sm`
* `en_core_web_sm` is a small English model trained on web text. It is designed to be lightweight and fast, making it suitable for various NLP tasks such as tokenization, POS tagging, dependency parsing, and NER.
* One of the key features of en_core_web_sm is its ability to perform NER. NER is the task of identifying and classifying named entities mentioned in text into predefined categories such as persons, organizations, locations, etc. The model has been trained to recognize various types of named entities and can provide entity labels along with their spans in the input text.

## 3. Retrain the model
* After training the model on the data, it can extract skills from the job description
![3](https://github.com/Galal-pic/GD-Project/assets/70837846/8d182d21-677d-4a37-8054-9e00a921cb53)
## 4. Model Evaluation 

* We tested the model on 100 job descriptions and it showed us these results

1. **Precision** = 0.89 
2. **Recall** = 0.84
3. **F1 Score** = 0.86
I am satisfied with these results, but I will work on other models to extract more details and more features

  

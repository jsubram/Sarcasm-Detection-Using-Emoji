# Sarcasm-Detection-Using-Emoji
Sarcasm Detection Using Emoji and Text Analysis.


The dataset.xlsx file has the data stats and the data before and after preprocessing. 
I have taken the 12.9K input data (emoji + text) separately and put it in a csv file in the 'Data to work with' folder(zip). These are supposed to be the input files for word_to_vec and emoji_to_vec codes.

The coursework codes and report are in Project II Group 10 zip folder.

### **Dependencies**:
1) Numpy
2) Keras
3) NLTK
4) emoji
5) gensim
5) scikit-learn
6) pandas
7) codecs
8) enchant

### **Instructions to execute Basline model code**:

1) Place data in /Data directory and run: 
```
python Baseline/Run_ML.py
```
### **Instructions to execute LSTM + Attention code**:

Change file names in settings.json and run:
```
python Model/LSTM+Attention.py
```

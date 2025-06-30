from nltk.corpus import stopwords
import numpy as np
from functools import reduce 
import pandas as pd 
from spacy.tokens import Doc 
import wikipedia
from tqdm import tqdm
import joblib
import os 

def extract_div(inp):
    stop = stopwords.words('french')
    li = [x.lemma_ for x in inp if (not ((x.pos_ == "PUNCT") or (x.pos_ == "SPACE"))) and (not x.lemma_ in stop)] 
    un, count = np.unique(li,return_counts=True)
    if sum(count) == 0:
        return 0
    else:
        return len(un)/sum(count)
    
class Text(object):
    """A class that embed many tools to help processing a Spacy document to a 'stylistic' vector"""

    def __init__(self, 
                 text : Doc) -> None:
        
        self.text = text

    def prep_lev0(self) -> None:

        stop = stopwords.words('french')
        condition = lambda a : (not a.pos_ in ["PUNCT","SPACE"]) and (not a.lemma_ in stop)
        get_dim = lambda a : ["{} : {}".format(x,a.morph.get(x)) for x in ["Gender","Number","VerbForm","Voice","Mood","Person","Tense"]]
        self.df_mot = pd.DataFrame(index=range(len([x.text for x in self.text if condition(x)])),columns=["mots","pos","dep","tense","pers","mood","token","genre","nombre","verbform","voix"])
        self.df_mot.loc[:,"mots"] = [x.text for x in self.text if condition(x)]
        self.df_mot.loc[:,["pos","dep","token"]] = [["POS -"+x.pos_,"DEP - "+x.dep_,x.lemma_] for x in self.text if condition(x)]
        self.df_mot.loc[:,["genre","nombre","verbform","voix","mood","pers","tense",]] = [get_dim(x) for x in self.text if condition(x)]
        
    def compte(self,
               inp : pd.DataFrame) -> list:
        out = {}
        t = len(inp)
        
        # Diversité Linguistiques
        mot = inp.mots.unique()
        out["div_mot"] = len(mot)/t
        
        token = inp.token.unique()
        out["div_tok"] = len(token)/t
        
        # Frequence des qualité du discours
        for col in ["genre","nombre","verbform","voix","mood","pers","tense","pos","dep"]:
            uni,nbi = np.unique(inp[col],return_counts=True)
            for un,nb in zip(uni,nbi):
                out[un] = nb/t
        
        if t >= 25:
            return out
        else:
            return []
        
    def fen(self,
            taille : int,
            deplacement : int) -> pd.DataFrame:
        
        col_add = reduce(lambda a,b : list(a)+list(b),[self.df_mot[i].unique() for i in [x for x in self.df_mot.columns if not x in ["mots","token"]]])
        fenetres = [int(i) for i in range(0,len(self.df_mot),deplacement)]
        self.df_X = pd.DataFrame(index=range(len(fenetres)),columns= ["div_mot","div_tok"]+col_add)
        
        for i,j in enumerate(fenetres):
            if (j+taille < len(self.df_mot)) and (j < len(self.df_mot)):
                scrap = self.compte(self.df_mot.loc[j:j+taille,:])
            else:
                scrap = self.compte(self.df_mot.loc[j:,:])
            
            if len(scrap) > 0:
                self.df_X.loc[i,:] = [scrap[x] if x in scrap.keys() else 0 for x in self.df_X.columns]
            else:
                continue
        #print(self.df_X)
        #sop += 1
        self.df_X.drop(self.df_X.index[self.df_X.sum(1) == 0],inplace=True)
        
def texts2vectors(textes : list,
                  saving_path : str = "df_wiki.joblib") -> pd.DataFrame:
    """Function for text vectorization

    Args :
        textes : a list of textes

    Return
        A DataFrame encoding the vectors for each texts
    """

    import pandas as pd 
    import spacy
    from tqdm import tqdm 
    
    df_vectors = []
    nlp = spacy.load("fr_core_news_lg", disable=["ner"])

    for doc in tqdm(nlp.pipe(textes,batch_size=3)):
        #doc = nlp(texte)
        doc = Text(doc)
        doc.prep_lev0()
        doc.fen(50,50)
        df_vectors.append(doc.df_X)
    joblib.dump(df_vectors, saving_path)

    df_vectors = pd.DataFrame([df.sum(0) for df in tqdm(df_vectors)])
    return df_vectors


def wikipedia_extraction(articles_titles : list,
                         saving_path : str):

    if os.path.exists(saving_path):
        articles_text = joblib.load(saving_path)
    else:
        # Set the language to French
        wikipedia.set_lang("fr")

        # Dictionary to store article texts
        articles_text = {}

        for title in tqdm(articles_titles):
            try:
                # Fetch the full content of the article
                content = wikipedia.page(title).content
                articles_text[title] = content
                print(f"Successfully fetched: {title}")
            except wikipedia.DisambiguationError as e:
                print(f"Disambiguation error for '{title}': {e.options}")
            except wikipedia.PageError:
                print(f"Page not found: {title}")
            except Exception as e:
                print(f"Error fetching '{title}': {e}")

        # Now articles_text contains the plain text for each article
        # Example: print(articles_text["France"])

        joblib.dump(articles_text,"articles_wikipedia.joblib")

    return articles_text


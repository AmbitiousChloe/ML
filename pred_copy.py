import random
import pandas as pd
import numpy as np
import re

vocab = ["be", "friendship", "fries", "washington", "koch", "face", "nightmare", "thousand", "by", "keller", "death", "paradise", "lost", "nothing", "third", "melting", "daring", "satisfy", "designed", "cities", "marley", "destination", "colorful", "joseph", "skies", "beauty", "choice", "doing", "unity", "lively", "lived", "these", "its", "god", "disillusionment", "you", "cost", "warm", "here", "am", "o", "also", "parent", "ambitions", "written", "homes", "kissed", "breathtaking", "bird", "banks", "happy", "noisy", "experience", "anywhere", "perhaps", "universe", "stinson", "ought", "expensive", "kansas", "will", "keys", "kid", "restart", "birds", "redemption", "entire", "wildest", "game", "vie", "playing", "turn", "succession", "celebrations", "audition", "furious", "carried", "large", "inhabitants", "louvre", "seaside", "imagined", "while", "colors", "forest", "replies", "instantly", "tower", "carnival", "err", "finite", "ball", "letter", "landscapes", "gasoline", "miraculous", "reflected", "might", "ur", "played", "soulless", "industry", "building", "painting", "her", "location", "person", "historic", "exists", "berlin", "kind", "style", "cards", "musical", "rules", "haute", "on", "term", "wealthy", "desert", "crust", "belongs", "future", "setback", "tropical", "plait", "crowded", "mon", "ratatouille", "corrupts", "sins", "version", "with", "fans", "players", "vein", "cover", "everything", "exercise", "points", "olympia", "was", "black", "conservative", "parisian", "jungle", "natural", "olympic", "irving", "esque", "politics", "league", "the", "metropolitan", "held", "get", "ny", "oasis", "dark", "strikes", "en", "this", "authentic", "cool", "accountants", "city", "surface", "sun", "history", "only", "and", "awesome", "n", "escargot", "down", "general", "summer", "itself", "tumbrels", "whom", "stupid", "of", "unconditionally", "them", "cats", "loud", "job", "steven", "gang", "wonder", "where", "cant", "lots", "earth", "prince", "french", "confidently", "suite", "dog", "oui", "times", "summers", "revolution", "art", "going", "want", "sleepless", "trap", "zoo", "mind", "day", "chocolate", "bigger", "charles", "like", "walk", "poverty", "shallow", "abundant", "deep", "iconic", "driving", "bread", "harsh", "days", "care", "alive", "infinity", "oscar", "either", "constantly", "passion", "big", "multicultural", "short", "dying", "deserve", "book", "eats", "square", "marvelous", "wine", "swimming", "succumb", "hamilton", "brodsky", "founded", "felt", "that", "firm", "acquired", "next", "peaceful", "at", "later", "less", "landscape", "sands", "paced", "patisserie", "white", "time", "ninja", "america", "been", "bandolero", "fearless", "party", "ge", "in", "father", "who", "lasts", "harder", "ice", "ultramodern", "welcome", "more", "comfortable", "high", "parties", "cop", "lucky", "heist", "ll", "waitress", "devouring", "springing", "life", "croissant", "princes", "whimsical", "dictatorship", "second", "toto", "feel", "americans", "go", "does", "joanne", "smell", "turtles", "baby", "us", "patrick", "take", "economically", "die", "would", "hello", "question", "adventure", "eiffel", "shy", "share", "exchange", "fused", "source", "jealous", "wait", "country", "side", "junk", "vida", "sous", "house", "limit", "leader", "janeiro", "torch", "means", "labour", "active", "rose", "bikini", "delicate", "jesus", "ugliness", "your", "keep", "taxis", "night", "culturally", "so", "luck", "shall", "someone", "eyes", "calculated", "2", "fashionable", "denial", "higher", "doctor", "minutes", "routine", "z", "specific", "good", "gave", "worrying", "rather", "sunny", "resides", "energy", "years", "three", "raised", "suit", "drinks", "stock", "viva", "opportunity", "had", "main", "marvels", "club", "ride", "rotten", "name", "m", "parrots", "influential", "underbelly", "he", "sea", "or", "baguettes", "naughty", "found", "fun", "economy", "harris", "lives", "urban", "bad", "ted", "gain", "size", "vegas", "feels", "lavish", "businesswomen", "buildings", "architectural", "mean", "wish", "state", "new", "word", "concrete", "realize", "yes", "strength", "meant", "tiki", "aha", "mountain", "crime", "cup", "captured", "ends", "tang", "fall", "dorothy", "bailar", "says", "sell", "taxes", "idea", "head", "price", "gonna", "think", "clears", "economic", "ostriches", "glitz", "news", "something", "psg", "deal", "bedbugs", "pele", "toronto", "surrender", "long", "pizzas", "celebration", "sleep", "enough", "kelk", "flash", "hands", "tomorrow", "beautiful", "clan", "somewhat", "along", "disney", "luxury", "partying", "grow", "mixed", "don", "timeless", "prowess", "jr", "riches", "course", "helen", "believe", "wanted", "vacations", "souffle", "yeah", "ba", "once", "turned", "own", "walking", "hub", "rats", "just", "meet", "que", "fiesta", "died", "man", "tradition", "butterflies", "climate", "seen", "responsibility", "d", "downtown", "some", "pack", "ambition", "over", "things", "artificial", "hon", "fast", "dubai", "about", "member", "dream", "week", "already", "quote", "traffic", "but", "migrant", "annual", "even", "what", "divide", "may", "prosperity", "buy", "extreme", "world", "carts", "blood", "mona", "lose", "beyond", "silver", "dessert", "started", "haven", "burden", "an", "count", "to", "living", "la", "mi", "pot", "boundless", "moveable", "arts", "pick", "bed", "tyson", "carnivals", "make", "imagine", "centre", "companies", "toilettes", "son", "university", "lot", "cozy", "pense", "erase", "richness", "experienced", "no", "knowledgeable", "community", "tasting", "rich", "until", "petroleum", "hearts", "looking", "meaningful", "fear", "opulence", "give", "culture", "playground", "skyline", "isn", "power", "emily", "judge", "ere", "weather", "anyone", "lisa", "imagination", "aux", "joy", "divided", "mosby", "purpose", "amazing", "men", "jordan", "safety", "race", "streets", "great", "si", "capitalism", "lies", "manhattan", "doesn", "everywhere", "which", "stops", "together", "again", "cook", "five", "up", "stars", "soccer", "artist", "apart", "aller", "sunshine", "away", "naturalistic", "underground", "yourself", "inclusive", "monsters", "shaffer", "pizza", "airlines", "sibling", "my", "works", "really", "close", "superpowers", "sku", "hey", "goodbye", "drop", "jay", "self", "people", "could", "true", "tourism", "empire", "tell", "smart", "guests", "balanced", "destiny", "gas", "girl", "technology", "inspire", "dances", "loaf", "argument", "amigo", "brokerage", "football", "silence", "see", "skyscraper", "brazil", "vamos", "live", "horses", "coastal", "fulfilment", "puffy", "flashy", "rainy", "without", "trips", "jobs", "spirit", "everyone", "case", "islands", "spread", "riots", "york", "clothing", "vivid", "unlike", "redeemer", "artistic", "put", "ever", "appetit", "made", "play", "exploit", "pockets", "rhythms", "cristiano", "become", "yorkers", "finish", "lets", "area", "el", "tears", "bateman", "innovation", "du", "decorations", "yet", "despair", "dancing", "stepping", "walt", "serenity", "dreams", "based", "dude", "teaches", "nation", "nice", "say", "blessed", "jive", "start", "visit", "starts", "back", "allons", "nocturnal", "gather", "palm", "rely", "very", "stealing", "david", "sports", "rio", "mexicans", "there", "his", "music", "blue", "worst", "re", "follower", "bolder", "projects", "palaces", "tax", "reason", "inevitable", "fossil", "successful", "jul", "movies", "lower", "sera", "classmates", "games", "crazy", "needs", "six", "business", "rumble", "endless", "specifically", "pursuit", "direction", "faster", "color", "guillotine", "empty", "enter", "requires", "one", "were", "slavery", "rigorous", "pretty", "anymore", "whole", "round", "usa", "better", "candy", "tunnels", "affordable", "yellow", "when", "apartment", "myself", "stays", "disgusting", "famous", "happens", "most", "dance", "dirty", "bon", "moment", "use", "magnificent", "teenage", "khalifa", "senses", "hell", "peter", "full", "inspires", "arabic", "birth", "stone", "writes", "heads", "enthusiastic", "soy", "war", "flying", "samba", "busy", "soup", "realization", "trash", "existence", "barney", "roll", "any", "alicia", "top", "best", "each", "expectations", "charo", "builds", "differences", "seems", "other", "can", "carry", "heaven", "criticism", "hierarchy", "mood", "beach", "mans", "emptiness", "wolf", "shape", "sport", "stole", "taking", "money", "forget", "kent", "human", "cold", "slave", "element", "favelas", "symphony", "intersection", "happen", "hot", "breads", "discover", "apple", "worry", "is", "young", "dollars", "flow", "states", "gun", "rhythm", "designating", "real", "bonita", "immigrants", "clouds", "ah", "enthusiasm", "panda", "they", "than", "variety", "built", "steve", "poorer", "greatness", "sriram", "statue", "bullish", "passionate", "talk", "viral", "situations", "well", "bugs", "now", "inequality", "andrew", "hollow", "prosperous", "women", "al", "love", "hard", "hear", "pissed", "serves", "wealth", "antidote", "loca", "wins", "into", "queens", "salute", "amie", "must", "je", "another", "annie", "change", "social", "antiquated", "king", "piece", "hosted", "countries", "s", "de", "saying", "skyscrapers", "enjoy", "romans", "settle", "land", "baguette", "workers", "emirates", "paris", "particular", "out", "classical", "vive", "cheese", "amounts", "route", "hit", "early", "carefree", "mutant", "treasure", "coffee", "license", "theres", "developed", "parado", "me", "architecture", "however", "as", "views", "vision", "income", "ed", "arab", "infrastructure", "answer", "eat", "flags", "has", "anything", "lights", "thrives", "jackson", "line", "brain", "find", "metropolis", "wallet", "equality", "masked", "nightlife", "facade", "are", "oh", "tiny", "romance", "stomach", "light", "nature", "show", "paint", "touched", "million", "insatiate", "festival", "fancy", "asked", "modern", "bottom", "heart", "our", "professional", "exceeded", "dedication", "paintbrush", "many", "souls", "stub", "bonjour", "plane", "rains", "parody", "such", "wrapper", "lindsey", "way", "colourful", "wings", "ha", "im", "come", "for", "we", "morning", "meets", "please", "moments", "millions", "average", "guess", "outdated", "part", "lifestyle", "cannot", "safe", "makes", "cityscape", "much", "born", "exploitation", "c", "upon", "perfume", "thing", "from", "food", "mian", "belfort", "under", "rural", "grey", "tall", "cuisine", "thought", "system", "fuel", "psycho", "olympics", "ow", "lines", "grandmaster", "memory", "former", "fashion", "tourist", "class", "rest", "lincoln", "few", "alone", "christ", "none", "tech", "if", "michael", "happiness", "probably", "low", "fraternity", "middle", "impossible", "development", "watercolor", "relentless", "deserves", "she", "golden", "controls", "tourists", "wake", "fulfilled", "brasil", "memories", "gold", "know", "not", "cultures", "barely", "after", "hudson", "succeed", "a", "popular", "song", "items", "yalla", "shining", "comes", "quit", "since", "feast", "then", "connect", "vibrant", "classic", "tallest", "rome", "containing", "forests", "fullest", "honeymoon", "decisions", "fifa", "thus", "absolute", "symbol", "eurocentric", "never", "trade", "little", "exciting", "year", "sky", "reach", "keeps", "how", "subway", "france", "grew", "poor", "excellent", "sao", "trendy", "hui", "romantic", "every", "huge", "air", "waves", "possibilities", "slice", "master", "lama", "end", "ve", "goes", "luxurious", "making", "universes", "attraction", "celebrate", "maybe", "dalai", "gras", "feeling", "despite", "un", "work", "bob", "finds", "illegal", "build", "appears", "loving", "takes", "pay", "mafia", "sing", "today", "still", "sacrifice", "because", "distinguishes", "businessmen", "extremely", "two", "magic", "thoreau", "accept", "wu", "racist", "ford", "activities", "sculpture", "street", "testament", "r", "wilde", "presenting", "enduring", "between", "record", "always", "stories", "train", "historical", "diet", "est", "fly", "soul", "off", "cultural", "similarities", "waiting", "e", "anxious", "quotes", "though", "let", "poetry", "alright", "absolutely", "gets", "renowned", "enjoyed", "possibly", "festive", "all", "wherever", "subways", "understanding", "vacation", "i", "same", "help", "thinking", "liable", "home", "nights", "copious", "diversity", "nyc", "london", "sound", "tate", "wall", "abraham", "wearing", "feelings", "museum", "rent", "places", "brings", "first", "center", "roots", "pain", "turns", "barefoot", "look", "excellence", "fight", "floating", "sand", "elegance", "during", "overrated", "rude", "different", "jungles", "whispers", "vi", "bayside", "greed", "staying", "dont", "uh", "century", "colette", "possible", "have", "opportunities", "brand", "useless", "it", "beaches", "glamour", "richer", "exist", "taken", "plate", "embarrass", "futuristic", "spectacular", "familia", "capital", "slums", "continuous", "greatest", "place", "crunches", "old", "chef", "gigantic", "bought", "score", "wisdom", "gorgeous", "ch", "sings", "festivals", "afford", "property", "right", "liberty", "thanks", "dries", "mouth", "ashamed", "cars", "t", "else", "pen", "laid", "do", "geographical", "taka", "jai", "united", "seasons", "movie", "said", "failure", "filled", "henry", "oil", "sleeps", "within", "gustave", "scene", "gaston", "thereafter"]

weights_first_layer = np.loadtxt('NN_weights_1st_layer.txt')
bias_first_layer = np.loadtxt('NN_biases_1st_layer.txt')
weights_sec_layer = np.loadtxt('NN_weights_2nd_layer.txt')
bias_sec_layer = np.loadtxt('NN_biases_2nd_layer.txt')

# weights_first_layer = np.loadtxt('weights_first_layer.txt')
# bias_first_layer = np.loadtxt('bias_first_layer.txt')
# weights_sec_layer = np.loadtxt('weights_second_layer.txt')
# bias_sec_layer = np.loadtxt('bias_second_layer.txt')

def get_number_list(s):
    return [int(n) for n in re.findall("(\d+)", str(s))]


def get_number(s, na):
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else na


def cat_in_s(s, cat):
    return int(cat in s) if not pd.isna(s) else 0


def to_numeric(s):
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


def to_dict_complete(s):
    samples = s.split(",")
    result_dict = {}
    for sample in samples:
        key, value = sample.split('=>')
        if value == "":
            break
        result_dict[key.strip()] = int(value.strip())
    return result_dict

def insert_feature(nparray, vocab):
    features = np.zeros((nparray.shape[0], len(vocab)), dtype=float)
    for i in range(nparray.shape[0]):
        text = nparray[i, 3]
        words = set(re.sub(r"[^\w\s]", " ", text).lower().split())
        for j, word in enumerate(vocab):
            if word in words:
                features[i, j] = 1.0
    return features

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def process_data(filename: str):
    # read the data as pandas dataframe
    df = pd.read_csv(filename)

    # pre process Q1 - Q4
    na_14 = 2
    df["Q1"] = df["Q1"].apply(lambda x: get_number(x, na_14))
    df["Q2"] = df["Q2"].apply(lambda x: get_number(x, na_14))
    df["Q3"] = df["Q3"].apply(lambda x: get_number(x, na_14))
    df["Q4"] = df["Q4"].apply(lambda x: get_number(x, na_14))

    # fill all the missing values in Q1-Q4 colums as 2
    df["Q1"].fillna(na_14, inplace = True)
    df["Q2"].fillna(na_14, inplace = True)
    df["Q3"].fillna(na_14, inplace = True)
    df["Q4"].fillna(na_14, inplace = True)
    
    Q1_Q4_all_categories = [1, 2, 3, 4, 5]

    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
        df[col] = pd.Categorical(df[col], categories=Q1_Q4_all_categories)

    Q1_onehot = pd.get_dummies(df['Q1'], prefix='Q1', dtype=int)
    Q2_onehot = pd.get_dummies(df['Q2'], prefix='Q2', dtype=int)
    Q3_onehot = pd.get_dummies(df['Q3'], prefix='Q3', dtype=int)
    Q4_onehot = pd.get_dummies(df['Q4'], prefix='Q4', dtype=int)

    q5_category = ["Partner", "Friends", "Siblings", "Co-worker"]
    
    df["Q5"].fillna(random.choice(q5_category))

    for cat in q5_category:
        cat_name = f"{cat}"
        df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    # pre process Q6 as ONEHOT VECTOR
    # fill all the missing values in Q6 as mean (randomo choose 3 or 4)
    df["Q6"].fillna("Skyscrapers=>3,Sport=>3,Art and Music=>3,Carnival=>3,Cuisine=>3,Economic=>3",  inplace = True)
    df["Q6_Skyscr"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Skyscrapers", random.choice([3, 4])))
    df["Q6_Sport"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Sport", random.choice([3, 4])))
    df["Q6_AM"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Art and Music", random.choice([3, 4])))
    df["Q6_Carnival"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Carnival", random.choice([3, 4])))
    df["Q6_Cuisine"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Cuisine", random.choice([3, 4])))
    df["Q6_Eco"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Economic", random.choice([3, 4])))
    
    Q6_all_categories = [1, 2, 3, 4, 5, 6]
    for question in ['Q6_Skyscr', 'Q6_Sport', 'Q6_AM', 'Q6_Carnival', 'Q6_Cuisine', 'Q6_Eco']:
        df[question] = pd.Categorical(df[question], categories=Q6_all_categories)

    Q6sky_oh = pd.get_dummies(df['Q6_Skyscr'], prefix='Q6sky', columns=Q6_all_categories, dtype=int)
    Q6spt_oh = pd.get_dummies(df['Q6_Sport'], prefix='Q6spt', columns=Q6_all_categories, dtype=int)
    Q6am_oh = pd.get_dummies(df['Q6_AM'], prefix='Q6am', columns=Q6_all_categories, dtype=int)
    Q6carni_oh = pd.get_dummies(df['Q6_Carnival'], prefix='Q6carni', columns=Q6_all_categories, dtype=int)
    Q6cuis_oh = pd.get_dummies(df['Q6_Cuisine'], prefix='Q6cuis', columns=Q6_all_categories, dtype=int)
    Q6eco_oh = pd.get_dummies(df['Q6_Eco'], prefix='Q6eco', columns=Q6_all_categories, dtype=int)
    
    df["Q7"] = df["Q7"].apply(to_numeric)
    df["Q8"] = df["Q8"].apply(to_numeric)
    df["Q9"] = df["Q9"].apply(to_numeric)

    df["Q7"].fillna(df["Q7"].mean(), inplace = True)
    df["Q8"].fillna(df["Q8"].mean(), inplace = True)
    df["Q9"].fillna(df["Q9"].mean(), inplace = True)

    q7_min = -30
    q7_max = 45
    q89_min = 1
    q89_max = 15
    df.loc[(df['Q7'] < q7_min), 'Q7'] = q7_min
    df.loc[(df['Q7'] > q7_max), 'Q7'] = q7_max
    df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    df.loc[(df['Q8'] > q89_max), 'Q8'] = q89_max
    df.loc[(df['Q9'] < q89_min), 'Q9'] = q89_min
    df.loc[(df['Q9'] > q89_max), 'Q9'] = q89_max
    
    # normalizing
    df['Q7'] = (df['Q7'] - df['Q7'].mean()) / (df['Q7'].std() + 0.0001)
    df['Q8'] = (df['Q8'] - df['Q8'].mean()) / (df['Q8'].std() + 0.0001)
    df['Q9'] = (df['Q9'] - df['Q9'].mean()) / (df['Q9'].std() + 0.0001)

    # fixing na in Q10
    df['Q10'] = df['Q10'].fillna(" ").astype(str)
        
    # add extra colums to df
    df = pd.concat([df, Q1_onehot, Q2_onehot, Q3_onehot, Q4_onehot, Q6sky_oh, Q6spt_oh, Q6am_oh, Q6carni_oh, Q6cuis_oh, Q6eco_oh], axis=1)

    # delete non useful columns
    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', "Q6_Skyscr", "Q6_Sport", "Q6_AM", "Q6_Carnival", "Q6_Cuisine", "Q6_Eco"]
    for col in delete_columns:
        if col in df.columns:
           del df[col]
    X = df.values
    features = insert_feature(X, vocab)
    print(X.shape, features.shape, len(vocab))
    X_numeric = np.delete(X, 3, axis=1).astype(np.float64)
    X = np.hstack((X_numeric, features)).astype(np.float64)
    
    return X

def predict_all(file: str):
    # pre-process data
    data = process_data(file)
    layer_1 = np.dot(data, weights_first_layer) + bias_first_layer
    layer_1_relu = np.maximum(layer_1, 0)
    layer_2 = np.dot(layer_1_relu, weights_sec_layer) + bias_sec_layer
    predictions = softmax(layer_2)
    cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
    final_predictions = []
    for pred in predictions:
        pred_city = cities[np.argmax(pred)]
        final_predictions.append(pred_city)
    # using the quote to improve prediction result
    return final_predictions

if __name__ == "__main__":
    predictions = predict_all("clean_dataset.csv")
    Label = np.array(['Paris', 'Dubai', 'Paris', 'Dubai', 'Dubai', 'Rio de Janeiro', 'Rio de Janeiro', 'Dubai', 'Dubai', 'Dubai', 'Rio de Janeiro', 'New York City', 'Rio de Janeiro', 'Dubai', 'Paris', 'Paris', 'Rio de Janeiro', 'Paris', 'Dubai', 'New York City', 'Dubai', 'Rio de Janeiro', 'New York City', 'Dubai', 'New York City', 'Paris', 'Rio de Janeiro', 'Rio de Janeiro', 'Paris', 'Dubai', 'Paris', 'Paris', 'Rio de Janeiro', 'Rio de Janeiro', 'New York City', 'New York City', 'New York City', 'Paris', 'Dubai', 'New York City', 'Dubai', 'New York City', 'New York City', 'New York City', 'Dubai', 'Dubai', 'New York City', 'Rio de Janeiro', 'Rio de Janeiro', 'Paris', 'Rio de Janeiro', 'Dubai', 'Dubai', 'Paris', 'Paris', 'New York City', 'Dubai', 'Paris', 'Rio de Janeiro', 'Paris', 'Paris', 'Paris', 'New York City', 'Rio de Janeiro', 'Paris', 'Dubai', 'Dubai', 'Rio de Janeiro', 'Dubai', 'New York City', 'Paris', 'Dubai', 'Rio de Janeiro', 'Dubai', 'Paris', 'Dubai', 'Paris', 'Rio de Janeiro', 'New York City', 'New York City', 'Paris', 'Rio de Janeiro', 'Paris', 'Rio de Janeiro', 'New York City', 'New York City', 'Paris', 'Rio de Janeiro', 'Rio de Janeiro', 'Dubai', 'Dubai', 'Dubai', 'Paris', 'Rio de Janeiro', 'Dubai', 'Dubai', 'New York City', 'Rio de Janeiro', 'New York City', 'Rio de Janeiro'])
    mean_accuracy = np.mean(np.array(predictions) == Label)
    print(mean_accuracy)
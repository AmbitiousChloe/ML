import random
import pandas as pd
import numpy as np
import re

vocab = ["anything", "yourself", "bateman", "also", "taxis", "jules", "festive", "starts", "heaven", "appears", "fyi", "containing", "passionate", "banks", "vogue", "gustave", "statue", "master", "pizza", "memory", "eurocentric", "henry", "twice", "lot", "people", "adventure", "toto", "technology", "celebration", "deserve", "worthwhile", "exploitation", "whom", "west", "vivid", "butterflies", "experienced", "dj", "culturally", "turns", "land", "turned", "americans", "culture", "walking", "what", "loca", "rich", "pack", "dalai", "beaches", "show", "flying", "name", "white", "quotes", "seems", "many", "god", "downtown", "economic", "overrated", "shallow", "serenity", "face", "mutant", "into", "breads", "here", "soy", "cristiano", "tells", "news", "confidently", "rural", "worrying", "of", "dance", "how", "get", "true", "ho", "isn", "property", "third", "sculpture", "might", "disgusting", "arab", "at", "vida", "die", "imagined", "waves", "meet", "belongs", "mona", "spirit", "joy", "hello", "will", "most", "ch", "business", "early", "feels", "very", "ashamed", "bailar", "bakeries", "toilettes", "classical", "joseph", "light", "lovers", "square", "exercise", "vacation", "ever", "ha", "each", "capital", "crunches", "mean", "someone", "is", "tiki", "rotten", "france", "distinguishes", "charles", "designating", "succeed", "cold", "la", "mafia", "infinity", "time", "martial", "que", "king", "celebrations", "probably", "brazil", "nation", "violations", "richer", "famous", "about", "care", "over", "nothing", "liable", "times", "blessed", "takes", "stub", "man", "person", "baguette", "nice", "paintbrush", "down", "superpowers", "d", "copious", "drop", "enduring", "economy", "settle", "haven", "three", "only", "american", "roots", "goodbye", "enough", "it", "marvels", "tears", "situations", "short", "imagine", "am", "ambitions", "definitely", "five", "loss", "rigorous", "romans", "exchange", "tumbrels", "discover", "dirty", "health", "must", "cigarettes", "sunny", "labour", "everywhere", "asked", "summer", "fearless", "keeps", "ft", "waitress", "brings", "ll", "esque", "earth", "become", "large", "while", "equality", "exists", "empire", "millions", "higher", "self", "ny", "senses", "underbelly", "seaside", "horses", "poor", "were", "erase", "bandolero", "bottom", "parent", "apart", "cities", "century", "forests", "along", "active", "extravagant", "divide", "est", "something", "developed", "fast", "houses", "own", "lama", "forest", "trips", "look", "line", "wrenching", "bad", "born", "everything", "acquired", "coolest", "dine", "fancy", "patagonia", "stock", "because", "bright", "i", "ninja", "days", "weather", "night", "walk", "serves", "perhaps", "ugliness", "hard", "feeling", "poorer", "kansas", "colorful", "awesome", "after", "rhythms", "happens", "named", "sleeps", "even", "hell", "fifa", "use", "elegance", "concrete", "traffic", "different", "really", "grey", "mi", "puffy", "jealous", "sacrifice", "comfortable", "intersection", "differences", "gold", "architecture", "diversity", "rhythm", "furious", "coffee", "worry", "she", "dreams", "salute", "slave", "jul", "but", "to", "do", "world", "cover", "deserves", "safety", "whole", "trendy", "these", "builds", "put", "resides", "can", "hui", "feelings", "romantic", "poverty", "mian", "anymore", "summers", "cityscape", "stepping", "eyes", "multicultural", "filled", "air", "flash", "already", "cook", "doing", "gets", "his", "language", "new", "birth", "blank", "shall", "festivals", "guillotine", "realize", "amounts", "failure", "businesswomen", "sound", "restart", "high", "us", "wherever", "other", "ride", "belfort", "tomorrow", "now", "politics", "mixed", "tunnels", "tiny", "steve", "hon", "buy", "wait", "vein", "dollars", "facade", "york", "happened", "hit", "routine", "jose", "bought", "parties", "enjoys", "balanced", "right", "lucky", "subway", "died", "natural", "wildest", "men", "french", "okay", "aha", "jay", "founded", "building", "trade", "tourists", "best", "playground", "apartment", "sunshine", "stinson", "gasoline", "enjoy", "shape", "her", "mon", "lets", "lexicon", "hub", "thanks", "bayside", "ford", "extreme", "girl", "future", "saves", "ambition", "party", "buildings", "slavery", "league", "simple", "specifically", "sleep", "jackson", "pen", "used", "tradition", "expectations", "less", "projects", "keller", "helen", "views", "gonna", "torch", "pick", "lasts", "arabic", "err", "steven", "every", "renowned", "who", "lincoln", "live", "oasis", "lower", "businessmen", "loving", "fraternity", "places", "which", "janeiro", "moveable", "wrapper", "el", "audition", "big", "cup", "jungles", "spectacular", "played", "influential", "lazy", "emily", "zoo", "wealthy", "street", "state", "pense", "partying", "piece", "social", "marvelous", "immigrants", "vamos", "an", "vacations", "lines", "amie", "and", "crowded", "bird", "tourist", "ve", "color", "fries", "conservative", "variety", "beach", "possibilities", "charo", "course", "pay", "dont", "dances", "book", "jr", "home", "does", "rude", "take", "cheaper", "young", "coastal", "futuristic", "money", "heist", "jobs", "islands", "tall", "driving", "ratatouille", "huge", "golden", "go", "rumble", "overwatch", "croissant", "nature", "such", "tomato", "surface", "well", "possibly", "taken", "ball", "criticism", "none", "good", "fullest", "that", "prince", "history", "seasons", "size", "mood", "travel", "strength", "shy", "everyone", "then", "alicia", "purpose", "faster", "reflected", "stealing", "en", "lots", "teaches", "exceeded", "migrant", "impossible", "either", "always", "mosby", "secrets", "rose", "sandy", "me", "carnival", "price", "cars", "cheese", "sounds", "passion", "satisfy", "jesus", "power", "experience", "on", "play", "stone", "remember", "affordable", "work", "eats", "croissants", "with", "until", "re", "greatness", "khalifa", "aller", "rains", "thing", "music", "afford", "irving", "hey", "gang", "family", "reach", "ah", "so", "whimsical", "main", "morning", "heart", "jungle", "myself", "connect", "together", "iconic", "else", "artists", "brain", "calculated", "turn", "timeless", "bed", "class", "obvious", "skyline", "country", "within", "movies", "never", "instantly", "lies", "keys", "break", "believe", "glamour", "noisy", "little", "brand", "our", "aldo", "hear", "east", "poetry", "wonder", "emirates", "week", "perfume", "sing", "former", "word", "brodsky", "chef", "requires", "enthusiasm", "alright", "setback", "we", "richness", "things", "colors", "reason", "fun", "much", "going", "oscar", "laugh", "enthusiastic", "racist", "wake", "mans", "judge", "miracle", "cop", "whispers", "abraham", "fly", "sun", "saying", "pissed", "points", "vie", "bigger", "anxious", "lived", "rio", "pizzas", "job", "under", "cake", "next", "broke", "looking", "black", "treasure", "goes", "extremely", "find", "between", "quit", "roll", "not", "artificial", "two", "carry", "happy", "start", "si", "style", "record", "lisa", "dedication", "useless", "swimming", "thoreau", "harder", "just", "energy", "kent", "complicated", "clears", "birds", "ted", "gave", "hearts", "bon", "heads", "ends", "parrots", "million", "taka", "museum", "wisdom", "wine", "tyson", "may", "sorry", "sports", "geographical", "grow", "bonjour", "shining", "teenage", "art", "prosperous", "count", "answer", "skyscraper", "classmates", "ed", "for", "samba", "de", "model", "miraculous", "exist", "sky", "olympics", "paradise", "train", "plait", "cant", "ge", "fashionable", "round", "luxury", "liberty", "lavish", "surrender", "climate", "koch", "kelk", "insist", "historical", "x", "loaf", "oh", "during", "palaces", "build", "sao", "lives", "hudson", "fossil", "hamilton", "the", "mind", "life", "olympic", "limit", "inequality", "soulless", "warm", "love", "companies", "t", "worst", "doesn", "insatiate", "where", "this", "feast", "change", "bugs", "beyond", "long", "second", "happiness", "tower", "universes", "development", "knowledgeable", "makes", "yet", "grand", "sera", "up", "ice", "glitz", "sand", "women", "rainy", "bullish", "than", "grandmaster", "want", "later", "gas", "fall", "element", "still", "dude", "rent", "general", "professional", "ow", "wins", "endless", "real", "whitewashed", "skyscrapers", "guests", "like", "problems", "cozy", "one", "fear", "welcome", "wish", "progress", "tallest", "he", "je", "lights", "deal", "stays", "was", "turtles", "m", "found", "landscape", "dream", "touched", "david", "university", "magic", "please", "games", "clothing", "have", "powers", "floating", "oui", "ur", "great", "cool", "part", "eiffel", "however", "rather", "riches", "deep", "united", "number", "heights", "plate", "suit", "vibrant", "carnivals", "manhattan", "average", "sku", "imagination", "c", "streets", "fused", "anyone", "nocturnal", "enjoyed", "artistic", "s", "abundant", "barely", "unity", "since", "lively", "thrives", "amigo", "raised", "started", "come", "ici", "playing", "sport", "rome", "louvre", "barney", "masked", "crime", "2", "guess", "similarities", "again", "rats", "yeah", "particular", "devouring", "held", "finance", "area", "symphony", "jai", "hollow", "workers", "honeymoon", "cuisine", "thereafter", "springing", "bikini", "almost", "fiesta", "laid", "drinks", "trading", "trash", "kind", "no", "realization", "meets", "back", "slice", "stocks", "vests", "firm", "lindsey", "queens", "al", "monsters", "succession", "wearing", "smart", "food", "think", "share", "super", "inevitable", "freedom", "captured", "by", "say", "human", "itself", "any", "keep", "inspire", "thousand", "patrick", "ere", "inhabitants", "walt", "alive", "are", "colourful", "vegas", "cats", "first", "jordan", "brasil", "attraction", "you", "follower", "overly", "mouth", "greatest", "continuous", "sell", "destination", "olympia", "loud", "flashy", "wanted", "sriram", "my", "harsh", "unlike", "u", "movie", "paris", "understanding", "system", "busy", "angry", "cost", "waiting", "somewhat", "fuel", "psg", "stops", "un", "sands", "entire", "eagle", "few", "letter", "pockets", "they", "fulfilled", "wealth", "visit", "theres", "roof", "made", "outdated", "club", "burden", "viva", "forget", "meant", "has", "head", "suite", "artist", "junk", "beautiful", "n", "metropolis", "festival", "uh", "football", "challenges", "metropolitan", "magnificent", "feel", "psycho", "empty", "some", "in", "minutes", "du", "soccer", "musical", "embarrass", "wings", "moment", "innovation", "don", "another", "homes", "blood", "christ", "need", "barefoot", "hosted", "code", "maybe", "son", "yalla", "happen", "tourism", "better", "sings", "nyc", "desert", "sea", "actually", "dessert", "location", "yellow", "way", "denial", "greed", "kid", "tax", "revolution", "spread", "direction", "had", "inspires", "year", "disney", "redeemer", "there", "taxes", "years", "hands", "batman", "wall", "possible", "taking", "more", "fashion", "comes", "gather", "same", "z", "seen", "perfect", "capitalism", "im", "weak", "landscapes", "infrastructure", "cultures", "arts", "wilde", "excellent", "subways", "works", "ought", "bolder", "candy", "flags", "community", "slums", "urban", "song", "sins", "classic", "said", "end", "petroleum", "familia", "constantly", "specific", "las", "pretty", "moments", "carts", "wolf", "historic", "would", "quote", "authentic", "or", "meaningful", "khaled", "your", "destiny", "decorations", "as", "replies", "story", "stars", "raise", "parisian", "place", "today", "safe", "paint", "sucked", "stories", "riding", "thus", "off", "present", "apple", "parody", "mountain", "souls", "accountants", "states", "death", "appetit", "city", "know", "breathtaking", "from", "a", "accept", "talk", "term", "license", "game", "day", "washington", "aux", "og", "peaceful", "thought", "modern", "favelas", "income", "fans", "lost", "mexicans", "idea", "toronto", "old", "kissed", "dark", "tropical", "plane", "economically", "dorothy", "carefree", "pony", "michael", "blue", "tech", "vision", "dubai", "baguettes", "lifestyle", "gaston", "cannot", "fight", "proceeded", "member", "despite", "ocean", "responsibility", "diet", "patisserie", "living", "hot", "disillusionment", "middle", "memories", "gigantic", "be", "if", "all", "leader", "once", "panda", "could", "soul", "annual", "choice", "its", "see", "anywhere", "illegal", "stomach", "expensive", "e", "items", "oil", "finish", "writes", "romance", "making", "them", "wet", "give", "rest", "let", "palm", "six", "eat", "built", "luck", "vi", "when", "becoming", "clouds", "felt", "excellence", "finds", "center", "flow", "full", "delicate", "been", "worlds", "beauty", "princes", "confucius", "race", "cultural", "ostriches", "father", "low", "make", "brokerage", "war"]

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
    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', "Q6_Skyscr", "Q6_Sport", "Q6_AM", "Q6_Carnival", "Q6_Cuisine", "Q6_Eco"] # Edit Accordingly
    for col in delete_columns:
        if col in df.columns:
           del df[col]
    X = df.values
    features = insert_feature(X, vocab)
    print(X[:,3])
    X_numeric = np.delete(X, 3, axis=1).astype(np.float64)
    
    X = np.hstack((X_numeric, features)).astype(np.float64)
    X = np.concatenate((X[:, :3], X[:, 4:]), axis=1)
    
    
    
    return X

def predict_all(file: str):
    # pre-process data
    data = process_data(file)
    layer_1 = np.dot(data, weights_first_layer) + bias_first_layer
    layer_1_relu = np.maximum(layer_1, 0)
    layer_2 = np.dot(layer_1_relu, weights_sec_layer) + bias_sec_layer
    predictions = softmax(layer_2)
    cities = ['Dubai', 'Rio de Janeiro', 'New York City' ,'Paris']
    final_predictions = []
    for pred in predictions:
        pred_city = cities[np.argmax(pred)]
        final_predictions.append(pred_city)
    # using the quote to improve prediction result
    return final_predictions

if __name__ == "__main__":
    predictions = predict_all("example_test.csv")
    print(predictions)
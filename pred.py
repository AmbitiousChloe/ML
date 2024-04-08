import random
import pandas as pd
import numpy as np
import re

vocab = ['2', 'a', 'about', 'abraham', 'absolute', 'absolutely', 'accept', 'acquired', 'act', 'active', 'activities', 'actually', 'adventure', 'afford', 'affordable', 'after', 'again', 'against', 'ah', 'aha', 'air', 'airlines', 'al', 'aldo', 'alicia', 'alive', 'all', 'aller', 'almost', 'alone', 'already', 'alright', 'also', 'always', 'am', 'amazing', 'ambitions', 'america', 'american', 'americans', 'amie', 'amigo', 'amounts', 'an', 'and', 'andrew', 'angry', 'annie', 'annual', 'another', 'antiquated', 'anxious', 'any', 'anymore', 'anyone', 'anything', 'anywhere', 'apart', 'apple', 'arab', 'arabic', 'architectural', 'architecture', 'are', 'area', 'art', 'artificial', 'artist', 'artistic', 'arts', 'as', 'ashamed', 'asked', 'at', 'attraction', 'audition', 'authentic', 'aux', 'average', 'away', 'awesome', 'ba', 'baby', 'back', 'backed', 'bad', 'baguette', 'baguettes', 'bailar', 'bakeries', 'ball', 'bandolero', 'barely', 'barney', 'based', 'batman', 'be', 'beach', 'beaches', 'beautiful', 'beauty', 'because', 'become', 'becoming', 'bed', 'bedbugs', 'been', 'believe', 'belongs', 'berlin', 'best', 'better', 'between', 'beyond', 'big', 'bigger', 'bikini', 'bird', 'birds', 'birth', 'black', 'blank', 'blue', 'bob', 'bonita', 'bonjour', 'book', 'born', 'bought', 'boundless', 'bradshaw', 'brain', 'brand', 'brasil', 'brazil', 'breads', 'break', 'breathtaking', 'bright', 'brings', 'broke', 'bugs', 'build', 'building', 'buildings', 'builds', 'built', 'bullish', 'burden', 'business', 'busy', 'but', 'butterflies', 'buy', 'by', 'c', 'cake', 'can', 'candy', 'cannot', 'capital', 'capitalism', 'captured', 'cards', 'care', 'carefree', 'carnival', 'carnivals', 'carrie', 'cars', 'case', 'cats', 'celebrate', 'celebration', 'celebrations', 'center', 'centre', 'century', 'ch', 'challenges', 'chaotic', 'charles', 'cheaper', 'cheese', 'chef', 'choice', 'chosen', 'christ', 'cigarettes', 'cities', 'city', 'cityscape', 'clan', 'class', 'classic', 'classical', 'classmates', 'clears', 'climate', 'close', 'clothing', 'clouds', 'club', 'coastal', 'coffee', 'cold', 'colette', 'color', 'colorful', 'colors', 'colourful', 'come', 'comes', 'comfortable', 'community', 'companies', 'complicated', 'concrete', 'confucius', 'connect', 'conservative', 'constantly', 'containing', 'continuous', 'controls', 'cook', 'cool', 'cop', 'copious', 'corrupts', 'cost', 'count', 'countries', 'country', 'court', 'cover', 'cozy', 'crime', 'criticism', 'croissant', 'croissants', 'crossroads', 'crowded', 'cuisine', 'cultural', 'culturally', 'culture', 'cultures', 'cup', 'd', 'dalai', 'dance', 'dances', 'dancing', 'daring', 'dark', 'day', 'days', 'de', 'deal', 'death', 'decisions', 'dedication', 'deep', 'definitely', 'denial', 'desert', 'deserve', 'deserves', 'designating', 'designed', 'despite', 'dessert', 'destination', 'developed', 'development', 'dictatorship', 'die', 'died', 'differences', 'different', 'dine', 'dirty', 'disgusting', 'disillusionment', 'disney', 'distinguishes', 'diversity', 'divide', 'divided', 'dj', 'do', 'doctor', 'does', 'doesn', 'dollars', 'don', 'dont', 'dorothy', 'down', 'downtown', 'dream', 'dreams', 'dries', 'drinks', 'driving', 'drop', 'du', 'dubai', 'dude', 'during', 'dying', 'e', 'each', 'eagle', 'early', 'earth', 'east', 'eat', 'eats', 'economic', 'economically', 'economy', 'ed', 'eiffel', 'either', 'el', 'elegance', 'else', 'embarrass', 'emily', 'emirates', 'empire', 'empty', 'en', 'end', 'endless', 'ends', 'enduring', 'energy', 'enjoy', 'enjoyed', 'enjoying', 'enjoys', 'enough', 'enter', 'enthusiasm', 'entire', 'equality', 'erase', 'esque', 'est', 'eurocentric', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'exceeded', 'excellence', 'excellent', 'exchange', 'exciting', 'exist', 'exists', 'expectations', 'expensive', 'experience', 'exploit', 'exploitation', 'extravagant', 'extremely', 'eyes', 'facade', 'fact', 'failure', 'fall', 'familia', 'family', 'famous', 'fancy', 'fans', 'fashion', 'fast', 'faster', 'father', 'favelas', 'fear', 'fearless', 'feast', 'feel', 'feeling', 'feels', 'felt', 'festival', 'festivals', 'festive', 'few', 'fifa', 'fight', 'filled', 'finance', 'find', 'finds', 'finish', 'finite', 'first', 'five', 'flags', 'flash', 'flashy', 'floating', 'flow', 'fly', 'flying', 'follower', 'food', 'football', 'for', 'ford', 'forest', 'forests', 'forget', 'fossil', 'found', 'founded', 'france', 'fraternity', 'freedom', 'french', 'frenchman', 'friendship', 'fries', 'from', 'ft', 'fuel', 'fulfilled', 'fulfilment', 'full', 'fullest', 'fun', 'furious', 'future', 'futuristic', 'fyi', 'gain', 'game', 'games', 'gang', 'garbage', 'gas', 'gasoline', 'gaston', 'gather', 'gathers', 'gave', 'ge', 'general', 'get', 'gets', 'gigantic', 'girl', 'give', 'glamour', 'glitters', 'glitz', 'go', 'god', 'goddamn', 'goes', 'going', 'gold', 'golden', 'gonna', 'good', 'goodbye', 'gorgeous', 'grand', 'grandmaster', 'great', 'greatest', 'greatness', 'greed', 'grew', 'grey', 'grow', 'guess', 'guests', 'gun', 'gustave', 'ha', 'had', 'hamilton', 'hands', 'happen', 'happened', 'happens', 'happiness', 'happy', 'hard', 'harder', 'harris', 'harvey', 'has', 'haute', 'have', 'haven', 'he', 'hear', 'heart', 'heaven', 'heights', 'heist', 'helen', 'hell', 'hello', 'help', 'henry', 'her', 'here', 'hey', 'hierarchy', 'high', 'higher', 'his', 'historical', 'history', 'ho', 'home', 'homes', 'hon', 'honeymoon', 'hope', 'horses', 'hot', 'house', 'houses', 'how', 'however', 'hub', 'hudson', 'huge', 'hui', 'i', 'ici', 'iconic', 'idea', 'if', 'il', 'impossible', 'in', 'inclusive', 'income', 'industry', 'inevitable', 'infinity', 'influential', 'infrastructure', 'inhabitants', 'innovation', 'insist', 'inspire', 'inspires', 'instantly', 'intersection', 'into', 'irving', 'is', 'islands', 'isn', 'it', 'its', 'itself', 'jackson', 'jai', 'janeiro', 'jay', 'je', 'jealous', 'jesus', 'jive', 'joanne', 'job', 'jobs', 'jordan', 'jose', 'joy', 'jr', 'judge', 'jules', 'jungle', 'jungles', 'junk', 'just', 'kansas', 'keep', 'keeps', 'kelk', 'keller', 'kent', 'keys', 'khaled', 'khalifa', 'king', 'kissed', 'know', 'knowledgeable', 'koch', 'la', 'labour', 'laid', 'lama', 'land', 'landmarks', 'landscape', 'landscapes', 'language', 'las', 'lasts', 'later', 'laugh', 'lavish', 'lazy', 'leader', 'learn', 'less', 'let', 'lets', 'letter', 'lexicon', 'liable', 'liberty', 'license', 'lies', 'life', 'lifestyle', 'light', 'lights', 'like', 'limit', 'lincoln', 'lindsey', 'line', 'lines', 'lisa', 'little', 'live', 'lived', 'lively', 'lives', 'living', 'll', 'loca', 'location', 'london', 'long', 'look', 'looking', 'lose', 'loss', 'lost', 'lot', 'lots', 'loud', 'louvre', 'love', 'loved', 'lovers', 'loving', 'low', 'lower', 'luck', 'lucky', 'luxurious', 'luxury', 'm', 'made', 'magic', 'main', 'make', 'makes', 'making', 'man', 'mans', 'many', 'mark', 'marley', 'marvel', 'marvelous', 'marvels', 'masked', 'master', 'may', 'maybe', 'me', 'mean', 'meaningful', 'meet', 'meets', 'memorable', 'memory', 'men', 'mere', 'metropolis', 'metropolitan', 'mexicans', 'mi', 'mian', 'michael', 'middle', 'might', 'migrant', 'million', 'millions', 'mind', 'minutes', 'miracle', 'miraculous', 'mixed', 'modern', 'moment', 'moments', 'mon', 'mona', 'money', 'monsieur', 'mood', 'more', 'mosby', 'most', 'mountain', 'mouth', 'moveable', 'movie', 'movies', 'much', 'multicultural', 'museum', 'music', 'musical', 'must', 'mutant', 'my', 'n', 'named', 'natural', 'naturalistic', 'nature', 'naughty', 'need', 'needs', 'never', 'new', 'next', 'nice', 'night', 'nightlife', 'nightmare', 'nights', 'ninja', 'no', 'nocturnal', 'noisy', 'none', 'not', 'nothing', 'novo', 'now', 'number', 'ny', 'nyc', 'o', 'oasis', 'ocean', 'of', 'oh', 'oil', 'okay', 'old', 'olympia', 'olympic', 'olympics', 'on', 'once', 'one', 'only', 'opportunities', 'opportunity', 'or', 'oscar', 'ostriches', 'other', 'ought', 'oui', 'our', 'outdated', 'over', 'overly', 'overrated', 'overwatch', 'ow', 'own', 'paced', 'paintbrush', 'painting', 'palm', 'panda', 'paradise', 'parado', 'parent', 'paris', 'parody', 'parrots', 'part', 'particular', 'parties', 'party', 'partying', 'passion', 'passionate', 'patagonia', 'patisserie', 'peaceful', 'pele', 'pen', 'pense', 'people', 'perfect', 'perfume', 'perhaps', 'person', 'peter', 'petroleum', 'pick', 'piece', 'pizza', 'pizzas', 'place', 'places', 'plait', 'plate', 'play', 'played', 'playground', 'playing', 'points', 'politics', 'pony', 'poor', 'poorer', 'popular', 'possible', 'possibly', 'poverty', 'power', 'powers', 'present', 'presenting', 'prince', 'princes', 'probably', 'problems', 'proceeded', 'progress', 'prosperity', 'prosperous', 'prowess', 'psg', 'psycho', 'purpose', 'pursuit', 'put', 'que', 'quote', 'quotes', 'r', 'race', 'racist', 'rains', 'rainy', 'raise', 'ratatouille', 'rats', 're', 'reach', 'real', 'reality', 'realize', 'really', 'redeemer', 'redemption', 'reflected', 'relentless', 'remember', 'renowned', 'rent', 'replies', 'requires', 'responsibility', 'rest', 'restart', 'revolution', 'rhythm', 'rhythms', 'rich', 'richer', 'riches', 'richest', 'richness', 'ride', 'riding', 'right', 'rio', 'roll', 'romance', 'romans', 'romantic', 'rome', 'roof', 'roots', 'rose', 'round', 'route', 'rules', 'rural', 's', 'sacrifice', 'safe', 'safety', 'said', 'salute', 'samba', 'same', 'sand', 'sands', 'sandy', 'satisfy', 'saves', 'say', 'saying', 'says', 'scene', 'science', 'score', 'sculpture', 'sea', 'seaside', 'seasons', 'secrets', 'see', 'seen', 'self', 'sell', 'sera', 'serenity', 'serves', 'setback', 'settle', 'shaffer', 'shall', 'shape', 'share', 'she', 'shining', 'short', 'show', 'si', 'side', 'silver', 'similarities', 'simple', 'sing', 'sings', 'sins', 'situations', 'size', 'skies', 'sku', 'sky', 'skyline', 'skyscraper', 'skyscrapers', 'slave', 'slavery', 'sleep', 'sleeps', 'slice', 'slums', 'so', 'soccer', 'social', 'some', 'somehow', 'someone', 'something', 'somewhat', 'son', 'song', 'sorry', 'soul', 'soulless', 'sound', 'sounds', 'source', 'sous', 'soy', 'specific', 'specifically', 'specter', 'spirit', 'sport', 'sports', 'spread', 'springing', 'square', 'sriram', 'stars', 'state', 'states', 'statue', 'staying', 'stays', 'stealing', 'stepping', 'steve', 'steven', 'still', 'stinson', 'stock', 'stocks', 'stole', 'stone', 'stops', 'stories', 'street', 'streets', 'strength', 'strikes', 'stupid', 'style', 'subway', 'subways', 'succeed', 'successful', 'succession', 'such', 'suit', 'suite', 'summer', 'summers', 'sun', 'sunny', 'sunshine', 'super', 'superpowers', 'surface', 'surrender', 'surrounded', 'swift', 'symbol', 'symphony', 'system', 't', 'taka', 'take', 'taken', 'takes', 'talk', 'tall', 'tallest', 'tang', 'tate', 'tax', 'taxes', 'taylor', 'tears', 'tech', 'technology', 'ted', 'teenage', 'tell', 'tennis', 'term', 'testament', 'than', 'thanks', 'that', 'the', 'them', 'then', 'there', 'thereafter', 'theres', 'these', 'they', 'thing', 'things', 'think', 'thinking', 'third', 'this', 'thought', 'thousand', 'thrives', 'thus', 'tiki', 'time', 'timeless', 'times', 'to', 'today', 'together', 'toilettes', 'tomato', 'tomorrow', 'top', 'torch', 'toronto', 'toto', 'touched', 'tourism', 'tourist', 'tourists', 'tower', 'trade', 'trading', 'tradition', 'train', 'trap', 'trash', 'travel', 'treasure', 'trips', 'tropical', 'true', 'tunnels', 'turn', 'turns', 'turtles', 'twice', 'two', 'tyson', 'u', 'ugliness', 'uh', 'ultramodern', 'um', 'un', 'under', 'underbelly', 'underground', 'understand', 'understanding', 'une', 'united', 'unity', 'universes', 'university', 'until', 'up', 'upon', 'ur', 'urban', 'us', 'use', 'used', 'useless', 'vacation', 'vacations', 'vamos', 'variety', 've', 'vegas', 'vein', 'version', 'very', 'vests', 'vi', 'vibrant', 'vida', 'vie', 'vision', 'visit', 'viva', 'vive', 'vivid', 'wait', 'waiting', 'wake', 'walk', 'walking', 'wall', 'wallet', 'walt', 'want', 'wanted', 'wants', 'war', 'warm', 'was', 'watercolor', 'waves', 'way', 'we', 'weak', 'wealth', 'wealthy', 'wearing', 'weather', 'welcome', 'well', 'west', 'wet', 'what', 'when', 'where', 'wherever', 'which', 'while', 'whimsical', 'whispers', 'white', 'whitewashed', 'who', 'whom', 'wilde', 'wildest', 'will', 'willing', 'windy', 'wine', 'wings', 'wins', 'wisdom', 'wish', 'with', 'within', 'women', 'wonder', 'word', 'work', 'workers', 'works', 'world', 'worlds', 'worm', 'worry', 'worrying', 'worst', 'worthwhile', 'would', 'wrapper', 'wrenching', 'writes', 'written', 'wu', 'x', 'yalla', 'yeah', 'year', 'years', 'yes', 'yet', 'york', 'you', 'young', 'your', 'yourself', 'z', 'zoo']

# weights_first_layer = np.loadtxt('NN_weights_1st_layer.txt')
# bias_first_layer = np.loadtxt('NN_biases_1st_layer.txt')
# weights_sec_layer = np.loadtxt('NN_weights_2nd_layer.txt')
# bias_sec_layer = np.loadtxt('NN_biases_2nd_layer.txt')

weights_first_layer = np.loadtxt('weights_first_layer.txt')
bias_first_layer = np.loadtxt('bias_first_layer.txt')
weights_sec_layer = np.loadtxt('weights_second_layer.txt')
bias_sec_layer = np.loadtxt('bias_second_layer.txt')

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
    layer_1 = np.dot(data, weights_first_layer.T) + bias_first_layer
    layer_1_relu = np.maximum(layer_1, 0)
    layer_2 = np.dot(layer_1_relu, weights_sec_layer.T) + bias_sec_layer
    predictions = softmax(layer_2)
    cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
    final_predictions = []
    for pred in predictions:
        pred_city = cities[np.argmax(pred)]
        final_predictions.append(pred_city)
    # using the quote to improve prediction result
    return final_predictions

if __name__ == "__main__":
    predictions = predict_all("example_test.csv")
    Label = np.array(['Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai'])
    mean_accuracy = np.mean(np.array(predictions) == Label)
    print(mean_accuracy)
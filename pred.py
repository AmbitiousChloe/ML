import random
import pandas as pd
import numpy as np
import re

vocab = ['0', '000', '01', '1', '100', '110', '1th', '2', '2016', '21st', '27', '3', '30', '5', '6', '72', 'a', 'about', 'abraham', 'accept', 'acquired', 'act', 'active', 'activities', 'actually', 'adventure', 'afford', 'affordable', 'after', 'again', 'against', 'ah', 'aha', 'air', 'airlines', 'al', 'aldo', 'alicia', 'alive', 'all', 'aller', 'allons', 'almost', 'alone', 'alot', 'already', 'alright', 'also', 'always', 'am', 'amazing', 'ambition', 'ambitions', 'america', 'american', 'americans', 'amie', 'amigo', 'amounts', 'an', 'and', 'andrew', 'angery', 'angry', 'annual', 'another', 'answer', 'antiquated', 'anxious', 'any', 'anymore', 'anyone', 'anything', 'anywhere', 'appears', 'appetit', 'apple', 'appleâ', 'arab', 'arabic', 'architectural', 'architecture', 'are', 'area', 'art', 'artificial', 'artist', 'artistic', 'artists', 'arts', 'as', 'ashamed', 'asked', 'at', 'attraction', 'aunque', 'authentic', 'aux', 'average', 'away', 'awayâ', 'awesome', 'ayy', 'ba', 'baby', 'back', 'backed', 'bad', 'baguette', 'baguettes', 'baguetteâ', 'bailao', 'bailar', 'bailã', 'balanced', 'ball', 'bandolero', 'banks', 'barely', 'barney', 'based', 'bateman', 'batman', 'be', 'beach', 'beaches', 'beautiful', 'beauty', 'because', 'become', 'becoming', 'bedbugs', 'been', 'believe', 'belongs', 'best', 'better', 'between', 'beyond', 'beyondâ', 'big', 'bigger', 'bigâ', 'bikini', 'bird', 'birds', 'birth', 'black', 'blessed', 'blu', 'blue', 'bob', 'bon', 'bonita', 'bonjour', 'book', 'born', 'bought', 'boundless', 'bradshaw', 'brain', 'brand', 'brasilâ', 'brazil', 'bread', 'breads', 'break', 'breathtaking', 'bright', 'brings', 'brodsky', 'broke', 'brothas', 'build', 'building', 'buildings', 'builds', 'built', 'bullish', 'burj', 'business', 'businessmen', 'businesswomen', 'busy', 'but', 'butterflies', 'buy', 'by', 'c', 'calculated', 'can', 'candy', 'cannot', 'cant', 'canâ', 'capital', 'capitalism', 'captured', 'care', 'carefree', 'carnaval', 'carnival', 'carnivale', 'carnivals', 'carrie', 'cars', 'case', 'cats', 'celebrate', 'celebration', 'celebrations', 'center', 'centre', 'century', 'ch', 'challenges', 'change', 'chaois', 'charles', 'charo', 'chbosky', 'cheaper', 'cheese', 'chef', 'chocolate', 'choice', 'chosen', 'christ', 'cigarettesâ', 'cities', 'city', 'cityscape', 'cityâ', 'clan', 'class', 'classic', 'classical', 'classmates', 'clears', 'climate', 'close', 'clothing', 'clouds', 'coastal', 'code', 'coffee', 'cold', 'colette', 'color', 'colorful', 'colors', 'colourful', 'come', 'comes', 'comfortable', 'community', 'companies', 'concrete', 'confidently', 'connect', 'constantly', 'containing', 'continuous', 'cook', 'cool', 'coolest', 'cop', 'copious', 'coreta', 'cost', 'count', 'countries', 'country', 'court', 'cover', 'cozy', 'crazy', 'crime', 'cristo', 'criticism', 'croissan', 'croissant', 'croissants', 'crossroads', 'crowded', 'crunches', 'crust', 'cuisine', 'cultural', 'culturally', 'culture', 'cup', 'd', 'dalai', 'dance', 'dances', 'dancing', 'daring', 'dark', 'david', 'day', 'days', 'de', 'deal', 'death', 'decisions', 'dedication', 'deep', 'definitely', 'delicate', 'denial', 'desert', 'deserve', 'deserves', 'designating', 'designed', 'despite', 'dessert', 'destination', 'destiny', 'developed', 'development', 'dictatorship', 'die', 'died', 'diet', 'different', 'digan', 'digennaro', 'dine', 'direction', 'dirty', 'disgusting', 'disillusionment', 'disney', 'distinguishes', 'diversity', 'divide', 'divided', 'dj', 'do', 'doctor', 'does', 'doesn', 'dog', 'doing', 'dollars', 'don', 'donc', 'donde', 'dont', 'donâ', 'dorothy', 'down', 'downtown', 'dream', 'dreams', 'dries', 'driving', 'du', 'dubai', 'dude', 'during', 'dying', 'e', 'each', 'eagle', 'early', 'earth', 'east', 'eat', 'economic', 'economically', 'economy', 'effile', 'eiffel', 'either', 'el', 'elegance', 'eleganceâ', 'elfâ', 'else', 'embarrass', 'emily', 'empire', 'empty', 'en', 'end', 'endless', 'ends', 'enduring', 'energy', 'enjoy', 'enjoyed', 'enjoying', 'enjoys', 'enough', 'enter', 'enthusiasm', 'entire', 'equality', 'erase', 'err', 'escargot', 'esque', 'est', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'exceeded', 'excellent', 'exchange', 'exciting', 'exercise', 'exist', 'exists', 'expectations', 'expensive', 'experience', 'experienced', 'exploit', 'exploitation', 'extravagant', 'extreme', 'extremely', 'eyes', 'facade', 'face', 'fact', 'failure', 'fall', 'familia', 'family', 'famous', 'fancy', 'fans', 'fashion', 'fashionable', 'fast', 'faster', 'father', 'favelas', 'fear', 'fearless', 'feast', 'feel', 'feeling', 'feelings', 'feels', 'feijoada', 'felt', 'festival', 'festivals', 'festive', 'fiesta', 'fight', 'filled', 'finance', 'find', 'first', 'five', 'fk', 'flags', 'flash', 'flashy', 'floating', 'fly', 'flying', 'foie', 'follower', 'food', 'footbal', 'football', 'footballers', 'for', 'ford', 'forest', 'forests', 'forget', 'fossil', 'found', 'founded', 'francais', 'france', 'fraternity', 'freedom', 'french', 'frenchman', 'fries', 'from', 'ft', 'fuel', 'fulfilled', 'fulfilment', 'full', 'fullest', 'fun', 'furious', 'futball', 'future', 'futuristic', 'fyi', 'gain', 'game', 'games', 'gang', 'garbage', 'gas', 'gasoline', 'gaston', 'gather', 'gathers', 'gave', 'general', 'geographical', 'get', 'gets', 'getâ', 'gigantic', 'give', 'glamour', 'glitters', 'glitz', 'go', 'god', 'goddamn', 'goes', 'going', 'gold', 'golden', 'gonna', 'good', 'goodbye', 'gooooaaaaalll', 'grand', 'grandmaster', 'gras', 'great', 'greatest', 'greatness', 'greed', 'grew', 'grey', 'grow', 'growingâ', 'guests', 'gustave', 'gusteau', 'ha', 'habibi', 'had', 'hamilton', 'hands', 'happen', 'happened', 'happens', 'happiness', 'happy', 'hard', 'harder', 'harris', 'harvey', 'has', 'haute', 'have', 'haven', 'he', 'heads', 'health', 'hear', 'heart', 'hearts', 'heaven', 'heights', 'heightsâ', 'heist', 'held', 'helen', 'hell', 'help', 'henry', 'here', 'hereâ', 'hey', 'hierarchy', 'high', 'higher', 'himid', 'his', 'historic', 'history', 'hit', 'ho', 'home', 'homes', 'homeâ', 'hon', 'honeymoon', 'hope', 'horses', 'hosted', 'hot', 'hotspot', 'houses', 'how', 'however', 'hub', 'hudson', 'huge', 'human', 'hypermodernized', 'i', 'ice', 'iconic', 'idea', 'idk', 'if', 'il', 'illegal', 'imagined', 'immigrants', 'impossible', 'imposssible', 'in', 'inclusive', 'income', 'industry', 'inevitable', 'infinity', 'influencers', 'influential', 'infrastructure', 'inhabitants', 'innovation', 'insideâ', 'inspire', 'inspires', 'instantly', 'into', 'irving', 'is', 'islands', 'isn', 'it', 'items', 'its', 'itself', 'itâ', 'iâ', 'jackson', 'jai', 'janeiro', 'janeiroâ', 'jay', 'je', 'jealous', 'jesus', 'jive', 'joanne', 'job', 'jobs', 'joga', 'jordan', 'jose', 'joseph', 'joy', 'jr', 'judge', 'jul', 'jules', 'jungle', 'jungles', 'junk', 'just', 'kansas', 'kanye', 'keep', 'keeps', 'kelk', 'keller', 'kent', 'keys', 'khaled', 'khalifa', 'kind', 'king', 'kissed', 'know', 'knowledgeable', 'kylian', 'la', 'labour', 'laid', 'lama', 'lamborghini', 'land', 'landmarks', 'landscape', 'landscapes', 'language', 'las', 'lasts', 'laugh', 'lavish', 'lazy', 'leader', 'learn', 'leavin', 'leroux', 'less', 'let', 'lets', 'letter', 'letâ', 'lexicon', 'liable', 'liberty', 'license', 'lies', 'life', 'lifestyle', 'light', 'lights', 'like', 'limit', 'lincoln', 'lindsey', 'line', 'lisa', 'little', 'live', 'lived', 'lively', 'lives', 'living', 'll', 'loaf', 'loca', 'location', 'london', 'long', 'look', 'looking', 'lose', 'loss', 'lost', 'lot', 'lots', 'loud', 'louvre', 'love', 'loved', 'lovers', 'loveâ', 'loving', 'lower', 'lucky', 'luxurious', 'luxury', 'lãºcio', 'm', 'made', 'mafia', 'magic', 'magnificent', 'main', 'make', 'makes', 'making', 'man', 'manhattan', 'mans', 'many', 'mark', 'marley', 'martial', 'marvel', 'marvelous', 'marvels', 'masked', 'matuidi', 'may', 'maybe', 'mbappe', 'me', 'mean', 'meaningful', 'meant', 'meet', 'meets', 'memorable', 'memories', 'memory', 'men', 'metropolis', 'metropolitan', 'mexicans', 'meâ', 'mi', 'michael', 'middle', 'might', 'migrant', 'million', 'millions', 'mind', 'minutes', 'miracle', 'mixed', 'modern', 'moment', 'moments', 'mon', 'mona', 'money', 'moneyâ', 'monsieur', 'more', 'morning', 'mosby', 'most', 'mountain', 'mouth', 'moveable', 'movie', 'movies', 'much', 'multicultural', 'mundo', 'museum', 'music', 'must', 'mutant', 'my', 'myself', 'n', 'name', 'named', 'nation', 'natural', 'naturalistic', 'nature', 'naughty', 'need', 'never', 'new', 'news', 'newww', 'newyork', 'newyorkkkk', 'next', 'neymar', 'neymarrrr', 'nice', 'night', 'nightlife', 'nightmare', 'nights', 'ninja', 'no', 'nocturnal', 'noisy', 'none', 'not', 'nothin', 'nothing', 'novo', 'now', 'nowhereâ', 'ny', 'nyc', 'nymar', 'o', 'oasis', 'obvious', 'ocean', 'of', 'ofâ', 'og', 'oh', 'ohh', 'ohâ', 'oil', 'oinion', 'okay', 'old', 'oliveira', 'olympia', 'olympic', 'olympics', 'on', 'once', 'one', 'only', 'opportunity', 'opprtunity', 'opulence', 'or', 'oscar', 'ostriches', 'other', 'ought', 'oui', 'our', 'outdated', 'over', 'overly', 'overrated', 'overwatch', 'ow', 'own', 'pack', 'pain', 'paint', 'paintbrush', 'painting', 'palaces', 'palm', 'panda', 'paradise', 'parado', 'paradono', 'parent', 'paris', 'parisian', 'parisâ', 'parody', 'parrots', 'part', 'particular', 'parties', 'party', 'partying', 'partyâ', 'passion', 'passionate', 'patagonia', 'patrick', 'pay', 'peaceful', 'pele', 'pen', 'pense', 'people', 'perfect', 'perfume', 'person', 'peter', 'petroleum', 'peux', 'pick', 'piece', 'pizza', 'pizzas', 'place', 'places', 'plait', 'plane', 'plate', 'play', 'played', 'players', 'playground', 'playing', 'please', 'pockets', 'points', 'politics', 'pony', 'poor', 'poorer', 'popular', 'possibilities', 'possible', 'possibly', 'poverty', 'power', 'powers', 'presenting', 'preservences', 'pretty', 'price', 'prince', 'princes', 'probably', 'problems', 'probs', 'proceeded', 'progress', 'prosperity', 'prosperityâ', 'prowess', 'psg', 'psycho', 'puffy', 'purpose', 'pursuit', 'put', 'que', 'quit', 'quote', 'quotes', 'r', 'rains', 'rainy', 'raise', 'ratatouille', 'rather', 'rats', 're', 'reach', 'real', 'reality', 'really', 'reason', 'redeemer', 'redemption', 'redentor', 'reflected', 'relentless', 'rely', 'remember', 'renowned', 'rent', 'replies', 'requires', 'resides', 'respectâ', 'responsibility', 'rest', 'revolution', 'rhythm', 'rhythms', 'rich', 'richer', 'riches', 'richest', 'richness', 'richï¼', 'riding', 'right', 'rigorous', 'rio', 'riots', 'riâ', 'romance', 'romanceâ', 'romans', 'romantic', 'romanticâ', 'rome', 'ronaldoâ', 'roof', 'roots', 'rose', 'rotten', 'round', 'route', 'routine', 'rude', 'rules', 'rural', 's', 'safe', 'safety', 'said', 'salute', 'samba', 'same', 'sands', 'sandy', 'say', 'saying', 'scene', 'sculpture', 'sea', 'seaside', 'seasons', 'second', 'secrets', 'see', 'seen', 'self', 'sell', 'sera', 'serenity', 'serves', 'setback', 'settle', 'shaffer', 'shall', 'shallow', 'shape', 'share', 'she', 'shining', 'show', 'si', 'sibling', 'side', 'silver', 'simple', 'sing', 'sings', 'sins', 'situations', 'siuu', 'siuuuu', 'size', 'skies', 'sku', 'sky', 'skyline', 'skyscraper', 'skyscrapers', 'slave', 'slavery', 'sleep', 'sleeps', 'slice', 'slums', 'smell', 'so', 'soccer', 'soccerâ', 'social', 'some', 'someone', 'something', 'somewhat', 'son', 'souffle', 'soul', 'soulless', 'souls', 'sound', 'sounds', 'soup', 'sous', 'soy', 'specific', 'specifically', 'specter', 'spirit', 'sport', 'sports', 'spread', 'spreadin', 'springing', 'square', 'sriram', 'stars', 'start', 'state', 'states', 'statue', 'statueâ', 'staying', 'stays', 'stayâ', 'stealing', 'stepping', 'steve', 'steven', 'still', 'stinson', 'stock', 'stocks', 'stole', 'stomach', 'stone', 'stories', 'story', 'street', 'streets', 'strength', 'strikes', 'stub', 'stupid', 'style', 'subway', 'subways', 'succeed', 'succession', 'such', 'suiiiiiiiiiiiii', 'suis', 'suit', 'suite', 'summer', 'summers', 'sun', 'sunny', 'sunshine', 'super', 'superpowers', 'surface', 'surrender', 'surrounded', 'swift', 'symbol', 'symphony', 'system', 't', 'take', 'taken', 'takes', 'taking', 'talk', 'tall', 'tallest', 'tang', 'tasting', 'tate', 'tax', 'taxes', 'taxis', 'taylor', 'tbh', 'teaches', 'tears', 'tech', 'technology', 'ted', 'teenage', 'tell', 'tells', 'tennis', 'term', 'testament', 'than', 'thanks', 'that', 'the', 'them', 'then', 'there', 'thereafter', 'theres', 'thereâ', 'these', 'they', 'thing', 'things', 'think', 'thinking', 'third', 'this', 'thoreau', 'though', 'thought', 'thousand', 'thrives', 'thus', 'time', 'timeless', 'times', 'to', 'today', 'together', 'toilettes', 'tomato', 'tombsâ', 'tomorrow', 'top', 'torch', 'toronto', 'toto', 'touched', 'tourism', 'tourist', 'tourists', 'tower', 'toâ', 'trade', 'trading', 'tradition', 'trae', 'traffic', 'train', 'trap', 'trash', 'travel', 'treasure', 'trips', 'tropical', 'true', 'tunnels', 'turn', 'turns', 'turtles', 'twice', 'two', 'tyson', 'u', 'ugliness', 'uh', 'ultramodern', 'um', 'un', 'unconditionally', 'under', 'underbelly', 'underground', 'understand', 'understanding', 'une', 'united', 'unity', 'universes', 'university', 'unlike', 'until', 'up', 'upon', 'urban', 'us', 'usa', 'use', 'used', 'useless', 'vacation', 'vacations', 'vamos', 'variety', 've', 'vegas', 'version', 'very', 'vests', 'vi', 'vibrant', 'vida', 'vie', 'views', 'violations', 'viral', 'vision', 'viva', 'vive', 'vivid', 'vous', 'voy', 'wait', 'waiting', 'waitress', 'waitâ', 'wake', 'walk', 'walkin', 'walking', 'wall', 'wallet', 'walt', 'wannabe', 'want', 'wanted', 'wants', 'war', 'warm', 'was', 'washington', 'watercolor', 'way', 'we', 'weak', 'wealth', 'wealthy', 'wearing', 'weather', 'welcome', 'were', 'west', 'wet', 'what', 'whatâ', 'when', 'where', 'wherever', 'which', 'while', 'whimsical', 'whispers', 'white', 'whitewashed', 'who', 'wilde', 'wildest', 'will', 'willing', 'windy', 'wine', 'wings', 'wins', 'wisdom', 'wish', 'with', 'within', 'without', 'women', 'womâ', 'wonder', 'word', 'work', 'workers', 'works', 'world', 'worlds', 'worldâ', 'worm', 'worry', 'worrying', 'worst', 'worthwhile', 'would', 'wrapper', 'writes', 'written', 'wu', 'x', 'yalayalayalayala', 'yalla', 'yallah', 'yeah', 'year', 'years', 'yellow', 'yes', 'yet', 'york', 'yorkers', 'yorkkk', 'yorkðÿžµ', 'yorrrkk', 'you', 'young', 'your', 'yourself', 'youâ', 'z', 'zoo', 'zooyork', 'â', 'ã', 'ä', 'å', 'åº', 'åÿžå', 'åœ', 'æ', 'æ²', 'æ¼', 'çš', 'çž', 'è', 'é', 'ðÿžµwhen', 'œainâ', 'œbrazil', 'œdon', 'œdonâ', 'œdubai', 'œeverything', 'œi', 'œin', 'œiâ', 'œmake', 'œnew', 'œnot', 'œone', 'œthese', 'œthey', 'œto', 'œwestern', 'œwhat', 'œwhen', 'œwhy', 'œyou', 'œè', 'žå']

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

    # replace all the outliers by given number
    q7_min = -30
    q7_max = 45
    q89_min = 1
    Q89_max = 15
    df.loc[(df['Q7'] < q7_min), 'Q7'] = q7_min
    df.loc[(df['Q7'] > q7_max), 'Q7'] = q7_max
    df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    df.loc[(df['Q9'] > Q89_max), 'Q9'] = Q89_max
    df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    df.loc[(df['Q9'] > Q89_max), 'Q9'] = Q89_max
    
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
    X_numeric = np.delete(X, 3, axis=1).astype(np.float64)
    X = np.hstack((X_numeric, features)).astype(np.float64)
    X = np.concatenate((X[:, :3], X[:, 4:]), axis=1)
    
    return X

def predict_all(file: str):
    # pre-process data
    data = process_data(file)
    layer_1 = np.dot(data, weights_first_layer.T) + bias_first_layer
    layer_1_relu = np.maximum(layer_1, 0)
    layer_2 = np.dot(layer_1_relu, weights_sec_layer.T) + bias_sec_layer
    predictions = softmax(layer_2)
    cities = ['Dubai', 'Rio de Janeiro', 'New York City' ,'Paris']
    final_predictions = []
    for pred in predictions:
        pred_city = cities[np.argmax(pred)]
        final_predictions.append(pred_city)
    # using the quote to improve prediction result
    return final_predictions

# if __name__ == "__main__":
#     predictions = predict_all("example_test.csv")
#     Label = ["Dubai", "Rio de Janeiro", "Dubai", "New York City", "Dubai", "Paris", "Dubai", "Paris", "Paris", "Paris", "Paris", "Paris", "New York City", "Dubai", "Rio de Janeiro", "Paris", "Rio de Janeiro", "Rio de Janeiro", "New York City", "Dubai", "Rio de Janeiro", "Dubai", "New York City", "Dubai", "Paris", "Paris", "Rio de Janeiro", "New York City", "Rio de Janeiro", "Paris", "New York City", "Paris", "Dubai", "Paris", "Paris", "Rio de Janeiro", "Rio de Janeiro", "Dubai", "Rio de Janeiro", "Paris", "Paris", "New York City", "Dubai", "Rio de Janeiro", "New York City", "Dubai", "Rio de Janeiro", "New York City", "Dubai", "Paris", "New York City", "Dubai", "Rio de Janeiro", "Rio de Janeiro", "Rio de Janeiro", "New York City", "New York City", "Rio de Janeiro", "Paris", "Rio de Janeiro", "Dubai", "Paris", "Rio de Janeiro", "New York City", "Paris", "Dubai", "New York City", "Paris", "Dubai", "Dubai", "Paris", "Rio de Janeiro", "Rio de Janeiro", "Dubai", "Dubai", "New York City", "Paris", "New York City", "Dubai", "Rio de Janeiro", "New York City", "Rio de Janeiro", "New York City", "Dubai", "New York City", "New York City", "New York City", "Paris", "Dubai", "Dubai", "Paris", "Dubai", "Paris", "Dubai", "New York City", "Dubai", "New York City", "New York City", "Rio de Janeiro", "Rio de Janeiro"]
#     mean_accuracy = np.mean(predictions == Label)
#     print(mean_accuracy)
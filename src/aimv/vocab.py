"""Vocabularies used to turn prompt phrases into checkable requirements.

Each term maps to a category; the ``*_MAP`` dicts link some terms to a
dedicated model class. Terms not covered by a dedicated model fall back to
CLAP zero-shot similarity.
"""
from __future__ import annotations

INSTRUMENT_TERMS = [
    "piano", "electric piano", "keyboard", "keys", "organ", "synth", "synthesizer", "synth pad", "pad", "pads",
    "harpsichord", "rhodes", "accordion", "shrutibox", "harmonium", "melodica",
    "guitar", "electric guitar", "acoustic guitar", "bass guitar", "bass", "double bass", "upright bass",
    "violin", "violins", "viola", "cello", "strings", "string", "harp", "banjo", "mandolin", "ukulele",
    "bouzouki", "sitar", "sarod", "koto", "erhu", "fiddle",
    "flute", "bansuri", "clarinet", "oboe", "bassoon", "saxophone", "sax", "trumpet", "trombone", "horn",
    "horns", "brass", "tuba", "harmonica", "didgeridoo", "digeridoo", "bagpipes", "recorder", "woodwinds",
    "woodwind", "windpipe", "pan flute", "bansuri flute", "irish bouzouki",
    "drums", "drum", "drum kit", "acoustic drums", "drum machine", "percussion", "tabla", "tablas",
    "djembe", "congas", "bongos", "cajon", "cymbal", "cymbals", "hi-hat", "snare", "kick", "tambourine",
    "shaker", "claps", "handclaps", "xylophone", "marimba", "vibraphone", "glockenspiel", "bells", "bell",
    "chimes", "gong", "steel drums", "timpani", "808", "triangle", "rimshot",
    "orchestra", "beat", "beats", "sampler", "turntable", "scratching",
]

# MTG-Jamendo instrument classes (Essentia) – term -> class
ESSENTIA_INSTRUMENT_MAP = {
    "accordion": "accordion", "acoustic guitar": "acousticguitar", "bass": "bass", "bass guitar": "bass",
    "double bass": "doublebass", "upright bass": "doublebass", "bell": "bell", "bells": "bell",
    "bongos": "bongo", "brass": "brass", "cello": "cello", "clarinet": "clarinet",
    "drum machine": "drummachine", "808": "drummachine", "drums": "drums", "drum": "drums",
    "drum kit": "drums", "acoustic drums": "drums", "electric guitar": "electricguitar",
    "electric piano": "electricpiano", "flute": "flute", "bansuri": "flute", "guitar": "guitar",
    "harmonica": "harmonica", "harp": "harp", "horn": "horn", "horns": "horn", "keyboard": "keyboard",
    "keys": "keyboard", "oboe": "oboe", "orchestra": "orchestra", "organ": "organ", "pad": "pad",
    "pads": "pad", "synth pad": "pad", "percussion": "percussion", "piano": "piano", "rhodes": "rhodes",
    "saxophone": "saxophone", "bansuri flute": "flute", "sax": "saxophone", "strings": "strings", "string": "strings",
    "synth": "synthesizer", "synthesizer": "synthesizer", "trombone": "trombone", "trumpet": "trumpet",
    "viola": "viola", "violin": "violin", "violins": "violin", "fiddle": "violin", "sampler": "sampler",
    "beat": "beat", "beats": "beat",
}

GENRE_TERMS = [
    "rock", "southern rock", "hard rock", "punk", "metal", "heavy metal", "pop", "k-pop", "j-pop", "indie",
    "jazz", "blues", "funk", "soul", "r&b", "disco", "gospel", "country", "folk", "irish folk", "bluegrass",
    "classical", "orchestral", "opera", "baroque", "hindustani", "hindustani classical", "carnatic", "hindi",
    "indian", "indian classical", "bollywood", "afrobeat", "african", "latin", "salsa", "bossa nova",
    "reggae", "reggaeton", "ska", "dub", "hip hop", "hip-hop", "rap", "trap", "electronic", "edm", "techno",
    "house", "deep house", "trance", "dubstep", "drum and bass", "ambient", "lo-fi", "lofi", "chillout",
    "new age", "soundtrack", "cinematic", "video game", "game music", "children's music", "lullaby",
    "christian", "worship", "dance", "ballad", "march", "waltz", "tango", "flamenco", "celtic", "arabic",
    "middle eastern", "chinese", "japanese", "greek", "turkish", "advertisement music", "jingle",
    "samba", "caribbean", "world music", "symphonic", "easy listening", "fanfare", "wedding music", "comedy",
]

# Discogs-400 style keywords (Essentia genre_discogs400) – term -> substrings
DISCOGS_GENRE_MAP = {
    "southern rock": ["Southern Rock"], "hard rock": ["Hard Rock"], "rock": ["Rock---"],
    "punk": ["Punk"], "metal": ["Metal"], "pop": ["Pop---"], "k-pop": ["K-pop"], "j-pop": ["J-pop"],
    "indie": ["Indie"], "jazz": ["Jazz---"], "blues": ["Blues---"], "funk": ["Funk"], "soul": ["Soul"],
    "r&b": ["RnB"], "disco": ["Disco"], "gospel": ["Gospel"], "country": ["Country"],
    "folk": ["Folk"], "irish folk": ["Celtic", "Folk"], "bluegrass": ["Bluegrass"],
    "classical": ["Classical---"], "orchestral": ["Classical---", "Score"], "opera": ["Opera"],
    "baroque": ["Baroque"], "hindustani": ["Hindustani"], "hindustani classical": ["Hindustani"],
    "carnatic": ["Carnatic"], "hindi": ["Hindustani", "Bollywood"], "indian": ["Hindustani", "Bollywood"],
    "indian classical": ["Hindustani", "Carnatic"], "bollywood": ["Bollywood"], "afrobeat": ["Afrobeat"],
    "african": ["African"], "latin": ["Latin---"], "salsa": ["Salsa"], "bossa nova": ["Bossa Nova"],
    "reggae": ["Reggae---"], "reggaeton": ["Reggaeton"], "ska": ["Ska"], "dub": ["Dub"],
    "hip hop": ["Hip Hop---"], "hip-hop": ["Hip Hop---"], "rap": ["Hip Hop---"], "trap": ["Trap"],
    "electronic": ["Electronic---"], "edm": ["Electro House", "Big Room", "Progressive House"],
    "techno": ["Techno"], "house": ["House"], "deep house": ["Deep House"], "trance": ["Trance"],
    "dubstep": ["Dubstep"], "drum and bass": ["Drum n Bass"], "ambient": ["Ambient"],
    "lo-fi": ["Lo-Fi"], "lofi": ["Lo-Fi"], "chillout": ["Chillout"], "new age": ["New Age"],
    "soundtrack": ["Soundtrack", "Score"], "cinematic": ["Score", "Soundtrack"],
    "video game": ["Video Game Music"], "game music": ["Video Game Music"], "ballad": ["Ballad"],
    "flamenco": ["Flamenco"], "tango": ["Tango"], "celtic": ["Celtic"], "greek": ["Greek"],
}

MOOD_TERMS = [
    "happy", "joyful", "cheerful", "fun", "playful", "upbeat", "uplifting", "positive", "hopeful",
    "energetic", "exciting", "powerful", "intense", "aggressive", "angry", "epic", "dramatic",
    "sad", "melancholic", "melancholy", "emotional", "sentimental", "passionate", "romantic", "love",
    "calm", "relaxing", "chill", "soothing", "peaceful", "mellow", "soft", "gentle", "meditative", "dreamy",
    "trippy", "ethereal", "angelic", "celestial", "mystical", "mysterious", "dark", "eerie", "spooky",
    "suspenseful", "tense", "haunting", "groovy", "funky", "catchy", "nostalgic", "spiritual",
    "sweet", "lively", "cool", "heavy", "harsh", "loud", "sorrowful", "triumphant", "inspiring",
    "motivational", "easygoing", "buoyant", "youthful", "engaging", "vigorous", "festive", "cozy", "euphoric",
    "lonely", "loneliness", "sensual", "spirited", "perky", "violent", "manic", "chaotic", "monotonous",
    "bored", "addictive", "soulful", "quirky", "weird", "poignant",
]

# MTG-Jamendo mood/theme classes (Essentia) – term -> class
ESSENTIA_MOOD_MAP = {
    "happy": "happy", "joyful": "happy", "cheerful": "happy", "fun": "fun", "playful": "fun",
    "upbeat": "upbeat", "uplifting": "uplifting", "positive": "positive", "hopeful": "hopeful",
    "energetic": "energetic", "exciting": "energetic", "powerful": "powerful", "intense": "powerful",
    "epic": "epic", "dramatic": "dramatic", "sad": "sad", "melancholic": "melancholic",
    "melancholy": "melancholic", "emotional": "emotional", "sentimental": "emotional",
    "passionate": "emotional", "romantic": "romantic", "love": "love", "calm": "calm",
    "relaxing": "relaxing", "chill": "relaxing", "soothing": "relaxing", "peaceful": "calm",
    "mellow": "soft", "soft": "soft", "gentle": "soft", "meditative": "meditative", "dreamy": "dream",
    "dark": "dark", "groovy": "groovy", "funky": "groovy", "heavy": "heavy", "inspiring": "inspiring",
    "motivational": "motivational", "cool": "cool",
}

# SunoCaps expert emotion labels (for emotion classification by zero-shot)
SUNOCAPS_EMOTIONS = ["happy", "exciting", "sad", "calm", "content", "angry", "disgust", "surprise"]

TEMPO_WORDS = {
    # phrase -> (min_bpm, max_bpm)
    "very slow": (40, 70), "slow": (50, 90), "slow tempo": (50, 90), "downtempo": (60, 100),
    "medium tempo": (85, 125), "moderate tempo": (85, 125), "mid-tempo": (85, 125), "medium": (85, 125),
    "medium fast": (105, 140), "medium-fast": (105, 140), "uptempo": (115, 170), "up-tempo": (115, 170),
    "fast": (120, 200), "fast tempo": (120, 200), "very fast": (150, 220), "quick": (120, 200),
    "mid tempo": (85, 125), "faster tempo": (120, 200),
}

VOCAL_TERMS = [
    "vocal", "vocals", "voice", "voices", "singer", "singers", "singing", "sings", "sung", "vocalist",
    "rap", "rapping", "rapper", "choir", "choral", "chant", "chanting", "humming", "whistling",
    "vocables", "lyrics", "a cappella", "acapella", "soprano", "alto", "tenor", "baritone", "falsetto",
    "harmonies", "backing vocals", "beatbox", "spoken word", "opera singer", "yodel",
    "male talking", "female talking", "man talking", "woman talking", "narrator", "narration",
]

OTHER_TERMS = [
    "low quality", "poor quality", "poor audio quality", "low quality audio", "amateur recording", "noisy",
    "noise", "mono", "stereo", "live performance", "live recording", "live", "studio", "lo-fi recording",
    "reverb", "echo", "delay", "distortion", "distorted", "clean sound", "effect pedal", "muffled",
    "crowd", "audience", "applause", "clapping", "cheering",
    "sound effect", "sound effects", "birds chirping", "birds", "water", "water flowing", "rain", "wind",
    "nature", "ambient noise", "background noise", "chatter", "people talking", "talking", "rustling",
    "whirring", "glass shattering", "ducks", "footsteps", "laughter", "nature atmosphere",
]

# Spelling variants and typos seen in SunoCaps prompts -> the term used in the lists above.
# The parser matches the variant and stores the term, e.g. "shimmering hi hats" -> instruments: hi-hat.
TERM_ALIASES = {
    # instruments
    "e-bass": "bass", "e-guitar": "electric guitar", "electronic guitar": "electric guitar",
    "guitar string": "guitar", "guitark": "guitar", "guitarist": "guitar", "sitarist": "sitar",
    "acordion": "accordion", "synthesiser": "synthesizer", "hi hat": "hi-hat", "drumset": "drum kit",
    "drumming": "drums", "baraban": "drum", "stringed instrument": "strings", "turn table": "turntable",
    "percussive elements": "percussion",
    # vocals
    "beat boxing": "beatbox", "beat box": "beatbox", "beatboxing": "beatbox", "vocalisation": "vocals",
    "vocalization": "vocals", "hummed": "humming", "hums": "humming", "yodeling": "yodel", "yodelling": "yodel",
    # genre
    "himdustani": "hindustani", "adverticement music": "advertisement music", "advertisement": "advertisement music",
    "carribean": "caribbean", "childrens song": "children's music", "kids music": "children's music",
    "songs for toddlers": "children's music", "kindergarten": "children's music",
    "in game audio": "video game", "in-game audio": "video game",
    # other
    "inferior quality audio": "low quality", "mediocre audio quality": "low quality", "home recording": "amateur recording",
}

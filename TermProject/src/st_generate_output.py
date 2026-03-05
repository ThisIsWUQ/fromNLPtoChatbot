# Generate an output text given the user's input

import streamlit as st
import numpy as np
import random
import tracery
import spacy
import markovify
from spacy.lang.en.stop_words import STOP_WORDS
from tracery.modifiers import base_english
from transformers import pipeline

# Set up working vocabularies

with open("TermProject/data/food_word_list.txt", "r", encoding="utf-8") as f:
    food_text = f.read()
food_vocab = food_text.split(" ")

with open("TermProject/data/flavor_word_list.txt", "r", encoding="utf-8") as f:
    flavor_text = f.read()
flavor_vocab = flavor_text.split(" ")

with open("TermProject/data/food_sensory_neutral.txt", "r", encoding="utf-8") as f:
    corpus_text_food_neutral = f.read()

with open("TermProject/data/food_sensory_anger.txt", "r", encoding="utf-8") as f:
    corpus_text_food_anger = f.read()

with open("TermProject/data/food_sensory_disgust.txt", "r", encoding="utf-8") as f:
    corpus_text_food_disgust = f.read()

with open("TermProject/data/food_sensory_fear.txt", "r", encoding="utf-8") as f:
    corpus_text_food_fear = f.read()

with open("TermProject/data/food_sensory_joy.txt", "r", encoding="utf-8") as f:
    corpus_text_food_joy = f.read()

with open("TermProject/data/food_sensory_sadness.txt", "r", encoding="utf-8") as f:
    corpus_text_food_sadness = f.read()

with open("TermProject/data/food_sensory_surprise.txt", "r", encoding="utf-8") as f:
    corpus_text_food_surprise = f.read()

with open("TermProject/data/abstract_bridge_word_list.txt", "r", encoding="utf-8") as f:
    bridge_text = f.read()
bridge_vocab = bridge_text.split(" ")

with open("TermProject/data/home_object_word_list.txt", "r", encoding="utf-8") as f:
    furniture_text = f.read()
furniture_vocab = furniture_text.split(" ")

with open("TermProject/data/atmosphere_word_list.txt", "r", encoding="utf-8") as f:
    atmosphere_text = f.read()
atmosphere_vocab = atmosphere_text.split(" ")

with open("TermProject/data/house_interior_neutral.txt", "r", encoding="utf-8") as f:
    corpus_text_home_neutral = f.read()

with open("TermProject/data/house_interior_anger.txt", "r", encoding="utf-8") as f:
    corpus_text_home_anger = f.read()

with open("TermProject/data/house_interior_disgust.txt", "r", encoding="utf-8") as f:
    corpus_text_home_disgust = f.read()

with open("TermProject/data/house_interior_fear.txt", "r", encoding="utf-8") as f:
    corpus_text_home_fear = f.read()

with open("TermProject/data/house_interior_joy.txt", "r", encoding="utf-8") as f:
    corpus_text_home_joy = f.read()

with open("TermProject/data/house_interior_sadness.txt", "r", encoding="utf-8") as f:
    corpus_text_home_sadness = f.read()

with open("TermProject/data/house_interior_surprise.txt", "r", encoding="utf-8") as f:
    corpus_text_home_surprise = f.read()

# Load an English language package

nlp = spacy.load('en_core_web_md')

# Load an emotion classifier model for English texts

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def extract_emotions(text, threshold=0.05):
    """
    Extract emotions in a given text using the emotion classifier
    """
    results = emotion_classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True) # sort results by score (descending)
    emotions = [r["label"] for r in results if r["score"] > threshold]

    return emotions

def extract_themes(text):
    """
    Extract main contents (i.e. nouns, verbs, adjectives)
    in a given text.
    """
    doc = nlp(text)
    themes = []

    # Get tokens of content words
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and token.is_alpha and token.text.lower() not in STOP_WORDS:
            themes.append(token.lemma_.lower())

    return themes

def nearest_neighbors(target, vocab, k=10, tolerance=0.000001, random_pick=True):
    """
    Find nearest neighbor words to `target` inside a restricted vocabulary
    using spaCy word vectors.

    Parameters:
        target (str): the word to compare
        vocab (list[str]): candidate words (food, flavors, etc.)
        k (int): how many top results to consider
        tolerance (float): how close scores must be to the best
        random_pick (bool): if True, pick randomly among close matches

    Returns:
        list of neighboring words (or a single word if random_pick=True)
    """
    target_token = nlp.vocab[target.lower()]

    if not target_token.has_vector:
        return []

    scores = {}

    # Calculate similarity score
    for word in vocab:
        token = nlp.vocab[word.lower()]
        if token.has_vector and word != target:
            score = target_token.similarity(token)
            scores[word] = score

    if not scores:
        return []

    # Sort by similarity
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:k]
    max_score = sorted_scores[0][1]

    # Keep words very close to the best score
    close_matches = [
        word for word, score in sorted_scores
        if abs(score - max_score) < tolerance
    ]

    if random_pick:
        return random.choice(close_matches)
    else:
        return close_matches

with open("TermProject/data/google-10000-english-no-swears.txt", "r", encoding="utf-8") as f:
    common_text = f.read()
common_words = common_text.split("\n")

def analogy(a, b, c, top_n=1):
    """
    Solve analogies of the form: A : B :: C : D
    Returns the best matching word D.
    """
    # Get vector representations
    A = nlp.vocab[a].vector
    B = nlp.vocab[b].vector
    C = nlp.vocab[c].vector

    # Compute target vector
    target = B - A + C

    # Search the entire vocabulary for closest words
    similarities = []
    for w in common_words: # only check common_words
        word = nlp.vocab[w] # get lexeme
        if word.has_vector and word.is_alpha and not word.is_stop:
            sim = np.dot(target, word.vector) / (
                np.linalg.norm(target) * np.linalg.norm(word.vector)
            )
            similarities.append((word.text, float(sim)))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top result(s), skipping the original input words
    results = [w for w, s in similarities if w not in {a, b, c}]
    return results[:top_n]


def map_theme_to_content(themes, vocab, random_pick=True):
    """
    Map theme words to the most semantically similar words
    (either food or furniture words).
    """
    contents = []

    for theme in themes:
        content = nearest_neighbors(theme, vocab, random_pick=True)
        if content:
            contents.append(content)

    return list(set(contents))

def map_emotion_to_sense(emotions, themes, vocab, random_pick=True):
    """
    Map emotion words extracted previously to the
    semantically similar words of sense as well as its analogy
    using spaCy word vectors.
    """
    senses = []

    for emotion in emotions:
        sense = nearest_neighbors(emotion, vocab, random_pick=True)
        if sense:
            senses.append(sense)

        try:
            anchor_theme = random.choice(themes)
            anchor_sense = random.choice(vocab)

            candidates = analogy(emotion, anchor_theme, anchor_sense, top_n=5)
            candidates = [c for c in candidates if c in vocab]

            if candidates:
                senses.append(random.choice(candidates))
        except:
            pass

        try:
            anchor_emotion = random.choice(emotions)
            anchor_sense  = random.choice(vocab)

            candidates = analogy(emotion, anchor_emotion, anchor_sense, top_n=5)
            candidates = [c for c in candidates if c in vocab]

            if candidates:
                senses.append(random.choice(candidates))
        except:
            pass

    return list(set(senses))

def generate_narration_food(foods, flavors):
    """
    Generate a narrative text of food, given from the mapped words
    """
    foods = foods
    flavors = flavors

    rules = {
        "origin": [
            "#open#", "#taste1#", "#taste2#", "#taste3#", "#taste4#", "#taste5#", "#taste6#", "#taste7#", "#taste8#", "#taste9#",
            "#taste10#", "#taste11#", "#taste12#", "#taste13#", "#taste14#", "#taste15#", "#taste16#", "#taste17#", "#taste18#", "#taste19#"
            "#taste20#", "#taste21#", "#taste22#", "#taste23#", "#taste24#", "#taste25#", "#taste26#", "#taste27#", "#taste28#", "#taste29#",
            "#taste30#", "#taste31#", "#taste32#", "#taste33#", "#taste34#", "#taste35#", "#taste36#", "#taste37#", "#taste38#", "#taste39#",
            "#taste40#", "#taste41#", "#taste42#", "#taste43#", "#taste44#", "#taste45#", "#taste46#", "#taste47#", "#taste48#", "#taste49#",
            "#taste50#", "#taste51#", "#taste52#", "#taste53#", "#taste54#", "#taste55#", "#taste56#", "#taste57#", "#taste58#", "#taste59#",
            "#taste60#", "#taste61#", "#taste62#", "#taste63#", "#taste64#", "#taste65#", "#taste66#", "#taste67#", "#taste68#", "#taste69#",
            "#taste70#", "#taste71#", "#taste72#", "#taste73#", "#taste74#", "#taste75#", "#taste76#", "#taste77#", "#taste78#", "#taste79#",
            "#taste80#", "#taste81#", "#taste82#", "#taste83#", "#taste84#", "#taste85#", "#taste86#", "#taste87#", "#taste88#", "#taste89#",
            "#taste90#", "#taste91#", "#taste92#", "#taste93#", "#taste94#", "#taste95#", "#close#"
        ],
        "open": "#flavor.capitalize# dish built around #food.a#.",
        "food": foods,
        "flavor": flavors,
        "taste1": "The #food# is and tastes distinctly #flavor#.",
        "taste2": "The #flavor# presence is carried by the #food#, and the #flavor# touch lingers on the #food#.",
        "taste3": "At its heart lies the #food#, which is touched by the #flavor#, and the #food# dissolves into the #flavor# finish.",
        "taste4": "The #food# releases the #flavor# scent whereas the #flavor# notes define the #food#.",
        "taste5": "The #flavor# fragrance surrounds the #food#, and the #flavor# quality unfolds through the #food#.",
        "taste6": "The #food# absorbs the #flavor# richness, yet there is something #flavor# about the #food#.",
        "taste7": "Although the #food# appears mild, it reveals #flavor.a# core.",
        "taste8": "The #food# begins gently, but a #flavor# intensity follows.",
        "taste9": "When the #food# is done, it becomes deeply #flavor#.",
        "taste10": "Moreover, the #food# remains unexpectedly #flavor#.",
        "taste11": "The #food# begins with #flavor#, then settles into this #flavor# flavor.",
        "taste12": "#food.capitalize# carries a #flavor# edge that softens over time.",
        "taste13": "A layer of #flavor# rises from the body of the #food#.",
        "taste14": "The #food# unfolds gradually, revealing #flavor# beneath its surface.",
        "taste15": "The #food# balances #flavor# intensity with a structure.",
        "taste16": "The #food# seems delicate, yet it delivers a #flavor# impact.",
        "taste17": "With each bite, the #food# shifts to #flavor#.",
        "taste18": "The #food# evolves, moving through phases of #flavor#.",
        "taste19": "When chilled, the #food# reveals a #flavor# clarity.",
        "taste20": "If overcooked, the #food# loses its #flavor# balance.",
        "taste21": "The #food# feels like a color of #flavor#.",
        "taste22": "The #food# hums with #flavor# resonance.",
        "taste23": "The #food# explores the boundary of #flavor#.",
        "taste24": "The #food# embodies a theory of #flavor#.",
        "taste25": "The #food# questions its own #flavor# identity.",
        "taste26": "Although the #food# presents an exterior, it gradually reveals a #flavor# interior that reshapes the entire experience.",
        "taste27": "The #food#, initially marked by #flavor#, transforms as it lingers, allowing a quieter flavor to emerge.",
        "taste28": "While the #food# resists immediate definition, its #flavor# presence accumulates with each moment.",
        "taste29": "The #food# opens gently, carrying #flavor#, before settling into a sustained calm.",
        "taste30": "Because the #food# retains its structure, the #flavor# unfolds in controlled stages.",
        "taste31": "Long after the #food# fades, a trace of #flavor# remains.",
        "taste32": "The surface suggests calm, yet beneath it the #food# burns with #flavor#.",
        "taste33": "Inside the #food#, #flavor# waits patiently.",
        "taste34": "What appears simple in the #food# conceals a complex #flavor# movement.",
        "taste35": "Each encounter with the #food# redraws the boundaries of #flavor#.",
        "taste36": "The #food# stretches toward #flavor#, almost resisting it.",
        "taste37": "Between texture and memory, the #food# negotiates its #flavor#.",
        "taste38": "A faint #flavor# shimmer vibrates within the #food#.",
        "taste39": "Across the palate, the #food# traces a line of #flavor#.",
        "taste40": "At first restrained, the #food# gradually leans into #flavor#.",
        "taste41": "The #food# drifts from neutrality toward #flavor# intensity.",
        "taste42": "#Flavor.capitalize#, suspended in the #food#.",
        "taste43": "The #food# — briefly #flavor#, then gone.",
        "taste44": "#Flavor.capitalize# operates as the dominant register within the #food#.",
        "taste45": "The #food# stabilizes its #flavor# through repetition.",
        "taste46": "Within the #food#, #flavor# functions as both surface and depth.",
        "taste47": "Structurally, the #food# depends on #flavor# contrast.",
        "taste48": "The #food# organizes #flavor# into a precise sequence.",
        "taste49": "The #food# wavers between restraint and #flavor# excess.",
        "taste50": "The #food# trembles at the edge of #flavor#.",
        "taste51": "Within the #food#, #flavor# threatens to overtake everything.",
        "taste52": "The #food# compresses its #flavor# into a single moment.",
        "taste53": "Every return to the #food# intensifies its #flavor#.",
        "taste54": "Layer upon layer, the #food# thickens with #flavor#.",
        "taste55": "The #food# saturates the air with #flavor#.",
        "taste56": "Gradually, the #flavor# slips away from the #food#.",
        "taste57": "The #food# forgets its own #flavor#.",
        "taste58": "The #flavor# thins until the #food# feels almost neutral.",
        "taste59": "A silence of #flavor# surrounds the #food#.",
        "taste60": "The #food# tastes like the sound of #flavor#.",
        "taste61": "The texture of the #food# speaks in #flavor# tones.",
        "taste62": "The #food# becomes an argument for #flavor#.",
        "taste63": "The #food# redefines what #flavor# can be.",
        "taste64": "The #food# confronts the limits of #flavor#.",
        "taste65": "Through the #food#, #flavor# discovers itself.",
        "taste66": "One bite of the #food#, and #flavor# shifts unexpectedly.",
        "taste67": "The #food# surprises with a sudden turn toward #flavor#.",
        "taste68": "The #food# waits quietly before revealing its #flavor#.",
        "taste69": "#Flavor.capitalize# echoes within the #food#.",
        "taste70": "#Flavor.capitalize# echoes within the #food#.",
        "taste71": "A memory of #flavor# returns through the #food#.",
        "taste72": "The #food# circles back to #flavor#.",
        "taste73": "#Flavor.capitalize# resurfaces each time the #food# settles.",
        "taste74": "#Flavor.capitalize# without excess, anchored in the #food#.",
        "taste75": "The #flavor# arrives late, but transforms the #food# entirely.",
        "taste76": "Over time, the #food# accumulates a deeper #flavor# complexity.",
        "taste77": "As it rests, the #food# settles into a composed #flavor# calm.",
        "taste78": "The #food# matures gradually, enriching its #flavor# dimension.",
        "taste79": "Cooling reshapes the #food# into a more introspective #flavor# tone.",
        "taste80": "The #food# evolves quietly, sustaining a continuous #flavor# unfolding.",
        "taste81": "Repeated tasting reveals a layered #flavor# architecture within the #food#.",
        "taste82": "The #food# ages across the plate, intensifying its #flavor# gravity.",
        "taste83": "With patience, the #food# reveals an unexpected #flavor# clarity.",
        "taste84": "The #food# transforms subtly, allowing #flavor# to reorganize.",
        "taste85": "Time polishes the edges of the #food#, refining its #flavor# expression.",
        "taste86": "The texture of the #food# reinforces its distinctly #flavor# character.",
        "taste87": "A subtle aroma rises from the #food#, thick with #flavor# resonance.",
        "taste88": "The surface of the #food# absorbs heat, intensifying its #flavor# depth.",
        "taste89": "The #food# balances firmness with a #flavor# softness.",
        "taste90": "The mouthfeel of the #food# carries a sustained #flavor# clarity.",
        "taste91": "Through contrast, the #food# sharpens its #flavor# identity.",
        "taste92": "The #food# diffuses slowly, releasing layers of #flavor# nuance.",
        "taste93": "Temperature shifts reveal the #food#’s underlying #flavor# structure.",
        "taste94": "The #food# holds a tactile memory of #flavor# intensity.",
        "taste95": "Each bite of the #food# concentrates the experience of #flavor#.",
        "close": "The #food# has a #flavor# depth."
    }

    grammar = tracery.Grammar(rules)
    grammar.add_modifiers(base_english)

    narration_food = " ".join([grammar.flatten("#origin#") for i in range(200)])

    return narration_food

def generate_narration_home(furniture, atmosphere):
    """
    Generate a narrative text of food, given from the mapped words
    """
    furniture = furniture
    atmosphere = atmosphere

    rules = {
        "origin": [
            "#open#", "#space1#", "#space2#", "#space3#", "#space4#", "#space5#", "#space6#", "#space7#", "#space8#", "#space9#",
            "#space10#", "#space11#", "#space12#", "#space13#", "#space14#", "#space15#", "#space16#", "#space17#", "#space18#", "#space19#",
            "#space20#", "#space21#", "#space22#", "#space23#", "#space24#", "#space25#", "#space26#", "#space27#", "#space28#", "#space29#",
            "#space30#", "#space31#", "#space32#", "#space33#", "#space34#", "#space35#", "#space36#", "#space37#", "#space38#", "#space39#",
            "#space40#", "#space41#", "#space42#", "#space43#", "#space44#", "#space45#", "#space46#", "#space47#", "#space48#", "#space49#",
            "#space50#", "#space51#", "#space52#", "#space53#", "#space54#", "#space55#", "#space56#", "#space57#", "#space58#", "#space59#",
            "#space60#", "#space61#", "#space62#", "#space63#", "#space64#", "#space65#", "#space66#", "#space67#", "#space68#", "#space69#",
            "#space70#", "#space71#", "#space72#", "#space73#", "#space74#", "#space75#", "#space76#", "#space77#", "#space78#", "#space79#",
            "#space80#", "#space81#", "#space82#", "#space83#", "#space84#", "#space85#", "#space86#", "#space87#", "#space88#", "#space89#",
            "#space90#", "#space91#", "#space92#", "#space93#", "#space94#", "#space95#", "#close#"
        ],
        "open": "#furniture.capitalize# is #atmosphere#.",
        "furniture": furniture,
        "atmosphere": atmosphere,
        "space1": "The #furniture# stands in a distinctly #atmosphere# atmosphere.",
        "space2": "The #atmosphere# presence is carried by the #furniture#, and the #atmosphere# mood lingers through the room.",
        "space3": "At its center lies the #furniture#, touched by #atmosphere#, as the space settles into a #atmosphere# calm.",
        "space4": "The #furniture# shapes a #atmosphere# field, while subtle #atmosphere# undertones define the interior.",
        "space5": "Although the #furniture# appears minimal, it reveals #atmosphere.a# core. The space begins quietly, but a #atmosphere# intensity unfolds.",
        "space6": "When illuminated, the #furniture# becomes deeply #atmosphere#. Even in shadow, it remains unexpectedly #atmosphere#.",
        "space7": "The #furniture# introduces #atmosphere#, then settles into a softer register. #furniture.capitalize# carries a #atmosphere# edge that softens over time.",
        "space8": "A layer of #atmosphere# emerges from the structure of the #furniture#. The space unfolds gradually, revealing #atmosphere# beneath its surfaces.",
        "space9": "The #furniture# balances #atmosphere# presence with structural clarity. It seems restrained, yet it delivers a #atmosphere# impact.",
        "space10": "With each step, the #furniture# shifts the room toward #atmosphere#. The interior evolves, moving through phases of #atmosphere#.",
        "space11": "In daylight, the #furniture# reveals a #atmosphere# clarity. In dim light, it loses some of its #atmosphere# precision.",
        "space12": "The #furniture# feels like a material expression of #atmosphere#. It hums with #atmosphere# resonance.",
        "space13": "The #furniture# explores the boundary of #atmosphere#. It embodies a theory of #atmosphere#. It questions its own #atmosphere# identity.",
        "space14": "Although the #furniture# presents a restrained exterior, it gradually reveals a #atmosphere# interior that reshapes the entire spatial experience.",
        "space15": "The #furniture#, initially marked by #atmosphere#, transforms as one inhabits the space, allowing a quieter quality to emerge.",
        "space16": "While the #furniture# resists immediate definition, its #atmosphere# presence accumulates with time.",
        "space17": "The #furniture# anchors the room gently, carrying #atmosphere#, before settling into a sustained calm.",
        "space18": "Because the #furniture# retains its structural clarity, the #atmosphere# atmosphere unfolds in controlled stages.",
        "space19": "The #furniture# arranges itself within a field of #atmosphere#, where proportions guide perception.",
        "space20": "Light brushes across the #furniture#, intensifying its #atmosphere# character.",
        "space21": "The surface of the #furniture# absorbs shadow, deepening the room’s #atmosphere# register.",
        "space22": "The #furniture# interrupts the space with a gesture of #atmosphere#, redirecting movement.",
        "space23": "Through careful placement, the #furniture# establishes a rhythm of #atmosphere# throughout the interior.",
        "space24": "The #furniture# negotiates between openness and enclosure, producing a #atmosphere# equilibrium.",
        "space25": "Edges soften around the #furniture#, dissolving into a #atmosphere# haze.",
        "space26": "The #furniture# reflects a quiet #atmosphere# logic that structures the entire room",
        "space27": "Against the wall, the #furniture# projects a subtle #atmosphere# tension.",
        "space28": "The geometry of the #furniture# introduces a disciplined #atmosphere# order.",
        "space29": "The #furniture# absorbs the room’s acoustics, transforming them into #atmosphere# stillness.",
        "space30": "Material textures within the #furniture# reinforce the #atmosphere# atmosphere.",
        "space31": "The #furniture# creates a threshold between functional clarity and #atmosphere# abstraction.",
        "space32": "From certain angles, the #furniture# appears almost #atmosphere#, reshaping spatial perception.",
        "space33": "The #furniture# draws the eye inward, cultivating a #atmosphere# intimacy.",
        "space34": "Subtle contrasts around the #furniture# heighten its #atmosphere# presence.",
        "space35": "The #furniture# stabilizes the interior, allowing #atmosphere# to circulate freely.",
        "space36": "As evening falls, the #furniture# shifts toward a deeper #atmosphere# tone.",
        "space37": "The scale of the #furniture# amplifies a sense of #atmosphere# enclosure.",
        "space38": "The #furniture# holds the room in a state of suspended #atmosphere# equilibrium.",
        "space39": "Through repetition and variation, the #furniture# sustains a layered #atmosphere# complexity.",
        "space40": "The #furniture# becomes a structural metaphor for #atmosphere# within the domestic landscape.",
        "space41": "The #furniture# proposes a hypothesis of #atmosphere#, testing how space can be inhabited.",
        "space42": "The #furniture# questions the boundary between utility and #atmosphere# expression.",
        "space43": "The #furniture# articulates a quiet discourse on #atmosphere# within the home.",
        "space44": "Within the room, the #furniture# performs #atmosphere# as spatial language.",
        "space45": "The #furniture# translates material form into #atmosphere# experience.",
        "space46": "Under morning light, the #furniture# reveals a crisp #atmosphere# clarity.",
        "space47": "In artificial light, the #furniture# radiates a controlled #atmosphere# warmth.",
        "space48": "Filtered sunlight reframes the #furniture# within a softer #atmosphere# glow.",
        "space49": "Reflections from the #furniture# scatter a faint #atmosphere# shimmer across the walls.",
        "space50": "The interplay of light and shadow turns the #furniture# into a vessel of #atmosphere# depth.",
        "space52": "Grain patterns within the #furniture# echo a subtle #atmosphere# rhythm.",
        "space53": "The weight of the #furniture# anchors the room in #atmosphere# stability.",
        "space54": "Through texture and form, the #furniture# negotiates a #atmosphere# balance.",
        "space55": "The #furniture# stands as a tactile expression of #atmosphere# restraint.",
        "space56": "The #furniture# directs circulation through a corridor of #atmosphere# tension.",
        "space57": "Moving around the #furniture#, one senses a gradual shift toward #atmosphere#.",
        "space58": "The #furniture# opens pathways that dissolve into #atmosphere# quietness.",
        "space59": "As one passes the #furniture#, the room transitions into a softer #atmosphere# register.",
        "space60": "The #furniture# subtly redirects movement, reinforcing the room’s #atmosphere# coherence.",
        "space61": "Each approach to the #furniture# reframes the interior within #atmosphere# perspective.",
        "space62": "The #furniture# slows movement, allowing #atmosphere# to settle into the space.",
        "space63": "Circulation bends around the #furniture#, intensifying its #atmosphere# presence.",
        "space64": "The #furniture# orchestrates motion, turning passage into #atmosphere# experience.",
        "space65": "Through its placement, the #furniture# transforms routine movement into #atmosphere# ritual.",
        "space66": "The #furniture# sharpens visual focus within a field of #atmosphere#.",
        "space67": "Textures across the #furniture# diffuse a subtle #atmosphere# softness.",
        "space68": "The scent of materials around the #furniture# deepens the room’s #atmosphere# mood.",
        "space69": "The #furniture# absorbs ambient sound, thickening the #atmosphere# stillness.",
        "space70": "Reflections across the #furniture# multiply the #atmosphere# dimension.",
        "space71": "The #furniture# holds light differently at each hour, reshaping #atmosphere# perception.",
        "space72": "Shadow gathers beneath the #furniture#, intensifying the #atmosphere# gravity.",
        "space73": "The tactile quality of the #furniture# reinforces the interior’s #atmosphere# intimacy.",
        "space74": "Through contrast, the #furniture# heightens the room’s #atmosphere# clarity.",
        "space75": "The #furniture# filters sensory input into a composed #atmosphere# equilibrium.",
        "space76": "Over time, the #furniture# accumulates a deeper layer of #atmosphere# meaning.",
        "space77": "As daylight fades, the #furniture# shifts toward a more introspective #atmosphere# tone.",
        "space78": "The #furniture# ages gracefully, enriching the #atmosphere# atmosphere.",
        "space79": "Seasonal light transforms the #furniture# into a marker of #atmosphere# change.",
        "space80": "With repeated use, the #furniture# settles into a stable #atmosphere# rhythm.",
        "space81": "The #furniture# records traces of inhabitation, intensifying its #atmosphere# character.",
        "space82": "Morning reveals the #furniture# in crisp #atmosphere# clarity, while evening softens it into reflection.",
        "space83": "The #furniture# evolves quietly, sustaining a continuous #atmosphere# unfolding.",
        "space84": "Time polishes the surfaces of the #furniture#, amplifying its #atmosphere# refinement.",
        "space85": "In still moments, the #furniture# seems suspended within a timeless #atmosphere# field.",
        "space86": "The #furniture# stages a dialogue between structure and #atmosphere# abstraction.",
        "space87": "The #furniture# performs #atmosphere# as both material fact and emotional suggestion.",
        "space88": "Within the domestic frame, the #furniture# theorizes #atmosphere# through proportion.",
        "space89": "The #furniture# negotiates identity through a language of #atmosphere#.",
        "space90": "The #furniture# renders #atmosphere# visible through spatial syntax.",
        "space91": "The #furniture# embodies a quiet epistemology of #atmosphere# experience.",
        "space92": "Through restraint, the #furniture# intensifies the #atmosphere# atmosphere.",
        "space93": "The #furniture# becomes an interface between inhabitation and #atmosphere# reflection.",
        "space94": "The #furniture# encodes #atmosphere# within its geometry.",
        "space95": "The #furniture# sustains a layered discourse of #atmosphere# across the interior.",
        "close": "The #furniture# gives the space a lasting #atmosphere# depth."
    }

    grammar = tracery.Grammar(rules)
    grammar.add_modifiers(base_english)

    narration_home = " ".join([grammar.flatten("#origin#") for i in range(200)])

    return narration_home

def generate_output_text(descriptive_text, corpus_text, vocab1, vocab2, weight=[ 0.6, 0.4 ], k=20, t=1000):
    """
    Generate an output text
    """
    descriptive_model = markovify.Text(descriptive_text, state_size=2)
    corpus_model = markovify.Text(corpus_text, state_size=2)
    model_combined = markovify.combine([ descriptive_model, corpus_model ], weight)

    matched_sentences = []

    for i in range(k):
        sentence = model_combined.make_sentence(tries=t)
        if sentence and any(word in sentence for word in vocab1 + vocab2):
            matched_sentences.append(sentence.strip())

    paragraph = " ".join(matched_sentences)

    return paragraph

def st_generate_output_text(user_input, mode):
    """
    Generate an output text in streamlit
    """
    output = "Generation failed."

    themes = extract_themes(user_input)
    print(themes)
    emotions = extract_emotions(user_input)
    print(emotions)
    print(emotions[0])

    # detect if the themes are food-heavy-loaded or space-heavy-loaded
    density_food = sum(1 for t in themes if t in food_vocab + flavor_vocab) / len(themes)
    density_space = sum(1 for t in themes if t in furniture_vocab + atmosphere_vocab) / len(themes)

    if density_food > 0.15:
        mode = "Interior"
        st.info("Food-heavy input detected. Translating into interior space.")
    if density_space > 0.15:
        mode = "Culinary"
        st.info("Space-heavy input detected. Translating into culinary taste.")

    # get a number of sentences in the input
    user_text = nlp(user_input)
    user_sentence = list(user_text.sents)
    num_sentence = len(user_sentence)

    # translate and generate an output text of the domain
    if mode == "Culinary":
        food_mapped = map_theme_to_content(themes, food_vocab)
        print(food_mapped)
        flavor_mapped = map_emotion_to_sense(emotions, themes, flavor_vocab)
        print(flavor_mapped)

        description_food = generate_narration_food(food_mapped, flavor_mapped)

        # select a corpus text by the main mood of the input
        if emotions[0] == "anger":
            corpus_text_food = corpus_text_food_anger
        elif emotions[0] == "disgust":
            corpus_text_food = corpus_text_food_disgust
        elif emotions[0] == "fear":
            corpus_text_food = corpus_text_food_fear
        elif emotions[0] == "joy":
            corpus_text_food = corpus_text_food_joy
        elif emotions[0] == "neutral":
            corpus_text_food = corpus_text_food_neutral
        elif emotions[0] == "sadness":
            corpus_text_food = corpus_text_food_sadness
        elif emotions[0] == "surprise":
            corpus_text_food = corpus_text_food_surprise
        else:
            corpus_text_food = corpus_text_food_neutral # use the neutral corpus as the default

        output = generate_output_text(description_food, corpus_text_food, food_mapped, flavor_mapped, k=num_sentence+round(num_sentence/0.5))

    else:
        bridge_mapped = map_theme_to_content(themes, bridge_vocab) # semantic bridge
        print(bridge_mapped)
        furniture_mapped = map_theme_to_content(bridge_mapped, furniture_vocab)
        print(furniture_mapped)
        atmosphere_mapped = map_emotion_to_sense(emotions, bridge_mapped, atmosphere_vocab)
        print(atmosphere_mapped)

        description_home = generate_narration_home(furniture_mapped, atmosphere_mapped)

        # select a corpus text by the main mood of the input
        if emotions[0] == "anger":
            corpus_text_home = corpus_text_home_anger
        elif emotions[0] == "disgust":
            corpus_text_home = corpus_text_home_disgust
        elif emotions[0] == "fear":
            corpus_text_home = corpus_text_home_fear
        elif emotions[0] == "joy":
            corpus_text_home = corpus_text_home_joy
        elif emotions[0] == "neutral":
            corpus_text_home = corpus_text_home_neutral
        elif emotions[0] == "sadness":
            corpus_text_home = corpus_text_home_sadness
        elif emotions[0] == "surprise":
            corpus_text_home = corpus_text_home_surprise
        else:
            corpus_text_home = corpus_text_home_neutral # use the neutral corpus as the default

        output = generate_output_text(description_home, corpus_text_home, furniture_mapped, atmosphere_mapped, k=num_sentence+round(num_sentence/0.5))


    return output











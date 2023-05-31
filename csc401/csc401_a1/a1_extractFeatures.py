#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import csv
# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCTUATION = { '!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~'}

bglpath = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
warpath = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
# bglpath = './Wordlists/BristolNorms+GilhoolyLogie.csv'
# warpath = './Wordlists/Ratings_Warriner_et_al.csv'

altIDs = '/u/cs401/A1/feats/Alt_IDs.txt'
rightIDs = '/u/cs401/A1/feats/Right_IDs.txt'
centerIDs = '/u/cs401/A1/feats/Center_IDs.txt'
leftIDs = '/u/cs401/A1/feats/Left_IDs.txt'
altFeats = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
leftFeats = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
rightFeats = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
centerFeats = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')

# altIDs = './feats/Alt_IDs.txt'
# rightIDs = './feats/Right_IDs.txt'
# centerIDs = './feats/Center_IDs.txt'
# leftIDs = './feats/Left_IDs.txt'
# altFeats = np.load('./feats/Alt_feats.dat.npy')
# leftFeats = np.load('./feats/Left_feats.dat.npy')
# rightFeats = np.load('./feats/Right_feats.dat.npy')
# centerFeats = np.load('./feats/Center_feats.dat.npy')

BGLDICT = {}
with open(bglpath, newline='') as f:
    for row in csv.DictReader(f):
        BGLDICT[row['WORD']] = {
            "AoA": row["AoA (100-700)"],
            "IMG": row["IMG"],
            "FAM": row["FAM"]
        }
WARDICT = {}
with open(warpath, newline='') as f:
    for row in csv.DictReader(f):
        WARDICT[row['Word']] = {
            "V.Mean.Sum": row["V.Mean.Sum"],
            "A.Mean.Sum": row["A.Mean.Sum"],
            "D.Mean.Sum": row["D.Mean.Sum"]
        }

LIWC_ID_DICT = {"Alt": {}, "Center": {}, "Left": {}, "Right": {}}
for i, id_file_name in enumerate([altIDs, rightIDs, centerIDs, leftIDs]):
    if i == 0:
        cat = "Alt"
    elif i == 1:
        cat = "Right"
    elif i == 2:
        cat = "Center"
    elif i == 3:
        cat = "Left"
    with open(id_file_name, "r") as id_file:
        for index, comment_id in enumerate(id_file):
            LIWC_ID_DICT[cat][comment_id.strip("\n")] = index

LIWC_DICT = {
    "Alt": altFeats, "Center": centerFeats, "Left": leftFeats, "Right": rightFeats}


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    feats = np.zeros(173)
    # pre_process comment
    sentences = comment.split("\n")
    all_words = []
    tokens = []
    tags = []
    for sentence in sentences:
        words = sentence.split(" ")
        all_words.extend(words)
        for word in words:
            comp = word.split("/")
            if len(comp) == 2:
                tokens.append(comp[0])
                tags.append(comp[1])

    # feat1 num of uppercase token len >3
    feat1 = 0
    new_tokens = []
    for token in tokens:
        new_tokens.append(token.lower())
        if len(token) > 3 and token.isupper():
            feat1 += 1
    feats[0] = feat1

    # change all to lowercase after feat1
    tokens = new_tokens

    # feat 2,3,4
    first_pp = 0
    second_pp = 0
    third_pp = 0

    # feat 18-29
    aoa = []
    img = []
    fam = []
    v = []
    a = []
    d = []
    punc_only_num = 0
    for token in tokens:
        if token in FIRST_PERSON_PRONOUNS:
            first_pp += 1
        if token in SECOND_PERSON_PRONOUNS:
            second_pp += 1
        if token in THIRD_PERSON_PRONOUNS:
            third_pp += 1
        # feat9 multi punctuation
        punc_num = 0
        for char in token:
            if char not in PUNCTUATION:
                break
            else:
                punc_num += 1
        if len(token) > 1 and len(token) == punc_num:
            feats[8] += 1
        # feat 16 avg length of non punc-only token
        if len(token) != punc_num:
            feats[15] += len(token)  # later divide by punc_only_num
            punc_only_num += 1
        # feat 14 slang
        if token in SLANG:
            feats[13] += 1

        if len(token) > 0 and token in BGLDICT:
            aoa.append(int(BGLDICT[token]["AoA"]))
            img.append(int(BGLDICT[token]["IMG"]))
            fam.append(int(BGLDICT[token]["FAM"]))

        if len(token) > 0 and token in WARDICT:
            v.append(float(WARDICT[token]["V.Mean.Sum"]))
            a.append(float(WARDICT[token]["A.Mean.Sum"]))
            d.append(float(WARDICT[token]["D.Mean.Sum"]))

    for i in range(len(tags)):
        # feat 5 coordinating conjunctions
        if tags[i] == 'CC':
            feats[4] += 1
        # feat 6 past tense verb
        if tags[i] == "VBD":
            feats[5] += 1
        # feat 7 future tense ver
        if tags[i] == "VB" and i >= 2:
            if tokens[i - 2] == 'going' and tokens[i - 1] == 'to':
                feats[6] += 1
        if tags[i] == "VB" and i >= 1:
            if tokens[i - 1] in ["'ll", "will", "gonna", "shall"]:
                feats[6] += 1
        # feat 8 commas
        if tags[i] == ",":
            feats[7] += 1
        # feat 10 common nouns
        if tags[i] == "NN" or tags[i] == "NNS":
            feats[9] += 1
        # feat 11 proper nouns
        if tags[i] == "NNP" or tags[i] == "NNPS":
            feats[10] += 1
        # feat 12 adverbs
        if tags[i] == "RB" or tags[i] == "RBR" or tags[i] == "RBS":
            feats[11] += 1
        # feat 13 wh-words
        if tags[i] == "WDT" or tags[i] == "WP" or tags[i] == "WP$" or tags[i] == "WRB":
            feats[12] += 1

    feats[1] = first_pp
    feats[2] = second_pp
    feats[3] = third_pp
    if len(tokens) > 0:
        feats[15] /= punc_only_num
    # feat 15 17 average num token per sentence and number of sentence
    feats[14] = len(tokens) / len(sentences)
    feats[16] = len(sentences)
    # feat 18-29
    if len(aoa) > 0:
        feats[17] = np.mean(aoa)
        feats[20] = np.std(aoa)
    if len(img) > 0:
        feats[18] = np.mean(img)
        feats[21] = np.std(img)
    if len(fam) > 0:
        feats[19] = np.mean(fam)
        feats[22] = np.std(fam)
    if len(v) > 0:
        feats[23] = np.mean(v)
        feats[26] = np.std(v)
    if len(a) > 0:
        feats[24] = np.mean(a)
        feats[27] = np.std(a)
    if len(d) > 0:
        feats[25] = np.mean(d)
        feats[28] = np.std(d)
    return feats


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    feature_index = LIWC_ID_DICT[comment_class][comment_id]
    feature = LIWC_DICT[comment_class][feature_index]
    feat[29:173] = feature



def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    cat_classes = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}
    # TODO: Call extract1 for each datatpoint to find the first 29 features.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    for i, comment in enumerate(data):
        feats_extract1 = extract1(comment['body'])
        feats_extract2 = feats_extract1[:]
        extract2(feats_extract2, comment['cat'], comment['id'])
        feats[i, :-1] = feats_extract2
        feats[i, -1] = cat_classes[comment['cat']]

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)


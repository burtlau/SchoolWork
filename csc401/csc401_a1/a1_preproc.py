#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  
        #modify this to handle other whitespace chars.
        #replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        
    if 4 in steps: #remove duplicate spaces.
        modComm = " ".join(modComm.split())

    if 5 in steps:
        # TODO: get Spacy document for modComm
        utt = nlp(modComm)
        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
        modComm = ""
        for sent in utt.sents:
            sentence = []
            for token in sent:
                word = token.lemma_
                if word[0] == '-':
                    word = token.text
                word = word + "/" + token.tag_
                sentence.append(word)
            modComm += " ".join(sentence) + '\n'


            
    
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            data = json.load(open(fullFile))
            # TODO: select appropriate args.max lines
            start = args.ID[0] % len(data)
            end = start + args.max
            if end < len(data):
                select_data = data[start: end]
            else:
                front_end = start + args.max - len(data)
                select_data = data[start: len(data)] + data[: front_end]
            # TODO: read those lines with something like `j = json.loads(line)`
            for line in select_data:
                j = json.loads(line)
                comment = {}
            # TODO: choose to retain fields from those lines that are relevant to you
                comment['id'] = j['id']
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                comment['cat'] = file
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
                comment['body'] = preproc1(j["body"])
            # TODO: append the result to 'allOutput'
                allOutput.append(comment)
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)


    indir = os.path.join(args.a1_dir, 'data')
    main(args)

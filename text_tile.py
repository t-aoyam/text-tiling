import nltk
import numpy as np
import argparse

tokenizer = nltk.word_tokenize
stemmer = nltk.stem.PorterStemmer()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file_path", type=str,
                        help="path for the text to be TextTiled")
    parser.add_argument("--seq_size", "-s", type=int, default=20,
                        help="the size of token sequence. hyperparameter k in Hearst (1997)")
    parser.add_argument("--num_bound", "-n", type=int, default=0,
                        help="number of boundaries. if not specified (i.e., 0), it will be automatically calculated based on the distribution of the height scores")
    parser.add_argument("--stem", action="store_true", default=False,
                        help="stem each token before tiling")
    parser.add_argument("--segment", action="store_true", default=False,
                        help="segment the text into real sentences. default is false because the input is expected to be separable with '\n'")

    args = parser.parse_args()
    text_file_path = args.text_file_path
    seq_size = args.seq_size
    num_bound = args.num_bound
    stem = args.stem
    segment = args.segment

    with open(text_file_path, encoding="utf-8") as f:
        sents = [sent.strip().lower() for sent in f.read().split('\n') if sent.strip() != '' ]

    with open(text_file_path, encoding="utf-8") as f:
        text = f.read()
    
    if segment:
        text_tiler = TextTiler(text=text, k=seq_size, num_bound=num_bound, stem=stem)
    else:
        text_tiler = TextTiler(sents=sents, k=seq_size, num_bound=num_bound, stem=stem)
    print(text_tiler.aligned_bin)

class TextTiler:
    def __init__(self, text=None, sents=None, k=20,
                 num_bound=0, stem=True, stop_words=None):
        if text==None and sents==None:
            raise ValueError('Either text in str or sentences in list has to be provided.')
        self.k = k
        self.stem = stem
        self.num_bound = num_bound
        self.text = text
        self.sents = sents
        if self.sents == None:
            self.sents= self.sent_segment()
        if self.text == None:
            self.text = ' '.join(self.sents)
        self.stop_words = stop_words
        self.tokenized = self.tokenize()
        self.vocab = set()
        self.new_vocabs, self.scores, = self.vocab_intro()
        self.heights = self.height_score()
        self.boundaries = self.detect_boundaries()
        self.aligned_bin, self.alignment, self.aligned_vocab = self.align_real_sentence()
    
    def tokenize(self):
        tokenized = tokenizer(self.text)
        if self.stem:
            tokenized = [stemmer.stem(token) for token in tokenized]
        return tokenized

    def sent_segment(self):
        sents = nltk.sent_tokenize(self.text)
        return sents
    
    def vocab_intro(self):
        if self.stop_words == None:
            self.stop_words = nltk.corpus.stopwords.words('english')
        scores = []
        new_vocabs = []
        num_gap = len(self.tokenized)//self.k
        for i in range(num_gap):
            gap_id = i + 1
            gap_loc = gap_id * self.k
            sent_1 = self.tokenized[gap_loc - self.k : gap_loc]
            sent_1 = [tok for tok in sent_1 if tok not in self.stop_words]
            if gap_id != num_gap:
                sent_2 = self.tokenized[gap_loc : gap_loc + self.k]
            else:
                sent_2 = self.tokenized[gap_loc :]
            sent_2 = [tok for tok in sent_2 if tok not in self.stop_words]
            new_vocab = set(sent_1 + sent_2).difference(self.vocab)
            num_new = len(new_vocab)
            new_vocabs.append(new_vocab)
            scores.append(num_new)
            self.vocab.update(set(sent_1 + sent_2))
        return new_vocabs, scores

    def height_score(self):
        heights = []
        for i, score in enumerate(self.scores):
            if i == 0 or i == len(self.scores) - 1:
                continue
            left = score - self.scores[i-1]
            right = score - self.scores[i+1]
            heights.append(left+right)
        return heights

    def detect_boundaries(self):
        if self.num_bound == 0:  # when the num_bound is unspecified, cutoff = mean + std
            mean = np.mean(self.heights)
            sigma = np.std(self.heights)
            cutoff = mean + sigma
            boundaries = [i+1 for i, height in enumerate(self.heights) if height >= cutoff]
        else:  # simply take the top n boundaries
            height_indices = [(i+1, height) for i, height in enumerate(self.heights)]
            height_indices.sort(key=lambda x:x[1], reverse=True)
            boundaries = [index for index, _ in height_indices[:self.num_bound-1]]
        boundaries.append(0)  # the first token sequence has to be a start of a new segment
        return sorted(boundaries)
    
    def align_real_sentence(self):
        aligned_locs = []
        sent_locs = []
        alignment = {}
        current = 0
        for i, sent in enumerate(self.sents):
            sent_locs.append((i, current))
            current += len(tokenizer(sent))
#        print(sent_locs)
        assert(len(sent_locs) == len(self.sents))

        boundary_locs = [boundary*self.k for boundary in self.boundaries]
        for boundary_loc in boundary_locs:
            min_dif = np.inf
            aligned_loc = 0
            for i, sent_loc in sent_locs:                
                dif = abs(boundary_loc - sent_loc)
                if dif < min_dif:
                    min_dif = dif
                    aligned_loc = i
            alignment[aligned_loc] = int(boundary_loc/self.k)
            aligned_locs.append(aligned_loc)
            aligned_bin = [1 if i in aligned_locs else 0 for i, _ in sent_locs]
            aligned_vocab = [{} if binary == 0 else self.new_vocabs[alignment[i]] for i, binary in enumerate(aligned_bin)]
        return aligned_bin, alignment, aligned_vocab

if __name__ == "__main__":
    main()
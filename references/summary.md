# MANDARIN TONE MODELING USING RECURRENT NEURAL NETWORKS

- ASR (*GMM*+HMM) -> phone boundaries and pitch -> RNN encoder -> pooling -> embedding -> softmax
- The 863 Project, 110-hour recordings spoken by 160 speakers (80 females and 80 males). 92,243 utterances (86,271 used)
- Inputs to RNN are *frames*, output the embedding of a segment
- cut by full syllables (声母+韵母?)
- add duration as a feature
- contextual embedding: input of softmax layer is (h1 + h2 + h3) where h1, h2, h3 are:
    - the average hidden vectors of the preceding syllable
    - ... of the current syllable
    - ... of the succeeding syllable

Shortcomings:

- GMM
- pitch hard to track especially when noise exists
- using full syllable, is it the better?
- not utilizing RNN to include contextual information across syllables (using average of hidden states is probably not
  so good)

# IMPROVING MISPRONUNCIATION DETECTION AND ENRICHING - CHAPTER 5

- Corpus: The 863 Project and iCALL
- *GMM*+HMM -> frame-level posteriors and phone boundaries -> DNN baseline and BLSTM -> Use ERN+BLSTM to generate soft
  targets (TODO: how?)
- The input of classifier is the posterior of tones calculated from the output posteriors of ASR model
- For DNN, each sample is the posterior of each *segment* obtained by averaging frame posteriors
- For BLSTM each sample is a sequence of posteriors of each *frame*
- Soft targets (more suitable for non-native pronunciation than hard targets), output "correct" or "mispronounced"

Shortcomings:

- GMM
- Uses posteriors from ASR model (probably not good)
- No contextual information between phones (LSTM is only used within the duration of a phone)

# My Model

- DNN+HMM ASR -> phone boundaries + annotated phone labels -> mel-spectrogram -> Resnet34StatsPool -> tone embedding ->
  transformer/LSTM

Advantage:

- State-of-the-art kaldi ASR
- Mel-spectrogram contains more information than F0 or pitch, more robust against noise
- Contextual information across multiple syllables thanks to RNN/Transformer
- "Triphone" (TODO: read more about "overlapped ditone"), half of previous segment, this segment, and half of next
  segment. This is useful because of tone co-articulation
import sentencepiece as spm

corpus_path = "arabe_corpus.txt"

model_prefix = "arabe_tokenizador"

vocab_size = 32000
character_coverage = 1.0

spm.SentencePieceTrainer.train(
    input=corpus_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    model_type='bpe'  
)

print("Tokenizador entrenado y guardado con el prefijo:", model_prefix)

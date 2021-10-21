import youtokentome as yttm

train_data_path = '/content/raw_train_librespeech_texts.txt'
model_path = "bpe_train_libreespeech.model"

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)


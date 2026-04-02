*HERE we first use MTCNN to detect face from and image in dataset *
then we resize it to 160*160 pixels

after we get face pixels  we use FACENET to extract face embeddings( vectors ) from that photo


for each photo we save an embedding (ex [0.99,0.23,.....] )in new line in the file "embeddings.npy"
and for each embedding we store a label (name of person) in label.npy file



then we use SVM model to learn embeddings and label


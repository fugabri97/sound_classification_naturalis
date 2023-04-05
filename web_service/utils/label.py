from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

CLASSES = ['bird', 'insect', 'speech']

def decode_label(enc_label):
    le = LabelEncoder()
    le.fit_transform(CLASSES)

    return le.inverse_transform(enc_label)
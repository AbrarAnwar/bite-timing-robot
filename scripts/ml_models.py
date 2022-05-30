from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
# from preprocessing import load_data
# from preprocessing import load_data_single
from sklearn.metrics import classification_report
import pickle


def SVM(c, kernel):
    clf = svm.SVC(C=c, kernel=kernel)
    return clf


def SGD():
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log')
    return clf


def MLP():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=300)
    return clf


model_list = [SVM(c=1, kernel='linear'), svm.SVC(C=1, gamma='scale', kernel='rbf'), SGD(), MLP()]
model_name = ["Linear SVM", "RBF SVM", "SGD", "MLP"]

models = {"Linear SVM": SVM(c=1, kernel='linear'),
            "RBF SVM": SVM(c=1, kernel='rbf'),
            "SGD": SGD(), 
            "MLP": MLP()
        }
models_path = {"Linear SVM":'/home/abrar/feeding_ws/src/bite-timing-robot/weights/linear_svm_social_15fps_nohand_all_audio.pkl'}



class SocialDiningModel:
    def __init__(self, name):
        self.name = name
        # model is based on name
        self.model = models[name]

        # load model accordingly
        self.load(models_path[name])

    def predict(self, X):
        return self.model.predict(X)

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

# import pdb; pdb.set_trace()
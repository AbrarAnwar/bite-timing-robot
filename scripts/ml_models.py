from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
# from preprocessing import load_data
# from preprocessing import load_data_single
from sklearn.metrics import classification_report
import pickle
from models.interleaved_net import PazNet
from models.tcn_model import TCNModels

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

paznet = PazNet()

models = {"Linear SVM": SVM(c=1, kernel='linear'),
            "RBF SVM": SVM(c=1, kernel='rbf'),
            "SGD": SGD(), 
            "MLP": MLP(),
            "tong_paznet" : paznet,
            "my_paznet" : paznet,
            'full_ssp_paznet' : paznet,
            "tcn_global": TCNModels(),
            
        }
models_path = {"Linear SVM":'/home/abrar/feeding_ws/src/bite-timing-robot/weights/linear_svm_social_15fps_nohand_all_audio.pkl',
                "tong_paznet": '/home/abrar/feeding_ws/src/bite-timing-robot/weights/interleaved_net_6s_he_nohand_15fps_0.001_0.0_0.005_32.h5',
                "my_paznet": '/home/abrar/feeding_ws/src/bite-timing-robot/weights/interleaved_paznet_body_face_gaze_headpose_speaking_loso-session_14_0.001_0.5_0.0001_128.h5',
                "tcn_global": '/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_70:30_0_ssp_0.001_0.5_0.0001_128.h5',
                'tcn_global_no_scaling': '/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_no_scaling_70:30_0_ssp_0.001_0.5_0.0001_128.h5',
                'tcn_global_no_rt_gene': '/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_num_bites_speaking_time_since_last_bite_15fps_no_scaling_70:30_0_ssp_0.001_0.5_0.0001_128.h5',
                'tcn_global_only_speaking': '/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_num_bites_speaking_time_since_last_bite_15fps_no_scaling_70:30_0_0.001_0.5_0.0001_128.h5',
                'tcn_global_only_speaking_no_global':  '/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_speaking_15fps_no_scaling_70:30_0_0.001_0.5_0.0001_128.h5',
                'full_ssp_paznet': '/home/abrar/feeding_ws/src/bite-timing-robot/weights/i_paznet_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_filter_scale2_15fps_70:30_0_full_ssp_rep100_0.001_0.5_0.0001_128.h5',
        }




class SocialDiningModel:
    def __init__(self, name):
        self.name = name
        # model is based on name
        if 'tcn' in name:
            name = 'tcn_global'
        self.model = models[name]


        # load model accordingly
        self.load(models_path[self.name])

    def predict(self, X):
        # if type(X) is list:
        #     return self.model.predict(*X)

        return self.model.model.predict(X)

    def load(self, path):
        print(self.name, "loading model from", path)
        if 'paznet' in self.name:
            self.model.load(path)
        elif 'tcn' in self.name:
            self.model.load(path)
        else:
            self.model = pickle.load(open(path, 'rb'))

# sdm = SocialDiningModel('tcn_global_only_speaking_no_global')

# import pdb; pdb.set_trace()
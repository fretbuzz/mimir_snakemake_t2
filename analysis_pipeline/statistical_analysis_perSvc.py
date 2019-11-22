'''
This function will implement the per-svc exfiltration model (maybe modded to be a hybrid at some point, but not now...)
'''
import sklearn

class persvc_ensemble_exfil_model():
    '''This class combines (takes an OR) of the different timegran ensemble models'''
    def __init__(self, time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, recipes_used,
                 skip_model_part, avg_exfil_per_min, avg_pkt_size, exfil_per_min_variance, pkt_size_variance,
                 no_labeled_data):
        print "persvc_ensemble_exfil_model"
        self.report_sections = {}
        self.list_of_optimal_fone_scores_at_this_exfil_rates = {}
        self.Xs = {}
        self.Ys = {}
        self.Xts = {}
        self.Yts = {}
        self.no_labeled_data = no_labeled_data
        self.trained_models = {}
        self.timegran_to_methods_to_attacks_found_dfs = {}
        self.timegran_to_methods_toattacks_found_training_df = {}
        self.time_gran_to_outtraffic = {}
        self.timegran_to_statistical_pipeline = {}
        self.base_output_name = base_output_name + '_lasso'
        self.time_gran_to_aggregate_mod_score_dfs = time_gran_to_aggregate_mod_score_dfs
        self.time_gran_to_predicted_test = {}
        self.time_gran_to_cm = {}

        for timegran, feature_df in self.time_gran_to_aggregate_mod_score_dfs.iteritems():
            self.list_of_optimal_fone_scores_at_this_exfil_rates[timegran] = []
            self.Xs[timegran] = []
            self.Ys[timegran] = []
            self.Xts[timegran] = []
            self.Yts[timegran] = []
            self.trained_models[timegran] = []
            self.time_gran_to_outtraffic[timegran] = []
            self.skip_model_part = skip_model_part

        self.recipes_used = recipes_used
        self.avg_exfil_per_min = avg_exfil_per_min
        self.avg_pkt_size = avg_pkt_size
        self.exfil_per_min_variance = exfil_per_min_variance
        self.pkt_size_variance = pkt_size_variance

    def train_pergran_models(self):
        # okay, first let's put the correct values into Xs,Ys,Xts,Yts
        # TODO

        # second, let's feed the correct values into the single_timegran_persvc_ensemble_exfil_model instances...
        # TODO

        # finally, put the model instances into the trained_models dict
        # TODO

        pass

    def apply_model_to_new_data(self):
        # TODO TODO TODO
        pass

    def generate_reports(self):
        # TODO TODO TODO
        pass

class single_timegran_persvc_ensemble_exfil_model():
    '''This class is an ensemble model of many models that each determine whether a particular svc is involved in the exfil'''

    def __init__(self, Xs, Ys, Xts, Yts):
        # Xs, Ys, Xts, and Yts are each dicts mapping a service to it's set of features/labels
        self.Xs = Xs
        self.Ys = Ys
        self.Xts = Xts
        self.Yts = Yts

        self.svc_to_model = {}

    def train_single_timegran_persvc_ensemble_model(self):
        for svc, X in self.Xs.iteritems():
            Y = self.Ys[svc]
            Xt = self.Xts[svc]
            Yt = self.Yts[svc]
            self.svc_to_model[svc] = single_svc_exfil_model(X, Y, Xt, Yt)

        for svc, single_svc_model in self.svc_to_model:
            single_svc_model.train_single_svc_exfil_model()

        # TODO: insert boosting here... (or is this even boosting??)

    def apply_single_timegran_persvc_ensemble_model_to_new_data(self):
        pass

class single_svc_exfil_model():
    '''This class is an ML model for if a particular svc is involved in the exfil'''
    def __init__(self, X, Y, Xt, Yt):
        self.X = X
        self.Y = Y
        self.Xt = Xt
        self.Yt = Yt

    def train_single_svc_exfil_model(self):
        clf = sklearn.linear_model.LogisticRegressionCV()

    def apply_single_svc_exfil_model_to_new_data(self):
        pass
'''
This function will implement the per-svc exfiltration model (maybe modded to be a hybrid at some point, but not now...)
'''
import sklearn
import process_roc
import generate_report
from jinja2 import FileSystemLoader, Environment
import pandas as pd
from statistical_analysis import construct_ROC_curve, drop_useless_columns_testTrain_Xs, drop_useless_columns_aggreg_DF
import numpy as np
import copy
import time

# FOR A PARTICUALR EXFIL RATE (implicitly, because that determines the input data...)
class exfil_detection_model():
    '''This class combines (takes an OR) of the different timegran ensemble models'''
    def __init__(self, time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, recipes_used,
                 skip_model_part, avg_exfil_per_min, avg_pkt_size, exfil_per_min_variance, pkt_size_variance,
                 no_labeled_data):

        time_gran_to_aggregate_mod_score_dfs = copy.deepcopy(time_gran_to_aggregate_mod_score_dfs)

        print "persvc_ensemble_exfil_model"
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
        self.base_output_name = base_output_name
        self.time_gran_to_aggregate_mod_score_dfs = time_gran_to_aggregate_mod_score_dfs
        self.time_gran_to_predicted_test = {}
        self.time_gran_to_cm = {}
        self.ROC_curve_p = ROC_curve_p
        self.time_gran_to_predicted_test = None
        self.type_of_model_to_time_gran_to_cm = {}
        self.type_of_model_to_time_gran_to_predicted_test = {} # rate_to_time_gran_to_predicted_test
        self.model_to_tg_report_sections = {}

        ###### # this list controls which individual models are calculated
        self.types_of_models = ['lasso', 'logistic', 'boosting lasso', 'boosting logisitic' 'persvc_boosting']
        ######

        for timegran, feature_df in self.time_gran_to_aggregate_mod_score_dfs.iteritems():
            self.list_of_optimal_fone_scores_at_this_exfil_rates[timegran] = []
            training_feature_df = feature_df[feature_df['is_test'] == 0]
            testing_feature_df = feature_df[feature_df['is_test'] == 1]

            self.Xs[timegran]  = [training_feature_df.loc[:, training_feature_df.columns != 'labels']]
            self.Ys[timegran]  = [training_feature_df.loc[:, training_feature_df.columns == 'labels']]
            self.Xts[timegran] = [testing_feature_df.loc[:, testing_feature_df.columns != 'labels']] # the t is for test
            self.Yts[timegran] = [testing_feature_df.loc[:, testing_feature_df.columns == 'labels']]

            self.time_gran_to_outtraffic[timegran] = []
            self.skip_model_part = skip_model_part

        for type_of_model in self.types_of_models:
            self.trained_models[type_of_model] = {}
            for timegran in self.Xs.keys():
                self.trained_models[type_of_model][timegran] = []

        self.recipes_used = recipes_used
        self.avg_exfil_per_min = avg_exfil_per_min
        self.avg_pkt_size = avg_pkt_size
        self.exfil_per_min_variance = exfil_per_min_variance
        self.pkt_size_variance = pkt_size_variance

    def train_pergran_models(self):
        # first, let's feed the correct values into the single_timegran_exfil_model instances...
        # TODO


        for type_of_model in self.types_of_models:
            self.type_of_model_to_time_gran_to_cm[type_of_model] = {}
            for timegran in self.Xs.keys():
                # this if-block exists b/c not all the model types are implemented...
                if type_of_model in ['lasso', 'logistic', 'boosting lasso', 'boosting logisitic']:
                    self.trained_models[type_of_model][timegran] = \
                        single_timegran_exfil_model(self.Xs[timegran], self.Ys[timegran], self.Xts[timegran],
                                                    self.Yts[timegran], type_of_model, timegran, self.base_output_name, self.recipes_used)

                    self.trained_models[type_of_model][timegran].train()

                    self.type_of_model_to_time_gran_to_cm[type_of_model][timegran] = self.trained_models[type_of_model][
                        timegran].train_confusion_matrix

                '''
                elif type_of_model == 'logistic':
                    self.trained_models[type_of_model][timegran] = \
                        single_timegran_exfil_model(self.Xs[timegran], self.Ys[timegran], self.Xts[timegran],
                                                    self.Yts[timegran], 'logistic', timegran, self.base_output_name, self.recipes_used)

                    self.trained_models[type_of_model][timegran].train()

                    self.type_of_model_to_time_gran_to_cm[type_of_model][timegran] = self.trained_models[type_of_model][
                        timegran].train_confusion_matrix

                elif type_of_model == 'boosting':
                    # TODO
                    pass
                elif type_of_model == 'persvc_boosting':
                    # TODO
                    pass
                '''

                # when all the model types are implemented, I can then remove the above if-block and enable the code below
                #self.trained_models[type_of_model][timegran] = single_timegran_exfil_model(self.Xs[timegran], self.Ys[timegran],
                #                                                        self.Xts[timegran], self.Yts[timegran], type_of_model, timegran)

                #print " self.trained_models[type_of_model][timegran]",  self.trained_models[type_of_model][timegran]

    def apply_to_new_data(self, time_gran_to_aggregate_mod_score_df, cur_base_output_name, recipes_used, avg_exfil_per_min,
                          avg_pkt_size, exfil_per_min_variance, pkt_size_variance):

        self.base_output_name = cur_base_output_name
        self.recipes_used = recipes_used
        self.avg_exfil_per_min = avg_exfil_per_min
        self.avg_pkt_size = avg_pkt_size
        self.exfil_per_min_variance = exfil_per_min_variance
        self.pkt_size_variance = pkt_size_variance

        if time_gran_to_aggregate_mod_score_df:
            for timegran, feature_df in time_gran_to_aggregate_mod_score_df.iteritems():
                self.Xts[timegran] = feature_df.loc[:, feature_df.columns != 'labels']  # the t is for test
                self.Yts[timegran] = feature_df.loc[:, feature_df.columns == 'labels']

        #print "self.Xts", self.Xts
        #print "self.Yts", self.Yts

        timegran_to_alerts = {}
        for type_of_model in self.types_of_models:
            self.type_of_model_to_time_gran_to_predicted_test[type_of_model] = {}
            for timegran in self.Xts.keys():
                if type(self.trained_models[type_of_model][timegran]) != list:
                    self.trained_models[type_of_model][timegran].apply_to_new_data(self.Xts[timegran], self.Yts[timegran],
                                                                                   self.base_output_name, self.recipes_used,
                                                                                   self.avg_exfil_per_min, self.exfil_per_min_variance)
                    timegran_to_alerts[timegran] = self.trained_models[type_of_model][timegran].yt_optimal_thresholded

                    #if self.Yts is None:
                    self.type_of_model_to_time_gran_to_predicted_test[type_of_model] = timegran_to_alerts
                    #else:
                    self.type_of_model_to_time_gran_to_cm[type_of_model][timegran] = \
                            self.trained_models[type_of_model][timegran].test_confusion_matrix

            # now need to combine the alerts at different time granularities
            ensemble_timegran = '(' + ','.join(self.Xts.keys()) + ')'
            multi_time_object = multi_time_alerts(self.type_of_model_to_time_gran_to_predicted_test[type_of_model],
                                                  self.Yts[0], self.Xts[0], self.base_output_name, type_of_model, self.recipes_used)

            alerts_for_multitime = multi_time_object.alerts
            self.type_of_model_to_time_gran_to_predicted_test[type_of_model][ensemble_timegran] = alerts_for_multitime
            self.trained_models[type_of_model][ensemble_timegran] = multi_time_object
            self.type_of_model_to_time_gran_to_cm[type_of_model][ensemble_timegran] = multi_time_object.confusion_matrix

    def generate_reports(self, auto_open_p, skip_heatmaps, using_pretrained_model):
        ## TODO: I need to add the other alert calculation methods (e.g., cilium, ide)
        # (^^ probably as a list of the alert levels, and then can find when the alerts are coming here)

        for type_of_model in self.trained_models.keys():
            self.model_to_tg_report_sections[type_of_model] = {}
            for timegran in self.trained_models[type_of_model].keys():
                # if model is not implemented yet, then self.trained_models[type_of_model][timegran] will be a list
                if type(self.trained_models[type_of_model][timegran]) != list:
                    self.model_to_tg_report_sections[type_of_model][timegran] = \
                        self.trained_models[type_of_model][timegran].generate_report_section(skip_heatmaps,
                                                                                             using_pretrained_model)

        for type_of_model, report_sections in self.model_to_tg_report_sections.iteritems():
            first_time_gran = self.trained_models[type_of_model].keys()[0]
            if type(self.trained_models[type_of_model][first_time_gran]) != list:
                output_location = self.base_output_name
                if using_pretrained_model:
                    output_location += '_' + str(self.avg_exfil_per_min) + ":"  + str(self.exfil_per_min_variance)

                output_location += '_' + type_of_model + 'NEW_MODEL'
                print "current report output loc", output_location
                print "self.recipes_used", self.recipes_used

                generate_report.join_report_sections(self.recipes_used, output_location, self.avg_exfil_per_min,
                                                     self.avg_pkt_size, self.exfil_per_min_variance, self.pkt_size_variance,
                                                     report_sections, auto_open_p, new_model=True)

                #time.sleep(3600) # TODO: <--- remove!!!

                #print "sleeping for a bit..."
                #time.sleep(3600)


class single_timegran_exfil_model():
    '''This class is an ensemble model of many models that each determine whether a particular svc is involved in the exfil'''

    def __init__(self, X, Y, Xt, Yt, model_to_fit, timegran, base_output_name, recipes_used):
        # might be oversimplifying with the [0]???
        Xt = Xt[0]
        X = X[0]
        Yt = Yt[0]
        Y = Y[0]

        exfil_paths = Xt['exfil_path']
        exfil_paths_train = X['exfil_path']

        try:
            X = X.drop(columns='real_ide_angles_')
        except:
            pass

        try:
            Xt = Xt.drop(columns='real_ide_angles_')
        except:
            pass

        Xt = drop_useless_columns_aggreg_DF(Xt)
        X = drop_useless_columns_aggreg_DF(X)

        # if having problems with X_train_exfil_weight, can try using the value returned by this func...
        X, Xt, dropped_feature_list, X_train_exfil_weight, X_test_exfil_weight = drop_useless_columns_testTrain_Xs(X, Xt)

        #print "Xt", Xt

        try:
            exfil_paths = exfil_paths.replace('0', '[]')
            exfil_paths_train = exfil_paths_train.replace('0', '[]')
        except:
            try:
                exfil_paths = exfil_paths.replace(0, '[]')
                exfil_paths_train = exfil_paths_train.replace(0, '[]')
            except:
                pass

        #X_train_exfil_weight = exfil_paths_train
        #X_test_exfil_weight = exfil_paths

        # TODO: need to add in the dropping of the other stuff too (honestly, just calling the function might be easiest...)
        # Xs, Ys, Xts, and Yts are each dicts mapping a service to it's set of features/labels
        self.X = X
        self.Y = Y
        self.Xt = Xt
        self.Yt = Yt

        self.svc_to_model = {}
        self.timegran = timegran

        self.model_to_fit = model_to_fit
        self.test_predictions = None
        self.train_thresholds = None
        self.train_f1s = None
        self.optimal_train_thresh_and_f1 = None
        self.y_optimal_thresholded = None
        self.yt_optimal_thresholded = None
        self.train_confusion_matrix = None
        self.test_confusion_matrix = None

        self.exfil_paths_train = exfil_paths_train
        self.exfil_weights_train = X_train_exfil_weight
        self.exfil_paths_test = exfil_paths
        self.exfil_weights_test = X_test_exfil_weight

        self.base_output_name = base_output_name
        self.recipes_used = recipes_used

        self.feature_activation_heatmaps_training = ['nonsense.png']
        self.feature_raw_heatmaps_training = ['nonsense.png']
        self.feature_activation_heatmaps, self.feature_raw_heatmaps = ['nonsense.png'],['nonsense.png']

        self.avg_exfil_per_min = None
        self.exfil_per_min_variance = None
        self.avg_pkt_size = None
        self.pkt_size_variance = None

        self.ROC_path = self.base_output_name + '_roc_' + self.model_to_fit + '_' + str(self.timegran)
        self.plot_name = 'roc_' + self.model_to_fit + '_' + str(self.timegran)
        self.title = 'ROC for ' + self.model_to_fit + ' at ' + str(self.timegran) + ' sec granularity'
        self.exp_name = self.recipes_used

    def train(self):
        if self.model_to_fit == 'logistic':
            self.clf = sklearn.linear_model.LogisticRegressionCV() ## putting positive here makes it works.

        elif self.model_to_fit == 'lasso':
            self.clf = sklearn.linear_model.LassoCV(cv=5, max_iter=80000) ## putting positive here makes it works.

        elif self.model_to_fit == 'boosting lasso':
            self.clf = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.linear_model.LassoCV(cv=5, max_iter=80000),
                                                           n_estimators=10, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

        elif self.model_to_fit == 'boosting logistic':
            self.clf = sklearn.ensemble.AdaBoostClassifier(
                base_estimator=sklearn.linear_model.LogisticRegressionCV(),
                n_estimators=10, learning_rate=1.0, algorithm='SAMME.R', random_state=None)



        print "self.X has NaN's here: ", np.where(np.isnan(self.X))
        print "self.X has Inf's here: ", np.where(np.isinf(self.X))

        print "self.X.columns.values", self.X.columns.values
        self.clf.fit(self.X, self.Y)
        self.train_predictions = self.clf.predict(X=self.X)

        # need to determine the optimal threshold point now.
        self._find_optimal_train_threshold()

        print "self.optimal_train_thresh_and_f1", self.optimal_train_thresh_and_f1

        self._generate_train_cm()

    def apply_to_new_data(self, Xt, Yt, base_output_name, recipes_used, avg_exfil_per_min, exfil_per_min_variance):
        self.base_output_name = base_output_name
        self.recipes_used = recipes_used
        self.ROC_path = self.base_output_name + '_roc_' + self.model_to_fit + '_' + str(self.timegran)
        self.plot_name = 'roc_' + self.model_to_fit + '_' + str(self.timegran)
        self.title = 'ROC for ' + self.model_to_fit + ' at ' + str(self.timegran) + ' sec granularity'
        self.exp_name = self.recipes_used
        self.avg_exfil_per_min =  avg_exfil_per_min
        self.exfil_per_min_variance = exfil_per_min_variance

        if Xt is not None:
            Xt = Xt.drop(columns='real_ide_angles_')

            exfil_paths = Xt['exfil_path']
            try:
                exfil_paths = exfil_paths.replace('0', '[]')
            except:
                try:
                    exfil_paths = exfil_paths.replace(0, '[]')
                except:
                    pass
            self.exfil_paths_test = exfil_paths

            Xt = drop_useless_columns_aggreg_DF(Xt)

            # if having problems with X_train_exfil_weight, can try using the value returned by this func...
            # the first param in the below func is pointless, but we need it to avoid crashing.
            _, Xt, _, _, X_test_exfil_weight = drop_useless_columns_testTrain_Xs(copy.deepcopy(Xt), Xt)

            self.exfil_weights_test = X_test_exfil_weight

            self.Xt = Xt
            self.Yt = Yt

        print '\n\n------\n\n'

        print "self.Xt", self.Xt, type(self.Xt)

        print "self.Xt has NaN's here: ", np.where(np.isnan(self.Xt))
        print "self.Xt has Inf's here: ", np.where(np.isinf(self.Xt))

        self.test_predictions = self.clf.predict(X=self.Xt)

        if self.Yt is not None:
            # can calculate performance vals...
            self._find_optimal_test_threshold()
            self._generate_test_cm()
        else:
            # cannot calculate performance vals... (so do nothing???)
            pass

    def _find_optimal_train_threshold(self):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=self.Y, y_score=self.train_predictions, pos_label=1)
        list_of_f1_scores = []
        for counter, threshold in enumerate(thresholds):
            # print counter,threshold
            y_pred = [int(i > threshold) for i in self.train_predictions]
            f1_score = sklearn.metrics.f1_score(self.Y, y_pred, pos_label=1, average='binary')
            list_of_f1_scores.append(f1_score)
        max_f1_score = max(list_of_f1_scores)
        max_f1_score_threshold_pos = [i for i, j in enumerate(list_of_f1_scores) if j == max_f1_score]
        threshold_corresponding_max_f1 = thresholds[max_f1_score_threshold_pos[0]]
        ########
        self.train_thresholds = thresholds
        self.train_f1s = list_of_f1_scores
        self.optimal_train_thresh_and_f1 = (threshold_corresponding_max_f1, max_f1_score)
        self.y_optimal_thresholded = [int(i > threshold_corresponding_max_f1) for i in self.train_predictions]

    def _find_optimal_test_threshold(self):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=self.Yt, y_score=self.test_predictions, pos_label=1)
        list_of_f1_scores = []
        for counter, threshold in enumerate(thresholds):
            # print counter,threshold
            yt_pred = [int(i > threshold) for i in self.test_predictions]
            f1_score = sklearn.metrics.f1_score(self.Yt, yt_pred, pos_label=1, average='binary')
            list_of_f1_scores.append(f1_score)
        max_f1_score = max(list_of_f1_scores)
        max_f1_score_threshold_pos = [i for i, j in enumerate(list_of_f1_scores) if j == max_f1_score]
        threshold_corresponding_max_f1 = thresholds[max_f1_score_threshold_pos[0]]
        ########
        self.test_thresholds = thresholds
        self.test_f1s = list_of_f1_scores
        self.optimal_test_thresh_and_f1 = (max_f1_score, threshold_corresponding_max_f1)
        self.yt_optimal_thresholded = [int(i > threshold_corresponding_max_f1) for i in self.test_predictions]

    def _generate_train_cm(self):
        y_train = self.Y['labels'].tolist()
        #print "y_train", y_train
        attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
            process_roc.determine_categorical_labels(y_train, self.y_optimal_thresholded, self.exfil_paths_train, self.exfil_weights_train.tolist())

        attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(attack_type_to_predictions,
                                                                                              attack_type_to_truth)
        categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values,
                                                                    attack_type_to_weights)
        ## re-name the row without any attacks in it...
        #print "categorical_cm_df.index", categorical_cm_df.index
        confusion_matrix = categorical_cm_df.rename({(): 'No Attack'}, axis='index')

        self.train_confusion_matrix = confusion_matrix

    def _generate_test_cm(self):
        print "\n===\n"
        y_test = self.Yt['labels'].tolist()
        print "y_test", y_test
        print "self.yt_optimal_thresholded", self.yt_optimal_thresholded
        attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
            process_roc.determine_categorical_labels(y_test, self.yt_optimal_thresholded, self.exfil_paths_test,
                                                     self.exfil_weights_test.tolist())

        attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(
            attack_type_to_predictions, attack_type_to_truth)

        print "attack_type_to_confusion_matrix_values", attack_type_to_confusion_matrix_values

        categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values,
                                                                    attack_type_to_weights)
        ## re-name the row without any attacks in it...
        print "categorical_cm_df.index", categorical_cm_df.index
        confusion_matrix = categorical_cm_df.rename({(): 'No Attack'}, axis='index')

        self.test_confusion_matrix = confusion_matrix

    def _generate_heatmap(self, training_p):
        pass # TODO: add this back in at some point!!! (need to process for this new file...)
        self.feature_activation_heatmaps = [''] # change...
        self.feature_raw_heatmaps = [''] # change...
        self.feature_activation_heatmaps_training = [''] # change...
        self.feature_raw_heatmaps_training  = [''] # change...
        '''
        if training_p:
            train_test = 'training_'
        else:
            train_test = 'test_'

        print "self.time_gran", self.time_gran
        if type(self.time_gran) == tuple:
            time_gran = "_".join([str(i) for i in self.time_gran]) + '_multi_'
        else:
            time_gran = str(self.time_gran)

        # make heatmaps so I can see which features are contributing
        current_heatmap_val_path = self.base_output_name + train_test + 'coef_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_val_path = 'temp_outputs/' + train_test + self.recipes_used +  '_heatmap_coef_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path = self.base_output_name + train_test + 'coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path = 'temp_outputs/' + train_test + self.recipes_used + '_heatmap_coef_contribs_at_' +  str(time_gran) + '.png'

        if training_p:
            coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(self.coef_dict, self.X_train, self.exfil_paths_train)
        else:
            coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(self.coef_dict, self.X_test, self.exfil_paths_test)


        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path, current_heatmap_path)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_val_path, current_heatmap_val_path)

        if training_p:
            self.feature_activation_heatmaps_training = ['../' + local_heatmap_path]
            self.feature_raw_heatmaps_training = ['../' + local_heatmap_val_path]
        else:
            self.feature_activation_heatmaps = ['../' + local_heatmap_path]
            self.feature_raw_heatmaps = ['../' + local_heatmap_val_path]
        print coef_impact_df
        '''

    def _generate_ROC(self, ROC_path, plot_name, title, exp_name):
        #'''        
        list_of_x_vals = []
        list_of_y_vals = []
        line_titles = []
        if not self.testing_data_present:
            #print "self.method_to_train_predictions", self.method_to_train_predictions
            #method_to_predictions = self.method_to_train_predictions
            predictions = self.train_predictions
            correct_labels = self.Y
        else:
            #method_to_predictions = self.method_to_test_predictions
            predictions = self.test_predictions
            correct_labels = self.Yt

        method_to_test_thresholds = {}
        method_to_predictions = {}
        method_to_predictions['ensemble'] = predictions
        for method, test_predictions in method_to_predictions.iteritems():
            # print "self.y_test",self.y_test
            # print "test_predictions", test_predictions
            #print "method", method
            # print "test_predictions", test_predictions
            test_predictions = np.nan_to_num(test_predictions)
            # print "self.y_test",self.y_test
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=correct_labels, y_score=test_predictions,
                                                             pos_label=1)
            list_of_x_vals.append(fpr)
            list_of_y_vals.append(tpr)
            line_titles.append(method)
            method_to_test_thresholds[method] = thresholds

        print "list_of_x_vals", list_of_x_vals
        print "list_of_y_vals", list_of_y_vals
        print "roc_path", ROC_path + plot_name
        exp_name = [i + '_NEW_MODEL' for i in exp_name]
        ax, _, plot_path = construct_ROC_curve(list_of_x_vals, list_of_y_vals, title,
                                               ROC_path + plot_name + 'NEW_MODEL_',
                                               line_titles, show_p=False, exp_name=exp_name)
        #'''
        return plot_path

    def generate_report_section(self, skip_heatmaps, using_pretrained_model):
        # (will probably need to pass in the comparison methods -- can then reassemble the dicts used in the orig method...)

        if not skip_heatmaps:
            pass # TODO: at some point..
            self._generate_heatmap(training_p=False)
            '''
            if not self.using_pretrained_model:
                self._generate_heatmap(training_p=True)
            if not self.no_testing:
                self._generate_heatmap(training_p=False)
            '''

        number_attacks_in_test = len(self.Yt[self.Yt['labels'] == 1])
        number_non_attacks_in_test = len(self.Yt[self.Yt['labels'] == 0])
        number_attacks_in_train = len(self.Y[self.Y['labels'] == 1])
        number_non_attacks_in_train = len(self.Y[self.Y['labels'] == 0])

        self.testing_data_present = (self.Xt.shape[0] != 0)
        if not self.testing_data_present: # in this case, there's no testing data present...
            percent_attacks = 0.0
        else:
            # so testing in this case.
            percent_attacks = (float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))
    
        env = Environment(
            loader=FileSystemLoader(searchpath="./report_templates")
        )
        table_section_template = env.get_template("table_section.html")

        self.coef_dict  = get_lin_mod_coef_dict(self.clf, self.X.columns.values, self.X.dtypes,
                                                sanity_check_number_coefs=(not using_pretrained_model),
                                                model_type=self.model_to_fit)
        print("self.X_train",self.X)
        print("self.coef_dict",self.coef_dict)
        self.coef_feature_df = pd.DataFrame.from_dict(self.coef_dict, orient='index')
        self.coef_feature_df.columns = ['Coefficient']

        self.model_params = self.clf.get_params()
        try:
            self.model_params['alpha_val'] = self.clf.alpha_
        except:
            pass

        coef_feature_df = self.coef_feature_df.to_html()
    
        if not using_pretrained_model:
            attacks_found_training=self.train_confusion_matrix.to_html()
            percent_attacks_train = (float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))
            feature_activation_heatmap_training=self.feature_activation_heatmaps_training[0]
            feature_raw_heatmap_training=self.feature_raw_heatmaps_training[0]
        else:
            attacks_found_training= 'none.png'
            percent_attacks_train = 0.0
            feature_activation_heatmap_training='none.png'
            feature_raw_heatmap_training='none.png'
    
        if not self.testing_data_present:
            ensemble_optimal_f1_scores_test = self.optimal_train_thresh_and_f1[1] # self.method_to_optimal_f1_scores_train[self.method_name]
            ensemble_df_test = '' #self.train_confusion_matrix #self.method_to_cm_df_train[self.method_name]
            ensemble_optimal_thresh_test = self.optimal_train_thresh_and_f1[0] #self.method_to_optimal_thresh_train[self.method_name]
        else:
            ensemble_optimal_f1_scores_test = self.optimal_test_thresh_and_f1[1] #self.method_to_optimal_f1_scores_test[self.method_name]
            ensemble_df_test = self.test_confusion_matrix.to_html() #self.method_to_cm_df_test[self.method_name]
            ensemble_optimal_thresh_test = self.optimal_test_thresh_and_f1[0] #self.method_to_optimal_thresh_test[self.method_name]

        roc_plot_path = self._generate_ROC(self.ROC_path, self.plot_name, self.title, self.exp_name)

        print "roc_plot_path", roc_plot_path
        print "time_gran", str(self.timegran) + " sec granularity"
        print "ideal_threshold", ensemble_optimal_thresh_test
        #print "attacks_found", ensemble_df_test.to_html()

        report_section = table_section_template.render(
            time_gran=str(self.timegran) + " sec granularity",
            roc=roc_plot_path,
            feature_table=coef_feature_df,
            model_params=self.model_params,
            optimal_fOne=ensemble_optimal_f1_scores_test,
            percent_attacks=percent_attacks,
            attacks_found=ensemble_df_test,
            attacks_found_training=attacks_found_training,
            percent_attacks_training=percent_attacks_train,
            feature_activation_heatmap=self.feature_activation_heatmaps[0],
            feature_activation_heatmap_training=feature_activation_heatmap_training,
            feature_raw_heatmap=self.feature_raw_heatmaps[0],
            feature_raw_heatmap_training=feature_raw_heatmap_training,
            ideal_threshold = ensemble_optimal_thresh_test
        )
    
        return report_section

class multi_time_alerts():
    def __init__(self, timegran_to_single_timegran_exfil_model, Yts, Xts, base_output_name, model_to_fit, recipes_used):
        self.ensemble_timegran = '(' + ','.join(self.timegran_to_single_timegran_exfil_model.keys()) + ')'
        self.timegran_to_single_timegran_exfil_model = timegran_to_single_timegran_exfil_model
        self.confusion_matrix = None
        longest_timegram = max(timegran_to_single_timegran_exfil_model.keys())

        ###### #TODO
        self.Yt = Yts[longest_timegram]
        Xt = Xts[longest_timegram]
        exfil_paths = Xt['exfil_path']
        try:
            exfil_paths = exfil_paths.replace('0', '[]')
        except:
            try:
                exfil_paths = exfil_paths.replace(0, '[]')
            except:
                pass
        self.exfil_paths_test = exfil_paths
        _, _, _, _, self.exfil_weights_test = drop_useless_columns_testTrain_Xs(copy.deepcopy(Xt), Xt)

        self.ROC_path = None
        self.plot_name  = None
        self.title  = None
        self.exp_name  = None

        self.ROC_path = base_output_name + '_roc_' + model_to_fit + '_' + str(self.ensemble_timegran)
        self.plot_name = 'roc_' + model_to_fit + '_' + str(self.ensemble_timegran)
        self.title = 'ROC for ' + model_to_fit + ' at ' + str(self.ensemble_timegran) + ' sec granularity'
        self.exp_name = recipes_used
        ##########

        self.alerts = timegran_to_single_timegran_exfil_model[longest_timegram].test_predictions

        timegran_to_alerts_at_longest_timegran = {}
        for timegran in timegran_to_single_timegran_exfil_model.keys():
            timegran_to_alerts_at_longest_timegran[timegran] = []

        for timegran, single_timegran_exfil_model in timegran_to_single_timegran_exfil_model.iteritems():
            ratio_of_current_timesteps_in_longest_timesteps = longest_timegram / timegran
            for i in range(0, ratio_of_current_timesteps_in_longest_timesteps):
                is_alert = 0
                for j in range(i, i + ratio_of_current_timesteps_in_longest_timesteps):
                    is_alert = is_alert or single_timegran_exfil_model.test_predictions[j]
                    timegran_to_alerts_at_longest_timegran[timegran].append(is_alert)

        all_alert_lists = timegran_to_alerts_at_longest_timegran.values()
        all_alert_lists_len = [len(i) for i in all_alert_lists]

        if min(all_alert_lists_len) != max(all_alert_lists_len):
            print "problem with multi-gran alerts"
            exit(7)

        for timegran, alerts_at_longest_timegran in timegran_to_alerts_at_longest_timegran.iteritems():
            for index in range(0,len(self.alerts)):
                self.alerts[index] = self.alerts[index] or alerts_at_longest_timegran[index]

    def generate_report_section(self, skip_heatmaps, using_pretrained_model):
    #def _generate_report_section(self, Yt, exfil_paths_test, exfil_weights_test, ROC_path, plot_name,
    #                             title, exp_name):
        # (will probably need to pass in the comparison methods -- can then reassemble the dicts used in the orig method...)

        number_attacks_in_test = len(self.Yt[self.Yt['labels'] == 1])
        number_non_attacks_in_test = len(self.Yt[self.Yt['labels'] == 0])

        percent_attacks = (float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))

        env = Environment(
            loader=FileSystemLoader(searchpath="./report_templates")
        )
        table_section_template = env.get_template("table_section.html")

        self.model_params = ''

        coef_feature_df = ''

        attacks_found_training = 'none.png'
        percent_attacks_train = 0.0
        feature_activation_heatmap_training = 'none.png'
        feature_raw_heatmap_training = 'none.png'

        ###################

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=self.Yt, y_score=self.alerts, pos_label=1)
        list_of_f1_scores = []
        for counter, threshold in enumerate(thresholds):
            # print counter,threshold
            yt_pred = [int(i > threshold) for i in self.alerts]
            f1_score = sklearn.metrics.f1_score(self.Yt, yt_pred, pos_label=1, average='binary')
            list_of_f1_scores.append(f1_score)
        max_f1_score = max(list_of_f1_scores)
        max_f1_score_threshold_pos = [i for i, j in enumerate(list_of_f1_scores) if j == max_f1_score]
        threshold_corresponding_max_f1 = thresholds[max_f1_score_threshold_pos[0]]
        yt_optimal_thresholded = [int(i > threshold_corresponding_max_f1) for i in self.alerts]

        y_test = self.Yt['labels'].tolist()
        print "y_test", y_test
        print "self.yt_optimal_thresholded", yt_optimal_thresholded
        attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
            process_roc.determine_categorical_labels(y_test, yt_optimal_thresholded, self.exfil_paths_test,
                                                     self.exfil_weights_test.tolist())

        attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(
            attack_type_to_predictions, attack_type_to_truth)

        print "attack_type_to_confusion_matrix_values", attack_type_to_confusion_matrix_values

        categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values,
                                                                    attack_type_to_weights)
        ## re-name the row without any attacks in it...
        print "categorical_cm_df.index", categorical_cm_df.index
        self.confusion_matrix = categorical_cm_df.rename({(): 'No Attack'}, axis='index')


        ensemble_optimal_f1_scores_test = max_f1_score # self.method_to_optimal_f1_scores_test[self.method_name]
        ensemble_df_test = self.confusion_matrix.to_html()  # self.method_to_cm_df_test[self.method_name]
        ensemble_optimal_thresh_test = threshold_corresponding_max_f1  # self.method_to_optimal_thresh_test[self.method_name]

        ####
        list_of_x_vals = []
        list_of_y_vals = []
        line_titles = []
        predictions = self.alerts
        correct_labels = self.Yt
        method_to_test_thresholds = {}
        method_to_predictions = {}
        method_to_predictions['ensemble'] = predictions
        for method, test_predictions in method_to_predictions.iteritems():
            test_predictions = np.nan_to_num(test_predictions)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=correct_labels, y_score=test_predictions,
                                                             pos_label=1)
            list_of_x_vals.append(fpr)
            list_of_y_vals.append(tpr)
            line_titles.append(method)
            method_to_test_thresholds[method] = thresholds

        print "roc_path", self.ROC_path + self.plot_name
        exp_name = [i + '_NEW_MODEL' for i in self.exp_name]
        ax, _, roc_plot_path = construct_ROC_curve(list_of_x_vals, list_of_y_vals, self.title,
                                               self.ROC_path + self.plot_name + 'NEW_MODEL_',
                                               line_titles, show_p=False, exp_name=exp_name)
        ##################

        print "roc_plot_path", roc_plot_path
        print "time_gran", str(self.ensemble_timegran) + " sec granularity"
        print "ideal_threshold", ensemble_optimal_thresh_test
        # print "attacks_found", ensemble_df_test.to_html()

        report_section = table_section_template.render(
            time_gran=str(self.ensemble_timegran) + " sec granularity",
            roc=roc_plot_path,
            feature_table=coef_feature_df,
            model_params=self.model_params,
            optimal_fOne=ensemble_optimal_f1_scores_test,
            percent_attacks=percent_attacks,
            attacks_found=ensemble_df_test,
            attacks_found_training=attacks_found_training,
            percent_attacks_training=percent_attacks_train,
            feature_activation_heatmap='none.png',
            feature_activation_heatmap_training=feature_activation_heatmap_training,
            feature_raw_heatmap='none.png',
            feature_raw_heatmap_training=feature_raw_heatmap_training,
            ideal_threshold=ensemble_optimal_thresh_test
        )

        return report_section

def get_lin_mod_coef_dict(clf, X_train_columns, X_train_dtypes, sanity_check_number_coefs, model_type):
    coef_dict = {}
    print "Coefficients: "
    print "LASSO model", clf.get_params()
    print '----------------------'
    print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns)

    if 'logistic' in model_type:
        model_coefs = clf.coef_[0]
    else:
        model_coefs = clf.coef_

    if sanity_check_number_coefs:
        if len(model_coefs) != (len(X_train_columns)):  # there is no plus one b/c the intercept is stored in clf.intercept_
            print "coef_ is different length than X_train_columns!", X_train_columns
            for counter, i in enumerate(X_train_dtypes):
                print counter, i, X_train_columns[counter]
                print model_coefs  # [counter]
                print len(model_coefs)

    for coef, feat in zip(model_coefs, X_train_columns):
        coef_dict[feat] = coef
    coef_dict['intercept'] = float(clf.intercept_)

    print "COEFS_HERE"
    for coef, feature in coef_dict.iteritems():
        print coef, feature

    print "coef_dict"
    return coef_dict


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